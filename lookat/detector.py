"""Face detection -> head pose + iris offset -> "is this person looking at me?".

Coordinate conventions used throughout (OpenCV camera frame):
    x -> image right, y -> image down, z -> away from the camera.

    yaw   > 0  : the face points towards the right-hand side of the image
    pitch > 0  : the face points upwards

A frontal face staring straight into the lens gives yaw = pitch = 0.
Mirroring the image flips the sign of yaw *and* of the horizontal iris offset
and of the position compensation, so the final |deviation| is unaffected.
"""

from __future__ import annotations

import logging
import math
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# OpenCV 5 no longer ships the Haar cascades inside the Python wheel, so the
# fallback detector fetches them on demand, like the landmark model.
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"

# --- MediaPipe FaceMesh landmark indices -----------------------------------
# "left"/"right" here mean left/right *in the image*, not the subject's own.
LEFT_EYE = {
    "outer": 33, "inner": 133, "top": 159, "bottom": 145, "iris": 468,
    "ear": (33, 160, 158, 133, 153, 144),
}
RIGHT_EYE = {
    "outer": 263, "inner": 362, "top": 386, "bottom": 374, "iris": 473,
    "ear": (362, 385, 387, 263, 373, 380),
}

# Six stable points used for solvePnP, with a generic head model in mm.
POSE_LANDMARKS = (1, 152, 33, 263, 61, 291)
MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),        # 1   nose tip
        (0.0, 63.6, 12.5),      # 152 chin
        (-43.3, -32.7, 26.0),   # 33  eye corner, image left
        (43.3, -32.7, 26.0),    # 263 eye corner, image right
        (-28.9, 28.9, 24.1),    # 61  mouth corner, image left
        (28.9, 28.9, 24.1),     # 291 mouth corner, image right
    ],
    dtype=np.float64,
)


@dataclass
class Observation:
    """Everything one detector pass learned about the scene."""

    ts: float = 0.0
    face_found: bool = False
    score: float = 0.0            # 0..1 "is looking at the screen"
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0
    gaze_yaw: float = 0.0         # head pose + iris deflection
    gaze_pitch: float = 0.0
    dev_yaw: float = 0.0          # gaze relative to the screen, after offsets
    dev_pitch: float = 0.0
    # How far the face is turned away from pointing straight at the camera,
    # with the viewer's position in frame accounted for but no screen offsets.
    # This is the honest "how frontal is this face" measure -- raw head_yaw /
    # head_pitch are inflated by perspective when the face is off-centre.
    frontal_yaw: float = 0.0
    frontal_pitch: float = 0.0
    eye_h: float = 0.5            # iris position in the eye opening, 0..1
    eye_v: float = 0.5
    eyes_open: bool = True
    openness: float = 1.0
    face_ratio: float = 0.0       # face width / frame width
    bbox: Optional[Tuple[int, int, int, int]] = None
    landmarks: Optional[np.ndarray] = None   # Nx2 pixels, debug only
    backend: str = ""
    note: str = ""
    identity: Optional[str] = None      # filled in by the identity stage
    identity_score: float = 0.0


def membership(angle: float, tolerance: float, softness: float) -> float:
    """1 inside the tolerance cone, linear falloff over `softness`, then 0."""
    a = abs(angle)
    if a <= tolerance:
        return 1.0
    if softness <= 0 or a >= tolerance + softness:
        return 0.0
    return 1.0 - (a - tolerance) / softness


def _euler_from_rotation(rmat: np.ndarray) -> Tuple[float, float, float]:
    forward = rmat @ np.array([0.0, 0.0, -1.0])  # face normal, out of the face
    yaw = math.degrees(math.atan2(forward[0], -forward[2]))
    pitch = math.degrees(math.atan2(-forward[1], -forward[2]))
    roll = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
    return yaw, pitch, roll


def _ratio(value: float, low: float, high: float) -> float:
    span = high - low
    if abs(span) < 1e-6:
        return 0.5
    return float(np.clip((value - low) / span, 0.0, 1.0))


def _eye_aspect_ratio(pts: np.ndarray, idx) -> float:
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 0.0
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * horizontal))


class GazeScorer:
    """Turns raw geometry into a 0..1 attention score. Shared by all backends."""

    def __init__(self, cfg):
        g = lambda k, d: cfg.get("gaze." + k, d)  # noqa: E731
        self.yaw_tol = float(g("yaw_tolerance_deg", 22))
        self.pitch_tol = float(g("pitch_tolerance_deg", 18))
        self.soft = float(g("softness_deg", 8))
        self.use_eye = bool(g("use_eye_offset", True))
        self.eye_gain_yaw = float(g("eye_gain_yaw_deg", 30))
        self.eye_gain_pitch = float(g("eye_gain_pitch_deg", 15))
        self.compensate = bool(g("compensate_position", True))
        self.hfov = float(g("horizontal_fov_deg", 60))
        self.yaw_offset = float(g("yaw_offset_deg", 0))
        self.pitch_offset = float(g("pitch_offset_deg", 0))
        self.require_open = bool(g("require_eyes_open", True))
        self.min_face_ratio = float(g("min_face_width_ratio", 0.05))

    def focal_px(self, width: int) -> float:
        return (width / 2.0) / math.tan(math.radians(self.hfov / 2.0))

    def apply(self, obs: Observation, frame_shape) -> Observation:
        height, width = frame_shape[:2]

        gaze_yaw, gaze_pitch = obs.head_yaw, obs.head_pitch
        if self.use_eye:
            gaze_yaw += (obs.eye_h - 0.5) * 2.0 * self.eye_gain_yaw
            gaze_pitch -= (obs.eye_v - 0.5) * 2.0 * self.eye_gain_pitch
        obs.gaze_yaw, obs.gaze_pitch = gaze_yaw, gaze_pitch

        # Angle from the camera axis to the face. A viewer standing off to the
        # side must turn their head to look at the camera; that expectation is
        # geometric, not a property of where they are looking.
        offset_x = offset_y = 0.0
        if obs.bbox:
            focal = self.focal_px(width)
            x, y, w, h = obs.bbox
            cx, cy = x + w / 2.0, y + h / 2.0
            offset_x = math.degrees(math.atan2(cx - width / 2.0, focal))
            offset_y = math.degrees(math.atan2(cy - height / 2.0, focal))

        obs.frontal_yaw = obs.head_yaw + offset_x
        obs.frontal_pitch = obs.head_pitch - offset_y

        dev_yaw, dev_pitch = gaze_yaw, gaze_pitch
        if self.compensate:
            dev_yaw += offset_x
            dev_pitch -= offset_y

        dev_yaw -= self.yaw_offset
        dev_pitch -= self.pitch_offset
        obs.dev_yaw, obs.dev_pitch = dev_yaw, dev_pitch

        score = membership(dev_yaw, self.yaw_tol, self.soft) * membership(
            dev_pitch, self.pitch_tol, self.soft
        )
        if self.require_open and not obs.eyes_open:
            score = 0.0
        if obs.face_ratio and obs.face_ratio < self.min_face_ratio:
            # Too far away for the pose estimate to mean anything.
            score *= max(0.0, obs.face_ratio / max(self.min_face_ratio, 1e-6))

        obs.score = float(np.clip(score, 0.0, 1.0))
        return obs


# --------------------------------------------------------------------------- 
# MediaPipe backend
# ---------------------------------------------------------------------------
def ensure_model(path: str, auto_download: bool = True) -> str:
    if os.path.exists(path):
        return path
    if not auto_download:
        raise FileNotFoundError(
            f"{path} missing. Run scripts/fetch_model.sh or set "
            "detector.auto_download_model=true"
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    log.info("downloading face landmark model from %s", MODEL_URL)
    tmp = path + ".part"
    urllib.request.urlretrieve(MODEL_URL, tmp)  # noqa: S310 - fixed, known URL
    os.replace(tmp, path)
    log.info("saved %s (%.1f MB)", path, os.path.getsize(path) / 1e6)
    return path


class MediaPipeDetector:
    """478-point face mesh incl. irises. Must be created in the thread using it."""

    name = "mediapipe"

    def __init__(self, cfg, video_mode: bool = True):
        """`video_mode=False` for unrelated still images: VIDEO mode tracks
        across calls, so feeding it separate photos leaks one face's region
        into the next and produces bad landmarks."""
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.mp = mp
        self.video_mode = video_mode
        self.scorer = GazeScorer(cfg)
        self.blink_threshold = float(cfg.get("gaze.blink_threshold", 0.6))
        self.ear_threshold = float(cfg.get("gaze.ear_threshold", 0.15))
        confidence = float(cfg.get("detector.min_confidence", 0.5))

        model_path = ensure_model(
            cfg.path(cfg.get("detector.model_path", "models/face_landmarker.task")),
            bool(cfg.get("detector.auto_download_model", True)),
        )
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            min_face_detection_confidence=confidence,
            min_face_presence_confidence=confidence,
            min_tracking_confidence=confidence,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def close(self) -> None:
        try:
            self.landmarker.close()
        except Exception:
            pass

    def process(self, frame: np.ndarray, ts_ms: int) -> Observation:
        import cv2

        obs = Observation(backend=self.name)
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)

        if self.video_mode:
            ts_ms = max(ts_ms, self._last_ts + 1)  # must be strictly increasing
            self._last_ts = ts_ms
            result = self.landmarker.detect_for_video(image, ts_ms)
        else:
            result = self.landmarker.detect(image)
        if not result.face_landmarks:
            return obs

        lm = result.face_landmarks[0]
        pts = np.array([(p.x * width, p.y * height) for p in lm], dtype=np.float64)
        obs.face_found = True
        obs.landmarks = pts

        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        obs.bbox = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
        obs.face_ratio = float((x1 - x0) / width)

        # --- head pose ------------------------------------------------------
        image_points = np.array([pts[i] for i in POSE_LANDMARKS], dtype=np.float64)
        focal = self.scorer.focal_px(width)
        camera_matrix = np.array(
            [[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]],
            dtype=np.float64,
        )
        flags = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_ITERATIVE)
        ok, rvec, _ = cv2.solvePnP(
            MODEL_POINTS, image_points, camera_matrix, np.zeros((4, 1)), flags=flags
        )
        if ok:
            rmat, _ = cv2.Rodrigues(rvec)
            obs.head_yaw, obs.head_pitch, obs.head_roll = _euler_from_rotation(rmat)
        else:
            obs.note = "solvePnP failed"
            return obs

        # --- iris offset ----------------------------------------------------
        h_ratios, v_ratios = [], []
        for eye in (LEFT_EYE, RIGHT_EYE):
            if eye["iris"] >= len(pts):
                continue
            iris = pts[eye["iris"]]
            outer, inner = pts[eye["outer"]], pts[eye["inner"]]
            top, bottom = pts[eye["top"]], pts[eye["bottom"]]
            h_ratios.append(_ratio(iris[0], min(outer[0], inner[0]), max(outer[0], inner[0])))
            v_ratios.append(_ratio(iris[1], top[1], bottom[1]))
        if h_ratios:
            obs.eye_h = float(np.mean(h_ratios))
            obs.eye_v = float(np.mean(v_ratios))

        # --- eyes open ------------------------------------------------------
        blink = None
        if result.face_blendshapes:
            scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
            left, right = scores.get("eyeBlinkLeft"), scores.get("eyeBlinkRight")
            if left is not None and right is not None:
                blink = max(left, right)
        if blink is not None:
            obs.openness = 1.0 - blink
            obs.eyes_open = blink < self.blink_threshold
        else:
            ear = min(
                _eye_aspect_ratio(pts, LEFT_EYE["ear"]),
                _eye_aspect_ratio(pts, RIGHT_EYE["ear"]),
            )
            obs.openness = float(np.clip(ear / 0.3, 0.0, 1.0))
            obs.eyes_open = ear >= self.ear_threshold

        return self.scorer.apply(obs, frame.shape)


# ---------------------------------------------------------------------------
# Haar cascade fallback (no mediapipe): coarse frontal-face presence only
# ---------------------------------------------------------------------------
def _load_cascade(name: str, cache_dir: str, auto_download: bool):
    """Load a Haar cascade from the OpenCV wheel, a local cache, or the web."""
    import cv2

    candidates = []
    bundled = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if bundled:
        candidates.append(os.path.join(bundled, name))
    cached = os.path.join(cache_dir, name)
    candidates.append(cached)

    for path in candidates:
        if os.path.exists(path):
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                return cascade

    if not auto_download:
        raise RuntimeError(
            f"{name} not found (OpenCV 5 no longer bundles it). Enable "
            "detector.auto_download_model, or install mediapipe."
        )

    os.makedirs(cache_dir, exist_ok=True)
    log.info("downloading %s", name)
    tmp = cached + ".part"
    urllib.request.urlretrieve(CASCADE_URL + name, tmp)  # noqa: S310 - fixed URL
    os.replace(tmp, cached)
    cascade = cv2.CascadeClassifier(cached)
    if cascade.empty():
        raise RuntimeError(f"downloaded {name} but OpenCV could not load it")
    return cascade


class HaarDetector:
    name = "haar"

    def __init__(self, cfg):
        import cv2

        self.cv2 = cv2
        self.scorer = GazeScorer(cfg)
        cache_dir = os.path.dirname(
            cfg.path(cfg.get("detector.model_path", "models/face_landmarker.task"))
        ) or "models"
        auto = bool(cfg.get("detector.auto_download_model", True))
        self.face = _load_cascade("haarcascade_frontalface_default.xml", cache_dir, auto)
        self.eyes = _load_cascade("haarcascade_eye.xml", cache_dir, auto)
        log.warning(
            "using the Haar fallback: frontal-face presence only, no real gaze "
            "estimation. Install mediapipe for proper tracking."
        )

    def close(self) -> None:
        pass

    def process(self, frame: np.ndarray, ts_ms: int) -> Observation:
        cv2 = self.cv2
        obs = Observation(backend=self.name, note="coarse frontal-face heuristic")
        height, width = frame.shape[:2]
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = self.face.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
        if len(faces) == 0:
            return obs

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        obs.face_found = True
        obs.bbox = (int(x), int(y), int(w), int(h))
        obs.face_ratio = w / width

        roi = gray[y : y + int(h * 0.6), x : x + w]
        eyes = self.eyes.detectMultiScale(roi, 1.15, 6, minSize=(int(w * 0.12),) * 2)
        obs.eyes_open = len(eyes) >= 1
        obs.openness = min(1.0, len(eyes) / 2.0)

        # The frontal cascade only fires on roughly frontal faces, so presence
        # itself is the signal; eye count refines it.
        score = {0: 0.25, 1: 0.65}.get(len(eyes), 0.9)
        if obs.face_ratio < self.scorer.min_face_ratio:
            score *= obs.face_ratio / max(self.scorer.min_face_ratio, 1e-6)
        obs.score = float(np.clip(score, 0.0, 1.0))
        return obs


def build_detector(cfg, video_mode: bool = True):
    """Instantiate the detector. Call this inside the thread that will use it."""
    wanted = str(cfg.get("detector.backend", "auto")).lower()
    if wanted in ("auto", "mediapipe"):
        try:
            return MediaPipeDetector(cfg, video_mode=video_mode)
        except Exception as exc:
            if wanted == "mediapipe":
                raise
            log.warning("mediapipe backend unavailable (%s); falling back to Haar", exc)
    return HaarDetector(cfg)
