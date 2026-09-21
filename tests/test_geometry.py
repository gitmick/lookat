"""Geometry and scoring tests. Run with:  python tests/test_geometry.py

No webcam and no pytest required -- the head pose is exercised by projecting a
synthetic head through a pinhole camera and solving it back.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lookat.attention import AttentionTracker
from lookat.config import load_config
from lookat.detector import (
    MODEL_POINTS,
    POSE_LANDMARKS,
    GazeScorer,
    Observation,
    _euler_from_rotation,
    membership,
)

W, H, HFOV = 640, 480, 60.0
FOCAL = (W / 2) / math.tan(math.radians(HFOV / 2))
K = np.array([[FOCAL, 0, W / 2], [0, FOCAL, H / 2], [0, 0, 1]], float)


def pose_towards(yaw_deg: float, pitch_deg: float):
    """Upright head whose forward ray points at the given tangent-plane angles."""
    d = np.array([math.tan(math.radians(yaw_deg)), -math.tan(math.radians(pitch_deg)), -1.0])
    d /= np.linalg.norm(d)
    z_col = -d
    x_col = np.cross(np.array([0.0, 1.0, 0.0]), z_col)
    x_col /= np.linalg.norm(x_col)
    y_col = np.cross(z_col, x_col)
    return np.column_stack([x_col, y_col, z_col])


def test_pose_round_trip():
    """solvePnP + euler extraction must recover gaze direction, sign included."""
    import cv2

    worst = 0.0
    for true_yaw in (-45, -30, -15, -5, 0, 5, 15, 30, 45):
        for true_pitch in (-30, -15, 0, 15, 30):
            rotation = pose_towards(true_yaw, true_pitch)
            translation = np.array([[40.0], [-25.0], [600.0]])
            camera_points = (rotation @ MODEL_POINTS.T) + translation
            projected = (K @ camera_points).T
            image_points = np.ascontiguousarray(projected[:, :2] / projected[:, 2:3])

            ok, rvec, _ = cv2.solvePnP(
                MODEL_POINTS, image_points, K, np.zeros((4, 1)), flags=cv2.SOLVEPNP_SQPNP
            )
            assert ok
            yaw, pitch, roll = _euler_from_rotation(cv2.Rodrigues(rvec)[0])
            worst = max(worst, abs(yaw - true_yaw), abs(pitch - true_pitch), abs(roll))
    assert worst < 0.5, f"pose error {worst:.2f} deg -- sign/convention bug"


def test_pose_landmarks_in_range():
    assert len(POSE_LANDMARKS) == len(MODEL_POINTS)
    assert max(POSE_LANDMARKS) < 468, "pose points must be mesh points, not iris points"


def test_membership_falloff():
    assert membership(0, 20, 8) == 1.0
    assert membership(-20, 20, 8) == 1.0
    assert membership(24, 20, 8) == 0.5
    assert membership(28, 20, 8) == 0.0
    assert membership(90, 20, 8) == 0.0


def _observation(**kwargs) -> Observation:
    base = dict(face_found=True, bbox=(W // 2 - 60, H // 2 - 80, 120, 160), face_ratio=0.19)
    base.update(kwargs)
    return Observation(**base)


def test_centred_face_looking_straight_scores_one():
    cfg = load_config()
    scorer = GazeScorer(cfg)
    obs = scorer.apply(_observation(head_yaw=0, head_pitch=0, eye_h=0.5, eye_v=0.5), (H, W, 3))
    assert obs.score == 1.0, obs


def test_turned_head_scores_zero():
    cfg = load_config()
    scorer = GazeScorer(cfg)
    obs = scorer.apply(_observation(head_yaw=55, head_pitch=0), (H, W, 3))
    assert obs.score == 0.0, obs


def test_closed_eyes_score_zero():
    cfg = load_config()
    scorer = GazeScorer(cfg)
    obs = scorer.apply(_observation(head_yaw=0, head_pitch=0, eyes_open=False), (H, W, 3))
    assert obs.score == 0.0


def test_eye_offset_moves_the_gaze():
    """Looking hard to one side with the head still must break attention."""
    cfg = load_config(overrides=["gaze.eye_gain_yaw_deg=30", "gaze.yaw_tolerance_deg=15",
                                 "gaze.softness_deg=5"])
    scorer = GazeScorer(cfg)
    straight = scorer.apply(_observation(eye_h=0.5), (H, W, 3))
    sideways = scorer.apply(_observation(eye_h=1.0), (H, W, 3))
    assert straight.score == 1.0
    assert sideways.gaze_yaw > 25
    assert sideways.score == 0.0


def test_position_compensation():
    """Someone standing off to the side, but looking into the lens, counts as
    looking. Without compensation their head yaw alone would fail."""
    cfg = load_config(overrides=["gaze.compensate_position=true", "gaze.yaw_tolerance_deg=12",
                                 "gaze.softness_deg=4", "gaze.use_eye_offset=false"])
    scorer = GazeScorer(cfg)
    # face near the right edge of the frame
    bbox = (W - 160, H // 2 - 80, 120, 160)
    cx = bbox[0] + bbox[2] / 2
    offset_deg = math.degrees(math.atan2(cx - W / 2, FOCAL))
    assert offset_deg > 12, "test setup should place the face outside the raw tolerance"

    looking = scorer.apply(_observation(bbox=bbox, head_yaw=-offset_deg), (H, W, 3))
    assert abs(looking.dev_yaw) < 0.5 and looking.score == 1.0, looking

    straight_ahead = scorer.apply(_observation(bbox=bbox, head_yaw=0.0), (H, W, 3))
    assert straight_ahead.score == 0.0, straight_ahead


def test_far_away_face_is_discounted():
    cfg = load_config(overrides=["gaze.min_face_width_ratio=0.10"])
    scorer = GazeScorer(cfg)
    obs = scorer.apply(_observation(face_ratio=0.02), (H, W, 3))
    assert 0.0 < obs.score < 0.3, obs


def test_tracker_hysteresis_and_dwell():
    cfg = load_config(overrides=[
        "attention.enter_frames=3", "attention.exit_frames=4",
        "attention.min_state_seconds=0.0", "attention.smoothing=1.0",
        "attention.lost_face_grace_seconds=0.0",
    ])
    tracker = AttentionTracker(cfg)
    now = 0.0

    def step(score, face=True):
        nonlocal now
        now += 0.1
        return tracker.update(_observation(face_found=face, score=score), now).attentive

    assert step(1.0) is False          # needs consecutive frames
    assert step(1.0) is False
    assert step(1.0) is True           # third frame flips it
    assert step(0.0) is True           # one bad frame does not
    assert step(0.0) is True
    assert step(0.0) is True
    assert step(0.0) is False          # fourth does
    # a single high frame in the middle resets the enter counter
    assert step(1.0) is False
    assert step(0.0) is False
    assert step(1.0) is False
    assert step(1.0) is False
    assert step(1.0) is True


def test_tracker_min_state_seconds_blocks_flapping():
    cfg = load_config(overrides=[
        "attention.enter_frames=1", "attention.exit_frames=1",
        "attention.min_state_seconds=1.0", "attention.smoothing=1.0",
    ])
    tracker = AttentionTracker(cfg)
    assert tracker.update(_observation(score=1.0), 10.0).attentive is True
    # still inside the dwell window -> must not flip back
    assert tracker.update(_observation(score=0.0), 10.3).attentive is True
    assert tracker.update(_observation(score=0.0), 11.5).attentive is False


def test_tracker_face_loss_grace():
    cfg = load_config(overrides=[
        "attention.enter_frames=1", "attention.exit_frames=1",
        "attention.min_state_seconds=0.0", "attention.smoothing=1.0",
        "attention.lost_face_grace_seconds=0.5",
    ])
    tracker = AttentionTracker(cfg)
    assert tracker.update(_observation(score=1.0), 0.0).attentive is True
    assert tracker.update(Observation(face_found=False), 0.2).attentive is True   # blink/dropout
    assert tracker.update(Observation(face_found=False), 1.0).attentive is False  # really gone


def test_config_overrides_and_types():
    cfg = load_config(overrides=["display.fullscreen=false", "camera.device=2",
                                 "gaze.yaw_tolerance_deg=33.5", "hooks.on_idle=echo hi"])
    assert cfg.get("display.fullscreen") is False
    assert cfg.get("camera.device") == 2
    assert cfg.get("gaze.yaw_tolerance_deg") == 33.5
    assert cfg.get("hooks.on_idle") == "echo hi"
    assert cfg.get("does.not.exist", "fallback") == "fallback"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
