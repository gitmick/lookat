"""Threaded frame grabber with OpenCV (USB/UVC webcams) and Picamera2 backends.

The grabber always keeps only the newest frame, so a slow detector can never
build up latency.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

_UNSET = object()


class CameraError(RuntimeError):
    pass


def macos_permission_hint() -> str:
    """macOS denies camera access silently -- the capture opens and then
    hands back nothing. Say so rather than leaving people guessing."""
    if sys.platform != "darwin":
        return ""
    return (
        "\n\nOn macOS the camera needs permission. Open"
        "\n  System Settings > Privacy & Security > Camera"
        "\nand enable the app you started this from (Terminal, iTerm, VS Code)."
        "\nIf it is not listed, quit that app fully and run lookat again to be asked."
    )


class _Backend:
    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class OpenCVBackend(_Backend):
    def __init__(self, device, width: int, height: int, fps: int):
        import cv2

        self.cv2 = cv2
        self.cap = None

        # Per-platform API preference; None means "let OpenCV choose".
        apis: list = []
        if isinstance(device, int):
            if sys.platform.startswith("win"):
                # Measured on OpenCV 5 / Windows 11: MSMF ~1.4s, generic ~1.9s
                # (it resolves to MSMF anyway, after loading the FFMPEG plugin
                # and logging a warning), DSHOW ~3.7s. DSHOW stays as a
                # fallback because some webcams only work through it.
                apis = [getattr(cv2, "CAP_MSMF", None), getattr(cv2, "CAP_DSHOW", None)]
            elif sys.platform.startswith("linux"):
                apis = [getattr(cv2, "CAP_V4L2", None)]  # skips a slow GStreamer probe
            elif sys.platform == "darwin":
                apis = [getattr(cv2, "CAP_AVFOUNDATION", None)]
        # Always keep the generic path as a last resort, without duplicates.
        apis = [a for a in apis if a is not None] + [None]
        seen: set = set()
        apis = [a for a in apis if not (a in seen or seen.add(a))]

        for api in apis:
            cap = cv2.VideoCapture(device) if api is None else cv2.VideoCapture(device, api)
            if cap.isOpened():
                self.cap = cap
                log.debug("opened camera %r with api %s", device, api)
                break
            cap.release()

        if self.cap is None:
            raise CameraError(
                f"could not open camera {device!r}. Try another index "
                "(--camera 1), or close any app already using the webcam."
                + macos_permission_hint()
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # not supported by every backend
            pass

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self) -> None:
        self.cap.release()


class Picamera2Backend(_Backend):
    """Raspberry Pi CSI cameras (Camera Module 1/2/3, HQ, GS)."""

    def __init__(self, width: int, height: int, fps: int):
        from picamera2 import Picamera2

        self.picam = Picamera2()
        # "RGB888" hands back BGR-ordered arrays, which is what OpenCV wants.
        config = self.picam.create_preview_configuration(
            main={"format": "RGB888", "size": (width, height)}
        )
        self.picam.configure(config)
        try:
            self.picam.set_controls({"FrameRate": float(fps)})
        except Exception:
            pass
        self.picam.start()
        time.sleep(0.5)  # let auto-exposure settle

    def read(self) -> Optional[np.ndarray]:
        return self.picam.capture_array()

    def close(self) -> None:
        self.picam.stop()
        self.picam.close()


def _open_backend(cfg, device=_UNSET) -> _Backend:
    name = str(cfg.get("camera.backend", "auto")).lower()
    width = int(cfg.get("camera.width", 640))
    height = int(cfg.get("camera.height", 480))
    fps = int(cfg.get("camera.fps", 30))
    if device is _UNSET:
        device = cfg.get("camera.device", 0)

    if name == "picamera2" or (name == "auto" and sys.platform.startswith("linux")):
        try:
            backend = Picamera2Backend(width, height, fps)
            log.info("camera: picamera2 %dx%d", width, height)
            return backend
        except Exception as exc:
            if name == "picamera2":
                raise CameraError(f"picamera2 backend failed: {exc}") from exc
            log.debug("picamera2 unavailable (%s), falling back to OpenCV", exc)

    # device: null means "pick one for me". Try the usual index first so the
    # common case costs nothing; the fallback below scans if that fails.
    if device is None:
        device = 0

    try:
        backend = OpenCVBackend(device, width, height, fps)
    except CameraError:
        if not isinstance(device, int):
            raise
        log.warning("camera %d did not open; looking for another one", device)
        found = [d for d in probe_cameras() if d[0] != device]
        if not found:
            raise
        device = found[0][0]
        log.info("falling back to camera %d", device)
        backend = OpenCVBackend(device, width, height, fps)

    log.info("camera: opencv device=%r %dx%d", device, width, height)
    backend.device = device
    return backend


class Camera:
    """Background frame grabber. `read()` returns (sequence_number, frame)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.flip = bool(cfg.get("camera.flip_horizontal", True))
        self.rotate = int(cfg.get("camera.rotate", 0)) % 360
        self._backend: Optional[_Backend] = None
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fail_count = 0
        self.device = cfg.get("camera.device", 0)

    def start(self) -> "Camera":
        self._backend = _open_backend(self.cfg)
        self.device = getattr(self._backend, "device", self.device)
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        # Wait briefly for the first frame so callers get something immediately.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and self._seq == 0 and not self._stop.is_set():
            time.sleep(0.02)
        if self._seq == 0:
            raise CameraError(
                "the camera opened but delivered no frames within 5s"
                + macos_permission_hint()
            )
        return self

    def _transform(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        if self.rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.flip:
            frame = cv2.flip(frame, 1)
        return frame

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._backend.read() if self._backend else None
            except Exception as exc:
                log.warning("camera read failed: %s", exc)
                frame = None

            if frame is None:
                self._fail_count += 1
                if self._fail_count > 150:
                    log.error("camera produced no frames for ~5s")
                    self._fail_count = 0
                time.sleep(0.03)
                continue

            self._fail_count = 0
            frame = self._transform(frame)
            with self._lock:
                self._frame = frame
                self._seq += 1

    def read(self) -> Tuple[int, Optional[np.ndarray]]:
        with self._lock:
            return self._seq, self._frame

    def switch_device(self, device) -> bool:
        """Swap to another camera while running. Reverts on failure.

        Most webcams cannot be opened twice, so the current one is released
        first; `read()` returns the last frame until the new one delivers.
        """
        if device == self.device:
            return True
        previous, old_backend = self.device, self._backend

        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if old_backend:
            old_backend.close()
        self._stop.clear()

        for candidate in (device, previous):
            try:
                self._backend = _open_backend(self.cfg, candidate)
            except CameraError as exc:
                log.error("camera %r failed to open: %s", candidate, exc)
                continue
            self.device = getattr(self._backend, "device", candidate)
            self._fail_count = 0
            self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
            self._thread.start()
            if candidate != device:
                log.warning("stayed on camera %r", self.device)
            return candidate == device

        self._backend = None
        log.error("no camera could be opened any more")
        return False

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._backend:
            self._backend.close()
            self._backend = None


def probe_cameras(max_index: int = 8, stop_after_misses: int = 3,
                  skip: set | None = None) -> list[tuple[int, int, int]]:
    """Return [(index, width, height)] for every camera index that delivers a
    frame. Handy for finding the right `camera.device` on Windows.

    Probing a missing index is slow, so give up after a few misses in a row.
    """
    found: list[tuple[int, int, int]] = []
    misses = 0
    skip = skip or set()
    for index in range(max_index):
        if index in skip:
            # Already in use by this process; opening it again would fail.
            found.append((index, 0, 0))
            misses = 0
            continue
        backend = None
        try:
            backend = OpenCVBackend(index, 640, 480, 30)
            frame = backend.read()
        except CameraError:
            frame = None
        finally:
            if backend is not None:
                backend.close()

        if frame is None:
            misses += 1
            if misses >= stop_after_misses and found:
                break
            if misses >= stop_after_misses + 2:
                break
            continue
        misses = 0
        found.append((index, frame.shape[1], frame.shape[0]))
    return found
