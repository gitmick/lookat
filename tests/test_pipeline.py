"""End-to-end smoke test without a webcam.

A fake camera backend replays a still image (or a solid colour), the real
detector/tracker/display stack runs on top, and we check that the state
switches, the hooks fire and everything shuts down cleanly.

Run with:  python tests/test_pipeline.py [path/to/face.jpg]
Set SDL_VIDEODRIVER=dummy to test the pygame display without a monitor.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lookat import app, camera as camera_module
from lookat.config import load_config


class FakeBackend(camera_module._Backend):
    def __init__(self, frame):
        self.frame = frame

    def read(self):
        time.sleep(0.01)
        return self.frame.copy()


def install_fake_camera(frame):
    camera_module._open_backend = lambda cfg: FakeBackend(frame)


def test_camera_thread_delivers_frames():
    install_fake_camera(np.zeros((480, 640, 3), np.uint8))
    cfg = load_config(overrides=["camera.flip_horizontal=true", "camera.rotate=90"])
    cam = camera_module.Camera(cfg).start()
    try:
        time.sleep(0.2)
        seq, frame = cam.read()
        assert seq > 0 and frame is not None
        assert frame.shape[:2] == (640, 480), f"rotate=90 should swap axes, got {frame.shape}"
    finally:
        cam.stop()
    print("PASS  camera thread + rotation")


def test_headless_run_switches_state(image_path=None):
    import cv2

    if image_path and os.path.exists(image_path):
        frame = cv2.imread(image_path)
        frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
        expect_attentive = True
    else:
        frame = np.full((480, 640, 3), 30, np.uint8)  # no face anywhere
        expect_attentive = False

    install_fake_camera(frame)
    marker = os.path.join(tempfile.gettempdir(), "lookat_hook_marker")
    for state in ("attentive", "idle"):
        path = f"{marker}_{state}"
        if os.path.exists(path):
            os.remove(path)

    overrides = [
        "display.backend=none",
        "camera.flip_horizontal=false",
        "detector.detection_fps=20",
        "attention.enter_frames=2",
        "attention.min_state_seconds=0.0",
        f"hooks.on_attentive=touch {marker}_attentive",
        f"hooks.on_idle=touch {marker}_idle",
    ]
    model = os.environ.get("LOOKAT_MODEL")
    if model:
        overrides += [f"detector.model_path={model}", "detector.auto_download_model=false"]

    cfg = load_config(overrides=overrides)

    result = {}
    stop = threading.Event()
    thread = threading.Thread(
        target=lambda: result.setdefault("rc", app.run(cfg, stop)), daemon=True
    )
    thread.start()
    time.sleep(3.0)

    # The state change is observed through the hook marker file.
    saw_attentive = os.path.exists(f"{marker}_attentive")

    stop.set()
    thread.join(timeout=8.0)

    assert not thread.is_alive(), "app.run did not shut down on SIGINT"
    assert result.get("rc") == 0, f"app.run returned {result.get('rc')}"
    if expect_attentive:
        assert saw_attentive, "a face looking at the camera should have fired the hook"
        print("PASS  headless run: detected the face, fired the hook, shut down")
    else:
        assert not saw_attentive, "a blank frame must not count as someone looking"
        print("PASS  headless run: blank frames stay idle, shut down cleanly")


def test_pygame_display_renders():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from lookat.attention import AttentionState
    from lookat.display import build_display, parse_colour

    assert parse_colour("#ff8000") == (255, 128, 0)
    assert parse_colour("#fff") == (255, 255, 255)
    assert parse_colour(None, (1, 2, 3)) == (1, 2, 3)

    cfg = load_config(overrides=["display.fullscreen=false", "display.window_size=[320,240]",
                                 "display.fade_seconds=0.05"])
    display = build_display(cfg)
    try:
        state = AttentionState(attentive=False, score=0.0)
        for _ in range(3):
            assert display.update("idle", state) is True
        state.attentive = True
        for _ in range(8):
            assert display.update("attentive", state) is True
        assert display.alpha > 0.9, f"crossfade did not complete: alpha={display.alpha}"
        assert display._shown == "attentive"

        # A per-person scene that inherits from `attentive`, rendered on demand.
        display.cfg.set("display.scenes.person", {"alice": {"text": "Hi Alice"}})
        assert "person.alice" in display.scene_names()
        for _ in range(8):
            assert display.update("person.alice", state) is True
        assert display._shown == "person.alice"

        display.debug = True
        preview = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        assert display.update("person.alice", state, preview, {"det fps": "12.0"}) is True
    finally:
        display.close()
    print("PASS  pygame display renders, crossfades, per-person scenes, overlay")


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else None
    test_camera_thread_delivers_frames()
    test_pygame_display_renders()
    test_headless_run_switches_state(image)
    print("\nall pipeline checks passed")
