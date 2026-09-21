"""Menu and in-app task tests.

    python tests/test_menu.py [face.jpg]

With a face photo it also runs the real in-app calibration and enrolment
against a replayed frame; without one, only the menu logic is exercised.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lookat.config import load_config, patch_yaml_values
from lookat.menu import Menu, MenuItem


class FakeKey:
    """Minimal stand-in for a pygame KEYDOWN event."""

    def __init__(self, key, unicode=""):
        self.type = 2  # pygame.KEYDOWN
        self.key = key
        self.unicode = unicode


class FakePygame:
    KEYDOWN = 2
    K_ESCAPE, K_m, K_UP, K_DOWN, K_k, K_j = 27, ord("m"), 273, 274, ord("k"), ord("j")
    K_RETURN, K_KP_ENTER, K_SPACE, K_RIGHT = 13, 271, 32, 275
    K_BACKSPACE = 8


PG = FakePygame()


def _menu(fired):
    return Menu(items=[
        MenuItem("first", lambda: fired.append("first"), lambda: "A"),
        MenuItem("disabled", lambda: fired.append("nope"), lambda: "-", enabled=lambda: False),
        MenuItem("third", lambda: fired.append("third"), lambda: "C"),
    ])


def test_navigation_skips_disabled():
    menu = _menu([])
    assert menu.index == 0
    menu.handle(FakeKey(PG.K_DOWN), PG)
    assert menu.current.label == "third", "must step over the disabled entry"
    menu.handle(FakeKey(PG.K_DOWN), PG)
    assert menu.current.label == "first", "wraps around"
    menu.handle(FakeKey(PG.K_UP), PG)
    assert menu.current.label == "third"


def test_enter_runs_the_action_and_never_a_disabled_one():
    fired = []
    menu = _menu(fired)
    menu.handle(FakeKey(PG.K_RETURN), PG)
    assert fired == ["first"]
    menu.index = 1                      # force onto the disabled entry
    menu.handle(FakeKey(PG.K_RETURN), PG)
    assert fired == ["first"], "a disabled entry must not fire"


def test_open_and_close():
    menu = _menu([])
    assert menu.open is False
    menu.toggle()
    assert menu.open is True
    menu.handle(FakeKey(PG.K_ESCAPE), PG)
    assert menu.open is False


def test_text_prompt():
    got = []
    menu = _menu([])
    menu.ask("Name?", got.append)
    for char in "anna":
        menu.handle(FakeKey(ord(char), char), PG)
    menu.handle(FakeKey(PG.K_BACKSPACE), PG)
    assert menu.text == "ann"
    menu.handle(FakeKey(ord("a"), "a"), PG)
    menu.handle(FakeKey(PG.K_RETURN), PG)
    assert got == ["anna"]
    assert menu.prompt is None, "prompt must close after submitting"


def test_prompt_swallows_navigation_keys():
    """While typing, 'j' is a letter, not 'move down'."""
    menu = _menu([])
    menu.ask("Name?", lambda name: None)
    menu.handle(FakeKey(PG.K_j, "j"), PG)
    assert menu.text == "j"
    assert menu.index == 0

    menu.cancel_prompt()
    assert menu.prompt is None


def test_empty_name_is_not_submitted():
    got = []
    menu = _menu([])
    menu.ask("Name?", got.append)
    menu.handle(FakeKey(PG.K_RETURN), PG)
    assert got == []


def test_choice_picker():
    picked = []
    menu = _menu([])
    menu.choose("Remove which person?", ["anna", "ben"], picked.append)
    menu.handle(FakeKey(PG.K_DOWN), PG)
    menu.handle(FakeKey(PG.K_RETURN), PG)
    assert picked == ["ben"]
    assert menu.choices is None, "picker must close after choosing"

    # the picker must swallow keys, not move the main list underneath
    menu.choose("Pick", ["a", "b"], picked.append)
    before = menu.index
    menu.handle(FakeKey(PG.K_DOWN), PG)
    menu.handle(FakeKey(PG.K_ESCAPE), PG)
    assert menu.choices is None and menu.index == before
    assert picked == ["ben"], "escape must not pick anything"


def test_choice_handler_errors_are_contained():
    menu = _menu([])

    def boom(_pick):
        raise RuntimeError("nope")

    menu.choose("Pick", ["a"], boom)
    menu.handle(FakeKey(PG.K_RETURN), PG)
    assert menu.choices is None
    assert "failed" in menu.status and menu.status_kind == "bad"


def test_removal_is_queued_and_applied_by_the_worker():
    """Removing must go through the worker: recognition iterates the same
    dict every frame."""
    from lookat.app import DetectionWorker, Shared, apply_task_result
    from lookat.hooks import HookRunner
    from lookat.identity import PeopleDatabase

    db_path = os.path.join(tempfile.mkdtemp(), "people.json")
    db = PeopleDatabase(db_path)
    db.add("anna", [np.ones((1, 128), dtype=np.float32)])
    db.add("ben", [np.zeros((1, 128), dtype=np.float32)])
    db.save()

    cfg = load_config(overrides=[f"identity.database={db_path}", "identity.enabled=true"])
    cfg.set("display.scenes.person", {"anna": {"text": "Hi anna"}})

    shared = Shared()
    worker = DetectionWorker(cfg, None, shared, HookRunner(cfg), threading.Event())

    class FakeIdentifier:
        def __init__(self, database):
            self.db = database

    worker.identifier = FakeIdentifier(PeopleDatabase(db_path))

    worker.request_removal("anna")
    worker.request_removal("nobody")
    worker._process_removals()

    assert "anna" not in worker.identifier.db.people
    assert "ben" in worker.identifier.db.people
    assert "anna" not in PeopleDatabase(db_path).people, "not persisted"

    menu = Menu()
    apply_task_result({"kind": "forget", "ok": True, "name": "anna"},
                      cfg, object(), menu, None)
    assert "anna" not in (cfg.get("display.scenes.person") or {})
    assert "Removed anna" in menu.status


def test_broken_value_accessor_does_not_crash_drawing():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    from lookat.display import build_display

    cfg = load_config(overrides=["display.fullscreen=false", "display.window_size=[640,360]"])
    display = build_display(cfg)
    try:
        menu = Menu(items=[
            MenuItem("ok", None, lambda: "fine"),
            MenuItem("broken", None, lambda: 1 / 0),
        ])
        menu.open = True
        menu.set_status("something happened", "bad")
        display.menu = menu
        from lookat.attention import AttentionState

        assert display.update("idle", AttentionState()) is True
        menu.ask("Name?", lambda n: None)
        menu.text = "anna"
        assert display.update("idle", AttentionState()) is True
        menu.cancel_prompt()
        menu.choose("Remove which person?", ["anna", "ben"], lambda n: None)
        assert display.update("idle", AttentionState()) is True
    finally:
        display.close()


def _make_image(path, colour=(20, 120, 200), size=(64, 48)):
    import pygame

    pygame.init()
    surface = pygame.Surface(size)
    surface.fill(colour)
    pygame.image.save(surface, str(path))


def test_image_lookup_and_person_fallback():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from lookat.display import build_display

    folder = tempfile.mkdtemp()
    cfg = load_config(overrides=["display.fullscreen=false", "display.window_size=[320,240]",
                                 "display.mode=images", f"display.images.folder={folder}"])
    display = build_display(cfg)
    try:
        assert display.mode == "images"
        assert display.image_for("idle") is None, "nothing there yet"

        _make_image(os.path.join(folder, "idle.jpg"))
        _make_image(os.path.join(folder, "attentive.png"))
        assert str(display.image_for("idle")).endswith("idle.jpg")
        assert str(display.image_for("attentive")).endswith("attentive.png")

        # a person with no picture of their own falls back to `attentive`
        assert str(display.image_for("person.bob")).endswith("attentive.png")
        _make_image(os.path.join(folder, "person-bob.webp"))
        assert str(display.image_for("person-bob".replace("-", "."))).endswith("person-bob.webp")
    finally:
        display.close()


def test_images_mode_renders_and_falls_back_to_a_placeholder():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from lookat.attention import AttentionState
    from lookat.display import build_display

    folder = tempfile.mkdtemp()
    _make_image(os.path.join(folder, "idle.jpg"))
    cfg = load_config(overrides=["display.fullscreen=false", "display.window_size=[320,240]",
                                 "display.mode=images", f"display.images.folder={folder}"])
    display = build_display(cfg)
    try:
        state = AttentionState()
        assert display.update("idle", state) is True
        # `attentive` has no file -- must render a placeholder, not raise
        assert display.update("attentive", state) is True

        # switching back to text must work at runtime
        display.set_mode("text")
        assert display.mode == "text" and cfg.get("display.mode") == "text"
        assert display.update("attentive", state) is True
        display.set_mode("images")
        assert display.update("idle", state) is True
    finally:
        display.close()


def test_patch_yaml_keeps_comments():
    path = os.path.join(tempfile.mkdtemp(), "c.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("gaze:\n  yaw_offset_deg: 0   # where the screen is\n  other: 1\n")
    missing = patch_yaml_values(path, {"yaw_offset_deg": 28.6, "absent": 2})
    body = open(path, encoding="utf-8").read()
    assert "28.6" in body and "# where the screen is" in body
    assert missing == ["absent"]


def test_in_app_calibration_and_enrolment(face: str):
    """The worker must run both tasks on the already-open camera."""
    import cv2

    from lookat import app, camera as cm
    from lookat.app import DetectionWorker, Shared, apply_task_result
    from lookat.hooks import HookRunner

    frame = cv2.imread(face)
    frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))

    class Still(cm._Backend):
        def read(self):
            time.sleep(0.02)
            return frame.copy()

    real = cm._open_backend
    cm._open_backend = lambda cfg, device=cm._UNSET: Still()
    db_path = os.path.join(tempfile.mkdtemp(), "people.json")
    try:
        overrides = [
            "camera.flip_horizontal=false", "detector.detection_fps=20",
            "identity.enabled=true", f"identity.database={db_path}",
            "gaze.yaw_offset_deg=0", "gaze.pitch_offset_deg=0",
        ]
        if os.environ.get("LOOKAT_MODEL"):
            overrides.append(f"detector.model_path={os.environ['LOOKAT_MODEL']}")
        if os.environ.get("LOOKAT_SFACE"):
            overrides.append(f"identity.model_path={os.environ['LOOKAT_SFACE']}")
        cfg = load_config(overrides=overrides)

        camera = cm.Camera(cfg).start()
        shared, stop = Shared(), threading.Event()
        worker = DetectionWorker(cfg, camera, shared, HookRunner(cfg), stop)
        worker.start()
        try:
            deadline = time.monotonic() + 25
            while worker.identifier is None and time.monotonic() < deadline:
                time.sleep(0.1)
            assert worker.identifier is not None, "recogniser never came up"

            # --- calibration ------------------------------------------------
            assert worker.request_task("calibrate", 2.0) is True
            assert worker.request_task("calibrate", 2.0) is False, "no two tasks at once"
            result = _await_result(shared, 25)
            assert result["kind"] == "calibrate" and result["ok"], result
            assert cfg.get("gaze.yaw_offset_deg") == result["values"]["yaw_offset_deg"]
            print(f"    calibrated: {result['values']}")

            # --- enrolment ---------------------------------------------------
            assert worker.request_task("enroll", 2.0, "anna") is True
            result = _await_result(shared, 25)
            assert result["kind"] == "enroll" and result["ok"], result
            assert "anna" in worker.identifier.db.people
            assert os.path.exists(db_path), "the database was not written"

            # the app then gives the new person a scene of their own
            menu = Menu()
            apply_task_result(result, cfg, object(), menu, None)
            assert "anna" in (cfg.get("display.scenes.person") or {})
            print(f"    enrolled anna, scene: {cfg.get('display.scenes.person')['anna']}")
        finally:
            stop.set()
            worker.join(timeout=5)
            camera.stop()
    finally:
        cm._open_backend = real


def _await_result(shared, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with shared.lock:
            result = shared.task_result
            if result is not None:
                shared.task_result = None
                return result
        time.sleep(0.05)
    raise AssertionError("the task never finished")


def main() -> int:
    face = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("LOOKAT_FACE_A")
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v) and k != "test_in_app_calibration_and_enrolment"]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {test.__name__}: {exc!r}")

    if face and os.path.exists(face):
        try:
            test_in_app_calibration_and_enrolment(face)
            print("PASS  test_in_app_calibration_and_enrolment")
        except Exception as exc:
            failed += 1
            print(f"FAIL  test_in_app_calibration_and_enrolment: {exc!r}")
    else:
        print("SKIP  test_in_app_calibration_and_enrolment (pass a face photo)")

    print(f"\n{'all good' if not failed else str(failed) + ' failed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
