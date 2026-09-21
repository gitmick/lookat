"""Face recognition tests.

Needs two photos of two different people. Pass them on the command line, or
set LOOKAT_FACE_A / LOOKAT_FACE_B. The models are found via LOOKAT_MODEL and
LOOKAT_SFACE, or downloaded into models/ as usual.

    python tests/test_identity.py personA.jpg personB.jpg
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lookat.app import pick_scene
from lookat.config import load_config
from lookat.identity import IdentityVoter, Match, PeopleDatabase


def _cfg(db_path: str):
    overrides = [f"identity.database={db_path}", "identity.enabled=true"]
    if os.environ.get("LOOKAT_MODEL"):
        overrides += [f"detector.model_path={os.environ['LOOKAT_MODEL']}"]
    if os.environ.get("LOOKAT_SFACE"):
        overrides += [f"identity.model_path={os.environ['LOOKAT_SFACE']}"]
    return load_config(overrides=overrides)


def test_voter_needs_agreement():
    voter = IdentityVoter(window=5, min_votes=3)
    assert voter.push(Match(name="alice", score=0.8)) is None      # 1 vote
    assert voter.push(Match(name="alice", score=0.8)) is None      # 2 votes
    assert voter.push(Match(name="alice", score=0.8)) == "alice"   # 3 -> accepted
    # one stray frame must not rename anybody
    assert voter.push(Match(name="bob", score=0.5)) == "alice"
    assert voter.push(Match()) == "alice"
    voter.reset()
    assert voter.current is None


def test_voter_switches_person_when_evidence_is_consistent():
    voter = IdentityVoter(window=4, min_votes=3)
    for _ in range(3):
        voter.push(Match(name="alice", score=0.8))
    assert voter.current == "alice"
    for _ in range(4):
        voter.push(Match(name="bob", score=0.8))
    assert voter.current == "bob"


def test_pick_scene():
    known = {"idle", "attentive", "person.alice"}
    assert pick_scene(False, None, known) == "idle"
    assert pick_scene(False, "alice", known) == "idle"        # away wins
    assert pick_scene(True, None, known) == "attentive"       # unknown face
    assert pick_scene(True, "alice", known) == "person.alice"
    assert pick_scene(True, "bob", known) == "attentive"      # no scene for bob


def test_database_round_trip():
    path = os.path.join(tempfile.mkdtemp(), "people.json")
    db = PeopleDatabase(path)
    db.add("alice", [np.arange(128, dtype=np.float32).reshape(1, -1)])
    db.add("bob", [np.ones((1, 128), dtype=np.float32)])
    db.save()

    again = PeopleDatabase(path)
    assert sorted(again.people) == ["alice", "bob"]
    assert again.people["alice"][0].shape == (1, 128)
    assert np.allclose(again.people["alice"][0], np.arange(128))
    assert again.enrolled_at["alice"]

    assert again.forget("alice") is True
    assert again.forget("nobody") is False
    again.save()
    assert sorted(PeopleDatabase(path).people) == ["bob"]


def test_recognises_two_specific_faces(face_a: str, face_b: str):
    """The real thing: enrol two people, then identify each of them."""
    import cv2

    from lookat.detector import build_detector
    from lookat.enroll import enroll_from_images
    from lookat.identity import FaceIdentifier

    db_path = os.path.join(tempfile.mkdtemp(), "people.json")
    cfg = _cfg(db_path)

    assert enroll_from_images(cfg, "alpha", [face_a]) == 0
    assert enroll_from_images(cfg, "beta", [face_b]) == 0

    identifier = FaceIdentifier(cfg)
    assert sorted(identifier.db.people) == ["alpha", "beta"]

    detector = build_detector(cfg, video_mode=False)
    try:
        results = {}
        for label, path in (("alpha", face_a), ("beta", face_b)):
            image = cv2.imread(path)
            image = cv2.resize(image, (640, int(image.shape[0] * 640 / image.shape[1])))
            obs = detector.process(image, 1)
            assert obs.face_found, f"no face in {path}"
            match = identifier.identify(image, obs)
            results[label] = match
            print(f"    {label:6} -> {match.name!s:8} score {match.score:.3f} "
                  f"(runner-up {match.runner_up} {match.runner_up_score:.3f})")
            assert match.name == label, f"{path} identified as {match.name}"
            assert match.score > identifier.threshold

        # The two must not be confusable.
        gap = min(r.score - r.runner_up_score for r in results.values())
        print(f"    smallest margin over the wrong person: {gap:.3f}")
        assert gap > 0.3, "the two faces are too close to tell apart reliably"
    finally:
        detector.close()


def test_pipeline_switches_person_scene(face_a: str, face_b: str):
    """The running app must pick person.<name> as the faces change."""
    import threading
    import time

    import cv2

    from lookat import app, camera as cm, display as dm
    from lookat.enroll import enroll_from_images

    db_path = os.path.join(tempfile.mkdtemp(), "people.json")
    setup = _cfg(db_path)
    assert enroll_from_images(setup, "alpha", [face_a]) == 0
    assert enroll_from_images(setup, "beta", [face_b]) == 0

    def load(path):
        image = cv2.imread(path)
        return cv2.resize(image, (640, int(image.shape[0] * 640 / image.shape[1])))

    first, second = load(face_a), load(face_b)

    class Swap(cm._Backend):
        def __init__(self):
            self.t0 = time.monotonic()

        def read(self):
            time.sleep(0.02)
            return (first if time.monotonic() - self.t0 < 3.0 else second).copy()

    seen: list[str] = []

    class Recorder:
        debug = False

        def __init__(self, cfg):
            pass

        def scene_names(self):
            return ["idle", "attentive", "person.alpha", "person.beta"]

        def update(self, scene, state, preview=None, stats=None):
            if not seen or seen[-1] != scene:
                seen.append(scene)
            time.sleep(0.02)
            return True

        def close(self):
            pass

    real_camera, real_display = cm._open_backend, app.build_display
    cm._open_backend = lambda cfg: Swap()
    app.build_display = lambda cfg: Recorder(cfg)
    try:
        cfg = _cfg(db_path)
        for override in ("camera.flip_horizontal=false", "detector.detection_fps=15",
                         "attention.enter_frames=2", "attention.min_state_seconds=0.0",
                         "identity.recognize_every=2", "identity.min_votes=2"):
            key, _, value = override.partition("=")
            cfg.set(key, value if not value.replace(".", "").isdigit() else float(value)
                    if "." in value else int(value))
        cfg.set("camera.flip_horizontal", False)

        stop = threading.Event()
        thread = threading.Thread(target=lambda: app.run(cfg, stop), daemon=True)
        thread.start()
        time.sleep(6.5)
        stop.set()
        thread.join(timeout=8)
    finally:
        cm._open_backend, app.build_display = real_camera, real_display

    print(f"    scene timeline: {' -> '.join(seen)}")
    assert "person.alpha" in seen, f"never recognised the first person: {seen}"
    assert "person.beta" in seen, f"never recognised the second person: {seen}"
    assert seen.index("person.alpha") < seen.index("person.beta")
    assert not thread.is_alive(), "app did not shut down"


def main() -> int:
    face_a = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("LOOKAT_FACE_A")
    face_b = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("LOOKAT_FACE_B")

    tests = [test_voter_needs_agreement, test_voter_switches_person_when_evidence_is_consistent,
             test_pick_scene, test_database_round_trip]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")

    if face_a and face_b and os.path.exists(face_a) and os.path.exists(face_b):
        for test in (test_recognises_two_specific_faces, test_pipeline_switches_person_scene):
            try:
                test(face_a, face_b)
                print(f"PASS  {test.__name__}")
            except Exception as exc:
                failed += 1
                print(f"FAIL  {test.__name__}: {exc}")
    else:
        print("SKIP  face tests (pass two photos of two different people)")

    print(f"\n{'all good' if not failed else str(failed) + ' failed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
