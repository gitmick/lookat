"""Enrol, list and forget faces.

    python run.py --enroll alice                 # 10s from the webcam
    python run.py --enroll bob --from photos/*.jpg
    python run.py --people
    python run.py --forget alice
"""

from __future__ import annotations

import glob
import logging
import os
import time
from typing import List

import numpy as np

from .camera import Camera
from .detector import build_detector
from .identity import FaceIdentifier, PeopleDatabase

log = logging.getLogger("lookat.enroll")


def _quality_ok(obs, min_ratio: float) -> tuple[bool, str]:
    if not obs.face_found:
        return False, "no face"
    if obs.face_ratio < min_ratio:
        return False, f"face too small ({obs.face_ratio:.2f})"
    # Frontality relative to the camera, not raw head pose: a face high or low
    # in the frame has a large raw pitch purely from perspective.
    if abs(obs.frontal_yaw) > 32 or abs(obs.frontal_pitch) > 32:
        return False, (f"turned too far (yaw {obs.frontal_yaw:+.0f}, "
                       f"pitch {obs.frontal_pitch:+.0f})")
    if not obs.eyes_open:
        return False, "eyes closed"
    return True, ""


def _dedupe(identifier: FaceIdentifier, features: List[np.ndarray],
            keep: int, similar: float = 0.97) -> List[np.ndarray]:
    """Keep a spread of distinct templates instead of `keep` near-identical
    ones from consecutive frames."""
    chosen: List[np.ndarray] = []
    for feature in features:
        if all(identifier.similarity(feature, other) < similar for other in chosen):
            chosen.append(feature)
        if len(chosen) >= keep:
            break
    return chosen or features[:keep]


def enroll_from_camera(cfg, name: str, seconds: float = 10.0, keep: int = 8,
                       add: bool = False) -> int:
    identifier = FaceIdentifier(cfg)
    camera = Camera(cfg).start()
    detector = build_detector(cfg)
    min_ratio = float(cfg.get("identity.min_face_ratio", 0.08))

    print(f"\nEnrolling '{name}'.")
    print(f"Look at the camera for {seconds:.0f} seconds. Turn your head a little,")
    print("and change your expression, so the templates cover some variation.\n")
    for count in (3, 2, 1):
        print(f"  starting in {count}...", flush=True)
        time.sleep(1.0)
    print("  capturing...\n", flush=True)

    features: List[np.ndarray] = []
    rejected: dict[str, int] = {}
    frames = 0
    start, last_seq = time.monotonic(), -1
    try:
        while time.monotonic() - start < seconds:
            seq, frame = camera.read()
            if frame is None or seq == last_seq:
                time.sleep(0.005)
                continue
            last_seq = seq
            frames += 1
            obs = detector.process(frame, int((time.monotonic() - start) * 1000))
            ok, why = _quality_ok(obs, min_ratio)
            if not ok:
                rejected[why.split(" (")[0]] = rejected.get(why.split(" (")[0], 0) + 1
                continue
            feature = identifier.embed(frame, obs)
            if feature is not None:
                features.append(feature)
    finally:
        camera.stop()
        detector.close()

    print(f"{len(features)} usable faces out of {frames} frames")
    for why, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
        print(f"  skipped {count:4d}  {why}")

    if len(features) < 5:
        print("\nNot enough usable frames. Check the lighting, and make sure you")
        print("are facing the camera and close enough to fill part of the frame.")
        return 1

    return _store(cfg, identifier, name, features, keep, add)


def enroll_from_images(cfg, name: str, patterns: List[str], keep: int = 8,
                       add: bool = False) -> int:
    import cv2

    identifier = FaceIdentifier(cfg)
    paths: List[str] = []
    for pattern in patterns:
        hits = sorted(glob.glob(pattern))
        if not hits and os.path.exists(pattern):
            hits = [pattern]
        paths.extend(hits)
    if not paths:
        print(f"no files matched: {' '.join(patterns)}")
        return 1

    # IMAGE mode: VIDEO mode would carry tracking between unrelated photos.
    detector = build_detector(cfg, video_mode=False)
    min_ratio = float(cfg.get("identity.min_face_ratio", 0.08))
    features: List[np.ndarray] = []
    try:
        for path in paths:
            image = cv2.imread(path)
            if image is None:
                print(f"  {os.path.basename(path):30} unreadable")
                continue
            if image.shape[1] > 1280:
                scale = 1280 / image.shape[1]
                image = cv2.resize(image, (1280, int(image.shape[0] * scale)))
            obs = detector.process(image, 1)
            ok, why = _quality_ok(obs, min_ratio)
            if not ok:
                print(f"  {os.path.basename(path):30} skipped: {why}")
                continue
            feature = identifier.embed(image, obs)
            if feature is None:
                print(f"  {os.path.basename(path):30} skipped: could not embed")
                continue
            features.append(feature)
            print(f"  {os.path.basename(path):30} ok")
    finally:
        detector.close()

    if not features:
        print("\nno usable faces found")
        return 1
    return _store(cfg, identifier, name, features, keep, add)


def _store(cfg, identifier: FaceIdentifier, name: str, features: List[np.ndarray],
           keep: int, add: bool) -> int:
    kept = _dedupe(identifier, features, keep)

    # Warn if this person looks like somebody already enrolled.
    db = identifier.db
    for other, samples in db.people.items():
        if other == name:
            continue
        best = max(identifier.similarity(kept[0], s) for s in samples)
        if best >= identifier.threshold:
            print(f"\n  WARNING: this face matches '{other}' at {best:.2f}.")
            print("  They will be hard to tell apart. Enrol more varied photos,")
            print("  or raise identity.threshold / identity.margin.")

    db.add(name, kept, replace=not add)
    db.save()

    spread = [identifier.similarity(kept[0], k) for k in kept[1:]]
    print(f"\nstored {len(kept)} templates for '{name}' in {db.path}")
    if spread:
        print(f"  template spread: {min(spread):.2f}..{max(spread):.2f} "
              "(lower = more variation captured)")
    print(f"  enrolled now: {', '.join(sorted(db.people)) or 'nobody'}")
    print("\nEnable it with  identity.enabled: true  in config.yaml")
    return 0


def list_people(cfg) -> int:
    db = PeopleDatabase(cfg.path(cfg.get("identity.database", "people.json")))
    if not db.people:
        print("nobody enrolled yet.  python run.py --enroll <name>")
        return 0
    print(f"{len(db.people)} enrolled ({db.path}):")
    for name in sorted(db.people):
        when = db.enrolled_at.get(name) or "unknown date"
        print(f"  {name:20} {len(db.people[name]):2d} templates   {when}")
    return 0


def forget_person(cfg, name: str) -> int:
    db = PeopleDatabase(cfg.path(cfg.get("identity.database", "people.json")))
    if not db.forget(name):
        print(f"'{name}' is not enrolled")
        return 1
    db.save()
    print(f"removed '{name}'. Remaining: {', '.join(sorted(db.people)) or 'nobody'}")
    return 0
