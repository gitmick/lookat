"""Face recognition: tell enrolled people apart.

Uses OpenCV's built-in SFace recogniser, so there is no extra Python
dependency -- only a 37 MB ONNX model, fetched on first use.

The five alignment landmarks SFace expects are taken straight from the
MediaPipe mesh we already compute, in the order SFace wants:
subject's right eye, left eye, nose tip, right mouth corner, left mouth corner
(the subject's right is the left-hand side of the image).

Measured separation on sample photos: same person 0.74-0.99, different people
0.07-0.15. The default threshold of 0.40 sits in that gap with room to spare.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

# MediaPipe mesh indices for the five points SFace aligns on.
ALIGN_LANDMARKS = (468, 473, 1, 61, 291)

DB_VERSION = 1


@dataclass
class Match:
    name: Optional[str] = None      # None means "not recognised"
    score: float = 0.0              # similarity to the best candidate
    runner_up: Optional[str] = None
    runner_up_score: float = 0.0

    @property
    def known(self) -> bool:
        return self.name is not None


def ensure_sface_model(path: str, auto_download: bool = True) -> str:
    if os.path.exists(path):
        return path
    if not auto_download:
        raise FileNotFoundError(
            f"{path} missing. Run scripts/fetch_model.sh --identity, or set "
            "identity.auto_download_model=true"
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    log.info("downloading the face recognition model (~37 MB)")
    tmp = path + ".part"
    urllib.request.urlretrieve(SFACE_URL, tmp)  # noqa: S310 - fixed, known URL
    os.replace(tmp, path)
    log.info("saved %s (%.1f MB)", path, os.path.getsize(path) / 1e6)
    return path


class PeopleDatabase:
    """Enrolled faces, stored as plain JSON so you can inspect and edit it."""

    def __init__(self, path: str):
        self.path = path
        self.people: Dict[str, List[np.ndarray]] = {}
        self.enrolled_at: Dict[str, str] = {}
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            log.error("could not read %s: %s", self.path, exc)
            return
        for name, entry in (data.get("people") or {}).items():
            samples = [np.asarray(s, dtype=np.float32).reshape(1, -1)
                       for s in entry.get("samples", [])]
            if samples:
                self.people[name] = samples
                self.enrolled_at[name] = entry.get("enrolled", "")
        log.info("loaded %d enrolled %s from %s",
                 len(self.people), "person" if len(self.people) == 1 else "people", self.path)

    def save(self) -> None:
        data = {
            "version": DB_VERSION,
            "model": "sface_2021dec",
            "people": {
                name: {
                    "enrolled": self.enrolled_at.get(name, ""),
                    "samples": [s.reshape(-1).tolist() for s in samples],
                }
                for name, samples in self.people.items()
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=1)
        os.replace(tmp, self.path)

    def add(self, name: str, samples: List[np.ndarray], replace: bool = True) -> None:
        if replace or name not in self.people:
            self.people[name] = list(samples)
        else:
            self.people[name].extend(samples)
        self.enrolled_at[name] = time.strftime("%Y-%m-%dT%H:%M:%S")

    def forget(self, name: str) -> bool:
        existed = self.people.pop(name, None) is not None
        self.enrolled_at.pop(name, None)
        return existed

    def __len__(self) -> int:
        return len(self.people)


class FaceIdentifier:
    """Embeds aligned faces and matches them against the database.

    Create it in the thread that uses it; the OpenCV DNN object is not
    guaranteed to be thread-safe.
    """

    def __init__(self, cfg):
        import cv2

        self.cv2 = cv2
        self.threshold = float(cfg.get("identity.threshold", 0.40))
        self.margin = float(cfg.get("identity.margin", 0.10))
        self.min_face_ratio = float(cfg.get("identity.min_face_ratio", 0.08))

        model_path = ensure_sface_model(
            cfg.path(cfg.get("identity.model_path", "models/face_recognition_sface.onnx")),
            bool(cfg.get("identity.auto_download_model", True)),
        )
        self.recognizer = cv2.FaceRecognizerSF.create(model_path, "")
        self.cosine = getattr(cv2, "FaceRecognizerSF_FR_COSINE", 0)
        self.db = PeopleDatabase(cfg.path(cfg.get("identity.database", "people.json")))

    # -- embedding ----------------------------------------------------------
    def embed(self, frame: np.ndarray, obs) -> Optional[np.ndarray]:
        """128-d feature for the face in `obs`, or None if it is unusable."""
        if not obs.face_found or obs.landmarks is None or obs.bbox is None:
            return None
        if len(obs.landmarks) <= max(ALIGN_LANDMARKS):
            return None  # the Haar backend has no landmarks
        if obs.face_ratio < self.min_face_ratio:
            return None

        x, y, w, h = obs.bbox
        row = np.array(
            [x, y, w, h] + [float(v) for i in ALIGN_LANDMARKS for v in obs.landmarks[i]] + [1.0],
            dtype=np.float32,
        )
        try:
            aligned = self.recognizer.alignCrop(frame, row)
            return self.recognizer.feature(aligned)
        except Exception as exc:
            log.debug("alignCrop/feature failed: %s", exc)
            return None

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(self.recognizer.match(a, b, self.cosine))

    # -- matching -----------------------------------------------------------
    def match(self, feature: Optional[np.ndarray]) -> Match:
        if feature is None or not self.db.people:
            return Match()

        scores: List[Tuple[str, float]] = []
        for name, samples in self.db.people.items():
            scores.append((name, max(self.similarity(feature, s) for s in samples)))
        scores.sort(key=lambda item: item[1], reverse=True)

        best_name, best_score = scores[0]
        second_name, second_score = scores[1] if len(scores) > 1 else (None, 0.0)

        result = Match(score=best_score, runner_up=second_name, runner_up_score=second_score)
        if best_score >= self.threshold and (best_score - second_score) >= self.margin:
            result.name = best_name
        return result

    def identify(self, frame: np.ndarray, obs) -> Match:
        return self.match(self.embed(frame, obs))


class IdentityVoter:
    """Majority vote over recent frames, so one bad embedding cannot rename
    the person on screen."""

    def __init__(self, window: int = 5, min_votes: int = 2):
        self.window = max(1, window)
        self.min_votes = max(1, min_votes)
        self.history: List[Optional[str]] = []
        self.current: Optional[str] = None
        self.score = 0.0

    def push(self, match: Match) -> Optional[str]:
        self.history.append(match.name)
        del self.history[: -self.window]
        if match.name:
            self.score = match.score

        counts: Dict[Optional[str], int] = {}
        for name in self.history:
            counts[name] = counts.get(name, 0) + 1
        winner, votes = max(counts.items(), key=lambda item: item[1])
        if winner is not None and votes >= self.min_votes:
            self.current = winner
        elif winner is None and votes >= self.min_votes:
            self.current = None
        return self.current

    def reset(self) -> None:
        self.history.clear()
        self.current = None
        self.score = 0.0
