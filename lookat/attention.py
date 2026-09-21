"""Temporal smoothing: turn a noisy per-frame score into a stable boolean.

Three mechanisms keep the screen from flickering:
  * an exponential moving average over the raw score,
  * hysteresis (different enter/exit thresholds) plus consecutive-frame counts,
  * a minimum dwell time in each state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class AttentionState:
    attentive: bool = False
    score: float = 0.0
    raw: float = 0.0
    changed_at: float = 0.0
    face_found: bool = False


class AttentionTracker:
    def __init__(self, cfg):
        a = lambda k, d: cfg.get("attention." + k, d)  # noqa: E731
        self.enter_score = float(a("enter_score", 0.6))
        self.exit_score = float(a("exit_score", 0.4))
        self.enter_frames = int(a("enter_frames", 3))
        self.exit_frames = int(a("exit_frames", 6))
        self.min_state = float(a("min_state_seconds", 0.8))
        self.grace = float(a("lost_face_grace_seconds", 0.7))
        self.alpha = float(a("smoothing", 0.35))

        self.state = AttentionState()
        self._above = 0
        self._below = 0
        self._last_face_ts = 0.0
        self._started = False

    def update(self, obs, now: float | None = None) -> AttentionState:
        now = time.monotonic() if now is None else now
        if not self._started:
            # Adopt the caller's clock, and start outside the dwell window so
            # the very first observations can take effect immediately.
            self.state.changed_at = now - self.min_state
            self._last_face_ts = now - self.grace - 1.0
            self._started = True

        if obs.face_found:
            self._last_face_ts = now
            raw = obs.score
        elif now - self._last_face_ts <= self.grace:
            raw = self.state.score  # brief dropout: hold the current level
        else:
            raw = 0.0

        self.state.raw = raw
        self.state.face_found = obs.face_found
        self.state.score += self.alpha * (raw - self.state.score)
        score = self.state.score

        if score >= self.enter_score:
            self._above += 1
        else:
            self._above = 0
        if score <= self.exit_score:
            self._below += 1
        else:
            self._below = 0

        settled = (now - self.state.changed_at) >= self.min_state
        if settled:
            if not self.state.attentive and self._above >= self.enter_frames:
                self.state.attentive = True
                self.state.changed_at = now
                self._above = self._below = 0
            elif self.state.attentive and self._below >= self.exit_frames:
                self.state.attentive = False
                self.state.changed_at = now
                self._above = self._below = 0

        return self.state
