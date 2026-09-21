"""Fire-and-forget shell commands on attention state changes."""

from __future__ import annotations

import logging
import shlex
import subprocess
import time

log = logging.getLogger(__name__)


class HookRunner:
    def __init__(self, cfg):
        self.on_attentive = cfg.get("hooks.on_attentive")
        self.on_idle = cfg.get("hooks.on_idle")
        self.debounce = float(cfg.get("hooks.debounce_seconds", 2.0))
        self._last_fire = 0.0
        self._procs: list[subprocess.Popen] = []

    @property
    def enabled(self) -> bool:
        return bool(self.on_attentive or self.on_idle)

    def fire(self, attentive: bool, person: str = "") -> None:
        """`{person}` in a hook command is replaced with the recognised name
        (empty when nobody is enrolled or the face is unknown)."""
        command = self.on_attentive if attentive else self.on_idle
        if not command:
            return
        if "{person}" in command:
            command = command.replace("{person}", shlex.quote(person))
        now = time.monotonic()
        if now - self._last_fire < self.debounce:
            log.debug("hook debounced: %s", command)
            return
        self._last_fire = now
        try:
            log.info("hook: %s", command)
            self._procs.append(subprocess.Popen(command, shell=True))  # noqa: S602
        except Exception as exc:
            log.error("hook failed (%s): %s", command, exc)
        self._procs = [p for p in self._procs if p.poll() is None]

    def shutdown(self) -> None:
        for proc in self._procs:
            if proc.poll() is None:
                proc.terminate()


def quote(command: str) -> str:
    return shlex.quote(command)
