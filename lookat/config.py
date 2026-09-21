"""Configuration loading: YAML file + defaults + CLI overrides."""

from __future__ import annotations

import ast
import copy
import os
import re
from typing import Any, Dict

import yaml

DEFAULTS: Dict[str, Any] = {
    "camera": {
        "backend": "auto",
        "device": None,      # None = auto: try 0, then scan
        "width": 640,
        "height": 480,
        "fps": 30,
        "flip_horizontal": True,
        "rotate": 0,
    },
    "detector": {
        "backend": "auto",
        "model_path": "models/face_landmarker.task",
        "auto_download_model": True,
        "detection_fps": 12,
        "min_confidence": 0.5,
    },
    "gaze": {
        "yaw_tolerance_deg": 22.0,
        "pitch_tolerance_deg": 18.0,
        "softness_deg": 8.0,
        "use_eye_offset": True,
        "eye_gain_yaw_deg": 30.0,
        "eye_gain_pitch_deg": 15.0,
        "compensate_position": True,
        "horizontal_fov_deg": 60.0,
        "yaw_offset_deg": 0.0,
        "pitch_offset_deg": 0.0,
        "require_eyes_open": True,
        "blink_threshold": 0.6,
        "ear_threshold": 0.15,
        "min_face_width_ratio": 0.05,
    },
    "identity": {
        "enabled": False,
        "model_path": "models/face_recognition_sface.onnx",
        "auto_download_model": True,
        "database": "people.json",
        "threshold": 0.40,
        "margin": 0.10,
        "min_face_ratio": 0.08,
        "recognize_every": 3,
        "vote_frames": 5,
        "min_votes": 2,
    },
    "attention": {
        "enter_score": 0.6,
        "exit_score": 0.4,
        "enter_frames": 3,
        "exit_frames": 6,
        "min_state_seconds": 0.8,
        "lost_face_grace_seconds": 0.7,
        "smoothing": 0.35,
    },
    "display": {
        "backend": "pygame",
        "fullscreen": True,
        "window_size": [1280, 720],
        "hide_cursor": True,
        "fps": 30,
        "fade_seconds": 0.35,
        "debug_overlay": False,
        "font_path": None,
        "scenes": {
            "idle": {
                "background": "#0d1117",
                "text": "...",
                "subtext": "",
                "text_color": "#3d4551",
                "subtext_color": "#272d36",
                "image": None,
                "image_fit": "cover",
                "font_scale": 0.22,
                "subtext_scale": 0.05,
            },
            "person": {},
            "attentive": {
                "background": "#0b3d2e",
                "text": "I see you",
                "subtext": "",
                "text_color": "#e8fff4",
                "subtext_color": "#7fd7b4",
                "image": None,
                "image_fit": "cover",
                "font_scale": 0.16,
                "subtext_scale": 0.045,
            },
        },
    },
    "updates": {
        "check_on_start": True,
    },
    "hooks": {
        "on_attentive": None,
        "on_idle": None,
        "debounce_seconds": 2.0,
    },
}


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(text: str) -> Any:
    """Turn a CLI string into a Python value ("3", "true", "[1,2]", "#fff")."""
    lowered = text.strip().lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", ""):
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


class Config:
    """Nested dict with dotted-path access."""

    def __init__(self, data: Dict[str, Any], base_dir: str = "."):
        self.data = data
        self.base_dir = base_dir

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def path(self, value: str | None) -> str | None:
        """Resolve a possibly-relative path against the config file's folder."""
        if not value:
            return None
        return value if os.path.isabs(value) else os.path.join(self.base_dir, value)


def load_config(config_path: str | None = None, overrides: list[str] | None = None) -> Config:
    data = copy.deepcopy(DEFAULTS)
    base_dir = os.getcwd()

    if config_path:
        with open(config_path, "r", encoding="utf-8") as handle:
            data = _deep_merge(data, yaml.safe_load(handle) or {})
        base_dir = os.path.dirname(os.path.abspath(config_path)) or "."

    cfg = Config(data, base_dir)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, _, raw = item.partition("=")
        cfg.set(key.strip(), _coerce(raw))
    return cfg


def patch_yaml_values(path: str, values: dict) -> list[str]:
    """Rewrite `key: value` lines in place, keeping comments and layout.

    Only keys that are unique in the file are touched; anything else is
    reported back so the caller can print it instead.
    """
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    unmatched = []
    for key, value in values.items():
        pattern = re.compile(rf"^(\s*){re.escape(key)}:(\s*)([^#\n]*)(#.*)?$")
        hits = [i for i, line in enumerate(lines) if pattern.match(line)]
        if len(hits) != 1:
            unmatched.append(key)
            continue
        index = hits[0]
        match = pattern.match(lines[index])
        indent, _, _, comment = match.groups()
        comment = f"  {comment.strip()}" if comment else ""
        lines[index] = f"{indent}{key}: {value}{comment}\n"

    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    return unmatched
