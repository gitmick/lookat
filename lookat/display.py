"""Fullscreen two-state display with a crossfade between the scenes."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def parse_colour(value, default=(0, 0, 0)):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value[:3])
    text = str(value).strip().lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    if len(text) != 6:
        log.warning("cannot parse colour %r, using default", value)
        return default
    return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))


class NullDisplay:
    """Headless mode: no window at all, only hooks and logging."""

    def __init__(self, cfg):
        self.debug = False

    def update(self, scene: str, state, preview=None, stats=None) -> bool:
        return True

    def close(self) -> None:
        pass


class PygameDisplay:
    def __init__(self, cfg):
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        self.pygame = pygame
        self.cfg = cfg
        pygame.init()
        pygame.font.init()

        size = tuple(cfg.get("display.window_size", [1280, 720]))
        self.logical_size = size
        self.fullscreen = bool(cfg.get("display.fullscreen", True))
        flags = pygame.FULLSCREEN | pygame.SCALED if self.fullscreen else pygame.RESIZABLE
        self.screen = pygame.display.set_mode(size, flags)
        pygame.display.set_caption("lookat")
        if cfg.get("display.hide_cursor", True):
            pygame.mouse.set_visible(False)

        self.clock = pygame.time.Clock()
        self.fps = int(cfg.get("display.fps", 30))
        self.fade = max(0.01, float(cfg.get("display.fade_seconds", 0.35)))
        self.debug = bool(cfg.get("display.debug_overlay", False))
        self.font_path = cfg.path(cfg.get("display.font_path"))
        self.mode = str(cfg.get("display.mode", "text")).lower()

        # Crossfade state: `_shown` is fully visible, `_incoming` fades in
        # over it. Scene names are "idle", "attentive" or "person.<name>".
        self._shown = "idle"
        self._incoming = "idle"
        self.alpha = 1.0
        self.menu = None          # set by the app; drawn on top when open
        self._scene_cache: dict[str, object] = {}
        self._debug_font = pygame.font.Font(None, 22)
        self._build_scenes()

    # -- scene rendering ----------------------------------------------------
    def _font(self, pixels: int):
        pixels = max(12, int(pixels))
        if self.font_path and os.path.exists(self.font_path):
            return self.pygame.font.Font(self.font_path, pixels)
        return self.pygame.font.Font(None, pixels)

    def _wrap(self, text: str, font, max_width: int) -> list[str]:
        lines: list[str] = []
        for paragraph in str(text).split("\n"):
            words, current = paragraph.split(" "), ""
            for word in words:
                candidate = f"{current} {word}".strip()
                if font.size(candidate)[0] <= max_width or not current:
                    current = candidate
                else:
                    lines.append(current)
                    current = word
            lines.append(current)
        return lines

    IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

    def images_folder(self) -> Path:
        return Path(self.cfg.path(self.cfg.get("display.images.folder", "images")))

    def _image_basename(self, scene: str) -> str:
        """idle -> "idle";  person.anna -> "person-anna"."""
        return scene.replace(".", "-")

    def image_for(self, scene: str) -> Optional[Path]:
        """The picture for a scene, or None. A person with no picture of
        their own falls back to the generic `attentive` one."""
        folder = self.images_folder()
        candidates = [self._image_basename(scene)]
        if scene.startswith("person."):
            candidates.append("attentive")
        for base in candidates:
            for suffix in self.IMAGE_SUFFIXES:
                path = folder / f"{base}{suffix}"
                if path.exists():
                    return path
        return None

    def _missing_image_spec(self, scene: str) -> dict:
        wanted = f"{self._image_basename(scene)}.jpg"
        return {
            "background": "#1b1418",
            "text": "no picture",
            "subtext": f"put one at  {self.images_folder() / wanted}",
            "text_color": "#8c7a82",
            "subtext_color": "#5d5057",
            "font_scale": 0.10,
            "subtext_scale": 0.032,
            "image": None,
        }

    def _scene_spec(self, name: str) -> dict:
        """A per-person scene inherits from `attentive` for anything it omits,
        so you only have to write the parts that differ."""
        if name.startswith("person."):
            base = dict(self.cfg.get("display.scenes.attentive", {}) or {})
            base.update(self.cfg.get(f"display.scenes.{name}", {}) or {})
            return base
        return dict(self.cfg.get(f"display.scenes.{name}", {}) or {})

    def scene_names(self) -> list[str]:
        people = self.cfg.get("display.scenes.person", {}) or {}
        return ["idle", "attentive"] + [f"person.{name}" for name in people]

    def _render_scene(self, name: str):
        pygame = self.pygame
        if self.mode == "images":
            picture = self.image_for(name)
            spec = ({"background": "#000000", "image": str(picture),
                     "image_fit": self.cfg.get("display.images.fit", "cover"),
                     "text": "", "subtext": ""}
                    if picture else self._missing_image_spec(name))
        else:
            spec = self._scene_spec(name)
        width, height = self.screen.get_size()
        surface = pygame.Surface((width, height)).convert()
        surface.fill(parse_colour(spec.get("background"), (0, 0, 0)))

        image_path = spec.get("image")
        if image_path and not os.path.isabs(image_path):
            image_path = self.cfg.path(image_path)
        if image_path:
            if os.path.exists(image_path):
                try:
                    image = pygame.image.load(image_path).convert()
                    iw, ih = image.get_size()
                    fit = str(spec.get("image_fit", "cover")).lower()
                    chooser = max if fit == "cover" else min
                    scale = chooser(width / iw, height / ih)
                    image = pygame.transform.smoothscale(
                        image, (max(1, int(iw * scale)), max(1, int(ih * scale)))
                    )
                    rect = image.get_rect(center=(width // 2, height // 2))
                    surface.blit(image, rect)
                except Exception as exc:
                    log.error("could not load image %s: %s", image_path, exc)
            else:
                log.error("scene image not found: %s", image_path)

        text = spec.get("text") or ""
        subtext = spec.get("subtext") or ""
        blocks = []
        if text:
            font = self._font(height * float(spec.get("font_scale", 0.2)))
            colour = parse_colour(spec.get("text_color"), (255, 255, 255))
            blocks += [(line, font, colour) for line in self._wrap(text, font, int(width * 0.9))]
        if subtext:
            font = self._font(height * float(spec.get("subtext_scale", 0.05)))
            colour = parse_colour(spec.get("subtext_color"), (180, 180, 180))
            blocks.append((None, None, None))  # spacer
            blocks += [(line, font, colour) for line in self._wrap(subtext, font, int(width * 0.9))]

        rendered = [
            (font.render(line, True, colour), font.get_linesize())
            if line is not None
            else (None, int(height * 0.04))
            for line, font, colour in blocks
        ]
        total = sum(step for _, step in rendered)
        y = (height - total) // 2
        for glyph, step in rendered:
            if glyph is not None:
                surface.blit(glyph, glyph.get_rect(centerx=width // 2, top=y))
            y += step
        return surface

    def _build_scenes(self) -> None:
        self._scene_cache = {name: self._render_scene(name) for name in self.scene_names()}

    def _surface(self, name: str):
        """Render unknown scenes on demand (a person enrolled while running)."""
        if name not in self._scene_cache:
            log.debug("rendering scene %s on demand", name)
            self._scene_cache[name] = self._render_scene(name)
        return self._scene_cache[name]

    # -- main loop ----------------------------------------------------------
    def _set_mode(self, size, flags) -> bool:
        """set_mode, retrying once through a display re-init.

        Switching to fullscreen can fail with "failed to create renderer" when
        SDL cannot build a scaled renderer for the requested size; tearing the
        display down and back up clears that.
        """
        pygame = self.pygame
        for reinit in (False, True):
            try:
                if reinit:
                    pygame.display.quit()
                    pygame.display.init()
                self.screen = pygame.display.set_mode(size, flags)
                if self.cfg.get("display.hide_cursor", True):
                    pygame.mouse.set_visible(False)
                return True
            except pygame.error as exc:
                log.warning("set_mode(%s, %s) failed: %s", size, flags, exc)
        return False

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.cfg.set("display.mode", mode)
        self.reload_scenes()

    def set_fullscreen(self, value: bool) -> bool:
        """Switch between fullscreen and windowed. Returns False (and stays
        on something usable) if the platform refuses."""
        pygame = self.pygame
        if value == self.fullscreen:
            return True

        was = self.fullscreen
        if value:
            attempts = [(self.logical_size, pygame.FULLSCREEN | pygame.SCALED),
                        ((0, 0), pygame.FULLSCREEN)]
        else:
            attempts = [(self.logical_size, pygame.RESIZABLE)]

        for size, flags in attempts:
            if self._set_mode(size, flags):
                self.fullscreen = value
                self.reload_scenes()
                return True

        # Nothing worked -- get back to a window we can still draw on.
        log.error("could not switch fullscreen; restoring the previous mode")
        restore = ((self.logical_size, pygame.FULLSCREEN | pygame.SCALED) if was
                   else (self.logical_size, pygame.RESIZABLE))
        if not self._set_mode(*restore):
            self._set_mode(self.logical_size, 0)
        self.fullscreen = was
        self.reload_scenes()
        return False

    def reload_scenes(self) -> None:
        """Re-render every scene, e.g. after a person was added to the config."""
        self._scene_cache.clear()
        self._build_scenes()
        if self._shown not in self._scene_cache:
            self._shown = "idle"

    def _handle_events(self) -> bool:
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            # While the menu is open it gets first refusal on every key.
            if self.menu is not None and self.menu.open:
                if self.menu.handle(event, pygame):
                    continue

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m and self.menu is not None:
                    self.menu.toggle()
                    continue
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if event.key == pygame.K_d:
                    self.debug = not self.debug
                if event.key == pygame.K_f:
                    self.set_fullscreen(not self.fullscreen)
            if event.type == pygame.VIDEORESIZE and not self.fullscreen:
                self.screen = pygame.display.set_mode(
                    (event.w, event.h), pygame.RESIZABLE
                )
                self._build_scenes()
        return True

    def _blit_preview(self, preview: np.ndarray) -> None:
        pygame = self.pygame
        width, height = self.screen.get_size()
        target_w = max(160, width // 4)
        ph, pw = preview.shape[:2]
        target_h = int(ph * target_w / pw)
        rgb = preview[:, :, ::-1]  # BGR -> RGB
        surface = pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))
        surface = pygame.transform.smoothscale(surface, (target_w, target_h))
        self.screen.blit(surface, (width - target_w - 12, 12))

    def update(self, scene: str, state, preview=None, stats=None) -> bool:
        pygame = self.pygame
        if not self._handle_events():
            return False

        dt = self.clock.tick(self.fps) / 1000.0

        if scene != self._incoming:
            # Start a new fade. If one was already running, the partially
            # faded scene becomes the new starting point.
            self._shown = self._incoming if self.alpha >= 0.5 else self._shown
            self._incoming = scene
            self.alpha = 0.0

        self.alpha = min(1.0, self.alpha + dt / self.fade)

        self.screen.blit(self._surface(self._shown), (0, 0))
        if self.alpha > 0.001 and self._incoming != self._shown:
            top = self._surface(self._incoming)
            top.set_alpha(int(self.alpha * 255))
            self.screen.blit(top, (0, 0))
        if self.alpha >= 1.0:
            self._shown = self._incoming

        if self.debug:
            if preview is not None:
                self._blit_preview(preview)
            lines = [
                f"scene    {scene}",
                f"score    {state.score:.2f}  (raw {state.raw:.2f})",
                f"face     {'yes' if state.face_found else 'no'}",
            ]
            for key, value in (stats or {}).items():
                lines.append(f"{key:<8} {value}")
            lines.append("keys: m menu  d debug  f fullscreen  q quit")
            y = 12
            for line in lines:
                glyph = self._debug_font.render(line, True, (235, 235, 235))
                shadow = self._debug_font.render(line, True, (0, 0, 0))
                self.screen.blit(shadow, (13, y + 1))
                self.screen.blit(glyph, (12, y))
                y += glyph.get_height() + 2

        if self.menu is not None and self.menu.open:
            self.menu.draw(self.screen, pygame, self._font)

        pygame.display.flip()
        return True

    def close(self) -> None:
        self.pygame.quit()


def build_display(cfg):
    backend = str(cfg.get("display.backend", "pygame")).lower()
    if backend in ("none", "null", "headless"):
        return NullDisplay(cfg)
    return PygameDisplay(cfg)
