"""In-app settings menu, drawn over the running display.

Press `m` to open. Up/Down to move, Enter to activate, Esc to close.
Entries are supplied by the application as `MenuItem`s, so the menu itself
knows nothing about cameras or calibration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

log = logging.getLogger(__name__)

PANEL_BG = (16, 20, 27, 236)
PANEL_EDGE = (70, 82, 100)
TEXT = (232, 238, 246)
DIM = (140, 154, 172)
ACCENT = (108, 196, 255)
SELECT_BG = (36, 52, 74)
BUSY = (255, 201, 92)
OK = (126, 224, 160)
BAD = (255, 130, 130)


@dataclass
class MenuItem:
    label: str
    action: Callable[[], None] | None = None
    # Returns the text shown on the right, e.g. "on" / "camera 0".
    value: Callable[[], str] | None = None
    hint: str = ""
    enabled: Callable[[], bool] = lambda: True


@dataclass
class Menu:
    items: List[MenuItem] = field(default_factory=list)
    open: bool = False
    index: int = 0
    title: str = "Settings"

    # Transient status line, e.g. "Calibrating... 4s" or "Saved".
    status: str = ""
    status_kind: str = "info"        # info | busy | ok | bad

    # Text entry sub-mode (used for "learn a new person").
    prompt: Optional[str] = None
    text: str = ""
    on_submit: Optional[Callable[[str], None]] = None

    # List-picker sub-mode (used for "remove a person", and its confirmation).
    choices: Optional[List[str]] = None
    choice_prompt: str = ""
    choice_index: int = 0
    on_choice: Optional[Callable[[str], None]] = None

    def set_status(self, message: str, kind: str = "info") -> None:
        self.status = message
        self.status_kind = kind

    def ask(self, prompt: str, on_submit: Callable[[str], None], initial: str = "") -> None:
        self.prompt = prompt
        self.text = initial
        self.on_submit = on_submit

    def cancel_prompt(self) -> None:
        self.prompt = None
        self.text = ""
        self.on_submit = None

    def choose(self, prompt: str, options: List[str],
               on_pick: Callable[[str], None]) -> None:
        """Pick one of `options`. Used instead of free text where the valid
        answers are known, so nobody can delete the wrong person by typo."""
        self.choices = list(options)
        self.choice_prompt = prompt
        self.choice_index = 0
        self.on_choice = on_pick

    def cancel_choice(self) -> None:
        self.choices = None
        self.choice_prompt = ""
        self.choice_index = 0
        self.on_choice = None

    def toggle(self) -> None:
        self.open = not self.open
        if not self.open:
            self.cancel_prompt()
            self.cancel_choice()

    # -- input --------------------------------------------------------------
    def handle(self, event, pygame) -> bool:
        """Consume a pygame event. Returns True if the menu handled it."""
        if event.type != pygame.KEYDOWN:
            return False

        if self.prompt is not None:
            return self._handle_prompt(event, pygame)
        if self.choices is not None:
            return self._handle_choice(event, pygame)

        if event.key in (pygame.K_ESCAPE, pygame.K_m):
            self.open = False
            return True
        if event.key in (pygame.K_UP, pygame.K_k):
            self._move(-1)
            return True
        if event.key in (pygame.K_DOWN, pygame.K_j):
            self._move(1)
            return True
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE, pygame.K_RIGHT):
            item = self.current
            if item and item.action and item.enabled():
                try:
                    item.action()
                except Exception as exc:              # a menu must not crash the app
                    log.exception("menu action %r failed", item.label)
                    self.set_status(f"{item.label} failed: {exc}", "bad")
            return True
        return False

    def _handle_prompt(self, event, pygame) -> bool:
        if event.key == pygame.K_ESCAPE:
            self.cancel_prompt()
            return True
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            name, submit = self.text.strip(), self.on_submit
            self.cancel_prompt()
            if name and submit:
                try:
                    submit(name)
                except Exception as exc:
                    log.exception("menu prompt handler failed")
                    self.set_status(f"failed: {exc}", "bad")
            return True
        if event.key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
            return True
        char = getattr(event, "unicode", "")
        if char and char.isprintable() and len(self.text) < 24:
            self.text += char
        return True

    def _handle_choice(self, event, pygame) -> bool:
        if event.key == pygame.K_ESCAPE:
            self.cancel_choice()
            return True
        if event.key in (pygame.K_UP, pygame.K_k):
            self.choice_index = (self.choice_index - 1) % len(self.choices)
            return True
        if event.key in (pygame.K_DOWN, pygame.K_j):
            self.choice_index = (self.choice_index + 1) % len(self.choices)
            return True
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            picked, handler = self.choices[self.choice_index], self.on_choice
            self.cancel_choice()
            if handler:
                try:
                    handler(picked)
                except Exception as exc:
                    log.exception("menu choice handler failed")
                    self.set_status(f"failed: {exc}", "bad")
            return True
        return True    # swallow everything else while a picker is open

    def _move(self, step: int) -> None:
        if not self.items:
            return
        for _ in range(len(self.items)):
            self.index = (self.index + step) % len(self.items)
            if self.items[self.index].enabled():
                return

    @property
    def current(self) -> Optional[MenuItem]:
        return self.items[self.index] if 0 <= self.index < len(self.items) else None

    # -- drawing ------------------------------------------------------------
    def draw(self, screen, pygame, font_for) -> None:
        width, height = screen.get_size()
        panel_w = min(int(width * 0.74), 760)
        row_h = max(30, int(height * 0.062))
        title_font = font_for(int(row_h * 0.72))
        item_font = font_for(int(row_h * 0.56))
        small_font = font_for(int(row_h * 0.42))

        rows = len(self.items)
        # Sub-modes need extra room below the list.
        if self.prompt is not None:
            bottom = row_h * 2.7
        elif self.choices is not None:
            bottom = row_h * (1.5 + 0.8 * len(self.choices))
        else:
            bottom = row_h * 1.9
        panel_h = int(row_h * 1.5 + rows * row_h + bottom)
        panel_h = min(panel_h, int(height * 0.94))
        x = (width - panel_w) // 2
        y = (height - panel_h) // 2

        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        pygame.draw.rect(panel, PANEL_EDGE, panel.get_rect(), width=2, border_radius=12)
        screen.blit(panel, (x, y))

        cursor_y = y + int(row_h * 0.4)
        screen.blit(title_font.render(self.title, True, ACCENT), (x + 24, cursor_y))
        cursor_y += int(row_h * 1.1)

        for i, item in enumerate(self.items):
            active = i == self.index
            greyed = not item.enabled()
            if active:
                pygame.draw.rect(screen, SELECT_BG,
                                 pygame.Rect(x + 12, cursor_y - 3, panel_w - 24, row_h - 4),
                                 border_radius=7)
            colour = DIM if greyed else (ACCENT if active else TEXT)
            screen.blit(item_font.render(item.label, True, colour), (x + 26, cursor_y))
            if item.value:
                try:
                    value = item.value()
                except Exception:          # a broken accessor must not kill the UI
                    value = "?"
                glyph = item_font.render(value, True, DIM if greyed else TEXT)
                screen.blit(glyph, (x + panel_w - 26 - glyph.get_width(), cursor_y))
            cursor_y += row_h

        cursor_y += int(row_h * 0.25)
        if self.prompt is not None:
            screen.blit(small_font.render(self.prompt, True, TEXT), (x + 26, cursor_y))
            cursor_y += int(row_h * 0.55)
            box = pygame.Rect(x + 26, cursor_y, panel_w - 52, int(row_h * 0.85))
            pygame.draw.rect(screen, (8, 11, 16), box, border_radius=6)
            pygame.draw.rect(screen, ACCENT, box, width=2, border_radius=6)
            caret = "_" if (pygame.time.get_ticks() // 450) % 2 else " "
            screen.blit(item_font.render(self.text + caret, True, TEXT),
                        (box.x + 10, box.y + 4))
        elif self.choices is not None:
            screen.blit(small_font.render(self.choice_prompt, True, TEXT), (x + 26, cursor_y))
            cursor_y += int(row_h * 0.6)
            for i, option in enumerate(self.choices):
                picked = i == self.choice_index
                if picked:
                    pygame.draw.rect(screen, SELECT_BG,
                                     pygame.Rect(x + 24, cursor_y - 2, panel_w - 48,
                                                 int(row_h * 0.78)),
                                     border_radius=6)
                colour = ACCENT if picked else TEXT
                screen.blit(item_font.render(option, True, colour), (x + 38, cursor_y))
                cursor_y += int(row_h * 0.8)
        else:
            hint = (self.current.hint if self.current else "") or ""
            if self.status:
                colour = {"busy": BUSY, "ok": OK, "bad": BAD}.get(self.status_kind, TEXT)
                screen.blit(small_font.render(self.status, True, colour), (x + 26, cursor_y))
                cursor_y += int(row_h * 0.5)
            if hint:
                screen.blit(small_font.render(hint, True, DIM), (x + 26, cursor_y))

        if self.prompt is not None:
            footer = "enter  confirm     esc  cancel"
        elif self.choices is not None:
            footer = "up/down  choose     enter  confirm     esc  cancel"
        else:
            footer = "up/down  move     enter  select     esc  close"
        glyph = small_font.render(footer, True, DIM)
        screen.blit(glyph, (x + 26, y + panel_h - glyph.get_height() - 12))
