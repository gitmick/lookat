"""Application wiring: camera thread -> detector thread -> display loop."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .attention import AttentionState, AttentionTracker
from .camera import Camera
from .config import load_config
from .detector import Observation, build_detector
from .display import build_display
from .hooks import HookRunner

log = logging.getLogger("lookat")


@dataclass
class Shared:
    lock: threading.Lock = field(default_factory=threading.Lock)
    state: AttentionState = field(default_factory=AttentionState)
    observation: Observation = field(default_factory=Observation)
    preview: Optional[np.ndarray] = None
    detector_fps: float = 0.0
    backend: str = "?"
    error: Optional[str] = None
    task_status: str = ""
    task_status_kind: str = "info"
    task_result: Optional[dict] = None


class DetectionWorker(threading.Thread):
    """Owns the detector. MediaPipe objects are not thread-safe, so the
    detector is constructed here rather than handed in."""

    def __init__(self, cfg, camera: Camera, shared: Shared, hooks: HookRunner, stop: threading.Event):
        super().__init__(name="detector", daemon=True)
        self.cfg = cfg
        self.camera = camera
        self.shared = shared
        self.hooks = hooks
        self.stop_event = stop
        self.tracker = AttentionTracker(cfg)
        self.interval = 1.0 / max(1.0, float(cfg.get("detector.detection_fps", 12)))
        self.want_preview = bool(cfg.get("display.debug_overlay", False))
        self.identity_enabled = bool(cfg.get("identity.enabled", False))
        self.recognize_every = max(1, int(cfg.get("identity.recognize_every", 3)))
        self.identifier = None
        self.detector = None
        self._task_lock = threading.Lock()
        self._task: Optional[dict] = None
        self._removals: list[str] = []

    def _ensure_identifier(self) -> None:
        """Build the recogniser on first use, inside the worker thread."""
        if self.identifier is not None or not self.identity_enabled:
            return
        try:
            from .identity import FaceIdentifier

            self.identifier = FaceIdentifier(self.cfg)
            if self.identifier.db.people:
                log.info("recognising: %s", ", ".join(sorted(self.identifier.db.people)))
            else:
                log.info("face recognition on, but nobody is enrolled yet")
        except Exception as exc:
            log.error("face recognition unavailable (%s); continuing without it", exc)
            self.identity_enabled = False

    def request_task(self, kind: str, seconds: float, name: str = "") -> bool:
        """Ask the worker to run a calibration or enrolment on the live camera.

        Both need frames, and the camera is already open here -- opening a
        second capture of the same webcam fails on most systems.
        """
        with self._task_lock:
            if self._task is not None:
                return False
            now = time.monotonic()
            self._task = {
                "kind": kind, "name": name, "lead": 3.0,
                "start": now + 3.0, "end": now + 3.0 + seconds,
                "samples": [], "frames": 0, "rejected": {},
            }
        return True

    def request_removal(self, name: str) -> None:
        """Delete an enrolled person. Queued for the worker thread, because
        recognition iterates the same dictionary on every frame."""
        with self._task_lock:
            self._removals.append(name)

    def _process_removals(self) -> None:
        with self._task_lock:
            pending, self._removals = self._removals, []
        for name in pending:
            if self.identifier is None:
                self._publish_task("Face recognition is not running", "bad")
                continue
            if self.identifier.db.forget(name):
                self.identifier.db.save()
                log.info("removed enrolled person %r", name)
                self._publish_task(f"Removed {name}", "ok",
                                   {"kind": "forget", "ok": True, "name": name})
            else:
                self._publish_task(f"{name} was not enrolled", "bad",
                                   {"kind": "forget", "ok": False, "name": name})

    @property
    def task_active(self) -> bool:
        return self._task is not None

    def cancel_task(self) -> None:
        with self._task_lock:
            self._task = None

    def _advance_task(self, task: dict, frame, obs, now: float) -> None:
        from .enroll import _quality_ok

        if now < task["start"]:
            left = int(task["start"] - now) + 1
            where = "the screen" if task["kind"] == "calibrate" else "the camera"
            self._publish_task(f"Look at {where}... {left}", "busy")
            return

        if now < task["end"]:
            task["frames"] += 1
            if task["kind"] == "calibrate":
                if obs.face_found:
                    task["samples"].append((
                        obs.dev_yaw + float(self.cfg.get("gaze.yaw_offset_deg", 0.0)),
                        obs.dev_pitch + float(self.cfg.get("gaze.pitch_offset_deg", 0.0)),
                    ))
            else:
                ok, why = _quality_ok(obs, float(self.cfg.get("identity.min_face_ratio", 0.08)))
                if ok and self.identifier is not None:
                    feature = self.identifier.embed(frame, obs)
                    if feature is not None:
                        task["samples"].append(feature)
                elif not ok:
                    key = why.split(" (")[0]
                    task["rejected"][key] = task["rejected"].get(key, 0) + 1
            left = task["end"] - now
            self._publish_task(
                f"{'Calibrating' if task['kind'] == 'calibrate' else 'Learning ' + task['name']}"
                f"... {left:.0f}s   ({len(task['samples'])} samples)", "busy")
            return

        with self._task_lock:
            self._task = None
        if task["kind"] == "calibrate":
            self._finish_calibration(task)
        else:
            self._finish_enrolment(task)

    def _publish_task(self, message: str, kind: str = "info", result: dict | None = None) -> None:
        with self.shared.lock:
            self.shared.task_status = message
            self.shared.task_status_kind = kind
            if result is not None:
                self.shared.task_result = result

    def _finish_calibration(self, task: dict) -> None:
        import numpy as np

        from .detector import GazeScorer

        samples = task["samples"]
        if len(samples) < 10:
            self._publish_task(
                f"Calibration failed: only {len(samples)} frames with a face", "bad",
                {"kind": "calibrate", "ok": False})
            return

        data = np.array(samples)
        yaw = float(np.median(data[:, 0]))
        pitch = float(np.median(data[:, 1]))
        yaw_spread = float(np.percentile(np.abs(data[:, 0] - yaw), 90))
        pitch_spread = float(np.percentile(np.abs(data[:, 1] - pitch), 90))
        values = {
            "yaw_offset_deg": round(yaw, 1),
            "pitch_offset_deg": round(pitch, 1),
            "yaw_tolerance_deg": round(max(12.0, yaw_spread + 8.0), 1),
            "pitch_tolerance_deg": round(max(10.0, pitch_spread + 8.0), 1),
        }
        for key, value in values.items():
            self.cfg.set(f"gaze.{key}", value)
        # Apply straight away -- the scorer caches these at construction.
        self.detector.scorer = GazeScorer(self.cfg)

        self._publish_task(
            f"Calibrated from {len(samples)} samples "
            f"(yaw {values['yaw_offset_deg']:+.1f}, pitch {values['pitch_offset_deg']:+.1f})",
            "ok", {"kind": "calibrate", "ok": True, "values": values})

    def _finish_enrolment(self, task: dict) -> None:
        from .enroll import _dedupe

        name, samples = task["name"], task["samples"]
        if self.identifier is None:
            self._publish_task("Face recognition is not available", "bad",
                               {"kind": "enroll", "ok": False})
            return
        if len(samples) < 5:
            worst = max(task["rejected"].items(), key=lambda kv: kv[1])[0] if task["rejected"] else ""
            detail = f" (mostly: {worst})" if worst else ""
            self._publish_task(
                f"Could not learn {name}: only {len(samples)} usable frames{detail}", "bad",
                {"kind": "enroll", "ok": False})
            return

        kept = _dedupe(self.identifier, samples, 8)
        self.identifier.db.add(name, kept, replace=True)
        self.identifier.db.save()
        self._publish_task(
            f"Learned {name} from {len(kept)} templates", "ok",
            {"kind": "enroll", "ok": True, "name": name})

    def run(self) -> None:
        try:
            detector = build_detector(self.cfg)
            self.detector = detector
        except Exception as exc:
            log.exception("detector could not start")
            with self.shared.lock:
                self.shared.error = str(exc)
            self.stop_event.set()
            return

        from .identity import IdentityVoter

        voter = IdentityVoter(
            int(self.cfg.get("identity.vote_frames", 5)),
            int(self.cfg.get("identity.min_votes", 2)),
        )
        self._ensure_identifier()

        with self.shared.lock:
            self.shared.backend = detector.name

        frame_count = 0
        last_seq = -1
        last_attentive = None
        smoothed_fps = 0.0
        t_start = time.monotonic()

        while not self.stop_event.is_set():
            loop_start = time.monotonic()
            seq, frame = self.camera.read()
            if frame is None or seq == last_seq:
                time.sleep(0.005)
                continue
            last_seq = seq

            ts_ms = int((loop_start - t_start) * 1000)
            try:
                obs = detector.process(frame, ts_ms)
            except Exception as exc:
                log.exception("detection failed: %s", exc)
                time.sleep(0.2)
                continue
            obs.ts = loop_start
            frame_count += 1

            if self.identity_enabled and self.identifier is None:
                self._ensure_identifier()      # switched on from the menu
            identifier = self.identifier if self.identity_enabled else None

            if identifier is not None:
                if not obs.face_found:
                    voter.reset()
                elif frame_count % self.recognize_every == 0:
                    try:
                        voter.push(identifier.identify(frame, obs))
                    except Exception as exc:
                        log.debug("recognition failed: %s", exc)
                if voter.current:
                    obs.identity = voter.current
                    obs.identity_score = voter.score
            else:
                voter.reset()

            if self._removals:
                self._process_removals()
                voter.reset()

            task = self._task
            if task is not None:
                self._advance_task(task, frame, obs, loop_start)

            state = self.tracker.update(obs, loop_start)

            preview = None
            if self.want_preview:
                from .overlay import draw_observation

                preview = draw_observation(frame, obs, state.attentive)

            elapsed = max(1e-6, time.monotonic() - loop_start)
            smoothed_fps = 0.8 * smoothed_fps + 0.2 * (1.0 / elapsed) if smoothed_fps else 1.0 / elapsed

            with self.shared.lock:
                self.shared.state = state
                self.shared.observation = obs
                self.shared.preview = preview
                self.shared.detector_fps = smoothed_fps

            if last_attentive is None or state.attentive != last_attentive:
                who = obs.identity or ""
                label = "LOOKING" if state.attentive else "away"
                if state.attentive and who:
                    label = f"LOOKING ({who})"
                if last_attentive is None:
                    # Report where we start, but do not fire hooks for it --
                    # startup is not a state *change*.
                    log.info("initial state: %s (score %.2f)", label, state.score)
                else:
                    log.info("state -> %s (score %.2f)", label, state.score)
                    self.hooks.fire(state.attentive, who)
                last_attentive = state.attentive

            remaining = self.interval - (time.monotonic() - loop_start)
            if remaining > 0:
                self.stop_event.wait(remaining)

        try:
            detector.close()
        except Exception:
            pass


PERSON_COLOURS = ["#1d3557", "#5c2018", "#2d4739", "#4a2f5e", "#5a4a12", "#12414a"]


def build_menu(cfg, display, camera, worker, config_path, stop):
    """Wire the on-screen menu to the running app."""
    from .camera import probe_cameras
    from .config import patch_yaml_values
    from .menu import Menu, MenuItem

    from .paths import checkout_dir

    menu = Menu(title="lookat  -  settings")
    cameras: dict = {"list": None, "scanning": False}
    updates: dict = {"checked": False, "checking": False, "available": False,
                     "message": "not checked"}

    def check_updates(force: bool = False):
        from .update import check_for_update

        def work():
            try:
                available, message = check_for_update()
                updates.update(available=available, message=message, checked=True)
            except Exception as exc:
                updates.update(available=False, message=f"check failed: {exc}", checked=True)
            finally:
                updates["checking"] = False

        if updates["checking"] or (updates["checked"] and not force):
            return
        if checkout_dir() is None:
            updates.update(checked=True, message="not a git checkout")
            return
        updates["checking"] = True
        threading.Thread(target=work, name="update-check", daemon=True).start()

    def scan_cameras():
        def work():
            try:
                found = probe_cameras(skip={camera.device} if isinstance(camera.device, int) else None)
                cameras["list"] = [index for index, _, _ in found] or [camera.device]
            except Exception as exc:
                log.error("camera scan failed: %s", exc)
                cameras["list"] = [camera.device]
            finally:
                cameras["scanning"] = False

        if cameras["scanning"] or cameras["list"] is not None:
            return
        cameras["scanning"] = True
        threading.Thread(target=work, name="camera-scan", daemon=True).start()

    # -- actions ------------------------------------------------------------
    def do_calibrate():
        if worker.request_task("calibrate", 8.0):
            menu.set_status("Starting calibration...", "busy")

    def do_learn():
        if not worker.identity_enabled:
            # Just turn it on rather than making the user find the other entry.
            worker.identity_enabled = True
            cfg.set("identity.enabled", True)
            menu.set_status("Switched face recognition on", "info")
        menu.ask("Name of the person:", lambda name: (
            menu.set_status(f"Starting to learn {name}...", "busy")
            if worker.request_task("enroll", 10.0, name) else None))

    def do_remove_person():
        identifier = worker.identifier
        if identifier is None or not identifier.db.people:
            menu.set_status("Nobody is enrolled", "info")
            return

        def confirm(name: str) -> None:
            menu.choose(f"Delete {name}'s face data for good?",
                        [f"No, keep {name}", f"Yes, remove {name}"],
                        lambda pick: finish(pick, name))

        def finish(pick: str, name: str) -> None:
            if not pick.startswith("Yes"):
                menu.set_status(f"Kept {name}", "info")
                return
            worker.request_removal(name)

        menu.choose("Remove which person?", sorted(identifier.db.people), confirm)

    def do_next_camera():
        if cameras["list"] is None:
            scan_cameras()
            menu.set_status("Scanning for cameras...", "busy")
            return
        options = cameras["list"]
        if len(options) < 2:
            menu.set_status(f"Only one camera found (device {camera.device})", "info")
            return
        try:
            nxt = options[(options.index(camera.device) + 1) % len(options)]
        except ValueError:
            nxt = options[0]
        menu.set_status(f"Switching to camera {nxt}...", "busy")
        if camera.switch_device(nxt):
            cfg.set("camera.device", nxt)
            menu.set_status(f"Now using camera {nxt}", "ok")
        else:
            menu.set_status(f"Camera {nxt} would not open, kept {camera.device}", "bad")

    def do_toggle_fullscreen():
        if display.set_fullscreen(not display.fullscreen):
            menu.set_status(f"Fullscreen {'on' if display.fullscreen else 'off'}", "ok")
        else:
            menu.set_status("This display refused the mode change", "bad")
        cfg.set("display.fullscreen", display.fullscreen)

    def do_toggle_identity():
        enabled = not worker.identity_enabled
        worker.identity_enabled = enabled
        cfg.set("identity.enabled", enabled)
        menu.set_status(f"Face recognition {'on' if enabled else 'off'}", "ok")

    def do_update():
        from .update import apply_update

        if checkout_dir() is None:
            menu.set_status("Not a git checkout -- reinstall by hand", "bad")
            return
        if not updates["checked"] or updates["checking"]:
            check_updates(force=True)
            menu.set_status("Checking for updates...", "busy")
            return
        if not updates["available"]:
            check_updates(force=True)
            menu.set_status(f"{updates['message']} -- checking again...", "busy")
            return

        menu.set_status("Updating...", "busy")
        ok, message = apply_update()
        menu.set_status(message, "ok" if ok else "bad")
        if ok:
            updates.update(available=False, message="up to date", checked=True)

    def do_save():
        if not config_path:
            menu.set_status("No config file to write to", "bad")
            return
        values = {
            "device": cfg.get("camera.device"),
            "fullscreen": str(bool(cfg.get("display.fullscreen"))).lower(),
            "enabled": str(bool(cfg.get("identity.enabled"))).lower(),
            "yaw_offset_deg": cfg.get("gaze.yaw_offset_deg"),
            "pitch_offset_deg": cfg.get("gaze.pitch_offset_deg"),
            "yaw_tolerance_deg": cfg.get("gaze.yaw_tolerance_deg"),
            "pitch_tolerance_deg": cfg.get("gaze.pitch_tolerance_deg"),
        }
        try:
            missing = patch_yaml_values(config_path, values)
        except OSError as exc:
            menu.set_status(f"Could not write the config: {exc}", "bad")
            return
        if missing:
            menu.set_status(f"Saved, except: {', '.join(missing)}", "bad")
        else:
            menu.set_status(f"Saved to {os.path.basename(config_path)}", "ok")

    def people_summary() -> str:
        identifier = worker.identifier
        if not worker.identity_enabled:
            return "off"
        if identifier is None:
            return "starting..."
        return ", ".join(sorted(identifier.db.people)) or "nobody yet"

    menu.items = [
        MenuItem("Calibrate gaze", do_calibrate,
                 lambda: f"{cfg.get('gaze.yaw_offset_deg', 0):+.0f} / "
                         f"{cfg.get('gaze.pitch_offset_deg', 0):+.0f}",
                 "Look at this screen for 8 seconds. Learns where the screen is."),
        MenuItem("Learn a new person", do_learn, people_summary,
                 "Look at the camera for 10 seconds, moving your head a little."),
        MenuItem("Remove a person", do_remove_person,
                 lambda: str(len(worker.identifier.db.people))
                         if worker.identifier is not None else "-",
                 "Deletes their face data from people.json for good.",
                 enabled=lambda: bool(worker.identifier is not None
                                      and worker.identifier.db.people)),
        MenuItem("Face recognition", do_toggle_identity,
                 lambda: "on" if worker.identity_enabled else "off",
                 "Show a different screen per recognised person."),
        MenuItem("Camera", do_next_camera,
                 lambda: ("scanning..." if cameras["scanning"]
                          else f"device {camera.device}"),
                 "Switch to the next camera that works."),
        MenuItem("Fullscreen", do_toggle_fullscreen,
                 lambda: "on" if display.fullscreen else "off"),
        MenuItem("Save settings", do_save, lambda: "",
                 "Write these settings into config.yaml."),
        MenuItem("Update", do_update,
                 lambda: ("checking..." if updates["checking"]
                          else ("update available" if updates["available"]
                                else updates["message"])),
                 "Pull the newest version from git and reinstall.",
                 enabled=lambda: checkout_dir() is not None),
        MenuItem("Quit", stop.set, lambda: "", "Close lookat."),
    ]
    def on_open():
        scan_cameras()
        check_updates()

    menu.on_open = on_open
    if cfg.get("updates.check_on_start", True):
        check_updates()
    return menu


def apply_task_result(result: dict, cfg, display, menu, config_path) -> None:
    """Runs on the main thread: persist what a finished task produced."""
    from .config import patch_yaml_values

    if not result.get("ok"):
        return

    if result["kind"] == "calibrate" and config_path:
        try:
            patch_yaml_values(config_path, result["values"])
        except OSError as exc:
            log.error("could not save the calibration: %s", exc)

    if result["kind"] == "forget":
        name = result["name"]
        people = cfg.get("display.scenes.person") or {}
        if name in people:
            people.pop(name)
            cfg.set("display.scenes.person", people)
            if hasattr(display, "reload_scenes"):
                display.reload_scenes()
            menu.set_status(
                f"Removed {name}. Their face data is gone; delete their entry "
                "under display.scenes.person in config.yaml to tidy up.", "ok")
        return

    if result["kind"] == "enroll":
        name = result["name"]
        people = cfg.get("display.scenes.person") or {}
        if name not in people:
            # Give the new person a screen of their own straight away,
            # otherwise nothing visible changes and it looks broken.
            colour = PERSON_COLOURS[len(people) % len(PERSON_COLOURS)]
            people[name] = {"background": colour, "text": f"Hi {name}"}
            cfg.set("display.scenes.person", people)
            if hasattr(display, "reload_scenes"):
                display.reload_scenes()
            menu.set_status(
                f"Learned {name}, and gave them a screen. "
                "Edit display.scenes.person in config.yaml to change it.", "ok")


def pick_scene(attentive: bool, person: str | None, known_scenes) -> str:
    """Which scene to show. A recognised person gets their own scene if one is
    configured, otherwise the generic `attentive` scene."""
    if not attentive:
        return "idle"
    if person:
        candidate = f"person.{person}"
        if candidate in known_scenes:
            return candidate
    return "attentive"


def run(cfg, stop: threading.Event | None = None, config_path: str | None = None) -> int:
    """Run until the stop event is set, the window closes, or a signal arrives.

    `stop` lets an embedding application shut the loop down; when omitted a
    fresh event is used and SIGINT/SIGTERM set it.
    """
    stop = stop or threading.Event()

    def _signal(_sig, _frame):
        log.info("shutting down")
        stop.set()

    try:
        signal.signal(signal.SIGINT, _signal)
        signal.signal(signal.SIGTERM, _signal)
    except ValueError:
        # Not on the main thread (embedded / under test): the caller owns
        # shutdown via the stop event.
        log.debug("signal handlers not installed (not the main thread)")

    shared = Shared()
    hooks = HookRunner(cfg)
    camera = Camera(cfg)
    camera.start()

    display = build_display(cfg)
    headless = display.__class__.__name__ == "NullDisplay"
    identity_enabled = bool(cfg.get("identity.enabled", False))
    # The display can toggle debug at runtime; the worker needs to know.
    worker = DetectionWorker(cfg, camera, shared, hooks, stop)
    worker.start()

    menu = None
    if not headless:
        menu = build_menu(cfg, display, camera, worker, config_path, stop)
        display.menu = menu

    log.info("running -- press m for the menu, q or Esc to quit")
    menu_was_open = False
    try:
        while not stop.is_set():
            with shared.lock:
                state = shared.state
                preview = shared.preview
                observation = shared.observation
                stats = {
                    "det fps": f"{shared.detector_fps:.1f}",
                    "backend": shared.backend,
                    "yaw": f"{observation.dev_yaw:+.1f}",
                    "pitch": f"{observation.dev_pitch:+.1f}",
                }
                if worker.identity_enabled:
                    stats["person"] = (
                        f"{observation.identity} ({observation.identity_score:.2f})"
                        if observation.identity else "unknown"
                    )
                error = shared.error
                task_status = shared.task_status
                task_status_kind = shared.task_status_kind
                task_result = shared.task_result
                shared.task_result = None

            if menu is not None:
                if task_status:
                    menu.set_status(task_status, task_status_kind)
                if task_result is not None:
                    apply_task_result(task_result, cfg, display, menu, config_path)
                if menu.open and getattr(menu, "on_open", None) and not menu_was_open:
                    menu.on_open()
                menu_was_open = menu.open

            known_scenes = set(getattr(display, "scene_names", lambda: [])())
            scene = pick_scene(state.attentive, observation.identity, known_scenes)
            if error:
                log.error("detector error: %s", error)
                break

            worker.want_preview = getattr(display, "debug", False)
            if not display.update(scene, state, preview, stats):
                break
            if headless:
                time.sleep(0.05)   # NullDisplay does not block on vsync
    finally:
        stop.set()
        worker.join(timeout=3.0)
        camera.stop()
        display.close()
        hooks.shutdown()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lookat",
        description="Webcam attention detection: the screen reacts when someone looks at it.",
    )
    parser.add_argument("-c", "--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="override any config value, e.g. --set gaze.yaw_tolerance_deg=30",
    )
    parser.add_argument("--camera", type=int, help="shortcut for --set camera.device=N")
    parser.add_argument("--windowed", action="store_true", help="do not go fullscreen")
    parser.add_argument("--debug", action="store_true", help="start with the debug overlay on")
    parser.add_argument("--headless", action="store_true", help="no window, hooks only")
    parser.add_argument("--calibrate", action="store_true", help="open the tuning tool instead")
    parser.add_argument(
        "--auto-calibrate", nargs="?", type=float, const=8.0, metavar="SECONDS",
        help="look at the screen for N seconds (default 8) and derive the offsets; "
             "no window needed",
    )
    parser.add_argument(
        "--write", action="store_true",
        help="with --auto-calibrate, save the result into the config file",
    )
    parser.add_argument(
        "--list-cameras", action="store_true", help="probe camera indices and exit"
    )

    people = parser.add_argument_group("face recognition")
    people.add_argument("--enroll", metavar="NAME", help="record a person's face and exit")
    people.add_argument(
        "--from", dest="from_images", nargs="+", metavar="IMG",
        help="with --enroll: use these photos instead of the webcam",
    )
    people.add_argument(
        "--add", action="store_true",
        help="with --enroll: add to the existing templates instead of replacing them",
    )
    people.add_argument(
        "--enroll-seconds", type=float, default=10.0, metavar="S",
        help="with --enroll: how long to record from the webcam (default 10)",
    )
    people.add_argument("--people", action="store_true", help="list enrolled people and exit")
    people.add_argument("--forget", metavar="NAME", help="remove an enrolled person and exit")
    parser.add_argument("-v", "--verbose", action="store_true")

    maintenance = parser.add_argument_group("installation")
    maintenance.add_argument("--version", action="store_true",
                             help="print the version and where everything lives")
    maintenance.add_argument("--where", action="store_true",
                             help="same as --version (config and data paths)")
    maintenance.add_argument("--update", action="store_true",
                             help="pull the latest version from git and reinstall")
    maintenance.add_argument("--check-update", action="store_true",
                             help="say whether an update is available, then exit")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from .paths import describe_paths, resolve_config

    if args.update or args.check_update:
        from .update import apply_update, check_for_update

        if args.check_update:
            available, message = check_for_update()
            print(("update available: " if available else "") + message)
            return 0
        ok, message = apply_update()
        print(message)
        return 0 if ok else 1

    # `-c/--config` counts as explicit only when it differs from the default.
    explicit = args.config if args.config != "config.yaml" else None
    resolved, created = resolve_config(explicit)
    if created:
        print(f"created a fresh config at {resolved}")
    config_path = str(resolved) if resolved.exists() else None
    if config_path is None:
        log.warning("config %s not found, using built-in defaults", resolved)

    if args.version or args.where:
        print(describe_paths(resolved))
        return 0

    if args.list_cameras:
        from .camera import probe_cameras

        found = probe_cameras()
        if not found:
            print("no working camera found")
            return 1
        for index, width, height in found:
            print(f"camera.device: {index}   ({width}x{height})")
        return 0

    cfg = load_config(config_path, args.overrides)
    if args.camera is not None:
        cfg.set("camera.device", args.camera)
    if args.windowed:
        cfg.set("display.fullscreen", False)
    if args.debug:
        cfg.set("display.debug_overlay", True)
    if args.headless:
        cfg.set("display.backend", "none")

    if args.enroll:
        from .enroll import enroll_from_camera, enroll_from_images

        cfg.set("identity.enabled", True)
        if args.from_images:
            return enroll_from_images(cfg, args.enroll, args.from_images, add=args.add)
        return enroll_from_camera(cfg, args.enroll, args.enroll_seconds, add=args.add)

    if args.people:
        from .enroll import list_people

        return list_people(cfg)

    if args.forget:
        from .enroll import forget_person

        return forget_person(cfg, args.forget)

    if args.auto_calibrate:
        from .calibrate import run_auto_calibration

        return run_auto_calibration(cfg, args.auto_calibrate, config_path, args.write)

    if args.calibrate:
        from .calibrate import run_calibration

        return run_calibration(cfg)

    try:
        return run(cfg, config_path=config_path)
    except Exception as exc:
        log.error("%s", exc)
        if args.verbose:
            raise
        return 1
