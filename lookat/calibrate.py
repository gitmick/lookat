"""Interactive tuning tool.

Shows the live camera with the pose/gaze overlay. Collect a handful of samples
while looking at the screen and while looking away, and it prints a ready-made
YAML block with offsets and tolerances that separate the two.

Needs a desktop session (it opens an OpenCV window).
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Tuple

import numpy as np

from .attention import AttentionTracker
from .config import patch_yaml_values
from .camera import Camera
from .detector import build_detector
from .overlay import draw_observation

log = logging.getLogger("lookat.calibrate")

HELP = [
    "SPACE / l : record a sample while LOOKING at the screen",
    "a         : record a sample while looking AWAY",
    "s         : suggest settings from the samples so far",
    "r         : reset samples",
    "q / Esc   : quit",
]


def _suggest(looking: List[Tuple[float, float]], away: List[Tuple[float, float]]) -> str:
    if len(looking) < 5:
        return "# need at least 5 'looking' samples (press SPACE while looking at the screen)"

    look = np.array(looking)
    yaw_offset = float(np.median(look[:, 0]))
    pitch_offset = float(np.median(look[:, 1]))

    yaw_spread = float(np.percentile(np.abs(look[:, 0] - yaw_offset), 90))
    pitch_spread = float(np.percentile(np.abs(look[:, 1] - pitch_offset), 90))
    yaw_tol = max(8.0, round(yaw_spread + 6.0, 1))
    pitch_tol = max(8.0, round(pitch_spread + 6.0, 1))

    warning = ""
    if len(away) >= 5:
        off = np.array(away)
        margin_yaw = float(np.percentile(np.abs(off[:, 0] - yaw_offset), 10))
        margin_pitch = float(np.percentile(np.abs(off[:, 1] - pitch_offset), 10))
        # Keep the tolerance clear of the "away" cluster.
        yaw_tol = min(yaw_tol, max(8.0, round(margin_yaw * 0.6, 1)))
        pitch_tol = min(pitch_tol, max(8.0, round(margin_pitch * 0.6, 1)))
        if margin_yaw < yaw_spread or margin_pitch < pitch_spread:
            warning = (
                "# WARNING: the 'looking' and 'away' samples overlap.\n"
                "# Move further from the camera edge, improve the lighting, or\n"
                "# accept that these two cases are hard to tell apart here.\n"
            )

    return (
        f"{warning}# paste into config.yaml, replacing the matching keys\n"
        "gaze:\n"
        f"  yaw_offset_deg: {yaw_offset:.1f}\n"
        f"  pitch_offset_deg: {pitch_offset:.1f}\n"
        f"  yaw_tolerance_deg: {yaw_tol:.1f}\n"
        f"  pitch_tolerance_deg: {pitch_tol:.1f}\n"
        f"# samples: {len(looking)} looking, {len(away)} away"
    )


def run_calibration(cfg) -> int:
    import cv2

    camera = Camera(cfg).start()
    detector = build_detector(cfg)
    tracker = AttentionTracker(cfg)

    looking: List[Tuple[float, float]] = []
    away: List[Tuple[float, float]] = []
    last_seq, t0 = -1, time.monotonic()
    obs = None
    state = tracker.state

    print("\n".join(HELP))
    try:
        while True:
            seq, frame = camera.read()
            if frame is not None and seq != last_seq:
                last_seq = seq
                obs = detector.process(frame, int((time.monotonic() - t0) * 1000))
                state = tracker.update(obs)

            if frame is None:
                time.sleep(0.02)
                continue

            canvas = draw_observation(frame, obs, state.attentive) if obs else frame.copy()
            height = canvas.shape[0]
            info = f"samples: looking {len(looking)}  away {len(away)}   [space/l, a, s, r, q]"
            cv2.putText(canvas, info, (8, height - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(canvas, info, (8, height - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow("lookat - calibration", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord(" "), ord("l")) and obs and obs.face_found:
                looking.append((obs.dev_yaw + cfg.get("gaze.yaw_offset_deg", 0.0),
                                obs.dev_pitch + cfg.get("gaze.pitch_offset_deg", 0.0)))
                print(f"looking sample {len(looking)}: yaw {looking[-1][0]:+.1f} pitch {looking[-1][1]:+.1f}")
            if key == ord("a") and obs and obs.face_found:
                away.append((obs.dev_yaw + cfg.get("gaze.yaw_offset_deg", 0.0),
                             obs.dev_pitch + cfg.get("gaze.pitch_offset_deg", 0.0)))
                print(f"away sample {len(away)}: yaw {away[-1][0]:+.1f} pitch {away[-1][1]:+.1f}")
            if key == ord("r"):
                looking.clear()
                away.clear()
                print("samples cleared")
            if key == ord("s"):
                print("\n" + _suggest(looking, away) + "\n")
    finally:
        cv2.destroyAllWindows()
        camera.stop()
        detector.close()

    print("\n" + _suggest(looking, away))
    return 0

# ---------------------------------------------------------------------------
# Non-interactive calibration
# ---------------------------------------------------------------------------
def run_auto_calibration(cfg, seconds: float = 8.0, config_path: str | None = None,
                         write: bool = False) -> int:
    """Sample while the user looks at the screen, then derive the offsets.

    No window and no keypresses -- useful over SSH, or when the display is the
    thing being calibrated.
    """
    import numpy as np

    camera = Camera(cfg).start()
    detector = build_detector(cfg)

    # Whatever offsets are configured now must be added back, so the result is
    # absolute rather than relative to the current settings.
    yaw_base = float(cfg.get("gaze.yaw_offset_deg", 0.0))
    pitch_base = float(cfg.get("gaze.pitch_offset_deg", 0.0))

    print(f"\nLook straight at the screen for {seconds:.0f} seconds.")
    print("Move your head around a little, the way you normally would.\n")
    for count in (3, 2, 1):
        print(f"  starting in {count}...", flush=True)
        time.sleep(1.0)
    print("  sampling...\n", flush=True)

    samples: List[Tuple[float, float]] = []
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
            if obs.face_found:
                samples.append((obs.dev_yaw + yaw_base, obs.dev_pitch + pitch_base))
    finally:
        camera.stop()
        detector.close()

    if len(samples) < 10:
        print(f"only {len(samples)} usable frames out of {frames} -- no face found.")
        print("Check the lighting and that you are inside the camera's view.")
        return 1

    data = np.array(samples)
    yaw_offset = float(np.median(data[:, 0]))
    pitch_offset = float(np.median(data[:, 1]))
    yaw_spread = float(np.percentile(np.abs(data[:, 0] - yaw_offset), 90))
    pitch_spread = float(np.percentile(np.abs(data[:, 1] - pitch_offset), 90))
    yaw_tol = round(max(12.0, yaw_spread + 8.0), 1)
    pitch_tol = round(max(10.0, pitch_spread + 8.0), 1)

    print(f"{len(samples)} samples from {frames} frames")
    print(f"  gaze while looking at the screen: yaw {yaw_offset:+.1f}  pitch {pitch_offset:+.1f}")
    print(f"  90th-percentile wobble:           yaw {yaw_spread:5.1f}  pitch {pitch_spread:5.1f}")
    if max(abs(yaw_offset), abs(pitch_offset)) > 25:
        print("\n  NOTE: that is a large offset. The camera is probably aimed well")
        print("  away from where you sit. Aiming it better will work much more")
        print("  reliably than correcting it in software.")

    values = {
        "yaw_offset_deg": round(yaw_offset, 1),
        "pitch_offset_deg": round(pitch_offset, 1),
        "yaw_tolerance_deg": yaw_tol,
        "pitch_tolerance_deg": pitch_tol,
    }

    if write and config_path and os.path.exists(config_path):
        missing = patch_yaml_values(config_path, values)
        print(f"\nwrote to {config_path}:")
        for key, value in values.items():
            flag = "  (NOT FOUND, set it by hand)" if key in missing else ""
            print(f"  {key}: {value}{flag}")
    else:
        print("\n# paste into config.yaml under `gaze:`")
        print("gaze:")
        for key, value in values.items():
            print(f"  {key}: {value}")
    return 0
