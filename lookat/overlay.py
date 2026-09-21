"""Debug drawing on the camera frame (used by the overlay and by --calibrate)."""

from __future__ import annotations

import math

import numpy as np

GREEN = (80, 230, 140)
RED = (70, 90, 240)
WHITE = (240, 240, 240)
GREY = (150, 150, 150)


def draw_observation(frame: np.ndarray, obs, attentive: bool | None = None) -> np.ndarray:
    import cv2

    canvas = frame.copy()
    height, width = canvas.shape[:2]
    colour = GREEN if (attentive if attentive is not None else obs.score > 0.5) else RED

    if obs.bbox:
        x, y, w, h = obs.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, 2)

    if obs.landmarks is not None:
        from .detector import LEFT_EYE, RIGHT_EYE, POSE_LANDMARKS

        for idx in POSE_LANDMARKS:
            px, py = obs.landmarks[idx]
            cv2.circle(canvas, (int(px), int(py)), 2, WHITE, -1)
        for eye in (LEFT_EYE, RIGHT_EYE):
            if eye["iris"] < len(obs.landmarks):
                px, py = obs.landmarks[eye["iris"]]
                cv2.circle(canvas, (int(px), int(py)), 3, (60, 200, 255), -1)
            for key in ("outer", "inner", "top", "bottom"):
                px, py = obs.landmarks[eye[key]]
                cv2.circle(canvas, (int(px), int(py)), 1, GREY, -1)

    # Gaze ray from the face centre.
    if obs.bbox and obs.face_found:
        x, y, w, h = obs.bbox
        cx, cy = int(x + w / 2), int(y + h / 2)
        length = w * 1.2
        dx = math.tan(math.radians(obs.gaze_yaw)) * length
        dy = -math.tan(math.radians(obs.gaze_pitch)) * length
        cv2.arrowedLine(canvas, (cx, cy), (int(cx + dx), int(cy + dy)), colour, 2, tipLength=0.2)

    lines = [
        f"backend {obs.backend}  face {'yes' if obs.face_found else 'no'}",
        f"head  yaw {obs.head_yaw:+6.1f}  pitch {obs.head_pitch:+6.1f}  roll {obs.head_roll:+6.1f}",
        f"gaze  yaw {obs.gaze_yaw:+6.1f}  pitch {obs.gaze_pitch:+6.1f}",
        f"dev   yaw {obs.dev_yaw:+6.1f}  pitch {obs.dev_pitch:+6.1f}",
        f"iris  h {obs.eye_h:.2f}  v {obs.eye_v:.2f}  open {obs.openness:.2f}",
        f"score {obs.score:.2f}   face/frame {obs.face_ratio:.2f}",
    ]
    if obs.note:
        lines.append(obs.note)

    y = 18
    for line in lines:
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
        y += 18

    if attentive is not None:
        label = "LOOKING" if attentive else "away"
        cv2.putText(canvas, label, (8, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(canvas, label, (8, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

    return canvas
