"""lookat -- webcam attention detection with a screen that reacts."""

import os as _os

# OpenCV's videoio backends log a wall of warnings while probing camera
# indices that do not exist. Silence them before cv2 is imported anywhere;
# `run.py --verbose` turns them back on.
_os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

__version__ = "1.1.0"
