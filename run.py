#!/usr/bin/env python3
"""Entry point: python run.py [--windowed] [--debug] [--calibrate]"""

import sys

from lookat.app import main

if __name__ == "__main__":
    sys.exit(main())
