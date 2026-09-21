"""Where things live.

Keeping the user's config, enrolled faces and models out of the checkout is
what makes `git pull` safe: an update can never conflict with local edits or
overwrite somebody's face database.

Resolution order for the config file:
  1. --config PATH
  2. $LOOKAT_CONFIG
  3. <checkout>/config.yaml, if you are running from a git clone and that file
     exists -- so an existing working setup keeps working untouched
  4. <user data dir>/config.yaml, created from the packaged default

Models and people.json are resolved relative to the config file, so they end
up beside it in whichever of those locations is in use.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "config.default.yaml"


def user_data_dir() -> Path:
    """Per-user directory for config, models and enrolled faces."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "lookat"
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
        return Path(base) / "lookat"
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / "lookat"


def checkout_dir() -> Path | None:
    """The git clone this package lives in, if it is running from one."""
    candidate = PACKAGE_DIR.parent
    if (candidate / ".git").exists():
        return candidate
    return None


def resolve_config(explicit: str | None = None) -> tuple[Path, bool]:
    """Return (path, created_now). Creates the user config on first run."""
    if explicit:
        return Path(explicit).expanduser(), False

    from_env = os.environ.get("LOOKAT_CONFIG")
    if from_env:
        return Path(from_env).expanduser(), False

    checkout = checkout_dir()
    if checkout and (checkout / "config.yaml").exists():
        return checkout / "config.yaml", False

    target = user_data_dir() / "config.yaml"
    if target.exists():
        return target, False

    target.parent.mkdir(parents=True, exist_ok=True)
    if DEFAULT_CONFIG.exists():
        shutil.copyfile(DEFAULT_CONFIG, target)
        log.info("created %s from the shipped defaults", target)
        return target, True

    log.warning("no packaged default config found at %s", DEFAULT_CONFIG)
    return target, False


def git_describe(repo: Path | None = None) -> str:
    """Short commit of the checkout, or '' when not running from git."""
    repo = repo or checkout_dir()
    if not repo:
        return ""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "describe", "--always", "--dirty", "--tags"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def describe_paths(config_path: Path) -> str:
    from . import __version__

    lines = [
        f"version      {__version__}" + (f"  ({git_describe()})" if git_describe() else ""),
        f"python       {sys.version.split()[0]}  ({sys.executable})",
        f"config       {config_path}",
        f"data dir     {config_path.parent}",
        f"checkout     {checkout_dir() or '(installed, not a git clone)'}",
    ]
    return "\n".join(lines)
