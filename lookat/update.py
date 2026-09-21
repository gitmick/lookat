"""Self-update from the git checkout this package was installed from."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from .paths import checkout_dir, git_describe

log = logging.getLogger(__name__)


AUTH_MARKERS = ("could not read Username", "Authentication failed", "Permission denied",
                "publickey", "terminal prompts disabled", "403")

AUTH_HELP = (
    "\nThe repository is private, so this machine needs access to it. Either:"
    "\n  gh auth login                      (then: gh auth setup-git)"
    "\nor switch the remote to SSH and add a key to your GitHub account:"
    "\n  git remote set-url origin git@github.com:gitmick/lookat.git"
)


def _git(repo: Path, *args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    return subprocess.run(["git", "-C", str(repo), *args], env=env,
                          capture_output=True, text=True, timeout=timeout, check=False)


def _last_line(proc: subprocess.CompletedProcess) -> str:
    text = (proc.stderr or proc.stdout or "").strip().splitlines()
    return text[-1].strip() if text else "unknown error"


def _with_auth_hint(message: str) -> str:
    return message + AUTH_HELP if any(m in message for m in AUTH_MARKERS) else message


def check_for_update(repo: Path | None = None) -> tuple[bool, str]:
    """(update_available, message). Contacts the remote, so it can be slow."""
    repo = repo or checkout_dir()
    if not repo:
        return False, "not a git checkout"

    fetch = _git(repo, "fetch", "--quiet")
    if fetch.returncode != 0:
        return False, _with_auth_hint(f"could not reach the remote: {_last_line(fetch)}")

    counts = _git(repo, "rev-list", "--left-right", "--count", "HEAD...@{u}")
    if counts.returncode != 0:
        return False, "no upstream branch configured"
    try:
        ahead, behind = (int(n) for n in counts.stdout.split())
    except ValueError:
        return False, "could not compare with the remote"

    if behind == 0:
        return False, "up to date"
    plural = "commit" if behind == 1 else "commits"
    extra = f", {ahead} local {'commit' if ahead == 1 else 'commits'} not pushed" if ahead else ""
    return True, f"{behind} new {plural}{extra}"


def apply_update(repo: Path | None = None, reinstall: bool = True) -> tuple[bool, str]:
    """Fast-forward the checkout and reinstall dependencies."""
    repo = repo or checkout_dir()
    if not repo:
        return False, "not running from a git checkout; reinstall by hand"

    dirty = _git(repo, "status", "--porcelain")
    if dirty.stdout.strip():
        files = ", ".join(line[3:] for line in dirty.stdout.strip().splitlines()[:3])
        return False, f"local changes would be overwritten ({files}) -- commit or stash them"

    before = git_describe(repo)
    pull = _git(repo, "pull", "--ff-only")
    if pull.returncode != 0:
        return False, _with_auth_hint(f"git pull failed: {_last_line(pull)}")

    after = git_describe(repo)
    if before == after:
        return True, "already up to date"

    if reinstall:
        install = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "-e", str(repo)],
            capture_output=True, text=True, timeout=900, check=False,
        )
        if install.returncode != 0:
            return False, (f"updated to {after}, but installing dependencies failed: "
                           f"{_last_line(install)}")

    return True, f"updated {before} -> {after}; restart lookat to use it"
