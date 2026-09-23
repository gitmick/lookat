#!/usr/bin/env bash
# lookat -- macOS install.
#
#   git clone git@github.com:gitmick/lookat.git ~/lookat
#   cd ~/lookat && ./scripts/install_mac.sh
#
# Re-run it any time; it is idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

say()  { printf '\033[36m==\033[0m %s\n' "$*"; }
warn() { printf '\033[33m!!\033[0m %s\n' "$*"; }
die()  { printf '\033[31mxx\033[0m %s\n' "$*" >&2; exit 1; }

ARCH="$(uname -m)"
say "macOS $(sw_vers -productVersion 2>/dev/null || echo '?') on ${ARCH}"

# MediaPipe publishes a version-independent wheel for Apple Silicon, but only
# cp39-cp312 wheels for Intel, and none at all after 0.10.21. So an Intel Mac
# needs Python 3.10-3.12; anything newer has no wheel to install.
MAX_MINOR=99
if [ "$ARCH" = "x86_64" ]; then
    MAX_MINOR=12
    warn "Intel Mac: MediaPipe stopped shipping Intel wheels after 0.10.21,"
    warn "so this installs that release (with numpy 1.x and OpenCV 4.x)."
    warn "It needs Python 3.10-3.12 -- 3.13 and newer have no Intel wheel."
fi

# --- Python ---------------------------------------------------------------
PY=""
REJECTED=""
for candidate in python3.12 python3.11 python3.10 python3.13 python3; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    version="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo 0.0)"
    major="${version%%.*}"; minor="${version##*.}"
    [ "$major" = "3" ] || continue
    if [ "$minor" -lt 10 ] 2>/dev/null; then continue; fi
    if [ "$minor" -gt "$MAX_MINOR" ] 2>/dev/null; then
        REJECTED="$REJECTED $candidate($version)"
        continue
    fi
    PY="$candidate"
    break
done

if [ -z "$PY" ]; then
    [ -n "$REJECTED" ] && warn "Too new for an Intel Mac:$REJECTED"
    warn "No usable Python found (need 3.10-$([ "$MAX_MINOR" = 99 ] && echo "newer" || echo "3.$MAX_MINOR"))."
    if command -v brew >/dev/null 2>&1; then
        say "Installing Python 3.12 via Homebrew..."
        brew install python@3.12
        PY="$(brew --prefix)/opt/python@3.12/bin/python3.12"
        [ -x "$PY" ] || PY=python3.12
    else
        die "Install Homebrew (https://brew.sh) then re-run, or get Python 3.12 from python.org"
    fi
fi
say "Using $PY ($("$PY" --version))"

# --- virtualenv -----------------------------------------------------------
if [ ! -d .venv ]; then
    say "Creating .venv"
    "$PY" -m venv .venv
fi
say "Installing dependencies (a few hundred MB, first time takes a while)"
./.venv/bin/python -m pip install --quiet --upgrade pip wheel
./.venv/bin/python -m pip install --quiet -e .

# --- launcher on PATH -----------------------------------------------------
BIN="$HOME/.local/bin"
mkdir -p "$BIN"
cat > "$BIN/lookat" <<LAUNCHER
#!/usr/bin/env bash
exec "$REPO/.venv/bin/lookat" "\$@"
LAUNCHER
chmod +x "$BIN/lookat"
say "Installed launcher at $BIN/lookat"

case ":$PATH:" in
    *":$BIN:"*) ;;
    *)  warn "$BIN is not on your PATH. Add this to ~/.zshrc:"
        printf '\n    export PATH="$HOME/.local/bin:$PATH"\n\n' ;;
esac

# --- models ---------------------------------------------------------------
say "Fetching the face landmark model"
./.venv/bin/python - <<'PYEOF'
from lookat.config import load_config
from lookat.detector import ensure_model
from lookat.paths import resolve_config
path, created = resolve_config(None)
cfg = load_config(str(path) if path.exists() else None)
ensure_model(cfg.path(cfg.get("detector.model_path")), True)
print("config:", path)
PYEOF

cat <<'DONE'

Done.

  lookat --version        where everything lives
  lookat --windowed --debug   check it sees you
  lookat                  run it

The first time it opens the camera, macOS will ask for permission. If you are
never asked, or you see a black picture, enable your terminal under
System Settings > Privacy & Security > Camera.

Press m in the window for the menu (calibrate, learn a person, camera, ...).
Update later with:  lookat --update
DONE
