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

if [ "$ARCH" = "x86_64" ]; then
    warn "Intel Mac: MediaPipe stopped shipping Intel wheels after 0.10.21,"
    warn "so this pins that older release. It works, but it is not the"
    warn "version used on Apple Silicon. macOS 14+ is required for OpenCV."
fi

# --- Python ---------------------------------------------------------------
PY=""
for candidate in python3.12 python3.11 python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        version="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo 0.0)"
        major="${version%%.*}"; minor="${version##*.}"
        if [ "$major" = "3" ] && [ "$minor" -ge 10 ] 2>/dev/null; then PY="$candidate"; break; fi
    fi
done

if [ -z "$PY" ]; then
    warn "No Python 3.10+ found."
    if command -v brew >/dev/null 2>&1; then
        say "Installing Python via Homebrew..."
        brew install python@3.12
        PY=python3.12
    else
        die "Install Homebrew (https://brew.sh) then re-run, or get Python from python.org"
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
