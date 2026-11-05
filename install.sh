#!/usr/bin/env sh
# Animora installer.
#
#   curl -fsSL https://raw.githubusercontent.com/ltorl/animora/main/install.sh | sh
#
# Environment variables (all optional):
#   ANIMORA_HOME     install location               (default: ~/.animora)
#   ANIMORA_BIN_DIR  where the `animora` link goes  (default: ~/.local/bin)
#   ANIMORA_REF      branch or tag to install        (default: main)
#   ANIMORA_PYTHON   Python interpreter to use       (default: python3)
#   ANIMORA_NO_BUILD set to 1 to skip compiling the native renderer now
#
# Uninstall:
#   curl -fsSL https://raw.githubusercontent.com/ltorl/animora/main/install.sh | sh -s -- --uninstall

set -eu

REPO="ltorl/animora"
REF="${ANIMORA_REF:-main}"
HOME_DIR="${ANIMORA_HOME:-$HOME/.animora}"
BIN_DIR="${ANIMORA_BIN_DIR:-$HOME/.local/bin}"
PYTHON="${ANIMORA_PYTHON:-python3}"

if [ -t 1 ]; then B="$(printf '\033[1m')"; G="$(printf '\033[32m')"; R="$(printf '\033[31m')"; Y="$(printf '\033[33m')"; N="$(printf '\033[0m')"
else B=""; G=""; R=""; Y=""; N=""; fi
say()  { printf '%s==>%s %s\n' "$B$G" "$N" "$*"; }
warn() { printf '%swarning:%s %s\n' "$Y" "$N" "$*" >&2; }
die()  { printf '%serror:%s %s\n' "$R" "$N" "$*" >&2; exit 1; }

if [ "${1:-}" = "--uninstall" ]; then
    say "Removing $HOME_DIR and $BIN_DIR/animora"
    rm -rf "$HOME_DIR"
    if [ -L "$BIN_DIR/animora" ]; then rm -f "$BIN_DIR/animora"; fi
    rm -rf "${XDG_CACHE_HOME:-$HOME/.cache}/animora"
    say "Animora uninstalled."
    exit 0
fi

# --- requirements -----------------------------------------------------------
command -v "$PYTHON" >/dev/null 2>&1 || die "$PYTHON not found. Install Python 3.9 or newer (https://www.python.org/downloads/)."
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' \
    || die "Python 3.9 or newer is required (found $("$PYTHON" -c 'import sys; print(sys.version.split()[0])'))."
"$PYTHON" -c 'import venv, ensurepip' >/dev/null 2>&1 \
    || die "The Python 'venv' module is missing. On Debian/Ubuntu: sudo apt install python3-venv"

CXX_FOUND=""
for c in "${ANIMORA_CXX:-}" "${CXX:-}" clang++ g++ c++; do
    if [ -n "$c" ] && command -v "$c" >/dev/null 2>&1; then CXX_FOUND="$c"; break; fi
done
if [ -z "$CXX_FOUND" ]; then
    case "$(uname -s)" in
        Darwin) warn "No C++ compiler found. Install the Xcode command-line tools: xcode-select --install" ;;
        *)      warn "No C++ compiler found. Install one, e.g. sudo apt install g++ (or dnf install gcc-c++)" ;;
    esac
    warn "Animora installs anyway, but rendering needs a compiler (run 'animora build' afterwards)."
fi
command -v ffmpeg >/dev/null 2>&1 || warn "ffmpeg not found: image sequences work, video encoding needs ffmpeg."

# --- download ---------------------------------------------------------------
TMP="$(mktemp -d 2>/dev/null || mktemp -d -t animora)"
trap 'rm -rf "$TMP"' EXIT INT TERM

URL="https://codeload.github.com/$REPO/tar.gz/$REF"
say "Downloading Animora ($REF)"
if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$URL" -o "$TMP/src.tar.gz" || die "download failed: $URL"
elif command -v wget >/dev/null 2>&1; then
    wget -qO "$TMP/src.tar.gz" "$URL" || die "download failed: $URL"
else
    die "curl or wget is required."
fi
mkdir -p "$TMP/src"
tar -xzf "$TMP/src.tar.gz" -C "$TMP/src" --strip-components=1 || die "could not unpack the archive."

# --- install ----------------------------------------------------------------
say "Installing into $HOME_DIR"
mkdir -p "$HOME_DIR"
rm -rf "$HOME_DIR/src"
mv "$TMP/src" "$HOME_DIR/src"

if [ ! -x "$HOME_DIR/venv/bin/python" ]; then
    "$PYTHON" -m venv "$HOME_DIR/venv" || die "could not create a virtual environment."
fi
VPY="$HOME_DIR/venv/bin/python"
"$VPY" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true
"$VPY" -m pip install --quiet --upgrade "$HOME_DIR/src" || die "pip install failed."
"$VPY" -m pip install --quiet pyyaml >/dev/null 2>&1 || warn "pyyaml not installed (only needed for .yaml scenes)."

mkdir -p "$BIN_DIR"
ln -sf "$HOME_DIR/venv/bin/animora" "$BIN_DIR/animora"

# --- compile the native renderer ---------------------------------------------
if [ -n "$CXX_FOUND" ] && [ "${ANIMORA_NO_BUILD:-0}" != "1" ]; then
    say "Compiling the native renderer (about 10 seconds)"
    "$HOME_DIR/venv/bin/animora" build --force || warn "native build failed; run 'animora doctor' for details."
fi

# --- done -------------------------------------------------------------------
say "Animora installed: $BIN_DIR/animora"
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        SHELL_RC="~/.profile"
        case "${SHELL:-}" in */zsh) SHELL_RC="~/.zshrc" ;; */bash) SHELL_RC="~/.bashrc" ;; esac
        warn "$BIN_DIR is not on your PATH. Add it with:"
        printf '    echo '\''export PATH="%s:$PATH"'\'' >> %s && export PATH="%s:$PATH"\n' "$BIN_DIR" "$SHELL_RC" "$BIN_DIR"
        ;;
esac
printf '\n  Examples: %s/src/examples\n  Docs:     %s/src/DOCS.md\n\n' "$HOME_DIR" "$HOME_DIR"
printf '  Try:  animora doctor\n        animora render %s/src/examples/bouncing_ball.json --preview\n\n' "$HOME_DIR"
