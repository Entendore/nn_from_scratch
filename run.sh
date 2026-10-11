#!/usr/bin/env bash
# Launch script for Neural Networks from Scratch v2.0
# Handles the libEGL shared library dependency

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Provide libEGL.so.1 if not available on the system
if ! ldconfig -p 2>/dev/null | grep -q "libEGL.so.1"; then
    PLAYWRIGHT_LIB=$(find /home/z/.cache/ms-playwright -name "libEGL.so" -path "*/chrome-linux64/*" 2>/dev/null | head -1)
    if [ -n "$PLAYWRIGHT_LIB" ]; then
        export LD_LIBRARY_PATH="$(dirname "$PLAYWRIGHT_LIB"):$LD_LIBRARY_PATH"
    fi
fi

# Set Qt platform if no display
if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ]; then
    export QT_QPA_PLATFORM=offscreen
fi

cd "$SCRIPT_DIR"
exec python3 app.py "$@"