#!/usr/bin/env bash
# Run the optional librealsense C++ depth streamer used by DEPTH_SOURCE=librealsense_socket.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

STREAMER="${STREAMER:-$ROOT_DIR/real_deploy/realsense_depth_streamer}"
DEPTH_SOCKET_PATH="${DEPTH_SOCKET_PATH:-/tmp/go2_realsense_depth.sock}"
REALSENSE_WIDTH="${REALSENSE_WIDTH:-424}"
REALSENSE_HEIGHT="${REALSENSE_HEIGHT:-240}"
REALSENSE_FPS="${REALSENSE_FPS:-30}"
FRAME_TIMEOUT_MS="${FRAME_TIMEOUT_MS:-200}"

[ -x "$STREAMER" ] || {
    echo "ERROR: RealSense depth streamer not found: $STREAMER"
    echo "Build it first:"
    echo "  ./scripts/build_realsense_depth_streamer.sh"
    exit 1
}

ARGS=(
    --socket_path "$DEPTH_SOCKET_PATH"
    --width "$REALSENSE_WIDTH"
    --height "$REALSENSE_HEIGHT"
    --fps "$REALSENSE_FPS"
    --timeout_ms "$FRAME_TIMEOUT_MS"
)

if [ -n "${REALSENSE_SERIAL:-}" ]; then
    ARGS+=(--serial "$REALSENSE_SERIAL")
fi

echo "=== RealSense depth streamer ==="
echo "Socket: $DEPTH_SOCKET_PATH"
echo "Stream: ${REALSENSE_WIDTH}x${REALSENSE_HEIGHT}@${REALSENSE_FPS}"
echo ""

exec "$STREAMER" "${ARGS[@]}"
