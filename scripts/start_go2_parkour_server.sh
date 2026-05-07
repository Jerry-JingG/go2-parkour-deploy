#!/usr/bin/env bash
# Start the Python parkour depth policy server using the librealsense depth socket.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

export CONDA_ENV="${CONDA_ENV:-rl_deploy}"
export PARKOUR_DEVICE="${PARKOUR_DEVICE:-cpu}"
export DEPTH_SOURCE="${DEPTH_SOURCE:-librealsense_socket}"
export DEPTH_SOCKET_PATH="${DEPTH_SOCKET_PATH:-/tmp/go2_realsense_depth.sock}"

exec "$ROOT_DIR/scripts/real_go2_parkour_depth_server.sh"
