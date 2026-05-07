#!/usr/bin/env bash
# Start the RealSense depth streamer for Go2 parkour real deployment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

export REALSENSE_SERIAL="${REALSENSE_SERIAL:-247122071665}"
export REALSENSE_WIDTH="${REALSENSE_WIDTH:-424}"
export REALSENSE_HEIGHT="${REALSENSE_HEIGHT:-240}"
export REALSENSE_FPS="${REALSENSE_FPS:-30}"

exec "$ROOT_DIR/scripts/realsense_depth_streamer.sh"
