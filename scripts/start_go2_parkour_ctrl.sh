#!/usr/bin/env bash
# Start the Go2 DDS controller for parkour real deployment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

export NETWORK_INTERFACE="${NETWORK_INTERFACE:-eth0}"

exec "$ROOT_DIR/scripts/real_go2_parkour_depth_ctrl.sh" "$NETWORK_INTERFACE"
