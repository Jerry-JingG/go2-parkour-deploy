#!/usr/bin/env bash
# Start the parkour policy server and record incoming proprio obs + outgoing actions + depth artifacts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

LOG_DIR="${PARKOUR_LOG_DIR:-$ROOT_DIR/real_deploy/logs}"
mkdir -p "$LOG_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"

export PARKOUR_LOG_CSV="${PARKOUR_LOG_CSV:-$LOG_DIR/parkour_obs_action_${timestamp}.csv}"
export PARKOUR_LOG_DEPTH_DIR="${PARKOUR_LOG_DEPTH_DIR:-$LOG_DIR/parkour_depth_${timestamp}}"
export PARKOUR_LOG_STEPS="${PARKOUR_LOG_STEPS:-20000}"

echo "Recording parkour obs/action CSV:"
echo "  $PARKOUR_LOG_CSV"
echo "Recording parkour depth artifacts (npz + png):"
echo "  $PARKOUR_LOG_DEPTH_DIR"
echo "  steps: $PARKOUR_LOG_STEPS"
echo ""

exec "$ROOT_DIR/scripts/start_go2_parkour_server.sh"
