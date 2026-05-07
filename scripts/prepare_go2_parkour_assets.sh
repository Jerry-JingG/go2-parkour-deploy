#!/usr/bin/env bash
# Copy the MuJoCo-validated parkour deployment assets into this repo.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

EXP_SOURCE="${1:-/home/jing/IsaacLab/Camera_offline_Labparkour/logs/rsl_rl/unitree_go2_parkour/2025-09-03_12-07-56}"
ASSET_DIR="${ASSET_DIR:-$ROOT_DIR/real_deploy/policies/parkour_depth}"

echo "Preparing parkour depth assets"
echo "  Source: $EXP_SOURCE"
echo "  Target: $ASSET_DIR"

for path in \
    "$EXP_SOURCE/exported_deploy/policy.pt" \
    "$EXP_SOURCE/exported_deploy/depth_latest.pt" \
    "$EXP_SOURCE/params/env.yaml" \
    "$EXP_SOURCE/params/agent.yaml"
do
    [ -f "$path" ] || { echo "ERROR: missing asset: $path"; exit 1; }
done

mkdir -p "$ASSET_DIR/exported_deploy" "$ASSET_DIR/params"
cp "$EXP_SOURCE/exported_deploy/policy.pt" "$ASSET_DIR/exported_deploy/policy.pt"
cp "$EXP_SOURCE/exported_deploy/depth_latest.pt" "$ASSET_DIR/exported_deploy/depth_latest.pt"
cp "$EXP_SOURCE/params/env.yaml" "$ASSET_DIR/params/env.yaml"
cp "$EXP_SOURCE/params/agent.yaml" "$ASSET_DIR/params/agent.yaml"

echo "Done."
