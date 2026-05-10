#!/usr/bin/env bash
# Foreground Python inference server for the Go2 parkour depth policy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ASSET_DIR="${ASSET_DIR:-$ROOT_DIR/real_deploy/policies/parkour_depth}"
SERVER_SCRIPT="${SERVER_SCRIPT:-$ROOT_DIR/real_deploy/parkour_depth_inference_server.py}"
SOCKET_PATH="${SOCKET_PATH:-/tmp/go2_parkour_depth.sock}"
CONDA_ENV="${CONDA_ENV:-Isaaclab}"
PARKOUR_DEVICE="${PARKOUR_DEVICE:-auto}"
DEPTH_SOURCE="${DEPTH_SOURCE:-realsense}"
ACTION_LPF_ALPHA="${ACTION_LPF_ALPHA:-0.5}"
ACTION_DELTA_LIMIT="${ACTION_DELTA_LIMIT:-0.25}"

if [ "$CONDA_ENV" != "none" ] && [ -z "${CONDA_BASE:-}" ]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    fi
    CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
fi

cleanup_socket() {
    rm -f "$SOCKET_PATH"
}

on_signal() {
    echo ""
    echo "Stopping parkour depth inference server..."
    cleanup_socket
    exit 130
}

echo "=== Go2 parkour depth inference server ==="
[ -f "$SERVER_SCRIPT" ] || { echo "ERROR: server script not found: $SERVER_SCRIPT"; exit 1; }
[ -f "$ASSET_DIR/exported_deploy/policy.pt" ] || {
    echo "ERROR: policy.pt not found in $ASSET_DIR/exported_deploy"
    echo "Run: ./scripts/prepare_go2_parkour_assets.sh"
    exit 1
}
[ -f "$ASSET_DIR/exported_deploy/depth_latest.pt" ] || {
    echo "ERROR: depth_latest.pt not found in $ASSET_DIR/exported_deploy"
    echo "Run: ./scripts/prepare_go2_parkour_assets.sh"
    exit 1
}
if [ "$CONDA_ENV" != "none" ]; then
    [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] || {
        echo "ERROR: conda profile not found: $CONDA_BASE/etc/profile.d/conda.sh"
        echo "Set CONDA_BASE, set CONDA_ENV to an existing env, or use CONDA_ENV=none for system python3."
        exit 1
    }
fi

cleanup_socket
trap cleanup_socket EXIT
trap on_signal INT TERM

if [ "$CONDA_ENV" != "none" ]; then
    set +u
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    set -u
fi

echo "Checking Python torch..."
python3 -c 'import torch; print("  torch:", torch.__version__)'
if [ "$DEPTH_SOURCE" = "realsense" ]; then
    python3 -c 'import pyrealsense2 as rs; print("  pyrealsense2: OK")'
elif [ "$DEPTH_SOURCE" = "ros1" ]; then
    python3 -c 'import rospy; import sensor_msgs.msg; print("  rospy/sensor_msgs: OK")'
fi

SERVER_ARGS=(
    --asset_dir "$ASSET_DIR"
    --socket_path "$SOCKET_PATH"
    --device "$PARKOUR_DEVICE"
    --depth_source "$DEPTH_SOURCE"
    --action_lpf_alpha "$ACTION_LPF_ALPHA"
    --action_delta_limit "$ACTION_DELTA_LIMIT"
)

if [ -n "${REALSENSE_SERIAL:-}" ]; then
    SERVER_ARGS+=(--realsense_serial "$REALSENSE_SERIAL")
fi
if [ -n "${ROS_DEPTH_TOPIC:-}" ]; then
    SERVER_ARGS+=(--ros_depth_topic "$ROS_DEPTH_TOPIC")
fi
if [ -n "${ROS_DEPTH_SCALE:-}" ]; then
    SERVER_ARGS+=(--ros_depth_scale "$ROS_DEPTH_SCALE")
fi
if [ -n "${DEPTH_SOCKET_PATH:-}" ]; then
    SERVER_ARGS+=(--depth_socket_path "$DEPTH_SOCKET_PATH")
fi
if [ -n "${DEPTH_CONNECT_TIMEOUT_S:-}" ]; then
    SERVER_ARGS+=(--depth_connect_timeout_s "$DEPTH_CONNECT_TIMEOUT_S")
fi
if [ -n "${FOOT_FORCE_THRESHOLD:-}" ]; then
    SERVER_ARGS+=(--foot_force_threshold "$FOOT_FORCE_THRESHOLD")
fi
if [ -n "${PARKOUR_LOG_CSV:-}" ]; then
    SERVER_ARGS+=(--log_csv "$PARKOUR_LOG_CSV" --log_steps "${PARKOUR_LOG_STEPS:-300}")
fi
if [ -n "${PARKOUR_LOG_DEPTH_DIR:-}" ]; then
    SERVER_ARGS+=(--log_depth_dir "$PARKOUR_LOG_DEPTH_DIR")
fi
if [ -n "${DEPTH_ROTATE:-}" ]; then
    SERVER_ARGS+=(--depth_rotate "$DEPTH_ROTATE")
fi
if [ -n "${DEPTH_FLIP:-}" ]; then
    SERVER_ARGS+=(--depth_flip "$DEPTH_FLIP")
fi

echo "Asset dir:       $ASSET_DIR"
echo "Socket:          $SOCKET_PATH"
echo "Conda env:       $CONDA_ENV"
echo "Device:          $PARKOUR_DEVICE"
echo "Depth source:    $DEPTH_SOURCE"
echo "Action LPF:      $ACTION_LPF_ALPHA"
echo "Action d-limit:  $ACTION_DELTA_LIMIT"
if [ -n "${FOOT_FORCE_THRESHOLD:-}" ]; then
    echo "Foot threshold:  $FOOT_FORCE_THRESHOLD"
fi
if [ -n "${PARKOUR_LOG_DEPTH_DIR:-}" ]; then
    echo "Depth log dir:   $PARKOUR_LOG_DEPTH_DIR"
fi
echo "Ctrl+C to stop."
echo ""

exec python3 "$SERVER_SCRIPT" "${SERVER_ARGS[@]}"
