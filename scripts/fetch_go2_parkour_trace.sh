#!/usr/bin/env bash
# Run locally. Start trace collection over SSH and fetch the tarball back.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ROBOT=""
REMOTE_REPO=""
OUT_DIR="$ROOT_DIR/traces"
RUN_SECONDS=20
NETWORK_INTERFACE="eth0"
DEPTH_SOURCE="librealsense_socket"
DEPTH_SOCKET_PATH="/tmp/go2_realsense_depth.sock"
SOCKET_PATH="/tmp/go2_parkour_depth.sock"
TRACE_STEPS=0
TRACE_EVERY=1
TRACE_RAW_DEPTH_EVERY=0
KEYBOARD_CONTROL=0
START_CAMERA="auto"

usage() {
    cat <<'EOF'
Usage:
  fetch_go2_parkour_trace.sh --robot USER@HOST --repo /path/to/go2_parkour_deploy [options]

Options:
  --robot USER@HOST             SSH target for the robot
  --repo PATH                   go2_parkour_deploy path on the robot
  --out DIR                     Local output directory [go2_parkour_deploy/traces]
  --seconds N                   Collection window [20]
  --network IFACE               Unitree DDS network interface [eth0]
  --depth_source NAME           realsense|ros1|librealsense_socket|mock [librealsense_socket]
  --depth_socket_path PATH      Librealsense depth socket [/tmp/go2_realsense_depth.sock]
  --socket_path PATH            Policy socket [/tmp/go2_parkour_depth.sock]
  --trace_steps N               Max trace rows, 0 means unlimited [0]
  --trace_every N               Record every N inference steps [1]
  --trace_raw_depth_every N     Save raw depth .npy every N recorded rows [0]
  --keyboard_control 0|1        Enable keyboard FSM transitions [0]
  --start_camera auto|0|1       Start C++ RealSense streamer for librealsense_socket [auto]
  -h, --help                    Show help
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --robot) ROBOT="$2"; shift 2 ;;
        --repo) REMOTE_REPO="$2"; shift 2 ;;
        --out) OUT_DIR="$2"; shift 2 ;;
        --seconds) RUN_SECONDS="$2"; shift 2 ;;
        --network) NETWORK_INTERFACE="$2"; shift 2 ;;
        --depth_source) DEPTH_SOURCE="$2"; shift 2 ;;
        --depth_socket_path) DEPTH_SOCKET_PATH="$2"; shift 2 ;;
        --socket_path) SOCKET_PATH="$2"; shift 2 ;;
        --trace_steps) TRACE_STEPS="$2"; shift 2 ;;
        --trace_every) TRACE_EVERY="$2"; shift 2 ;;
        --trace_raw_depth_every) TRACE_RAW_DEPTH_EVERY="$2"; shift 2 ;;
        --keyboard_control) KEYBOARD_CONTROL="$2"; shift 2 ;;
        --start_camera) START_CAMERA="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

if [ -z "$ROBOT" ] || [ -z "$REMOTE_REPO" ]; then
    usage
    exit 2
fi

quote() {
    printf '%q' "$1"
}

mkdir -p "$OUT_DIR"
REMOTE_LOG="$(mktemp)"

REMOTE_CMD="cd $(quote "$REMOTE_REPO") && bash scripts/collect_go2_parkour_trace_robot.sh"
REMOTE_CMD+=" --seconds $(quote "$RUN_SECONDS")"
REMOTE_CMD+=" --network $(quote "$NETWORK_INTERFACE")"
REMOTE_CMD+=" --depth_source $(quote "$DEPTH_SOURCE")"
REMOTE_CMD+=" --depth_socket_path $(quote "$DEPTH_SOCKET_PATH")"
REMOTE_CMD+=" --socket_path $(quote "$SOCKET_PATH")"
REMOTE_CMD+=" --trace_steps $(quote "$TRACE_STEPS")"
REMOTE_CMD+=" --trace_every $(quote "$TRACE_EVERY")"
REMOTE_CMD+=" --trace_raw_depth_every $(quote "$TRACE_RAW_DEPTH_EVERY")"
REMOTE_CMD+=" --keyboard_control $(quote "$KEYBOARD_CONTROL")"
REMOTE_CMD+=" --start_camera $(quote "$START_CAMERA")"

echo "Starting remote trace collection on $ROBOT"
set +e
ssh "$ROBOT" "$REMOTE_CMD" | tee "$REMOTE_LOG"
SSH_STATUS="${PIPESTATUS[0]}"
set -e

REMOTE_TARBALL="$(awk -F= '/^TRACE_TARBALL=/{print $2}' "$REMOTE_LOG" | tail -n 1)"
rm -f "$REMOTE_LOG"

if [ -z "$REMOTE_TARBALL" ]; then
    echo "ERROR: remote collector did not print TRACE_TARBALL." >&2
    exit 1
fi

LOCAL_TARBALL="$OUT_DIR/$(basename "$REMOTE_TARBALL")"
echo "Fetching $REMOTE_TARBALL"
scp "$ROBOT:$REMOTE_TARBALL" "$LOCAL_TARBALL"

if [ "$SSH_STATUS" -ne 0 ]; then
    echo "WARNING: remote collector exited with status $SSH_STATUS, but a tarball was fetched." >&2
fi

echo "LOCAL_TARBALL=$LOCAL_TARBALL"
