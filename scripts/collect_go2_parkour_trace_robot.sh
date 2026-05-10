#!/usr/bin/env bash
# Run on the robot. Collect a bounded Go2 parkour real-deploy trace and package it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_SECONDS=20
NETWORK_INTERFACE="${NETWORK_INTERFACE:-eth0}"
DEPTH_SOURCE="${DEPTH_SOURCE:-librealsense_socket}"
DEPTH_SOCKET_PATH="${DEPTH_SOCKET_PATH:-/tmp/go2_realsense_depth.sock}"
SOCKET_PATH="${SOCKET_PATH:-/tmp/go2_parkour_depth.sock}"
TRACE_STEPS="${PARKOUR_TRACE_STEPS:-0}"
TRACE_EVERY="${PARKOUR_TRACE_EVERY:-1}"
TRACE_RAW_DEPTH_EVERY="${PARKOUR_TRACE_RAW_DEPTH_EVERY:-0}"
TRACE_ROOT=""
OUT_TARBALL=""
KEYBOARD_CONTROL="${KEYBOARD_CONTROL:-0}"
START_CAMERA="auto"

usage() {
    cat <<'EOF'
Usage:
  collect_go2_parkour_trace_robot.sh [options]

Options:
  --seconds N                  Collection window after controller starts [20]
  --network IFACE              Unitree DDS network interface [eth0]
  --depth_source NAME          realsense|ros1|librealsense_socket|mock [librealsense_socket]
  --depth_socket_path PATH     Librealsense depth socket [/tmp/go2_realsense_depth.sock]
  --socket_path PATH           Policy socket [/tmp/go2_parkour_depth.sock]
  --trace_steps N              Max trace rows, 0 means unlimited [0]
  --trace_every N              Record every N inference steps [1]
  --trace_raw_depth_every N    Save raw depth .npy every N recorded rows [0]
  --trace_root PATH            Trace working directory [/tmp/go2_parkour_trace_TIMESTAMP]
  --out PATH                   Output tar.gz [/tmp/go2_parkour_trace_TIMESTAMP.tar.gz]
  --keyboard_control 0|1       Enable keyboard FSM transitions [0]
  --start_camera auto|0|1      Start C++ RealSense streamer for librealsense_socket [auto]
  -h, --help                   Show help
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --seconds) RUN_SECONDS="$2"; shift 2 ;;
        --network) NETWORK_INTERFACE="$2"; shift 2 ;;
        --depth_source) DEPTH_SOURCE="$2"; shift 2 ;;
        --depth_socket_path) DEPTH_SOCKET_PATH="$2"; shift 2 ;;
        --socket_path) SOCKET_PATH="$2"; shift 2 ;;
        --trace_steps) TRACE_STEPS="$2"; shift 2 ;;
        --trace_every) TRACE_EVERY="$2"; shift 2 ;;
        --trace_raw_depth_every) TRACE_RAW_DEPTH_EVERY="$2"; shift 2 ;;
        --trace_root) TRACE_ROOT="$2"; shift 2 ;;
        --out) OUT_TARBALL="$2"; shift 2 ;;
        --keyboard_control) KEYBOARD_CONTROL="$2"; shift 2 ;;
        --start_camera) START_CAMERA="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

TRACE_ID="go2_parkour_trace_$(date +%Y%m%d_%H%M%S)"
TRACE_ROOT="${TRACE_ROOT:-/tmp/$TRACE_ID}"
OUT_TARBALL="${OUT_TARBALL:-/tmp/$TRACE_ID.tar.gz}"
TRACE_DATA_DIR="$TRACE_ROOT/data"
LOG_DIR="$TRACE_ROOT/logs"
CONFIG_DIR="$TRACE_ROOT/config"

mkdir -p "$TRACE_DATA_DIR" "$LOG_DIR" "$CONFIG_DIR"

SERVER_LOG="$LOG_DIR/server.log"
CTRL_LOG="$LOG_DIR/controller.log"
CAMERA_LOG="$LOG_DIR/camera.log"

SERVER_PID=""
CTRL_PID=""
CAMERA_PID=""

stop_pid() {
    local pid="$1"
    local name="$2"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "Stopping $name: PID $pid"
        kill -INT "$pid" 2>/dev/null || true
        sleep 1
    fi
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
        sleep 1
    fi
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$pid" 2>/dev/null || true
    fi
    if [ -n "$pid" ]; then
        wait "$pid" 2>/dev/null || true
    fi
}

cleanup_processes() {
    stop_pid "$CTRL_PID" "controller"
    stop_pid "$SERVER_PID" "server"
    stop_pid "$CAMERA_PID" "camera"
}

on_signal() {
    echo "Interrupted; finalizing trace package."
    cleanup_processes
    package_trace
    exit 130
}

wait_for_socket() {
    local path="$1"
    local timeout_s="$2"
    local waited=0
    while [ ! -S "$path" ] && [ "$waited" -lt "$timeout_s" ]; do
        sleep 1
        waited=$((waited + 1))
    done
    [ -S "$path" ]
}

copy_if_exists() {
    local src="$1"
    local dst="$2"
    if [ -e "$src" ]; then
        cp -a "$src" "$dst"
    fi
}

package_trace() {
    copy_if_exists "$ROOT_DIR/real_deploy/unitree_rl_lab/deploy/robots/go2/config/config.yaml" "$CONFIG_DIR/config.yaml"
    copy_if_exists "$ROOT_DIR/real_deploy/unitree_rl_lab/deploy/robots/go2/config/policy/parkour_depth/params/deploy.yaml" "$CONFIG_DIR/deploy.yaml"
    copy_if_exists "$ROOT_DIR/real_deploy/policies/parkour_depth/params/env.yaml" "$CONFIG_DIR/env.yaml"
    copy_if_exists "$ROOT_DIR/real_deploy/policies/parkour_depth/params/agent.yaml" "$CONFIG_DIR/agent.yaml"

    local git_rev="unknown"
    if command -v git >/dev/null 2>&1; then
        git_rev="$(cd "$ROOT_DIR" && git rev-parse HEAD 2>/dev/null || true)"
        git_rev="${git_rev:-unknown}"
    fi

    cat > "$TRACE_ROOT/manifest.json" <<EOF
{
  "schema": "go2_parkour_trace_bundle_v1",
  "created_at": "$(date -Iseconds)",
  "repo": "$ROOT_DIR",
  "git_rev": "$git_rev",
  "network_interface": "$NETWORK_INTERFACE",
  "depth_source": "$DEPTH_SOURCE",
  "socket_path": "$SOCKET_PATH",
  "depth_socket_path": "$DEPTH_SOCKET_PATH",
  "run_seconds": $RUN_SECONDS,
  "trace_steps": $TRACE_STEPS,
  "trace_every": $TRACE_EVERY,
  "trace_raw_depth_every": $TRACE_RAW_DEPTH_EVERY
}
EOF

    mkdir -p "$(dirname "$OUT_TARBALL")"
    tar -czf "$OUT_TARBALL" -C "$(dirname "$TRACE_ROOT")" "$(basename "$TRACE_ROOT")"
    echo "TRACE_TARBALL=$OUT_TARBALL"
}

trap on_signal INT TERM

cd "$ROOT_DIR"

echo "=== Go2 parkour trace collection ==="
echo "Trace root:      $TRACE_ROOT"
echo "Output tarball:  $OUT_TARBALL"
echo "Run seconds:     $RUN_SECONDS"
echo "Network:         $NETWORK_INTERFACE"
echo "Depth source:    $DEPTH_SOURCE"
echo "Policy socket:   $SOCKET_PATH"

if [ "$DEPTH_SOURCE" = "librealsense_socket" ]; then
    should_start_camera=0
    if [ "$START_CAMERA" = "1" ]; then
        should_start_camera=1
    elif [ "$START_CAMERA" = "auto" ] && [ ! -S "$DEPTH_SOCKET_PATH" ]; then
        should_start_camera=1
    fi

    if [ "$should_start_camera" = "1" ]; then
        echo "Starting RealSense depth streamer..."
        DEPTH_SOCKET_PATH="$DEPTH_SOCKET_PATH" "$ROOT_DIR/scripts/start_go2_parkour_camera.sh" > "$CAMERA_LOG" 2>&1 &
        CAMERA_PID="$!"
        if ! wait_for_socket "$DEPTH_SOCKET_PATH" 20; then
            echo "ERROR: depth socket did not appear: $DEPTH_SOCKET_PATH" >&2
            echo "Check log: $CAMERA_LOG" >&2
            cleanup_processes
            package_trace
            exit 1
        fi
    fi
fi

echo "Starting Python inference server..."
PARKOUR_TRACE_DIR="$TRACE_DATA_DIR" \
PARKOUR_TRACE_STEPS="$TRACE_STEPS" \
PARKOUR_TRACE_EVERY="$TRACE_EVERY" \
PARKOUR_TRACE_RAW_DEPTH_EVERY="$TRACE_RAW_DEPTH_EVERY" \
SOCKET_PATH="$SOCKET_PATH" \
DEPTH_SOURCE="$DEPTH_SOURCE" \
DEPTH_SOCKET_PATH="$DEPTH_SOCKET_PATH" \
"$ROOT_DIR/scripts/real_go2_parkour_depth_server.sh" > "$SERVER_LOG" 2>&1 &
SERVER_PID="$!"

if ! wait_for_socket "$SOCKET_PATH" 60; then
    echo "ERROR: policy socket did not appear: $SOCKET_PATH" >&2
    echo "Check log: $SERVER_LOG" >&2
    cleanup_processes
    package_trace
    exit 1
fi

echo "Starting Go2 controller..."
PARKOUR_TRACE_DIR="$TRACE_DATA_DIR" \
PARKOUR_TRACE_STEPS="$TRACE_STEPS" \
PARKOUR_TRACE_EVERY="$TRACE_EVERY" \
UNITREE_RL_LAB_KEYBOARD_CONTROL="$KEYBOARD_CONTROL" \
"$ROOT_DIR/scripts/real_go2_parkour_depth_ctrl.sh" "$NETWORK_INTERFACE" > "$CTRL_LOG" 2>&1 &
CTRL_PID="$!"

echo "Controller running. Enter Policy mode manually, then stop or wait for collection window."
sleep "$RUN_SECONDS"

cleanup_processes
package_trace
