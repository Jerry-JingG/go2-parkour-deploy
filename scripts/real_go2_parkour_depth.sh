#!/usr/bin/env bash
# Start the full Route-B parkour depth deployment:
#   1. Python RealSense/TorchScript inference server
#   2. Unitree DDS Go2 low-level controller

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SOCKET_PATH="${SOCKET_PATH:-/tmp/go2_parkour_depth.sock}"
NETWORK_INTERFACE="${NETWORK_INTERFACE:-eth0}"
SERVER_LOG="${SERVER_LOG:-/tmp/go2_parkour_depth_server.log}"
SERVER_PID_FILE="${SERVER_PID_FILE:-/tmp/go2_parkour_depth_server.pid}"

cleanup_server() {
    if [ -f "$SERVER_PID_FILE" ]; then
        local pid
        pid="$(cat "$SERVER_PID_FILE" 2>/dev/null || true)"
        if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            echo "Parkour server stopped: PID $pid"
        fi
        rm -f "$SERVER_PID_FILE"
    fi
    rm -f "$SOCKET_PATH"
}

on_exit() {
    cleanup_server
}

on_signal() {
    echo ""
    echo "Stopping Go2 parkour depth deployment..."
    cleanup_server
    trap - EXIT
    exit 130
}

if [ "${1:-}" = "stop" ]; then
    cleanup_server
    echo "Done."
    exit 0
fi

echo "=== Go2 parkour depth pre-flight checks ==="
[ -x "$ROOT_DIR/real_deploy/unitree_rl_lab/deploy/robots/go2/build/go2_ctrl" ] || {
    echo "ERROR: go2_ctrl not built."
    echo "Run: ./scripts/build_go2_parkour_ctrl.sh"
    exit 1
}
[ -f "$ROOT_DIR/real_deploy/policies/parkour_depth/exported_deploy/policy.pt" ] || {
    echo "ERROR: policy assets not prepared."
    echo "Run: ./scripts/prepare_go2_parkour_assets.sh"
    exit 1
}
if command -v ip >/dev/null 2>&1; then
    ip link show "$NETWORK_INTERFACE" >/dev/null 2>&1 || {
        echo "ERROR: network interface not found: $NETWORK_INTERFACE"
        echo "Check with: ip addr"
        exit 1
    }
fi

cleanup_server
trap on_exit EXIT
trap on_signal INT TERM

echo "[1/2] Starting parkour depth inference server..."
SOCKET_PATH="$SOCKET_PATH" "$ROOT_DIR/scripts/real_go2_parkour_depth_server.sh" > "$SERVER_LOG" 2>&1 &
echo $! > "$SERVER_PID_FILE"
echo "  PID: $(cat "$SERVER_PID_FILE")"
echo "  Log: $SERVER_LOG"

WAITED=0
while [ ! -S "$SOCKET_PATH" ] && [ "$WAITED" -lt 60 ]; do
    server_pid="$(cat "$SERVER_PID_FILE" 2>/dev/null || true)"
    if [ -n "${server_pid:-}" ] && ! kill -0 "$server_pid" 2>/dev/null; then
        echo "ERROR: parkour server exited before socket was ready."
        echo "Check log: tail -n 120 $SERVER_LOG"
        exit 1
    fi
    sleep 1
    WAITED=$((WAITED + 1))
    echo -n "."
done
echo ""

[ -S "$SOCKET_PATH" ] || {
    echo "ERROR: parkour server did not create socket within 60s."
    echo "Check log: tail -n 120 $SERVER_LOG"
    exit 1
}

echo "[1/2] Server ready: $SOCKET_PATH"
echo "[2/2] Starting Go2 controller..."
echo "========================================================"
echo "Remote: LT+A FixStand, LT+X record stand burst, Start policy, LT+B stop."
echo "Keyboard when NETWORK_INTERFACE=lo: 0 FixStand, R record stand burst, 1 policy, W/S velocity, P stop."
echo "Keep the robot suspended for first tests."
echo "========================================================"
echo ""

SOCKET_PATH="$SOCKET_PATH" "$ROOT_DIR/scripts/real_go2_parkour_depth_ctrl.sh" "$NETWORK_INTERFACE"
