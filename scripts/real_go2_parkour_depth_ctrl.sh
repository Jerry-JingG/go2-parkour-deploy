#!/usr/bin/env bash
# Foreground Go2 DDS controller for the parkour depth Route-B deployment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

GO2_BUILD_DIR="${GO2_BUILD_DIR:-$ROOT_DIR/real_deploy/unitree_rl_lab/deploy/robots/go2/build}"
GO2_CTRL="${GO2_CTRL:-$GO2_BUILD_DIR/go2_ctrl}"
SOCKET_PATH="${SOCKET_PATH:-/tmp/go2_parkour_depth.sock}"
NETWORK_INTERFACE="${1:-${NETWORK_INTERFACE:-eth0}}"
KEYBOARD_CONTROL="${KEYBOARD_CONTROL:-}"

if [ -z "$KEYBOARD_CONTROL" ]; then
    if [ "$NETWORK_INTERFACE" = "lo" ]; then
        KEYBOARD_CONTROL=1
    else
        KEYBOARD_CONTROL=0
    fi
fi
export UNITREE_RL_LAB_KEYBOARD_CONTROL="$KEYBOARD_CONTROL"

echo "=== Go2 parkour depth controller ==="
[ -x "$GO2_CTRL" ] || {
    echo "ERROR: go2_ctrl not found or not executable: $GO2_CTRL"
    echo "Build it first:"
    echo "  ./scripts/build_go2_parkour_ctrl.sh"
    exit 1
}
[ -S "$SOCKET_PATH" ] || {
    echo "ERROR: parkour policy socket not found: $SOCKET_PATH"
    echo "Start the server first:"
    echo "  ./scripts/real_go2_parkour_depth_server.sh"
    exit 1
}

if command -v ip >/dev/null 2>&1; then
    ip link show "$NETWORK_INTERFACE" >/dev/null 2>&1 || {
        echo "ERROR: network interface not found: $NETWORK_INTERFACE"
        echo "Check with: ip addr"
        exit 1
    }
fi

if [ -d "$ROOT_DIR/unitree_sdk2/lib" ]; then
    case "$(uname -m)" in
        aarch64|arm64) SDK_ARCH="aarch64" ;;
        *) SDK_ARCH="x86_64" ;;
    esac
    export LD_LIBRARY_PATH="$ROOT_DIR/unitree_sdk2/lib/$SDK_ARCH:$ROOT_DIR/unitree_sdk2/thirdparty/lib/$SDK_ARCH:${LD_LIBRARY_PATH:-}"
fi

echo "Network interface: $NETWORK_INTERFACE"
echo "Socket:            $SOCKET_PATH"
echo "Keyboard control:  $KEYBOARD_CONTROL"
echo ""
echo "Remote control:"
echo "  LT/L2 + A   enter FixStand"
echo "  Start       enter parkour policy"
echo "  Left stick  forward velocity only"
echo "  LT/L2 + B   Passive / stop policy"
echo ""
if [ "$KEYBOARD_CONTROL" != "0" ]; then
    echo "Keyboard control:"
    echo "  0           enter FixStand"
    echo "  1           enter parkour policy"
    echo "  W/S         increase/decrease forward velocity"
    echo "  Space       zero velocity"
    echo "  P or 9      Passive / stop policy"
    echo ""
fi
echo "Keep the robot suspended for first policy tests."
echo ""

cd "$GO2_BUILD_DIR"
exec "$GO2_CTRL" --network "$NETWORK_INTERFACE"
