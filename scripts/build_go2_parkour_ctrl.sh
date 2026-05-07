#!/usr/bin/env bash
# Build the Route-B Go2 controller for the parkour depth policy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

GO2_DIR="${GO2_DIR:-$ROOT_DIR/real_deploy/unitree_rl_lab/deploy/robots/go2}"
GO2_BUILD_DIR="${GO2_BUILD_DIR:-$GO2_DIR/build}"
TXL_SOCKET_ONLY="${TXL_SOCKET_ONLY:-ON}"

CMAKE_ARGS=("-DTXL_SOCKET_ONLY=$TXL_SOCKET_ONLY")

if [ -n "${UNITREE_SDK_ROOT:-}" ]; then
    CMAKE_ARGS+=("-DUNITREE_SDK_ROOT=$UNITREE_SDK_ROOT")
elif [ -d "$ROOT_DIR/unitree_sdk2/include" ]; then
    CMAKE_ARGS+=("-DUNITREE_SDK_ROOT=$ROOT_DIR/unitree_sdk2")
elif [ -d "/home/jing/robot_lab/unitree_rl_lab_go2_deploy/unitree_sdk2/include" ]; then
    CMAKE_ARGS+=("-DUNITREE_SDK_ROOT=/home/jing/robot_lab/unitree_rl_lab_go2_deploy/unitree_sdk2")
fi

echo "Building Go2 parkour controller"
echo "  Source: $GO2_DIR"
echo "  Build:  $GO2_BUILD_DIR"
echo "  TXL_SOCKET_ONLY=$TXL_SOCKET_ONLY"
if [ -n "${UNITREE_SDK_ROOT:-}" ]; then
    echo "  UNITREE_SDK_ROOT=$UNITREE_SDK_ROOT"
elif [ -d "$ROOT_DIR/unitree_sdk2/include" ]; then
    echo "  UNITREE_SDK_ROOT=$ROOT_DIR/unitree_sdk2"
elif [ -d "/home/jing/robot_lab/unitree_rl_lab_go2_deploy/unitree_sdk2/include" ]; then
    echo "  UNITREE_SDK_ROOT=/home/jing/robot_lab/unitree_rl_lab_go2_deploy/unitree_sdk2"
else
    echo "  UNITREE_SDK_ROOT not set; using system /usr/local SDK paths"
fi

cmake -S "$GO2_DIR" -B "$GO2_BUILD_DIR" "${CMAKE_ARGS[@]}"
cmake --build "$GO2_BUILD_DIR" -j "$(nproc)"

echo "Done: $GO2_BUILD_DIR/go2_ctrl"
