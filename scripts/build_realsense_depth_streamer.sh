#!/usr/bin/env bash
# Build the optional librealsense C++ depth streamer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SRC="${SRC:-$ROOT_DIR/real_deploy/realsense_depth_streamer.cpp}"
OUT="${OUT:-$ROOT_DIR/real_deploy/realsense_depth_streamer}"
CXX_BIN="${CXX:-g++}"

CXXFLAGS=(-std=c++17 -O2 -Wall -Wextra)
LIBS=(-lrealsense2 -pthread)

if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists realsense2; then
    read -r -a PKG_CFLAGS <<<"$(pkg-config --cflags realsense2)"
    read -r -a PKG_LIBS <<<"$(pkg-config --libs realsense2)"
    CXXFLAGS+=("${PKG_CFLAGS[@]}")
    LIBS=("${PKG_LIBS[@]}" -pthread)
fi

echo "Building RealSense depth streamer"
echo "  Source: $SRC"
echo "  Output: $OUT"

"$CXX_BIN" "${CXXFLAGS[@]}" "$SRC" -o "$OUT" "${LIBS[@]}"

echo "Done: $OUT"
