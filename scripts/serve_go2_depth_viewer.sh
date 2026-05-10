#!/usr/bin/env bash
# Serve the MuJoCo depth debug directory in a browser-friendly way.

set -euo pipefail

DEPTH_DIR="${MUJOCO_DEPTH_DUMP_DIR:-/tmp/go2_mujoco_depth}"
PORT="${PORT:-8765}"
HOST="${HOST:-127.0.0.1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --dir PATH       Depth debug directory. Default: $DEPTH_DIR
  --host HOST      Bind host. Default: $HOST
  --port PORT      HTTP port. Default: $PORT
  --python PATH    Python executable. Default: $PYTHON_BIN
  -h, --help       Show this help.

Open this URL in your browser after starting:
  http://$HOST:$PORT/index.html
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dir)
            DEPTH_DIR="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option '$1'" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [ ! -d "$DEPTH_DIR" ]; then
    echo "ERROR: depth debug directory does not exist yet: $DEPTH_DIR" >&2
    echo "Run ./scripts/start_go2_sim2sim.sh first so it can create depth images." >&2
    exit 1
fi

if [ ! -f "$DEPTH_DIR/index.html" ]; then
    echo "ERROR: viewer index not found: $DEPTH_DIR/index.html" >&2
    echo "Run ./scripts/start_go2_sim2sim.sh again to generate it." >&2
    exit 1
fi

echo "Serving Go2 depth viewer from: $DEPTH_DIR"
echo "Open: http://$HOST:$PORT/index.html"
echo "Press Ctrl-C to stop."
echo ""

cd "$DEPTH_DIR"
exec "$PYTHON_BIN" -m http.server "$PORT" --bind "$HOST"
