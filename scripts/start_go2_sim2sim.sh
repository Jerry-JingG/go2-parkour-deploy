#!/usr/bin/env bash
# Start Go2 MuJoCo sim2sim. Defaults to flat scene, keyboard control,
# command_x=0.0, depth camera visualization, and real-deploy action filtering.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SCENE="${SCENE:-flat}"
RL_LIB="${RL_LIB:-rsl_rl}"
TASK="${TASK:-unitree_go2_parkour}"
EXPID="${EXPID:-2025-09-03_12-07-56}"
INTERFACE="${INTERFACE:-lo}"
COMMAND_X="${COMMAND_X:-0.0}"
N_EVAL="${N_EVAL:-10}"
KEYBOARD_CONTROL="${KEYBOARD_CONTROL:-1}"
USE_JOYSTICK="${USE_JOYSTICK:-0}"
SHOW_DEPTH="${SHOW_DEPTH:-1}"
PYTHON_BIN="${PYTHON_BIN:-}"
LABPARKOUR_ROOT="${LABPARKOUR_ROOT:-}"
RECORD="${RECORD:-0}"
RECORD_DIR="${RECORD_DIR:-$ROOT_DIR/mujoco_logs}"
RECORD_CSV="${RECORD_CSV:-}"
RECORD_DEPTH_DIR="${RECORD_DEPTH_DIR:-}"
RECORD_DEPTH_EVERY="${RECORD_DEPTH_EVERY:-5}"
RECORD_FOOT_FORCE_THRESHOLD="${RECORD_FOOT_FORCE_THRESHOLD:-2.0}"
ACTION_LPF_ALPHA="${ACTION_LPF_ALPHA:-0.5}"
ACTION_DELTA_LIMIT="${ACTION_DELTA_LIMIT:-0.25}"
ACTION_CLIP="${ACTION_CLIP:-4.8}"
STAND_RECORD_SECONDS="${STAND_RECORD_SECONDS:-5.0}"

if [ -z "$LABPARKOUR_ROOT" ]; then
    for candidate in \
        "/home/jing/IsaacLab/Camera_offline_Labparkour" \
        "$ROOT_DIR/../IsaacLab/Camera_offline_Labparkour" \
        "$HOME/IsaacLab/Camera_offline_Labparkour"; do
        if [ -d "$candidate/parkour_isaaclab" ]; then
            LABPARKOUR_ROOT="$candidate"
            break
        fi
    done
fi

if [ -z "$PYTHON_BIN" ]; then
    PYTHON_CANDIDATES=()
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
        PYTHON_CANDIDATES+=("$CONDA_PREFIX/bin/python")
    fi
    PYTHON_CANDIDATES+=(python python3)

    for candidate in "${PYTHON_CANDIDATES[@]}"; do
        if [ -x "$candidate" ]; then
            resolved_python="$candidate"
        elif command -v "$candidate" >/dev/null 2>&1; then
            resolved_python="$(command -v "$candidate")"
        else
            continue
        fi

        if "$resolved_python" -c "import torch" >/dev/null 2>&1; then
            PYTHON_BIN="$resolved_python"
            break
        fi
    done
    if [ -z "$PYTHON_BIN" ]; then
        if command -v python >/dev/null 2>&1; then
            PYTHON_BIN="$(command -v python)"
        elif command -v python3 >/dev/null 2>&1; then
            PYTHON_BIN="$(command -v python3)"
        else
            echo "ERROR: neither python nor python3 was found." >&2
            exit 1
        fi
    fi
fi

usage() {
    cat <<EOF
Usage: $(basename "$0") [options] [-- extra python args]

Options:
  --scene flat|parkour       Scene to run. Default: $SCENE
  --flat                     Shortcut for --scene flat.
  --parkour                  Shortcut for --scene parkour.
  --rl_lib NAME              RL library name. Default: $RL_LIB
  --task NAME                Task name. Default: $TASK
  --expid ID                 Experiment id. Default: $EXPID
  --interface IFACE          Network interface. Default: $INTERFACE
  --command_x VALUE          Initial/constant x velocity. Default: $COMMAND_X
  --n_eval N                 Number of eval episodes. Default: $N_EVAL
  --keyboard-control         Enable keyboard FSM/control. Default.
  --no-keyboard-control      Disable keyboard FSM/control.
  --use-joystick             Enable joystick input.
  --show-depth               Show MuJoCo depth camera window. Default.
  --no-show-depth            Do not show depth camera window.
  --python PATH              Python executable. Default: auto-detect torch-enabled python.
  --labparkour-root PATH     Camera_offline_Labparkour root. Default: auto-detect.
  --record                   Record MuJoCo obs/action CSV and depth trace.
  --record-dir PATH          Recording root. Default: $RECORD_DIR
  --record-csv PATH          CSV output path. Default: auto under --record-dir.
  --record-depth-dir PATH    Depth trace directory. Default: auto beside CSV.
  --record-depth-every N     Save depth every N policy steps. Default: $RECORD_DEPTH_EVERY
  --record-foot-threshold V  Foot-force threshold written to CSV. Default: $RECORD_FOOT_FORCE_THRESHOLD
  --action-lpf-alpha V       Action low-pass alpha. Default: $ACTION_LPF_ALPHA
  --action-delta-limit V     Max per-step action delta. Default: $ACTION_DELTA_LIMIT
  --action-clip V            Final action clip, 0 disables. Default: $ACTION_CLIP
  --stand-record-seconds V   Seconds recorded when pressing r. Default: $STAND_RECORD_SECONDS
  -h, --help                 Show this help.

Examples:
  $(basename "$0")
  $(basename "$0") --scene parkour
  $(basename "$0") --parkour --command_x 0.2 --no-show-depth
EOF
}

EXTRA_ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --scene)
            SCENE="$2"
            shift 2
            ;;
        --flat)
            SCENE="flat"
            shift
            ;;
        --parkour|--non-flat|--nonflat)
            SCENE="parkour"
            shift
            ;;
        --rl_lib)
            RL_LIB="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --expid)
            EXPID="$2"
            shift 2
            ;;
        --interface)
            INTERFACE="$2"
            shift 2
            ;;
        --command_x)
            COMMAND_X="$2"
            shift 2
            ;;
        --n_eval)
            N_EVAL="$2"
            shift 2
            ;;
        --keyboard-control|--keyboard_control)
            KEYBOARD_CONTROL=1
            shift
            ;;
        --no-keyboard-control|--no-keyboard_control)
            KEYBOARD_CONTROL=0
            shift
            ;;
        --use-joystick|--use_joystick)
            USE_JOYSTICK=1
            shift
            ;;
        --show-depth|--show_depth)
            SHOW_DEPTH=1
            shift
            ;;
        --no-show-depth|--no-show_depth)
            SHOW_DEPTH=0
            shift
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --labparkour-root)
            LABPARKOUR_ROOT="$2"
            shift 2
            ;;
        --record)
            RECORD=1
            shift
            ;;
        --record-dir)
            RECORD_DIR="$2"
            shift 2
            ;;
        --record-csv|--record_csv)
            RECORD=1
            RECORD_CSV="$2"
            shift 2
            ;;
        --record-depth-dir|--record_depth_dir)
            RECORD=1
            RECORD_DEPTH_DIR="$2"
            shift 2
            ;;
        --record-depth-every|--record_depth_every)
            RECORD_DEPTH_EVERY="$2"
            shift 2
            ;;
        --record-foot-threshold|--record_foot_threshold|--record-foot-force-threshold|--record_foot_force_threshold)
            RECORD_FOOT_FORCE_THRESHOLD="$2"
            shift 2
            ;;
        --action-lpf-alpha|--action_lpf_alpha)
            ACTION_LPF_ALPHA="$2"
            shift 2
            ;;
        --action-delta-limit|--action_delta_limit)
            ACTION_DELTA_LIMIT="$2"
            shift 2
            ;;
        --action-clip|--action_clip)
            ACTION_CLIP="$2"
            shift 2
            ;;
        --stand-record-seconds|--stand_record_seconds)
            STAND_RECORD_SECONDS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

case "$SCENE" in
    flat)
        ENTRYPOINT="$ROOT_DIR/scripts/go2_deploy_flat.py"
        MUJOCO_TERRAIN_MODE="flat"
        ;;
    parkour|non-flat|nonflat)
        SCENE="parkour"
        ENTRYPOINT="$ROOT_DIR/scripts/go2_deploy.py"
        MUJOCO_TERRAIN_MODE="parkour"
        ;;
    *)
        echo "ERROR: unknown scene '$SCENE'. Use flat or parkour." >&2
        exit 1
        ;;
esac

ARGS=(
    "$ENTRYPOINT"
    --rl_lib "$RL_LIB"
    --task "$TASK"
    --expid "$EXPID"
    --interface "$INTERFACE"
    --command_x "$COMMAND_X"
    --n_eval "$N_EVAL"
    --action_lpf_alpha "$ACTION_LPF_ALPHA"
    --action_delta_limit "$ACTION_DELTA_LIMIT"
    --action_clip "$ACTION_CLIP"
    --stand_record_seconds "$STAND_RECORD_SECONDS"
)

if [ "$KEYBOARD_CONTROL" = "1" ]; then
    ARGS+=(--keyboard_control)
fi

if [ "$USE_JOYSTICK" = "1" ]; then
    ARGS+=(--use_joystick)
fi

if [ "$SHOW_DEPTH" = "1" ]; then
    ARGS+=(--show_depth)
fi

if [ "$RECORD" = "1" ]; then
    mkdir -p "$RECORD_DIR"
    timestamp="$(date +%Y%m%d_%H%M%S)"
    if [ -z "$RECORD_CSV" ]; then
        RECORD_CSV="$RECORD_DIR/mujoco_obs_action_${timestamp}.csv"
    fi
    if [ -z "$RECORD_DEPTH_DIR" ]; then
        RECORD_DEPTH_DIR="$RECORD_DIR/mujoco_depth_${timestamp}"
    fi
    ARGS+=(--record_csv "$RECORD_CSV")
    ARGS+=(--record_depth_dir "$RECORD_DEPTH_DIR")
    ARGS+=(--record_depth_every "$RECORD_DEPTH_EVERY")
    ARGS+=(--record_foot_force_threshold "$RECORD_FOOT_FORCE_THRESHOLD")
fi

ARGS+=("${EXTRA_ARGS[@]}")

cd "$ROOT_DIR"

export MUJOCO_TERRAIN_MODE

if [ -n "$LABPARKOUR_ROOT" ]; then
    export PYTHONPATH="$ROOT_DIR:$LABPARKOUR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
else
    export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
    echo "WARNING: parkour_isaaclab root was not auto-detected." >&2
    echo "Set LABPARKOUR_ROOT or pass --labparkour-root /path/to/Camera_offline_Labparkour if import fails." >&2
fi

echo "=== Go2 MuJoCo sim2sim ==="
echo "Scene: $SCENE"
echo "Terrain flag mode: $MUJOCO_TERRAIN_MODE"
echo "Interface: $INTERFACE"
echo "Keyboard control: $KEYBOARD_CONTROL"
echo "Command x: $COMMAND_X"
echo "Show depth: $SHOW_DEPTH"
echo "Action LPF alpha: $ACTION_LPF_ALPHA"
echo "Action delta limit: $ACTION_DELTA_LIMIT"
echo "Action clip: $ACTION_CLIP"
echo "Stand record seconds: $STAND_RECORD_SECONDS"
echo "Record: $RECORD"
if [ "$RECORD" = "1" ]; then
    echo "Record CSV: $RECORD_CSV"
    echo "Record depth: $RECORD_DEPTH_DIR"
    echo "Record depth every: $RECORD_DEPTH_EVERY"
fi
echo "LabParkour root: ${LABPARKOUR_ROOT:-not found}"
echo "Python: $PYTHON_BIN"
echo "Entrypoint: $ENTRYPOINT"
echo ""
echo "Keyboard FSM: 0 stand, 1 policy, r record stand burst, W/S velocity, P/9 passive."
echo "Depth viewer: run ./scripts/serve_go2_depth_viewer.sh and open http://127.0.0.1:8765/index.html"
echo ""

exec "$PYTHON_BIN" "${ARGS[@]}"
