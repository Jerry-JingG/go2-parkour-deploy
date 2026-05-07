# Go2 Parkour Depth Real Deployment

This Route-B deployment runs the same MuJoCo-tested parkour depth policy:

```text
real_deploy/policies/parkour_depth/exported_deploy/policy.pt
real_deploy/policies/parkour_depth/exported_deploy/depth_latest.pt
```

The C++ side owns Unitree DDS, FSM, standing posture, low-level PD commands, and safety transitions. The Python side owns RealSense depth preprocessing, depth encoder inference, observation history, and policy inference. The two processes communicate through a Unix socket.

## Prepare Assets

The default expid is `2025-09-03_12-07-56`.

```bash
./scripts/prepare_go2_parkour_assets.sh
```

This copies `policy.pt`, `depth_latest.pt`, `env.yaml`, and `agent.yaml` into:

```bash
real_deploy/policies/parkour_depth
```

## Build Controller

On the robot, install Unitree SDK2 first or set `UNITREE_SDK_ROOT` to a copied SDK directory.

```bash
./scripts/build_go2_parkour_ctrl.sh
```

If the SDK is inside the repo:

```bash
UNITREE_SDK_ROOT=$PWD/unitree_sdk2 ./scripts/build_go2_parkour_ctrl.sh
```

The executable is:

```bash
real_deploy/unitree_rl_lab/deploy/robots/go2/build/go2_ctrl
```

## Start Deployment

Use two terminals for `pyrealsense2` or ROS depth. Use three terminals for the direct librealsense streamer.

Terminal 1:

```bash
CONDA_ENV=Isaaclab PARKOUR_DEVICE=auto DEPTH_SOURCE=realsense \
./scripts/real_go2_parkour_depth_server.sh
```

If `pyrealsense2` is not compatible with the robot's glibc, run a ROS RealSense driver and read the depth topic instead:

```bash
source /opt/ros/noetic/setup.bash
CONDA_ENV=rl_deploy PARKOUR_DEVICE=cpu DEPTH_SOURCE=ros1 \
ROS_DEPTH_TOPIC=/camera/depth/image_rect_raw \
./scripts/real_go2_parkour_depth_server.sh
```

ROS depth encodings supported by the server:

```text
16UC1 / mono16  millimeters by default
32FC1           meters by default
```

Another option is to use librealsense directly through the C++ depth streamer. This avoids `pyrealsense2` and ROS, but requires `librealsense2-dev` on the robot:

```bash
./scripts/build_realsense_depth_streamer.sh
```

Then use three terminals for first tests.

Terminal 0:

```bash
./scripts/start_go2_parkour_camera.sh
```

Terminal 1:

```bash
./scripts/start_go2_parkour_server.sh
```

To record incoming proprio observations and outgoing policy actions:

```bash
./scripts/start_go2_parkour_server_record.sh
```

The default CSV path is:

```text
real_deploy/logs/parkour_obs_action_<timestamp>.csv
```

Terminal 2:

```bash
./scripts/start_go2_parkour_ctrl.sh
```

For a single combined launcher:

```bash
NETWORK_INTERFACE=eth0 ./scripts/real_go2_parkour_depth.sh
```

Replace `eth0` with the DDS interface shown by `ip addr`.

## Controls

Remote:

```text
LT/L2 + A   enter FixStand
Start       enter parkour policy
Left stick  forward velocity only
LT/L2 + B   Passive / stop policy
```

Keyboard is enabled automatically when `NETWORK_INTERFACE=lo`, or manually with `KEYBOARD_CONTROL=1`:

```text
0           enter FixStand
1           enter parkour policy
W/S         increase/decrease forward velocity
Space       zero velocity
P or 9      Passive / stop policy
```

## Local Smoke Tests

Python model chain without RealSense:

```bash
conda run --no-capture-output -n Isaaclab \
python real_deploy/parkour_depth_inference_server.py \
  --depth_source mock --device cpu --self_test_steps 3
```

Build check:

```bash
./scripts/build_go2_parkour_ctrl.sh
ldd real_deploy/unitree_rl_lab/deploy/robots/go2/build/go2_ctrl
```

`ldd` should not show ONNXRuntime.

## Safety Notes

First tests should be suspended. Start in `FixStand`, enter policy for 10-20 seconds, and verify the stop controls before ground testing. Initial forward velocity is capped at `0.35 m/s` and ramped at `0.35 m/s^2` in `deploy.yaml`.

Useful runtime knobs:

```bash
PARKOUR_LOG_CSV=/tmp/parkour_depth_log.csv
PARKOUR_LOG_STEPS=20000
ACTION_LPF_ALPHA=0.5
ACTION_DELTA_LIMIT=0.25
DEPTH_ROTATE=0
DEPTH_FLIP=none
REALSENSE_SERIAL=<serial>
ROS_DEPTH_TOPIC=/camera/depth/image_rect_raw
DEPTH_SOCKET_PATH=/tmp/go2_realsense_depth.sock
```
