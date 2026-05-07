#!/usr/bin/env python3
"""Parkour depth-policy inference server for Route-B Go2 real deployment."""

from __future__ import annotations

import argparse
import csv
import os
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Go2 parkour depth policy inference server")
    default_asset_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "policies",
        "parkour_depth",
    )
    parser.add_argument("--asset_dir", type=str, default=default_asset_dir)
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--depth_encoder", type=str, default=None)
    parser.add_argument("--socket_path", type=str, default="/tmp/go2_parkour_depth.sock")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--depth_source",
        type=str,
        default="realsense",
        choices=["realsense", "ros1", "librealsense_socket", "mock"],
    )
    parser.add_argument("--realsense_serial", type=str, default="")
    parser.add_argument("--realsense_width", type=int, default=424)
    parser.add_argument("--realsense_height", type=int, default=240)
    parser.add_argument("--realsense_fps", type=int, default=30)
    parser.add_argument("--ros_depth_topic", type=str, default="/camera/depth/image_rect_raw")
    parser.add_argument(
        "--ros_depth_scale",
        type=float,
        default=0.0,
        help="Depth scale for ROS images. 0 means auto: 0.001 for 16UC1/mono16, 1.0 for 32FC1.",
    )
    parser.add_argument("--frame_timeout_ms", type=int, default=200)
    parser.add_argument("--depth_socket_path", type=str, default="/tmp/go2_realsense_depth.sock")
    parser.add_argument(
        "--depth_connect_timeout_s",
        type=float,
        default=30.0,
        help="How long to wait for --depth_source librealsense_socket to become available.",
    )
    parser.add_argument("--depth_resize", type=int, nargs=2, default=[58, 87], metavar=("H", "W"))
    parser.add_argument("--depth_max_distance", type=float, default=2.0)
    parser.add_argument("--depth_update_interval", type=int, default=5)
    parser.add_argument("--depth_rotate", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--depth_flip", type=str, default="none", choices=["none", "horizontal", "vertical", "both"])
    parser.add_argument("--num_prop", type=int, default=53)
    parser.add_argument("--num_scan", type=int, default=132)
    parser.add_argument("--num_priv_explicit", type=int, default=9)
    parser.add_argument("--num_priv_latent", type=int, default=29)
    parser.add_argument("--num_hist", type=int, default=10)
    parser.add_argument("--action_dim", type=int, default=12)
    parser.add_argument("--action_lpf_alpha", type=float, default=0.5)
    parser.add_argument("--action_delta_limit", type=float, default=0.25)
    parser.add_argument("--action_clip", type=float, default=4.8)
    parser.add_argument("--log_csv", type=str, default=None)
    parser.add_argument("--log_steps", type=int, default=300)
    parser.add_argument("--self_test_steps", type=int, default=0)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(device_arg)


def recv_exact(conn: socket.socket, n_bytes: int) -> bytes:
    buf = b""
    while len(buf) < n_bytes:
        chunk = conn.recv(n_bytes - len(buf))
        if not chunk:
            raise ConnectionError("client disconnected")
        buf += chunk
    return buf


class DepthSource:
    def read_depth_m(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class MockDepthSource(DepthSource):
    def __init__(self, height: int = 60, width: int = 106, distance_m: float = 2.0):
        self._depth = np.full((height, width), distance_m, dtype=np.float32)

    def read_depth_m(self) -> np.ndarray:
        return self._depth


class RealSenseDepthSource(DepthSource):
    def __init__(self, args: argparse.Namespace):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is required for --depth_source realsense. "
                "Use --depth_source mock for local socket tests."
            ) from exc

        self._rs = rs
        self._timeout_ms = args.frame_timeout_ms
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        if args.realsense_serial:
            self._config.enable_device(args.realsense_serial)
        self._config.enable_stream(
            rs.stream.depth,
            args.realsense_width,
            args.realsense_height,
            rs.format.z16,
            args.realsense_fps,
        )
        profile = self._pipeline.start(self._config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

    def read_depth_m(self) -> np.ndarray:
        frames = self._pipeline.wait_for_frames(self._timeout_ms)
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            raise TimeoutError("RealSense depth frame was not available")
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        return depth * float(self._depth_scale)

    def close(self) -> None:
        self._pipeline.stop()


class ROS1DepthSource(DepthSource):
    def __init__(self, args: argparse.Namespace):
        try:
            import rospy
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise RuntimeError(
                "rospy and sensor_msgs are required for --depth_source ros1. "
                "Source your ROS environment before starting this server."
            ) from exc

        self._rospy = rospy
        self._timeout_s = args.frame_timeout_ms / 1000.0
        self._depth_scale_arg = args.ros_depth_scale
        self._lock = threading.Lock()
        self._frame_event = threading.Event()
        self._latest_depth: np.ndarray | None = None
        self._latest_time = 0.0

        if not rospy.core.is_initialized():
            rospy.init_node("go2_parkour_depth_policy_server", anonymous=True, disable_signals=True)
        self._subscriber = rospy.Subscriber(
            args.ros_depth_topic,
            Image,
            self._callback,
            queue_size=1,
            buff_size=2**24,
        )
        first_wait_s = max(2.0, self._timeout_s)
        if not self._frame_event.wait(first_wait_s):
            raise TimeoutError(f"No ROS depth frame received on {args.ros_depth_topic} within {first_wait_s:.1f}s")

    def _encoding_dtype_and_scale(self, encoding: str) -> tuple[np.dtype, float]:
        enc = encoding.lower()
        if enc in ("16uc1", "mono16"):
            scale = self._depth_scale_arg if self._depth_scale_arg > 0.0 else 0.001
            return np.dtype(np.uint16), scale
        if enc == "32fc1":
            scale = self._depth_scale_arg if self._depth_scale_arg > 0.0 else 1.0
            return np.dtype(np.float32), scale
        raise ValueError(f"Unsupported ROS depth image encoding: {encoding}")

    def _image_to_depth_m(self, msg) -> np.ndarray:
        dtype, scale = self._encoding_dtype_and_scale(msg.encoding)
        dtype = dtype.newbyteorder(">" if msg.is_bigendian else "<")
        row_stride = msg.step // dtype.itemsize
        depth = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, row_stride)
        depth = depth[:, : msg.width].astype(np.float32, copy=True)
        return depth * scale

    def _callback(self, msg) -> None:
        try:
            depth = self._image_to_depth_m(msg)
        except Exception as exc:
            print(f"[WARN] Dropping ROS depth frame: {exc}", flush=True)
            return
        with self._lock:
            self._latest_depth = depth
            self._latest_time = time.monotonic()
            self._frame_event.set()

    def read_depth_m(self) -> np.ndarray:
        deadline = time.monotonic() + self._timeout_s
        while True:
            with self._lock:
                if self._latest_depth is not None and time.monotonic() - self._latest_time <= self._timeout_s:
                    return self._latest_depth.copy()

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError("Timed out waiting for a fresh ROS depth frame")
            self._frame_event.clear()
            self._frame_event.wait(remaining)

    def close(self) -> None:
        self._subscriber.unregister()


class LibrealsenseSocketDepthSource(DepthSource):
    MAGIC = 0x48545044  # "DPTH" little-endian

    def __init__(self, args: argparse.Namespace):
        self._socket_path = args.depth_socket_path
        self._timeout_s = args.frame_timeout_ms / 1000.0
        self._connect_timeout_s = args.depth_connect_timeout_s
        self._sock: socket.socket | None = None
        self._connect()

    def _connect(self) -> None:
        self.close()
        deadline = time.monotonic() + max(self._connect_timeout_s, self._timeout_s)
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(self._timeout_s)
                sock.connect(self._socket_path)
                self._sock = sock
                return
            except OSError as exc:
                last_error = exc
                time.sleep(0.05)
        raise TimeoutError(
            f"Failed to connect to depth socket {self._socket_path}: {last_error}. "
            "Start ./scripts/realsense_depth_streamer.sh first, or remove a stale socket file."
        )

    def _recv_exact(self, n_bytes: int) -> bytes:
        assert self._sock is not None
        buf = b""
        while len(buf) < n_bytes:
            chunk = self._sock.recv(n_bytes - len(buf))
            if not chunk:
                raise ConnectionError("depth socket closed")
            buf += chunk
        return buf

    def _read_once(self) -> np.ndarray:
        header = self._recv_exact(12)
        magic, width, height = struct.unpack("<III", header)
        if magic != self.MAGIC:
            raise ValueError(f"Bad depth socket magic: 0x{magic:08x}")
        data = self._recv_exact(width * height * 4)
        depth = np.frombuffer(data, dtype="<f4").reshape(height, width)
        return depth.astype(np.float32, copy=True)

    def read_depth_m(self) -> np.ndarray:
        try:
            return self._read_once()
        except (ConnectionError, OSError):
            self._connect()
            return self._read_once()

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None


def make_depth_source(args: argparse.Namespace) -> DepthSource:
    if args.depth_source == "mock":
        return MockDepthSource(distance_m=args.depth_max_distance)
    if args.depth_source == "ros1":
        return ROS1DepthSource(args)
    if args.depth_source == "librealsense_socket":
        return LibrealsenseSocketDepthSource(args)
    return RealSenseDepthSource(args)


def apply_depth_orientation(depth: np.ndarray, rotate: int, flip: str) -> np.ndarray:
    if rotate:
        depth = np.rot90(depth, k=rotate // 90)
    if flip in ("horizontal", "both"):
        depth = np.fliplr(depth)
    if flip in ("vertical", "both"):
        depth = np.flipud(depth)
    return np.ascontiguousarray(depth)


@dataclass
class ParkourState:
    args: argparse.Namespace
    device: torch.device
    depth_source: DepthSource
    policy: torch.jit.ScriptModule
    depth_encoder: torch.jit.ScriptModule

    def __post_init__(self) -> None:
        h, w = self.args.depth_resize
        self.history = torch.zeros(
            1, self.args.num_hist, self.args.num_prop, dtype=torch.float32, device=self.device
        )
        self.depth_buffer = torch.zeros(2, h, w, dtype=torch.float32, device=self.device)
        self.dummy_scan = torch.zeros(1, self.args.num_scan, dtype=torch.float32, device=self.device)
        self.priv_explicit = torch.zeros(1, self.args.num_priv_explicit, dtype=torch.float32, device=self.device)
        self.priv_latent = torch.zeros(1, self.args.num_priv_latent, dtype=torch.float32, device=self.device)
        self.depth_latent: torch.Tensor | None = None
        self.depth_yaw = torch.zeros(1, 2, dtype=torch.float32, device=self.device)
        self.prev_action: torch.Tensor | None = None
        self.step = 0

    def reset_temporal_state(self) -> None:
        self.history.zero_()
        self.depth_buffer.zero_()
        self.depth_latent = None
        self.depth_yaw.zero_()
        self.prev_action = None
        self.step = 0

    def preprocess_depth(self) -> torch.Tensor:
        depth_np = self.depth_source.read_depth_m()
        depth_np = apply_depth_orientation(depth_np, self.args.depth_rotate, self.args.depth_flip)
        depth_np = np.nan_to_num(
            depth_np,
            nan=self.args.depth_max_distance,
            posinf=self.args.depth_max_distance,
            neginf=0.0,
        )
        depth_np = np.clip(depth_np, 0.0, self.args.depth_max_distance)
        depth = torch.from_numpy(depth_np).to(dtype=torch.float32, device=self.device)
        depth = depth[:-2, 4:-4]
        depth = F.interpolate(
            depth[None, None],
            size=tuple(self.args.depth_resize),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        return depth / self.args.depth_max_distance - 0.5

    def current_depth_image(self) -> torch.Tensor:
        if self.step % self.args.depth_update_interval == 0:
            processed = self.preprocess_depth()
            self.depth_buffer = torch.cat([self.depth_buffer[1:], processed.unsqueeze(0)], dim=0)
        return self.depth_buffer[-2].unsqueeze(0)

    def build_observation(self, prop: torch.Tensor) -> torch.Tensor:
        if self.depth_latent is None or self.step % self.args.depth_update_interval == 0:
            prop_depth = prop.clone()
            prop_depth[:, 6:8] = 0.0
            depth_latent_and_yaw = self.depth_encoder(self.current_depth_image(), prop_depth)
            self.depth_latent = depth_latent_and_yaw[:, :-2]
            self.depth_yaw = depth_latent_and_yaw[:, -2:]

        prop_policy = prop.clone()
        prop_policy[:, 6:8] = 1.5 * self.depth_yaw
        return torch.cat(
            [
                prop_policy,
                self.dummy_scan,
                self.priv_explicit,
                self.priv_latent,
                self.history.reshape(1, -1),
            ],
            dim=1,
        ).clamp(-100.0, 100.0)

    def update_history(self, prop: torch.Tensor) -> None:
        prop_hist = prop.clone()
        prop_hist[:, 6:8] = 0.0
        if self.step <= 1:
            self.history[:] = prop_hist.unsqueeze(1).repeat(1, self.args.num_hist, 1)
        else:
            self.history = torch.cat([self.history[:, 1:], prop_hist.unsqueeze(1)], dim=1)

    def call_policy(self, obs: torch.Tensor) -> torch.Tensor:
        assert self.depth_latent is not None
        try:
            return self.policy(obs, scandots_latent=self.depth_latent)
        except Exception:
            return self.policy(obs, self.depth_latent)

    def filter_action(self, action: torch.Tensor) -> torch.Tensor:
        action = action.reshape(-1)
        if self.prev_action is None:
            self.prev_action = torch.zeros_like(action)
        alpha = self.args.action_lpf_alpha
        if alpha < 1.0:
            action = alpha * action + (1.0 - alpha) * self.prev_action
        if self.args.action_delta_limit > 0:
            delta = torch.clamp(
                action - self.prev_action,
                -self.args.action_delta_limit,
                self.args.action_delta_limit,
            )
            action = self.prev_action + delta
        if self.args.action_clip > 0:
            action = action.clamp(-self.args.action_clip, self.args.action_clip)
        self.prev_action = action.detach()
        return action

    def infer(self, floats: Iterable[float]) -> tuple[list[float], dict[str, float]]:
        prop = torch.tensor(list(floats), dtype=torch.float32, device=self.device).unsqueeze(0)
        if prop.shape[1] != self.args.num_prop:
            raise ValueError(f"expected {self.args.num_prop} proprio floats, got {prop.shape[1]}")
        with torch.inference_mode():
            obs = self.build_observation(prop)
            action = self.call_policy(obs).squeeze(0)
            action = self.filter_action(action)
            self.update_history(prop)
        self.step += 1
        stats = {
            "command_x": float(prop[0, 10].detach().cpu()),
            "action_min": float(action.min().detach().cpu()),
            "action_max": float(action.max().detach().cpu()),
        }
        return action.detach().cpu().tolist(), stats


class CsvLogger:
    def __init__(self, path: str | None, max_steps: int):
        self._max_steps = max_steps
        self._file = None
        self._writer = None
        if path:
            log_dir = os.path.dirname(path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            self._file = open(path, "w", newline="")
            fieldnames = (
                ["step", "command_x", "action_min", "action_max"]
                + [f"obs_{i:02d}" for i in range(53)]
                + [f"act_{i:02d}" for i in range(12)]
            )
            self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
            self._writer.writeheader()

    def write(self, step: int, obs: tuple[float, ...], action: list[float], stats: dict[str, float]) -> None:
        if self._writer is None or step >= self._max_steps:
            return
        row = {"step": step, **stats}
        row.update({f"obs_{i:02d}": float(obs[i]) for i in range(min(len(obs), 53))})
        row.update({f"act_{i:02d}": float(action[i]) for i in range(min(len(action), 12))})
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def load_torchscript(path: str, device: torch.device) -> torch.jit.ScriptModule:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    module = torch.jit.load(path, map_location=device)
    module.eval()
    return module


def handle_connection(conn: socket.socket, state: ParkourState, logger: CsvLogger) -> None:
    state.reset_temporal_state()
    while True:
        header = recv_exact(conn, 4)
        n_floats = struct.unpack("<I", header)[0]
        data = recv_exact(conn, n_floats * 4)
        floats = struct.unpack(f"<{n_floats}f", data)
        action, stats = state.infer(floats)
        if len(action) != state.args.action_dim:
            raise ValueError(f"expected action_dim={state.args.action_dim}, got {len(action)}")
        logger.write(state.step, floats, action, stats)
        response = struct.pack(f"<I{len(action)}f", len(action), *action)
        conn.sendall(response)
        if state.step % 200 == 0:
            print(
                f"[step {state.step}] command_x={stats['command_x']:.3f} "
                f"action=[{stats['action_min']:.3f}, {stats['action_max']:.3f}]",
                flush=True,
            )


def run_self_test(state: ParkourState, steps: int) -> None:
    obs = [0.0] * state.args.num_prop
    obs[10] = 0.2
    for _ in range(steps):
        action, stats = state.infer(obs)
    print(
        f"Self-test OK: {steps} steps, action_dim={len(action)}, "
        f"action_min={stats['action_min']:.4f}, action_max={stats['action_max']:.4f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if not (0.0 < args.action_lpf_alpha <= 1.0):
        raise ValueError("--action_lpf_alpha must be in (0, 1]")

    policy_path = args.policy or os.path.join(args.asset_dir, "exported_deploy", "policy.pt")
    depth_encoder_path = args.depth_encoder or os.path.join(args.asset_dir, "exported_deploy", "depth_latest.pt")
    device = resolve_device(args.device)

    print("=== Go2 parkour depth inference server ===")
    print(f"Policy:        {policy_path}")
    print(f"Depth encoder: {depth_encoder_path}")
    print(f"Socket:        {args.socket_path}")
    print(f"Depth source:  {args.depth_source}")
    print(f"Device:        {device}")

    policy = load_torchscript(policy_path, device)
    depth_encoder = load_torchscript(depth_encoder_path, device)
    depth_source = make_depth_source(args)
    logger = CsvLogger(args.log_csv, args.log_steps)

    state = ParkourState(args, device, depth_source, policy, depth_encoder)
    if args.self_test_steps > 0:
        run_self_test(state, args.self_test_steps)
        depth_source.close()
        logger.close()
        return

    if os.path.exists(args.socket_path):
        os.unlink(args.socket_path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.socket_path)
    server.listen(1)
    print(f"Ready on {args.socket_path}", flush=True)

    try:
        while True:
            conn, _ = server.accept()
            print("client connected", flush=True)
            try:
                handle_connection(conn, state, logger)
            except ConnectionError:
                print("client disconnected, waiting for reconnect...", flush=True)
            finally:
                conn.close()
    finally:
        logger.close()
        depth_source.close()
        server.close()
        if os.path.exists(args.socket_path):
            os.unlink(args.socket_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted", flush=True)
        sys.exit(130)
