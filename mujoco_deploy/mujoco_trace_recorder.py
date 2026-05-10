import csv
import os
from pathlib import Path

import cv2
import numpy as np


RAW_FOOT_FORCE_UNITREE_LABELS = ("fr", "fl", "rr", "rl")
RAW_FOOT_FORCE_ISAAC_LABELS = ("fl", "fr", "rl", "rr")
ISAAC_TO_UNITREE_FOOT_ORDER = (1, 0, 3, 2)


CSV_FIELDS = (
    ["step", "command_x", "action_min", "action_max", "foot_force_threshold"]
    + [f"depth_yaw_{i:02d}" for i in range(2)]
    + ["policy_obs_06", "policy_obs_07"]
    + [f"terrain_flag_{i:02d}" for i in range(2)]
    + [f"contact_obs_{i:02d}" for i in range(4)]
    + [f"raw_foot_force_contact_{i:02d}" for i in range(4)]
    + [f"raw_foot_force_unitree_{label}" for label in RAW_FOOT_FORCE_UNITREE_LABELS]
    + [f"raw_foot_force_isaac_{label}" for label in RAW_FOOT_FORCE_ISAAC_LABELS]
    + [f"raw_foot_force_est_unitree_{label}" for label in RAW_FOOT_FORCE_UNITREE_LABELS]
    + [f"raw_foot_force_est_isaac_{label}" for label in RAW_FOOT_FORCE_ISAAC_LABELS]
    + [
        "depth_capture_path",
        "depth_policy_input_path",
        "depth_capture_vis_path",
        "depth_policy_input_vis_path",
    ]
    + [f"obs_{i:02d}" for i in range(53)]
    + [f"act_{i:02d}" for i in range(12)]
)


class MujocoTraceRecorder:
    def __init__(
        self,
        csv_path: str,
        depth_dir: str | None = None,
        depth_every: int = 5,
        foot_force_threshold: float = 2.0,
        depth_max_distance: float = 2.0,
    ):
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.depth_dir = Path(depth_dir).expanduser().resolve() if depth_dir else self._default_depth_dir()
        self.depth_every = max(1, int(depth_every))
        self.foot_force_threshold = float(foot_force_threshold)
        self.depth_max_distance = float(depth_max_distance)

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        for name in ("capture", "policy_input", "capture_vis", "policy_input_vis"):
            (self.depth_dir / name).mkdir(parents=True, exist_ok=True)

        self._file = self.csv_path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._file.flush()

        print(f"[INFO] Recording MuJoCo obs/action CSV: {self.csv_path}")
        print(f"[INFO] Recording MuJoCo depth trace: {self.depth_dir}")

    def _default_depth_dir(self) -> Path:
        stem = self.csv_path.stem
        if stem.startswith("mujoco_obs_action_"):
            stem = "mujoco_depth_" + stem[len("mujoco_obs_action_"):]
        else:
            stem = stem + "_depth"
        return self.csv_path.parent / stem

    def write(
        self,
        *,
        policy_step_index: int,
        proprio_obs,
        policy_action,
        depth_yaw,
        policy_obs_6_8,
        foot_force_isaac=None,
        depth_trace=None,
    ) -> None:
        obs_np = _to_numpy(proprio_obs).reshape(-1)
        action_np = _to_numpy(policy_action).reshape(-1)
        depth_yaw_np = _to_numpy(depth_yaw).reshape(-1)
        policy_obs_6_8_np = _to_numpy(policy_obs_6_8).reshape(-1)

        row = {field: "" for field in CSV_FIELDS}
        row["step"] = int(policy_step_index) + 1
        row["command_x"] = _value_or_nan(obs_np, 10)
        row["action_min"] = float(np.nanmin(action_np)) if action_np.size else float("nan")
        row["action_max"] = float(np.nanmax(action_np)) if action_np.size else float("nan")
        row["foot_force_threshold"] = self.foot_force_threshold
        row["depth_yaw_00"] = _value_or_nan(depth_yaw_np, 0)
        row["depth_yaw_01"] = _value_or_nan(depth_yaw_np, 1)
        row["policy_obs_06"] = _value_or_nan(policy_obs_6_8_np, 0)
        row["policy_obs_07"] = _value_or_nan(policy_obs_6_8_np, 1)

        for i in range(2):
            row[f"terrain_flag_{i:02d}"] = _value_or_nan(obs_np, 11 + i)
        for i in range(4):
            row[f"contact_obs_{i:02d}"] = _value_or_nan(obs_np, 49 + i)
        for i in range(53):
            row[f"obs_{i:02d}"] = _value_or_nan(obs_np, i)
        for i in range(12):
            row[f"act_{i:02d}"] = _value_or_nan(action_np, i)

        foot_isaac = _normalize_foot_forces(foot_force_isaac)
        foot_unitree = [foot_isaac[idx] for idx in ISAAC_TO_UNITREE_FOOT_ORDER]
        foot_contact = [
            (1.0 if value > self.foot_force_threshold else 0.0) - 0.5
            if np.isfinite(value)
            else float("nan")
            for value in foot_isaac
        ]
        for i, value in enumerate(foot_contact):
            row[f"raw_foot_force_contact_{i:02d}"] = value
        for label, value in zip(RAW_FOOT_FORCE_UNITREE_LABELS, foot_unitree):
            row[f"raw_foot_force_unitree_{label}"] = value
            row[f"raw_foot_force_est_unitree_{label}"] = value
        for label, value in zip(RAW_FOOT_FORCE_ISAAC_LABELS, foot_isaac):
            row[f"raw_foot_force_isaac_{label}"] = value
            row[f"raw_foot_force_est_isaac_{label}"] = value

        if depth_trace is not None and policy_step_index % self.depth_every == 0:
            paths = self._write_depth(policy_step_index, depth_trace)
            row.update(paths)

        self._writer.writerow(row)
        self._file.flush()

    def _write_depth(self, policy_step_index: int, depth_trace: dict) -> dict[str, str]:
        step_name = f"step_{policy_step_index:06d}"
        capture_path = self.depth_dir / "capture" / f"{step_name}.npz"
        policy_input_path = self.depth_dir / "policy_input" / f"{step_name}.npz"
        capture_vis_path = self.depth_dir / "capture_vis" / f"{step_name}.png"
        policy_input_vis_path = self.depth_dir / "policy_input_vis" / f"{step_name}.png"

        raw_depth = _squeeze_image(_to_numpy(depth_trace.get("raw_depth_m"))).astype(np.float32)
        processed_capture = _squeeze_image(
            _to_numpy(depth_trace.get("processed_capture_depth"))
        ).astype(np.float32)
        policy_input = _squeeze_image(_to_numpy(depth_trace.get("policy_depth_input"))).astype(np.float32)

        np.savez(capture_path, raw_depth_m=raw_depth, processed_capture_depth=processed_capture)
        np.savez(policy_input_path, policy_depth_input=policy_input)
        _write_depth_vis(capture_vis_path, raw_depth, 0.0, self.depth_max_distance)
        _write_depth_vis(policy_input_vis_path, policy_input, -0.5, 0.5)

        return {
            "depth_capture_path": str(capture_path),
            "depth_policy_input_path": str(policy_input_path),
            "depth_capture_vis_path": str(capture_vis_path),
            "depth_policy_input_vis_path": str(policy_input_vis_path),
        }

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


def _to_numpy(value) -> np.ndarray:
    if value is None:
        return np.array([], dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _value_or_nan(values: np.ndarray, index: int) -> float:
    if index >= values.size:
        return float("nan")
    return float(values[index])


def _normalize_foot_forces(value) -> list[float]:
    if value is None:
        return [float("nan")] * 4
    values = _to_numpy(value).reshape(-1)
    if values.size < 4:
        return [float("nan")] * 4
    return [float(values[i]) for i in range(4)]


def _squeeze_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def _write_depth_vis(path: Path, image: np.ndarray, vmin: float, vmax: float) -> None:
    arr = _squeeze_image(image)
    arr = np.nan_to_num(arr.astype(np.float32), nan=vmax, posinf=vmax, neginf=vmin)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    gray = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    gray = (gray * 255.0).astype(np.uint8)
    color_map = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    cv2.imwrite(str(path), cv2.applyColorMap(gray, color_map))
