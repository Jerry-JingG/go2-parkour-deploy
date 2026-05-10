#!/usr/bin/env python3
"""Verify Go2 real-deploy traces against the IsaacLab parkour deploy inference chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a Go2 parkour real-deploy trace bundle")
    parser.add_argument("trace", type=str, help="Trace tar.gz or extracted trace directory")
    parser.add_argument("--asset_dir", type=str, default="real_deploy/policies/parkour_depth")
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--depth_encoder", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--report_dir", type=str, default=None)
    parser.add_argument("--socket_tol", type=float, default=1e-6)
    parser.add_argument("--proprio_tol", type=float, default=1e-5)
    parser.add_argument("--tensor_tol", type=float, default=1e-4)
    parser.add_argument("--action_tol", type=float, default=1e-4)
    parser.add_argument("--processed_action_tol", type=float, default=1e-5)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_or_find_trace(trace_path: Path, tmp_dir: Path) -> Path:
    if trace_path.is_dir():
        return trace_path
    if not tarfile.is_tarfile(trace_path):
        raise ValueError(f"Trace is neither a directory nor a tar archive: {trace_path}")
    with tarfile.open(trace_path, "r:*") as archive:
        archive.extractall(tmp_dir)
    matches = list(tmp_dir.rglob("server_trace.npz"))
    if not matches:
        raise FileNotFoundError("server_trace.npz not found in trace archive")
    return matches[0].parent.parent


def find_trace_files(root: Path) -> tuple[Path, Path, Path, Path | None]:
    server_npz = next(root.rglob("server_trace.npz"), None)
    server_meta = next(root.rglob("server_trace_meta.json"), None)
    cpp_jsonl = next(root.rglob("cpp_proprio.jsonl"), None)
    cpp_meta = next(root.rglob("cpp_trace_meta.json"), None)
    missing = [
        name
        for name, path in (
            ("server_trace.npz", server_npz),
            ("server_trace_meta.json", server_meta),
            ("cpp_proprio.jsonl", cpp_jsonl),
        )
        if path is None
    ]
    if missing:
        raise FileNotFoundError(f"Missing trace files: {', '.join(missing)}")
    assert server_npz is not None and server_meta is not None and cpp_jsonl is not None
    return server_npz, server_meta, cpp_jsonl, cpp_meta


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def metric(value: float, tolerance: float) -> dict[str, Any]:
    return {
        "max_abs": value,
        "tolerance": tolerance,
        "passed": bool(np.isfinite(value) and value <= tolerance),
    }


def wrap_to_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def euler_xyz_from_quat_wxyz(quat: np.ndarray) -> tuple[float, float, float]:
    q = quat.astype(np.float64)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("Zero quaternion in trace")
    w, x, y, z = q / norm
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + w * z)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(r21, r22)
    pitch = math.asin(max(-1.0, min(1.0, -r20)))
    yaw = math.atan2(r10, r00)
    return wrap_to_pi(roll), wrap_to_pi(pitch), wrap_to_pi(yaw)


def select_cpp_obs(row: dict[str, Any]) -> np.ndarray:
    groups = row["obs_groups"]
    if "obs" in groups:
        return np.asarray(groups["obs"], dtype=np.float32)
    if len(groups) == 1:
        return np.asarray(next(iter(groups.values())), dtype=np.float32)
    raise KeyError(f"C++ trace row has no obs group and multiple groups: {list(groups)}")


def recompute_parkour_proprio(row: dict[str, Any], default_joint_pos: np.ndarray, foot_threshold: float) -> np.ndarray:
    root_ang_vel_b = np.asarray(row["root_ang_vel_b"], dtype=np.float32)
    quat = np.asarray(row["root_quat_w"], dtype=np.float32)
    joint_pos = np.asarray(row["joint_pos"], dtype=np.float32)
    joint_vel = np.asarray(row["joint_vel"], dtype=np.float32)
    previous_action = np.asarray(row["previous_action"], dtype=np.float32)
    foot_force = np.asarray(row["foot_force_est"], dtype=np.float32)
    roll, pitch, yaw = euler_xyz_from_quat_wxyz(quat)

    obs: list[float] = []
    obs.extend((root_ang_vel_b * 0.25).tolist())
    obs.extend([roll, pitch])
    obs.extend([0.0, yaw, yaw])
    obs.extend([0.0, 0.0])
    obs.append(float(row["parkour_command_x"]))
    obs.extend([1.0, 0.0])
    obs.extend((joint_pos - default_joint_pos).tolist())
    obs.extend((joint_vel * 0.05).tolist())
    obs.extend(previous_action.tolist())
    for idx in (1, 0, 3, 2):
        obs.append((1.0 if foot_force[idx] > foot_threshold else 0.0) - 0.5)
    return np.asarray(obs, dtype=np.float32)


def load_deploy_cfg(root: Path) -> dict[str, Any]:
    candidates = list(root.rglob("deploy.yaml"))
    if not candidates or yaml is None:
        return {}
    with candidates[0].open("r", encoding="utf-8") as infile:
        loaded = yaml.safe_load(infile)
    return loaded or {}


def cfg_vector(value: Any, length: int, fallback: float) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    return np.full(length, fallback, dtype=np.float32)


def call_policy(policy: torch.jit.ScriptModule, obs: torch.Tensor, depth_latent: torch.Tensor) -> torch.Tensor:
    try:
        return policy(obs, scandots_latent=depth_latent)
    except Exception:
        return policy(obs, depth_latent)


def replay_action_filter(raw_actions: np.ndarray, alpha: float, delta_limit: float, action_clip: float) -> np.ndarray:
    outputs = np.zeros_like(raw_actions, dtype=np.float32)
    prev = np.zeros(raw_actions.shape[1], dtype=np.float32)
    for i, raw in enumerate(raw_actions.astype(np.float32)):
        action = raw.copy()
        if alpha < 1.0:
            action = alpha * action + (1.0 - alpha) * prev
        if delta_limit > 0.0:
            delta = np.clip(action - prev, -delta_limit, delta_limit)
            action = prev + delta
        if action_clip > 0.0:
            action = np.clip(action, -action_clip, action_clip)
        outputs[i] = action
        prev = action.copy()
    return outputs


def replay_torch(
    data: dict[str, np.ndarray],
    policy_path: Path,
    depth_path: Path,
    device_arg: str,
) -> dict[str, float]:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    policy = torch.jit.load(str(policy_path), map_location=device)
    depth_encoder = torch.jit.load(str(depth_path), map_location=device)
    policy.eval()
    depth_encoder.eval()

    action_replay: list[np.ndarray] = []
    obs_after_replay: list[np.ndarray] = []
    depth_out_replay: list[np.ndarray] = []
    depth_out_recorded: list[np.ndarray] = []

    with torch.inference_mode():
        for i in range(len(data["step"])):
            if bool(data["depth_encoder_ran"][i]):
                depth_image = torch.from_numpy(data["depth_image_used"][i]).to(device=device, dtype=torch.float32).unsqueeze(0)
                prop_depth = torch.from_numpy(data["prop_socket"][i]).to(device=device, dtype=torch.float32).unsqueeze(0)
                prop_depth[:, 6:8] = 0.0
                depth_out = depth_encoder(depth_image, prop_depth)
                depth_out_replay.append(depth_out.detach().cpu().numpy().squeeze(0))
                depth_out_recorded.append(
                    np.concatenate([data["depth_latent"][i], data["depth_yaw"][i]]).astype(np.float32)
                )

            obs = torch.from_numpy(data["policy_obs_before"][i]).to(device=device, dtype=torch.float32).unsqueeze(0)
            latent = torch.from_numpy(data["depth_latent"][i]).to(device=device, dtype=torch.float32).unsqueeze(0)
            action = call_policy(policy, obs, latent)
            action_replay.append(action.detach().cpu().numpy().squeeze(0))
            obs_after_replay.append(obs.detach().cpu().numpy().squeeze(0))

    metrics: dict[str, float] = {
        "policy_action_replay": max_abs(np.stack(action_replay), data["action_raw"]),
        "policy_obs_after_replay": max_abs(np.stack(obs_after_replay), data["policy_obs_after"]),
    }
    if depth_out_replay:
        metrics["depth_encoder_replay"] = max_abs(np.stack(depth_out_replay), np.stack(depth_out_recorded))
    else:
        metrics["depth_encoder_replay"] = float("nan")
    return metrics


def write_reports(report_dir: Path, report: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "verification_report.json").open("w", encoding="utf-8") as outfile:
        json.dump(report, outfile, indent=2, sort_keys=True)

    lines = [
        "# Go2 Parkour Trace Verification",
        "",
        f"Overall: {'PASS' if report['passed'] else 'FAIL'}",
        "",
        "| Check | Max Abs | Tolerance | Result |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, item in report["metrics"].items():
        value = item["max_abs"]
        value_text = "nan" if not np.isfinite(value) else f"{value:.6g}"
        lines.append(
            f"| {name} | {value_text} | {item['tolerance']:.6g} | {'PASS' if item['passed'] else 'FAIL'} |"
        )
    if report["issues"]:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in report["issues"])
    (report_dir / "verification_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    trace_path = Path(args.trace).resolve()
    asset_dir = Path(args.asset_dir).resolve()
    policy_path = Path(args.policy).resolve() if args.policy else asset_dir / "exported_deploy" / "policy.pt"
    depth_path = (
        Path(args.depth_encoder).resolve() if args.depth_encoder else asset_dir / "exported_deploy" / "depth_latest.pt"
    )

    if not policy_path.is_file():
        raise FileNotFoundError(policy_path)
    if not depth_path.is_file():
        raise FileNotFoundError(depth_path)

    default_report_dir = trace_path.with_suffix("")
    if trace_path.name.endswith(".tar.gz"):
        default_report_dir = trace_path.with_name(trace_path.name[:-7] + "_verification")
    report_dir = Path(args.report_dir).resolve() if args.report_dir else default_report_dir

    with tempfile.TemporaryDirectory(prefix="go2_parkour_trace_") as tmp:
        root = extract_or_find_trace(trace_path, Path(tmp))
        server_npz_path, server_meta_path, cpp_jsonl_path, cpp_meta_path = find_trace_files(root)
        server_meta = load_json(server_meta_path)
        cpp_meta = load_json(cpp_meta_path) if cpp_meta_path else {}
        cpp_rows = load_jsonl(cpp_jsonl_path)
        data = {key: value for key, value in np.load(server_npz_path).items()}

        issues: list[str] = []
        metrics: dict[str, dict[str, Any]] = {}

        if len(cpp_rows) == 0:
            raise ValueError("C++ trace has zero rows")
        if len(data.get("step", [])) == 0:
            raise ValueError("Server trace has zero rows")

        server_policy_hash = server_meta.get("policy_sha256")
        server_depth_hash = server_meta.get("depth_encoder_sha256")
        local_policy_hash = sha256_file(policy_path)
        local_depth_hash = sha256_file(depth_path)
        if server_policy_hash and server_policy_hash != local_policy_hash:
            issues.append("Local policy.pt SHA256 differs from trace metadata")
        if server_depth_hash and server_depth_hash != local_depth_hash:
            issues.append("Local depth_latest.pt SHA256 differs from trace metadata")

        cpp_by_step: dict[int, dict[str, Any]] = {}
        for row in cpp_rows:
            cpp_by_step.setdefault(int(row["step"]), row)
        aligned_cpp: list[dict[str, Any]] = []
        aligned_server_indices: list[int] = []
        for i, step in enumerate(data["step"].astype(np.int64).tolist()):
            row = cpp_by_step.get(int(step))
            if row is not None:
                aligned_cpp.append(row)
                aligned_server_indices.append(i)
        if not aligned_cpp:
            raise ValueError("No overlapping C++ and server trace steps")

        idx = np.asarray(aligned_server_indices, dtype=np.int64)
        server_prop = data["prop_socket"][idx]
        cpp_obs = np.stack([select_cpp_obs(row) for row in aligned_cpp])
        metrics["cpp_obs_vs_server_prop"] = metric(max_abs(cpp_obs, server_prop), args.socket_tol)

        default_joint_pos = np.asarray(cpp_meta.get("default_joint_pos", aligned_cpp[0].get("default_joint_pos", [])), dtype=np.float32)
        if default_joint_pos.size == 0:
            default_joint_pos = np.asarray(
                [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5],
                dtype=np.float32,
            )

        deploy_cfg = load_deploy_cfg(root)
        parkour_cfg = deploy_cfg.get("parkour", {}) if isinstance(deploy_cfg, dict) else {}
        foot_threshold = float(parkour_cfg.get("foot_force_threshold", 2.0))
        recomputed = np.stack([recompute_parkour_proprio(row, default_joint_pos, foot_threshold) for row in aligned_cpp])
        metrics["recomputed_cpp_proprio"] = metric(max_abs(recomputed, cpp_obs), args.proprio_tol)

        action_filtered = data["action_filtered"][idx]
        cpp_action_raw = np.stack([np.asarray(row["action_raw"], dtype=np.float32) for row in aligned_cpp])
        metrics["cpp_action_vs_server_filtered"] = metric(max_abs(cpp_action_raw, action_filtered), args.action_tol)

        action_cfg = deploy_cfg.get("actions", {}).get("JointPositionAction", {}) if isinstance(deploy_cfg, dict) else {}
        scale = cfg_vector(action_cfg.get("scale"), action_filtered.shape[1], 0.25)
        offset = cfg_vector(action_cfg.get("offset"), action_filtered.shape[1], 0.0)
        if not np.any(offset):
            offset = default_joint_pos
        expected_processed = action_filtered * scale + offset
        cpp_processed = np.stack([np.asarray(row["processed_action"], dtype=np.float32) for row in aligned_cpp])
        metrics["cpp_processed_action"] = metric(max_abs(cpp_processed, expected_processed), args.processed_action_tol)

        server_args = server_meta.get("args", {})
        alpha = float(server_args.get("action_lpf_alpha", 0.5))
        delta_limit = float(server_args.get("action_delta_limit", 0.25))
        action_clip = float(server_args.get("action_clip", 4.8))
        filtered_replay = replay_action_filter(data["action_raw"], alpha, delta_limit, action_clip)
        metrics["action_filter_replay"] = metric(max_abs(filtered_replay, data["action_filtered"]), args.action_tol)

        if int(server_args.get("trace_every", 1)) != 1:
            issues.append("Torch replay is exact only when server trace_every is 1")
        torch_metrics = replay_torch(data, policy_path, depth_path, args.device)
        metrics["depth_encoder_replay"] = metric(torch_metrics["depth_encoder_replay"], args.tensor_tol)
        metrics["policy_action_replay"] = metric(torch_metrics["policy_action_replay"], args.action_tol)
        metrics["policy_obs_after_replay"] = metric(torch_metrics["policy_obs_after_replay"], args.tensor_tol)

        for name, item in metrics.items():
            if not item["passed"]:
                issues.append(f"{name} exceeded tolerance: {item['max_abs']} > {item['tolerance']}")

        report = {
            "passed": not issues,
            "trace_root": str(root),
            "server_trace": str(server_npz_path),
            "cpp_trace": str(cpp_jsonl_path),
            "aligned_steps": len(aligned_cpp),
            "server_rows": int(len(data["step"])),
            "cpp_rows": int(len(cpp_rows)),
            "policy_path": str(policy_path),
            "depth_encoder_path": str(depth_path),
            "policy_sha256": local_policy_hash,
            "depth_encoder_sha256": local_depth_hash,
            "metrics": metrics,
            "issues": issues,
        }
        write_reports(report_dir, report)

    print(f"Verification {'PASS' if report['passed'] else 'FAIL'}")
    print(f"Report: {report_dir / 'verification_summary.md'}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
