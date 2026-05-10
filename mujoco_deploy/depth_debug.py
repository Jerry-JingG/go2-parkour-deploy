import os
from pathlib import Path

import cv2
import numpy as np


def save_depth_debug_image(
    name: str,
    image,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    dump_dir: str | None = None,
) -> Path | None:
    """Write a depth/debug image as a continuously refreshed PNG."""
    if os.environ.get("MUJOCO_DEPTH_DUMP", "1").lower() in {"0", "false", "off", "no"}:
        return None

    output_dir = Path(dump_dir or os.environ.get("MUJOCO_DEPTH_DUMP_DIR", "/tmp/go2_mujoco_depth"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_depth_viewer_index(output_dir)

    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    output_path = output_dir / f"{safe_name}.png"

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = np.nan_to_num(arr).astype(np.float32)
        if rgb.max(initial=0.0) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        if rgb.shape[-1] == 3:
            cv2.imwrite(str(output_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(output_path), cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGRA))
        return output_path

    depth = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        finite = np.array([0.0], dtype=np.float32)

    lo = float(np.percentile(finite, 1)) if vmin is None else float(vmin)
    hi = float(np.percentile(finite, 99)) if vmax is None else float(vmax)
    if hi <= lo:
        hi = lo + 1e-6

    gray = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    gray = (gray * 255.0).astype(np.uint8)
    color_map = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    color = cv2.applyColorMap(gray, color_map)
    cv2.imwrite(str(output_path), color)

    stats_path = output_dir / f"{safe_name}.txt"
    stats_path.write_text(
        f"shape={tuple(arr.shape)}\n"
        f"min={float(np.min(depth)):.6f}\n"
        f"max={float(np.max(depth)):.6f}\n"
        f"mean={float(np.mean(depth)):.6f}\n"
        f"scale_min={lo:.6f}\n"
        f"scale_max={hi:.6f}\n"
    )
    return output_path


def _ensure_depth_viewer_index(output_dir: Path) -> None:
    index_path = output_dir / "index.html"
    if index_path.exists():
        return

    index_path.write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Go2 MuJoCo Depth Debug</title>
  <style>
    body {
      margin: 0;
      background: #111;
      color: #eee;
      font-family: system-ui, sans-serif;
    }
    main {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 16px;
      padding: 16px;
    }
    section {
      min-width: 0;
    }
    h1 {
      margin: 16px 16px 0;
      font-size: 20px;
      font-weight: 600;
    }
    h2 {
      margin: 0 0 8px;
      font-size: 14px;
      font-weight: 500;
      color: #bbb;
    }
    img {
      width: 100%;
      image-rendering: pixelated;
      background: #222;
      border: 1px solid #333;
    }
    pre {
      min-height: 80px;
      margin: 8px 0 0;
      padding: 8px;
      overflow: auto;
      background: #1b1b1b;
      border: 1px solid #333;
      color: #ccc;
      font-size: 12px;
    }
  </style>
</head>
<body>
  <h1>Go2 MuJoCo Depth Debug</h1>
  <main>
    <section>
      <h2>distance_to_camera.png</h2>
      <img id="distance_img" alt="distance_to_camera">
      <pre id="distance_txt"></pre>
    </section>
    <section>
      <h2>processed_image.png</h2>
      <img id="processed_img" alt="processed_image">
      <pre id="processed_txt"></pre>
    </section>
  </main>
  <script>
    const refreshMs = 300;

    function updateImage(id, path) {
      document.getElementById(id).src = path + "?t=" + Date.now();
    }

    async function updateText(id, path) {
      try {
        const response = await fetch(path + "?t=" + Date.now(), { cache: "no-store" });
        document.getElementById(id).textContent = response.ok ? await response.text() : "waiting...";
      } catch {
        document.getElementById(id).textContent = "waiting...";
      }
    }

    function refresh() {
      updateImage("distance_img", "distance_to_camera.png");
      updateImage("processed_img", "processed_image.png");
      updateText("distance_txt", "distance_to_camera.txt");
      updateText("processed_txt", "processed_image.txt");
    }

    refresh();
    setInterval(refresh, refreshMs);
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
