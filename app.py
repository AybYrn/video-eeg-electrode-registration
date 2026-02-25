"""
Gradio PoC UI for Video EEG Electrode Registration

Implemented:
- Centered title with color #FF4400
- Intro text inside a read-only textbox
- Video uploader area kept stable (doesn't grow with video resolution)
- Primary buttons share the same styling (Save/Refresh/Refresh report list)
- Delete button has its own styling
- Reports tab:
  - No "Showing: ..." status textbox
  - PNGs displayed as a thumbnail grid (Gallery)
  - Clicking a thumbnail updates big preview + download
"""

import time
import shutil
from pathlib import Path
import glob

import gradio as gr

import json
import pickle
import numpy as np
from datetime import datetime

import plotly.graph_objects as go

# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "data" / "raw" / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_VALIDATION_DIR = BASE_DIR / "reports" / "validation"

# Codespaces workspace mount
WORKSPACES_MOUNT = "/workspaces"

# Copy in chunks for smooth progress updates
CHUNK_SIZE = 8 * 1024 * 1024  # 8MB
LARGE_FILE_WARN_GB = 1.0

INTRO_TEXT = (
    "This PoC UI is designed for GitHub Codespaces:\n"
    "• Upload a video and save it into data/raw/videos/\n"
    "• See copy progress, file size, free disk space, and an ETA\n"
    "• Manage saved videos (list & delete) to free disk space\n"
    "• Browse validation PNGs under reports/validation/"
)

CSS = r"""
/* ---------- Title ---------- */
#poc-title {
  text-align: center;
  color: #FF4400;
  font-weight: 800;
  font-size: 40px;
  margin: 10px 0 6px 0;
}

/* ---------- Video area: stable without clipping controls ---------- */
#video-wrap {
  max-height: 520px;
  border-radius: 10px;
  overflow: visible !important;
}
#video-wrap video {
  width: 100% !important;
  max-height: 420px !important;
  object-fit: contain;
  background: #111;
}

/* ---------- Primary buttons: unified style (non-delete) ---------- */
button#save-btn,
button#refresh-btn,
button#refresh-reports-btn {
  background: #FFB33F !important;
  color: #111 !important;
  border: 2px solid transparent !important;
  font-weight: 800 !important;
  border-radius: 10px !important;
}
button#save-btn:hover,
button#refresh-btn:hover,
button#refresh-reports-btn:hover {
  border: 2px solid #134E8E !important;
}

/* ---------- Delete button: special style ---------- */
button#delete-btn {
  background: #C00707 !important;
  color: white !important;
  border: 2px solid transparent !important;
  font-weight: 800 !important;
  border-radius: 10px !important;
}
button#delete-btn:hover {
  border: 2px solid #FFB33F !important;
}

/* ---------- Gallery tweaks (optional) ---------- */
#reports-gallery {
  padding-top: 6px;
}
"""

# -------------------------
# Helpers
# -------------------------
def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{n} B"

def _disk_free_bytes(path: str) -> int:
    return shutil.disk_usage(path).free

def _disk_status():
    """Return a short human-readable disk status for /workspaces."""
    free = _disk_free_bytes(WORKSPACES_MOUNT)
    total = shutil.disk_usage(WORKSPACES_MOUNT).total
    used = total - free
    return f"Disk ({WORKSPACES_MOUNT}) used: {_human_bytes(used)} / {_human_bytes(total)} | free: {_human_bytes(free)}"

def cleanup_demo_files(delete_videos: bool = False):
    """
    Delete demo folders: results/demo_*
    Optionally delete videos under data/raw/videos/*
    Returns a status string.
    """
    removed_folders = 0
    removed_files = 0
    removed_bytes = 0

    # 1) Remove results/demo_*
    results_dir = BASE_DIR / "results"
    if results_dir.exists():
        for p in results_dir.iterdir():
            if p.is_dir() and p.name.startswith("demo_"):
                # sum folder size (best-effort)
                for fp in p.rglob("*"):
                    if fp.is_file():
                        try:
                            removed_bytes += fp.stat().st_size
                        except Exception:
                            pass
                shutil.rmtree(p, ignore_errors=True)
                removed_folders += 1

    # 2) Optionally remove videos
    if delete_videos and VIDEO_DIR.exists():
        for fp in VIDEO_DIR.iterdir():
            if fp.is_file():
                try:
                    removed_bytes += fp.stat().st_size
                except Exception:
                    pass
                try:
                    fp.unlink()
                    removed_files += 1
                except Exception:
                    pass

    msg = (
        f"✅ Cleanup completed.\n"
        f"- Removed demo folders: {removed_folders}\n"
        f"- Removed video files: {removed_files}\n"
        f"- Freed (approx): {_human_bytes(removed_bytes)}\n"
        f"{_disk_status()}"
    )

    return msg

def _list_saved_videos() -> list[str]:
    exts = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}
    files = [p.name for p in VIDEO_DIR.iterdir() if p.is_file() and p.suffix in exts]
    files.sort()
    return files

def refresh_saved_videos():
    files = _list_saved_videos()
    if not files:
        return gr.update(choices=[], value=None), "ℹ️ No saved videos found in data/raw/videos."
    return gr.update(choices=files, value=files[-1]), f"✅ Found {len(files)} saved video(s)."

def electrodes_json_to_figure(json_path: str):
    """
    Build a Plotly 3D scatter from electrodes_3d.json.
    Shows landmarks (NAS/LPA/RPA/INION) and electrodes.
    """
    if not json_path:
        return go.Figure()

    p = Path(json_path)
    if not p.exists():
        return go.Figure()

    with open(p, "r") as f:
        data = json.load(f)

    landmarks = data.get("landmarks", {})
    electrodes = data.get("electrodes", {})

    def _pts_from_dict(d):
        xs, ys, zs, labels = [], [], [], []
        for name, item in d.items():
            pos = item.get("position", None)
            if not pos or len(pos) != 3:
                continue
            xs.append(pos[0]); ys.append(pos[1]); zs.append(pos[2])
            labels.append(name)
        return xs, ys, zs, labels

    lx, ly, lz, llabels = _pts_from_dict(landmarks)
    ex, ey, ez, elabels = _pts_from_dict(electrodes)

    fig = go.Figure()

    # Electrodes
    if ex:
        fig.add_trace(
            go.Scatter3d(
                x=ex, y=ey, z=ez,
                mode="markers",
                name="Electrodes",
                text=elabels,
                hovertemplate="<b>%{text}</b><br>x=%{x:.2f} mm<br>y=%{y:.2f} mm<br>z=%{z:.2f} mm<extra></extra>",
                marker=dict(size=4),
            )
        )

    # Landmarks
    if lx:
        fig.add_trace(
            go.Scatter3d(
                x=lx, y=ly, z=lz,
                mode="markers+text",
                name="Landmarks",
                text=llabels,
                textposition="top center",
                hovertemplate="<b>%{text}</b><br>x=%{x:.2f} mm<br>y=%{y:.2f} mm<br>z=%{z:.2f} mm<extra></extra>",
                marker=dict(size=7, symbol="diamond"),
            )
        )

    # ---------- Head wireframe / silhouette (PoC) ----------
    # We draw a transparent sphere-ish wireframe around the points.
    # Compute a center & radius from landmarks/electrodes so it adapts to scale.
    all_x = (lx + ex) if (lx or ex) else []
    all_y = (ly + ey) if (ly or ey) else []
    all_z = (lz + ez) if (lz or ez) else []

    if all_x:
        cx = float(np.mean(all_x))
        cy = float(np.mean(all_y))
        cz = float(np.mean(all_z))

        # radius: use max distance from center, then inflate a bit
        dists = np.sqrt((np.array(all_x) - cx) ** 2 + (np.array(all_y) - cy) ** 2 + (np.array(all_z) - cz) ** 2)
        r = float(np.max(dists)) * 1.15

        # Build a coarse sphere wireframe (few rings) to keep it lightweight
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0.15 * np.pi, 0.95 * np.pi, 20)  # avoid poles for nicer look
        uu, vv = np.meshgrid(u, v)

        xs = cx + r * np.cos(uu) * np.sin(vv)
        ys = cy + r * np.sin(uu) * np.sin(vv)
        zs = cz + r * np.cos(vv)

        fig.add_trace(
            go.Surface(
                x=xs,
                y=ys,
                z=zs,
                opacity=0.08,          # very transparent
                showscale=False,
                name="Head",
                hoverinfo="skip",
            )
        )

        # Optional: "ears" as small spheres (simple markers) at LPA/RPA if present
        if "LPA" in landmarks and "RPA" in landmarks:
            lpa = landmarks["LPA"]["position"]
            rpa = landmarks["RPA"]["position"]
            fig.add_trace(
                go.Scatter3d(
                    x=[lpa[0], rpa[0]],
                    y=[lpa[1], rpa[1]],
                    z=[lpa[2], rpa[2]],
                    mode="markers",
                    name="Ears (LPA/RPA)",
                    hoverinfo="skip",
                    marker=dict(size=6, symbol="circle"),
                )
            )

    fig.update_layout(
        title="3D Electrode Positions (mm)",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
    )

    return fig

# -------------------------
# Video: save/delete
# -------------------------
def save_video(video, progress=gr.Progress()):
    """
    Save uploaded video into data/raw/videos with progress.
    Progress tracks server-side copy, not browser upload.
    """
    if video is None:
        return gr.update(), "❌ No video uploaded."

    src = Path(video)
    if not src.exists():
        return gr.update(), f"❌ File not found: {src}"

    total = src.stat().st_size
    total_h = _human_bytes(total)

    free = _disk_free_bytes(WORKSPACES_MOUNT)
    free_h = _human_bytes(free)

    warn = ""
    if total >= LARGE_FILE_WARN_GB * (1024**3):
        warn = f"\n⚠️ Large file warning: {total_h}. Codespaces disk can fill up quickly."

    # Keep a small safety buffer (200MB)
    if free < total + 200 * 1024 * 1024:
        return gr.update(), (
            f"❌ Not enough free space on {WORKSPACES_MOUNT}.\n"
            f"Free: {free_h}\n"
            f"File: {total_h}\n"
            f"Tip: delete old videos using the dropdown below."
        )

    # Avoid overwriting
    dst = VIDEO_DIR / src.name
    if dst.exists():
        stem, suf = dst.stem, dst.suffix
        i = 1
        while True:
            cand = VIDEO_DIR / f"{stem}_{i}{suf}"
            if not cand.exists():
                dst = cand
                break
            i += 1

    copied = 0
    start = time.time()
    progress(0, desc="Preparing copy...")

    with src.open("rb") as fsrc, dst.open("wb") as fdst:
        while True:
            buf = fsrc.read(CHUNK_SIZE)
            if not buf:
                break
            fdst.write(buf)
            copied += len(buf)

            elapsed = max(time.time() - start, 1e-6)
            rate = copied / elapsed
            remaining = max(total - copied, 0)
            eta = remaining / rate if rate > 0 else 0.0

            progress(
                min(copied / total, 1.0),
                desc=f"Copying... {_human_bytes(copied)} / {total_h} | ETA ~ {eta:.1f}s",
            )

    progress(1, desc="Done ✅")

    files = _list_saved_videos()
    dd_update = gr.update(choices=files, value=(dst.name if dst.name in files else (files[-1] if files else None)))

    status = (
        f"✅ Saved to: {dst.relative_to(BASE_DIR)}\n"
        f"File size: {total_h}\n"
        f"Free space ({WORKSPACES_MOUNT}): {free_h}"
        f"{warn}"
    )
    return dd_update, status

def delete_selected_video(filename: str):
    """Delete the selected video from data/raw/videos."""
    if not filename:
        return gr.update(), "ℹ️ No video selected."

    target = VIDEO_DIR / filename
    if not target.exists():
        return gr.update(), f"❌ File not found: {target}"

    target.unlink()

    files = _list_saved_videos()
    dd_update = gr.update(choices=files, value=(files[-1] if files else None))

    free = _disk_free_bytes(WORKSPACES_MOUNT)
    status = f"🗑️ Deleted: {filename}\nFree space ({WORKSPACES_MOUNT}): {_human_bytes(free)}"
    return dd_update, status

# -------------------------
# Reports (PNGs): list + gallery + preview
# -------------------------
def _validation_png_paths() -> list[Path]:
    if not REPORTS_VALIDATION_DIR.exists():
        return []
    paths = sorted(Path(p) for p in glob.glob(str(REPORTS_VALIDATION_DIR / "*.png")))
    return paths

def list_validation_pngs():
    """
    Returns:
    - dropdown update (choices = names)
    - gallery items (list of (path, caption))
    """
    paths = _validation_png_paths()
    names = [p.name for p in paths]

    dd = gr.update(choices=names, value=(names[0] if names else None))

    # Gallery expects items like: [(image, caption), ...]
    gallery_items = [(str(p), p.name) for p in paths]
    return dd, gallery_items

def preview_by_filename(filename: str):
    """Return only the file path for download."""
    if not filename:
        return None
    fpath = REPORTS_VALIDATION_DIR / filename
    if not fpath.exists():
        return None
    return str(fpath)

def preview_by_gallery_select(evt: gr.SelectData):
    paths = _validation_png_paths()
    if not paths:
        return gr.update()
    idx = evt.index if isinstance(evt.index, int) else 0
    idx = max(0, min(idx, len(paths) - 1))
    fpath = paths[idx]
    return gr.update(value=fpath.name)

# -------------------------
# Script2 constants (matches scripts/Script2.py)
# -------------------------
LANDMARK_NAS = 0
LANDMARK_LPA = 1
LANDMARK_RPA = 2
ELECTRODE_START_ID = 100

ARC_TO_CHORD = 0.92
CIRCUMFERENCE_TO_EAR = 0.26

def _estimate_inion_3d(nas, lpa, rpa):
    """Estimate INION from NAS, LPA, RPA using the same logic as Script2."""
    origin = (lpa + rpa) / 2.0
    ear_axis = rpa - lpa
    ear_len = np.linalg.norm(ear_axis)
    if ear_len < 1e-6:
        return None
    ear_axis = ear_axis / ear_len

    nas_vec = nas - origin
    forward = nas_vec - np.dot(nas_vec, ear_axis) * ear_axis
    forward_len = np.linalg.norm(forward)
    if forward_len < 1e-6:
        return None
    forward = forward / forward_len

    up = np.cross(ear_axis, forward)
    up = up / np.linalg.norm(up)

    inion_distance = forward_len * 1.05
    z_offset = -0.08 * ear_len
    inion = origin - forward * inion_distance + up * z_offset
    return inion

def _build_head_transform(nas, lpa, rpa, measured_mm):
    """Build head coordinate system + scale (same as Script2)."""
    inion = _estimate_inion_3d(nas, lpa, rpa)
    if inion is None:
        return None

    origin = (lpa + rpa) / 2.0

    x_axis = rpa - lpa
    raw_ear_dist = np.linalg.norm(x_axis)
    x_axis = x_axis / raw_ear_dist

    y_vec = nas - inion
    y_axis = y_vec - np.dot(y_vec, x_axis) * x_axis
    y_axis = y_axis / np.linalg.norm(y_axis)

    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    R = np.array([x_axis, y_axis, z_axis])
    scale = measured_mm / raw_ear_dist

    return {"origin": origin, "rotation": R, "scale": scale, "raw_ear_dist": raw_ear_dist, "inion": inion}

def _apply_head_transform(point, transform):
    centered = point - transform["origin"]
    rotated = transform["rotation"] @ centered
    return rotated * transform["scale"]

def generate_demo_points(num_electrodes: int = 24):
    """
    Create a synthetic positions_3d dict that matches Script2 expectations:
      - keys: 0,1,2 for NAS/LPA/RPA
      - electrodes: int >= 100
      - values: np.array([x,y,z]) in arbitrary units
    """
    positions = {}

    # Landmarks in arbitrary units
    positions[LANDMARK_LPA] = np.array([-0.085, 0.000, 0.000], dtype=float)
    positions[LANDMARK_RPA] = np.array([+0.085, 0.000, 0.000], dtype=float)
    positions[LANDMARK_NAS] = np.array([0.000, +0.120, +0.020], dtype=float)

    # Put electrodes on a rough dome (synthetic)
    # This is only for PoC: it lets Script2 run and produce outputs.
    rng = np.random.default_rng(42)
    for i in range(num_electrodes):
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0.15 * np.pi, 0.55 * np.pi)  # upper hemisphere band
        r = 0.09 + rng.uniform(-0.005, 0.005)

        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi) + 0.02
        z = r * np.cos(phi) + 0.03
        positions[ELECTRODE_START_ID + i] = np.array([x, y, z], dtype=float)

    return positions

def create_demo_results_folder(num_electrodes: int = 24):
    """Create results/demo_<timestamp>/points_3d_intermediate.pkl"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = BASE_DIR / "results" / f"demo_{ts}"
    folder.mkdir(parents=True, exist_ok=True)

    points = generate_demo_points(num_electrodes=num_electrodes)
    pkl_path = folder / "points_3d_intermediate.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(points, f)

    return str(folder.name), str(pkl_path)

def run_script2_like(results_folder_name: str, method: str, value_mm: float):
    """
    Run the equivalent of Script2 main() non-interactively on a results folder.
    Writes:
      - electrodes_3d.json
      - electrodes_3d.ply
      - electrodes.elc
    Returns paths.
    """
    results_dir = BASE_DIR / "results" / results_folder_name
    pkl_path = results_dir / "points_3d_intermediate.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing: {pkl_path}")

    with open(pkl_path, "rb") as f:
        positions_3d = pickle.load(f)

    # Convert UI method to measured_mm
    if method == "caliper":
        measured_mm = float(value_mm)
    elif method == "arc":
        measured_mm = float(value_mm) * ARC_TO_CHORD
    elif method == "circumference":
        measured_mm = float(value_mm) * CIRCUMFERENCE_TO_EAR
    else:
        measured_mm = 150.0

    # Verify landmarks exist
    for k in (LANDMARK_NAS, LANDMARK_LPA, LANDMARK_RPA):
        if k not in positions_3d:
            raise ValueError("Missing landmarks in pkl (need NAS/LPA/RPA).")

    nas = positions_3d[LANDMARK_NAS]
    lpa = positions_3d[LANDMARK_LPA]
    rpa = positions_3d[LANDMARK_RPA]

    transform = _build_head_transform(nas, lpa, rpa, measured_mm)
    if transform is None:
        raise RuntimeError("Could not build head transform.")

    # Transform all points
    final_points = {obj_id: _apply_head_transform(pt, transform) for obj_id, pt in positions_3d.items()}
    final_points["INION"] = _apply_head_transform(transform["inion"], transform)

    # Save JSON/PLY/ELC (same conventions as Script2)
    out_json = results_dir / "electrodes_3d.json"
    out_ply = results_dir / "electrodes_3d.ply"
    out_elc = results_dir / "electrodes.elc"

    output = {
        "units": "mm",
        "measurement": {
            "method": method,
            "ear_to_ear_mm": float(measured_mm),
            "scale_factor": float(transform["scale"]),
        },
        "landmarks": {},
        "electrodes": {},
    }

    for obj_id, pos in final_points.items():
        pos_list = pos.tolist()
        if obj_id == "INION":
            output["landmarks"]["INION"] = {"position": pos_list}
        elif obj_id == LANDMARK_NAS:
            output["landmarks"]["NAS"] = {"position": pos_list}
        elif obj_id == LANDMARK_LPA:
            output["landmarks"]["LPA"] = {"position": pos_list}
        elif obj_id == LANDMARK_RPA:
            output["landmarks"]["RPA"] = {"position": pos_list}
        elif isinstance(obj_id, int) and obj_id >= ELECTRODE_START_ID:
            output["electrodes"][f"E{obj_id - ELECTRODE_START_ID}"] = {"position": pos_list}

    output["num_electrodes"] = len(output["electrodes"])

    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)

    # PLY
    ply_lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(final_points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for obj_id, pos in final_points.items():
        if obj_id == "INION":
            r, g, b = 255, 165, 0
        elif obj_id == LANDMARK_NAS:
            r, g, b = 255, 0, 0
        elif obj_id == LANDMARK_LPA:
            r, g, b = 0, 0, 255
        elif obj_id == LANDMARK_RPA:
            r, g, b = 0, 255, 0
        else:
            r, g, b = 0, 100, 255
        ply_lines.append(f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f} {r} {g} {b}")
    out_ply.write_text("\n".join(ply_lines))

    # ELC
    elc_lines = [
        "# Electrode positions",
        "# Generated by EEG Electrode Registration Pipeline (PoC)",
        f"NumberPositions= {len(output['electrodes']) + 3}",
        "UnitPosition mm",
        "Positions",
    ]
    for name in ["NAS", "LPA", "RPA"]:
        pos = output["landmarks"][name]["position"]
        elc_lines.append(f"{name}\t{pos[0]:.2f}\t{pos[1]:.2f}\t{pos[2]:.2f}")
    for ename in sorted(output["electrodes"].keys(), key=lambda x: int(x[1:])):
        pos = output["electrodes"][ename]["position"]
        elc_lines.append(f"{ename}\t{pos[0]:.2f}\t{pos[1]:.2f}\t{pos[2]:.2f}")
    elc_lines.append("Labels")
    elc_lines.append("NAS\tLPA\tRPA\t" + "\t".join(sorted(output["electrodes"].keys(), key=lambda x: int(x[1:]))))
    out_elc.write_text("\n".join(elc_lines))

    return str(out_json), str(out_ply), str(out_elc)

# -------------------------
# UI
# -------------------------
with gr.Blocks() as demo:
    gr.HTML('<div id="poc-title">Video EEG Electrode Registration (PoC UI)</div>')

    gr.Textbox(
        value=INTRO_TEXT,
        label="",
        show_label=False,
        interactive=False,
        lines=5,
    )

    with gr.Tabs():
        # -------- Upload tab --------
        with gr.Tab("Upload & Manage Videos"):
            with gr.Column(elem_id="video-wrap"):
                video_input = gr.Video(label="Video")

            with gr.Row():
                save_btn = gr.Button("Save video data", elem_id="save-btn")
                refresh_btn = gr.Button("Refresh saved videos", elem_id="refresh-btn")

            status = gr.Textbox(
                label="",
                show_label=False,
                placeholder="Status will appear here…",
                interactive=False,
                lines=6,
            )

            gr.Markdown("## Saved videos (data/raw/videos)")

            with gr.Row():
                saved_dd = gr.Dropdown(choices=[], value=None, label="Saved videos")
                delete_btn = gr.Button("Delete selected video", elem_id="delete-btn")

            save_btn.click(save_video, inputs=video_input, outputs=[saved_dd, status])
            refresh_btn.click(refresh_saved_videos, inputs=None, outputs=[saved_dd, status])
            delete_btn.click(delete_selected_video, inputs=saved_dd, outputs=[saved_dd, status])

            demo.load(refresh_saved_videos, inputs=None, outputs=[saved_dd, status])

        # -------- Reports tab --------
        with gr.Tab("Reports (Validation PNGs)"):
            with gr.Row():
                refresh_reports_btn = gr.Button("Refresh report list", elem_id="refresh-reports-btn")
                reports_dd = gr.Dropdown(choices=[], value=None, label="Select a report PNG")

            reports_gallery = gr.Gallery(
                label="Thumbnails",
                show_label=True,
                columns=4,
                rows=2,
                height="auto",
                elem_id="reports-gallery",
            )

            refresh_reports_btn.click(
                list_validation_pngs,
                inputs=None,
                outputs=[reports_dd, reports_gallery],
            )

            demo.load(
                list_validation_pngs,
                inputs=None,
                outputs=[reports_dd, reports_gallery],
            )

            reports_gallery.select(
                preview_by_gallery_select,
                inputs=None,
                outputs=[reports_dd],
            )
        # -------- Demo & Script2 tab --------
        with gr.Tab("Demo & Script2"):

            def _list_json_outputs():
                base = BASE_DIR / "results"
                base.mkdir(exist_ok=True)
                paths = sorted(base.glob("**/electrodes_3d.json"))
                choices = [str(p.relative_to(BASE_DIR)) for p in paths]
                return gr.update(choices=choices, value=(choices[-1] if choices else None))

            def _plot_from_picker(rel_path):
                if not rel_path:
                    return go.Figure()
                return electrodes_json_to_figure(str(BASE_DIR / rel_path))

            gr.Markdown("## Generate a demo results folder and run Script2 (non-interactive)")

            with gr.Row():
                num_el = gr.Slider(8, 64, value=24, step=1, label="Number of demo electrodes")
                gen_btn = gr.Button("Generate demo results", elem_id="refresh-reports-btn")  # primary style

            demo_results_dd = gr.Dropdown(choices=[], value=None, label="Select results folder (demo)")
            gen_status = gr.Textbox(label="", show_label=False, interactive=False, lines=2)

            with gr.Row():
                method = gr.Dropdown(
                    choices=["caliper", "arc", "circumference", "default"],
                    value="default",
                    label="Measurement method"
                )
                mm = gr.Number(value=150.0, label="Measurement value (mm)")

            run_btn = gr.Button("Run Script2", elem_id="save-btn")
            run_status = gr.Textbox(label="", show_label=False, interactive=False, lines=6)

            gr.Markdown("### 3D Preview (from electrodes_3d.json)")
            json_picker = gr.Dropdown(choices=[], value=None, label="Select a JSON to preview (results/*/electrodes_3d.json)")
            plot3d = gr.Plot(label="3D Scatter")

            demo.load(_list_json_outputs, inputs=None, outputs=[json_picker])
            json_picker.change(_plot_from_picker, inputs=[json_picker], outputs=[plot3d])

            gr.Markdown("### Disk cleanup (Codespaces)")

            with gr.Row():
                delete_videos_chk = gr.Checkbox(
                    value=False,
                    label="Also delete videos in data/raw/videos (optional)"
                )
                cleanup_btn = gr.Button("Clean demo files", elem_id="refresh-reports-btn")  # primary style

            cleanup_status = gr.Textbox(label="", show_label=False, interactive=False, lines=4)

            # Clicking cleanup updates status, refreshes dropdown choices, and refreshes saved videos list
            def _cleanup(delete_videos):
                msg = cleanup_demo_files(delete_videos=bool(delete_videos))
                # refresh dropdown after deleting demo folders
                dd_update = _refresh_results_folders()
                # also refresh saved videos dropdown in Upload tab
                saved_dd_update, saved_status_msg = refresh_saved_videos()
                return msg, dd_update, saved_dd_update, saved_status_msg

            cleanup_btn.click(
                _cleanup,
                inputs=[delete_videos_chk],
                outputs=[cleanup_status, demo_results_dd, saved_dd, status],
            )

            out_json = gr.File(label="electrodes_3d.json")
            out_ply = gr.File(label="electrodes_3d.ply")
            out_elc = gr.File(label="electrodes.elc")

            def _refresh_results_folders():
                # list results/* that have points_3d_intermediate.pkl
                base = BASE_DIR / "results"
                base.mkdir(exist_ok=True)
                folders = []
                for p in base.iterdir():
                    if p.is_dir() and (p / "points_3d_intermediate.pkl").exists():
                        folders.append(p.name)
                folders.sort()
                return gr.update(choices=folders, value=(folders[-1] if folders else None))

            def _gen(num_electrodes):
                folder_name, pkl_path = create_demo_results_folder(int(num_electrodes))
                dd = _refresh_results_folders()
                return dd, f"✅ Created: results/{folder_name}\n✅ {pkl_path}"

            def _run(folder_name, method, mm_val):
                if not folder_name:
                    return None, None, None, "❌ Select a results folder first.", gr.update(), go.Figure()

                try:
                    j, p, e = run_script2_like(folder_name, method, float(mm_val))

                    # list all json outputs
                    base = BASE_DIR / "results"
                    paths = sorted(base.glob("**/electrodes_3d.json"))
                    choices = [str(p.relative_to(BASE_DIR)) for p in paths]

                    rel = str(Path(j).relative_to(BASE_DIR))

                    fig = electrodes_json_to_figure(j)

                    return (
                        j,
                        p,
                        e,
                        f"✅ Script2 done.\n- {j}\n- {p}\n- {e}",
                        gr.update(choices=choices, value=rel),   # 🔥 FIX HERE
                        fig,
                    )

                except Exception as ex:
                    return None, None, None, f"❌ Error: {ex}", gr.update(), go.Figure()
        
            gen_btn.click(_gen, inputs=[num_el], outputs=[demo_results_dd, gen_status])
            run_btn.click(_run,inputs=[demo_results_dd, method, mm],outputs=[out_json, out_ply, out_elc, run_status, json_picker, plot3d],)
            demo.load(_refresh_results_folders, inputs=None, outputs=[demo_results_dd])

demo.queue().launch(
    server_name="0.0.0.0",
    server_port=7860,
    css=CSS,
)