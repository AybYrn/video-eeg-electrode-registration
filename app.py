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

demo.queue().launch(
    server_name="0.0.0.0",
    server_port=7860,
    css=CSS,
)