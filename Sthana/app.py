"""
Flask backend for Sthāna (library seat monitor).

Run:   python app.py      then open http://localhost:5000

Pages
    /                 overview (landing page)
    /monitor          live monitor: images, video, demo controls
    /history          every saved run (annotated frame, crops, detections)
    /history/<id>     one run in detail
    /records          CSV viewer: desk observations, runs, events
    /map              live library map, zone heatmap, time-lapse, spatial reasoning
"""
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

import config
from graph import SpatialGNN
from pipeline import SeatPipeline
from seat_state import SeatManager
from storage import FILENAMES, RunStore
from video import VideoRunner
from viz import draw_annotated

app = Flask(__name__)
for folder in (config.OUTPUT_DIR, config.IMAGE_DIR, config.UPLOAD_DIR):
    folder.mkdir(parents=True, exist_ok=True)

store = RunStore(config.OUTPUT_DIR)

print("=" * 60, flush=True)
print("LOADING MODELS", flush=True)
print("=" * 60, flush=True)
pipeline = SeatPipeline()

seats = SeatManager(
    grace_minutes=config.GRACE_PERIOD_MINUTES,
    match_iou=config.DESK_MATCH_IOU,
    empty_confirm_frames=config.EMPTY_CONFIRM_FRAMES,
    on_event=store.log_event,
    gnn=SpatialGNN(
        k=config.GNN_NEIGHBOURS, radius=config.GNN_RADIUS, beta=config.GNN_STRENGTH,
        layers=config.GNN_LAYERS, temporal=config.GNN_TEMPORAL,
        uncertain=config.UNCERTAIN_EVIDENCE,
    ) if config.GNN_ENABLED else None,
    inferred_max_missed=config.INFERRED_MAX_MISSED,
)

frame_lock = threading.Lock()
current = {
    "jpeg": None, "frame_id": 0, "source": None, "width": 0, "height": 0,
    "persons": [], "processed_at": None, "inference_ms": 0,
    "run_id": None, "run_saved": False,
}
folder_cursor = {"index": -1}
video_save = {"n": 0, "signature": None}   # decides which video frames keep images


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def list_folder(exts):
    return sorted(p.name for p in config.IMAGE_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in exts)


def error(msg, code=400):
    return jsonify({"error": msg}), code


def process_image(image, source, kind="image"):
    """Run the full pipeline on one frame, update seats, store the run."""
    t0 = time.time()
    analysis = pipeline.analyze(image)
    ms = int((time.time() - t0) * 1000)

    seats.update(analysis)
    snapshot = seats.snapshot()

    if kind == "video":
        # CSV rows for every frame; images every Nth frame or when any seat changes state
        video_save["n"] += 1
        signature = tuple((d["id"], d["state"]) for d in snapshot["desks"])
        save_images = config.SAVE_RUN_IMAGES and (
            signature != video_save["signature"]
            or (video_save["n"] - 1) % max(1, config.VIDEO_SAVE_EVERY_N) == 0
        )
        video_save["signature"] = signature
    else:
        save_images = config.SAVE_RUN_IMAGES

    run_id = store.save_run(image, analysis, snapshot, source, kind, ms, save_images)

    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Could not encode frame")

    with frame_lock:
        current["frame_id"] += 1
        current.update({
            "jpeg": buf.tobytes(), "source": source,
            "width": image.shape[1], "height": image.shape[0],
            "persons": analysis["persons"],
            "processed_at": datetime.now().strftime("%H:%M:%S"),
            "inference_ms": ms, "run_id": run_id, "run_saved": save_images,
        })
        frame_id = current["frame_id"]

    print(f"[frame {frame_id}] {source}: {len(analysis['desks'])} desks, "
          f"{len(analysis['persons'])} persons, {ms} ms, run {run_id}"
          f"{' (saved)' if save_images else ''}", flush=True)
    return {"frame_id": frame_id, "desks": len(analysis["desks"]),
            "persons": len(analysis["persons"]), "inference_ms": ms,
            "run_id": run_id, "saved": save_images}


video = VideoRunner(
    process_fn=lambda frame, name: process_image(frame, name, kind="video"),
    advance_clock_fn=lambda seconds: seats.clock.advance(seconds),
)


def stop_video_for_manual_input():
    """A manual image/camera frame replaces the video feed."""
    if video.is_running():
        video.stop()


@app.context_processor
def inject_brand():
    return {
        "app_name": config.APP_NAME,
        "app_name_native": config.APP_NAME_NATIVE,
        "tagline": config.APP_TAGLINE,
        "year": datetime.now().year,
        "grace_minutes": config.GRACE_PERIOD_MINUTES,
        "floor_name": config.FLOOR_NAME,
        "bucket_minutes": config.HEATMAP_BUCKET_MINUTES,
    }


# ------------------------------------------------------------
# Pages
# ------------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html", page="home")


@app.route("/monitor")
def monitor():
    return render_template("monitor.html", page="monitor")


@app.route("/history")
def history():
    return render_template("history.html", page="history")


@app.route("/history/<run_id>")
def run_detail(run_id):
    if store.get_run(run_id) is None:
        abort(404)
    return render_template("run.html", page="history", run_id=run_id)


@app.route("/records")
def records():
    return render_template("records.html", page="records")


@app.route("/map")
def library_map():
    return render_template("map.html", page="map")


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith(("/api/", "/runs/", "/download/")):
        return error("Not found", 404)
    return render_template("404.html", page=""), 404


# ------------------------------------------------------------
# Live state API
# ------------------------------------------------------------
@app.route("/api/state")
def api_state():
    snap = seats.snapshot()
    with frame_lock:
        snap["frame"] = {
            "frame_id": current["frame_id"] if current["jpeg"] else None,
            "source": current["source"],
            "width": current["width"],
            "height": current["height"],
            "processed_at": current["processed_at"],
            "inference_ms": current["inference_ms"],
            "persons": [{"box": p["box"], "confidence": p["confidence"],
                         "desk_index": p.get("desk_index")} for p in current["persons"]],
        }
        snap["last_run"] = {"id": current["run_id"], "saved": current["run_saved"]}
    snap["video"] = video.status()
    snap["map"]["floor"] = config.FLOOR_NAME
    return jsonify(snap)


@app.route("/api/frame")
def api_frame():
    with frame_lock:
        data = current["jpeg"]
    if data is None:
        return error("No frame processed yet", 404)
    return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.route("/api/annotated")
def api_annotated():
    with frame_lock:
        data, persons = current["jpeg"], list(current["persons"])
    if data is None:
        return error("No frame processed yet", 404)
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    ok, buf = cv2.imencode(".jpg", draw_annotated(image, seats.snapshot(), persons))
    return Response(buf.tobytes(), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.route("/api/images")
def api_images():
    return jsonify({"images": list_folder(config.IMAGE_EXTENSIONS),
                    "videos": list_folder(config.VIDEO_EXTENSIONS),
                    "folder": str(config.IMAGE_DIR)})


# ------------------------------------------------------------
# Image input API
# ------------------------------------------------------------
@app.route("/api/upload", methods=["POST"])
def api_upload():
    f = request.files.get("image")
    if f is None or f.filename == "":
        return error("No image uploaded (form field 'image')")
    image = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return error("Could not decode image")
    stop_video_for_manual_input()
    return jsonify(process_image(image, f.filename))


@app.route("/api/process", methods=["POST"])
def api_process():
    name = (request.get_json(silent=True) or {}).get("filename", "")
    folder = config.IMAGE_DIR.resolve()
    path = (folder / name).resolve()
    if not name or path.parent != folder or not path.is_file():
        return error(f"Image not found in data folder: {name}")
    image = cv2.imread(str(path))
    if image is None:
        return error(f"Could not read {name}")
    stop_video_for_manual_input()
    images = list_folder(config.IMAGE_EXTENSIONS)
    if name in images:
        folder_cursor["index"] = images.index(name)
    return jsonify(process_image(image, name))


@app.route("/api/process_next", methods=["POST"])
def api_process_next():
    images = list_folder(config.IMAGE_EXTENSIONS)
    if not images:
        return error(f"No images in {config.IMAGE_DIR}")
    stop_video_for_manual_input()
    folder_cursor["index"] = (folder_cursor["index"] + 1) % len(images)
    name = images[folder_cursor["index"]]
    image = cv2.imread(str(config.IMAGE_DIR / name))
    if image is None:
        return error(f"Could not read {name}")
    result = process_image(image, name)
    result["filename"] = name
    return jsonify(result)


@app.route("/api/camera", methods=["POST"])
def api_camera():
    stop_video_for_manual_input()
    cap = cv2.VideoCapture(config.CAMERA_SOURCE)
    ok, image = False, None
    try:
        for _ in range(5):  # let the webcam auto-expose
            ok, image = cap.read()
    finally:
        cap.release()
    if not ok or image is None:
        return error(f"Could not read from camera source {config.CAMERA_SOURCE!r}")
    return jsonify(process_image(image, "camera"))


# ------------------------------------------------------------
# Video API
# ------------------------------------------------------------
def _video_options(src):
    """Read playback options from a JSON body or form fields."""
    def num(key, default):
        try:
            return float(src.get(key, default))
        except (TypeError, ValueError):
            return default

    def flag(key, default):
        v = src.get(key, default)
        return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "on", "yes")

    return {
        "sample_seconds": num("sample_seconds", config.VIDEO_SAMPLE_SECONDS),
        "speed": num("speed", config.VIDEO_SPEED),
        "sync_clock": flag("sync_clock", config.VIDEO_SYNC_CLOCK),
        "loop": flag("loop", config.VIDEO_LOOP),
        "reset": flag("reset", True),
        "start_time": str(src.get("start_time", "") or "").strip(),
    }


def _start_video(source, name, live, opts):
    video.stop()
    start_time = opts.pop("start_time")
    if opts.pop("reset"):
        seats.reset()
        folder_cursor["index"] = -1
    if start_time:   # e.g. "08:00": the recording's real start, so history gets real times
        seats.set_clock_time(start_time)
    video_save.update(n=0, signature=None)
    video.start(source, name, live=live, **opts)
    print(f"[video] started {name} {opts}", flush=True)
    return jsonify({"ok": True, "video": video.status()})


@app.route("/api/video/upload", methods=["POST"])
def api_video_upload():
    f = request.files.get("video")
    if f is None or f.filename == "":
        return error("No video uploaded (form field 'video')")
    name = secure_filename(f.filename) or "video.mp4"
    if Path(name).suffix.lower() not in config.VIDEO_EXTENSIONS:
        return error(f"Unsupported video type: {name}")
    path = config.UPLOAD_DIR / name
    f.save(str(path))
    try:
        return _start_video(str(path), name, False, _video_options(request.form))
    except ValueError as e:
        return error(str(e))


@app.route("/api/video/start", methods=["POST"])
def api_video_start():
    body = request.get_json(silent=True) or {}
    opts = _video_options(body)
    try:
        if body.get("camera"):
            return _start_video(config.CAMERA_SOURCE, "live camera", True, opts)
        name = body.get("filename", "")
        folder = config.IMAGE_DIR.resolve()
        path = (folder / name).resolve()
        if not name or path.parent != folder or not path.is_file():
            return error(f"Video not found in data folder: {name}")
        return _start_video(str(path), name, False, opts)
    except ValueError as e:
        return error(str(e))


@app.route("/api/video/pause", methods=["POST"])
def api_video_pause():
    video.pause()
    return jsonify({"ok": True, "video": video.status()})


@app.route("/api/video/resume", methods=["POST"])
def api_video_resume():
    video.resume()
    return jsonify({"ok": True, "video": video.status()})


@app.route("/api/video/stop", methods=["POST"])
def api_video_stop():
    video.stop()
    return jsonify({"ok": True, "video": video.status()})


# ------------------------------------------------------------
# History + records API
# ------------------------------------------------------------
def _int_arg(name, default):
    try:
        return int(request.args.get(name, default))
    except (TypeError, ValueError):
        return default


@app.route("/api/runs")
def api_runs():
    return jsonify(store.list_runs(
        q=request.args.get("q", "").strip(),
        kind=request.args.get("kind", "").strip(),
        page=_int_arg("page", 1),
        per_page=_int_arg("per_page", 24),
    ))


@app.route("/api/runs/<run_id>")
def api_run(run_id):
    meta = store.get_run(run_id)
    if meta is None:
        return error("Run not found", 404)
    return jsonify(meta)


@app.route("/runs/<run_id>/<path:rel>")
def run_file(run_id, rel):
    path = store.file_path(run_id, rel)
    if path is None:
        abort(404)
    return send_file(path, max_age=86400)


@app.route("/api/records/<name>")
def api_records(name):
    if name not in FILENAMES:
        return error("Unknown table", 404)
    return jsonify(store.query(
        name,
        q=request.args.get("q", "").strip(),
        state=request.args.get("state", "").strip(),
        page=_int_arg("page", 1),
        per_page=_int_arg("per_page", 50),
    ))


@app.route("/download/<name>.csv")
def download_csv(name):
    path = store.files.get(name)
    if path is None or not path.exists():
        abort(404)
    return send_file(path, as_attachment=True, download_name=FILENAMES[name],
                     mimetype="text/csv", max_age=0)


# ------------------------------------------------------------
# Demo / admin API
# ------------------------------------------------------------
@app.route("/api/desks/<desk_id>/elapsed", methods=["POST"])
def api_set_elapsed(desk_id):
    try:
        minutes = float((request.get_json(silent=True) or {}).get("minutes"))
        seats.set_elapsed(desk_id, minutes)
    except KeyError as e:
        return error(str(e).strip("'"), 404)
    except (TypeError, ValueError) as e:
        return error(str(e) or "minutes must be a number")
    return jsonify({"ok": True})


@app.route("/api/settings", methods=["POST"])
def api_settings():
    body = request.get_json(silent=True) or {}
    try:
        if "grace_minutes" in body:
            seats.set_grace(body["grace_minutes"])
    except (TypeError, ValueError) as e:
        return error(str(e))
    return jsonify({"ok": True, "grace_minutes": seats.grace_seconds / 60})


@app.route("/api/clock/advance", methods=["POST"])
def api_advance():
    try:
        seats.advance_clock(float((request.get_json(silent=True) or {}).get("minutes", 0)))
    except (TypeError, ValueError):
        return error("minutes must be a number")
    return jsonify({"ok": True})


@app.route("/api/clock/set", methods=["POST"])
def api_clock_set():
    try:
        seats.set_clock_time((request.get_json(silent=True) or {}).get("time", ""))
    except ValueError as e:
        return error(str(e))
    return jsonify({"ok": True})


@app.route("/api/timeline")
def api_timeline():
    return jsonify(store.timeline(
        bucket_min=_int_arg("bucket", config.HEATMAP_BUCKET_MINUTES),
        day=request.args.get("day") or None,
    ))


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Clears live seats and timers. Saved history and CSVs are kept."""
    video.stop()
    seats.reset()
    folder_cursor["index"] = -1
    with frame_lock:
        current.update({"jpeg": None, "source": None, "persons": [], "processed_at": None,
                        "run_id": None, "run_saved": False})
    return jsonify({"ok": True})


if __name__ == "__main__":
    print(f"\nOpen http://localhost:{config.PORT}\n", flush=True)
    # debug=False so the reloader doesn't load the models twice
    app.run(host=config.HOST, port=config.PORT, debug=False, threaded=True)
