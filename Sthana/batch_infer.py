"""
batch_infer.py: run all three models on a whole folder, standalone.

Does not need the Flask app. For every image (and, optionally, video) in the data
folder it runs:

    1. Desk model      (full frame)   -> desk boxes + desk crops
    2. Person model    (full frame)   -> person boxes, each matched to one desk
    3. Object model    (desk crops)   -> belongings on each desk

and writes each model's own results plus a combined per-desk observation:

    batch_outputs/<run>/
    ├── desks/     annotated/  crops/   desks.csv
    ├── persons/   annotated/           persons.csv
    ├── objects/   annotated/           objects.csv        (one annotated crop per desk)
    ├── combined/  annotated/           seat_observations.csv  (+ .parquet if pandas+pyarrow)
    └── summary.json   row counts, collection window, settings, per-image counts

Per desk, the combined observation is:
    PERSON_PRESENT   a person overlaps the desk (or the object model sees a human)
    BELONGINGS_ONLY  no person, but bag / book / bottle / charger / phone / laptop
    EMPTY            neither
It records what the camera observed. The 30-minute rule is time-based and is applied
by the app, not here.

Usage (from the seat_monitor folder):
    python batch_infer.py
    python batch_infer.py --data data --models models --out batch_outputs
    python batch_infer.py --videos --video-step 5        # also sample videos, 1 frame / 5 s
    python batch_infer.py --no-images                    # CSVs only (much faster to write)
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ultralytics is not installed:  pip install ultralytics")

# ---------------------------------------------------------------------------
# Settings (same defaults as the app's config.py)
# ---------------------------------------------------------------------------
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}

# The object model stores its class names as "1", "2", ... so map by class id
OBJECT_CLASS_LABELS = {0: "bag", 1: "book", 2: "bottle", 3: "chair",
                       4: "charger", 5: "phone", 6: "human", 7: "laptop"}
BELONGINGS = {"bag", "book", "bottle", "charger", "phone", "laptop"}   # chair ignored
HUMAN_LABELS = {"human", "person"}

DESK_PAD_RATIO = 0.15        # grow desk box before checking person overlap
PERSON_DESK_OVERLAP = 0.30   # intersection / min(person area, padded desk area)
MAX_OBJECTS_PER_DESK = 5

OBS_COLORS = {"PERSON_PRESENT": (48, 59, 255), "BELONGINGS_ONLY": (0, 149, 255), "EMPTY": (89, 199, 52)}
DESK_COLOR, PERSON_COLOR, OBJECT_COLOR = (89, 199, 52), (255, 132, 10), (10, 214, 255)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clamp_box(b, w, h):
    x1, y1, x2, y2 = (int(round(v)) for v in b)
    return [max(0, min(x1, w)), max(0, min(y1, h)), max(0, min(x2, w)), max(0, min(y2, h))]


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def inter(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def pad(b, r, w, h):
    px, py = int((b[2] - b[0]) * r), int((b[3] - b[1]) * r)
    return clamp_box([b[0] - px, b[1] - py, b[2] + px, b[3] + py], w, h)


def parse(result, names, w, h, keep=None):
    out = []
    if result.boxes is None:
        return out
    for box in result.boxes:
        cid = int(box.cls[0])
        if keep is not None and cid not in keep:
            continue
        b = clamp_box(box.xyxy[0].tolist(), w, h)
        if b[2] <= b[0] or b[3] <= b[1]:
            continue
        out.append({"class_id": cid, "label": str(names.get(cid, cid)).lower(),
                    "confidence": round(float(box.conf[0]), 4), "box": b})
    return out


def tag(img, text, x, y, color, scale=0.6, ink=(255, 255, 255)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    y = max(th + 8, y)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 8, y), color, -1)
    cv2.putText(img, text, (x + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, ink, 2, cv2.LINE_AA)


def safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


WHATSAPP = re.compile(r"(\d{4}-\d{2}-\d{2}) at (\d{1,2})\.(\d{2})\.(\d{2})\s*([AP]M)", re.I)
STAMP = re.compile(r"(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})")


def capture_time(path):
    """Best guess of when a photo was taken: EXIF, then filename, then file time."""
    try:
        from PIL import Image
        exif = Image.open(path).getexif()
        raw = exif.get(36867) or exif.get(306)          # DateTimeOriginal / DateTime
        if not raw:
            raw = exif.get_ifd(0x8769).get(36867)
        if raw:
            return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S").isoformat(), "exif"
    except Exception:
        pass
    m = WHATSAPP.search(path.name)                      # "WhatsApp Image 2026-09-26 at 8.42.40 AM"
    if m:
        t = datetime.strptime(f"{m[1]} {m[2]}:{m[3]}:{m[4]} {m[5].upper()}", "%Y-%m-%d %I:%M:%S %p")
        return t.isoformat(), "filename"
    m = STAMP.search(path.name)                         # IMG_20260926_084240.jpg
    if m:
        try:
            return datetime(*map(int, m.groups())).isoformat(), "filename"
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"), "file_mtime"


def load(path, name):
    path = Path(path)
    if not path.exists():
        sys.exit(f"{name} weights not found: {path}")
    print(f"  {name:<14} {path}")
    return YOLO(str(path))


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------
def frames(data_dir, use_videos, video_step):
    """Yields (frame_id, source_file, frame_index, image, capture_time, time_source)."""
    files = sorted(p for p in Path(data_dir).rglob("*") if p.is_file())
    for p in files:
        ext = p.suffix.lower()
        if ext in IMAGE_EXT:
            img = cv2.imread(str(p))
            if img is None:
                print(f"  ! could not read {p.name}, skipped")
                continue
            t, src = capture_time(p)
            yield p.stem, p, 0, img, t, src
        elif ext in VIDEO_EXT and use_videos:
            cap = cv2.VideoCapture(str(p))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            if fps <= 1 or fps > 240:
                fps = 25.0
            step = max(1, round(video_step * fps))
            # a file's modified time is roughly when recording ended -> start = end - duration
            duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
            start = datetime.fromtimestamp(p.stat().st_mtime) - timedelta(seconds=duration)
            i = 0
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                if i % step == 0:
                    sec = i / fps
                    t = (start + timedelta(seconds=sec)).isoformat(timespec="seconds")
                    yield f"{p.stem}_t{int(sec):05d}s", p, i, img, t, "video_mtime_minus_duration"
                i += 1
            cap.release()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Run desk, person and object models on a whole folder.")
    ap.add_argument("--data", default=str(here / "data"), help="folder with images/videos (searched recursively)")
    ap.add_argument("--models", default=str(here / "models"), help="folder with the three .pt files")
    ap.add_argument("--desk-model", default="weights_deskDetection.pt")
    ap.add_argument("--person-model", default="yolov8n.pt")
    ap.add_argument("--object-model", default="weights_objectDetection.pt")
    ap.add_argument("--out", default=str(here / "batch_outputs"))
    ap.add_argument("--desk-conf", type=float, default=0.25)
    ap.add_argument("--person-conf", type=float, default=0.25)
    ap.add_argument("--object-conf", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="e.g. cpu, 0 (GPU)")
    ap.add_argument("--videos", action="store_true", help="also sample frames from videos")
    ap.add_argument("--video-step", type=float, default=2.0, help="seconds between sampled video frames")
    ap.add_argument("--no-images", action="store_true", help="write CSVs only, no annotated images/crops")
    args = ap.parse_args()

    run = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) / run
    dirs = {k: out / k for k in ("desks", "persons", "objects", "combined")}
    for k, d in dirs.items():
        (d / "annotated").mkdir(parents=True, exist_ok=True)
    (dirs["desks"] / "crops").mkdir(parents=True, exist_ok=True)
    save = not args.no_images

    print("Loading models")
    models = Path(args.models)
    desk_m = load(models / args.desk_model, "desk model")
    person_m = load(models / args.person_model, "person model")
    object_m = load(models / args.object_model, "object model")
    person_ids = {c for c, n in person_m.names.items() if str(n).lower() == "person"}
    if not person_ids:
        sys.exit("person model has no 'person' class")
    object_names = {**dict(object_m.names), **OBJECT_CLASS_LABELS}
    pred = dict(imgsz=args.imgsz, verbose=False)
    if args.device is not None:
        pred["device"] = args.device

    desk_rows, person_rows, object_rows, seat_rows, per_image = [], [], [], [], []
    obs_count = Counter()
    times = []
    t_start = time.time()
    print(f"\nProcessing {args.data}")

    for n, (fid, path, fidx, img, ctime, tsrc) in enumerate(frames(args.data, args.videos, args.video_step), 1):
        h, w = img.shape[:2]
        t0 = time.time()
        common = {"frame_id": fid, "source_file": str(path.name), "frame_index": fidx,
                  "capture_time": ctime, "time_source": tsrc, "width": w, "height": h}

        # ---- 1. desks -------------------------------------------------------
        desks = parse(desk_m.predict(img, conf=args.desk_conf, **pred)[0], desk_m.names, w, h)
        desks.sort(key=lambda d: (d["box"][1], d["box"][0]))          # top-left to bottom-right
        for i, d in enumerate(desks, 1):
            d["desk_id"] = f"{fid}_desk{i:02d}"
            x1, y1, x2, y2 = d["box"]
            crop_rel = ""
            if save:
                crop_rel = f"desks/crops/{safe(d['desk_id'])}.jpg"
                cv2.imwrite(str(out / crop_rel), img[y1:y2, x1:x2])
            desk_rows.append({**common, "desk_id": d["desk_id"], "desk_label": d["label"],
                              "desk_conf": d["confidence"], "x1": x1, "y1": y1, "x2": x2, "y2": y2, "crop": crop_rel})

        # ---- 2. persons (full frame), matched to one desk each ----------------
        persons = parse(person_m.predict(img, conf=args.person_conf, classes=list(person_ids), **pred)[0],
                        person_m.names, w, h, keep=person_ids)
        padded = [pad(d["box"], DESK_PAD_RATIO, w, h) for d in desks]
        best_overlap = [0.0] * len(desks)
        for j, p in enumerate(persons, 1):
            best_i, best = None, 0.0
            for i, pb in enumerate(padded):
                s = inter(p["box"], pb) / max(1, min(area(p["box"]), area(pb)))
                if s > best:
                    best_i, best = i, s
            assigned = best_i is not None and best >= PERSON_DESK_OVERLAP
            if assigned:
                best_overlap[best_i] = max(best_overlap[best_i], best)
            x1, y1, x2, y2 = p["box"]
            person_rows.append({**common, "person_id": f"{fid}_person{j:02d}", "person_conf": p["confidence"],
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                "desk_id": desks[best_i]["desk_id"] if assigned else "",
                                "desk_overlap": round(best, 3)})

        # ---- 3. objects on every desk crop ------------------------------------
        # (run on all desks so the objects table is complete; the combined
        #  observation still gives a matched person priority)
        combined_img = img.copy() if save else None
        for i, d in enumerate(desks):
            x1, y1, x2, y2 = d["box"]
            crop = img[y1:y2, x1:x2]
            objs = []
            if crop.size:
                ch, cw = crop.shape[:2]
                objs = parse(object_m.predict(crop, conf=args.object_conf, **pred)[0], object_names, cw, ch)
                objs.sort(key=lambda o: -o["confidence"])
                objs = objs[:MAX_OBJECTS_PER_DESK]
                if save:
                    vis = crop.copy()
                    for o in objs:
                        a, b, c, e = o["box"]
                        cv2.rectangle(vis, (a, b), (c, e), OBJECT_COLOR, 2)
                        tag(vis, f"{o['label']} {o['confidence']:.2f}", a, b, OBJECT_COLOR, 0.5, (29, 29, 29))
                    cv2.imwrite(str(dirs["objects"] / "annotated" / f"{safe(d['desk_id'])}.jpg"), vis)
            for o in objs:
                a, b, c, e = o["box"]
                object_rows.append({**common, "desk_id": d["desk_id"], "label": o["label"], "class_id": o["class_id"],
                                    "object_conf": o["confidence"],
                                    "x1": a + x1, "y1": b + y1, "x2": c + x1, "y2": e + y1,
                                    "counts_as_belonging": o["label"] in BELONGINGS})

            labels = [o["label"] for o in objs]
            belongings = sorted({l for l in labels if l in BELONGINGS})
            if best_overlap[i] > 0:
                obs, why = "PERSON_PRESENT", f"person overlap {best_overlap[i]:.2f}"
            elif any(l in HUMAN_LABELS for l in labels):
                obs, why = "PERSON_PRESENT", "human seen in desk crop"
            elif belongings:
                obs, why = "BELONGINGS_ONLY", "belongings: " + ", ".join(belongings)
            else:
                obs, why = "EMPTY", "no person or belongings"
            obs_count[obs] += 1
            seat_rows.append({**common, "desk_id": d["desk_id"], "desk_conf": d["confidence"],
                              "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                              "observation": obs, "reason": why,
                              "person_overlap": round(best_overlap[i], 3),
                              "objects": ";".join(labels), "belongings": ";".join(belongings),
                              "data_type": "observed"})
            if save:
                col = OBS_COLORS[obs]
                cv2.rectangle(combined_img, (x1, y1), (x2, y2), col, 3)
                tag(combined_img, f"{i + 1:02d} {obs.replace('_', ' ').title()}", x1, y1, col)

        # ---- per-model annotated frames ----------------------------------------
        if save:
            dimg, pimg = img.copy(), img.copy()
            for i, d in enumerate(desks, 1):
                x1, y1, x2, y2 = d["box"]
                cv2.rectangle(dimg, (x1, y1), (x2, y2), DESK_COLOR, 3)
                tag(dimg, f"desk {i:02d} {d['confidence']:.2f}", x1, y1, DESK_COLOR)
            for p in persons:
                x1, y1, x2, y2 = p["box"]
                cv2.rectangle(pimg, (x1, y1), (x2, y2), PERSON_COLOR, 3)
                tag(pimg, f"person {p['confidence']:.2f}", x1, y1, PERSON_COLOR)
                cv2.rectangle(combined_img, (x1, y1), (x2, y2), PERSON_COLOR, 1)
            name = f"{safe(fid)}.jpg"
            cv2.imwrite(str(dirs["desks"] / "annotated" / name), dimg)
            cv2.imwrite(str(dirs["persons"] / "annotated" / name), pimg)
            cv2.imwrite(str(dirs["combined"] / "annotated" / name), combined_img)

        ms = int((time.time() - t0) * 1000)
        times.append(ctime)
        c = Counter(r["observation"] for r in seat_rows if r["frame_id"] == fid)
        per_image.append({"frame_id": fid, "source_file": path.name, "capture_time": ctime, "desks": len(desks),
                          "persons": len(persons), "occupied": c["PERSON_PRESENT"],
                          "belongings_only": c["BELONGINGS_ONLY"], "empty": c["EMPTY"], "ms": ms})
        print(f"  [{n:>4}] {fid[:48]:<48} desks {len(desks):>3}  people {len(persons):>3}  "
              f"occupied {c['PERSON_PRESENT']:>3}  held {c['BELONGINGS_ONLY']:>3}  empty {c['EMPTY']:>3}  {ms} ms")

    if not per_image:
        sys.exit(f"No images found in {args.data}" + ("" if args.videos else " (add --videos to include videos)"))

    # ---- write tables ---------------------------------------------------------
    def write(path, rows):
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)

    write(dirs["desks"] / "desks.csv", desk_rows)
    write(dirs["persons"] / "persons.csv", person_rows)
    write(dirs["objects"] / "objects.csv", object_rows)
    write(dirs["combined"] / "seat_observations.csv", seat_rows)
    write(out / "per_image.csv", per_image)

    parquet = False
    try:
        import pandas as pd
        pd.DataFrame(seat_rows).to_parquet(dirs["combined"] / "seat_observations.parquet", index=False)
        parquet = True
    except Exception as e:
        print(f"\n  (Parquet skipped: {e.__class__.__name__}. pip install pandas pyarrow to enable)")

    summary = {
        "run": run,
        "data_folder": str(Path(args.data).resolve()),
        "frames": len(per_image),
        "rows": {"desks": len(desk_rows), "persons": len(person_rows), "objects": len(object_rows),
                 "seat_observations": len(seat_rows)},
        "observations": dict(obs_count),
        "collection_window": {"first": min(times), "last": max(times),
                              "time_sources": dict(Counter(r["time_source"] for r in seat_rows))},
        "settings": {"desk_conf": args.desk_conf, "person_conf": args.person_conf, "object_conf": args.object_conf,
                     "imgsz": args.imgsz, "person_desk_overlap": PERSON_DESK_OVERLAP, "desk_pad_ratio": DESK_PAD_RATIO,
                     "object_classes": OBJECT_CLASS_LABELS, "belongings": sorted(BELONGINGS),
                     "models": {"desk": args.desk_model, "person": args.person_model, "object": args.object_model}},
        "data_type": "all rows are observed (model output on real frames); nothing inferred or synthetic",
        "parquet": parquet,
        "seconds": round(time.time() - t_start, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDone")
    print(f"  frames             {summary['frames']}")
    print(f"  desk rows          {len(desk_rows)}")
    print(f"  person rows        {len(person_rows)}")
    print(f"  object rows        {len(object_rows)}")
    print(f"  seat observations  {len(seat_rows)}  {dict(obs_count)}")
    print(f"  window             {summary['collection_window']['first']}  ->  {summary['collection_window']['last']}")
    print(f"  output             {out}")


if __name__ == "__main__":
    main()
