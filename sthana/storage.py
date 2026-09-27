"""
Run storage. Every analysis of a frame becomes a "run".

outputs/
├── runs/<run_id>/
│   ├── original.jpg              frame as received
│   ├── annotated.jpg             desk boxes coloured by state + person / object boxes
│   ├── thumb.jpg                 small annotated preview for the History grid
│   ├── desks/D01_crop.jpg        cropped desk (what the object model sees)
│   ├── desks/D01_objects.jpg     same crop with object detections drawn
│   └── meta.json                 everything above, machine-readable
├── runs.csv                      one row per analysis
├── desk_observations.csv         one row per desk per analysis
└── events.csv                    one row per seat state change

CSV rows are written for every analysis. Images are optional per run
(video frames save images only every Nth frame or on a state change).
"""
import csv
import json
import math
import re
import threading
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2

from pipeline import OBS_PERSON
from viz import draw_annotated, draw_crop_objects

RUN_ID_RE = re.compile(r"^[0-9A-Za-z_-]{1,64}$")

FILENAMES = {
    "runs": "runs.csv",
    "desks": "desk_observations.csv",
    "events": "events.csv",
}

FIELDS = {
    "runs": [
        "run_id", "timestamp", "seat_time", "source", "kind", "desks", "available",
        "reclaimable", "partial", "occupied", "persons", "inference_ms", "has_images",
    ],
    "desks": [
        "run_id", "timestamp", "seat_time", "source", "desk_id", "zone", "row", "col",
        "state", "observation", "objects", "object_count", "desk_conf", "evidence",
        "gnn_occupied", "gnn_held", "gnn_free", "gnn_flag",
        "x1", "y1", "x2", "y2", "timer_elapsed_s", "timer_remaining_s", "reason",
        "crop", "objects_image",
    ],
    "events": ["timestamp", "seat_clock", "desk_id", "state", "text"],
}


class RunStore:
    def __init__(self, root, jpeg_quality=90, thumb_width=560):
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.files = {k: self.root / v for k, v in FILENAMES.items()}
        self.lock = threading.RLock()
        self.jpeg = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        self.thumb_width = thumb_width
        self._checked = set()

    # ============================================================
    # Writing
    # ============================================================
    def _archive_if_old(self, name):
        """A CSV from an older version (different columns) is renamed, not mixed."""
        if name in self._checked:
            return
        self._checked.add(name)
        path = self.files[name]
        if not path.exists() or path.stat().st_size == 0:
            return
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != FIELDS[name]:
            old = path.with_name(f"{path.stem}.old-{datetime.now():%Y%m%d-%H%M%S}.csv")
            path.rename(old)
            print(f"[storage] {path.name} had old columns; archived as {old.name}", flush=True)

    def _append(self, name, rows):
        if not rows:
            return
        self._archive_if_old(name)
        path = self.files[name]
        new_file = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS[name], extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerows(rows)

    def _new_run_id(self, now):
        base = now.strftime("%Y%m%d-%H%M%S-") + f"{now.microsecond // 1000:03d}"
        run_id, i = base, 1
        while (self.runs_dir / run_id).exists():
            i += 1
            run_id = f"{base}-{i}"
        return run_id

    def _write_jpg(self, path, img):
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img, self.jpeg)

    def save_run(self, image, analysis, snapshot, source, kind, inference_ms, save_images):
        """Store one analysis. Returns the run_id."""
        now = datetime.now()
        ts = now.isoformat(timespec="seconds")
        seat_time = snapshot.get("seat_time", ts)   # demo clock / video time
        states = {d["id"]: d for d in snapshot["desks"]}
        dets = analysis["desks"]
        summary = snapshot["summary"]

        with self.lock:
            run_id = self._new_run_id(now)
            run_dir = self.runs_dir / run_id

            if save_images:
                run_dir.mkdir(parents=True, exist_ok=True)
                self._write_jpg(run_dir / "original.jpg", image)
                annotated = draw_annotated(image, snapshot, analysis["persons"])
                self._write_jpg(run_dir / "annotated.jpg", annotated)
                h, w = annotated.shape[:2]
                tw = min(self.thumb_width, w)
                thumb = cv2.resize(annotated, (tw, max(1, round(h * tw / w))),
                                   interpolation=cv2.INTER_AREA)
                self._write_jpg(run_dir / "thumb.jpg", thumb)

            desks_meta, desk_rows = [], []
            for det in sorted(dets, key=lambda d: d.get("desk_id") or ""):
                did = det.get("desk_id") or "D??"
                seat = states.get(did, {})
                state = seat.get("state", "UNKNOWN")
                x1, y1, x2, y2 = det["box"]
                objects = [{"label": o["label"], "confidence": o["confidence"], "box": o["box"]}
                           for o in det["objects"]]

                crop_rel = objects_rel = ""
                if save_images:
                    crop = image[y1:y2, x1:x2]
                    if crop.size:
                        crop_rel = f"desks/{did}_crop.jpg"
                        objects_rel = f"desks/{did}_objects.jpg"
                        if det["observation"] == OBS_PERSON and not objects:
                            note = "person present - object model skipped"
                        elif not objects:
                            note = "no objects detected"
                        else:
                            note = ""
                        self._write_jpg(run_dir / crop_rel, crop)
                        self._write_jpg(run_dir / objects_rel,
                                        draw_crop_objects(crop, objects, x1, y1, state, note))

                elapsed, remaining = seat.get("elapsed_s"), seat.get("remaining_s")
                gnn = seat.get("gnn") or {}
                gp = gnn.get("p") or ["", "", ""]
                desks_meta.append({
                    "id": did,
                    "state": state,
                    "label": seat.get("label", state),
                    "message": seat.get("message", ""),
                    "observation": det["observation"],
                    "reason": det.get("reason", ""),
                    "box": det["box"],
                    "confidence": det["confidence"],
                    "objects": objects,
                    "elapsed_s": elapsed,
                    "remaining_s": remaining,
                    "crop": crop_rel,
                    "objects_image": objects_rel,
                    "zone": seat.get("zone"), "row": seat.get("row"), "col": seat.get("col"),
                    "evidence": det.get("evidence"), "weak_objects": det.get("weak_objects", []),
                    "gnn": gnn or None,
                })
                desk_rows.append({
                    "run_id": run_id, "timestamp": ts, "seat_time": seat_time, "source": source,
                    "desk_id": did, "zone": seat.get("zone", ""), "row": seat.get("row", ""),
                    "col": seat.get("col", ""), "evidence": det.get("evidence", ""),
                    "gnn_occupied": gp[0], "gnn_held": gp[1], "gnn_free": gp[2],
                    "gnn_flag": gnn.get("flag", ""),
                    "state": state, "observation": det["observation"],
                    "objects": ";".join(o["label"] for o in objects),
                    "object_count": len(objects), "desk_conf": det["confidence"],
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "timer_elapsed_s": "" if elapsed is None else round(elapsed),
                    "timer_remaining_s": "" if remaining is None else round(remaining),
                    "reason": det.get("reason", ""),
                    "crop": crop_rel, "objects_image": objects_rel,
                })

            persons = []
            for p in analysis["persons"]:
                idx = p.get("desk_index")
                persons.append({
                    "box": p["box"], "confidence": p["confidence"],
                    "desk_id": dets[idx].get("desk_id") if idx is not None else None,
                })

            if save_images:
                meta = {
                    "run_id": run_id, "timestamp": ts, "seat_time": seat_time,
                    "source": source, "kind": kind,
                    "image_size": analysis.get("image_size"), "inference_ms": inference_ms,
                    "summary": summary, "grace_minutes": snapshot.get("grace_minutes"),
                    "desks": desks_meta, "persons": persons,
                    "files": {"original": "original.jpg", "annotated": "annotated.jpg",
                              "thumb": "thumb.jpg"},
                }
                (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            self._append("runs", [{
                "run_id": run_id, "timestamp": ts, "seat_time": seat_time,
                "source": source, "kind": kind,
                "desks": len(dets), "available": summary["available"],
                "reclaimable": summary["reclaimable"], "partial": summary["partial"],
                "occupied": summary["occupied"], "persons": len(persons),
                "inference_ms": inference_ms, "has_images": 1 if save_images else 0,
            }])
            self._append("desks", desk_rows)
        return run_id

    def log_event(self, event):
        """SeatManager callback: one row per state change / demo action."""
        with self.lock:
            self._append("events", [{
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "seat_clock": event.get("time", ""),
                "desk_id": event.get("desk") or "",
                "state": event.get("state") or "",
                "text": event.get("text", ""),
            }])

    # ============================================================
    # Reading
    # ============================================================
    def _read(self, name):
        with self.lock:
            self._archive_if_old(name)
        path = self.files[name]
        if not path.exists():
            return list(FIELDS[name]), []
        with self.lock:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                cols = reader.fieldnames or list(FIELDS[name])
        return cols, rows

    @staticmethod
    def _page(rows, page, per_page):
        per_page = max(1, min(500, int(per_page)))
        total = len(rows)
        pages = max(1, math.ceil(total / per_page))
        page = max(1, min(int(page), pages))
        start = (page - 1) * per_page
        return rows[start:start + per_page], total, page, pages, per_page

    def query(self, name, q="", state="", page=1, per_page=50):
        """Newest-first, filtered, paginated view of one CSV."""
        cols, rows = self._read(name)
        rows.reverse()
        counts = dict(Counter(r.get("state", "") for r in rows)) if "state" in cols else {}
        counts["_all"] = len(rows)
        if state:
            rows = [r for r in rows if r.get("state") == state]
        if q:
            ql = q.lower()
            rows = [r for r in rows if any(ql in (v or "").lower() for v in r.values())]
        page_rows, total, page, pages, per_page = self._page(rows, page, per_page)
        return {"name": name, "columns": cols, "rows": page_rows, "total": total,
                "page": page, "pages": pages, "per_page": per_page, "counts": counts}

    def _saved_runs(self):
        _, rows = self._read("runs")
        return [r for r in rows if r.get("has_images") == "1"
                and (self.runs_dir / r["run_id"] / "meta.json").exists()]

    def list_runs(self, q="", kind="", page=1, per_page=24):
        _, all_rows = self._read("runs")
        saved = list(reversed(self._saved_runs()))
        rows = saved
        if kind:
            rows = [r for r in rows if r.get("kind") == kind]
        if q:
            ql = q.lower()
            rows = [r for r in rows if ql in r.get("source", "").lower() or ql in r["run_id"].lower()]
        page_rows, total, page, pages, per_page = self._page(rows, page, per_page)
        for r in page_rows:
            r["thumb"] = f"/runs/{r['run_id']}/thumb.jpg"
        desk_rows = 0
        if self.files["desks"].exists():
            with self.lock, open(self.files["desks"], encoding="utf-8") as f:
                desk_rows = max(0, sum(1 for _ in f) - 1)
        return {
            "runs": page_rows, "total": total, "page": page, "pages": pages,
            "stats": {
                "analyses": len(all_rows), "saved": len(saved), "desk_rows": desk_rows,
                "last": saved[0]["timestamp"] if saved else None,
            },
        }

    def get_run(self, run_id):
        if not RUN_ID_RE.match(run_id or ""):
            return None
        meta_path = self.runs_dir / run_id / "meta.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        base = f"/runs/{run_id}/"
        meta["urls"] = {k: base + v for k, v in meta.get("files", {}).items()}
        meta["urls"]["meta"] = base + "meta.json"
        for d in meta["desks"]:
            d["crop_url"] = base + d["crop"] if d.get("crop") else None
            d["objects_url"] = base + d["objects_image"] if d.get("objects_image") else None
        ids = [r["run_id"] for r in self._saved_runs()]
        meta["prev"] = meta["next"] = None
        if run_id in ids:
            i = ids.index(run_id)
            meta["prev"] = ids[i - 1] if i > 0 else None             # older
            meta["next"] = ids[i + 1] if i + 1 < len(ids) else None  # newer
        return meta

    def file_path(self, run_id, rel):
        """Safe path to a file inside one run folder, or None."""
        if not RUN_ID_RE.match(run_id or ""):
            return None
        base = (self.runs_dir / run_id).resolve()
        path = (base / rel).resolve()
        if base not in path.parents or not path.is_file():
            return None
        return path

    # ============================================================
    # Time-lapse heatmap
    # ============================================================
    def timeline(self, bucket_min=60, day=None):
        """
        Occupancy per seat and per zone in time buckets for one day.
        Occupancy = share of observations in a bucket where the seat was not
        available (occupied or held by belongings). Time = seat_time (the demo /
        video clock), falling back to the wall-clock timestamp.
        """
        bucket_min = max(5, min(240, int(bucket_min)))
        _, rows = self._read("desks")
        parsed = []
        for r in rows:
            t = r.get("seat_time") or r.get("timestamp")
            try:
                parsed.append((datetime.fromisoformat(t), r))
            except (TypeError, ValueError):
                continue
        days = sorted({t.date().isoformat() for t, _ in parsed})
        if not days:
            return {"day": None, "days": [], "bucket_min": bucket_min, "buckets": [],
                    "seats": [], "seat_matrix": {}, "seat_state": {}, "zones": [],
                    "zone_matrix": {}, "overall": [], "samples": [], "simulated": False}
        if day not in days:
            day = days[-1]
        parsed = [(t, r) for t, r in parsed if t.date().isoformat() == day]

        def floor(t):
            m = (t.hour * 60 + t.minute) // bucket_min * bucket_min
            return t.replace(hour=m // 60, minute=m % 60, second=0, microsecond=0)

        cells = defaultdict(Counter)        # (bucket, desk) -> state counts
        zcells = defaultdict(Counter)       # (bucket, zone) -> busy/total
        layout = {}
        for t, r in parsed:
            b = floor(t)
            did = r.get("desk_id") or "?"
            st = r.get("state", "")
            cells[(b, did)][st] += 1
            busy = st in ("OCCUPIED", "PARTIALLY_OCCUPIED")
            z = r.get("zone") or "Unzoned"
            zcells[(b, z)]["n"] += 1
            zcells[(b, z)]["busy"] += busy
            zcells[(b, "_all")]["n"] += 1
            zcells[(b, "_all")]["busy"] += busy
            layout[did] = {"id": did, "zone": z, "row": _int(r.get("row")), "col": _int(r.get("col"))}

        first = min(b for b, _ in cells)
        last = max(b for b, _ in cells)
        buckets = []
        b = first
        while b <= last:
            buckets.append(b)
            b += timedelta(minutes=bucket_min)

        seat_ids = sorted(layout, key=lambda d: (layout[d]["row"] or 0, layout[d]["col"] or 0, d))
        seat_matrix, seat_state = {}, {}
        for did in seat_ids:
            occ, dom = [], []
            for b in buckets:
                c = cells.get((b, did))
                if not c:
                    occ.append(None)
                    dom.append(None)
                    continue
                n = sum(c.values())
                occ.append(round((c["OCCUPIED"] + c["PARTIALLY_OCCUPIED"]) / n, 3))
                dom.append(c.most_common(1)[0][0])
            seat_matrix[did] = occ
            seat_state[did] = dom

        zone_names = sorted({v["zone"] for v in layout.values()})
        def ratio(b, z):
            c = zcells.get((b, z))
            return round(c["busy"] / c["n"], 3) if c and c["n"] else None
        zone_matrix = {z: [ratio(b, z) for b in buckets] for z in zone_names}
        overall = [ratio(b, "_all") for b in buckets]
        samples = [zcells.get((b, "_all"), Counter())["n"] for b in buckets]
        simulated = any((r.get("source") or "").upper().startswith("SIMULATED") for _, r in parsed)
        return {
            "day": day, "days": days, "bucket_min": bucket_min,
            "buckets": [b.strftime("%H:%M") for b in buckets],
            "seats": [layout[d] for d in seat_ids],
            "seat_matrix": seat_matrix, "seat_state": seat_state,
            "zones": zone_names, "zone_matrix": zone_matrix,
            "overall": overall, "samples": samples, "simulated": simulated,
        }


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
