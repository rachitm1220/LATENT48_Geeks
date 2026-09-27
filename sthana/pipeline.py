"""
Perception pipeline:  frame -> per-desk observation

    1. Desk model   (full frame)  -> desk boxes
    2. Person model (full frame)  -> person boxes, each assigned to at most one desk
    3. For desks WITHOUT a person: crop the desk and run the object model
         - "human" in crop           -> PERSON_PRESENT
         - any belonging class       -> BELONGINGS_ONLY
         - nothing relevant          -> EMPTY

The pipeline only reports what the camera sees. Time-based decisions
(30-minute rule) live in seat_state.py.
"""
import threading
from pathlib import Path

from ultralytics import YOLO

import config

OBS_PERSON = "PERSON_PRESENT"
OBS_BELONGINGS = "BELONGINGS_ONLY"
OBS_EMPTY = "EMPTY"


# ------------------------------------------------------------
# Box helpers  (boxes are [x1, y1, x2, y2] ints)
# ------------------------------------------------------------
def box_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def intersection_area(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def clamp_box(b, w, h):
    return [
        max(0, min(int(b[0]), w)),
        max(0, min(int(b[1]), h)),
        max(0, min(int(b[2]), w)),
        max(0, min(int(b[3]), h)),
    ]


def pad_box(b, ratio, w, h):
    px = int((b[2] - b[0]) * ratio)
    py = int((b[3] - b[1]) * ratio)
    return clamp_box([b[0] - px, b[1] - py, b[2] + px, b[3] + py], w, h)


def _load_model(path, name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{name} weights not found: {path}\n"
            f"Copy the .pt file there or change the path in config.py"
        )
    print(f"Loading {name}: {path}", flush=True)
    return YOLO(str(path))


class SeatPipeline:
    def __init__(self):
        self.desk_model = _load_model(config.DESK_MODEL_PATH, "desk model")
        self.person_model = _load_model(config.PERSON_MODEL_PATH, "person model")
        self.object_model = _load_model(config.OBJECT_MODEL_PATH, "object model")

        self.person_class_ids = [
            cid for cid, n in self.person_model.names.items() if str(n).lower() == "person"
        ]
        if not self.person_class_ids:
            raise ValueError("Person model has no 'person' class")

        # Object model names: use the id->name table from config when given,
        # falling back to the names stored in the weights for unknown ids.
        self.object_names = dict(self.object_model.names)
        if config.OBJECT_CLASS_LABELS:
            self.object_names.update(config.OBJECT_CLASS_LABELS)

        # YOLO models are not guaranteed thread-safe; Flask is threaded.
        self.lock = threading.Lock()

        print(f"Desk classes:   {self.desk_model.names}", flush=True)
        print(f"Person class id: {self.person_class_ids}", flush=True)
        print(f"Object classes (in weights): {self.object_model.names}", flush=True)
        print(f"Object classes (used):       {self.object_names}", flush=True)

    # --------------------------------------------------------
    @staticmethod
    def _parse(result, names, w, h, allowed_ids=None):
        out = []
        if result.boxes is None:
            return out
        for box in result.boxes:
            cid = int(box.cls[0])
            if allowed_ids is not None and cid not in allowed_ids:
                continue
            b = clamp_box(box.xyxy[0].tolist(), w, h)
            if b[2] <= b[0] or b[3] <= b[1]:
                continue
            out.append({
                "box": b,
                "confidence": round(float(box.conf[0]), 3),
                "label": str(names[cid]).lower(),
                "class_id": cid,
            })
        return out

    def detect_desks(self, image):
        h, w = image.shape[:2]
        r = self.desk_model.predict(
            source=image, conf=config.DESK_CONF, imgsz=config.IMAGE_SIZE, verbose=False
        )[0]
        desks = self._parse(r, self.desk_model.names, w, h)
        if config.DESK_CLASS_NAMES:
            wanted = {n.lower() for n in config.DESK_CLASS_NAMES}
            desks = [d for d in desks if d["label"] in wanted]
        return desks

    def detect_persons(self, image):
        h, w = image.shape[:2]
        r = self.person_model.predict(
            source=image, conf=config.PERSON_CONF, imgsz=config.IMAGE_SIZE,
            classes=self.person_class_ids, verbose=False,
        )[0]
        return self._parse(r, self.person_model.names, w, h, set(self.person_class_ids))

    def detect_objects(self, crop, offset_x, offset_y):
        """
        Run the object model on a desk crop (boxes returned in full-frame coordinates).
        Returns (objects, weak): objects >= OBJECT_CONF drive the seat state as before;
        weak ones (OBJECT_WEAK_CONF..OBJECT_CONF) only lower the evidence confidence,
        which tells the spatial model this seat is uncertain.
        """
        h, w = crop.shape[:2]
        low = min(config.OBJECT_WEAK_CONF, config.OBJECT_CONF)
        r = self.object_model.predict(
            source=crop, conf=low, imgsz=config.IMAGE_SIZE, verbose=False
        )[0]
        objs = self._parse(r, self.object_names, w, h)
        objs.sort(key=lambda o: o["confidence"], reverse=True)
        for o in objs:
            b = o["box"]
            o["box"] = [b[0] + offset_x, b[1] + offset_y, b[2] + offset_x, b[3] + offset_y]
        strong = [o for o in objs if o["confidence"] >= config.OBJECT_CONF]
        weak = [o for o in objs if o["confidence"] < config.OBJECT_CONF]
        return strong[: config.MAX_OBJECTS_PER_DESK], weak[: config.MAX_OBJECTS_PER_DESK]

    # --------------------------------------------------------
    def analyze(self, image):
        with self.lock:
            h, w = image.shape[:2]
            desks = self.detect_desks(image)
            persons = self.detect_persons(image)

            # ---- assign each person to the single best-overlapping desk
            padded = [pad_box(d["box"], config.DESK_PAD_RATIO, w, h) for d in desks]
            person_overlap = [0.0] * len(desks)
            person_evidence = [0.0] * len(desks)
            for p in persons:
                p["desk_index"] = None
                best_i, best_score = None, 0.0
                for i, pb in enumerate(padded):
                    inter = intersection_area(p["box"], pb)
                    denom = max(1, min(box_area(p["box"]), box_area(pb)))
                    score = inter / denom
                    if score > best_score:
                        best_i, best_score = i, score
                if best_i is not None and best_score >= config.PERSON_DESK_OVERLAP:
                    p["desk_index"] = best_i
                    person_overlap[best_i] = max(person_overlap[best_i], best_score)
                    # evidence: detector confidence x how clearly the person sits at this desk
                    ev = p["confidence"] * min(1.0, best_score / 0.6)
                    person_evidence[best_i] = max(person_evidence[best_i], ev)

            # ---- classify each desk
            results = []
            relevant = config.BELONGING_CLASSES | config.PERSON_CLASSES_IN_OBJECT_MODEL
            for i, d in enumerate(desks):
                objects, weak = [], []
                if person_overlap[i] > 0:
                    obs = OBS_PERSON
                    reason = f"person detected (overlap {person_overlap[i]:.2f})"
                    evidence = person_evidence[i]
                else:
                    x1, y1, x2, y2 = d["box"]
                    crop = image[y1:y2, x1:x2]
                    if crop.size:
                        objects, weak = self.detect_objects(crop, x1, y1)
                    humans = [o for o in objects if o["label"] in config.PERSON_CLASSES_IN_OBJECT_MODEL]
                    belongings = [o for o in objects if o["label"] in config.BELONGING_CLASSES]
                    weak_rel = [o for o in weak if o["label"] in relevant]
                    if humans:
                        obs = OBS_PERSON
                        reason = "person detected in desk crop"
                        evidence = max(o["confidence"] for o in humans)
                    elif belongings:
                        obs = OBS_BELONGINGS
                        reason = "belongings: " + ", ".join(sorted({o["label"] for o in belongings}))
                        evidence = max(o["confidence"] for o in belongings)
                    else:
                        obs = OBS_EMPTY
                        reason = "no person or belongings"
                        # a faint detection (e.g. a 0.4 bag) makes "empty" less certain
                        faint = max((o["confidence"] for o in weak_rel), default=0.0)
                        evidence = 0.9 * (1.0 - faint / max(config.OBJECT_CONF, 1e-6))
                        if weak_rel:
                            reason += f" (faint: {', '.join(sorted({o['label'] for o in weak_rel}))})"

                # a shaky desk detection lowers confidence a little too
                evidence *= 0.6 + 0.4 * d["confidence"]
                results.append({
                    "box": d["box"],
                    "confidence": d["confidence"],
                    "observation": obs,
                    "reason": reason,
                    "objects": objects,
                    "weak_objects": [{"label": o["label"], "confidence": o["confidence"]} for o in weak],
                    "evidence": round(max(0.0, min(1.0, evidence)), 3),
                })

            return {"desks": results, "persons": persons, "image_size": [w, h]}
