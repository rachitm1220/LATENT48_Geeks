"""
Drawing helpers (OpenCV, BGR colours).

draw_annotated()      full frame: desk boxes coloured by seat state, person and object boxes
draw_crop_objects()   one desk crop with the object-model detections drawn on it
"""
import cv2

from seat_state import PARTIAL, RECLAIMABLE, STATE_META

FONT = cv2.FONT_HERSHEY_SIMPLEX
OBJECT_BGR = (10, 214, 255)     # #ffd60a  yellow
PERSON_BGR = (255, 132, 10)     # #0a84ff  blue
WHITE = (255, 255, 255)
INK = (29, 29, 29)


def _tag(img, text, x, y, bg, scale, thick, fg=WHITE):
    """Filled label whose bottom-left corner sits at (x, y); kept inside the image."""
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    pad = max(3, int(th * 0.4))
    h, w = img.shape[:2]
    x = max(0, min(int(x), w - tw - 2 * pad))
    top = max(0, int(y) - th - 2 * pad)
    cv2.rectangle(img, (x, top), (x + tw + 2 * pad, top + th + 2 * pad), bg, -1)
    cv2.putText(img, text, (x + pad, top + th + pad), FONT, scale, fg, thick, cv2.LINE_AA)


def _dashed_rect(img, p1, p2, color, thick, dash=18):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thick)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thick)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thick)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thick)


def draw_annotated(image, snapshot, persons):
    """Full frame with every visible desk coloured by its current seat state."""
    img = image.copy()
    h, w = img.shape[:2]
    thick = max(2, w // 450)
    scale = max(0.55, w / 1800)
    desks = [d for d in snapshot["desks"] if d.get("visible", True)]

    # translucent fill for all desks in one blend
    overlay = img.copy()
    for d in desks:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), STATE_META[d["state"]]["bgr"], -1)
    img = cv2.addWeighted(overlay, 0.16, img, 0.84, 0)

    for p in persons:
        x1, y1, x2, y2 = p["box"]
        _dashed_rect(img, (x1, y1), (x2, y2), PERSON_BGR, max(1, thick // 2 + 1), dash=max(8, w // 150))

    for d in desks:
        x1, y1, x2, y2 = d["box"]
        color = STATE_META[d["state"]]["bgr"]
        if d["state"] == RECLAIMABLE:
            _dashed_rect(img, (x1, y1), (x2, y2), color, thick * 2, dash=max(10, w // 90))
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thick * 2)

        for o in d.get("objects", []):
            ox1, oy1, ox2, oy2 = o["box"]
            cv2.rectangle(img, (ox1, oy1), (ox2, oy2), OBJECT_BGR, thick)

        text = f"{d['id']}  {d['label']}"
        if d["state"] == PARTIAL and d.get("remaining_s") is not None:
            m, s = divmod(int(d["remaining_s"]), 60)
            text += f"  {m:02d}:{s:02d}"
        _tag(img, text, x1, y1, color, scale, 2)
    return img


def draw_crop_objects(crop, objects, offset_x, offset_y, state, note=""):
    """Desk crop with object detections; the frame border shows the seat state."""
    img = crop.copy()
    h, w = img.shape[:2]
    scale = max(0.4, min(0.9, w / 650))
    thick = max(1, round(w / 350))

    for o in objects:
        x1, y1, x2, y2 = o["box"]
        x1, x2 = x1 - offset_x, x2 - offset_x
        y1, y2 = y1 - offset_y, y2 - offset_y
        cv2.rectangle(img, (x1, y1), (x2, y2), OBJECT_BGR, thick + 1)
        _tag(img, f"{o['label']} {o['confidence']:.2f}", x1, y1, OBJECT_BGR, scale, 1, fg=INK)

    color = STATE_META.get(state, STATE_META["UNKNOWN"])["bgr"]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, max(3, thick * 3))

    if note:
        (tw, th), _ = cv2.getTextSize(note, FONT, scale * 0.9, 1)
        cv2.rectangle(img, (0, h - th - 14), (min(w, tw + 16), h), (0, 0, 0), -1)
        cv2.putText(img, note, (8, h - 7), FONT, scale * 0.9, WHITE, 1, cv2.LINE_AA)
    return img
