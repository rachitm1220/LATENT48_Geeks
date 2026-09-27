"""
Central configuration for Sthana (library seat monitor).
Edit the paths / thresholds here; nothing else needs to change.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# BRANDING  (shown in the website header, footer and titles)
# ------------------------------------------------------------
APP_NAME = "Sthāna"
APP_NAME_NATIVE = "स्थान"                # Sanskrit: place, seat, position
APP_TAGLINE = "Every seat. Seen."

# ------------------------------------------------------------
# MODEL WEIGHTS  (copy your .pt files into seat_monitor/models/)
# ------------------------------------------------------------
MODEL_DIR = BASE_DIR / "models"
DESK_MODEL_PATH = MODEL_DIR / "weights_deskDetection.pt"
PERSON_MODEL_PATH = MODEL_DIR / "yolov8n.pt"            # COCO model, "person" class
OBJECT_MODEL_PATH = MODEL_DIR / "weights_objectDetection.pt"

# ------------------------------------------------------------
# FOLDERS
# ------------------------------------------------------------
IMAGE_DIR = BASE_DIR / "data"          # sample frames for the demo ("camera feed")
# Every analysis ("run") is stored here:
#   outputs/runs/<run_id>/  original.jpg, annotated.jpg, thumb.jpg,
#                           desks/<id>_crop.jpg, desks/<id>_objects.jpg, meta.json
#   outputs/runs.csv, outputs/desk_observations.csv, outputs/events.csv
OUTPUT_DIR = BASE_DIR / "outputs"
SAVE_RUN_IMAGES = True                 # save images for image uploads / folder images

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Optional live camera: 0 = default webcam, or an RTSP/HTTP URL, or a video file path
CAMERA_SOURCE = 0

# ------------------------------------------------------------
# VIDEO INPUT
# ------------------------------------------------------------
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}
UPLOAD_DIR = BASE_DIR / "uploads"      # uploaded videos are saved here
VIDEO_SAMPLE_SECONDS = 2.0             # analyze one frame every N seconds of video
VIDEO_SPEED = 1.0                      # playback speed (5 = five times faster than real time)
VIDEO_SYNC_CLOCK = True                # seat timers follow video time instead of wall time
VIDEO_LOOP = False
# Video frames: CSV rows are written for EVERY analyzed frame, but images are
# saved only every Nth frame and whenever any seat changes state (keeps disk use sane).
VIDEO_SAVE_EVERY_N = 10

# ------------------------------------------------------------
# DETECTION THRESHOLDS
# ------------------------------------------------------------
IMAGE_SIZE = 640
DESK_CONF = 0.25
PERSON_CONF = 0.25
OBJECT_CONF = 0.50
# Detections between OBJECT_WEAK_CONF and OBJECT_CONF don't change the seat state,
# they only lower its evidence confidence (used by the spatial model).
OBJECT_WEAK_CONF = 0.25
MAX_OBJECTS_PER_DESK = 5

# Only keep these desk-model classes (None = keep every class the desk model outputs)
DESK_CLASS_NAMES = None

# Class-id -> name for the object model (same table as objectDetection_rachit.py).
# The model's own names are just "1", "2", ... so we map by class id instead.
# Set to None to use the names stored inside the .pt file.
OBJECT_CLASS_LABELS = {
    0: "bag",
    1: "book",
    2: "bottle",
    3: "chair",
    4: "charger",
    5: "phone",
    6: "human",
    7: "laptop",
}

# Object-model labels that count as "belongings" (chair is ignored on purpose)
BELONGING_CLASSES = {"bag", "book", "bottle", "charger", "phone", "laptop"}
# Object-model labels that mean a person is at the desk
PERSON_CLASSES_IN_OBJECT_MODEL = {"human", "person"}

# ------------------------------------------------------------
# PERSON <-> DESK ASSOCIATION
# ------------------------------------------------------------
# The desk box is grown by this fraction on every side before checking overlap,
# because a seated person usually sits just in front of / beside the desk top.
DESK_PAD_RATIO = 0.15
# overlap = intersection / min(person_area, padded_desk_area)
PERSON_DESK_OVERLAP = 0.30

# ------------------------------------------------------------
# SEAT TRACKING + TIMER
# ------------------------------------------------------------
# A detected desk is matched to an existing seat ID when IoU >= this (fixed camera)
DESK_MATCH_IOU = 0.30
# Minutes a desk may hold belongings (no person) before it becomes reclaimable
GRACE_PERIOD_MINUTES = 30
# Consecutive EMPTY frames needed before a running belongings timer is cleared
# (protects against one frame where the bag is missed). 1 = clear immediately.
EMPTY_CONFIRM_FRAMES = 1

# ------------------------------------------------------------
# LIBRARY MAP + ZONES
# ------------------------------------------------------------
FLOOR_NAME = "Reading Hall"            # shown on the Map page
# Zones are rectangles in normalised camera coordinates (0..1). A seat belongs to the
# first zone that contains its centre. None = split the view into equal vertical strips.
ZONES = None
# ZONES = [
#     {"name": "Window side", "x1": 0.00, "y1": 0.0, "x2": 0.40, "y2": 1.0},
#     {"name": "Centre",      "x1": 0.40, "y1": 0.0, "x2": 0.70, "y2": 1.0},
#     {"name": "Stacks",      "x1": 0.70, "y1": 0.0, "x2": 1.00, "y2": 1.0},
# ]
AUTO_ZONE_NAMES = ["Zone A", "Zone B", "Zone C"]
# Time-lapse heatmap: default bucket size in minutes (15, 30 or 60)
HEATMAP_BUCKET_MINUTES = 60

# ------------------------------------------------------------
# SPATIAL REASONING (graph model over neighbouring seats)
# ------------------------------------------------------------
GNN_ENABLED = True
GNN_NEIGHBOURS = 4          # k nearest seats per node
GNN_RADIUS = 2.6            # max edge length, in desk sizes (perspective-aware)
GNN_LAYERS = 2              # message-passing rounds
GNN_STRENGTH = 1.4          # how strongly neighbours move an uncertain seat
GNN_TEMPORAL = 0.6          # weight of the seat's own previous estimate
UNCERTAIN_EVIDENCE = 0.5    # below this, a seat is flagged "uncertain"
# A desk hidden in the latest frame (e.g. someone walking past) stays on the map as
# "inferred" for this many frames, with its state estimated from history + neighbours.
INFERRED_MAX_MISSED = 5

# ------------------------------------------------------------
# SERVER
# ------------------------------------------------------------
HOST = "0.0.0.0"
PORT = 5000
