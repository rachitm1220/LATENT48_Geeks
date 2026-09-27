# Sthāna (स्थान)

*Sanskrit: a place, a seat.* Real-time library desk availability from three vision models and one rule.

| Model | File | Runs on | Purpose |
|---|---|---|---|
| Desk detector | `models/weights_deskDetection.pt` | full frame | finds desks |
| Person detector | `models/yolov8n.pt` | full frame | finds people, each assigned to one desk |
| Object detector | `models/weights_objectDetection.pt` | desk crop (only desks with no person) | bag / book / bottle / charger / phone / laptop |

## Seat logic

```
person at desk                        -> OCCUPIED            red
no person, belongings on desk         -> PARTIALLY_OCCUPIED  orange (timer starts)
belongings unattended >= 30 min       -> RECLAIMABLE         green, dashed ("keep things aside and sit")
no person, no belongings              -> AVAILABLE           green
```

## Run it

```bash
cd Sthana
python -m venv venv
venv\Scripts\activate          # Windows  (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

Copy the three `.pt` files into `models/` and your images/videos into `data/`, then:

```bash
python app.py
```

Open http://localhost:5000

## Pages

| Page | URL | What it's for |
|---|---|---|
| Overview | `/` | Landing page: live counts, how it works, the 30-minute rule, latest capture |
| Live | `/monitor` | Upload images / play videos / camera, live seat list, demo controls, activity |
| History | `/history` | Grid of every saved run, searchable, filter Images / Video |
| Run detail | `/history/<run_id>` | Annotated + original frame, and for every desk: crop, object detections, state, box |
| Records | `/records` | The three CSVs as searchable tables with state filters, pagination and download |
| Map | `/map` | Live library map (Status / Heat / Graph), zone occupancy, time-lapse heatmap, spatial reasoning |

## What gets saved

Every analysis is a **run**:

```
outputs/
├── runs/<run_id>/
│   ├── original.jpg            frame as received
│   ├── annotated.jpg           desks coloured by state + person / object boxes
│   ├── thumb.jpg               small preview for History
│   ├── desks/D01_crop.jpg      cropped desk (what the object model sees)
│   ├── desks/D01_objects.jpg   the crop with object detections drawn
│   └── meta.json               all of the above as data
├── runs.csv                    one row per analysis (counts per state, people, inference ms)
├── desk_observations.csv       one row per desk per analysis (state, objects, box, timer, crop path)
└── events.csv                  one row per state change / demo action
```

- Images and folder images: every run saves images (`SAVE_RUN_IMAGES`).
- Video: CSV rows for every analyzed frame; images every `VIDEO_SAVE_EVERY_N`th frame **and** whenever any seat changes state.
- **Reset** on the Live page clears live seats only. History and CSVs are kept. To start fresh, delete the `outputs/` folder.
- CSVs are read whole on each Records request, which is fine for tens of thousands of rows (a long demo day).

## Demo: orange -> green in one minute

1. Live → analyze an image where a desk has a bag but nobody sitting. It shows orange with a 30:00 countdown.
2. Demo controls → Belongings timer: pick that seat (or click its card), enter `29`, **Set timer**.
3. Wait one minute: the seat turns green (dashed), a banner and a notification appear, and it's logged in Activity and `events.csv`.

Or use **+1 / +5 / +30 min**, or set the grace period to 1 minute.

## Library map, heatmaps and spatial reasoning

**Live map.** Seats are laid out as a grid in the order the camera sees them (rows by depth, left to right).
- *Status*: green / orange / red, dashed = reclaimable, hatched = hidden this frame (inferred), dot = uncertain.
- *Heat*: occupancy intensity per seat (occupied 100%, held 70%, reclaimable 30%, free 0%) on a blue scale.
- *Graph*: seats as nodes, neighbour edges, colour = the spatial model's estimate.
- **Best seat now**: the free seat with the quietest neighbourhood.

**Zones.** Set rectangles in `config.py` (`ZONES`, normalised camera coordinates) or leave `None` for three vertical strips.
Rename the room with `FLOOR_NAME`. One camera = one floor/room.

**Time-lapse.** Built from `desk_observations.csv` using `seat_time` (the seat clock), bucketed per 15 / 30 / 60 min:
slider + play, replay grid, occupancy line, zone x time and seat x time heatmaps. To get real times of day:
- Video: set **Recording starts at** (e.g. 08:00) with *Timers follow video time* on. An 8-hour recording then fills 08:00-16:00.
- Images: Live → Demo controls → **Set clock** 08:00, analyze; Set clock 10:00, analyze; and so on.
- Demo only: `python simulate_day.py` writes a clearly tagged "SIMULATED demo day" (the Map page says so).
  Remove it with `python simulate_day.py --remove`. Stop the app before running it.

**Spatial reasoning (GNN)** — `graph.py`. Each seat is a node; edges join the k nearest seats within a radius measured
in desk sizes (perspective-aware), weight exp(-(d/σ)²). Node input = detector observation weighted by its evidence
confidence, blended with the seat's previous estimate. Two message-passing rounds:
`H ← softmax(log H + β·(1−c)·(Â·H·W))`, W = 3x3 class-compatibility matrix (busy areas cluster).
The (1−c) gate means confident detections barely move; uncertain or hidden seats lean on neighbours and history.
It adds: *inferred* seats when a desk is occluded, *uncertain*/*disagree* flags on weak detections, and the quiet-seat
recommendation. It never overrides a confident detection or the 30-minute timer. Weights are hand-set; with labelled
sequences they can be fitted with cross-entropy on the same forward pass. Tune in `config.py` (`GNN_*`).

Evidence confidence comes from the detectors: person confidence x how clearly they overlap the desk, object
confidence for belongings, and for "empty" desks it drops when a faint (0.25-0.5) bag/book is seen (`OBJECT_WEAK_CONF`).

## Video

Live → **Video** tab: upload a video or pick one from `data/`, or choose **Live camera** (`CAMERA_SOURCE` in `config.py`).
Options: analyze every N seconds, speed 1–30×, timers follow video time, loop, reset seats on start.
For video set `EMPTY_CONFIRM_FRAMES = 2` so one missed detection doesn't reset a timer.

## Design notes

- **Apple-inspired system**: SF Pro (Inter as fallback), large tight headlines, light body copy, monochrome surfaces, one accent blue. Green / orange / red appear only where they mean seat state.
- **Soft depth, no hard borders**: cards use layered soft shadows; the nav becomes translucent glass on scroll.
- **Scroll storytelling** on the Overview: fade/slide reveals, subtle parallax on the floor plan, and a sticky section where scrolling drives the 30-minute timeline until the seat turns green.
- **Bento grid** for features; pill buttons with a press scale and ripple.
- **Dark mode**: Apple-style switch, follows the system until chosen, remembered per browser, smooth colour transition.
- **Accessible**: semantic landmarks, skip link, visible focus rings, ARIA on tabs / switches / menu, AA-contrast text colours ("ink" variants of the status colours), `prefers-reduced-motion` respected everywhere.
- **Light**: no framework or build step. Plain Flask templates + about 50 KB of CSS/JS, lazy-loaded thumbnails.

## API

| Method | Endpoint | Notes |
|---|---|---|
| GET | `/api/state` | live seats, timers, events, frame, video status, last run |
| GET | `/api/runs?q=&kind=&page=&per_page=` | saved runs + stats |
| GET | `/api/runs/<run_id>` | meta.json with image URLs, prev/next |
| GET | `/runs/<run_id>/<file>` | any saved image / meta.json |
| GET | `/api/records/<desks\|runs\|events>?q=&state=&page=&per_page=` | CSV rows, newest first |
| GET | `/download/<desks\|runs\|events>.csv` | download a CSV |
| GET | `/api/frame`, `/api/annotated`, `/api/images` | latest frame, rendered overlay, data folder listing |
| POST | `/api/upload`, `/api/process`, `/api/process_next`, `/api/camera` | image input |
| POST | `/api/video/upload`, `/api/video/start`, `/api/video/pause`, `/resume`, `/stop` | video input |
| POST | `/api/desks/<id>/elapsed`, `/api/settings`, `/api/clock/advance`, `/api/clock/set`, `/api/reset` | demo controls |
| GET | `/api/timeline?bucket=60&day=YYYY-MM-DD` | time-lapse data: buckets, seat/zone matrices, overall |

## Files

```
app.py          Flask routes (pages + API)
pipeline.py     desk + person + object detection -> per-desk observation
seat_state.py   seat tracking, 30-min timer, demo clock, events
storage.py      runs, crops, CSVs, history/records queries
viz.py          OpenCV drawing for saved images
video.py        background video / camera player
layout.py       seat grid (rows/cols) and zones
graph.py        spatial reasoning: seat graph + message passing
simulate_day.py writes / removes a clearly tagged simulated day for the time-lapse demo
config.py       paths, thresholds, branding (APP_NAME)
templates/      base, home, monitor, map, history, run, records, 404
static/css/     site.css (design system), pages.css (page layouts)
static/js/      site.js (theme, nav, reveals), one script per page
```

Rename the project by changing `APP_NAME` / `APP_NAME_NATIVE` / `APP_TAGLINE` in `config.py`.
