"""
Temporal seat-state engine.

Observation (from pipeline)      Seat state (shown on dashboard)
---------------------------      --------------------------------
PERSON_PRESENT              ->   OCCUPIED            red
BELONGINGS_ONLY  < grace    ->   PARTIALLY_OCCUPIED  orange   (timer running)
BELONGINGS_ONLY  >= grace   ->   RECLAIMABLE         green    "keep things aside and sit"
EMPTY                       ->   AVAILABLE           green

The timer starts the first time a seat is seen with belongings only, and is
cleared as soon as a person is seen again (or the desk becomes empty).
States are re-derived from the clock on every read, so an orange seat turns
green on its own without needing a new camera frame.

The clock is a SimClock so the demo can fast-forward time or set a seat's
timer directly (e.g. to 29 minutes).
"""
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import layout
from pipeline import OBS_BELONGINGS, OBS_EMPTY, OBS_PERSON

OBS_INDEX = {OBS_PERSON: 0, OBS_BELONGINGS: 1, OBS_EMPTY: 2}
GNN_TO_STATE = {"occupied": "OCCUPIED", "held": "PARTIALLY_OCCUPIED", "free": "AVAILABLE"}
# how much each displayed state contributes to the occupancy heat (0 = free, 1 = full)
HEAT = {"OCCUPIED": 1.0, "PARTIALLY_OCCUPIED": 0.7, "RECLAIMABLE": 0.3, "AVAILABLE": 0.0}

AVAILABLE = "AVAILABLE"
OCCUPIED = "OCCUPIED"
PARTIAL = "PARTIALLY_OCCUPIED"
RECLAIMABLE = "RECLAIMABLE"
UNKNOWN = "UNKNOWN"

STATE_META = {
    AVAILABLE: {
        "color": "green", "hex": "#34c759", "bgr": (89, 199, 52),
        "label": "Available",
        "message": "Seat is free.",
    },
    OCCUPIED: {
        "color": "red", "hex": "#ff3b30", "bgr": (48, 59, 255),
        "label": "Occupied",
        "message": "A person is using this seat.",
    },
    PARTIAL: {
        "color": "orange", "hex": "#ff9500", "bgr": (0, 149, 255),
        "label": "Partially occupied",
        "message": "Belongings on the desk, owner is away. Please wait for the timer.",
    },
    RECLAIMABLE: {
        "color": "green", "hex": "#34c759", "bgr": (89, 199, 52),
        "label": "Available (reclaimable)",
        "message": "This seat can be occupied now by keeping the things aside.",
    },
    UNKNOWN: {
        "color": "grey", "hex": "#8e8e93", "bgr": (147, 142, 142),
        "label": "Unknown",
        "message": "No observation yet.",
    },
}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class SimClock:
    """Wall clock plus an adjustable offset (for the demo fast-forward)."""

    def __init__(self):
        self._offset = 0.0
        self._lock = threading.Lock()

    def now(self):
        with self._lock:
            return time.time() + self._offset

    def advance(self, seconds):
        with self._lock:
            self._offset += seconds

    def reset(self):
        with self._lock:
            self._offset = 0.0

    def set_now(self, ts):
        """Make now() return ts (the clock keeps running from there)."""
        with self._lock:
            self._offset = ts - time.time()

    @property
    def offset(self):
        with self._lock:
            return self._offset


@dataclass
class Desk:
    desk_id: str
    box: list
    confidence: float = 0.0
    observation: str = UNKNOWN
    reason: str = ""
    objects: list = field(default_factory=list)
    belongings_since: float = None
    empty_streak: int = 0
    visible: bool = True
    last_seen: float = 0.0
    last_state: str = UNKNOWN
    evidence: float = 0.0          # detector confidence in the current observation
    weak_objects: list = field(default_factory=list)
    missed: int = 0                # consecutive frames this desk was not detected
    gnn: dict = None               # latest spatial-model output


class SeatManager:
    def __init__(self, grace_minutes=30, match_iou=0.3, empty_confirm_frames=1, on_event=None,
                 gnn=None, inferred_max_missed=5):
        self.on_event = on_event          # callback(event_dict), e.g. write to events.csv
        self.gnn = gnn                    # graph.SpatialGNN or None
        self.inferred_max_missed = inferred_max_missed
        self.frame_size = None
        self.clock = SimClock()
        self.lock = threading.RLock()
        self.grace_seconds = grace_minutes * 60
        self.match_iou = match_iou
        self.empty_confirm_frames = max(1, int(empty_confirm_frames))
        self.desks = {}
        self._next_id = 1
        self.events = deque(maxlen=200)
        self._event_id = 0

    # --------------------------------------------------------
    def _log(self, desk_id, text, state=None, now=None):
        self._event_id += 1
        ts = now if now is not None else self.clock.now()
        event = {
            "id": self._event_id,
            "time": datetime.fromtimestamp(ts).strftime("%H:%M:%S"),
            "desk": desk_id,
            "state": state,
            "text": text,
        }
        self.events.append(event)
        if self.on_event:
            try:
                self.on_event(event)
            except Exception as e:  # logging must never break seat tracking
                print(f"[events] could not log event: {e}", flush=True)

    def _new_desk(self, box):
        desk_id = f"D{self._next_id:02d}"
        self._next_id += 1
        desk = Desk(desk_id=desk_id, box=box)
        self.desks[desk_id] = desk
        self._log(desk_id, f"{desk_id} registered")
        return desk

    # --------------------------------------------------------
    def update(self, analysis):
        """Merge one frame's pipeline output. Adds 'desk_id' to each detection."""
        with self.lock:
            now = self.clock.now()
            unmatched = set(self.desks)
            for det in sorted(analysis["desks"], key=lambda d: -d["confidence"]):
                best_id, best_iou = None, 0.0
                for did in unmatched:
                    v = iou(det["box"], self.desks[did].box)
                    if v > best_iou:
                        best_id, best_iou = did, v
                if best_id is not None and best_iou >= self.match_iou:
                    desk = self.desks[best_id]
                    unmatched.discard(best_id)
                else:
                    desk = self._new_desk(det["box"])

                desk.box = det["box"]
                desk.confidence = det["confidence"]
                desk.objects = det["objects"]
                desk.reason = det.get("reason", "")
                desk.evidence = det.get("evidence", 1.0)
                desk.weak_objects = det.get("weak_objects", [])
                desk.visible = True
                desk.missed = 0
                desk.last_seen = now
                det["desk_id"] = desk.desk_id
                self._observe(desk, det["observation"], now)

            for did in unmatched:
                self.desks[did].visible = False
                self.desks[did].missed += 1

            self.frame_size = analysis.get("image_size") or self.frame_size
            self._refresh_states(now)
            self._run_gnn()

    # --------------------------------------------------------
    # Spatial reasoning
    # --------------------------------------------------------
    def _on_map(self, d):
        return d.visible or d.missed <= self.inferred_max_missed

    def _run_gnn(self):
        """Refine every on-map seat with the graph model (once per analyzed frame)."""
        if self.gnn is None:
            return
        nodes = []
        for d in self.desks.values():
            if not self._on_map(d):
                d.gnn = None
                continue
            x1, y1, x2, y2 = d.box
            nodes.append({
                "id": d.desk_id,
                "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                "size": [max(1, x2 - x1), max(1, y2 - y1)],
                "obs": OBS_INDEX.get(d.observation),
                "evidence": d.evidence,
                "visible": d.visible,
                "prev": d.gnn["p"] if d.gnn else None,
            })
        for did, result in self.gnn.run(nodes).items():
            self.desks[did].gnn = result

    def _observe(self, desk, obs, now):
        if obs == OBS_PERSON:
            desk.belongings_since = None
            desk.empty_streak = 0
        elif obs == OBS_BELONGINGS:
            desk.empty_streak = 0
            if desk.belongings_since is None:
                desk.belongings_since = now
        elif obs == OBS_EMPTY:
            desk.empty_streak += 1
            if desk.belongings_since is not None and desk.empty_streak < self.empty_confirm_frames:
                return  # debounce: keep the belongings timer for now
            desk.belongings_since = None
        desk.observation = obs

    # --------------------------------------------------------
    def _derive(self, desk, now):
        if desk.observation == OBS_PERSON:
            return OCCUPIED
        if desk.observation == OBS_BELONGINGS:
            since = desk.belongings_since if desk.belongings_since is not None else now
            return RECLAIMABLE if now - since >= self.grace_seconds else PARTIAL
        if desk.observation == OBS_EMPTY:
            return AVAILABLE
        return UNKNOWN

    def _refresh_states(self, now):
        for desk in self.desks.values():
            state = self._derive(desk, now)
            if state != desk.last_state:
                if state == RECLAIMABLE:
                    text = (f"{desk.desk_id}: belongings unattended for "
                            f"{self.grace_seconds / 60:g} min. This seat can be occupied "
                            f"now by keeping the things aside.")
                elif state == PARTIAL:
                    text = (f"{desk.desk_id}: belongings only, owner away. "
                            f"Timer started ({self.grace_seconds / 60:g} min).")
                elif state == OCCUPIED:
                    text = f"{desk.desk_id}: person seated."
                elif state == AVAILABLE:
                    text = f"{desk.desk_id}: seat is empty and available."
                else:
                    text = f"{desk.desk_id}: {state}"
                self._log(desk.desk_id, text, state, now)
                desk.last_state = state

    # --------------------------------------------------------
    # Demo / admin controls
    # --------------------------------------------------------
    def set_elapsed(self, desk_id, minutes):
        """Pretend the belongings have already been unattended for `minutes`."""
        with self.lock:
            desk = self.desks.get(desk_id)
            if desk is None:
                raise KeyError(f"Unknown desk {desk_id}")
            if desk.observation != OBS_BELONGINGS:
                raise ValueError(
                    f"{desk_id} is not partially occupied (orange); "
                    f"the timer only runs while only belongings are on the desk."
                )
            now = self.clock.now()
            desk.belongings_since = now - float(minutes) * 60
            self._log(desk_id, f"Demo: {desk_id} timer set to {float(minutes):g} min elapsed", now=now)
            self._refresh_states(now)

    def set_grace(self, minutes):
        minutes = float(minutes)
        if minutes <= 0:
            raise ValueError("Grace period must be > 0")
        with self.lock:
            self.grace_seconds = minutes * 60
            self._log(None, f"Grace period set to {minutes:g} min")
            self._refresh_states(self.clock.now())

    def advance_clock(self, minutes):
        with self.lock:
            self.clock.advance(float(minutes) * 60)
            self._log(None, f"Demo: clock fast-forwarded {float(minutes):g} min")
            self._refresh_states(self.clock.now())

    def set_clock_time(self, hhmm):
        """Demo: set the seat clock to today's HH:MM (e.g. "08:00") for time-lapse demos."""
        try:
            h, m = (int(x) for x in str(hhmm).strip().split(":")[:2])
            if not (0 <= h < 24 and 0 <= m < 60):
                raise ValueError
        except ValueError:
            raise ValueError("Time must look like 08:00")
        with self.lock:
            target = datetime.fromtimestamp(self.clock.now()).replace(
                hour=h, minute=m, second=0, microsecond=0)
            self.clock.set_now(target.timestamp())
            self._log(None, f"Demo: clock set to {h:02d}:{m:02d}")
            self._refresh_states(self.clock.now())

    def reset(self):
        with self.lock:
            self.desks.clear()
            self._next_id = 1
            self.events.clear()
            self.clock.reset()
            self._log(None, "System reset")

    # --------------------------------------------------------
    def snapshot(self):
        with self.lock:
            now = self.clock.now()
            self._refresh_states(now)
            desks, counts = [], {AVAILABLE: 0, OCCUPIED: 0, PARTIAL: 0, RECLAIMABLE: 0, UNKNOWN: 0}
            for desk_id in sorted(self.desks):
                d = self.desks[desk_id]
                if not d.visible:
                    continue  # only desks seen in the latest frame
                state = d.last_state
                counts[state] += 1
                meta = STATE_META[state]
                elapsed = remaining = progress = None
                if d.observation == OBS_BELONGINGS and d.belongings_since is not None:
                    elapsed = max(0.0, now - d.belongings_since)
                    remaining = max(0.0, self.grace_seconds - elapsed)
                    progress = min(1.0, elapsed / self.grace_seconds)
                desks.append({
                    "id": d.desk_id,
                    "box": d.box,
                    "confidence": d.confidence,
                    "visible": d.visible,
                    "observation": d.observation,
                    "reason": d.reason,
                    "state": state,
                    "label": meta["label"],
                    "color": meta["color"],
                    "hex": meta["hex"],
                    "message": meta["message"],
                    "objects": [{"label": o["label"], "confidence": o["confidence"], "box": o["box"]}
                                for o in d.objects],
                    "elapsed_s": elapsed,
                    "remaining_s": remaining,
                    "progress": progress,
                    "evidence": d.evidence,
                    "weak_objects": d.weak_objects,
                    "gnn": d.gnn,
                })
            library = self._library_map(desks)
            return {
                "desks": desks,
                "map": library,
                "seat_time": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
                "summary": {
                    "total": len(desks),
                    "available": counts[AVAILABLE],
                    "reclaimable": counts[RECLAIMABLE],
                    "partial": counts[PARTIAL],
                    "occupied": counts[OCCUPIED],
                },
                "grace_minutes": self.grace_seconds / 60,
                "clock_offset_min": self.clock.offset / 60,
                "server_time": datetime.fromtimestamp(now).strftime("%H:%M:%S"),
                "events": list(self.events)[::-1][:30],
            }

    # --------------------------------------------------------
    def _library_map(self, visible_desks):
        """Seat grid, zones, heat, graph edges and the best-seat recommendation."""
        fw, fh = self.frame_size or (1, 1)
        by_id = {d["id"]: d for d in visible_desks}
        seats = []
        for did in sorted(self.desks):
            d = self.desks[did]
            if not self._on_map(d):
                continue
            x1, y1, x2, y2 = d.box
            center = [((x1 + x2) / 2) / fw, ((y1 + y2) / 2) / fh]
            size = [(x2 - x1) / fw, (y2 - y1) / fh]
            if d.visible:
                state = d.last_state
                heat = HEAT.get(state, 0.0)
            else:  # hidden this frame: show the spatial model's estimate
                g = d.gnn or {"p": [1 / 3] * 3, "state": "free"}
                state = GNN_TO_STATE[g["state"]]
                heat = g["p"][0] + 0.7 * g["p"][1]
            seats.append({
                "id": did, "state": state, "visible": d.visible, "inferred": not d.visible,
                "center": [round(center[0], 4), round(center[1], 4)],
                "size": [round(size[0], 4), round(size[1], 4)],
                "heat": round(heat, 3), "evidence": round(d.evidence, 3),
                "gnn": d.gnn,
                "remaining_s": (by_id.get(did) or {}).get("remaining_s"),
                "objects": [o["label"] for o in d.objects] if d.visible else [],
            })

        pos, n_rows, n_cols = layout.grid_layout(seats)
        zone_list = layout.zones()
        zone_stats = {z["name"]: {"name": z["name"], "total": 0, "occupied": 0, "held": 0,
                                  "free": 0, "inferred": 0, "heat_sum": 0.0}
                      for z in zone_list}
        for s in seats:
            s["row"], s["col"] = pos.get(s["id"], (0, 0))
            s["zone"] = layout.assign_zone(s["center"], zone_list)
            z = zone_stats.setdefault(s["zone"], {"name": s["zone"], "total": 0, "occupied": 0,
                                                  "held": 0, "free": 0, "inferred": 0, "heat_sum": 0.0})
            z["total"] += 1
            z["heat_sum"] += s["heat"]
            z["inferred"] += s["inferred"]
            if s["state"] == OCCUPIED:
                z["occupied"] += 1
            elif s["state"] == PARTIAL:
                z["held"] += 1
            else:
                z["free"] += 1
        # copy zone/row/col onto the per-desk entries (used by storage + Live page)
        for s in seats:
            if s["id"] in by_id:
                by_id[s["id"]].update(zone=s["zone"], row=s["row"], col=s["col"])

        zones_out = []
        for z in zone_stats.values():
            t = z.pop("total")
            heat = z.pop("heat_sum")
            zones_out.append({**z, "total": t,
                              "occupancy": round((z["occupied"] + z["held"]) / t, 3) if t else None,
                              "heat": round(heat / t, 3) if t else None})

        # best seat: a free seat (visible) whose neighbourhood is quietest
        free = [s for s in seats if s["visible"] and s["state"] in (AVAILABLE, RECLAIMABLE)]
        def quiet(s):
            nb = (s["gnn"] or {}).get("nbr_busy")
            return (nb if nb is not None else 0.5) + (0.15 if s["state"] == RECLAIMABLE else 0)
        recommend = []
        for s in sorted(free, key=quiet)[:3]:
            g = s["gnn"] or {}
            nbrs = g.get("neighbours", [])
            busy = sum(1 for n in nbrs for t in seats if t["id"] == n and t["state"] in (OCCUPIED, PARTIAL))
            recommend.append({"id": s["id"], "zone": s["zone"], "state": s["state"],
                              "neighbours": len(nbrs), "busy_neighbours": busy,
                              "nbr_busy": g.get("nbr_busy")})

        flags = {}
        for s in seats:
            f = (s["gnn"] or {}).get("flag")
            if f:
                flags[f] = flags.get(f, 0) + 1
        return {
            "floor": None, "rows": n_rows, "cols": n_cols, "seats": seats, "zones": zones_out,
            "edges": [list(e) for e in (self.gnn.edges if self.gnn else [])
                      if e[0] in pos and e[1] in pos],
            "recommend": recommend, "gnn_enabled": self.gnn is not None, "flags": flags,
        }
