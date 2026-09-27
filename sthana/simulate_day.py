"""
Generate a SIMULATED day of seat observations for demoing the time-lapse heatmap.

Real data needs hours of footage. This writes a plausible day (08:00-20:00) for a
grid of seats into outputs/desk_observations.csv and outputs/runs.csv, tagged
source = "SIMULATED demo day" so it is always distinguishable from real data.
The Map page shows a "Includes simulated data" notice whenever these rows are present.

    python simulate_day.py                 # today, 08:00-20:00 every 10 min
    python simulate_day.py --date 2026-09-28 --start 8 --end 18 --step 15
    python simulate_day.py --remove        # delete all simulated rows again

Model: each seat is a small Markov chain (free / occupied / held-by-belongings).
The chance of a free seat being taken follows a daily demand curve, a per-zone
popularity factor, and how many neighbours are already occupied (people cluster).
Stop the app before running this.
"""
import argparse
import csv
import random
from datetime import datetime, timedelta

import config
import layout
from storage import FIELDS, RunStore

SOURCE = "SIMULATED demo day"
# (hour, demand 0..1): quiet morning, lunchtime peak, afternoon peak, evening drop
DEMAND = [(7, 0.05), (8, 0.15), (9, 0.4), (10, 0.65), (11, 0.85), (12, 0.9), (13, 0.75),
          (14, 0.6), (15, 0.8), (16, 0.85), (17, 0.7), (18, 0.5), (19, 0.35), (20, 0.2), (21, 0.05)]


def demand(t):
    h = t.hour + t.minute / 60
    for (h0, d0), (h1, d1) in zip(DEMAND, DEMAND[1:]):
        if h0 <= h <= h1:
            return d0 + (d1 - d0) * (h - h0) / (h1 - h0)
    return 0.05


def build_seats(rows=5, cols=6):
    seats = []
    for r in range(rows):
        for c in range(cols):
            cx, cy = (c + 0.5) / cols, 0.3 + r * 0.14
            seats.append({"id": f"D{r * cols + c + 1:02d}", "center": [cx, cy], "size": [0.12, 0.08]})
    pos, _, _ = layout.grid_layout(seats)
    zones = layout.zones()
    for s in seats:
        s["row"], s["col"] = pos[s["id"]]
        s["zone"] = layout.assign_zone(s["center"], zones)
    return seats


def neighbours(seats, s):
    return [t for t in seats if t is not s and abs(t["row"] - s["row"]) <= 1 and abs(t["col"] - s["col"]) <= 1]


def simulate(date, start, end, step, seed):
    rng = random.Random(seed)
    seats = build_seats()
    zone_names = sorted({s["zone"] for s in seats})
    # first zone is the popular one (e.g. window side), last the quietest
    zone_factor = {z: 1.35 - 0.7 * i / max(1, len(zone_names) - 1) for i, z in enumerate(zone_names)}
    state = {s["id"]: "free" for s in seats}
    held_since = {}
    grace = config.GRACE_PERIOD_MINUTES

    t = datetime.combine(date, datetime.min.time()).replace(hour=start)
    stop = t.replace(hour=end)
    desk_rows, run_rows = [], []
    while t <= stop:
        d = demand(t)
        new = {}
        for s in seats:
            cur = state[s["id"]]
            nb = neighbours(seats, s)
            nb_occ = sum(state[n["id"]] != "free" for n in nb) / max(1, len(nb))
            if cur == "free":
                p = min(0.9, d * zone_factor[s["zone"]] * 0.16 + 0.06 * nb_occ * d)
                new[s["id"]] = "occupied" if rng.random() < p else "free"
            elif cur == "occupied":
                r = rng.random()
                leave = 0.09 + 0.3 * max(0.0, 0.6 - d)
                new[s["id"]] = "held" if r < 0.07 else ("free" if r < 0.07 + leave else "occupied")
            else:  # held: owner may return, or abandon
                r = rng.random()
                new[s["id"]] = "occupied" if r < 0.3 else ("free" if r < 0.36 else "held")
        for sid, st in new.items():
            if st == "held" and state[sid] != "held":
                held_since[sid] = t
            if st != "held":
                held_since.pop(sid, None)
        state = new

        run_id = f"SIM-{t:%Y%m%d-%H%M}"
        ts = t.isoformat(timespec="seconds")
        counts = {"AVAILABLE": 0, "RECLAIMABLE": 0, "PARTIALLY_OCCUPIED": 0, "OCCUPIED": 0}
        for s in seats:
            st = state[s["id"]]
            if st == "occupied":
                label = "OCCUPIED"
            elif st == "held":
                mins = (t - held_since[s["id"]]).total_seconds() / 60
                label = "RECLAIMABLE" if mins >= grace else "PARTIALLY_OCCUPIED"
            else:
                label = "AVAILABLE"
            counts[label] += 1
            obs = {"OCCUPIED": "PERSON_PRESENT", "AVAILABLE": "EMPTY"}.get(label, "BELONGINGS_ONLY")
            desk_rows.append({
                "run_id": run_id, "timestamp": ts, "seat_time": ts, "source": SOURCE,
                "desk_id": s["id"], "zone": s["zone"], "row": s["row"], "col": s["col"],
                "state": label, "observation": obs,
                "objects": "bag;book" if obs == "BELONGINGS_ONLY" else "",
                "object_count": 2 if obs == "BELONGINGS_ONLY" else 0,
                "reason": "simulated",
            })
        run_rows.append({
            "run_id": run_id, "timestamp": ts, "seat_time": ts, "source": SOURCE, "kind": "simulated",
            "desks": len(seats), "available": counts["AVAILABLE"], "reclaimable": counts["RECLAIMABLE"],
            "partial": counts["PARTIALLY_OCCUPIED"], "occupied": counts["OCCUPIED"],
            "persons": counts["OCCUPIED"], "inference_ms": 0, "has_images": 0,
        })
        t += timedelta(minutes=step)
    return desk_rows, run_rows


def remove_simulated(store):
    for name in ("desks", "runs"):
        path = store.files[name]
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or FIELDS[name]
            keep = [r for r in reader if not (r.get("source") or "").upper().startswith("SIMULATED")]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(keep)
        print(f"{path.name}: kept {len(keep)} real rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="YYYY-MM-DD (default: today)")
    ap.add_argument("--start", type=int, default=8, help="start hour (default 8)")
    ap.add_argument("--end", type=int, default=20, help="end hour (default 20)")
    ap.add_argument("--step", type=int, default=10, help="minutes between observations (default 10)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--remove", action="store_true", help="delete all simulated rows and exit")
    args = ap.parse_args()

    store = RunStore(config.OUTPUT_DIR)
    if args.remove:
        remove_simulated(store)
        return
    date = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else datetime.now().date()
    desk_rows, run_rows = simulate(date, args.start, args.end, args.step, args.seed)
    store._append("desks", desk_rows)
    store._append("runs", run_rows)
    print(f"Wrote {len(run_rows)} simulated snapshots ({len(desk_rows)} seat rows) for {date}, "
          f"{args.start:02d}:00-{args.end:02d}:00. Source is tagged '{SOURCE}'.")
    print("Remove them any time with:  python simulate_day.py --remove")


if __name__ == "__main__":
    main()
