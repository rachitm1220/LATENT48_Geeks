"""
Library map layout: turns desk boxes from the camera into a seat grid and zones.

    grid_layout()   rows by vertical position (tolerant to perspective), columns left->right
    assign_zone()   first configured zone that contains the seat centre
"""
from statistics import median

import config


def zones():
    """Configured zones, or equal vertical strips of the camera view."""
    if config.ZONES:
        return config.ZONES
    names = config.AUTO_ZONE_NAMES or ["Zone A"]
    n = len(names)
    return [{"name": name, "x1": i / n, "y1": 0.0, "x2": (i + 1) / n, "y2": 1.0}
            for i, name in enumerate(names)]


def assign_zone(center, zone_list=None):
    if center is None:
        return "Unzoned"
    zone_list = zone_list or zones()
    x, y = center
    for z in zone_list:
        if z["x1"] <= x <= z["x2"] and z["y1"] <= y <= z["y2"]:
            return z["name"]
    # outside every rectangle: nearest zone centre
    def dist(z):
        return ((x - (z["x1"] + z["x2"]) / 2) ** 2 + (y - (z["y1"] + z["y2"]) / 2) ** 2)
    return min(zone_list, key=dist)["name"]


def grid_layout(seats):
    """
    seats: list of {"id", "center": [x, y], "size": [w, h]} in normalised coords.
    Returns ({id: (row, col)}, n_rows, n_cols).

    Rows: seats sorted top->bottom; a new row starts when a seat's centre is lower
    than the current row's mean by more than half a typical desk height.
    """
    placed = [s for s in seats if s.get("center")]
    if not placed:
        return {}, 0, 0
    heights = [s["size"][1] for s in placed if s.get("size")] or [0.1]
    tol = 0.5 * median(heights)

    rows = []
    for s in sorted(placed, key=lambda s: s["center"][1]):
        cy = s["center"][1]
        if rows and abs(cy - rows[-1]["mean"]) <= tol:
            row = rows[-1]
            row["seats"].append(s)
            row["mean"] = sum(x["center"][1] for x in row["seats"]) / len(row["seats"])
        else:
            rows.append({"mean": cy, "seats": [s]})

    pos, n_cols = {}, 0
    for r, row in enumerate(rows):
        ordered = sorted(row["seats"], key=lambda s: s["center"][0])
        n_cols = max(n_cols, len(ordered))
        for c, s in enumerate(ordered):
            pos[s["id"]] = (r, c)
    return pos, len(rows), n_cols
