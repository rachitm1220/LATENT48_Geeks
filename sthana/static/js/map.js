/* =====================================================================
   Map page
   - live seat grid (Status / Heat / Graph views), polled every 2 s
   - zone occupancy cards
   - time-lapse: slider + play, replay grid, occupancy line, zone and seat matrices
   - spatial reasoning (GNN) stats and flagged seats
   Heat uses one blue ramp (light -> dark); status colours stay reserved for seat state.
   ===================================================================== */
(() => {
  "use strict";
  const { $, $$, api, esc, fmtClock, STATE, pill } = window.S;

  /* ---------- colour scale (sequential, one hue) ------------------ */
  const RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
                "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];
  const isDark = () => document.documentElement.getAttribute("data-theme") === "dark";

  function hex2rgb(h) { const n = parseInt(h.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; }
  function mix(a, b, t) { return a.map((x, i) => Math.round(x + (b[i] - x) * t)); }
  // light mode: low = light (recedes into the page); dark mode: anchor flips, low = dark
  function heatRGB(v) {
    const ramp = isDark() ? [...RAMP].reverse() : RAMP;
    const x = Math.max(0, Math.min(1, v)) * (ramp.length - 1);
    const i = Math.floor(x);
    return mix(hex2rgb(ramp[i]), hex2rgb(ramp[Math.min(i + 1, ramp.length - 1)]), x - i);
  }
  const heatColor = (v) => `rgb(${heatRGB(v).join(",")})`;
  function inkOn(v) {           // readable text on a heat fill
    const [r, g, b] = heatRGB(v);
    const lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
    return lum > 0.55 ? "#0b1b33" : "#ffffff";
  }
  const pct = (v) => (v == null ? "—" : `${Math.round(v * 100)}%`);
  const busyOf = (g) => (g && g.p ? g.p[0] + 0.7 * g.p[1] : null);

  const FLAG_TEXT = {
    consistent: "Detector and neighbours agree.",
    uncertain: "Weak detector evidence. Neighbours and history are weighed in.",
    disagree: "Weak evidence, and the neighbourhood points the other way. Worth a second look.",
    inferred: "Hidden in the latest frame. Estimated from its history and neighbours.",
  };
  const FLAG_LABEL = { uncertain: "Uncertain", disagree: "Disagrees", inferred: "Inferred", consistent: "Consistent" };

  /* ---------- tooltip --------------------------------------------- */
  const tip = $("#viz-tip");
  function showTip(text, x, y) {
    tip.innerHTML = text;
    tip.hidden = false;
    const r = tip.getBoundingClientRect();
    const left = Math.min(window.innerWidth - r.width - 8, Math.max(8, x + 14));
    const top = y - r.height - 12 < 8 ? y + 18 : y - r.height - 12;
    tip.style.transform = `translate(${left}px, ${top}px)`;
  }
  const hideTip = () => { tip.hidden = true; };
  document.addEventListener("pointermove", (e) => {
    const el = e.target.closest("[data-tip]");
    if (el) showTip(el.dataset.tip, e.clientX, e.clientY);
    else if (!e.target.closest(".line-chart")) hideTip();
  });
  document.addEventListener("focusin", (e) => {
    const el = e.target.closest("[data-tip]");
    if (!el) return;
    const r = el.getBoundingClientRect();
    showTip(el.dataset.tip, r.left + r.width / 2, r.top);
  });
  document.addEventListener("focusout", hideTip);

  /* ================================================================
     LIVE MAP
     ================================================================ */
  let mode = "status";
  let live = null;
  let selected = null;

  function tipFor(s) {
    const g = s.gnn || {};
    const lines = [`<b>${esc(s.id)}</b> · ${esc((STATE[s.state] || {}).label || s.state)}${s.inferred ? " (inferred)" : ""}`,
      `${esc(s.zone)} · row ${s.row + 1}, seat ${s.col + 1}`];
    if (mode === "heat") lines.push(`Heat ${pct(s.heat)}`);
    if (g.p) lines.push(`Model: occupied ${pct(g.p[0])} · held ${pct(g.p[1])} · free ${pct(g.p[2])}`);
    return lines.join("<br>");
  }

  function tileHTML(s, recoIds) {
    const g = s.gnn || {};
    let cls = "seat-tile";
    let style = `grid-row:${s.row + 1};grid-column:${s.col + 1};`;
    let sub = "";
    if (mode === "status") {
      cls += ` st-${s.state}`;
      if (s.state === "PARTIALLY_OCCUPIED" && s.remaining_s != null) sub = fmtClock(s.remaining_s);
      else if (s.state === "RECLAIMABLE") sub = "free";
    } else {
      const v = mode === "heat" ? s.heat : busyOf(g);
      style += `background:${heatColor(v ?? 0)};color:${inkOn(v ?? 0)};`;
      sub = pct(v);
      cls += " filled";
    }
    if (s.inferred) cls += " inferred";
    if (g.flag === "uncertain" || g.flag === "disagree") cls += " flagged";
    if (s.id === selected) cls += " selected";
    if (recoIds.has(s.id)) cls += " reco";
    const label = `${s.id}, ${(STATE[s.state] || {}).label || s.state}${s.inferred ? ", inferred" : ""}`;
    return `<button type="button" class="${cls}" style="${style}" data-id="${esc(s.id)}"
              data-tip="${esc(tipFor(s))}" aria-label="${esc(label)}" aria-pressed="${s.id === selected}">
              <span class="t-id">${esc(s.id)}</span>${sub ? `<span class="t-sub">${esc(sub)}</span>` : ""}</button>`;
  }

  function renderLive() {
    const m = live.map;
    $("#m-clock").textContent = live.server_time.slice(0, 5);
    const seats = m.seats;
    $("#m-empty").hidden = seats.length > 0;
    $(".map-room").hidden = seats.length === 0;
    const recoIds = new Set((m.recommend || []).slice(0, 1).map((r) => r.id));
    const grid = $("#m-grid");
    grid.style.gridTemplateColumns = `repeat(${Math.max(1, m.cols)}, minmax(var(--tile-min), 1fr))`;
    const focusedId = document.activeElement?.closest?.(".seat-tile")?.dataset.id;
    grid.innerHTML = seats.map((s) => tileHTML(s, recoIds)).join("");
    if (focusedId) grid.querySelector(`[data-id="${CSS.escape(focusedId)}"]`)?.focus({ preventScroll: true });
    grid.dataset.mode = mode;
    requestAnimationFrame(drawOverlays);
    renderLegend();
    renderReco();
    renderDetail();
    renderZones();
    renderGnn();
  }

  function tileCenters() {
    const wrap = $(".map-canvas-wrap").getBoundingClientRect();
    const out = {};
    $$("#m-grid .seat-tile").forEach((t) => {
      const r = t.getBoundingClientRect();
      out[t.dataset.id] = [r.left - wrap.left + r.width / 2, r.top - wrap.top + r.height / 2, r.width];
    });
    return out;
  }

  function drawOverlays() {
    const wrapEl = $(".map-canvas-wrap");
    const w = wrapEl.clientWidth, h = wrapEl.clientHeight;
    const centers = tileCenters();

    // heat glow (Heat view): soft blobs behind the tiles
    const cv = $("#m-heat");
    const dpr = window.devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = h * dpr;
    cv.style.width = `${w}px`; cv.style.height = `${h}px`;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (mode === "heat" && live) {
      for (const s of live.map.seats) {
        const c = centers[s.id];
        if (!c || s.heat <= 0.02) continue;
        const r = c[2] * 1.5;
        const [R, G, B] = heatRGB(Math.max(0.55, s.heat));
        const grad = ctx.createRadialGradient(c[0], c[1], 0, c[0], c[1], r);
        grad.addColorStop(0, `rgba(${R},${G},${B},${0.45 * s.heat})`);
        grad.addColorStop(1, `rgba(${R},${G},${B},0)`);
        ctx.fillStyle = grad;
        ctx.fillRect(c[0] - r, c[1] - r, r * 2, r * 2);
      }
    }

    // graph edges (Graph view)
    const svg = $("#m-graph");
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    if (mode !== "graph" || !live) { svg.innerHTML = ""; return; }
    svg.innerHTML = live.map.edges.map(([a, b, wt]) => {
      const p = centers[a], q = centers[b];
      if (!p || !q) return "";
      const hot = selected && (a === selected || b === selected);
      return `<line x1="${p[0]}" y1="${p[1]}" x2="${q[0]}" y2="${q[1]}" class="edge${hot ? " hot" : ""}"
                style="stroke-width:${(1 + 2.5 * wt).toFixed(2)};opacity:${(hot ? 0.9 : 0.25 + 0.45 * wt).toFixed(2)}"/>`;
    }).join("");
  }

  function scaleBar() {
    const stops = Array.from({ length: 11 }, (_, i) => `${heatColor(i / 10)} ${i * 10}%`).join(",");
    return `<span class="scale-bar" style="background:linear-gradient(90deg,${stops})" aria-hidden="true"></span>`;
  }

  function renderLegend() {
    let html;
    if (mode === "status") {
      html = `<span><i class="dot ok"></i>Available</span><span><i class="dot reclaim"></i>Free after grace period</span>
              <span><i class="dot warn"></i>Belongings only</span><span><i class="dot bad"></i>Occupied</span>
              <span><i class="swatch-hatch"></i>Inferred (hidden)</span><span><i class="swatch-flag"></i>Uncertain</span>`;
    } else if (mode === "heat") {
      html = `<span class="scale">Free ${scaleBar()} Full</span><span class="meta">Occupied 100% · held 70% · reclaimable 30% · free 0%</span>`;
    } else {
      html = `<span class="scale">Free ${scaleBar()} Busy</span><span class="meta">Model estimate per seat. Lines join neighbouring seats; thicker = closer.</span>`;
    }
    $("#m-legend").innerHTML = html;
  }

  function renderReco() {
    const r = live.map.recommend || [];
    if (!r.length) {
      $("#m-reco-id").textContent = live.map.seats.length ? "Full" : "–";
      $("#m-reco-why").textContent = live.map.seats.length ? "No free seat right now." : "Waiting for seats.";
      $("#m-reco-alt").innerHTML = "";
      return;
    }
    const best = r[0];
    $("#m-reco-id").textContent = best.id;
    const quiet = best.neighbours
      ? `${best.busy_neighbours} of ${best.neighbours} neighbouring seats busy.`
      : "No close neighbours.";
    $("#m-reco-why").textContent = `${best.zone} · ${best.state === "RECLAIMABLE" ? "reclaimable, move the things aside" : "free"}. ${quiet}`;
    $("#m-reco-alt").innerHTML = r.slice(1).map((x) =>
      `<button type="button" class="chip chip-btn" data-select="${esc(x.id)}">Also ${esc(x.id)} · ${esc(x.zone)}</button>`).join("");
  }

  function probBars(p) {
    const rows = [["Occupied", p[0], "bad"], ["Held", p[1], "warn"], ["Free", p[2], "ok"]];
    return `<div class="prob">${rows.map(([l, v, c]) => `
      <div class="prob-row"><span><i class="dot ${c}"></i>${l}</span>
        <span class="prob-bar"><span style="width:${Math.round(v * 100)}%"></span></span><b>${pct(v)}</b></div>`).join("")}</div>`;
  }

  function renderDetail() {
    const box = $("#m-detail");
    const s = live.map.seats.find((x) => x.id === selected);
    if (!s) {
      box.innerHTML = `<p class="meta">Select a seat to see its details, neighbours and what the spatial model thinks.</p>`;
      return;
    }
    const g = s.gnn || {};
    const byId = Object.fromEntries(live.map.seats.map((x) => [x.id, x]));
    const nbrs = (g.neighbours || []).map((id) => {
      const n = byId[id];
      const cls = n ? (STATE[n.state] || {}).cls : "";
      return `<button type="button" class="chip chip-btn" data-select="${esc(id)}"><i class="dot ${cls}"></i>${esc(id)}</button>`;
    }).join("");
    box.innerHTML = `
      <div class="desk-top"><h3>${esc(s.id)}</h3>${pill(s.state)}</div>
      <p class="meta">${esc(s.zone)} · row ${s.row + 1}, seat ${s.col + 1}${s.objects.length ? ` · ${esc(s.objects.join(", "))}` : ""}</p>
      ${s.state === "PARTIALLY_OCCUPIED" && s.remaining_s != null ? `<p class="detail-timer">Free in <b>${fmtClock(s.remaining_s)}</b></p>` : ""}
      <div class="detail-block">
        <p class="detail-label">Detector confidence</p>
        <div class="prob-row single"><span class="prob-bar"><span style="width:${Math.round((s.visible ? s.evidence : 0) * 100)}%"></span></span><b>${s.visible ? pct(s.evidence) : "hidden"}</b></div>
      </div>
      ${g.p ? `<div class="detail-block"><p class="detail-label">Spatial model <span class="flag-chip f-${esc(g.flag)}">${esc(FLAG_LABEL[g.flag] || g.flag)}</span></p>
        ${probBars(g.p)}<p class="meta">${esc(FLAG_TEXT[g.flag] || "")}</p></div>` : ""}
      ${nbrs ? `<div class="detail-block"><p class="detail-label">Neighbours · ${g.nbr_busy == null ? "" : `${pct(g.nbr_busy)} busy`}</p><div class="chips">${nbrs}</div></div>` : ""}`;
  }

  function renderZones() {
    const zones = live.map.zones.filter((z) => z.total > 0);
    if (!zones.length) {
      $("#m-zones").innerHTML = `<div class="empty small"><h3>No zones yet.</h3><p>Zones fill in once seats are detected.</p></div>`;
      return;
    }
    const occ = zones.map((z) => z.occupancy);
    const max = Math.max(...occ), min = Math.min(...occ);
    $("#m-zones").innerHTML = zones.map((z) => {
      const badge = zones.length > 1 && max !== min
        ? (z.occupancy === max ? `<span class="pill">Busiest</span>` : z.occupancy === min ? `<span class="pill ok">Quietest</span>` : "")
        : "";
      return `<div class="card zone-card">
        <div class="desk-top"><h3>${esc(z.name)}</h3>${badge}</div>
        <p class="zone-num">${pct(z.occupancy)}</p>
        <div class="zone-bar" aria-hidden="true"><span style="width:${Math.round(z.occupancy * 100)}%;background:${heatColor(Math.max(0.35, z.occupancy))}"></span></div>
        <p class="meta">${z.occupied} occupied · ${z.held} held · ${z.free} free${z.inferred ? ` · ${z.inferred} inferred` : ""}</p>
      </div>`;
    }).join("");
  }

  function renderGnn() {
    const m = live.map;
    const f = m.flags || {};
    const n = m.seats.length, e = m.edges.length;
    const tiles = [
      ["Nodes", n], ["Edges", e], ["Avg. neighbours", n ? (2 * e / n).toFixed(1) : "0"],
      ["Uncertain", f.uncertain || 0], ["Inferred", f.inferred || 0], ["Disagree", f.disagree || 0],
    ];
    $("#g-stats").innerHTML = m.gnn_enabled
      ? tiles.map(([l, v]) => `<div class="stat-tile"><span class="stat-num">${v}</span><span class="stat-label">${l}</span></div>`).join("")
      : `<div class="card"><p class="meta">Spatial reasoning is off (GNN_ENABLED = False in config.py).</p></div>`;
    const flagged = m.seats.filter((s) => s.gnn && s.gnn.flag !== "consistent");
    $("#g-flags").innerHTML = flagged.length
      ? flagged.map((s) => `<button type="button" class="flag-item" data-select="${esc(s.id)}">
          <span class="desk-top"><b>${esc(s.id)}</b><span class="flag-chip f-${esc(s.gnn.flag)}">${esc(FLAG_LABEL[s.gnn.flag])}</span></span>
          <span class="meta">Detector ${s.visible ? pct(s.evidence) : "hidden"} · model: occupied ${pct(s.gnn.p[0])}, free ${pct(s.gnn.p[2])}</span></button>`).join("")
      : `<p class="meta">Nothing flagged. Every seat's detection agrees with its neighbourhood.</p>`;
  }

  function select(id, scroll) {
    selected = selected === id ? null : id;
    if (live) renderLive();
    if (scroll && selected) $("#m-stage").scrollIntoView({ behavior: "smooth", block: "center" });
  }

  document.addEventListener("click", (e) => {
    const t = e.target.closest("#m-grid .seat-tile");
    if (t) return select(t.dataset.id, false);
    const s = e.target.closest("[data-select]");
    if (s) { selected = null; select(s.dataset.select, !!s.closest("#g-flags")); }
  });

  $$("#m-mode button").forEach((b) => b.addEventListener("click", () => {
    mode = b.dataset.mode;
    $$("#m-mode button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    if (live) renderLive();
  }));

  async function pollLive() {
    try {
      live = await api("/api/state");
      if (live.map.floor) { $("#m-floor").textContent = live.map.floor; $("#m-floor-2").textContent = live.map.floor; }
      if (selected && !live.map.seats.some((s) => s.id === selected)) selected = null;
      renderLive();
    } catch (e) { /* server restarting: keep last view */ }
  }

  /* ================================================================
     TIME-LAPSE
     ================================================================ */
  let tl = null;
  let idx = 0;
  let bucket = window.MAP_DEFAULT_BUCKET || 60;
  let day = "";
  let playTimer = null;

  function setBucketButtons() {
    $$("#t-bucket button").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.min) === bucket)));
  }

  async function loadTimeline(keepIdx) {
    try {
      const p = new URLSearchParams({ bucket, day });
      tl = await api(`/api/timeline?${p}`);
    } catch (e) { return; }
    day = tl.day || "";
    $("#t-day").innerHTML = tl.days.length
      ? tl.days.slice().reverse().map((d) => `<option value="${d}" ${d === tl.day ? "selected" : ""}>${new Date(d + "T00:00").toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" })}</option>`).join("")
      : `<option>No data</option>`;
    $("#t-sim").hidden = !tl.simulated;
    const has = tl.buckets.length > 0;
    $("#t-empty").hidden = has;
    $(".player").hidden = !has;
    $(".time-grid").hidden = !has;
    $("#t-zone-card").hidden = !has;
    $("#t-seat-card").hidden = !has;
    $("#t-scale").hidden = !has;
    if (!has) return;
    const slider = $("#t-slider");
    slider.max = tl.buckets.length - 1;
    if (!keepIdx) {
      // start at the busiest bucket so the first view is interesting
      let best = 0;
      tl.overall.forEach((v, i) => { if (v != null && v > (tl.overall[best] ?? -1)) best = i; });
      idx = best;
    }
    idx = Math.min(idx, tl.buckets.length - 1);
    renderTimeline();
  }

  function renderTimeline() {
    $("#t-slider").value = idx;
    const t = tl.buckets[idx];
    $("#t-now").textContent = t;
    $("#t-now-2").textContent = t;
    const ov = tl.overall[idx];
    $("#t-now-meta").textContent = ov == null ? "no observations" : `${pct(ov)} occupied · ${tl.samples[idx]} observations`;
    $("#t-slider").setAttribute("aria-valuetext", `${t}, ${pct(ov)} occupied`);
    renderReplay();
    renderLine();
    renderMatrix($("#t-zone-matrix"), tl.zones, (z) => tl.zone_matrix[z], (z) => z, null);
    renderMatrix($("#t-seat-matrix"), tl.seats.map((s) => s.id), (id) => tl.seat_matrix[id],
      (id) => { const s = tl.seats.find((x) => x.id === id); return `${id}<small>${esc(s?.zone || "")}</small>`; },
      (id, i) => tl.seat_state[id][i]);
    $("#t-scale").innerHTML = `<span class="scale">0% ${scaleBar()} 100%</span><span class="swatch-empty"></span><span class="meta">no observations</span>`;
  }

  function renderReplay() {
    const seats = tl.seats.filter((s) => s.row != null && s.col != null);
    const cols = Math.max(1, ...seats.map((s) => s.col + 1));
    const grid = $("#t-grid");
    grid.style.gridTemplateColumns = `repeat(${cols}, minmax(var(--tile-min-sm), 1fr))`;
    grid.innerHTML = seats.map((s) => {
      const v = tl.seat_matrix[s.id][idx];
      const st = tl.seat_state[s.id][idx];
      const style = `grid-row:${s.row + 1};grid-column:${s.col + 1};` +
        (v == null ? "" : `background:${heatColor(v)};color:${inkOn(v)};`);
      const tipText = `<b>${esc(s.id)}</b> at ${tl.buckets[idx]}<br>${v == null ? "No observations" : `Occupied ${pct(v)} of the time`}${st ? `<br>Mostly ${esc((STATE[st] || {}).label || st)}` : ""}`;
      return `<span class="seat-tile filled small${v == null ? " nodata" : ""}" style="${style}" data-tip="${esc(tipText)}" tabindex="0">
                <span class="t-id">${esc(s.id)}</span></span>`;
    }).join("");
  }

  function renderMatrix(el, keys, rowFn, labelFn, stateFn) {
    const n = tl.buckets.length;
    const every = n > 48 ? 8 : n > 24 ? 4 : n > 12 ? 2 : 1;
    el.style.gridTemplateColumns = `minmax(92px, max-content) repeat(${n}, minmax(22px, 1fr))`;
    let html = `<div class="mx-row" role="row"><span class="mx-corner" role="columnheader"></span>` +
      tl.buckets.map((b, i) => `<span class="mx-head${i === idx ? " now" : ""}" role="columnheader">${i % every === 0 ? b : ""}</span>`).join("") + `</div>`;
    for (const k of keys) {
      const vals = rowFn(k);
      html += `<div class="mx-row" role="row"><span class="mx-label" role="rowheader">${labelFn(k)}</span>` +
        vals.map((v, i) => {
          const st = stateFn ? stateFn(k, i) : null;
          const tipText = `<b>${esc(k)}</b> · ${tl.buckets[i]}<br>${v == null ? "No observations" : `${pct(v)} occupied`}${st ? `<br>Mostly ${esc((STATE[st] || {}).label || st)}` : ""}`;
          const style = v == null ? "" : `background:${heatColor(v)}`;
          return `<span class="mx-cell${v == null ? " nodata" : ""}${i === idx ? " now" : ""}" role="cell" style="${style}"
                    data-tip="${esc(tipText)}" data-i="${i}" aria-label="${esc(`${k} ${tl.buckets[i]}: ${pct(v)}`)}"></span>`;
        }).join("") + `</div>`;
    }
    el.innerHTML = html;
  }

  function renderLine() {
    const box = $("#t-chart");
    const W = Math.max(280, box.clientWidth), H = 220;
    const pad = { l: 42, r: 12, t: 12, b: 28 };
    const n = tl.buckets.length;
    const x = (i) => pad.l + (n === 1 ? (W - pad.l - pad.r) / 2 : (i * (W - pad.l - pad.r)) / (n - 1));
    const y = (v) => pad.t + (1 - v) * (H - pad.t - pad.b);
    // line segments break where there is no data
    let d = "", area = "", seg = [];
    const flush = () => {
      if (!seg.length) return;
      d += seg.map((i, k) => `${k ? "L" : "M"}${x(i).toFixed(1)},${y(tl.overall[i]).toFixed(1)}`).join("");
      area += `M${x(seg[0]).toFixed(1)},${y(0)}` + seg.map((i) => `L${x(i).toFixed(1)},${y(tl.overall[i]).toFixed(1)}`).join("") +
        `L${x(seg[seg.length - 1]).toFixed(1)},${y(0)}Z`;
      seg = [];
    };
    tl.overall.forEach((v, i) => (v == null ? flush() : seg.push(i)));
    flush();
    const every = n > 48 ? 8 : n > 24 ? 4 : n > 12 ? 2 : 1;
    const xl = tl.buckets.map((b, i) => (i % every === 0 ? `<text x="${x(i)}" y="${H - 8}" class="ax" text-anchor="middle">${b}</text>` : "")).join("");
    const grid = [0, 0.5, 1].map((v) => `<line x1="${pad.l}" x2="${W - pad.r}" y1="${y(v)}" y2="${y(v)}" class="gl"/>
      <text x="${pad.l - 8}" y="${y(v) + 4}" class="ax" text-anchor="end">${v * 100}%</text>`).join("");
    const cur = tl.overall[idx];
    box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img"
        aria-label="Share of seats occupied or held, by time of day. ${pct(cur)} at ${tl.buckets[idx]}.">
      ${grid}<path d="${area}" class="area"/><path d="${d}" class="line"/>
      <line x1="${x(idx)}" x2="${x(idx)}" y1="${pad.t}" y2="${y(0)}" class="now-line"/>
      ${cur == null ? "" : `<circle cx="${x(idx)}" cy="${y(cur)}" r="5" class="now-dot"/>`}
      <line class="hover-line" x1="0" x2="0" y1="${pad.t}" y2="${y(0)}" visibility="hidden"/>
      ${xl}<rect class="hit" x="${pad.l}" y="0" width="${W - pad.l - pad.r}" height="${H}"/></svg>`;
    const svg = box.querySelector("svg");
    const hl = box.querySelector(".hover-line");
    const nearest = (ev) => {
      const r = svg.getBoundingClientRect();
      const px = ((ev.clientX - r.left) / r.width) * W;
      return Math.max(0, Math.min(n - 1, Math.round(((px - pad.l) / (W - pad.l - pad.r)) * (n - 1))));
    };
    svg.querySelector(".hit").addEventListener("pointermove", (ev) => {
      const i = nearest(ev);
      hl.setAttribute("x1", x(i)); hl.setAttribute("x2", x(i)); hl.setAttribute("visibility", "visible");
      showTip(`<b>${tl.buckets[i]}</b><br>${tl.overall[i] == null ? "No observations" : `${pct(tl.overall[i])} occupied`}`, ev.clientX, ev.clientY);
    });
    svg.querySelector(".hit").addEventListener("pointerleave", () => { hl.setAttribute("visibility", "hidden"); hideTip(); });
    svg.querySelector(".hit").addEventListener("click", (ev) => { stop(); idx = nearest(ev); renderTimeline(); });
  }

  function stop() {
    clearInterval(playTimer);
    playTimer = null;
    $("#t-play").classList.remove("playing");
    $("#t-play").setAttribute("aria-label", "Play time-lapse");
  }
  $("#t-play").addEventListener("click", () => {
    if (!tl || !tl.buckets.length) return;
    if (playTimer) return stop();
    if (idx >= tl.buckets.length - 1) idx = 0;
    $("#t-play").classList.add("playing");
    $("#t-play").setAttribute("aria-label", "Pause time-lapse");
    renderTimeline();
    playTimer = setInterval(() => {
      if (idx >= tl.buckets.length - 1) return stop();
      idx += 1;
      renderTimeline();
    }, 800);
  });
  $("#t-slider").addEventListener("input", (e) => { stop(); idx = Number(e.target.value); renderTimeline(); });
  $("#t-day").addEventListener("change", (e) => { stop(); day = e.target.value; loadTimeline(false); });
  $$("#t-bucket button").forEach((b) => b.addEventListener("click", () => {
    stop(); bucket = Number(b.dataset.min); setBucketButtons(); loadTimeline(false);
  }));
  document.addEventListener("click", (e) => {
    const c = e.target.closest(".mx-cell");
    if (c) { stop(); idx = Number(c.dataset.i); renderTimeline(); }
  });

  /* ---------- redraw on theme / resize ---------------------------- */
  document.addEventListener("themechange", () => { if (live) renderLive(); if (tl && tl.buckets.length) renderTimeline(); });
  let rt;
  window.addEventListener("resize", () => {
    clearTimeout(rt);
    rt = setTimeout(() => { if (live) drawOverlays(); if (tl && tl.buckets.length) renderLine(); }, 150);
  });

  setBucketButtons();
  pollLive();
  setInterval(pollLive, 2000);
  loadTimeline(false);
  setInterval(() => { if (!playTimer) loadTimeline(true); }, 60000);
})();
