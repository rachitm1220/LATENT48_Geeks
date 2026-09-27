/* =====================================================================
   Live monitor
   Polls /api/state every second and redraws the frame overlay and seat
   cards, so an orange seat turns green on screen the moment its timer ends.
   ===================================================================== */
(() => {
  "use strict";
  const { $, $$, api, postJSON, esc, fmtClock, STATE, pill, toast } = window.S;

  let state = null;
  let loadedFrameId = null;
  let frameReady = false;
  let lastEventId = 0;
  let autoplayTimer = null;
  let knownDeskIds = "";
  let videoTabChosen = false;
  const frameImg = new Image();

  // Overlay colours (vivid system colours read well on photos in both themes)
  const COLORS = {
    AVAILABLE: "#34c759",
    RECLAIMABLE: "#34c759",
    PARTIALLY_OCCUPIED: "#ff9500",
    OCCUPIED: "#ff3b30",
    UNKNOWN: "#8e8e93",
  };
  const PERSON = "#0a84ff";
  const OBJECT = "#ffd60a";

  /* ---------- helpers --------------------------------------------- */
  function setStatus(msg, isError = false) {
    const el = $("#status");
    el.textContent = msg;
    el.classList.toggle("error", isError);
  }

  async function withBusy(fn) {
    $("#busy").hidden = false;
    try {
      const r = await fn();
      await poll();
      return r;
    } catch (e) {
      setStatus(e.message, true);
    } finally {
      $("#busy").hidden = true;
    }
  }

  /* ---------- polling + rendering --------------------------------- */
  async function poll() {
    try {
      state = await api("/api/state");
      render();
    } catch (e) {
      setStatus("Server not reachable: " + e.message, true);
    }
  }

  function render() {
    $("#clock-time").textContent = state.server_time;
    $("#clock-offset").textContent = state.clock_offset_min
      ? `Clock +${state.clock_offset_min.toFixed(1)} min` : "";
    if (document.activeElement !== $("#grace")) $("#grace").value = +state.grace_minutes.toFixed(2);
    $("#grace-display").innerHTML = `${+state.grace_minutes.toFixed(1)}<span> min</span>`;
    const [hh, mm] = state.server_time.split(":");
    $("#clock-display").innerHTML = `${hh}<span>:${mm}</span>`;

    renderFrame();
    renderVideo();
    renderSummary();
    renderAlerts();
    renderDesks();
    renderDemoSelect();
    renderEvents();
  }

  function renderFrame() {
    const f = state.frame;
    const run = state.last_run || {};
    const link = $("#last-run-link");
    link.hidden = !(run.id && run.saved);
    if (run.id && run.saved) link.href = `/history/${encodeURIComponent(run.id)}`;

    if (!f || f.frame_id == null) {
      $("#placeholder").hidden = false;
      $("#view").hidden = true;
      $("#frame-info").textContent = "";
      loadedFrameId = null;
      frameReady = false;
      return;
    }
    $("#frame-info").textContent =
      `${f.source} · ${f.width}×${f.height} · ${f.inference_ms} ms · ${f.processed_at}`;
    if (f.frame_id !== loadedFrameId) {
      loadedFrameId = f.frame_id;
      frameReady = false;
      frameImg.onload = () => { frameReady = true; draw(); };
      frameImg.src = `/api/frame?v=${f.frame_id}`;
    } else {
      draw();
    }
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
    else ctx.rect(x, y, w, h);
  }

  function drawLabel(ctx, text, x, y, color, font, ink = "#fff") {
    ctx.font = `600 ${font}px -apple-system, "SF Pro Text", Inter, system-ui, sans-serif`;
    const w = ctx.measureText(text).width + font * 0.9;
    const h = font * 1.6;
    const lx = Math.max(0, Math.min(x, ctx.canvas.width - w));
    const ly = Math.max(y - h - 4, 0);
    ctx.fillStyle = color;
    roundRect(ctx, lx, ly, w, h, h / 2);
    ctx.fill();
    ctx.fillStyle = ink;
    ctx.fillText(text, lx + font * 0.45, ly + font * 1.13);
  }

  function draw() {
    if (!frameReady || !state) return;
    $("#placeholder").hidden = true;
    const c = $("#view");
    c.hidden = false;
    const ctx = c.getContext("2d");
    c.width = frameImg.naturalWidth;
    c.height = frameImg.naturalHeight;
    ctx.drawImage(frameImg, 0, 0);

    const lw = Math.max(2, Math.round(c.width / 380));
    const font = Math.max(14, Math.round(c.width / 58));

    if ($("#show-persons").checked) {
      ctx.setLineDash([lw * 3, lw * 2]);
      ctx.lineWidth = lw;
      ctx.strokeStyle = PERSON;
      for (const p of state.frame.persons) {
        const [x1, y1, x2, y2] = p.box;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }
      ctx.setLineDash([]);
    }

    for (const d of state.desks) {
      const [x1, y1, x2, y2] = d.box;
      const color = COLORS[d.state] || COLORS.UNKNOWN;

      ctx.globalAlpha = 0.16;
      ctx.fillStyle = color;
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      ctx.globalAlpha = 1;

      ctx.lineWidth = lw * 2;
      ctx.strokeStyle = color;
      ctx.setLineDash(d.state === "RECLAIMABLE" ? [lw * 5, lw * 3] : []);
      roundRect(ctx, x1, y1, x2 - x1, y2 - y1, lw * 3);
      ctx.stroke();
      ctx.setLineDash([]);

      if ($("#show-objects").checked) {
        ctx.lineWidth = lw;
        ctx.strokeStyle = OBJECT;
        for (const o of d.objects) {
          const [a, b, cc, dd] = o.box;
          ctx.strokeRect(a, b, cc - a, dd - b);
          drawLabel(ctx, o.label, a, b, OBJECT, Math.round(font * 0.72), "#1d1d1f");
        }
      }

      let text = `${d.id} · ${(STATE[d.state] || {}).label || d.label}`;
      if (d.state === "PARTIALLY_OCCUPIED" && d.remaining_s != null) text += ` · ${fmtClock(d.remaining_s)}`;
      drawLabel(ctx, text, x1, y1, color, font);
    }
  }

  function renderVideo() {
    const v = state.video || { state: "idle" };
    const active = v.state === "playing" || v.state === "paused";
    $("#btn-video-pause").disabled = !active;
    $("#btn-video-stop").disabled = !active;
    $("#btn-video-pause").textContent = v.state === "paused" ? "Resume" : "Pause";

    // open the Video tab automatically if a video is already running
    if (active && !videoTabChosen) { selectTab("video"); videoTabChosen = true; }

    const box = $("#video-status");
    if (v.state === "idle") { box.hidden = true; return; }
    box.hidden = false;
    box.classList.toggle("error", v.state === "error");

    const labels = { playing: "Playing", paused: "Paused", finished: "Finished", stopped: "Stopped", error: "Error" };
    const pos = fmtClock(v.position_s || 0);
    const dur = v.duration_s ? ` / ${fmtClock(v.duration_s)}` : "";
    const pct = v.duration_s ? Math.min(100, (100 * v.position_s) / v.duration_s) : null;
    const opts = v.live ? "live" :
      `${v.speed}× · every ${v.sample_seconds}s${v.sync_clock ? " · video time" : ""}${v.loop ? " · loop" : ""}`;
    box.innerHTML = `
      <div class="vs-row"><b>${labels[v.state] || esc(v.state)}</b><span>${esc(v.name)}</span>
      <span class="meta">${pos}${dur} · ${v.frames_analyzed} frames · ${esc(opts)}</span></div>
      ${v.error ? `<div>${esc(v.error)}</div>` : ""}
      ${pct != null ? `<div class="vbar"><span style="width:${pct}%"></span></div>` : ""}`;
  }

  function renderSummary() {
    const s = state.summary;
    $("#summary").innerHTML = `
      <div class="stat-tile"><span class="stat-num ok">${s.available + s.reclaimable}</span><span class="stat-label">Free</span></div>
      <div class="stat-tile"><span class="stat-num warn">${s.partial}</span><span class="stat-label">Away</span></div>
      <div class="stat-tile"><span class="stat-num bad">${s.occupied}</span><span class="stat-label">Occupied</span></div>
      <div class="stat-tile"><span class="stat-num">${s.total}</span><span class="stat-label">Desks</span></div>`;
  }

  function renderAlerts() {
    const rec = state.desks.filter((d) => d.state === "RECLAIMABLE");
    $("#alerts").innerHTML = rec.map((d) => `
      <div class="alert">
        <span class="alert-icon" aria-hidden="true">✓</span>
        <div><b>${esc(d.id)} is free now.</b> ${esc(d.message)}
        ${d.objects.length ? `<span class="meta">${esc(d.objects.map((o) => o.label).join(", "))} on desk.</span>` : ""}</div>
      </div>`).join("");
  }

  function renderDesks() {
    if (!state.desks.length) {
      $("#desk-list").innerHTML = `<div class="empty small"><h3>No desks yet.</h3><p>Analyze a frame to see seats here.</p></div>`;
      return;
    }
    const selected = $("#demo-desk").value;
    // keep keyboard focus on the same seat across the 1-second refresh
    const focusedId = document.activeElement?.closest?.(".desk-card")?.dataset.id;
    $("#desk-list").innerHTML = state.desks.map((d) => {
      const objs = d.objects.length
        ? `<p class="desk-objs">${d.objects.map((o) => `${esc(o.label)} <span class="muted">${o.confidence.toFixed(2)}</span>`).join(" · ")}</p>`
        : "";
      let timer = "";
      if (d.progress != null) {
        const pct = Math.round(d.progress * 100);
        const txt = d.state === "RECLAIMABLE"
          ? `Unattended ${fmtClock(d.elapsed_s)}. Grace period over.`
          : `Free in <b>${fmtClock(d.remaining_s)}</b> <span class="muted">· away ${fmtClock(d.elapsed_s)}</span>`;
        timer = `<div class="timer"><div class="bar"><span style="width:${pct}%"></span></div><p>${txt}</p></div>`;
      }
      return `
        <button type="button" class="desk-card s-${d.state}${d.id === selected ? " selected" : ""}" data-id="${esc(d.id)}"
                aria-pressed="${d.id === selected}">
          <span class="desk-top"><b>${esc(d.id)}</b>${pill(d.state)}</span>
          <span class="desk-msg">${esc(d.message)}</span>
          ${d.gnn && d.gnn.flag !== "consistent" ? `<span class="meta">Spatial model: ${esc(d.gnn.flag)} · occupied ${Math.round(d.gnn.p[0] * 100)}%</span>` : ""}
          ${objs}
          ${timer}
        </button>`;
    }).join("");
    if (focusedId) $(`.desk-card[data-id="${CSS.escape(focusedId)}"]`)?.focus({ preventScroll: true });
  }

  function renderDemoSelect() {
    const ids = state.desks.map((d) => d.id).join(",");
    if (ids === knownDeskIds) return;
    knownDeskIds = ids;
    const sel = $("#demo-desk");
    const prev = sel.value;
    sel.innerHTML = state.desks.length
      ? state.desks.map((d) => `<option value="${esc(d.id)}">${esc(d.id)}</option>`).join("")
      : `<option value="">No seats</option>`;
    const firstOrange = state.desks.find((d) => d.state === "PARTIALLY_OCCUPIED");
    if (prev && ids.split(",").includes(prev)) sel.value = prev;
    else if (firstOrange) sel.value = firstOrange.id;
  }

  function renderEvents() {
    if (!state.events.length) {
      $("#events").innerHTML = `<li class="muted">Nothing yet.</li>`;
      return;
    }
    $("#events").innerHTML = state.events.map((e) => {
      const cls = e.state ? (STATE[e.state] || {}).cls : "";
      return `<li><i class="dot ${cls || ""}" aria-hidden="true"></i><span class="ev-time">${esc(e.time)}</span><span>${esc(e.text)}</span></li>`;
    }).join("");
    const newest = state.events[0];
    if (newest && newest.id > lastEventId) {
      if (lastEventId && newest.state === "RECLAIMABLE") toast(newest.text, 7000);
      lastEventId = newest.id;
    }
  }

  /* ---------- tabs (Images / Video) ------------------------------- */
  function selectTab(which) {
    for (const name of ["image", "video"]) {
      const on = name === which;
      const tab = $(`#tab-${name}`);
      tab.setAttribute("aria-selected", String(on));
      tab.tabIndex = on ? 0 : -1;
      $(`#panel-${name}`).hidden = !on;
    }
  }
  $$('[role="tab"]').forEach((tab) => {
    tab.addEventListener("click", () => { videoTabChosen = true; selectTab(tab.id.replace("tab-", "")); });
    tab.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const next = tab.id === "tab-image" ? "video" : "image";
      selectTab(next);
      $(`#tab-${next}`).focus();
    });
  });

  /* ---------- image inputs ---------------------------------------- */
  async function loadLists() {
    try {
      const { images, videos } = await api("/api/images");
      $("#image-select").innerHTML = `<option value="">Data folder (${images.length})</option>` +
        images.map((n) => `<option>${esc(n)}</option>`).join("");
      $("#video-select").innerHTML = `<option value="">Data folder videos (${videos.length})</option>` +
        videos.map((n) => `<option>${esc(n)}</option>`).join("") +
        `<option value="__camera__">Live camera</option>`;
    } catch (e) { setStatus(e.message, true); }
  }

  const done = (name, r) => setStatus(`${name}: ${r.desks} desks, ${r.persons} people${r.saved ? " · saved to history" : ""}`);

  $("#upload").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("image", file);
    const r = await withBusy(() => api("/api/upload", { method: "POST", body: fd }));
    if (r) done(file.name, r);
    e.target.value = "";
  });

  $("#btn-process").addEventListener("click", async () => {
    const name = $("#image-select").value;
    if (!name) return setStatus("Pick an image from the data folder first.", true);
    const r = await withBusy(() => postJSON("/api/process", { filename: name }));
    if (r) done(name, r);
  });

  async function processNext() {
    const r = await withBusy(() => postJSON("/api/process_next"));
    if (r) { $("#image-select").value = r.filename; done(r.filename, r); }
  }
  $("#btn-next").addEventListener("click", processNext);

  $("#btn-camera").addEventListener("click", async () => {
    const r = await withBusy(() => postJSON("/api/camera"));
    if (r) done("Camera", r);
  });

  $("#autoplay").addEventListener("change", (e) => {
    clearInterval(autoplayTimer);
    if (e.target.checked) {
      const sec = Math.max(1, Number($("#autoplay-sec").value) || 5);
      processNext();
      autoplayTimer = setInterval(processNext, sec * 1000);
    }
  });

  $("#show-persons").addEventListener("change", draw);
  $("#show-objects").addEventListener("change", draw);

  /* ---------- video inputs ---------------------------------------- */
  function videoOptions() {
    return {
      sample_seconds: Number($("#video-sample").value) || 2,
      speed: Number($("#video-speed").value) || 1,
      sync_clock: $("#video-sync").checked,
      loop: $("#video-loop").checked,
      reset: $("#video-reset").checked,
      start_time: $("#video-start").value || "",
    };
  }
  function afterVideoStart() {
    if ($("#video-reset").checked) knownDeskIds = "";
    $("#autoplay").checked = false;
    clearInterval(autoplayTimer);
  }

  // upload with progress (videos can be large)
  $("#video-upload").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("video", file);
    for (const [k, v] of Object.entries(videoOptions())) fd.append(k, v);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/video/upload");
    xhr.upload.onprogress = (ev) => {
      if (ev.lengthComputable) setStatus(`Uploading ${file.name}: ${Math.round((100 * ev.loaded) / ev.total)}%`);
    };
    xhr.onload = () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch (_) { /* ignore */ }
      if (xhr.status >= 200 && xhr.status < 300) { setStatus(`Playing ${file.name}`); afterVideoStart(); poll(); }
      else setStatus(data.error || `Upload failed (${xhr.status})`, true);
    };
    xhr.onerror = () => setStatus("Upload failed", true);
    xhr.send(fd);
    e.target.value = "";
  });

  $("#btn-video-start").addEventListener("click", async () => {
    const sel = $("#video-select").value;
    if (!sel) return setStatus("Pick a video from the data folder, or upload one.", true);
    const body = { ...videoOptions(), ...(sel === "__camera__" ? { camera: true } : { filename: sel }) };
    const r = await withBusy(() => postJSON("/api/video/start", body));
    if (r) { setStatus(`Playing ${sel === "__camera__" ? "live camera" : sel}`); afterVideoStart(); }
  });

  $("#btn-video-pause").addEventListener("click", () => {
    const paused = state && state.video && state.video.state === "paused";
    withBusy(() => postJSON(paused ? "/api/video/resume" : "/api/video/pause"));
  });
  $("#btn-video-stop").addEventListener("click", () => withBusy(() => postJSON("/api/video/stop")));

  /* ---------- demo controls --------------------------------------- */
  $("#btn-set-elapsed").addEventListener("click", async () => {
    const id = $("#demo-desk").value;
    const minutes = Number($("#demo-min").value);
    if (!id) return setStatus("No seat selected.", true);
    const r = await withBusy(() => postJSON(`/api/desks/${encodeURIComponent(id)}/elapsed`, { minutes }));
    if (r) setStatus(`${id}: timer set to ${minutes} min.`);
  });

  $("#btn-grace").addEventListener("click", async () => {
    const minutes = Number($("#grace").value);
    const r = await withBusy(() => postJSON("/api/settings", { grace_minutes: minutes }));
    if (r) setStatus(`Grace period is now ${r.grace_minutes} min.`);
  });

  $("#btn-clock-set").addEventListener("click", async () => {
    const time = $("#clock-set").value;
    if (!time) return setStatus("Pick a time first.", true);
    const r = await withBusy(() => postJSON("/api/clock/set", { time }));
    if (r) setStatus(`Clock set to ${time}. New frames are recorded at this time of day.`);
  });

  $$(".ff").forEach((b) => b.addEventListener("click", () =>
    withBusy(() => postJSON("/api/clock/advance", { minutes: Number(b.dataset.min) }))));

  $("#btn-reset").addEventListener("click", async () => {
    if (!confirm("Clear live seats and timers? Saved history is kept.")) return;
    knownDeskIds = "";
    await withBusy(() => postJSON("/api/reset"));
    setStatus("Live view reset. History is untouched.");
  });

  // clicking a seat selects it for the timer
  $("#desk-list").addEventListener("click", (e) => {
    const card = e.target.closest(".desk-card");
    if (!card) return;
    $("#demo-desk").value = card.dataset.id;
    renderDesks();
    $("#demo-min").focus({ preventScroll: true });
  });

  loadLists();
  poll();
  setInterval(poll, 1000);
})();
