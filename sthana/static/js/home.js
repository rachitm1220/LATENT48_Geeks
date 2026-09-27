/* Landing page: live numbers, latest capture, scroll-driven 30-minute timeline, hero parallax */
(() => {
  "use strict";
  const { $, api, esc, fmtDate, reduceMotion } = window.S;

  /* ---------- Right now: live seat counts ------------------------- */
  async function loadNow() {
    try {
      const s = await api("/api/state");
      const sum = s.summary;
      if (!sum.total) {
        $("#now-updated").textContent = "Waiting for the first frame.";
        return;
      }
      $("#now-free").textContent = sum.available + sum.reclaimable;
      $("#now-away").textContent = sum.partial;
      $("#now-busy").textContent = sum.occupied;
      $("#now-updated").textContent = `${sum.total} desks · updated ${s.frame.processed_at || s.server_time}.`;
    } catch (e) {
      $("#now-updated").textContent = "Live data unavailable.";
    }
  }
  loadNow();
  setInterval(loadNow, 5000);

  /* ---------- Latest saved capture -------------------------------- */
  api("/api/runs?per_page=1").then((d) => {
    const run = d.runs[0];
    if (!run) return;
    $("#latest").hidden = false;
    $("#latest-link").href = `/history/${encodeURIComponent(run.run_id)}`;
    const img = $("#latest-img");
    img.src = `/runs/${encodeURIComponent(run.run_id)}/annotated.jpg`;
    img.alt = `Annotated capture of ${run.source}`;
    $("#latest-source").textContent = run.source;
    $("#latest-time").textContent = fmtDate(run.timestamp);
    const free = Number(run.available) + Number(run.reclaimable);
    $("#latest-chips").innerHTML = `
      <span class="chip"><i class="dot ok"></i><b>${free}</b> free</span>
      <span class="chip"><i class="dot warn"></i><b>${esc(run.partial)}</b> away</span>
      <span class="chip"><i class="dot bad"></i><b>${esc(run.occupied)}</b> occupied</span>`;
    window.S.observeReveals($("#latest"));
  }).catch(() => {});

  /* ---------- Scroll-driven timeline + hero parallax -------------- */
  const track = $("#rule-track");
  const timeline = $("#timeline");
  const grace = Number(timeline.dataset.grace) || 30;
  const shot = $("#hero-shot");
  let ticking = false;

  function update() {
    ticking = false;
    const vh = window.innerHeight;

    // Timeline progress: 0 when the section reaches the top, 1 near the end
    const r = track.getBoundingClientRect();
    const span = Math.max(1, r.height - vh);
    let p = Math.min(1, Math.max(0, -r.top / span * 1.15));
    if (reduceMotion.matches) p = 1;
    timeline.style.setProperty("--p", p.toFixed(4));
    timeline.classList.toggle("done", p >= 0.999);
    $("#rule-min").textContent = Math.min(grace - 1, Math.floor(p * grace));

    // Parallax: the floor plan drifts up slightly slower than the page
    if (shot && !reduceMotion.matches) {
      const y = Math.min(window.scrollY, vh) * -0.06;
      shot.style.setProperty("--parallax", `${y.toFixed(1)}px`);
    }
  }
  const onScroll = () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } };
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll);
  update();
})();
