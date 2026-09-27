/* Run detail: annotated / original frame, then every desk with its crop and object view */
(() => {
  "use strict";
  const { $, $$, api, esc, fmtDate, fmtClock, STATE, pill } = window.S;

  const runId = $("#run-root").dataset.runId;
  let meta = null;
  let filter = "";

  function setNav(el, id) {
    if (id) {
      el.href = `/history/${encodeURIComponent(id)}`;
      el.removeAttribute("aria-disabled");
    } else {
      el.removeAttribute("href");
      el.setAttribute("aria-disabled", "true");
    }
  }

  function showImage(which) {
    const url = meta.urls[which];
    $("#r-img").src = url;
    $("#r-img").alt = `${which === "annotated" ? "Annotated" : "Original"} frame from ${meta.source}`;
    $("#r-img-link").href = url;
    $$("#r-view button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.view === which)));
  }

  function deskCard(d) {
    const objs = d.objects.length
      ? `<ul class="obj-list">${d.objects.map((o) => `<li><span>${esc(o.label)}</span><span class="meta">${o.confidence.toFixed(2)}</span></li>`).join("")}</ul>`
      : `<p class="meta">${d.observation === "PERSON_PRESENT" ? "Person present, object model skipped." : "No objects detected."}</p>`;
    const timer = d.remaining_s != null
      ? `<p class="meta">Unattended ${fmtClock(d.elapsed_s)}${d.state === "RECLAIMABLE" ? " · grace period over" : ` · free in ${fmtClock(d.remaining_s)}`}</p>`
      : "";
    const img = (url, label) => url
      ? `<figure><button type="button" class="thumb-btn" data-full="${esc(url)}" data-cap="${esc(`${d.id} · ${label}`)}">
           <img src="${esc(url)}" alt="${esc(`${d.id} ${label.toLowerCase()}`)}" loading="lazy" decoding="async"></button>
           <figcaption>${label}</figcaption></figure>`
      : "";
    return `
      <article class="card desk-detail" data-reveal>
        <header class="desk-top"><h3>${esc(d.id)}</h3>${pill(d.state)}</header>
        <div class="desk-imgs">${img(d.crop_url, "Crop")}${img(d.objects_url, "Detections")}</div>
        <p class="desk-reason">${esc(d.reason)}</p>
        ${timer}
        ${objs}
        <p class="meta mono">box ${d.box.join(", ")} · conf ${Number(d.confidence).toFixed(2)}</p>
      </article>`;
  }

  function renderDesks() {
    const list = meta.desks.filter((d) => !filter || d.state === filter);
    $("#r-desks").innerHTML = list.length
      ? list.map(deskCard).join("")
      : `<div class="empty"><h3>No desks in this state.</h3></div>`;
    window.S.observeReveals($("#r-desks"));
  }

  function renderFilter() {
    const counts = {};
    meta.desks.forEach((d) => { counts[d.state] = (counts[d.state] || 0) + 1; });
    const opts = [["", `All ${meta.desks.length}`]].concat(
      Object.keys(STATE).filter((s) => counts[s]).map((s) => [s, `${STATE[s].label} ${counts[s]}`]));
    $("#r-filter").innerHTML = opts.map(([v, l]) =>
      `<button type="button" data-state="${v}" aria-pressed="${v === filter}">${esc(l)}</button>`).join("");
  }

  async function load() {
    try {
      meta = await api(`/api/runs/${encodeURIComponent(runId)}`);
    } catch (e) {
      $("#r-title").textContent = "Run not found.";
      return;
    }
    document.title = `${meta.source} · ${document.title.split("·").pop().trim()}`;
    $("#r-title").textContent = meta.source;
    const [w, h] = meta.image_size || [];
    $("#r-sub").textContent = `${fmtDate(meta.timestamp)} · ${meta.kind === "video" ? "Video frame" : "Image"}` +
      `${w ? ` · ${w}×${h}` : ""} · ${meta.inference_ms} ms · ${meta.run_id}`;
    setNav($("#r-prev"), meta.prev);
    setNav($("#r-next"), meta.next);

    const s = meta.summary;
    $("#r-chips").innerHTML = `
      <span class="chip"><i class="dot ok"></i><b>${s.available + s.reclaimable}</b> free</span>
      <span class="chip"><i class="dot warn"></i><b>${s.partial}</b> away</span>
      <span class="chip"><i class="dot bad"></i><b>${s.occupied}</b> occupied</span>
      <span class="chip"><b>${meta.persons.length}</b> people</span>`;
    $("#r-files").innerHTML = `
      <a class="link-more small" href="${meta.urls.original}" download>Original</a>
      <a class="link-more small" href="${meta.urls.annotated}" download>Annotated</a>
      <a class="link-more small" href="${meta.urls.meta}" target="_blank" rel="noopener">meta.json</a>
      <span class="meta">outputs/runs/${esc(meta.run_id)}/</span>`;

    showImage("annotated");
    renderFilter();
    renderDesks();
  }

  $("#r-view").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b && meta) showImage(b.dataset.view);
  });

  $("#r-filter").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    filter = b.dataset.state;
    $$("#r-filter button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    renderDesks();
  });

  // lightbox for desk images
  const box = $("#lightbox");
  $("#r-desks").addEventListener("click", (e) => {
    const b = e.target.closest(".thumb-btn");
    if (!b) return;
    $("#lightbox-img").src = b.dataset.full;
    $("#lightbox-img").alt = b.dataset.cap;
    $("#lightbox-cap").textContent = b.dataset.cap;
    box.showModal();
  });
  box.addEventListener("click", (e) => { if (e.target === box) box.close(); });

  // arrow keys step through runs
  document.addEventListener("keydown", (e) => {
    if (box.open || /INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) return;
    if (e.key === "ArrowLeft" && meta?.prev) location.href = `/history/${encodeURIComponent(meta.prev)}`;
    if (e.key === "ArrowRight" && meta?.next) location.href = `/history/${encodeURIComponent(meta.next)}`;
  });

  load();
})();
