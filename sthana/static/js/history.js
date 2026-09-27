/* History: grid of saved runs with search, type filter and "load more" */
(() => {
  "use strict";
  const { $, $$, api, esc, fmtDate } = window.S;

  const PER_PAGE = 24;
  let page = 1, pages = 1, q = "", kind = "", loading = false, requestId = 0;

  function card(run) {
    const free = Number(run.available) + Number(run.reclaimable);
    const id = encodeURIComponent(run.run_id);
    return `
      <a class="run-card" href="/history/${id}" data-reveal>
        <div class="run-thumb">
          <img src="${esc(run.thumb)}" alt="Annotated frame from ${esc(run.source)}" loading="lazy" decoding="async" width="560" height="420">
          <span class="run-kind">${run.kind === "video" ? "Video" : "Image"}</span>
        </div>
        <div class="run-body">
          <p class="run-source" title="${esc(run.source)}">${esc(run.source)}</p>
          <p class="meta">${fmtDate(run.timestamp)} · ${esc(run.desks)} desks</p>
          <div class="chips">
            <span class="chip"><i class="dot ok"></i><b>${free}</b></span>
            <span class="chip"><i class="dot warn"></i><b>${esc(run.partial)}</b></span>
            <span class="chip"><i class="dot bad"></i><b>${esc(run.occupied)}</b></span>
          </div>
        </div>
      </a>`;
  }

  function skeletons(n) {
    return Array.from({ length: n }, () =>
      `<div class="run-card is-skeleton" aria-hidden="true"><div class="run-thumb skeleton"></div><div class="run-body"><div class="skeleton" style="height:16px;width:70%"></div><div class="skeleton" style="height:12px;width:45%;margin-top:10px"></div></div></div>`).join("");
  }

  async function load(reset) {
    if (loading && !reset) return;
    loading = true;
    const my = ++requestId;
    if (reset) { page = 1; $("#run-grid").innerHTML = skeletons(8); }
    try {
      const params = new URLSearchParams({ page, per_page: PER_PAGE, q, kind });
      const d = await api(`/api/runs?${params}`);
      if (my !== requestId) return;            // a newer search replaced this one
      pages = d.pages;

      $("#st-saved").textContent = d.stats.saved.toLocaleString();
      $("#st-analyses").textContent = d.stats.analyses.toLocaleString();
      $("#st-desks").textContent = d.stats.desk_rows.toLocaleString();
      $("#st-last").textContent = d.stats.last ? fmtDate(d.stats.last, false) : "—";

      const grid = $("#run-grid");
      if (reset) grid.innerHTML = "";
      if (!d.total) {
        grid.innerHTML = d.stats.saved
          ? `<div class="empty"><h3>No matches.</h3><p>Try another search.</p></div>`
          : `<div class="empty"><h3>Nothing saved yet.</h3><p>Analyze an image or play a video. Each run appears here.</p><a class="btn btn-primary" href="/monitor">Open Live Monitor</a></div>`;
      } else {
        grid.insertAdjacentHTML("beforeend", d.runs.map(card).join(""));
        window.S.observeReveals(grid);
      }
      $("#h-count").textContent = d.total ? `${d.total.toLocaleString()} run${d.total === 1 ? "" : "s"}` : "";
      $("#h-more").hidden = page >= pages;
    } catch (e) {
      $("#run-grid").innerHTML = `<div class="empty"><h3>Couldn't load history.</h3><p>${esc(e.message)}</p></div>`;
    } finally {
      if (my === requestId) loading = false;
    }
  }

  let t;
  $("#h-search").addEventListener("input", (e) => {
    clearTimeout(t);
    t = setTimeout(() => { q = e.target.value.trim(); load(true); }, 250);
  });

  $$("#h-kind button").forEach((b) => b.addEventListener("click", () => {
    $$("#h-kind button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    kind = b.dataset.kind;
    load(true);
  }));

  $("#h-more").addEventListener("click", () => { page += 1; load(false); });

  load(true);
})();
