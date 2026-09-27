/* Records: paginated, searchable views of the three CSV files.
   The URL keeps the current table, filter, search and page (?t=&state=&q=&page=). */
(() => {
  "use strict";
  const { $, $$, api, esc, fmtDate, fmtClock, STATE, pill } = window.S;

  const runLink = (r) => r.run_id
    ? `<a class="mono" href="/history/${encodeURIComponent(r.run_id)}">${esc(r.run_id)}</a>` : "—";
  const savedRunLink = (r) => (r.has_images === "1" ? runLink(r) : `<span class="mono muted">${esc(r.run_id)}</span>`);
  const num = (key) => (r) => `<span class="num">${esc(r[key])}</span>`;

  // Columns shown per table: [header, render(row)]. Downloads include every column.
  const TABLES = {
    desks: {
      states: true,
      cols: [
        ["Time", (r) => fmtDate(r.timestamp)],
        ["Desk", (r) => `<b>${esc(r.desk_id)}</b>`],
        ["Zone", (r) => esc(r.zone || "—")],
        ["State", (r) => pill(r.state)],
        ["Objects", (r) => esc((r.objects || "").split(";").filter(Boolean).join(", ")) || '<span class="muted">—</span>'],
        ["Free in", (r) => (r.timer_remaining_s !== "" && r.state === "PARTIALLY_OCCUPIED" ? fmtClock(r.timer_remaining_s) : '<span class="muted">—</span>')],
        ["Desk conf", num("desk_conf")],
        ["Box", (r) => `<span class="mono muted">${esc([r.x1, r.y1, r.x2, r.y2].join(", "))}</span>`],
        ["Source", (r) => `<span class="clip" title="${esc(r.source)}">${esc(r.source)}</span>`],
        ["Run", runLink],
        ["Crop", (r) => r.crop
          ? `<a href="/history/${encodeURIComponent(r.run_id)}" class="crop-link"><img src="/runs/${encodeURIComponent(r.run_id)}/${esc(r.crop)}" alt="${esc(r.desk_id)} crop" loading="lazy" decoding="async" width="72" height="54"></a>`
          : '<span class="muted">—</span>'],
      ],
    },
    runs: {
      states: false,
      cols: [
        ["Time", (r) => fmtDate(r.timestamp)],
        ["Source", (r) => `<span class="clip" title="${esc(r.source)}">${esc(r.source)}</span>`],
        ["Type", (r) => (r.kind === "video" ? "Video" : "Image")],
        ["Desks", num("desks")],
        ["Free", (r) => `<span class="num ok-ink">${Number(r.available) + Number(r.reclaimable)}</span>`],
        ["Away", (r) => `<span class="num warn-ink">${esc(r.partial)}</span>`],
        ["Occupied", (r) => `<span class="num bad-ink">${esc(r.occupied)}</span>`],
        ["People", num("persons")],
        ["ms", num("inference_ms")],
        ["Run", savedRunLink],
      ],
    },
    events: {
      states: true,
      cols: [
        ["Time", (r) => fmtDate(r.timestamp)],
        ["Seat clock", (r) => `<span class="mono">${esc(r.seat_clock)}</span>`],
        ["Desk", (r) => (r.desk_id ? `<b>${esc(r.desk_id)}</b>` : '<span class="muted">—</span>')],
        ["State", (r) => (r.state ? pill(r.state) : '<span class="muted">—</span>')],
        ["Event", (r) => esc(r.text)],
      ],
    },
  };

  // state from URL
  const params = new URLSearchParams(location.search);
  let table = TABLES[params.get("t")] ? params.get("t") : "desks";
  let stateFilter = params.get("state") || "";
  let q = params.get("q") || "";
  let page = Number(params.get("page")) || 1;
  let perPage = 50;
  let requestId = 0;
  $("#rec-search").value = q;

  function syncUrl() {
    const p = new URLSearchParams();
    if (table !== "desks") p.set("t", table);
    if (stateFilter) p.set("state", stateFilter);
    if (q) p.set("q", q);
    if (page > 1) p.set("page", page);
    history.replaceState(null, "", `${location.pathname}${p.toString() ? "?" + p : ""}`);
  }

  function renderTabs() {
    $$("#rec-tabs button").forEach((b) => {
      const on = b.dataset.table === table;
      b.setAttribute("aria-selected", String(on));
      b.tabIndex = on ? 0 : -1;
    });
    $("#rec-download").href = `/download/${table}.csv`;
  }

  function renderStates(counts) {
    const wrap = $("#rec-states");
    if (!TABLES[table].states) { wrap.innerHTML = ""; return; }
    const opts = [["", "All", counts._all || 0]].concat(
      Object.keys(STATE).filter((s) => s !== "UNKNOWN").map((s) => [s, STATE[s].label, counts[s] || 0]));
    wrap.innerHTML = opts.map(([v, label, n]) => {
      const cls = v ? STATE[v].cls : "";
      return `<button type="button" class="chip chip-btn" data-state="${v}" aria-pressed="${v === stateFilter}">
        ${v ? `<i class="dot ${cls}" aria-hidden="true"></i>` : ""}${esc(label)} <b>${n.toLocaleString()}</b></button>`;
    }).join("");
  }

  function renderTable(d) {
    const { cols } = TABLES[table];
    const head = `<thead><tr>${cols.map(([h]) => `<th scope="col">${esc(h)}</th>`).join("")}</tr></thead>`;
    const body = d.rows.length
      ? d.rows.map((r) => `<tr>${cols.map(([, fn]) => `<td>${fn(r)}</td>`).join("")}</tr>`).join("")
      : `<tr><td colspan="${cols.length}" class="empty-cell">${
          d.counts._all ? "No rows match." : "No records yet. Analyze a frame on the Live page."}</td></tr>`;
    $("#rec-table").innerHTML = `${head}<tbody>${body}</tbody>`;

    const from = d.total ? (d.page - 1) * d.per_page + 1 : 0;
    const to = Math.min(d.total, d.page * d.per_page);
    $("#rec-count").textContent = `${from.toLocaleString()}–${to.toLocaleString()} of ${d.total.toLocaleString()}`;
    $("#rec-page").textContent = `${d.page} / ${d.pages}`;
    $("#rec-prev").disabled = d.page <= 1;
    $("#rec-next").disabled = d.page >= d.pages;
  }

  async function load() {
    const my = ++requestId;
    renderTabs();
    syncUrl();
    $("#rec-table").setAttribute("aria-busy", "true");
    try {
      const p = new URLSearchParams({ q, state: stateFilter, page, per_page: perPage });
      const d = await api(`/api/records/${table}?${p}`);
      if (my !== requestId) return;
      page = d.page;
      renderStates(d.counts);
      renderTable(d);
      syncUrl();
    } catch (e) {
      $("#rec-table").innerHTML = `<tbody><tr><td class="empty-cell">Couldn't load records: ${esc(e.message)}</td></tr></tbody>`;
    } finally {
      $("#rec-table").removeAttribute("aria-busy");
    }
  }

  // events
  $("#rec-tabs").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b || b.dataset.table === table) return;
    table = b.dataset.table; stateFilter = ""; page = 1;
    load();
  });
  $("#rec-tabs").addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const keys = Object.keys(TABLES);
    const i = (keys.indexOf(table) + (e.key === "ArrowRight" ? 1 : keys.length - 1)) % keys.length;
    table = keys[i]; stateFilter = ""; page = 1;
    load().then(() => $(`#rec-tabs [data-table="${table}"]`).focus());
  });
  $("#rec-states").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    stateFilter = b.dataset.state; page = 1;
    load();
  });
  let t;
  $("#rec-search").addEventListener("input", (e) => {
    clearTimeout(t);
    t = setTimeout(() => { q = e.target.value.trim(); page = 1; load(); }, 250);
  });
  $("#rec-per").addEventListener("change", (e) => { perPage = Number(e.target.value); page = 1; load(); });
  $("#rec-prev").addEventListener("click", () => { page -= 1; load(); });
  $("#rec-next").addEventListener("click", () => { page += 1; load(); });

  load();
})();
