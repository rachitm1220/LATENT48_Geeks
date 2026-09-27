/* =====================================================================
   site.js: shared behaviour for every page
   - theme switch (light / dark, remembered, follows system until chosen)
   - glass navigation on scroll + mobile menu
   - scroll-reveal animations
   - pill button ripple
   - small helpers used by the page scripts (window.S)
   ===================================================================== */
(() => {
  "use strict";

  const root = document.documentElement;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  /* ---------- Theme ----------------------------------------------- */
  const toggle = document.getElementById("theme-toggle");
  const metaColor = document.getElementById("meta-theme-color");

  function storedTheme() {
    try { return localStorage.getItem("theme"); } catch (e) { return null; }
  }

  function applyTheme(theme, animate) {
    if (animate && !reduceMotion.matches) {
      root.classList.add("theme-anim");
      clearTimeout(applyTheme.t);
      applyTheme.t = setTimeout(() => root.classList.remove("theme-anim"), 500);
    }
    root.setAttribute("data-theme", theme);
    toggle?.setAttribute("aria-checked", String(theme === "dark"));
    if (metaColor) metaColor.content = theme === "dark" ? "#000000" : "#ffffff";
    document.dispatchEvent(new CustomEvent("themechange", { detail: theme }));
  }

  applyTheme(root.getAttribute("data-theme") || "light", false);

  toggle?.addEventListener("click", () => {
    const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    try { localStorage.setItem("theme", next); } catch (e) { /* private mode: still works for this page */ }
    applyTheme(next, true);
  });

  // Follow the OS setting until the user picks a theme explicitly
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
    if (!storedTheme()) applyTheme(e.matches ? "dark" : "light", true);
  });

  /* ---------- Navigation ------------------------------------------ */
  const nav = document.getElementById("nav");
  const burger = document.getElementById("nav-burger");
  const menu = document.getElementById("mobile-menu");

  const onScroll = () => nav.classList.toggle("scrolled", window.scrollY > 4);
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });

  function setMenu(open) {
    nav.classList.toggle("menu-open", open);
    burger.setAttribute("aria-expanded", String(open));
    burger.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    menu.setAttribute("aria-hidden", String(!open));
    menu.querySelectorAll("a").forEach((a) => (a.tabIndex = open ? 0 : -1));
    document.body.style.overflow = open ? "hidden" : "";
    if (open) menu.querySelector("a")?.focus();
  }
  burger?.addEventListener("click", () => setMenu(!nav.classList.contains("menu-open")));
  menu?.addEventListener("click", (e) => { if (e.target.closest("a")) setMenu(false); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && nav.classList.contains("menu-open")) { setMenu(false); burger.focus(); }
  });
  window.matchMedia("(min-width: 834px)").addEventListener("change", (e) => { if (e.matches) setMenu(false); });

  /* ---------- Scroll reveal --------------------------------------- */
  function observeReveals(scope = document) {
    const items = scope.querySelectorAll("[data-reveal]:not(.in)");
    if (reduceMotion.matches || !("IntersectionObserver" in window)) {
      items.forEach((el) => el.classList.add("in"));
      return;
    }
    items.forEach((el) => revealer.observe(el));
  }
  const revealer = "IntersectionObserver" in window
    ? new IntersectionObserver((entries) => {
        for (const entry of entries) {
          // reveal on entry, and also anything already scrolled past (anchor jumps)
          if (entry.isIntersecting || entry.boundingClientRect.top < 0) {
            const el = entry.target;
            el.classList.add("in");
            revealer.unobserve(el);
            // hand control back to the element's own transitions once revealed
            const done = (ev) => {
              if (ev.target !== el || ev.propertyName !== "opacity") return;
              el.removeEventListener("transitionend", done);
              el.removeAttribute("data-reveal");
            };
            el.addEventListener("transitionend", done);
          }
        }
      }, { rootMargin: "0px 0px -8% 0px", threshold: 0.12 })
    : null;
  observeReveals();

  // Safety net for very fast scrolling: reveal anything already on or above screen
  let revealTick = false;
  window.addEventListener("scroll", () => {
    if (revealTick) return;
    revealTick = true;
    requestAnimationFrame(() => {
      revealTick = false;
      const limit = window.innerHeight * 0.95;
      document.querySelectorAll("[data-reveal]:not(.in)").forEach((el) => {
        if (el.getBoundingClientRect().top < limit) el.classList.add("in");
      });
    });
  }, { passive: true });

  /* ---------- Button ripple --------------------------------------- */
  document.addEventListener("pointerdown", (e) => {
    const btn = e.target.closest(".btn");
    if (!btn || btn.disabled || reduceMotion.matches) return;
    const r = btn.getBoundingClientRect();
    const size = Math.max(r.width, r.height);
    const dot = document.createElement("span");
    dot.className = "ripple";
    dot.style.width = dot.style.height = `${size}px`;
    dot.style.left = `${e.clientX - r.left - size / 2}px`;
    dot.style.top = `${e.clientY - r.top - size / 2}px`;
    btn.appendChild(dot);
    dot.addEventListener("animationend", () => dot.remove());
  });

  /* ---------- Shared helpers -------------------------------------- */
  const $ = (sel, scope = document) => scope.querySelector(sel);
  const $$ = (sel, scope = document) => Array.from(scope.querySelectorAll(sel));

  async function api(path, opts = {}) {
    const r = await fetch(path, opts);
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || r.statusText);
    return data;
  }

  const postJSON = (path, body) => api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // seconds -> "mm:ss"
  function fmtClock(sec) {
    sec = Math.max(0, Math.round(Number(sec) || 0));
    const m = Math.floor(sec / 60), s = sec % 60;
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }

  // ISO timestamp -> "27 Sep, 12:30:05"
  function fmtDate(iso, withSeconds = true) {
    if (!iso) return "—";
    const d = new Date(iso);
    if (isNaN(d)) return esc(iso);
    return d.toLocaleString(undefined, {
      day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
      ...(withSeconds ? { second: "2-digit" } : {}),
    });
  }

  // seat state -> label + pill/dot class
  const STATE = {
    AVAILABLE: { label: "Available", cls: "ok" },
    RECLAIMABLE: { label: "Reclaimable", cls: "reclaim" },
    PARTIALLY_OCCUPIED: { label: "Partially occupied", cls: "warn" },
    OCCUPIED: { label: "Occupied", cls: "bad" },
    UNKNOWN: { label: "Unknown", cls: "" },
  };
  const pill = (state) => {
    const s = STATE[state] || { label: state || "—", cls: "" };
    return `<span class="pill ${s.cls}"><i class="dot ${s.cls}" aria-hidden="true"></i>${esc(s.label)}</span>`;
  };

  function toast(text, ms = 5000) {
    const t = document.createElement("div");
    t.className = "toast";
    t.setAttribute("role", "status");
    t.textContent = text;
    document.body.appendChild(t);
    setTimeout(() => { t.classList.add("out"); t.addEventListener("animationend", () => t.remove()); }, ms);
  }

  window.S = { $, $$, api, postJSON, esc, fmtClock, fmtDate, STATE, pill, toast, observeReveals, reduceMotion };
})();
