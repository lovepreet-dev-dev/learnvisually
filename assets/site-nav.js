/* ══════════════════════════════════════════════════════════════
   SITE SHELL

   The navigation furniture every documentation site is expected to
   have, generated from the catalog so it can never drift out of sync
   with the pages that actually exist:

     · a collapsible sidebar listing every concept, grouped by category,
       with the current page marked
     · a search palette on ⌘K / Ctrl+K (and "/"), matching title,
       summary and category
     · previous / next links so the whole syllabus can be read in order
     · a footer

   Injected into the existing markup rather than requiring every page to
   be rewritten.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";

  const catalog = window.MLAlgorithms;
  if (!catalog) return;

  const page = document.body.dataset.page;
  const atRoot = page === "home";
  const prefix = atRoot ? "./pages/" : "./";
  const rootPrefix = atRoot ? "./" : "../";

  const params = new URLSearchParams(window.location.search);
  const currentId = page === "algorithm" ? params.get("id") : null;
  const currentCategory = document.body.dataset.category || null;

  /* Reading order: categories in catalog order, pages in their own
     order within each. This is what prev/next walks. */
  const ordered = catalog.categories.flatMap((category) => catalog.byCategory(category.id));

  function algorithmHref(id) {
    return `${prefix}algorithm.html?id=${id}`;
  }

  function categoryHref(category) {
    return `${prefix}${category.page}`;
  }

  /* ── Sidebar ──────────────────────────────────────────────── */
  function buildSidebar() {
    const nav = document.createElement("nav");
    nav.className = "site-sidebar";
    nav.id = "site-sidebar";
    nav.setAttribute("aria-label", "All concepts");

    nav.innerHTML = `
      <div class="sidebar-head">
        <button type="button" class="sidebar-search" id="sidebar-search">
          <span>Search concepts…</span>
          <kbd>⌘K</kbd>
        </button>
      </div>
      <a class="sidebar-link sidebar-home${atRoot ? " is-current" : ""}" href="${rootPrefix}index.html">Home</a>
      ${catalog.categories
        .map((category) => {
          const entries = catalog.byCategory(category.id);
          const open = currentCategory === category.id || entries.some((entry) => entry.id === currentId);
          return `
            <section class="sidebar-group${open ? " is-open" : ""}">
              <button type="button" class="sidebar-group-title" aria-expanded="${open}">
                <span>${category.title}</span>
                <span class="sidebar-count">${entries.length}</span>
              </button>
              <div class="sidebar-group-body">
                <a class="sidebar-link sidebar-overview${currentCategory === category.id ? " is-current" : ""}"
                   href="${categoryHref(category)}">Overview</a>
                ${entries
                  .map(
                    (entry) => `
                      <a class="sidebar-link${entry.id === currentId ? " is-current" : ""}"
                         href="${algorithmHref(entry.id)}">${entry.title}</a>
                    `
                  )
                  .join("")}
              </div>
            </section>
          `;
        })
        .join("")}
    `;
    return nav;
  }

  /* ── Search palette ───────────────────────────────────────── */
  function buildPalette() {
    const overlay = document.createElement("div");
    overlay.className = "palette-overlay";
    overlay.id = "palette-overlay";
    overlay.hidden = true;
    overlay.innerHTML = `
      <div class="palette" role="dialog" aria-modal="true" aria-label="Search concepts">
        <input type="text" id="palette-input" aria-label="Search concepts"
               placeholder="Search concepts, categories, formulas…" autocomplete="off"
               role="combobox" aria-expanded="true" aria-controls="palette-results" aria-autocomplete="list" />
        <div class="palette-results" id="palette-results" role="listbox" aria-label="Search results"></div>
        <div class="palette-foot">
          <span><kbd>↑</kbd><kbd>↓</kbd> navigate</span>
          <span><kbd>↵</kbd> open</span>
          <span><kbd>esc</kbd> close</span>
        </div>
      </div>
    `;
    return overlay;
  }

  const searchIndex = [
    ...catalog.categories.map((category) => ({
      title: category.title,
      subtitle: category.description,
      kind: "Category",
      href: categoryHref(category),
      haystack: `${category.title} ${category.description}`.toLowerCase(),
    })),
    ...ordered.map((entry) => ({
      title: entry.title,
      subtitle: entry.summary,
      kind: catalog.categories.find((category) => category.id === entry.category).title,
      href: algorithmHref(entry.id),
      haystack: `${entry.title} ${entry.summary} ${entry.category} ${entry.engine}`.toLowerCase(),
    })),
  ];

  function score(item, query) {
    const title = item.title.toLowerCase();
    if (title === query) return 0;
    if (title.startsWith(query)) return 1;
    if (title.includes(query)) return 2;
    if (item.haystack.includes(query)) return 3;
    /* Loose subsequence match so "knn" finds "k-Nearest Neighbour". */
    let cursor = 0;
    for (const ch of query) {
      cursor = title.indexOf(ch, cursor);
      if (cursor === -1) return null;
      cursor += 1;
    }
    return 4;
  }

  function search(query) {
    const q = query.trim().toLowerCase();
    if (!q) return searchIndex.slice(0, 8);
    return searchIndex
      .map((item) => ({ item, rank: score(item, q) }))
      .filter((row) => row.rank !== null)
      .sort((a, b) => a.rank - b.rank || a.item.title.length - b.item.title.length)
      .slice(0, 12)
      .map((row) => row.item);
  }

  /* ── Prev / next ──────────────────────────────────────────── */
  function buildPager() {
    if (page !== "algorithm" || !currentId) return null;
    const index = ordered.findIndex((entry) => entry.id === currentId);
    if (index === -1) return null;

    const previous = ordered[index - 1];
    const next = ordered[index + 1];
    const pager = document.createElement("nav");
    pager.className = "pager";
    pager.setAttribute("aria-label", "Previous and next concept");
    pager.innerHTML = `
      ${
        previous
          ? `<a class="pager-link pager-prev" href="${algorithmHref(previous.id)}">
               <span class="pager-dir">← Previous</span>
               <span class="pager-title">${previous.title}</span>
             </a>`
          : "<span></span>"
      }
      ${
        next
          ? `<a class="pager-link pager-next" href="${algorithmHref(next.id)}">
               <span class="pager-dir">Next →</span>
               <span class="pager-title">${next.title}</span>
             </a>`
          : "<span></span>"
      }
    `;
    return pager;
  }

  /* Deliberately link-free. The sidebar carries every page, the home
     page ends in a full index, and the palette searches everything —
     a footer repeating 23 of those links directly below the index was
     the same list twice. */
  function buildFooter() {
    const footer = document.createElement("footer");
    footer.className = "site-footer";
    footer.innerHTML = `
      <div class="footer-bar">
        <div class="footer-brand">
          <span class="brand">Machine Learning Studio</span>
          <span>${ordered.length} interactive concepts across ${catalog.categories.length} tracks.</span>
        </div>
        <div class="footer-meta">
          <span>Static site — no accounts, no build step.</span>
          <span>Press <kbd>⌘K</kbd> to search from anywhere.</span>
          <span><a href="${rootPrefix}privacy.html">Privacy Policy</a></span>
        </div>
      </div>
    `;
    return footer;
  }


  /* ── Theme ────────────────────────────────────────────────────
     Follows the OS by default. Once the user picks a theme it is
     remembered and wins over the OS preference. The inline script in
     each page's <head> sets the initial attribute so there is no flash
     of the wrong theme before this file runs. */
  function currentTheme() {
    return document.documentElement.getAttribute("data-theme") || "light";
  }

  function applyTheme(theme, persist) {
    document.documentElement.setAttribute("data-theme", theme);
    if (persist) window.localStorage.setItem("mls-theme", theme);
    if (window.MLUtils && window.MLUtils.invalidatePalette) {
      window.MLUtils.invalidatePalette();
    }
    const button = document.getElementById("theme-toggle");
    if (button) {
      const dark = theme === "dark";
      button.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
      button.setAttribute("aria-pressed", String(dark));
      button.textContent = dark ? "☀" : "☾";
    }
    /* Charts paint their colours into SVG attributes, so they have to be
       redrawn rather than restyled. */
    window.dispatchEvent(new CustomEvent("mls:themechange", { detail: { theme } }));
  }

  function buildThemeToggle() {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "theme-toggle";
    button.id = "theme-toggle";
    button.addEventListener("click", () => {
      applyTheme(currentTheme() === "dark" ? "light" : "dark", true);
    });
    return button;
  }

  /* ── Assembly ─────────────────────────────────────────────── */
  function init() {
    const shell = document.querySelector(".site-shell");
    if (!shell) return;

    document.body.classList.add("has-shell");

    /* Keyboard users should be able to jump past the sidebar. */
    const skip = document.createElement("a");
    skip.className = "skip-link";
    skip.href = "#main-content";
    skip.textContent = "Skip to content";
    document.body.insertBefore(skip, document.body.firstChild);

    /* Sidebar toggle lives in the topbar next to the brand. */
    const topbar = shell.querySelector(".topbar");
    if (topbar) {
      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "sidebar-toggle";
      toggle.id = "sidebar-toggle";
      toggle.setAttribute("aria-label", "Toggle navigation sidebar");
      toggle.innerHTML = "<span></span><span></span><span></span>";
      topbar.insertBefore(toggle, topbar.firstChild);

      const searchButton = document.createElement("button");
      searchButton.type = "button";
      searchButton.className = "topbar-search";
      searchButton.id = "topbar-search";
      searchButton.innerHTML = `<span aria-hidden="true">⌕</span><span class="topbar-search-label">Search</span><kbd>⌘K</kbd>`;
      topbar.appendChild(searchButton);
      topbar.appendChild(buildThemeToggle());
    }

    /* Wrap everything below the topbar so the sidebar can sit beside it. */
    const layout = document.createElement("div");
    layout.className = "shell-layout";
    const main = document.createElement("main");
    main.className = "shell-main";
    main.id = "main-content";

    const children = Array.from(shell.children).filter((node) => !node.classList.contains("topbar"));
    children.forEach((node) => main.appendChild(node));

    layout.appendChild(buildSidebar());
    layout.appendChild(main);
    shell.appendChild(layout);

    const pager = buildPager();
    if (pager) main.appendChild(pager);
    main.appendChild(buildFooter());

    document.body.appendChild(buildPalette());

    wireSidebar();
    wirePalette();
    applyTheme(currentTheme(), false);

    /* Track the OS preference only while the user has no explicit choice. */
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    media.addEventListener("change", (event) => {
      if (window.localStorage.getItem("mls-theme")) return;
      applyTheme(event.matches ? "dark" : "light", false);
    });
  }

  function wireSidebar() {
    const sidebar = document.getElementById("site-sidebar");
    const toggle = document.getElementById("sidebar-toggle");

    /* Closed by default so pages open at full width; it stays open
       only after the reader opens it. Stored under a new key so the
       old default ("open") from earlier visits does not carry over. */
    let stored = null;
    try {
      stored = window.localStorage.getItem("mls-sidebar-v2");
    } catch (error) {
      stored = null;
    }
    if (stored !== "open") document.body.classList.add("sidebar-closed");

    if (toggle) {
      toggle.addEventListener("click", () => {
        const closed = document.body.classList.toggle("sidebar-closed");
        try {
          window.localStorage.setItem("mls-sidebar-v2", closed ? "closed" : "open");
        } catch (error) {
          /* Private mode: the choice just isn't remembered. */
        }
      });
    }

    /* Collapsible category groups. */
    sidebar.querySelectorAll(".sidebar-group-title").forEach((button) => {
      button.addEventListener("click", () => {
        const group = button.closest(".sidebar-group");
        const open = group.classList.toggle("is-open");
        button.setAttribute("aria-expanded", String(open));
      });
    });

    /* Keep the active entry in view on load. */
    const current = sidebar.querySelector(".sidebar-link.is-current");
    if (current) {
      window.requestAnimationFrame(() => {
        const box = current.getBoundingClientRect();
        if (box.top < 0 || box.bottom > window.innerHeight) {
          current.scrollIntoView({ block: "center" });
        }
      });
    }

    /* Tapping the backdrop closes the sidebar on small screens. */
    document.addEventListener("click", (event) => {
      if (window.innerWidth > 1080) return;
      if (document.body.classList.contains("sidebar-closed")) return;
      if (sidebar.contains(event.target) || event.target.closest("#sidebar-toggle")) return;
      document.body.classList.add("sidebar-closed");
    });
  }

  function wirePalette() {
    const overlay = document.getElementById("palette-overlay");
    const input = document.getElementById("palette-input");
    const results = document.getElementById("palette-results");
    let active = 0;
    let items = [];

    function paint(query) {
      items = search(query);
      active = 0;
      results.innerHTML = items.length
        ? items
            .map(
              (item, index) => `
                <a class="palette-item${index === 0 ? " is-active" : ""}" href="${item.href}"
                   role="option" aria-selected="${index === 0}" data-index="${index}">
                  <span class="palette-kind">${item.kind}</span>
                  <span class="palette-title">${item.title}</span>
                  <span class="palette-sub">${item.subtitle}</span>
                </a>
              `
            )
            .join("")
        : `<p class="palette-empty">Nothing matches “${query}”.</p>`;
    }

    function move(delta) {
      if (!items.length) return;
      active = (active + delta + items.length) % items.length;
      results.querySelectorAll(".palette-item").forEach((node, index) => {
        node.classList.toggle("is-active", index === active);
        node.setAttribute("aria-selected", String(index === active));
        if (index === active) node.scrollIntoView({ block: "nearest" });
      });
    }

    function open() {
      overlay.hidden = false;
      document.body.classList.add("palette-open");
      input.value = "";
      paint("");
      input.focus();
    }

    function close() {
      overlay.hidden = true;
      document.body.classList.remove("palette-open");
    }

    document.getElementById("topbar-search")?.addEventListener("click", open);
    document.getElementById("sidebar-search")?.addEventListener("click", open);

    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) close();
    });

    input.addEventListener("input", () => paint(input.value));

    input.addEventListener("keydown", (event) => {
      if (event.key === "ArrowDown") {
        event.preventDefault();
        move(1);
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        move(-1);
      } else if (event.key === "Enter") {
        event.preventDefault();
        const target = results.querySelectorAll(".palette-item")[active];
        if (target) window.location.href = target.getAttribute("href");
      } else if (event.key === "Escape") {
        close();
      }
    });

    document.addEventListener("keydown", (event) => {
      const typingTarget =
        event.target.tagName === "INPUT" ||
        event.target.tagName === "TEXTAREA" ||
        event.target.tagName === "SELECT";

      if ((event.key === "k" || event.key === "K") && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        overlay.hidden ? open() : close();
        return;
      }
      if (event.key === "/" && !typingTarget && overlay.hidden) {
        event.preventDefault();
        open();
        return;
      }
      if (event.key === "Escape" && !overlay.hidden) close();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
