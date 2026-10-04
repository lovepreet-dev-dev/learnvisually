(function () {
  const catalog = window.MLAlgorithms;
  const U = window.MLUtils;
  /* Resolved per render so charts follow the active theme. */
  const C = () => U.chartColors();

  if (!catalog || document.body.dataset.page !== "algorithm") {
    return;
  }

  const params = new URLSearchParams(window.location.search);
  const algorithmId = params.get("id");
  const definition =
    catalog.get(algorithmId) || Object.values(catalog.algorithms).sort((a, b) => a.order - b.order)[0];
  const root = U.qs("#algorithm-page-root");

  /* ── Default datasets ──────────────────────────────────────────
     These seed the editable tables. Nothing here is fixed: every lab
     copies its defaults into mutable state that the learner edits. */

  const knnDefaults = [
    { id: "A1", x: 1.8, y: 7.2, label: "A" },
    { id: "A2", x: 2.6, y: 6.4, label: "A" },
    { id: "A3", x: 3.1, y: 8.0, label: "A" },
    { id: "A4", x: 3.8, y: 6.9, label: "A" },
    { id: "A5", x: 4.2, y: 5.8, label: "A" },
    { id: "B1", x: 6.1, y: 3.8, label: "B" },
    { id: "B2", x: 7.2, y: 2.8, label: "B" },
    { id: "B3", x: 8.0, y: 4.3, label: "B" },
    { id: "B4", x: 7.3, y: 5.2, label: "B" },
    { id: "B5", x: 5.8, y: 2.4, label: "B" },
  ];

  const knnPresets = {
    "Well separated": knnDefaults,
    "Overlapping classes": [
      { id: "A1", x: 3.4, y: 5.6, label: "A" },
      { id: "A2", x: 4.2, y: 6.2, label: "A" },
      { id: "A3", x: 5.0, y: 5.1, label: "A" },
      { id: "A4", x: 4.6, y: 4.4, label: "A" },
      { id: "B1", x: 5.2, y: 5.8, label: "B" },
      { id: "B2", x: 5.9, y: 4.6, label: "B" },
      { id: "B3", x: 4.9, y: 3.7, label: "B" },
      { id: "B4", x: 6.4, y: 5.4, label: "B" },
    ],
    "One noisy outlier": [
      { id: "A1", x: 2.0, y: 7.0, label: "A" },
      { id: "A2", x: 2.8, y: 7.6, label: "A" },
      { id: "A3", x: 3.4, y: 6.8, label: "A" },
      { id: "A4", x: 2.4, y: 6.2, label: "A" },
      { id: "B1", x: 7.4, y: 2.8, label: "B" },
      { id: "B2", x: 8.0, y: 3.6, label: "B" },
      { id: "B3", x: 7.0, y: 2.0, label: "B" },
      { id: "Bx", x: 3.0, y: 7.1, label: "B" },
    ],
    "Imbalanced classes": [
      { id: "A1", x: 2.2, y: 6.4, label: "A" },
      { id: "A2", x: 3.0, y: 7.2, label: "A" },
      { id: "A3", x: 3.8, y: 6.0, label: "A" },
      { id: "A4", x: 2.6, y: 5.2, label: "A" },
      { id: "A5", x: 4.4, y: 6.8, label: "A" },
      { id: "A6", x: 3.4, y: 8.0, label: "A" },
      { id: "B1", x: 6.6, y: 3.4, label: "B" },
      { id: "B2", x: 7.4, y: 4.0, label: "B" },
    ],
  };

  const treeClassificationData = [
    { id: "S1", hours: 1.2, attendance: 2.4, label: 0 },
    { id: "S2", hours: 2.0, attendance: 3.2, label: 0 },
    { id: "S3", hours: 2.4, attendance: 4.8, label: 0 },
    { id: "S4", hours: 3.0, attendance: 6.1, label: 1 },
    { id: "S5", hours: 4.1, attendance: 6.6, label: 1 },
    { id: "S6", hours: 4.8, attendance: 7.2, label: 1 },
    { id: "S7", hours: 3.5, attendance: 3.6, label: 0 },
    { id: "S8", hours: 5.4, attendance: 5.8, label: 1 },
    { id: "S9", hours: 2.8, attendance: 5.8, label: 1 },
    { id: "S10", hours: 4.2, attendance: 4.0, label: 1 },
  ];

  const treeRegressionData = [
    { id: "R1", hours: 1.0, score: 42 },
    { id: "R2", hours: 1.8, score: 47 },
    { id: "R3", hours: 2.4, score: 51 },
    { id: "R4", hours: 3.1, score: 60 },
    { id: "R5", hours: 4.0, score: 68 },
    { id: "R6", hours: 4.8, score: 75 },
    { id: "R7", hours: 5.6, score: 83 },
    { id: "R8", hours: 6.3, score: 89 },
  ];

  const linearDatasets = {
    separable: [
      { id: "P1", x: 2.0, y: 7.1, label: 1 },
      { id: "P2", x: 2.8, y: 6.4, label: 1 },
      { id: "P3", x: 3.6, y: 7.8, label: 1 },
      { id: "P4", x: 4.4, y: 6.2, label: 1 },
      { id: "N1", x: 6.2, y: 2.8, label: -1 },
      { id: "N2", x: 7.2, y: 3.6, label: -1 },
      { id: "N3", x: 7.7, y: 2.0, label: -1 },
      { id: "N4", x: 8.2, y: 4.2, label: -1 },
    ],
    overlap: [
      { id: "P1", x: 2.2, y: 6.4, label: 1 },
      { id: "P2", x: 3.3, y: 5.8, label: 1 },
      { id: "P3", x: 4.4, y: 5.2, label: 1 },
      { id: "P4", x: 5.0, y: 4.7, label: 1 },
      { id: "N1", x: 4.8, y: 5.0, label: -1 },
      { id: "N2", x: 5.8, y: 4.2, label: -1 },
      { id: "N3", x: 6.6, y: 3.4, label: -1 },
      { id: "N4", x: 6.1, y: 5.4, label: -1 },
    ],
    xor: [
      { id: "P1", x: 2.0, y: 2.0, label: 1 },
      { id: "P2", x: 7.8, y: 8.0, label: 1 },
      { id: "P3", x: 2.5, y: 1.4, label: 1 },
      { id: "P4", x: 8.3, y: 8.5, label: 1 },
      { id: "N1", x: 2.1, y: 8.0, label: -1 },
      { id: "N2", x: 8.0, y: 2.1, label: -1 },
      { id: "N3", x: 1.4, y: 7.2, label: -1 },
      { id: "N4", x: 7.4, y: 2.6, label: -1 },
    ],
  };

  const partitionPoints = [
    { id: "P1", x: 1.4, y: 2.0 },
    { id: "P2", x: 2.0, y: 1.6 },
    { id: "P3", x: 2.4, y: 2.6 },
    { id: "P4", x: 3.1, y: 7.0 },
    { id: "P5", x: 3.6, y: 7.8 },
    { id: "P6", x: 4.2, y: 6.4 },
    { id: "P7", x: 6.5, y: 3.8 },
    { id: "P8", x: 7.2, y: 4.5 },
    { id: "P9", x: 7.8, y: 3.0 },
    { id: "P10", x: 8.5, y: 7.6 },
    { id: "P11", x: 8.9, y: 6.8 },
    { id: "P12", x: 8.0, y: 8.3 },
  ];

  const hierarchicalPoints = [
    { id: "A", x: 1.2, y: 2.0 },
    { id: "B", x: 1.9, y: 2.8 },
    { id: "C", x: 3.8, y: 7.0 },
    { id: "D", x: 4.5, y: 7.5 },
    { id: "E", x: 7.0, y: 3.0 },
    { id: "F", x: 7.8, y: 3.7 },
  ];

  const dbscanPoints = [
    { id: "A", x: 1.4, y: 2.0 },
    { id: "B", x: 1.8, y: 2.7 },
    { id: "C", x: 2.4, y: 1.8 },
    { id: "D", x: 2.6, y: 2.5 },
    { id: "E", x: 5.1, y: 6.0 },
    { id: "F", x: 5.6, y: 6.6 },
    { id: "G", x: 6.2, y: 6.1 },
    { id: "H", x: 6.4, y: 5.5 },
    { id: "I", x: 8.3, y: 2.0 },
    { id: "J", x: 8.9, y: 2.6 },
    { id: "K", x: 9.2, y: 1.7 },
    { id: "L", x: 4.2, y: 1.4 },
    { id: "M", x: 7.5, y: 8.1 },
  ];

  const spectralPoints = [
    { id: "S1", x: 1.5, y: 4.0 },
    { id: "S2", x: 2.3, y: 5.0 },
    { id: "S3", x: 3.3, y: 5.4 },
    { id: "S4", x: 4.1, y: 4.7 },
    { id: "S5", x: 5.8, y: 3.2 },
    { id: "S6", x: 6.7, y: 3.7 },
    { id: "S7", x: 7.4, y: 4.6 },
    { id: "S8", x: 6.1, y: 5.3 },
  ];

  /* Resolved on each call rather than once at load, so cluster colours
     follow the active theme instead of freezing the palette that
     happened to be active when the script first ran. */
  const clusterPaletteAt = (index) => {
    const palette = [C().a, C().b, C().c, C().d];
    return palette[((index % palette.length) + palette.length) % palette.length];
  };

  function pagePrefix() {
    return "./";
  }

  function categoryOf(categoryId) {
    return catalog.categories.find((category) => category.id === categoryId);
  }

  function algorithmHref(id) {
    return `${pagePrefix()}algorithm.html?id=${id}`;
  }

  function categoryHref(categoryId) {
    const category = categoryOf(categoryId);
    return `${pagePrefix()}${category.page}`;
  }

  function normalize(vector) {
    const total = vector.reduce((acc, value) => acc + value, 0);
    if (total <= 0) {
      return vector.map(() => 1 / vector.length);
    }
    return vector.map((value) => value / total);
  }

  /* A derivation step may supply either `tex` (typeset with KaTeX) or a
     plain `expr` string. Mixing both is fine — older cards keep working
     while newer ones get proper mathematical notation. */
  function renderDerivationChain(steps) {
    if (!steps || !steps.length) return '';
    return `<div class="derivation-chain">${steps.map((s, i) => `
      <div class="derive-line">
        <span class="derive-num">${i + 1}</span>
        <span class="derive-expr">${s.tex ? U.tex(s.tex) : s.expr}</span>
        ${s.result !== undefined && s.result !== '' ? `<span class="derive-result">${s.result}</span>` : ''}
      </div>
      ${s.note ? `<div class="derive-annotation">${s.note}</div>` : ''}
    `).join('')}</div>`;
  }

  function renderFormulaCards(container, cards) {
    container.innerHTML = `
      <div class="math-grid">
        ${cards.map((card) => `
          <article class="formula-card">
            <div class="formula-card-body">
              <h3>${card.title}</h3>
              <p>${card.description}</p>
              ${card.tex
                ? `<div class="formula-tex">${U.tex(card.tex, true)}</div>`
                : `<code class="formula">${card.formula}</code>`}
              ${renderDerivationChain(card.derivation)}
              ${card.insight ? `<div class="insight-box">${card.insight}</div>` : ''}
            </div>
          </article>
        `).join('')}
      </div>
    `;
  }

  /* Markup for the editable-dataset slot every lab now carries. The
     editor itself is mounted into `#${id}` by U.dataEditor. */
  function editorSlot(id) {
    return `<div class="editor-slot" id="${id}"></div>`;
  }

  const DRAG_HINT = `<div class="drag-hint">✥ Drag any point on the chart to move it — the table, formulas and result follow.</div>`;

  /* Shapes that make clustering behave in interestingly different ways.
     Offering them as presets lets a learner discover *why* k-means fails
     on rings and DBSCAN does not, rather than being told. */
  const clusterPresets = {
    "Three blobs": [
      { id: "P1", x: 1.4, y: 2.0 }, { id: "P2", x: 2.0, y: 1.6 }, { id: "P3", x: 2.4, y: 2.6 },
      { id: "P4", x: 3.1, y: 7.0 }, { id: "P5", x: 3.6, y: 7.8 }, { id: "P6", x: 4.2, y: 6.4 },
      { id: "P7", x: 6.5, y: 3.8 }, { id: "P8", x: 7.2, y: 4.5 }, { id: "P9", x: 7.8, y: 3.0 },
      { id: "P10", x: 8.5, y: 7.6 }, { id: "P11", x: 8.9, y: 6.8 }, { id: "P12", x: 8.0, y: 8.3 },
    ],
    "Two moons": [
      { id: "M1", x: 1.6, y: 5.0 }, { id: "M2", x: 2.4, y: 6.6 }, { id: "M3", x: 3.8, y: 7.4 },
      { id: "M4", x: 5.2, y: 7.0 }, { id: "M5", x: 6.0, y: 5.8 }, { id: "M6", x: 6.2, y: 4.6 },
      { id: "N1", x: 3.4, y: 4.2 }, { id: "N2", x: 4.6, y: 3.2 }, { id: "N3", x: 6.0, y: 2.6 },
      { id: "N4", x: 7.4, y: 2.8 }, { id: "N5", x: 8.4, y: 3.8 }, { id: "N6", x: 8.8, y: 5.0 },
    ],
    "Unequal sizes": [
      { id: "B1", x: 2.0, y: 2.2 }, { id: "B2", x: 2.6, y: 2.8 }, { id: "B3", x: 1.8, y: 3.0 },
      { id: "B4", x: 2.4, y: 1.6 }, { id: "B5", x: 3.0, y: 2.4 }, { id: "B6", x: 2.2, y: 2.6 },
      { id: "B7", x: 2.8, y: 3.2 }, { id: "B8", x: 1.6, y: 2.4 },
      { id: "S1", x: 7.8, y: 7.4 }, { id: "S2", x: 8.4, y: 8.0 },
    ],
    "Dense core plus noise": [
      { id: "C1", x: 4.6, y: 4.8 }, { id: "C2", x: 5.0, y: 5.4 }, { id: "C3", x: 5.4, y: 4.6 },
      { id: "C4", x: 4.8, y: 4.2 }, { id: "C5", x: 5.6, y: 5.2 }, { id: "C6", x: 5.2, y: 5.0 },
      { id: "X1", x: 1.2, y: 8.6 }, { id: "X2", x: 9.0, y: 1.4 }, { id: "X3", x: 8.8, y: 8.8 },
      { id: "X4", x: 1.0, y: 1.2 },
    ],
  };

  /* Every clustering lab shares the same {id, x, y} shape, so they share
     one editor factory too. */
  function pointsEditor(node, rows, onChange, options = {}) {
    return U.dataEditor(node, {
      title: options.title || "Data points",
      hint:
        options.hint ||
        "Type new coordinates, add points to thicken a region, or delete one to thin it out. The algorithm re-runs from scratch on every edit.",
      columns: [
        { key: "id", label: "ID", type: "text" },
        { key: "x", label: "x", min: 0, max: 10, step: 0.1 },
        { key: "y", label: "y", min: 0, max: 10, step: 0.1 },
      ],
      rows,
      presets: options.presets === null ? undefined : options.presets || clusterPresets,
      minRows: options.minRows || 3,
      newRow: (current) => ({
        id: `P${current.length + 1}`,
        x: Number(U.round(Math.random() * 9 + 0.5, 1)),
        y: Number(U.round(Math.random() * 9 + 0.5, 1)),
      }),
      onChange,
    });
  }

  /* Draw a labelled scatter point. Passing `onDrag` turns it into a
     handle the learner can pick up and move around the plot; the
     callback receives the new position in data coordinates. */
  function drawScatterPoint(plot, chart, point, color, radius = 7, onDrag = null) {
    const circle = U.svgEl("circle", {
      cx: chart.xScale(point.x),
      cy: chart.yScale(point.y),
      r: radius,
      fill: color,
      opacity: "0.92",
    });
    plot.appendChild(circle);
    if (onDrag) {
      U.draggable(circle, plot, chart, (position) => onDrag(point, position));
    }
    plot.appendChild(
      U.svgEl("text", {
        x: chart.xScale(point.x) + 10,
        y: chart.yScale(point.y) - 9,
        class: "svg-label",
      })
    ).textContent = point.id;
    return circle;
  }

  /* Search engines get one shell for all 42 concepts, so the breadcrumb
     trail and the page's subject have to be described per render too. */
  function writeStructuredData(canonical, category) {
    const site = "https://www.learnvisually.fun";
    const graph = {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "BreadcrumbList",
          itemListElement: [
            { "@type": "ListItem", position: 1, name: "Home", item: `${site}/` },
            {
              "@type": "ListItem",
              position: 2,
              name: category.title,
              item: `${site}/pages/${category.page}`,
            },
            { "@type": "ListItem", position: 3, name: definition.title, item: canonical },
          ],
        },
        {
          "@type": "LearningResource",
          name: definition.title,
          description: definition.summary,
          url: canonical,
          inLanguage: "en",
          learningResourceType: "Interactive concept page",
          educationalLevel: "Undergraduate",
          isAccessibleForFree: true,
          image: `${site}/assets/og-image.jpg`,
          about: { "@type": "Thing", name: category.title },
          publisher: { "@type": "EducationalOrganization", name: "Machine Learning Studio", url: `${site}/` },
        },
      ],
    };
    let node = document.getElementById("ld-json");
    if (!node) {
      node = document.createElement("script");
      node.type = "application/ld+json";
      node.id = "ld-json";
      document.head.appendChild(node);
    }
    node.textContent = JSON.stringify(graph);
  }

  function renderShell() {
    const category = categoryOf(definition.category);
    document.title = `${definition.title} | Machine Learning Studio`;

    /* One HTML shell serves every concept, so the per-page metadata has
       to be written at render time or every page would share the same
       description and share card. */
    const canonical = `https://www.learnvisually.fun/pages/algorithm.html?id=${definition.id}`;
    const metaMap = {
      'meta[name="description"]': ["content", definition.summary],
      'meta[property="og:title"]': ["content", `${definition.title} | Machine Learning Studio`],
      'meta[property="og:description"]': ["content", definition.summary],
      'meta[property="og:url"]': ["content", canonical],
      'meta[name="twitter:title"]': ["content", `${definition.title} | Machine Learning Studio`],
      'meta[name="twitter:description"]': ["content", definition.summary],
      'link[rel="canonical"]': ["href", canonical],
    };
    Object.entries(metaMap).forEach(([selector, [attribute, value]]) => {
      const node = document.head.querySelector(selector);
      if (node) node.setAttribute(attribute, value);
    });

    writeStructuredData(canonical, category);

    root.innerHTML = `
      <section class="page-hero algorithm-shell">
        <div class="breadcrumbs">
          <a href="../index.html">Home</a>
          <span>/</span>
          <a href="${categoryHref(definition.category)}">${category.title}</a>
          <span>/</span>
          <span>${definition.title}</span>
        </div>
        <div class="eyebrow">${category.title}</div>
        <h1>${definition.title}</h1>
        <p class="page-lead">${definition.subtitle}</p>
        <div class="hero-meta">
          ${(definition.heroMeta || [
            "✎ <b>Editable</b> input data",
            "∑ <b>Live</b> formulas with your numbers",
            "▤ <b>Step-by-step</b> logic trail",
            "◈ Deep-dive notes below",
          ]).map((item) => `<span>${item}</span>`).join("")}
        </div>
      </section>

      <section id="algorithm-workspace"></section>

      <section class="lab">
        <div class="section-header">
          <div>
            <div class="eyebrow">Detailed Explanation</div>
            <h2>Interpretation, math, usage, and failure modes</h2>
          </div>
        </div>
        <div class="detail-grid">
          <article class="detail-card">
            <h3>Intuition</h3>
            <p>${definition.detail.intuition}</p>
            ${definition.detail.analogy ? `<ul>${definition.detail.analogy.map(a => `<li>${a}</li>`).join('')}</ul>` : ''}
          </article>
          <article class="detail-card">
            <h3>Core Mathematics</h3>
            <p>${definition.detail.math}</p>
            ${definition.detail.keyFormulas
              ? `<div class="detail-formulas">${definition.detail.keyFormulas
                  .map((f) => `<div class="formula-tex">${U.tex(f, true)}</div>`)
                  .join('')}</div>`
              : ''}
          </article>
          <article class="detail-card">
            <h3>When To Use It</h3>
            <p>${definition.detail.use}</p>
          </article>
          <article class="detail-card">
            <h3>Watch Out For</h3>
            <p>${definition.detail.caution}</p>
          </article>
        </div>
      </section>

    `;

    const navMap = {
      foundations: "./introduction.html",
      classification: "./classification.html",
      "graphical-models": "./graphical-models.html",
      clustering: "./clustering.html",
      neural: "./neural.html",
    };
    U.qsa(".nav-links a").forEach((link) => {
      if (link.getAttribute("href") === navMap[definition.category]) {
        link.setAttribute("aria-current", "page");
      }
    });
  }

  function mountSection(rootNode, title, intro, content) {
    rootNode.innerHTML = `
      <section class="lab">
        <div class="section-header">
          <div>
            <div class="eyebrow">Interactive Lab</div>
            <h2>${title}</h2>
          </div>
        </div>
        <p class="section-intro">${intro}</p>
        ${content}
      </section>
    `;

    /* Callouts carry the running result on labs that have no metric
       tiles, so they need announcing when they change. Marked here once
       rather than in every lab's markup. */
    U.qsa(".callout", rootNode).forEach((node) => {
      node.setAttribute("aria-live", "polite");
    });
  }

  /* ── Overview of machine learning ─────────────────────────────
     The other concept pages let you manipulate an algorithm; this one
     lets you manipulate the *dataset budget*, because that is the
     decision this stage of a project actually turns on. */
  function mountConceptOverview(rootNode) {
    mountSection(
      rootNode,
      "The end-to-end workflow, sized to your data",
      "Set how much labelled data you actually have and how you want to spend it. The split arithmetic, the reliability of your test estimate, and the choice of validation strategy all follow from those numbers — this page computes them rather than describing them.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-group">
              <label for="overview-task">Task family</label>
              <select id="overview-task">
                <option value="classification">Classification</option>
                <option value="regression">Regression</option>
                <option value="clustering">Clustering</option>
                <option value="sequence">Sequence modelling</option>
              </select>
            </div>
            <div class="control-group">
              <label for="overview-n">Total labelled examples n</label>
              <input id="overview-n" type="number" min="20" max="2000000" step="10" value="1000" />
            </div>
            <div class="control-grid">
              <div class="control-group">
                <label for="overview-test">Test share</label>
                <div class="range-row">
                  <input id="overview-test" type="range" min="0.05" max="0.5" step="0.01" value="0.2" />
                  <span class="range-value" id="overview-test-value">0.20</span>
                </div>
              </div>
              <div class="control-group">
                <label for="overview-val">Validation share</label>
                <div class="range-row">
                  <input id="overview-val" type="range" min="0" max="0.4" step="0.01" value="0.2" />
                  <span class="range-value" id="overview-val-value">0.20</span>
                </div>
              </div>
            </div>
            <div class="control-group">
              <label for="overview-acc">Expected accuracy p</label>
              <div class="range-row">
                <input id="overview-acc" type="range" min="0.5" max="0.99" step="0.01" value="0.85" />
                <span class="range-value" id="overview-acc-value">0.85</span>
              </div>
            </div>
            <div class="control-group">
              <label for="overview-folds">Cross-validation folds k</label>
              <input id="overview-folds" type="number" min="2" max="20" value="5" />
            </div>
            <div class="step-controls">
              <button class="button primary" id="overview-step">Reveal next stage</button>
              <button class="button secondary" id="overview-reset">Reset</button>
            </div>
            <div class="callout" id="overview-callout"></div>
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg id="overview-pipeline" viewBox="0 0 620 210" aria-label="Machine learning workflow"></svg>
            </div>
            <div class="plot-card">
              <svg id="overview-split" viewBox="0 0 560 200" aria-label="Data split"></svg>
              <div class="legend">
                <span><i style="background:#0d7a72"></i> Train</span>
                <span><i style="background:#2563a8"></i> Validation</span>
                <span><i style="background:#c2410c"></i> Test</span>
              </div>
            </div>
            <div class="readout"><div class="output-grid" id="overview-output"></div></div>
            <div class="equation-card" id="overview-math"></div>
            <div class="steps"><ol id="overview-steps"></ol></div>
          </div>
        </div>
      `
    );

    const taskInput = rootNode.querySelector("#overview-task");
    const nInput = rootNode.querySelector("#overview-n");
    const testInput = rootNode.querySelector("#overview-test");
    const testValue = rootNode.querySelector("#overview-test-value");
    const valInput = rootNode.querySelector("#overview-val");
    const valValue = rootNode.querySelector("#overview-val-value");
    const accInput = rootNode.querySelector("#overview-acc");
    const accValue = rootNode.querySelector("#overview-acc-value");
    const foldsInput = rootNode.querySelector("#overview-folds");
    const stepButton = rootNode.querySelector("#overview-step");
    const resetButton = rootNode.querySelector("#overview-reset");
    const callout = rootNode.querySelector("#overview-callout");
    const pipeline = rootNode.querySelector("#overview-pipeline");
    const splitPlot = rootNode.querySelector("#overview-split");
    const output = rootNode.querySelector("#overview-output");
    const mathNode = rootNode.querySelector("#overview-math");
    const stepsNode = rootNode.querySelector("#overview-steps");

    let revealed = 0;

    const taskInfo = {
      classification: {
        objective: "L = -\\frac{1}{n}\\sum_i \\sum_c y_{ic}\\log \\hat{p}_{ic}",
        objectiveName: "Cross-entropy loss",
        metric: "Accuracy, precision/recall, ROC-AUC",
        note: "Labels are discrete, so the model outputs a probability per class and is scored on how much mass it puts on the truth.",
        supervised: true,
      },
      regression: {
        objective: "L = \\frac{1}{n}\\sum_i (y_i - \\hat{y}_i)^2",
        objectiveName: "Mean squared error",
        metric: "RMSE, MAE, R²",
        note: "Targets are continuous, so error is measured as a distance rather than a hit or miss.",
        supervised: true,
      },
      clustering: {
        objective: "L = \\sum_{k}\\sum_{i \\in C_k} \\lVert \\mathbf{x}_i - \\boldsymbol{\\mu}_k \\rVert^2",
        objectiveName: "Within-cluster sum of squares",
        metric: "Silhouette, Davies–Bouldin (no ground truth available)",
        note: "There are no labels at all, so the objective is defined purely by the geometry of the data — and there is no test set in the usual sense.",
        supervised: false,
      },
      sequence: {
        objective: "L = -\\sum_t \\log p(o_t \\mid o_{<t})",
        objectiveName: "Negative log-likelihood",
        metric: "Perplexity, per-step accuracy",
        note: "Examples are not independent, so the split must respect time — a random shuffle would let the model see the future.",
        supervised: true,
      },
    };

    function render() {
      const task = taskInput.value;
      const info = taskInfo[task];
      const n = Math.max(Number(nInput.value) || 100, 20);
      let testShare = Number(testInput.value);
      let valShare = Number(valInput.value);
      const accuracy = Number(accInput.value);
      const folds = U.clamp(Math.round(Number(foldsInput.value) || 5), 2, 20);

      testValue.textContent = testShare.toFixed(2);
      valValue.textContent = valShare.toFixed(2);
      accValue.textContent = accuracy.toFixed(2);

      if (testShare + valShare > 0.9) {
        valShare = Math.max(0, 0.9 - testShare);
        valValue.textContent = valShare.toFixed(2);
      }

      const nTest = Math.round(n * testShare);
      const nVal = Math.round(n * valShare);
      const nTrain = n - nTest - nVal;

      /* Standard error of a proportion — the honest width of a test-set
         accuracy estimate. This is the number people forget. */
      const stderr = nTest > 0 ? Math.sqrt((accuracy * (1 - accuracy)) / nTest) : NaN;
      const margin = 1.96 * stderr;

      const cvTrain = Math.round(n * (1 - 1 / folds));
      const cvVal = Math.round(n / folds);

      callout.innerHTML = info.supervised
        ? `A test set of <strong>${nTest}</strong> examples gives a 95% confidence interval of roughly <strong>±${U.round(margin * 100, 1)}pp</strong> around an accuracy of ${U.round(accuracy * 100, 0)}%. ${margin > 0.05 ? "That is too wide to distinguish most model changes — enlarge the test share or collect more data." : "That is tight enough to compare models meaningfully."}`
        : `Clustering has no labels, so there is no held-out accuracy to estimate. Validation means stability and internal criteria, not a test score.`;

      U.renderMetrics(output, [
        { label: "Train", value: `${nTrain} (${U.round((nTrain / n) * 100, 0)}%)` },
        { label: "Validation", value: `${nVal} (${U.round((nVal / n) * 100, 0)}%)` },
        { label: "Test", value: `${nTest} (${U.round((nTest / n) * 100, 0)}%)` },
        { label: "95% CI half-width", value: info.supervised && nTest > 0 ? `±${U.round(margin * 100, 2)}pp` : "n/a" },
      ]);

      renderFormulaCards(mathNode, [
        {
          title: `${info.objectiveName} — what the model minimises`,
          description: info.note,
          tex: info.objective,
          derivation: [
            {
              tex: `n_{\\text{train}} = ${n} - ${nTest} - ${nVal} = ${nTrain}`,
              result: String(nTrain),
              note: "Only this portion is ever used to fit parameters.",
            },
            { tex: `\\text{evaluation metric: } \\text{${info.metric}}`, result: "" },
          ],
          insight:
            "<strong>The objective and the metric are rarely the same thing.</strong> You minimise cross-entropy because it is differentiable; you report accuracy or AUC because that is what someone cares about. Optimising one does not guarantee the other improves.",
        },
        {
          title: "How reliable is your test estimate?",
          description:
            "A test accuracy is an estimate from a finite sample, so it comes with a standard error. Below roughly a few hundred test examples, that error swamps most real differences between models.",
          tex: "\\mathrm{SE} = \\sqrt{\\frac{p(1-p)}{n_{\\text{test}}}}, \\qquad \\text{CI}_{95\\%} = p \\pm 1.96\\,\\mathrm{SE}",
          derivation: info.supervised && nTest > 0
            ? [
                {
                  tex: `\\mathrm{SE} = \\sqrt{\\frac{${U.texNum(accuracy, 2)}(1 - ${U.texNum(accuracy, 2)})}{${nTest}}} = \\sqrt{\\frac{${U.texNum(accuracy * (1 - accuracy), 4)}}{${nTest}}}`,
                  result: U.round(stderr, 5),
                },
                {
                  tex: `\\text{CI} = ${U.texNum(accuracy, 2)} \\pm ${U.texNum(margin, 4)} = [${U.texNum(accuracy - margin, 4)},\\; ${U.texNum(accuracy + margin, 4)}]`,
                  result: `±${U.round(margin * 100, 2)}pp`,
                  note: "Drag the test share up and watch the interval tighten — the price is fewer training examples.",
                },
              ]
            : [{ tex: "\\text{no labelled test set in this task family}", result: "" }],
          insight:
            "<strong>This is the real reason for the 80/20 rule of thumb.</strong> It is not tradition — it is the point where the test set is usually large enough to measure something and the training set is still large enough to learn something.",
        },
        {
          title: `${folds}-fold cross-validation`,
          description:
            "When data is scarce, holding out a fixed validation set wastes it. Cross-validation reuses every example as both training and validation data, at the cost of k times the compute.",
          tex: "\\mathrm{CV}_k = \\frac{1}{k}\\sum_{j=1}^{k} L\\!\\left(f^{(-j)}, D_j\\right)",
          derivation: [
            {
              tex: `\\text{per fold: } n_{\\text{train}} = ${cvTrain}, \\quad n_{\\text{val}} = ${cvVal}`,
              result: "",
              note: `Each model trains on ${U.round((1 - 1 / folds) * 100, 0)}% of the data and is scored on the remaining ${U.round((1 / folds) * 100, 0)}%.`,
            },
            {
              tex: `\\text{models trained} = ${folds}`,
              result: String(folds),
              note: folds >= 10
                ? "High k means low bias but high variance and high cost. Leave-one-out is the extreme case."
                : "k = 5 or 10 is the usual compromise between bias, variance and compute.",
            },
          ],
          insight:
            "<strong>The test set stays out of cross-validation.</strong> If you tune hyperparameters against the test set — even indirectly, by picking the best of many runs — you have turned it into a validation set and your reported number is optimistic.",
        },
      ]);

      const steps = [
        `<strong>Define the task.</strong> This is a <strong>${task}</strong> problem, so the objective is ${info.objectiveName.toLowerCase()}.`,
        `<strong>Split the data.</strong> ${n} examples become ${nTrain} train / ${nVal} validation / ${nTest} test.`,
        `<strong>Fit on train only.</strong> Parameters are estimated from ${nTrain} examples and nothing else.`,
        `<strong>Tune on validation.</strong> Hyperparameters are chosen against ${nVal > 0 ? `${nVal} held-out examples` : `${folds}-fold cross-validation, since no fixed validation set was reserved`}.`,
        `<strong>Report on test, once.</strong> ${info.supervised ? `The ${nTest}-example test set gives ±${U.round(margin * 100, 1)}pp of resolution.` : "Clustering is judged by internal criteria and stability instead."}`,
      ];
      U.renderSteps(stepsNode, steps, revealed);

      /* ── pipeline diagram ── */
      U.clear(pipeline);
      const stages = [
        { label: "Raw data", sub: `${n} examples` },
        { label: "Split", sub: `${nTrain}/${nVal}/${nTest}` },
        { label: "Fit", sub: info.objectiveName.split(" ")[0] },
        { label: "Validate", sub: nVal > 0 ? "holdout" : `${folds}-fold CV` },
        { label: "Test once", sub: info.supervised ? `±${U.round(margin * 100, 1)}pp` : "internal" },
      ];
      const boxWidth = 104;
      const gap = 24;
      stages.forEach((stage, index) => {
        const x = 14 + index * (boxWidth + gap);
        const active = index < revealed;
        pipeline.appendChild(
          U.svgEl("rect", {
            x,
            y: 62,
            width: boxWidth,
            height: 72,
            rx: 12,
            fill: active ? C().a : C().plotBg,
            stroke: active ? C().a : C().faint,
            "stroke-width": "2",
          })
        );
        pipeline.appendChild(
          U.svgEl("text", {
            x: x + boxWidth / 2,
            y: 92,
            "text-anchor": "middle",
            "font-size": "12.5",
            "font-weight": "700",
            fill: active ? C().onFill : C().ink,
          })
        ).textContent = stage.label;
        pipeline.appendChild(
          U.svgEl("text", {
            x: x + boxWidth / 2,
            y: 111,
            "text-anchor": "middle",
            "font-size": "10.5",
            "font-family": "var(--mono)",
            fill: active ? U.tint(C().onFill, 0.85) : C().neutral,
          })
        ).textContent = stage.sub;
        if (index < stages.length - 1) {
          const ax = x + boxWidth + 4;
          pipeline.appendChild(
            U.svgEl("path", {
              d: `M ${ax} 98 L ${ax + gap - 8} 98 M ${ax + gap - 14} 93 L ${ax + gap - 8} 98 L ${ax + gap - 14} 103`,
              stroke: index < revealed - 1 ? C().a : C().faint,
              "stroke-width": "2",
              fill: "none",
            })
          );
        }
      });
      pipeline.appendChild(
        U.svgEl("text", { x: 14, y: 32, class: "svg-title" })
      ).textContent = "Workflow — press “Reveal next stage” to walk it";
      pipeline.appendChild(
        U.svgEl("text", { x: 14, y: 172, class: "svg-label" })
      ).textContent = "Information only ever flows left to right. Any leak backwards inflates your reported score.";

      /* ── split bar ── */
      U.clear(splitPlot);
      const totalWidth = 520;
      const segments = [
        { label: "Train", count: nTrain, color: C().a },
        { label: "Validation", count: nVal, color: C().c },
        { label: "Test", count: nTest, color: C().b },
      ];
      splitPlot.appendChild(U.svgEl("text", { x: 20, y: 28, class: "svg-title" })).textContent =
        `Where your ${n} examples go`;
      let cursorX = 20;
      segments.forEach((segment) => {
        const width = (segment.count / n) * totalWidth;
        if (width <= 0) return;
        splitPlot.appendChild(
          U.svgEl("rect", { x: cursorX, y: 56, width, height: 54, fill: segment.color, rx: 6, opacity: "0.92" })
        );
        if (width > 52) {
          splitPlot.appendChild(
            U.svgEl("text", {
              x: cursorX + width / 2,
              y: 80,
              "text-anchor": "middle",
              fill: C().onFill,
              "font-size": "12",
              "font-weight": "700",
            })
          ).textContent = segment.label;
          splitPlot.appendChild(
            U.svgEl("text", {
              x: cursorX + width / 2,
              y: 97,
              "text-anchor": "middle",
              fill: U.tint(C().plotBg, 0.92),
              "font-size": "11",
              "font-family": "var(--mono)",
            })
          ).textContent = String(segment.count);
        }
        cursorX += width + 2;
      });

      /* Confidence interval ruler under the bar. */
      if (info.supervised && nTest > 0) {
        const ciWidth = U.clamp(margin * 8 * totalWidth, 4, totalWidth);
        const centre = 20 + totalWidth / 2;
        splitPlot.appendChild(
          U.svgEl("line", {
            x1: centre - ciWidth / 2,
            y1: 148,
            x2: centre + ciWidth / 2,
            y2: 148,
            stroke: C().danger,
            "stroke-width": "4",
            "stroke-linecap": "round",
          })
        );
        splitPlot.appendChild(
          U.svgEl("circle", { cx: centre, cy: 148, r: 5, fill: C().ink })
        );
        splitPlot.appendChild(
          U.svgEl("text", { x: centre, y: 172, class: "svg-label", "text-anchor": "middle" })
        ).textContent = `95% CI on test accuracy: ±${U.round(margin * 100, 2)} percentage points`;
        splitPlot.appendChild(
          U.svgEl("text", { x: 20, y: 134, class: "svg-label" })
        ).textContent = "Resolution of your test estimate (wider = less able to tell models apart)";
      }
    }

    stepButton.addEventListener("click", () => {
      revealed = Math.min(revealed + 1, 5);
      render();
    });
    resetButton.addEventListener("click", () => {
      revealed = 0;
      render();
    });
    [taskInput, nInput, testInput, valInput, accInput, foldsInput].forEach((input) =>
      input.addEventListener("input", render)
    );
    U.onRedraw(render);
    render();
  }

  /* ── Different forms of learning ──────────────────────────────
     Rendered as one dataset whose label coverage you control, because
     supervised / semi-supervised / unsupervised is a continuum, not
     three unrelated boxes. */
  function mountConceptForms(rootNode) {
    mountSection(
      rootNode,
      "One dataset, four learning paradigms",
      "These are not four unrelated subjects — they are four answers to the question “how much supervision do I have?”. Slide the label coverage from 100% to 0% and watch the same dataset turn from a supervised problem into a semi-supervised one and finally into a clustering problem.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-group">
              <label for="forms-mode">Paradigm</label>
              <select id="forms-mode">
                <option value="supervised">Supervised</option>
                <option value="semi">Semi-supervised</option>
                <option value="unsupervised">Unsupervised</option>
                <option value="reinforcement">Reinforcement</option>
                <option value="generative">Generative</option>
              </select>
            </div>
            <div class="control-group">
              <label for="forms-coverage">Label coverage</label>
              <div class="range-row">
                <input id="forms-coverage" type="range" min="0" max="1" step="0.05" value="1" />
                <span class="range-value" id="forms-coverage-value">100%</span>
              </div>
            </div>
            <div class="control-group">
              <label for="forms-n">Examples n</label>
              <input id="forms-n" type="number" min="8" max="240" step="4" value="60" />
            </div>
            <div class="control-group">
              <label for="forms-cost">Cost per label (currency)</label>
              <input id="forms-cost" type="number" min="0" step="0.5" value="2" />
            </div>
            <div class="control-group">
              <label for="forms-seed">Dataset seed</label>
              <input id="forms-seed" type="number" min="1" max="999" value="7" />
            </div>
            <div class="step-controls">
              <button class="button primary" id="forms-step">Reveal next point</button>
              <button class="button secondary" id="forms-reset">Reset</button>
            </div>
            <div class="callout" id="forms-callout"></div>
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg id="forms-plot" viewBox="0 0 560 280" aria-label="Labelled versus unlabelled data"></svg>
              <div class="legend">
                <span><i style="background:#0d7a72"></i> Labelled class A</span>
                <span><i style="background:#c2410c"></i> Labelled class B</span>
                <span><i style="background:#c3ccd6"></i> Unlabelled</span>
              </div>
            </div>
            <div class="readout"><div class="output-grid" id="forms-output"></div></div>
            <div class="equation-card" id="forms-math"></div>
            <div class="steps"><ol id="forms-steps"></ol></div>
          </div>
        </div>
      `
    );

    const modeInput = rootNode.querySelector("#forms-mode");
    const coverageInput = rootNode.querySelector("#forms-coverage");
    const coverageValue = rootNode.querySelector("#forms-coverage-value");
    const nInput = rootNode.querySelector("#forms-n");
    const costInput = rootNode.querySelector("#forms-cost");
    const seedInput = rootNode.querySelector("#forms-seed");
    const stepButton = rootNode.querySelector("#forms-step");
    const resetButton = rootNode.querySelector("#forms-reset");
    const callout = rootNode.querySelector("#forms-callout");
    const plot = rootNode.querySelector("#forms-plot");
    const output = rootNode.querySelector("#forms-output");
    const mathNode = rootNode.querySelector("#forms-math");
    const stepsNode = rootNode.querySelector("#forms-steps");

    let revealed = 0;

    const paradigms = {
      supervised: {
        title: "Supervised learning",
        tex: "\\hat{f} = \\operatorname*{arg\\,min}_{f \\in \\mathcal{F}} \\; \\frac{1}{n}\\sum_{i=1}^{n} L\\big(f(\\mathbf{x}_i),\\, y_i\\big)",
        signal: "Every example carries a target.",
        coverage: 1,
        insight:
          "<strong>Labels are the bottleneck, not the algorithm.</strong> Doubling your labelled data usually beats swapping the model. Set the cost per label above and watch what full coverage actually costs.",
      },
      semi: {
        title: "Semi-supervised learning",
        tex: "\\hat{f} = \\operatorname*{arg\\,min}_{f} \\; \\underbrace{\\sum_{i \\in \\mathcal{L}} L(f(\\mathbf{x}_i), y_i)}_{\\text{labelled}} + \\lambda \\underbrace{\\sum_{j \\in \\mathcal{U}} R(f, \\mathbf{x}_j)}_{\\text{unlabelled structure}}",
        signal: "A few labels plus a lot of unlabelled structure.",
        coverage: 0.2,
        insight:
          "<strong>The unlabelled points still tell you something</strong> — where the data is dense, and therefore where a decision boundary probably should not go. That is what the regulariser R encodes.",
      },
      unsupervised: {
        title: "Unsupervised learning",
        tex: "\\hat{\\theta} = \\operatorname*{arg\\,min}_{\\theta} \\; \\sum_{k}\\sum_{i \\in C_k} \\lVert \\mathbf{x}_i - \\boldsymbol{\\mu}_k \\rVert^2",
        signal: "No targets at all — only the geometry of the inputs.",
        coverage: 0,
        insight:
          "<strong>Without labels there is no single right answer.</strong> Different objectives give different groupings of the same points, and none of them is wrong — which is why evaluating clustering is genuinely hard.",
      },
      reinforcement: {
        title: "Reinforcement learning",
        tex: "\\pi^{*} = \\operatorname*{arg\\,max}_{\\pi} \\; \\mathbb{E}_{\\pi}\\!\\left[\\sum_{t=0}^{\\infty} \\gamma^{t} r_t\\right]",
        signal: "A delayed, scalar reward instead of a per-example target.",
        coverage: 0,
        insight:
          "<strong>The supervision is delayed and sparse.</strong> You are told the score at the end, not the right move at each step — which is the credit assignment problem that makes RL so much harder than supervised learning.",
      },
      generative: {
        title: "Generative modelling",
        tex: "\\hat{\\theta} = \\operatorname*{arg\\,max}_{\\theta} \\; \\sum_{i=1}^{n} \\log p_\\theta(\\mathbf{x}_i)",
        signal: "Model the distribution of the inputs themselves.",
        coverage: 0,
        insight:
          "<strong>Learning p(x) rather than p(y|x)</strong> means you can sample new data, detect outliers as low-probability points, and fill in missing features — all from the same fitted model.",
      },
    };

    function dataset() {
      const n = U.clamp(Math.round(Number(nInput.value) || 60), 8, 240);
      const rng = U.seededRandom(Number(seedInput.value) || 7);
      return Array.from({ length: n }, (_, index) => {
        const cls = index % 2;
        return {
          id: `p${index}`,
          x: U.clamp(U.sampleNormal(rng, cls ? 6.8 : 3.2, 1.15), 0.3, 9.7),
          y: U.clamp(U.sampleNormal(rng, cls ? 3.4 : 6.6, 1.15), 0.3, 9.7),
          label: cls,
        };
      });
    }

    function render() {
      const mode = modeInput.value;
      const info = paradigms[mode];
      const points = dataset();
      const n = points.length;
      const coverage = Number(coverageInput.value);
      const cost = Number(costInput.value) || 0;
      coverageValue.textContent = `${Math.round(coverage * 100)}%`;

      const labelledCount = Math.round(n * coverage);
      const labelled = new Set(points.slice(0, labelledCount).map((point) => point.id));

      const impliedMode =
        coverage >= 0.95 ? "supervised" : coverage <= 0.02 ? "unsupervised" : "semi-supervised";

      callout.innerHTML = `With <strong>${labelledCount}</strong> of ${n} examples labelled (${Math.round(coverage * 100)}%), this dataset is a <strong>${impliedMode}</strong> problem. Labelling all of it would cost <strong>${U.round(n * cost, 2)}</strong>; you have spent <strong>${U.round(labelledCount * cost, 2)}</strong>.`;

      U.renderMetrics(output, [
        { label: "Labelled", value: `${labelledCount} / ${n}` },
        { label: "Unlabelled", value: String(n - labelledCount) },
        { label: "Labelling cost", value: U.round(labelledCount * cost, 2) },
        { label: "Regime", value: impliedMode },
      ]);

      renderFormulaCards(mathNode, [
        {
          title: `${info.title} — the objective`,
          description: info.signal,
          tex: info.tex,
          derivation: [
            {
              tex: `|\\mathcal{L}| = ${labelledCount}, \\qquad |\\mathcal{U}| = ${n - labelledCount}`,
              result: `${Math.round(coverage * 100)}% labelled`,
              note: "Slide the coverage control and watch which term of the objective has anything to sum over.",
            },
          ],
          insight: info.insight,
        },
        {
          title: "The continuum",
          description:
            "Supervised and unsupervised learning are the two endpoints of a single axis. Everything in between is semi-supervised, and in practice almost all real datasets live in between.",
          tex: "\\mathcal{D} = \\underbrace{\\{(\\mathbf{x}_i, y_i)\\}_{i \\in \\mathcal{L}}}_{\\text{labelled}} \\;\\cup\\; \\underbrace{\\{\\mathbf{x}_j\\}_{j \\in \\mathcal{U}}}_{\\text{unlabelled}}",
          derivation: [
            {
              tex: `\\text{coverage} = \\frac{|\\mathcal{L}|}{|\\mathcal{L}| + |\\mathcal{U}|} = \\frac{${labelledCount}}{${n}} = ${U.texNum(coverage, 2)}`,
              result: `${Math.round(coverage * 100)}%`,
            },
            {
              tex: `\\text{cost} = |\\mathcal{L}| \\times c = ${labelledCount} \\times ${U.texNum(cost, 2)}`,
              result: U.round(labelledCount * cost, 2),
              note: "Labelling is usually the dominant cost of a supervised project, which is exactly why the semi-supervised middle ground matters commercially.",
            },
          ],
          insight:
            "<strong>Drag the slider to zero.</strong> The points do not move — only your knowledge of them changes. The structure that clustering finds was there the whole time; supervision just tells you what to call it.",
        },
      ]);

      U.renderSteps(
        stepsNode,
        [
          `<strong>Look at the inputs.</strong> ${n} points sit in feature space regardless of what you know about them.`,
          `<strong>Count the supervision.</strong> ${labelledCount} carry a target; ${n - labelledCount} do not.`,
          `<strong>Pick the objective.</strong> ${info.title} minimises the expression in the formula card above.`,
          `<strong>Notice what changes.</strong> Moving coverage from 100% to 0% turns the same data from ${impliedMode === "supervised" ? "a classification problem into a clustering problem" : "one regime into another"} without touching a single coordinate.`,
        ],
        revealed
      );

      const chart = U.makeChart(plot, {
        xDomain: [0, 10],
        yDomain: [0, 10],
        title: `${info.title} — grey points have no target`,
      });
      points.forEach((point) => {
        const isLabelled = labelled.has(point.id);
        plot.appendChild(
          U.svgEl("circle", {
            cx: chart.xScale(point.x),
            cy: chart.yScale(point.y),
            r: isLabelled ? 6 : 4.5,
            fill: isLabelled ? (point.label ? C().b : C().a) : C().faint,
            stroke: isLabelled ? "#fff" : "none",
            "stroke-width": "1.5",
            opacity: isLabelled ? "0.95" : "0.85",
          })
        );
      });
    }

    stepButton.addEventListener("click", () => {
      revealed = Math.min(revealed + 1, 4);
      render();
    });
    resetButton.addEventListener("click", () => {
      revealed = 0;
      render();
    });
    modeInput.addEventListener("change", () => {
      coverageInput.value = paradigms[modeInput.value].coverage;
      render();
    });
    [coverageInput, nInput, costInput, seedInput].forEach((input) => input.addEventListener("input", render));
    U.onRedraw(render);
    render();
  }

  /* ── Applications and tooling ─────────────────────────────────
     Turned into a weighted decision matrix so the "which algorithm
     should I use" question becomes arithmetic you can inspect and
     disagree with, rather than a list to memorise. */
  function mountConceptTools(rootNode) {
    const criteria = [
      { key: "interpretability", label: "Interpretability" },
      { key: "scale", label: "Scales to large data" },
      { key: "uncertainty", label: "Calibrated uncertainty" },
      { key: "nonlinearity", label: "Handles non-linearity" },
      { key: "smallData", label: "Works on small data" },
    ];

    const families = [
      { name: "Decision trees", scores: { interpretability: 5, scale: 3, uncertainty: 2, nonlinearity: 4, smallData: 4 }, stack: "scikit-learn", page: "decision-trees" },
      { name: "Linear / logistic", scores: { interpretability: 5, scale: 5, uncertainty: 3, nonlinearity: 1, smallData: 4 }, stack: "scikit-learn", page: "logistic-regression" },
      { name: "Kernel SVM", scores: { interpretability: 2, scale: 1, uncertainty: 2, nonlinearity: 5, smallData: 4 }, stack: "scikit-learn", page: "support-vector-machines" },
      { name: "Neural networks", scores: { interpretability: 1, scale: 5, uncertainty: 2, nonlinearity: 5, smallData: 1 }, stack: "PyTorch", page: "backpropagation" },
      { name: "Bayesian models", scores: { interpretability: 4, scale: 2, uncertainty: 5, nonlinearity: 3, smallData: 5 }, stack: "PyMC", page: "bayesian-estimation" },
      { name: "k-NN", scores: { interpretability: 4, scale: 1, uncertainty: 2, nonlinearity: 5, smallData: 3 }, stack: "scikit-learn", page: "nearest-neighbour" },
      { name: "HMM / sequence", scores: { interpretability: 3, scale: 3, uncertainty: 4, nonlinearity: 2, smallData: 4 }, stack: "hmmlearn", page: "hidden-markov-models" },
    ];

    mountSection(
      rootNode,
      "Choosing a model family, as arithmetic",
      "“Which algorithm should I use?” is not a lookup — it is a weighted trade-off. Set how much each requirement matters for your problem and the ranking below recomputes. The point is not that the scores are gospel; it is that making the weights explicit turns an argument into a calculation you can inspect and disagree with.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-group">
              <label for="tools-domain">Preset from a domain</label>
              <select id="tools-domain">
                <option value="custom">Custom weights</option>
                <option value="vision">Computer vision</option>
                <option value="healthcare">Clinical decision support</option>
                <option value="finance">Finance / risk</option>
                <option value="nlp">Natural language</option>
                <option value="science">Scientific inference</option>
              </select>
            </div>
            ${criteria
              .map(
                (criterion) => `
                  <div class="control-group">
                    <label for="w-${criterion.key}">${criterion.label}</label>
                    <div class="range-row">
                      <input id="w-${criterion.key}" class="tool-weight" data-key="${criterion.key}" type="range" min="0" max="5" step="1" value="3" />
                      <span class="range-value" id="wv-${criterion.key}">3</span>
                    </div>
                  </div>
                `
              )
              .join("")}
            <div class="callout" id="tools-callout"></div>
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg id="tools-plot" viewBox="0 0 560 300" aria-label="Weighted score by model family"></svg>
            </div>
            <div class="readout"><div class="output-grid" id="tools-output"></div></div>
            <div class="equation-card" id="tools-math"></div>
            <div class="table-panel">
              <table>
                <thead>
                  <tr>
                    <th>Family</th>
                    ${criteria.map((criterion) => `<th>${criterion.label.split(" ")[0]}</th>`).join("")}
                    <th>Score</th>
                    <th>Stack</th>
                  </tr>
                </thead>
                <tbody id="tools-table"></tbody>
              </table>
            </div>
            <div class="steps"><ol id="tools-steps"></ol></div>
          </div>
        </div>
      `
    );

    const domainInput = rootNode.querySelector("#tools-domain");
    const weightInputs = U.qsa(".tool-weight", rootNode);
    const callout = rootNode.querySelector("#tools-callout");
    const plot = rootNode.querySelector("#tools-plot");
    const output = rootNode.querySelector("#tools-output");
    const mathNode = rootNode.querySelector("#tools-math");
    const tableNode = rootNode.querySelector("#tools-table");
    const stepsNode = rootNode.querySelector("#tools-steps");

    const presets = {
      vision: { interpretability: 1, scale: 5, uncertainty: 1, nonlinearity: 5, smallData: 1 },
      healthcare: { interpretability: 5, scale: 2, uncertainty: 5, nonlinearity: 2, smallData: 4 },
      finance: { interpretability: 4, scale: 4, uncertainty: 5, nonlinearity: 3, smallData: 2 },
      nlp: { interpretability: 1, scale: 5, uncertainty: 2, nonlinearity: 5, smallData: 1 },
      science: { interpretability: 4, scale: 1, uncertainty: 5, nonlinearity: 3, smallData: 5 },
    };

    function weights() {
      const result = {};
      weightInputs.forEach((input) => {
        result[input.dataset.key] = Number(input.value);
      });
      return result;
    }

    function render() {
      const w = weights();
      criteria.forEach((criterion) => {
        rootNode.querySelector(`#wv-${criterion.key}`).textContent = String(w[criterion.key]);
      });

      const totalWeight = criteria.reduce((acc, criterion) => acc + w[criterion.key], 0);
      const scored = families
        .map((family) => {
          const raw = criteria.reduce((acc, criterion) => acc + w[criterion.key] * family.scores[criterion.key], 0);
          return { ...family, raw, normalized: totalWeight > 0 ? raw / (totalWeight * 5) : 0 };
        })
        .sort((a, b) => b.raw - a.raw);

      const best = scored[0];
      const runnerUp = scored[1];

      callout.innerHTML = totalWeight === 0
        ? "Every weight is zero — raise at least one requirement to get a ranking."
        : `Top match: <strong>${best.name}</strong> (${U.round(best.normalized * 100, 0)}% fit), ahead of ${runnerUp.name} by ${U.round((best.raw - runnerUp.raw), 0)} points. Usual stack: <strong>${best.stack}</strong>.`;

      U.renderMetrics(output, [
        { label: "Best fit", value: best.name },
        { label: "Fit score", value: `${U.round(best.normalized * 100, 0)}%` },
        { label: "Runner-up", value: runnerUp.name },
        { label: "Typical stack", value: best.stack },
      ]);

      const dominant = criteria.reduce((a, b) => (w[a.key] >= w[b.key] ? a : b));

      renderFormulaCards(mathNode, [
        {
          title: "Weighted decision score",
          description:
            "Each family gets a score on every criterion; you supply how much each criterion matters. The total is a simple weighted sum, normalised so 100% would be a family that scores maximum on everything you care about.",
          tex: "S(m) = \\frac{\\sum_{c} w_c \\, s_{mc}}{5\\sum_{c} w_c}",
          derivation: [
            {
              tex: `S(\\text{${best.name}}) = \\frac{${criteria.map((criterion) => `${w[criterion.key]} \\times ${best.scores[criterion.key]}`).join(" + ")}}{5 \\times ${totalWeight}}`,
              result: `${U.round(best.normalized * 100, 0)}%`,
            },
            {
              tex: `\\text{dominant criterion: } \\text{${dominant.label}} \\;(w = ${w[dominant.key]})`,
              result: "",
              note: "Set one weight to 5 and the rest to 0 to see which family wins on that criterion alone.",
            },
          ],
          insight:
            "<strong>The weights are the actual decision.</strong> Any two engineers who agree on the scores can still disagree on the model — because they disagree about what the project needs. Making that explicit is more useful than any benchmark table.",
        },
        {
          title: "There is no free lunch",
          description:
            "No family wins on every row of the table. Interpretability trades against flexibility; calibrated uncertainty trades against scale. That is not a gap in the tooling — it is a theorem.",
          tex: "\\sum_{f} \\mathbb{E}\\big[\\text{error} \\mid a, f\\big] \\;\\text{ is the same for every algorithm } a",
          derivation: [
            {
              tex: `\\text{families evaluated} = ${families.length}, \\quad \\text{criteria} = ${criteria.length}`,
              result: "",
              note: "Averaged over all possible problems, every algorithm performs identically. Performance comes from matching the algorithm's assumptions to your particular problem.",
            },
          ],
          insight:
            "<strong>This is why the interactive pages matter more than the ranking.</strong> Knowing that k-NN struggles at scale is a fact; watching its neighbourhood search degrade as you add points is understanding.",
        },
      ]);

      tableNode.innerHTML = scored
        .map(
          (family, index) => `
            <tr style="${index === 0 ? "background:var(--accent-soft);font-weight:600" : ""}">
              <td><a href="${algorithmHref(family.page)}">${family.name}</a></td>
              ${criteria.map((criterion) => `<td>${family.scores[criterion.key]}</td>`).join("")}
              <td>${U.round(family.normalized * 100, 0)}%</td>
              <td>${family.stack}</td>
            </tr>
          `
        )
        .join("");

      U.renderSteps(
        stepsNode,
        [
          `<strong>State what the problem needs.</strong> Your highest weight is <strong>${dominant.label}</strong>.`,
          `<strong>Score each family.</strong> Every family gets 1–5 on each criterion (the table on the right).`,
          `<strong>Combine.</strong> Weighted sum, normalised — <strong>${best.name}</strong> comes out on top at ${U.round(best.normalized * 100, 0)}%.`,
          `<strong>Pick the stack.</strong> ${best.stack} is the usual implementation route for that family.`,
          `<strong>Then verify empirically.</strong> This ranking is a starting hypothesis, not a result — validate it on your own data.`,
        ],
        5
      );

      /* Horizontal ranked bars. */
      U.clear(plot);
      plot.appendChild(U.svgEl("text", { x: 16, y: 24, class: "svg-title" })).textContent =
        "Weighted fit by model family";
      const barLeft = 150;
      const barMax = 380;
      scored.forEach((family, index) => {
        const y = 46 + index * 34;
        const width = Math.max(family.normalized * barMax, 2);
        plot.appendChild(
          U.svgEl("text", { x: barLeft - 10, y: y + 13, class: "svg-label", "text-anchor": "end" })
        ).textContent = family.name;
        plot.appendChild(
          U.svgEl("rect", { x: barLeft, y, width: barMax, height: 20, rx: 5, fill: C().plotBg })
        );
        plot.appendChild(
          U.svgEl("rect", {
            x: barLeft,
            y,
            width,
            height: 20,
            rx: 5,
            fill: index === 0 ? C().a : C().neutral,
            opacity: index === 0 ? 1 : 0.6,
          })
        );
        plot.appendChild(
          U.svgEl("text", {
            x: barLeft + width + 8,
            y: y + 14,
            class: "svg-label",
            fill: index === 0 ? C().a : C().neutral,
          })
        ).textContent = `${U.round(family.normalized * 100, 0)}%`;
      });
    }

    domainInput.addEventListener("change", () => {
      const preset = presets[domainInput.value];
      if (!preset) return;
      weightInputs.forEach((input) => {
        input.value = preset[input.dataset.key];
      });
      render();
    });
    weightInputs.forEach((input) =>
      input.addEventListener("input", () => {
        domainInput.value = "custom";
        render();
      })
    );
    U.onRedraw(render);
    render();
  }

  function mountGaussian(rootNode) {
    const focus = definition.options.focus;
    mountSection(
      rootNode,
      "Gaussian evidence and posterior explorer",
      "Edit the sample and prior, then reveal the derivation. The formula cards update with the current numbers so the algebra stays tied to the data.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-group">
              <label for="gaussian-samples">Observed samples (comma or space separated)</label>
              <textarea id="gaussian-samples">2.2, 1.7, 2.6, 1.9, 2.4, 2.1, 2.8</textarea>
              <p class="caption">Add a wild value like <code>9</code> and watch the MLE lurch while the posterior, anchored by the prior, resists.</p>
            </div>
            <div class="control-grid">
              <div class="control-group">
                <label for="prior-mean">Prior mean μ₀</label>
                <input id="prior-mean" type="number" step="0.1" value="1.5" />
              </div>
              <div class="control-group">
                <label for="prior-variance">Prior variance σ²₀</label>
                <input id="prior-variance" type="number" step="0.1" min="0.1" value="0.9" />
              </div>
              <div class="control-group">
                <label for="obs-variance">Observation variance σ²</label>
                <input id="obs-variance" type="number" step="0.1" min="0.1" value="0.6" />
              </div>
              <div class="control-group">
                <label for="query-x">Query x*</label>
                <input id="query-x" type="number" step="0.1" value="2.3" />
              </div>
            </div>
            <div class="step-controls">
              <button class="button primary" id="gaussian-step">Next step</button>
              <button class="button secondary" id="gaussian-reset">Reset steps</button>
            </div>
            <div class="callout" id="gaussian-callout"></div>
            <div class="legend">
              <span><i style="background:#6f8a88"></i> Prior</span>
              <span><i style="background:#1a7a6e"></i> MLE fit</span>
              <span><i style="background:#c96b28"></i> Posterior predictive</span>
            </div>
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg id="gaussian-plot" viewBox="0 0 560 280" aria-label="Gaussian estimation plot"></svg>
            </div>
            <div class="readout"><div class="output-grid" id="gaussian-output"></div></div>
            <div class="equation-card" id="gaussian-math"></div>
            <div class="steps"><ol id="gaussian-steps"></ol></div>
          </div>
        </div>
      `
    );

    const samplesInput = rootNode.querySelector("#gaussian-samples");
    const priorMeanInput = rootNode.querySelector("#prior-mean");
    const priorVarianceInput = rootNode.querySelector("#prior-variance");
    const obsVarianceInput = rootNode.querySelector("#obs-variance");
    const queryInput = rootNode.querySelector("#query-x");
    const stepButton = rootNode.querySelector("#gaussian-step");
    const resetButton = rootNode.querySelector("#gaussian-reset");
    const plot = rootNode.querySelector("#gaussian-plot");
    const output = rootNode.querySelector("#gaussian-output");
    const mathNode = rootNode.querySelector("#gaussian-math");
    const stepsNode = rootNode.querySelector("#gaussian-steps");
    const callout = rootNode.querySelector("#gaussian-callout");
    let revealed = 0;

    function render() {
      const samples = U.parseNumberList(samplesInput.value);
      const priorMean = Number(priorMeanInput.value) || 0;
      const priorVariance = Math.max(Number(priorVarianceInput.value) || 1, 0.05);
      const obsVariance = Math.max(Number(obsVarianceInput.value) || 1, 0.05);
      const query = Number(queryInput.value) || 0;

      if (!samples.length) {
        U.renderMetrics(output, [{ label: "Status", value: "Add samples" }]);
        U.renderSteps(stepsNode, [], 0);
        renderFormulaCards(mathNode, []);
        U.clear(plot);
        return;
      }

      const sampleMean = U.mean(samples);
      const mleVariance = U.variance(samples);
      const { posteriorMean, posteriorVariance } = U.normalPosterior(
        samples,
        obsVariance,
        priorMean,
        priorVariance
      );
      const predictiveDensity = U.gaussianPdf(query, posteriorMean, posteriorVariance + obsVariance);
      const logLikelihood =
        -0.5 * samples.length * Math.log(2 * Math.PI * obsVariance) -
        samples.reduce((acc, value) => acc + ((value - sampleMean) ** 2) / (2 * obsVariance), 0);

      callout.innerHTML = {
        gaussian: "This page emphasizes the Gaussian summary itself: mean and variance compress the sample into a tractable model.",
        mle: "This page emphasizes the likelihood term. The MLE is the parameter setting that maximizes it.",
        map: "This page emphasizes how the prior shifts the estimate away from the pure MLE.",
        bayesian: "This page emphasizes the full posterior, not only a single point estimate.",
        generative: "This page emphasizes the data-generation story: likelihood, prior, and posterior all belong to the same generative account.",
      }[focus];

      U.renderMetrics(output, [
        { label: "Sample mean", value: U.round(sampleMean, 3) },
        { label: "MLE variance", value: U.round(mleVariance, 3) },
        { label: "Posterior mean", value: U.round(posteriorMean, 3) },
        { label: "p(x=query)", value: U.round(predictiveDensity, 4) },
      ]);

      const dataPrecision = U.round(samples.length / obsVariance, 4);
      const priorPrecision = U.round(1 / priorVariance, 4);
      const totalPrecision = U.round(dataPrecision + priorPrecision, 4);
      const sampleSum = U.round(samples.reduce((a, b) => a + b, 0), 3);

      renderFormulaCards(mathNode, [
        {
          title: "Sample mean",
          description: "The arithmetic average compresses all n observations to one number — the sufficient statistic for a Gaussian.",
          tex: "\\bar{x} = \\frac{1}{n}\\sum_{i=1}^{n} x_i",
          derivation: [
            { expr: `Sum: ${samples.map(v => U.round(v,2)).join(' + ')} = ${sampleSum}`, result: sampleSum },
            { expr: `x̄ = ${sampleSum} / ${samples.length}`, result: U.round(sampleMean, 4) },
            { expr: `Sample variance s² = Σ(xᵢ − x̄)² / n`, result: U.round(mleVariance, 4), note: `Each squared deviation: ${samples.map(v => `(${U.round(v,2)}−${U.round(sampleMean,2)})²=${U.round((v-sampleMean)**2,3)}`).slice(0,4).join(', ')}${samples.length>4?', …':''}` },
          ],
          insight: `<strong>Why x̄?</strong> Taking the derivative of the Gaussian log-likelihood ln p(D|μ) = −n/2·ln(2πσ²) − Σ(xᵢ−μ)²/(2σ²) with respect to μ and setting it to zero gives μ̂ = x̄ exactly.`,
        },
        {
          title: focus === "mle" ? "Log-likelihood" : "Gaussian likelihood",
          description: focus === "mle"
            ? "MLE asks: for which μ is the observed data most probable? Setting ∂ℓ/∂μ = 0 yields μ̂ = x̄."
            : "Each observation contributes one Gaussian density term; they multiply because samples are independent.",
          tex: focus === "mle"
            ? "\\ell(\\mu) = -\\frac{n}{2}\\ln(2\\pi\\sigma^2) - \\sum_{i=1}^{n}\\frac{(x_i - \\mu)^2}{2\\sigma^2}"
            : "p(D \\mid \\mu, \\sigma^2) = \\prod_{i=1}^{n} \\mathcal{N}(x_i \\mid \\mu, \\sigma^2)",
          derivation: focus === "mle" ? [
            { expr: `n = ${samples.length},  σ²_obs = ${U.round(obsVariance,3)}`, result: '' },
            { expr: `−(n/2)·ln(2πσ²) = −(${samples.length}/2)·ln(${U.round(2*Math.PI*obsVariance,3)})`, result: U.round(-0.5*samples.length*Math.log(2*Math.PI*obsVariance),3) },
            { expr: `−Σ(xᵢ−x̄)²/(2σ²) = −${U.round(samples.reduce((a,v)=>a+(v-sampleMean)**2,0),3)}/(2·${U.round(obsVariance,3)})`, result: U.round(-samples.reduce((a,v)=>a+(v-sampleMean)**2,0)/(2*obsVariance),3) },
            { expr: `ℓ(μ̂ = x̄) total`, result: U.round(logLikelihood, 4), note: "This is the maximum value of ℓ; any other μ gives a smaller log-likelihood." },
          ] : [
            { expr: `First term: N(${U.round(samples[0],2)} | μ, ${U.round(obsVariance,2)})`, result: U.round(U.gaussianPdf(samples[0], sampleMean, obsVariance), 4) },
            { expr: `Full product: Π over ${samples.length} terms`, result: `exp(${U.round(logLikelihood,3)})` },
          ],
          insight: focus === "mle"
            ? `<strong>Key insight:</strong> The MLE for the Gaussian mean is unbiased (<em>E[x̄] = μ</em>), but the MLE for variance divides by <em>n</em>, introducing a small downward bias — which is why sample variance divides by <em>n−1</em>.`
            : `<strong>Independence assumption:</strong> p(D|μ) = Π p(xᵢ|μ) only holds when samples are i.i.d. The log converts the product to a manageable sum.`,
        },
        {
          title: focus === "map" ? "MAP posterior mode" : "Posterior precision update",
          description: "Bayes' rule combines prior precision and data precision additively — a beautiful conjugacy property of the Gaussian family.",
          tex: "\\tau_{\\text{post}} = \\tau_{0} + n\\tau, \\qquad \\mu_{\\text{post}} = \\frac{n\\tau\\,\\bar{x} + \\tau_0\\mu_0}{\\tau_{\\text{post}}}, \\qquad \\tau = \\tfrac{1}{\\sigma^2}",
          derivation: [
            { expr: `Prior precision:  τ₀ = 1/σ²_prior = 1/${U.round(priorVariance,3)}`, result: priorPrecision },
            { expr: `Data precision:   τ_n = n/σ²_obs  = ${samples.length}/${U.round(obsVariance,3)}`, result: dataPrecision },
            { expr: `Total precision:  τ_post = ${priorPrecision} + ${dataPrecision}`, result: totalPrecision },
            { expr: `σ²_post = 1/τ_post = 1/${totalPrecision}`, result: U.round(1/totalPrecision, 4) },
            { expr: `μ_post = (τ_n·x̄ + τ₀·μ₀) / τ_post = (${dataPrecision}·${U.round(sampleMean,3)} + ${priorPrecision}·${U.round(priorMean,3)}) / ${totalPrecision}`, result: U.round(posteriorMean, 4), note: "This is a precision-weighted average of the sample mean and prior mean." },
          ],
          insight: `<strong>Conjugate prior magic:</strong> When the prior p(μ) = N(μ₀, σ²_prior) and the likelihood is Gaussian, the posterior is also Gaussian. The posterior precision is just the sum of precisions — as clean as addition.`,
        },
        {
          title: focus === "bayesian" ? "Posterior uncertainty" : "Predictive density",
          description: focus === "bayesian"
            ? "The posterior variance tells you how uncertain you remain about μ after seeing n data points."
            : "Predictive inference integrates over posterior uncertainty, adding σ²_obs and σ²_post sources of variation.",
          tex: focus === "bayesian"
            ? "\\sigma^2_{\\text{post}} = \\frac{1}{\\tau_{\\text{post}}} = \\frac{1}{\\tau_0 + n\\tau}"
            : "p(x^* \\mid D) = \\mathcal{N}\\!\\left(x^* \\mid \\mu_{\\text{post}},\\; \\sigma^2_{\\text{post}} + \\sigma^2\\right)",
          derivation: focus === "bayesian" ? [
            { expr: `σ²_post = 1 / ${totalPrecision}`, result: U.round(posteriorVariance, 4) },
            { expr: `σ_post = √${U.round(posteriorVariance,4)}`, result: U.round(Math.sqrt(posteriorVariance), 4), note: "As n → ∞, σ²_post → 0 and the posterior collapses onto the true μ." },
          ] : [
            { expr: `Predictive variance = σ²_post + σ²_obs = ${U.round(posteriorVariance,4)} + ${U.round(obsVariance,3)}`, result: U.round(posteriorVariance+obsVariance, 4) },
            { expr: `p(x*=${U.round(query,2)} | D) = N(${U.round(query,2)} | ${U.round(posteriorMean,3)}, ${U.round(posteriorVariance+obsVariance,3)})`, result: U.round(predictiveDensity, 5) },
          ],
          insight: focus === "bayesian"
            ? `<strong>Shrinkage:</strong> With n=${samples.length} observations, uncertainty dropped from σ²_prior=${U.round(priorVariance,3)} to σ²_post=${U.round(posteriorVariance,4)} — a ${U.round((1-posteriorVariance/priorVariance)*100,1)}% reduction.`
            : `<strong>Two sources of randomness:</strong> σ²_post captures parameter uncertainty (goes to 0 as n→∞), while σ²_obs captures irreducible measurement noise (stays constant).`,
        },
      ]);

      const steps = [
        `<strong>Summarize the sample.</strong> There are <strong>${samples.length}</strong> observations and their mean is <strong>${U.round(sampleMean, 3)}</strong>.`,
        `<strong>Compute the Gaussian fit.</strong> The sample variance estimate is <strong>${U.round(mleVariance, 3)}</strong>.`,
        `<strong>Write the likelihood.</strong> The Gaussian model contributes a product of densities over the sample.`,
        `<strong>Combine with the prior.</strong> Prior precision <span class="mono">${U.round(1 / priorVariance, 3)}</span> and data precision <span class="mono">${U.round(samples.length / obsVariance, 3)}</span> add.`,
        `<strong>Read the posterior.</strong> The posterior mean is <strong>${U.round(posteriorMean, 3)}</strong> with variance <strong>${U.round(posteriorVariance, 3)}</strong>.`,
      ];
      U.renderSteps(stepsNode, steps, revealed);

      const xMin = Math.min(...samples, priorMean, query) - 1;
      const xMax = Math.max(...samples, priorMean, query) + 1;
      const xs = U.linspace(xMin, xMax, 180);
      const priorCurve = xs.map((x) => ({ x, y: U.gaussianPdf(x, priorMean, priorVariance) }));
      const mleCurve = xs.map((x) => ({ x, y: U.gaussianPdf(x, sampleMean, Math.max(mleVariance, 0.08)) }));
      const posteriorCurve = xs.map((x) => ({ x, y: U.gaussianPdf(x, posteriorMean, obsVariance + posteriorVariance) }));
      const yMax = Math.max(...priorCurve.map((p) => p.y), ...mleCurve.map((p) => p.y), ...posteriorCurve.map((p) => p.y)) * 1.25;
      const chart = U.makeChart(plot, {
        xDomain: [xMin, xMax],
        yDomain: [0, yMax],
        title: "Prior, MLE fit, and posterior predictive",
      });

      plot.appendChild(U.svgEl("path", { d: U.pathFromPoints(priorCurve, chart.xScale, chart.yScale), fill: "none", stroke: C().neutral, "stroke-width": "2.4", "stroke-dasharray": "4 6" }));
      plot.appendChild(U.svgEl("path", { d: U.pathFromPoints(mleCurve, chart.xScale, chart.yScale), class: "curve-primary" }));
      plot.appendChild(U.svgEl("path", { d: U.pathFromPoints(posteriorCurve, chart.xScale, chart.yScale), class: "curve-secondary" }));
      samples.forEach((sample) => {
        const x = chart.xScale(sample);
        plot.appendChild(U.svgEl("line", { x1: x, y1: chart.yScale(0), x2: x, y2: chart.yScale(yMax * 0.08), stroke: C().ink, "stroke-width": "1.8" }));
      });
      plot.appendChild(U.svgEl("line", { x1: chart.xScale(query), y1: chart.yScale(0), x2: chart.xScale(query), y2: chart.yScale(yMax * 0.9), stroke: C().b, "stroke-width": "2", "stroke-dasharray": "6 6" }));
    }

    stepButton.addEventListener("click", () => {
      revealed = Math.min(revealed + 1, 5);
      render();
    });
    resetButton.addEventListener("click", () => {
      revealed = 0;
      render();
    });
    [samplesInput, priorMeanInput, priorVarianceInput, obsVarianceInput, queryInput].forEach((input) =>
      input.addEventListener("input", () => {
        revealed = 0;
        render();
      })
    );
    U.onRedraw(render);
    render();
  }

  function mountBiasVariance(rootNode) {
    mountSection(
      rootNode,
      "Bias-variance simulation",
      "Change the true distribution, sample size, and shrinkage factor. The page simulates repeated datasets and computes the bias-variance decomposition numerically.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-grid">
              <div class="control-group">
                <label for="true-mean">True mean</label>
                <input id="true-mean" type="number" step="0.5" value="3" />
              </div>
              <div class="control-group">
                <label for="true-std">True standard deviation</label>
                <input id="true-std" type="number" step="0.1" min="0.2" value="1.2" />
              </div>
              <div class="control-group">
                <label for="sample-size">Sample size</label>
                <input id="sample-size" type="number" min="2" max="60" value="8" />
              </div>
              <div class="control-group">
                <label for="simulations">Simulations</label>
                <input id="simulations" type="number" min="50" max="1000" value="250" />
              </div>
            </div>
            <div class="control-group">
              <label for="shrinkage">Estimator shrinkage factor</label>
              <div class="range-row">
                <input id="shrinkage" type="range" min="0.2" max="1.4" step="0.05" value="0.9" />
                <span class="range-value" id="shrinkage-value">0.90</span>
              </div>
            </div>
            <div class="step-controls">
              <button class="button primary" id="bias-step">Next step</button>
              <button class="button secondary" id="bias-reset">Reset steps</button>
            </div>
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg id="bias-plot" viewBox="0 0 560 280" aria-label="Bias variance plot"></svg>
            </div>
            <div class="readout"><div class="output-grid" id="bias-output"></div></div>
            <div class="equation-card" id="bias-math"></div>
            <div class="steps"><ol id="bias-steps"></ol></div>
          </div>
        </div>
      `
    );

    const trueMeanInput = rootNode.querySelector("#true-mean");
    const trueStdInput = rootNode.querySelector("#true-std");
    const sampleSizeInput = rootNode.querySelector("#sample-size");
    const simulationsInput = rootNode.querySelector("#simulations");
    const shrinkageInput = rootNode.querySelector("#shrinkage");
    const shrinkageValue = rootNode.querySelector("#shrinkage-value");
    const stepButton = rootNode.querySelector("#bias-step");
    const resetButton = rootNode.querySelector("#bias-reset");
    const plot = rootNode.querySelector("#bias-plot");
    const output = rootNode.querySelector("#bias-output");
    const mathNode = rootNode.querySelector("#bias-math");
    const stepsNode = rootNode.querySelector("#bias-steps");
    let revealed = 0;

    function render() {
      const trueMean = Number(trueMeanInput.value) || 0;
      const trueStd = Math.max(Number(trueStdInput.value) || 1, 0.05);
      const sampleSize = Math.max(Number(sampleSizeInput.value) || 2, 2);
      const simulations = Math.max(Number(simulationsInput.value) || 50, 20);
      const shrinkage = Number(shrinkageInput.value) || 1;
      shrinkageValue.textContent = shrinkage.toFixed(2);

      const rng = U.seededRandom(27);
      const estimates = [];
      for (let sim = 0; sim < simulations; sim += 1) {
        const sample = [];
        for (let index = 0; index < sampleSize; index += 1) {
          sample.push(U.sampleNormal(rng, trueMean, trueStd));
        }
        estimates.push(shrinkage * U.mean(sample));
      }

      const estimateMean = U.mean(estimates);
      const bias = estimateMean - trueMean;
      const variance = U.variance(estimates);
      const mse = U.mean(estimates.map((value) => (value - trueMean) ** 2));

      U.renderMetrics(output, [
        { label: "Average estimate", value: U.round(estimateMean, 3) },
        { label: "Bias", value: U.round(bias, 3) },
        { label: "Variance", value: U.round(variance, 3) },
        { label: "MSE", value: U.round(mse, 3) },
      ]);

      renderFormulaCards(mathNode, [
        { title: "Estimator", description: "The simulation uses a shrunk sample mean: c = 1 recovers the plain average, c < 1 pulls every estimate toward zero.", tex: `\\hat{\\mu} = c\\,\\bar{x} = ${U.round(shrinkage, 2)}\\,\\bar{x}` },
        { title: "Bias", description: "Average displacement from the true parameter across every simulated dataset.", tex: "\\text{Bias}(\\hat{\\mu}) = \\mathbb{E}[\\hat{\\mu}] - \\mu", derivation: [{ tex: `\\mathbb{E}[\\hat{\\mu}] - \\mu = ${U.texNum(estimateMean, 3)} - ${U.texNum(trueMean, 3)}`, result: U.round(bias, 3) }] },
        { title: "Variance", description: "How much the estimate jumps around depending on which dataset happened to arrive.", tex: "\\text{Var}(\\hat{\\mu}) = \\mathbb{E}\\big[(\\hat{\\mu} - \\mathbb{E}[\\hat{\\mu}])^2\\big]", derivation: [{ tex: `\\text{Var}(\\hat{\\mu}) = ${U.texNum(variance, 4)}`, result: U.round(variance, 4), note: "Shrink harder and this falls — at the cost of bias." }] },
        { title: "Mean squared error", description: "The decomposition that makes the tradeoff concrete — the two terms add, so lowering one at the other's expense only helps if the drop is bigger than the rise.", tex: "\\text{MSE} = \\text{Bias}^2 + \\text{Var}", derivation: [{ tex: `\\text{MSE} = (${U.texNum(bias, 3)})^2 + ${U.texNum(variance, 4)} = ${U.texNum(bias * bias, 4)} + ${U.texNum(variance, 4)}`, result: U.round(mse, 4), note: "Sweep the shrinkage slider and watch MSE dip below the unbiased estimator's value — that dip is the whole reason regularisation works." }] },
      ]);

      const steps = [
        `<strong>Generate ${simulations} datasets.</strong> Each dataset contains ${sampleSize} draws from the true Gaussian.`,
        `<strong>Apply the estimator.</strong> The estimator multiplies the sample mean by <span class="mono">${U.round(shrinkage, 2)}</span>.`,
        `<strong>Compute bias.</strong> The empirical mean estimate is <strong>${U.round(estimateMean, 3)}</strong>, so the bias is <strong>${U.round(bias, 3)}</strong>.`,
        `<strong>Compute variance.</strong> Across runs the estimator varies by <strong>${U.round(variance, 3)}</strong>.`,
        `<strong>Compute MSE.</strong> Squared bias and variance sum to <strong>${U.round(mse, 3)}</strong>.`,
      ];
      U.renderSteps(stepsNode, steps, revealed);

      const bars = [
        { label: "bias²", value: bias * bias, color: C().b },
        { label: "variance", value: variance, color: C().a },
        { label: "mse", value: mse, color: C().ink },
      ];
      const yMax = Math.max(...bars.map((item) => item.value), 0.02) * 1.25;
      const chart = U.makeChart(plot, {
        xDomain: [0, 4],
        yDomain: [0, yMax],
        title: "Bias-variance decomposition",
      });
      bars.forEach((bar, index) => {
        const x1 = chart.xScale(index + 0.45);
        const x2 = chart.xScale(index + 1.05);
        const y = chart.yScale(bar.value);
        plot.appendChild(U.svgEl("rect", { x: x1, y, width: x2 - x1, height: chart.yScale(0) - y, fill: bar.color, rx: "12", opacity: "0.9" }));
        plot.appendChild(U.svgEl("text", { x: (x1 + x2) / 2, y: chart.yScale(0) + 18, class: "svg-label", "text-anchor": "middle" })).textContent = bar.label;
      });
    }

    stepButton.addEventListener("click", () => {
      revealed = Math.min(revealed + 1, 5);
      render();
    });
    resetButton.addEventListener("click", () => {
      revealed = 0;
      render();
    });
    [trueMeanInput, trueStdInput, sampleSizeInput, simulationsInput, shrinkageInput].forEach((input) =>
      input.addEventListener("input", () => {
        revealed = 0;
        render();
      })
    );
    U.onRedraw(render);
    render();
  }

  function mountCleanup(rootNode) {
    mountSection(
      rootNode,
      "Missing-value and noise cleanup lab",
      "Choose an imputation rule and a noise clamp. The formulas show exactly how missing values and extreme z-scores are transformed before modeling begins.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-group">
              <label for="cleanup-method">Missing-value strategy</label>
              <select id="cleanup-method">
                <option value="mean">Mean imputation</option>
                <option value="median">Median imputation</option>
                <option value="drop">Drop incomplete rows</option>
              </select>
            </div>
            <div class="control-group">
              <label for="noise-clamp">Clamp unusually noisy values at ± this z-score</label>
              <div class="range-row">
                <input id="noise-clamp" type="range" min="0.8" max="3" step="0.1" value="1.8" />
                <span class="range-value" id="noise-clamp-value">1.8</span>
              </div>
            </div>
            <div class="step-controls">
              <button class="button primary" id="cleanup-apply">Apply cleanup</button>
            </div>
            <div class="callout" id="cleanup-summary"></div>
            <div class="editor-slot" id="cleanup-data"></div>
          </div>
          <div class="two-column">
            <div class="table-panel">
              <table>
                <thead>
                  <tr>
                    <th>Example</th>
                    <th>Feature A</th>
                    <th>Feature B</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody id="cleanup-table"></tbody>
              </table>
            </div>
            <div class="plot-card">
              <svg id="cleanup-plot" viewBox="0 0 560 280" aria-label="Cleaned features plot"></svg>
            </div>
            <div class="equation-card" id="cleanup-math"></div>
          </div>
        </div>
      `
    );

    /* `null` marks a missing value. The editor writes an empty string,
       which we translate back into null so the imputation path fires. */
    const cleanupDefaults = [
      { id: "E1", a: 1.4, b: 2.0 },
      { id: "E2", a: 1.8, b: "" },
      { id: "E3", a: "", b: 1.7 },
      { id: "E4", a: 2.6, b: 2.9 },
      { id: "E5", a: 7.8, b: 0.5 },
      { id: "E6", a: 2.1, b: 3.2 },
      { id: "E7", a: 2.5, b: 2.7 },
      { id: "E8", a: 1.7, b: 8.4 },
    ];
    let rawRows = cleanupDefaults.map((row) => ({ ...row }));
    let baseData = [];

    function syncBaseData() {
      baseData = rawRows.map((row) => ({
        id: row.id,
        a: row.a === "" || row.a === null ? null : Number(row.a),
        b: row.b === "" || row.b === null ? null : Number(row.b),
      }));
    }
    syncBaseData();

    const methodInput = rootNode.querySelector("#cleanup-method");
    const clampInput = rootNode.querySelector("#noise-clamp");
    const clampValue = rootNode.querySelector("#noise-clamp-value");
    const button = rootNode.querySelector("#cleanup-apply");
    const summary = rootNode.querySelector("#cleanup-summary");
    const tableBody = rootNode.querySelector("#cleanup-table");
    const plot = rootNode.querySelector("#cleanup-plot");
    const mathNode = rootNode.querySelector("#cleanup-math");

    function render() {
      const method = methodInput.value;
      const clampZ = Number(clampInput.value) || 1.8;
      clampValue.textContent = clampZ.toFixed(1);
      const aValues = baseData.map((row) => row.a).filter(Number.isFinite);
      const bValues = baseData.map((row) => row.b).filter(Number.isFinite);
      const fillA = method === "median" ? U.median(aValues) : U.mean(aValues);
      const fillB = method === "median" ? U.median(bValues) : U.mean(bValues);

      const cleaned = [];
      let imputedCount = 0;
      let clampedCount = 0;
      baseData.forEach((row) => {
        if (method === "drop" && (!Number.isFinite(row.a) || !Number.isFinite(row.b))) {
          cleaned.push({ id: row.id, a: row.a, b: row.b, status: "Dropped for missing value", dropped: true });
          return;
        }
        const next = {
          id: row.id,
          a: Number.isFinite(row.a) ? row.a : fillA,
          b: Number.isFinite(row.b) ? row.b : fillB,
          dropped: false,
          changed: !Number.isFinite(row.a) || !Number.isFinite(row.b),
          statusParts: [],
        };
        if (!Number.isFinite(row.a)) {
          next.statusParts.push(`imputed A=${U.round(fillA, 2)}`);
          imputedCount += 1;
        }
        if (!Number.isFinite(row.b)) {
          next.statusParts.push(`imputed B=${U.round(fillB, 2)}`);
          imputedCount += 1;
        }
        cleaned.push(next);
      });
      const retained = cleaned.filter((row) => !row.dropped);
      const means = { a: U.mean(retained.map((row) => row.a)), b: U.mean(retained.map((row) => row.b)) };
      const stds = { a: Math.sqrt(U.variance(retained.map((row) => row.a))), b: Math.sqrt(U.variance(retained.map((row) => row.b))) };

      retained.forEach((row) => {
        ["a", "b"].forEach((feature) => {
          if (stds[feature] <= 1e-6) {
            return;
          }
          const z = (row[feature] - means[feature]) / stds[feature];
          if (Math.abs(z) > clampZ) {
            row[feature] = means[feature] + Math.sign(z) * clampZ * stds[feature];
            row.changed = true;
            row.statusParts.push(`clamped ${feature.toUpperCase()}`);
            clampedCount += 1;
          }
        });
      });
      cleaned.forEach((row) => {
        row.status = row.dropped ? row.status : row.statusParts.length ? row.statusParts.join(", ") : "Kept as-is";
      });

      summary.innerHTML = `${retained.length} rows remain. ${imputedCount} feature values were imputed and ${clampedCount} were clamped.`;
      tableBody.innerHTML = cleaned
        .map(
          (row) => `
            <tr>
              <td>${row.id}</td>
              <td>${Number.isFinite(row.a) ? U.round(row.a, 2) : "?"}</td>
              <td>${Number.isFinite(row.b) ? U.round(row.b, 2) : "?"}</td>
              <td>${row.status}</td>
            </tr>
          `
        )
        .join("");

      renderFormulaCards(mathNode, [
        { title: "Imputation rule", description: "Missing values must be resolved before the model sees the data — and every choice here changes the distribution the model learns from.", tex: method === "drop" ? "D' = \\{\\, \\mathbf{x}_i \\in D : \\mathbf{x}_i \\text{ complete} \\,\\}" : `\\tilde{x}_{j} = \\text{${method}}(\\{x_{ij} : x_{ij} \\text{ observed}\\})`, derivation: method === "drop" ? [{ tex: "\\text{rows with any missing feature are removed}", result: "" }] : [{ tex: `\\tilde{x}_A = ${U.texNum(fillA, 2)}, \\quad \\tilde{x}_B = ${U.texNum(fillB, 2)}`, result: "" }] },
        { title: "Noise rule", description: "Winsorising pulls extreme values back to a threshold instead of deleting them, keeping the row but limiting its leverage.", tex: `x' = \\mu + \\operatorname{sign}(z)\\cdot\\min\\!\\left(|z|,\\; ${clampZ.toFixed(1)}\\right)\\cdot\\sigma, \\qquad z = \\frac{x - \\mu}{\\sigma}` },
      ]);

      const chart = U.makeChart(plot, {
        xDomain: [0, 8.8],
        yDomain: [0, 8.8],
        title: "Original points vs cleaned points",
      });
      baseData.forEach((row) => {
        if (!Number.isFinite(row.a) || !Number.isFinite(row.b)) {
          return;
        }
        plot.appendChild(U.svgEl("circle", { cx: chart.xScale(row.a), cy: chart.yScale(row.b), r: 5, fill: C().neutral, opacity: "0.24" }));
      });
      cleaned.forEach((row) => {
        if (row.dropped || !Number.isFinite(row.a) || !Number.isFinite(row.b)) {
          return;
        }
        drawScatterPoint(plot, chart, row, row.changed ? C().b : C().a, row.changed ? 8 : 7);
      });
    }

    U.dataEditor(rootNode.querySelector("#cleanup-data"), {
      title: "Raw dataset",
      hint: "Leave a cell blank to mark it missing, or type an extreme number to create an outlier — then watch the imputation and clamping rules react to your edit.",
      columns: [
        { key: "id", label: "Example", type: "text" },
        { key: "a", label: "Feature A", type: "text" },
        { key: "b", label: "Feature B", type: "text" },
      ],
      rows: rawRows,
      minRows: 3,
      newRow: (current) => ({ id: `E${current.length + 1}`, a: 2, b: 2 }),
      onChange: (rows) => {
        rawRows = rows;
        syncBaseData();
        render();
      },
    });

    button.addEventListener("click", render);
    [methodInput, clampInput].forEach((input) => input.addEventListener("input", render));
    U.onRedraw(render);
    render();
  }

  function mountKde(rootNode) {
    mountSection(
      rootNode,
      "Kernel density estimation",
      "Adjust the bandwidth and reveal one kernel at a time. The formula panel keeps the KDE sum visible while the plot shows how local bumps aggregate into a global density.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-group">
              <label for="kde-samples">1D samples</label>
              <textarea id="kde-samples">0.9, 1.1, 1.8, 2.0, 2.1, 2.8, 3.1, 3.3, 4.4</textarea>
            </div>
            <div class="control-group">
              <label for="kde-bandwidth">Bandwidth</label>
              <div class="range-row">
                <input id="kde-bandwidth" type="range" min="0.15" max="1.2" step="0.05" value="0.45" />
                <span class="range-value" id="kde-bandwidth-value">0.45</span>
              </div>
            </div>
            <div class="step-controls">
              <button class="button primary" id="kde-step">Next step</button>
              <button class="button secondary" id="kde-reset">Reset steps</button>
            </div>
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg id="kde-plot" viewBox="0 0 560 280" aria-label="Kernel density estimation plot"></svg>
            </div>
            <div class="readout"><div class="output-grid" id="kde-output"></div></div>
            <div class="equation-card" id="kde-math"></div>
            <div class="steps"><ol id="kde-steps"></ol></div>
          </div>
        </div>
      `
    );

    const samplesInput = rootNode.querySelector("#kde-samples");
    const bandwidthInput = rootNode.querySelector("#kde-bandwidth");
    const bandwidthValue = rootNode.querySelector("#kde-bandwidth-value");
    const stepButton = rootNode.querySelector("#kde-step");
    const resetButton = rootNode.querySelector("#kde-reset");
    const plot = rootNode.querySelector("#kde-plot");
    const mathNode = rootNode.querySelector("#kde-math");
    const stepsNode = rootNode.querySelector("#kde-steps");
    const output = rootNode.querySelector("#kde-output");
    let revealed = 0;

    function render() {
      const samples = U.parseNumberList(samplesInput.value);
      const bandwidth = Math.max(Number(bandwidthInput.value) || 0.4, 0.08);
      bandwidthValue.textContent = bandwidth.toFixed(2);
      if (!samples.length) {
        return;
      }

      const xMin = Math.min(...samples) - 1;
      const xMax = Math.max(...samples) + 1;
      const xs = U.linspace(xMin, xMax, 180);
      const kernels = samples.map((center) =>
        xs.map((x) => ({ x, y: U.gaussianPdf(x, center, bandwidth * bandwidth) / samples.length }))
      );
      const total = xs.map((x) => ({
        x,
        y:
          samples.reduce((acc, center) => acc + U.gaussianPdf(x, center, bandwidth * bandwidth), 0) /
          samples.length,
      }));
      const yMax = Math.max(...total.map((point) => point.y), ...kernels.flat().map((point) => point.y)) * 1.25;

      /* Summary of the estimate itself, so the effect of the bandwidth
         is legible as numbers and not only as a shape. */
      const peak = total.reduce((best, point) => (point.y > best.y ? point : best), total[0]);
      const peakDensity = peak.y;
      const step = xs[1] - xs[0];
      const area = total.reduce((acc, point) => acc + point.y * step, 0);
      const modes = total.filter(
        (point, index) =>
          index > 0 && index < total.length - 1 && point.y > total[index - 1].y && point.y > total[index + 1].y
      ).length;

      U.renderMetrics(output, [
        { label: "Samples", value: String(samples.length) },
        { label: "Bandwidth h", value: bandwidth.toFixed(2) },
        { label: "Peak at x", value: U.round(peak.x, 2) },
        { label: "Modes", value: String(modes) },
      ]);

      renderFormulaCards(mathNode, [
        {
          title: "KDE formula",
          description: "Average one Gaussian kernel per sample.",
          tex: "\\hat{p}(x) = \\frac{1}{Nh} \\sum_{i=1}^{N} K\\!\\left(\\frac{x - x_i}{h}\\right)",
          derivation: (() => {
            const lines = [];
            lines.push({
              tex: `N = ${samples.length}, \\qquad h = ${bandwidth.toFixed(2)}`,
              result: bandwidth.toFixed(2),
            });
            const visibleSamples = samples.slice(0, Math.min(revealed, samples.length));
            if (visibleSamples.length > 0) {
              lines.push({
                tex: visibleSamples
                  .slice(0, 3)
                  .map((value) => `K\\!\\left(\\frac{x - ${U.round(value, 2)}}{${bandwidth.toFixed(2)}}\\right)`)
                  .join(" + ") + (visibleSamples.length > 3 ? " + \\cdots" : ""),
                result: `${visibleSamples.length} of ${samples.length}`,
                note: "One kernel per sample, each centred on its own observation.",
              });
              if (revealed > samples.length) {
                lines.push({
                  tex: `\\hat{p}(x) = \\frac{1}{${samples.length}}\\sum_{i=1}^{${samples.length}} K_i(x)`,
                  result: U.round(peakDensity, 4),
                  note: "Averaging keeps the total area at 1, so the result is a genuine density.",
                });
              }
            }
            return lines;
          })()
        },
        {
          title: "Bandwidth effect",
          description:
            "The bandwidth is the only real choice here, and it matters more than the shape of the kernel. Too small and the estimate is a spiky copy of the sample; too large and genuine structure is smoothed away.",
          tex: `K(u) = \\frac{1}{\\sqrt{2\\pi}}e^{-u^2/2}, \\qquad h = ${bandwidth.toFixed(2)}`,
          derivation: [
            {
              tex: `\\int \\hat{p}(x)\\,dx \\approx ${U.texNum(area, 3)}`,
              result: U.round(area, 3),
              note: "Numerically integrating the estimate returns 1, confirming it is a probability density.",
            },
            {
              tex: `\\text{modes detected} = ${modes}`,
              result: String(modes),
              note: "Slide the bandwidth down and spurious modes appear, one per sample in the limit; slide it up and they merge into a single bump.",
            },
          ],
          insight:
            "<strong>There is a rule of thumb, not an answer.</strong> Silverman's rule gives h ≈ 1.06 σ n^(-1/5) for roughly Gaussian data, but bandwidth selection is genuinely a modelling decision — which is the price of not assuming a parametric form.",
        },
      ]);

      const steps = samples.map(
        (sample, index) =>
          `<strong>Kernel ${index + 1}.</strong> Center a Gaussian bump at <strong>${U.round(sample, 2)}</strong>.`
      );
      steps.push("<strong>Average the kernels.</strong> The final density is the pointwise average of all kernel contributions.");
      U.renderSteps(stepsNode, steps, revealed);

      const chart = U.makeChart(plot, {
        xDomain: [xMin, xMax],
        yDomain: [0, yMax],
        title: "Kernel contributions and final density",
      });
      samples.forEach((sample) => {
        const x = chart.xScale(sample);
        plot.appendChild(U.svgEl("line", { x1: x, y1: chart.yScale(0), x2: x, y2: chart.yScale(yMax * 0.08), stroke: C().ink, "stroke-width": "1.6" }));
      });
      for (let index = 0; index < Math.min(revealed, samples.length); index += 1) {
        plot.appendChild(U.svgEl("path", { d: U.pathFromPoints(kernels[index], chart.xScale, chart.yScale), fill: "none", stroke: index % 2 ? C().b : C().a, "stroke-width": "2.2", opacity: "0.6" }));
      }
      if (revealed > samples.length) {
        plot.appendChild(U.svgEl("path", { d: U.pathFromPoints(total, chart.xScale, chart.yScale), class: "curve-primary" }));
      }
    }

    stepButton.addEventListener("click", () => {
      const count = U.parseNumberList(samplesInput.value).length;
      revealed = Math.min(revealed + 1, count + 1);
      render();
    });
    resetButton.addEventListener("click", () => {
      revealed = 0;
      render();
    });
    [samplesInput, bandwidthInput].forEach((input) =>
      input.addEventListener("input", () => {
        revealed = 0;
        render();
      })
    );
    U.onRedraw(render);
    render();
  }

  function mountKnn(rootNode) {
    mountSection(
      rootNode,
      "k-nearest neighbour voting",
      "Type your own training points, drag them around the chart, then move the query and reveal neighbours one at a time. The distance formula and the vote are recomputed from whatever data is in the table.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-grid">
              <div class="control-group">
                <label for="knn-k">k neighbours</label>
                <input id="knn-k" type="number" min="1" max="9" value="3" />
              </div>
              <div class="control-group">
                <label for="knn-metric">Distance</label>
                <select id="knn-metric">
                  <option value="euclidean">Euclidean</option>
                  <option value="manhattan">Manhattan</option>
                </select>
              </div>
              <div class="control-group">
                <label for="knn-query-x">Query x</label>
                <input id="knn-query-x" type="number" min="0" max="10" step="0.1" value="5.4" />
              </div>
              <div class="control-group">
                <label for="knn-query-y">Query y</label>
                <input id="knn-query-y" type="number" min="0" max="10" step="0.1" value="4.4" />
              </div>
            </div>
            <div class="step-controls">
              <button class="button primary" id="knn-step">Next neighbour</button>
              <button class="button secondary" id="knn-reset">Reset steps</button>
            </div>
            ${editorSlot("knn-data")}
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg id="knn-plot" viewBox="0 0 560 280" aria-label="k nearest neighbours plot"></svg>
              ${DRAG_HINT}
            </div>
            <div class="readout"><div class="output-grid" id="knn-output"></div></div>
            <div class="equation-card" id="knn-math"></div>
            <div class="steps"><ol id="knn-steps"></ol></div>
            <div class="table-panel">
              <table>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Point</th>
                    <th>Class</th>
                    <th>Distance</th>
                  </tr>
                </thead>
                <tbody id="knn-table"></tbody>
              </table>
            </div>
          </div>
        </div>
      `
    );

    const kInput = rootNode.querySelector("#knn-k");
    const metricInput = rootNode.querySelector("#knn-metric");
    const queryXInput = rootNode.querySelector("#knn-query-x");
    const queryYInput = rootNode.querySelector("#knn-query-y");
    const stepButton = rootNode.querySelector("#knn-step");
    const resetButton = rootNode.querySelector("#knn-reset");
    const plot = rootNode.querySelector("#knn-plot");
    const output = rootNode.querySelector("#knn-output");
    const mathNode = rootNode.querySelector("#knn-math");
    const stepsNode = rootNode.querySelector("#knn-steps");
    const table = rootNode.querySelector("#knn-table");

    let points = knnDefaults.map((point) => ({ ...point }));
    let revealed = 0;

    const editor = U.dataEditor(rootNode.querySelector("#knn-data"), {
      title: "Training points",
      hint: "Each row is one labelled example. Change a coordinate, flip a class, delete a point, or add your own — the neighbourhood re-sorts immediately.",
      columns: [
        { key: "id", label: "Name", type: "text" },
        { key: "x", label: "x", min: 0, max: 10, step: 0.1 },
        { key: "y", label: "y", min: 0, max: 10, step: 0.1 },
        {
          key: "label",
          label: "Class",
          type: "select",
          options: [
            { value: "A", label: "A" },
            { value: "B", label: "B" },
          ],
        },
      ],
      rows: points,
      presets: knnPresets,
      minRows: 2,
      newRow: (current) => ({
        id: `P${current.length + 1}`,
        x: Number(U.round(Math.random() * 9 + 0.5, 1)),
        y: Number(U.round(Math.random() * 9 + 0.5, 1)),
        label: current.length % 2 ? "B" : "A",
      }),
      onChange: (rows) => {
        points = rows;
        revealed = 0;
        render();
      },
    });

    function render() {
      const k = U.clamp(Math.round(Number(kInput.value) || 3), 1, Math.max(points.length, 1));
      const metric = metricInput.value;
      const query = { id: "Q", x: Number(queryXInput.value) || 0, y: Number(queryYInput.value) || 0 };
      const sorted = points
        .map((point) => ({ ...point, distance: U.distance(point, query, metric) }))
        .sort((a, b) => a.distance - b.distance);

      if (!sorted.length) {
        U.renderMetrics(output, [{ label: "Data", value: "Add at least one point" }]);
        U.makeChart(plot, { xDomain: [0, 10], yDomain: [0, 10], title: "No training data" });
        mathNode.innerHTML = "";
        table.innerHTML = "";
        U.renderSteps(stepsNode, [], 0);
        return;
      }

      const visibleCount = Math.max(0, Math.min(k, revealed - 2));
      const visible = sorted.slice(0, visibleCount);
      const votes = visible.reduce(
        (acc, point) => {
          acc[point.label] = (acc[point.label] || 0) + 1;
          return acc;
        },
        { A: 0, B: 0 }
      );
      const prediction =
        visibleCount < k ? "Pending" : votes.A === votes.B ? "Tie" : votes.A > votes.B ? "Class A" : "Class B";

      U.renderMetrics(output, [
        { label: "Training points", value: String(points.length) },
        { label: "k", value: String(k) },
        { label: "Vote A : B", value: `${votes.A} : ${votes.B}` },
        { label: "Prediction", value: prediction },
      ]);

      const nn1 = sorted[0];
      const dx1 = nn1.x - query.x;
      const dy1 = nn1.y - query.y;
      const metricTex =
        metric === "euclidean"
          ? "d(\\mathbf{x}, \\mathbf{q}) = \\sqrt{(x_1 - q_1)^2 + (x_2 - q_2)^2}"
          : "d(\\mathbf{x}, \\mathbf{q}) = |x_1 - q_1| + |x_2 - q_2|";

      const distanceDerivation = [
        {
          tex: `\\mathbf{q} = (${U.texNum(query.x, 2)},\\; ${U.texNum(query.y, 2)}) \\quad \\text{nearest} = \\text{${nn1.id}}\\;(${U.texNum(nn1.x, 2)},\\; ${U.texNum(nn1.y, 2)})`,
          result: "",
        },
      ];
      if (metric === "euclidean") {
        distanceDerivation.push({
          tex: `\\Delta x = ${U.texNum(nn1.x, 2)} - ${U.texNum(query.x, 2)} = ${U.texNum(dx1, 2)}, \\quad \\Delta y = ${U.texNum(nn1.y, 2)} - ${U.texNum(query.y, 2)} = ${U.texNum(dy1, 2)}`,
          result: "",
        });
        distanceDerivation.push({
          tex: `d = \\sqrt{(${U.texNum(dx1, 2)})^2 + (${U.texNum(dy1, 2)})^2} = \\sqrt{${U.texNum(dx1 * dx1 + dy1 * dy1, 3)}}`,
          result: U.round(nn1.distance, 4),
        });
      } else {
        distanceDerivation.push({
          tex: `d = |${U.texNum(dx1, 2)}| + |${U.texNum(dy1, 2)}| = ${U.texNum(Math.abs(dx1), 2)} + ${U.texNum(Math.abs(dy1), 2)}`,
          result: U.round(nn1.distance, 4),
        });
      }
      sorted.slice(1, k).forEach((point, index) => {
        const dx = point.x - query.x;
        const dy = point.y - query.y;
        distanceDerivation.push({
          tex:
            metric === "euclidean"
              ? `\\#${index + 2}\\;\\text{${point.id}}:\\; \\sqrt{(${U.texNum(dx, 2)})^2 + (${U.texNum(dy, 2)})^2}`
              : `\\#${index + 2}\\;\\text{${point.id}}:\\; |${U.texNum(dx, 2)}| + |${U.texNum(dy, 2)}|`,
          result: U.round(point.distance, 3),
        });
      });

      const voteLines = Object.entries(votes).map(([className, count]) => ({
        tex: `\\sum_{i=1}^{${visibleCount || k}} \\mathbb{1}[y_i = \\text{${className}}] = ${count}`,
        result: `${count}/${visibleCount || k}`,
      }));

      renderFormulaCards(mathNode, [
        {
          title: "Distance formula",
          description:
            metric === "euclidean"
              ? "Euclidean distance is the straight-line separation in feature space — it treats all dimensions equally."
              : "Manhattan distance sums absolute differences per dimension, like navigating city blocks.",
          tex: metricTex,
          derivation: distanceDerivation,
          insight:
            "<strong>Feature scaling matters:</strong> if x spans 0–100 but y spans 0–1, Euclidean distance is dominated by x. Try setting one column to a much larger range in the table and watch the neighbours stop making sense — that is why standardising features comes first.",
        },
        {
          title: `Majority vote (k = ${k})`,
          description:
            "After sorting by distance, count how many of the k nearest neighbours belong to each class. The most common class wins.",
          tex: "\\hat{y} = \\arg\\max_{c} \\sum_{i=1}^{k} \\mathbb{1}[y_i = c]",
          derivation:
            visibleCount > 0
              ? [
                  ...voteLines,
                  {
                    tex: `\\hat{y} = \\text{${prediction}}`,
                    result: prediction,
                    note: `${votes.A} A-vote${votes.A !== 1 ? "s" : ""} against ${votes.B} B-vote${votes.B !== 1 ? "s" : ""} among the ${visibleCount} revealed neighbour${visibleCount !== 1 ? "s" : ""}.`,
                  },
                ]
              : [{ tex: "\\text{press ``Next neighbour'' to reveal the vote}", result: "" }],
          insight:
            "<strong>Odd k avoids ties:</strong> larger k reduces variance but smooths the boundary; smaller k captures finer local structure. Load the “One noisy outlier” preset and compare k = 1 with k = 5.",
        },
      ]);

      const steps = [
        "<strong>Measure distances.</strong> Compute the chosen metric from every training point to the query.",
        "<strong>Sort by closeness.</strong> Rank all points — the nearest become the strongest evidence.",
        ...sorted
          .slice(0, k)
          .map(
            (point, index) =>
              `<strong>Neighbour ${index + 1}.</strong> ${point.id} (class <strong>${point.label}</strong>) — ${
                metric === "euclidean" ? "Euclidean" : "Manhattan"
              } distance = <strong>${U.round(point.distance, 4)}</strong>.`
          ),
        `<strong>Vote.</strong> Tally: A=${votes.A}, B=${votes.B} → prediction = <strong>${prediction}</strong>.`,
      ];
      U.renderSteps(stepsNode, steps, revealed);

      table.innerHTML = sorted
        .map(
          (point, index) => `
            <tr style="background:${index < visibleCount ? U.tint(C().a, 0.08) : "transparent"}">
              <td>${index + 1}</td>
              <td>${point.id} (${U.round(point.x, 1)}, ${U.round(point.y, 1)})</td>
              <td>${point.label}</td>
              <td>${U.round(point.distance, 3)}</td>
            </tr>
          `
        )
        .join("");

      const chart = U.makeChart(plot, { xDomain: [0, 10], yDomain: [0, 10], title: "Neighbourhood around the query" });

      /* Shade the ball that contains the k revealed neighbours so the
         "local evidence" idea is visible, not just described. */
      if (visibleCount > 0 && metric === "euclidean") {
        const radius = visible[visible.length - 1].distance;
        plot.appendChild(
          U.svgEl("ellipse", {
            cx: chart.xScale(query.x),
            cy: chart.yScale(query.y),
            rx: Math.abs(chart.xScale(radius) - chart.xScale(0)),
            ry: Math.abs(chart.yScale(0) - chart.yScale(radius)),
            fill: U.tint(C().ink, 0.06),
            stroke: U.tint(C().ink, 0.35),
            "stroke-dasharray": "5 5",
          })
        );
      }

      visible.forEach((point) => {
        plot.appendChild(
          U.svgEl("line", {
            x1: chart.xScale(query.x),
            y1: chart.yScale(query.y),
            x2: chart.xScale(point.x),
            y2: chart.yScale(point.y),
            stroke: C().ink,
            "stroke-width": "2",
            opacity: "0.55",
          })
        );
      });

      points.forEach((point, index) => {
        drawScatterPoint(
          plot,
          chart,
          point,
          point.label === "A" ? C().a : C().b,
          visible.some((entry) => entry.id === point.id) ? 9 : 7,
          (target, position) => {
            target.x = Number(U.round(position.x, 2));
            target.y = Number(U.round(position.y, 2));
            editor.setCell(index, "x", target.x);
            editor.setCell(index, "y", target.y);
            scheduleRender();
          }
        );
      });

      const qx = chart.xScale(query.x);
      const qy = chart.yScale(query.y);
      const marker = U.svgEl("polygon", {
        points: `${qx},${qy - 11} ${qx + 11},${qy} ${qx},${qy + 11} ${qx - 11},${qy}`,
        fill: C().ink,
      });
      plot.appendChild(marker);
      U.draggable(marker, plot, chart, (position) => {
        queryXInput.value = U.round(position.x, 2);
        queryYInput.value = U.round(position.y, 2);
        scheduleRender();
      });
      plot.appendChild(U.svgEl("text", { x: qx + 14, y: qy - 12, class: "svg-label" })).textContent = "Query (drag me)";
    }

    const scheduleRender = U.rafThrottle(render);

    stepButton.addEventListener("click", () => {
      const k = U.clamp(Math.round(Number(kInput.value) || 3), 1, Math.max(points.length, 1));
      revealed = Math.min(revealed + 1, k + 3);
      render();
    });
    resetButton.addEventListener("click", () => {
      revealed = 0;
      render();
    });
    [kInput, metricInput, queryXInput, queryYInput].forEach((input) =>
      input.addEventListener("input", () => {
        revealed = 0;
        render();
      })
    );
    U.onRedraw(render);
    render();
  }

  function gini(rows) {
    if (!rows.length) {
      return 0;
    }
    const p = rows.filter((row) => row.label === 1).length / rows.length;
    return 1 - p * p - (1 - p) * (1 - p);
  }

  function mse(rows) {
    if (!rows.length) {
      return 0;
    }
    const avg = U.mean(rows.map((row) => row.score));
    return U.mean(rows.map((row) => (row.score - avg) ** 2));
  }

  function getCandidateSplits(rows, mode) {
    if (mode === "classification") {
      const features = ["hours", "attendance"];
      const parent = gini(rows);
      return features.flatMap((feature) => {
        const values = [...new Set(rows.map((row) => row[feature]).sort((a, b) => a - b))];
        return values.slice(0, -1).map((value, index) => {
          const threshold = 0.5 * (value + values[index + 1]);
          const left = rows.filter((row) => row[feature] <= threshold);
          const right = rows.filter((row) => row[feature] > threshold);
          const score = parent - (left.length / rows.length) * gini(left) - (right.length / rows.length) * gini(right);
          return { feature, threshold, score, left, right };
        });
      });
    }
    const values = [...new Set(rows.map((row) => row.hours).sort((a, b) => a - b))];
    const parent = mse(rows);
    return values.slice(0, -1).map((value, index) => {
      const threshold = 0.5 * (value + values[index + 1]);
      const left = rows.filter((row) => row.hours <= threshold);
      const right = rows.filter((row) => row.hours > threshold);
      const score = parent - (left.length / rows.length) * mse(left) - (right.length / rows.length) * mse(right);
      return { feature: "hours", threshold, score, left, right };
    });
  }

  function mountTree(rootNode) {
    const allowModeSwitch = definition.id === "classification-and-regression-trees";
    const initialMode = definition.options.mode || "classification";
    mountSection(
      rootNode,
      "Greedy tree split explorer",
      "Advance the tree one split at a time. Every candidate threshold is scored explicitly, so the greedy choice is visible instead of implicit. Edit the training table and the ranking of splits changes with it.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-grid">
              <div class="control-group">
                <label for="tree-mode">Mode</label>
                <select id="tree-mode" ${allowModeSwitch ? "" : "disabled"}>
                  <option value="classification">Classification tree</option>
                  <option value="regression">Regression tree</option>
                </select>
              </div>
              <div class="control-group">
                <label for="tree-depth">Maximum depth shown</label>
                <input id="tree-depth" type="number" min="1" max="4" value="2" />
              </div>
            </div>
            <div class="step-controls">
              <button class="button primary" id="tree-step">Next split</button>
              <button class="button secondary" id="tree-reset">Reset tree</button>
            </div>
            <div class="callout" id="tree-callout"></div>
            ${editorSlot("tree-data")}
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg id="tree-plot" viewBox="0 0 560 280" aria-label="Tree plot"></svg>
              ${DRAG_HINT}
            </div>
            <div class="equation-card" id="tree-math"></div>
            <div class="steps"><ol id="tree-steps"></ol></div>
            <div class="table-panel">
              <table>
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Threshold</th>
                    <th>Gain</th>
                    <th>Leaf</th>
                  </tr>
                </thead>
                <tbody id="tree-candidates"></tbody>
              </table>
            </div>
          </div>
        </div>
      `
    );

    const modeInput = rootNode.querySelector("#tree-mode");
    const depthInput = rootNode.querySelector("#tree-depth");
    const stepButton = rootNode.querySelector("#tree-step");
    const resetButton = rootNode.querySelector("#tree-reset");
    const plot = rootNode.querySelector("#tree-plot");
    const mathNode = rootNode.querySelector("#tree-math");
    const stepsNode = rootNode.querySelector("#tree-steps");
    const candidatesNode = rootNode.querySelector("#tree-candidates");
    const callout = rootNode.querySelector("#tree-callout");
    const editorNode = rootNode.querySelector("#tree-data");

    let classificationRows = treeClassificationData.map((row) => ({ ...row }));
    let regressionRows = treeRegressionData.map((row) => ({ ...row }));
    let state;
    let editor;

    modeInput.value = initialMode;

    function activeRows() {
      return modeInput.value === "classification" ? classificationRows : regressionRows;
    }

    /* The two modes need different columns, so the editor is rebuilt
       whenever the mode changes rather than trying to be generic. */
    function buildEditor() {
      const isClassification = modeInput.value === "classification";
      editor = U.dataEditor(editorNode, {
        title: isClassification ? "Training examples" : "Training examples (hours → score)",
        hint: isClassification
          ? "Two features and a binary label. Move a point across the boundary and watch which threshold becomes the greedy winner."
          : "One feature and a continuous target. Each leaf will predict the mean score of the rows that fall inside it.",
        columns: isClassification
          ? [
              { key: "id", label: "ID", type: "text" },
              { key: "hours", label: "Hours", min: 0, max: 10, step: 0.1 },
              { key: "attendance", label: "Attendance", min: 0, max: 10, step: 0.1 },
              {
                key: "label",
                label: "Label",
                type: "select",
                options: [
                  { value: 0, label: "0 — fail" },
                  { value: 1, label: "1 — pass" },
                ],
              },
            ]
          : [
              { key: "id", label: "ID", type: "text" },
              { key: "hours", label: "Hours", min: 0, max: 12, step: 0.1 },
              { key: "score", label: "Score", min: 0, max: 100, step: 1 },
            ],
        rows: isClassification ? classificationRows : regressionRows,
        minRows: 3,
        newRow: (current) =>
          isClassification
            ? {
                id: `S${current.length + 1}`,
                hours: Number(U.round(Math.random() * 5 + 0.8, 1)),
                attendance: Number(U.round(Math.random() * 5 + 2, 1)),
                label: current.length % 2,
              }
            : {
                id: `R${current.length + 1}`,
                hours: Number(U.round(Math.random() * 6 + 0.8, 1)),
                score: Math.round(Math.random() * 45 + 45),
              },
        onChange: (rows) => {
          const normalized = rows.map((row) => ({ ...row, label: row.label !== undefined ? Number(row.label) : undefined }));
          if (isClassification) classificationRows = normalized;
          else regressionRows = normalized;
          resetState();
          render();
        },
      });
    }

    function resetState() {
      const mode = modeInput.value;
      const rows = activeRows();
      state = {
        mode,
        leaves: [
          {
            id: "root",
            depth: 0,
            rows,
            region:
              mode === "classification"
                ? { xMin: 0, xMax: 10, yMin: 0, yMax: 10 }
                : { xMin: 0, xMax: 12 },
          },
        ],
        history: [],
      };
    }

    function takeSplit() {
      const maxDepth = Math.max(Number(depthInput.value) || 2, 1);
      const choices = state.leaves
        .filter((leaf) => leaf.depth < maxDepth)
        .map((leaf) => {
          const candidates = getCandidateSplits(leaf.rows, state.mode).sort((a, b) => b.score - a.score);
          return { leaf, best: candidates[0] };
        })
        .filter((entry) => entry.best && entry.best.score > 1e-6);
      if (!choices.length) return;

      const choice = choices.sort((a, b) => b.best.score - a.best.score)[0];
      const leaf = choice.leaf;
      const best = choice.best;
      const leftRegion =
        state.mode === "classification"
          ? best.feature === "hours"
            ? { xMin: leaf.region.xMin, xMax: best.threshold, yMin: leaf.region.yMin, yMax: leaf.region.yMax }
            : { xMin: leaf.region.xMin, xMax: leaf.region.xMax, yMin: leaf.region.yMin, yMax: best.threshold }
          : { xMin: leaf.region.xMin, xMax: best.threshold };
      const rightRegion =
        state.mode === "classification"
          ? best.feature === "hours"
            ? { xMin: best.threshold, xMax: leaf.region.xMax, yMin: leaf.region.yMin, yMax: leaf.region.yMax }
            : { xMin: leaf.region.xMin, xMax: leaf.region.xMax, yMin: best.threshold, yMax: leaf.region.yMax }
          : { xMin: best.threshold, xMax: leaf.region.xMax };
      state.leaves = state.leaves
        .filter((entry) => entry.id !== leaf.id)
        .concat([
          { id: `${leaf.id}L`, depth: leaf.depth + 1, rows: best.left, region: leftRegion },
          { id: `${leaf.id}R`, depth: leaf.depth + 1, rows: best.right, region: rightRegion },
        ]);
      state.history.push({
        feature: best.feature,
        threshold: best.threshold,
        score: best.score,
        depth: leaf.depth,
        region: leaf.region,
      });
    }

    function render() {
      if (!state || state.mode !== modeInput.value) resetState();

      const maxDepth = Math.max(Number(depthInput.value) || 2, 1);
      const nextCandidates = state.leaves
        .filter((leaf) => leaf.depth < maxDepth)
        .flatMap((leaf) =>
          getCandidateSplits(leaf.rows, state.mode).map((candidate) => ({ leafId: leaf.id, ...candidate }))
        )
        .sort((a, b) => b.score - a.score)
        .slice(0, 6);
      const topCandidate = nextCandidates[0];

      callout.innerHTML = topCandidate
        ? `Best available split: <strong>${topCandidate.feature} ≤ ${U.round(topCandidate.threshold, 2)}</strong> — gain <strong>${U.round(topCandidate.score, 4)}</strong>.`
        : "No additional split improves the current tree within the selected depth.";

      /* Build the substituted derivation BEFORE rendering the cards. */
      const parentRows = state.leaves.flatMap((leaf) => leaf.rows);
      const n = parentRows.length;
      let derivLines = [];

      if (topCandidate && state.mode === "classification") {
        const parentGini = gini(parentRows);
        const p1 = parentRows.filter((row) => Number(row.label) === 1).length;
        const p0 = n - p1;
        const gLeft = gini(topCandidate.left);
        const gRight = gini(topCandidate.right);
        const weighted = (topCandidate.left.length / n) * gLeft + (topCandidate.right.length / n) * gRight;
        derivLines = [
          { tex: `n = ${n},\\quad n_{1} = ${p1},\\quad n_{0} = ${p0}`, result: "" },
          {
            tex: `p_1 = \\tfrac{${p1}}{${n}} = ${U.texNum(p1 / n, 3)}, \\quad p_0 = \\tfrac{${p0}}{${n}} = ${U.texNum(p0 / n, 3)}`,
            result: "",
          },
          {
            tex: `G(\\text{parent}) = 1 - (${U.texNum(p1 / n, 3)})^2 - (${U.texNum(p0 / n, 3)})^2`,
            result: U.round(parentGini, 4),
          },
          {
            tex: `\\text{split: } \\text{${topCandidate.feature}} \\le ${U.texNum(topCandidate.threshold, 2)} \\;\\Rightarrow\\; n_L = ${topCandidate.left.length},\\; n_R = ${topCandidate.right.length}`,
            result: "",
          },
          { tex: `G(\\text{left}) = ${U.texNum(gLeft, 4)}, \\quad G(\\text{right}) = ${U.texNum(gRight, 4)}`, result: "" },
          {
            tex: `\\tfrac{${topCandidate.left.length}}{${n}}(${U.texNum(gLeft, 4)}) + \\tfrac{${topCandidate.right.length}}{${n}}(${U.texNum(gRight, 4)})`,
            result: U.round(weighted, 4),
          },
          {
            tex: `\\text{Gain} = ${U.texNum(parentGini, 4)} - ${U.texNum(weighted, 4)}`,
            result: U.round(topCandidate.score, 4),
            note: "The split with the largest gain is chosen at each node — that is the whole greedy criterion.",
          },
        ];
      } else if (topCandidate && state.mode === "regression") {
        const parentMSE = mse(parentRows);
        const mLeft = mse(topCandidate.left);
        const mRight = mse(topCandidate.right);
        const weighted = (topCandidate.left.length / n) * mLeft + (topCandidate.right.length / n) * mRight;
        derivLines = [
          {
            tex: `\\bar{y} = ${U.texNum(U.mean(parentRows.map((row) => row.score)), 2)}, \\quad \\text{MSE(parent)} = ${U.texNum(parentMSE, 4)}`,
            result: "",
          },
          {
            tex: `\\text{split: hours} \\le ${U.texNum(topCandidate.threshold, 2)} \\;\\Rightarrow\\; n_L = ${topCandidate.left.length},\\; n_R = ${topCandidate.right.length}`,
            result: "",
          },
          { tex: `\\text{MSE}_L = ${U.texNum(mLeft, 4)}, \\quad \\text{MSE}_R = ${U.texNum(mRight, 4)}`, result: "" },
          {
            tex: `\\tfrac{${topCandidate.left.length}}{${n}}(${U.texNum(mLeft, 4)}) + \\tfrac{${topCandidate.right.length}}{${n}}(${U.texNum(mRight, 4)})`,
            result: U.round(weighted, 4),
          },
          {
            tex: `\\text{Gain} = ${U.texNum(parentMSE, 4)} - ${U.texNum(weighted, 4)}`,
            result: U.round(topCandidate.score, 4),
            note: `Each leaf predicts the mean of its group — the left leaf would predict ${U.round(U.mean(topCandidate.left.map((row) => row.score)), 2)}.`,
          },
        ];
      } else {
        derivLines = [{ tex: "\\text{no split improves the objective}", result: "" }];
      }

      renderFormulaCards(mathNode, [
        {
          title: state.mode === "classification" ? "Gini impurity" : "Squared-error criterion",
          description:
            state.mode === "classification"
              ? "Gini impurity measures how often a randomly drawn example would be misclassified. A pure node has G = 0; a perfectly mixed binary node has G = 0.5."
              : "Regression trees minimise within-leaf variance by choosing the threshold that most reduces mean squared error.",
          tex:
            state.mode === "classification"
              ? "G(t) = 1 - \\sum_{c} p_c^{\\,2}"
              : "\\text{MSE}(t) = \\frac{1}{n_t}\\sum_{i \\in t} (y_i - \\bar{y}_t)^2",
          derivation: derivLines,
          insight:
            state.mode === "classification"
              ? "<strong>Why Gini and not entropy?</strong> Both produce very similar splits in practice. Gini avoids the logarithm, which makes it cheaper to evaluate across thousands of candidate thresholds."
              : "<strong>Piecewise-constant prediction:</strong> each leaf predicts the mean of its training points. More splits means lower training error but higher variance — the core overfitting tradeoff.",
        },
        {
          title: "Greedy split selection",
          description:
            "At every node the algorithm scores every (feature, threshold) pair and picks the one with the largest impurity reduction. This local greedy choice is not globally optimal, but it is tractable.",
          tex: "\\text{Gain} = G(\\text{parent}) - \\left[\\frac{n_L}{n}G(\\text{left}) + \\frac{n_R}{n}G(\\text{right})\\right]",
          derivation: topCandidate
            ? [
                {
                  tex: `\\text{best} = \\left(\\text{${topCandidate.feature}} \\le ${U.texNum(topCandidate.threshold, 2)}\\right)`,
                  result: U.round(topCandidate.score, 4),
                  note: `Chosen from ${nextCandidates.length} candidate threshold${nextCandidates.length !== 1 ? "s" : ""} evaluated at this step.`,
                },
              ]
            : [{ tex: "\\text{no valid split within the depth limit}", result: "" }],
          insight:
            "<strong>Depth controls overfitting:</strong> a single-split stump is a weak learner. Raise the depth limit and keep splitting until each leaf holds one point — the regions become a memorised copy of your table.",
        },
      ]);

      U.renderSteps(
        stepsNode,
        state.history.map(
          (entry, index) =>
            `<strong>Split ${index + 1}.</strong> Apply <span class="mono">${entry.feature} ≤ ${U.round(entry.threshold, 2)}</span> — this improved the objective by <strong>${U.round(entry.score, 4)}</strong>.`
        ),
        state.history.length
      );

      candidatesNode.innerHTML = nextCandidates
        .map(
          (candidate) => `
            <tr>
              <td>${candidate.feature}</td>
              <td>${U.round(candidate.threshold, 2)}</td>
              <td>${U.round(candidate.score, 3)}</td>
              <td>${candidate.leafId}</td>
            </tr>
          `
        )
        .join("");

      if (state.mode === "classification") {
        const xs = classificationRows.map((row) => row.hours);
        const ys = classificationRows.map((row) => row.attendance);
        const chart = U.makeChart(plot, {
          xDomain: [Math.min(0, ...xs) - 0.5, Math.max(6, ...xs) + 0.5],
          yDomain: [Math.min(0, ...ys) - 0.5, Math.max(8, ...ys) + 0.5],
          title: "Classification tree regions",
        });
        state.history.forEach((entry) => {
          if (entry.feature === "hours") {
            plot.appendChild(
              U.svgEl("line", {
                x1: chart.xScale(entry.threshold),
                y1: chart.yScale(entry.region.yMin),
                x2: chart.xScale(entry.threshold),
                y2: chart.yScale(entry.region.yMax),
                class: "cluster-line",
              })
            );
          } else {
            plot.appendChild(
              U.svgEl("line", {
                x1: chart.xScale(entry.region.xMin),
                y1: chart.yScale(entry.threshold),
                x2: chart.xScale(entry.region.xMax),
                y2: chart.yScale(entry.threshold),
                class: "cluster-line",
              })
            );
          }
        });
        classificationRows.forEach((row, index) => {
          drawScatterPoint(
            plot,
            chart,
            { id: row.id, x: row.hours, y: row.attendance },
            Number(row.label) ? C().a : C().b,
            7,
            (_target, position) => {
              row.hours = Number(U.round(position.x, 2));
              row.attendance = Number(U.round(position.y, 2));
              editor.setCell(index, "hours", row.hours);
              editor.setCell(index, "attendance", row.attendance);
              resetState();
              scheduleRender();
            }
          );
        });
      } else {
        const xs = regressionRows.map((row) => row.hours);
        const ys = regressionRows.map((row) => row.score);
        const chart = U.makeChart(plot, {
          xDomain: [Math.min(0, ...xs), Math.max(7, ...xs) + 0.4],
          yDomain: [Math.min(...ys) - 6, Math.max(...ys) + 6],
          title: "Regression tree predictions",
        });
        state.history.forEach((entry) => {
          plot.appendChild(
            U.svgEl("line", {
              x1: chart.xScale(entry.threshold),
              y1: chart.yScale(chart.yDomain[0]),
              x2: chart.xScale(entry.threshold),
              y2: chart.yScale(chart.yDomain[1]),
              class: "cluster-line",
            })
          );
        });
        state.leaves.forEach((leaf) => {
          if (!leaf.rows.length) return;
          const prediction = U.mean(leaf.rows.map((row) => row.score));
          plot.appendChild(
            U.svgEl("line", {
              x1: chart.xScale(Math.max(leaf.region.xMin, chart.xDomain[0])),
              y1: chart.yScale(prediction),
              x2: chart.xScale(Math.min(leaf.region.xMax, chart.xDomain[1])),
              y2: chart.yScale(prediction),
              stroke: C().a,
              "stroke-width": "4",
            })
          );
        });
        regressionRows.forEach((row, index) => {
          drawScatterPoint(
            plot,
            chart,
            { id: row.id, x: row.hours, y: row.score },
            C().neutral,
            7,
            (_target, position) => {
              row.hours = Number(U.round(position.x, 2));
              row.score = Number(U.round(position.y, 1));
              editor.setCell(index, "hours", row.hours);
              editor.setCell(index, "score", row.score);
              resetState();
              scheduleRender();
            }
          );
        });
      }
    }

    const scheduleRender = U.rafThrottle(render);

    stepButton.addEventListener("click", () => {
      takeSplit();
      render();
    });
    resetButton.addEventListener("click", () => {
      resetState();
      render();
    });
    depthInput.addEventListener("input", () => {
      resetState();
      render();
    });
    modeInput.addEventListener("change", () => {
      buildEditor();
      resetState();
      render();
    });

    buildEditor();
    resetState();
    U.onRedraw(render);
    render();
  }

  function mountLinear(rootNode) {
    const forcedAlgorithm = definition.options.algorithm;
    const allowFeatureMap = forcedAlgorithm === "svm";
    const defaultFeatureMap = definition.options.featureMap || "linear";
    const rbfCenters = [
      { x: 2, y: 2 },
      { x: 2, y: 8 },
      { x: 8, y: 2 },
      { x: 8, y: 8 },
    ];

    mountSection(
      rootNode,
      `${definition.title} trainer`,
      "Edit the training points, set the learning rate, then step the update rule. Each click shows the exact arithmetic that moved the boundary — the same numbers you can see changing in the table.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-grid">
              <div class="control-group">
                <label for="linear-dataset">Dataset preset</label>
                <select id="linear-dataset">
                  <option value="separable">Linearly separable</option>
                  <option value="overlap">Overlapping classes</option>
                  <option value="xor">XOR challenge</option>
                </select>
              </div>
              <div class="control-group">
                <label for="feature-map">Feature representation</label>
                <select id="feature-map" ${allowFeatureMap ? "" : "disabled"}>
                  <option value="linear">Original 2D space</option>
                  <option value="rbf">Radial basis lift</option>
                </select>
              </div>
              <div class="control-group">
                <label for="linear-lr">Learning rate η</label>
                <div class="range-row">
                  <input id="linear-lr" type="range" min="0.02" max="1" step="0.02" value="0.3" />
                  <span class="range-value" id="linear-lr-value">0.30</span>
                </div>
              </div>
              <div class="control-group">
                <label for="linear-lambda">${forcedAlgorithm === "svm" ? "Regularisation λ" : "Auto-run steps"}</label>
                ${
                  forcedAlgorithm === "svm"
                    ? `<div class="range-row">
                         <input id="linear-lambda" type="range" min="0" max="0.4" step="0.01" value="0.04" />
                         <span class="range-value" id="linear-lambda-value">0.04</span>
                       </div>`
                    : `<button class="button secondary" id="linear-run">Run 20 steps</button>`
                }
              </div>
            </div>
            <div class="step-controls">
              <button class="button primary" id="linear-step">Train one step</button>
              <button class="button secondary" id="linear-reset">Reset model</button>
            </div>
            <div class="callout" id="linear-callout"></div>
            ${editorSlot("linear-data")}
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg id="linear-plot" viewBox="0 0 560 280" aria-label="Linear classification plot"></svg>
              ${DRAG_HINT}
            </div>
            <div class="readout"><div class="output-grid" id="linear-output"></div></div>
            <div class="equation-card" id="linear-math"></div>
            <div class="steps"><ol id="linear-steps"></ol></div>
          </div>
        </div>
      `
    );

    const datasetInput = rootNode.querySelector("#linear-dataset");
    const featureMapInput = rootNode.querySelector("#feature-map");
    const lrInput = rootNode.querySelector("#linear-lr");
    const lrValue = rootNode.querySelector("#linear-lr-value");
    const lambdaInput = rootNode.querySelector("#linear-lambda");
    const lambdaValue = rootNode.querySelector("#linear-lambda-value");
    const runButton = rootNode.querySelector("#linear-run");
    const stepButton = rootNode.querySelector("#linear-step");
    const resetButton = rootNode.querySelector("#linear-reset");
    const callout = rootNode.querySelector("#linear-callout");
    const plot = rootNode.querySelector("#linear-plot");
    const output = rootNode.querySelector("#linear-output");
    const mathNode = rootNode.querySelector("#linear-math");
    const stepsNode = rootNode.querySelector("#linear-steps");

    let points = linearDatasets.separable.map((point) => ({ ...point }));
    let state;

    featureMapInput.value = defaultFeatureMap;

    const editor = U.dataEditor(rootNode.querySelector("#linear-data"), {
      title: "Training points",
      hint: "Label +1 and −1 are the two classes. Drag a point across the boundary, or add one deep inside the wrong class, and step the trainer to watch the update rule react.",
      columns: [
        { key: "id", label: "ID", type: "text" },
        { key: "x", label: "x₁", min: 0, max: 10, step: 0.1 },
        { key: "y", label: "x₂", min: 0, max: 10, step: 0.1 },
        {
          key: "label",
          label: "y",
          type: "select",
          options: [
            { value: 1, label: "+1" },
            { value: -1, label: "−1" },
          ],
        },
      ],
      rows: points,
      presets: {
        "Linearly separable": linearDatasets.separable,
        "Overlapping classes": linearDatasets.overlap,
        "XOR challenge": linearDatasets.xor,
      },
      minRows: 2,
      newRow: (current) => ({
        id: `P${current.length + 1}`,
        x: Number(U.round(Math.random() * 9 + 0.5, 1)),
        y: Number(U.round(Math.random() * 9 + 0.5, 1)),
        label: current.length % 2 ? -1 : 1,
      }),
      onChange: (nextRows) => {
        points = nextRows.map((row) => ({ ...row, label: Number(row.label) }));
        resetState();
        render();
      },
    });

    function learningRate() {
      return Number(lrInput.value) || 0.3;
    }

    function lambda() {
      return lambdaInput ? Number(lambdaInput.value) : 0.04;
    }

    function featureVector(point) {
      if (featureMapInput.value === "rbf" && forcedAlgorithm === "svm") {
        const gamma = 0.18;
        return [
          1,
          ...rbfCenters.map((center) =>
            Math.exp(-gamma * ((point.x - center.x) ** 2 + (point.y - center.y) ** 2))
          ),
        ];
      }
      return [1, point.x, point.y];
    }

    function computeLDA(rows) {
      const positives = rows.filter((row) => row.label === 1);
      const negatives = rows.filter((row) => row.label === -1);
      if (!positives.length || !negatives.length) return null;
      const mu1 = [U.mean(positives.map((row) => row.x)), U.mean(positives.map((row) => row.y))];
      const mu0 = [U.mean(negatives.map((row) => row.x)), U.mean(negatives.map((row) => row.y))];
      const covariance = [
        [0, 0],
        [0, 0],
      ];
      rows.forEach((row) => {
        const meanVec = row.label === 1 ? mu1 : mu0;
        const diff = [row.x - meanVec[0], row.y - meanVec[1]];
        covariance[0][0] += diff[0] * diff[0];
        covariance[0][1] += diff[0] * diff[1];
        covariance[1][0] += diff[1] * diff[0];
        covariance[1][1] += diff[1] * diff[1];
      });
      const denom = Math.max(rows.length - 2, 1);
      covariance[0][0] /= denom;
      covariance[0][1] /= denom;
      covariance[1][0] /= denom;
      covariance[1][1] /= denom;
      const inverse = U.inverse2x2(covariance);
      const diff = [mu1[0] - mu0[0], mu1[1] - mu0[1]];
      const w = U.matVec(inverse, diff);
      const quad0 =
        0.5 *
        (mu0[0] * (inverse[0][0] * mu0[0] + inverse[0][1] * mu0[1]) +
          mu0[1] * (inverse[1][0] * mu0[0] + inverse[1][1] * mu0[1]));
      const quad1 =
        0.5 *
        (mu1[0] * (inverse[0][0] * mu1[0] + inverse[0][1] * mu1[1]) +
          mu1[1] * (inverse[1][0] * mu1[0] + inverse[1][1] * mu1[1]));
      const prior = Math.log(positives.length / negatives.length);
      return { w, b: -quad1 + quad0 + prior, mu1, mu0, covariance };
    }

    function scorePoint(point, params) {
      if (forcedAlgorithm === "lda") {
        if (!params) return 0;
        return params.b + params.w[0] * point.x + params.w[1] * point.y;
      }
      return U.dot(params.weights, featureVector(point));
    }

    function trainingAccuracy(params) {
      if (!points.length) return 0;
      let correct = 0;
      points.forEach((row) => {
        const prediction = scorePoint(row, params) >= 0 ? 1 : -1;
        if (prediction === row.label) correct += 1;
      });
      return correct / points.length;
    }

    function resetState() {
      state = {
        weights: Array(featureVector({ x: 0, y: 0 }).length).fill(0),
        step: 0,
        sampleIndex: 0,
        history: [],
        lastId: null,
        lastUpdate: null,
      };
    }

    function takeStep() {
      if (forcedAlgorithm === "lda") {
        state.step = Math.min(state.step + 1, 3);
        return;
      }
      if (!points.length) return;

      const row = points[state.sampleIndex % points.length];
      const x = featureVector(row);
      const y = row.label;
      const score = U.dot(state.weights, x);
      const eta = learningRate();
      const before = [...state.weights];

      if (forcedAlgorithm === "perceptron") {
        if (y * score <= 0) {
          U.addScaled(state.weights, x, eta * y);
          state.history.push(
            `Perceptron update on <strong>${row.id}</strong>: y·(wᵀx) = ${U.round(y * score, 3)} ≤ 0, so <span class="mono">w ← w + ηyx</span>.`
          );
          state.lastUpdate = { kind: "update", row, x, y, score, eta, before, after: [...state.weights] };
        } else {
          state.history.push(
            `Perceptron check on <strong>${row.id}</strong>: y·(wᵀx) = ${U.round(y * score, 3)} > 0, already correct, so the weights stay fixed.`
          );
          state.lastUpdate = { kind: "skip", row, x, y, score, eta, before, after: [...state.weights] };
        }
      } else if (forcedAlgorithm === "logistic") {
        const target = y === 1 ? 1 : 0;
        const probability = U.sigmoid(score);
        U.addScaled(state.weights, x, eta * (target - probability));
        state.history.push(
          `Logistic step on <strong>${row.id}</strong>: p = ${U.round(probability, 3)}, residual (y − p) = ${U.round(target - probability, 3)}, then move along the cross-entropy gradient.`
        );
        state.lastUpdate = {
          kind: "update",
          row,
          x,
          y,
          score,
          eta,
          probability,
          target,
          before,
          after: [...state.weights],
        };
      } else {
        const margin = y * score;
        const reg = lambda();
        state.weights = state.weights.map((weight, index) => (index === 0 ? weight : weight * (1 - eta * reg)));
        if (margin < 1) {
          U.addScaled(state.weights, x, eta * y);
          state.history.push(
            `Large-margin step on <strong>${row.id}</strong>: margin ${U.round(margin, 3)} is below 1, so apply the hinge update.`
          );
        } else {
          state.history.push(
            `Large-margin step on <strong>${row.id}</strong>: margin ${U.round(margin, 3)} is already safe, so only the shrinkage term applies.`
          );
        }
        state.lastUpdate = {
          kind: margin < 1 ? "update" : "shrink",
          row,
          x,
          y,
          score,
          eta,
          margin,
          reg,
          before,
          after: [...state.weights],
        };
      }

      state.sampleIndex += 1;
      state.step += 1;
      state.lastId = row.id;
      if (state.history.length > 8) state.history = state.history.slice(-8);
    }

    function weightTex(weights) {
      return `\\begin{bmatrix} ${weights.map((value) => U.texNum(value, 2)).join(" & ")} \\end{bmatrix}`;
    }

    function render() {
      if (!state || state.weights.length !== featureVector({ x: 0, y: 0 }).length) resetState();

      const ldaParams = forcedAlgorithm === "lda" ? computeLDA(points) : null;
      const params = forcedAlgorithm === "lda" ? ldaParams : state;

      if (forcedAlgorithm === "lda" && !ldaParams) {
        U.renderMetrics(output, [{ label: "Data", value: "Need at least one point per class" }]);
        U.makeChart(plot, { xDomain: [0, 10], yDomain: [0, 10], title: "Add points of both classes" });
        mathNode.innerHTML = "";
        return;
      }

      const accuracy = trainingAccuracy(params);
      const scores = points.map((row) => scorePoint(row, params));
      const avgMargin = points.length ? U.mean(points.map((row, index) => row.label * scores[index])) : 0;
      const norm =
        forcedAlgorithm === "lda"
          ? Math.hypot(ldaParams.w[0], ldaParams.w[1])
          : Math.sqrt(state.weights.reduce((acc, value) => acc + value * value, 0));

      U.renderMetrics(output, [
        { label: "Steps", value: String(state.step) },
        { label: "Accuracy", value: `${Math.round(accuracy * 100)}%` },
        { label: "Weight norm", value: U.round(norm, 3) },
        { label: "Avg. margin", value: U.round(avgMargin, 3) },
      ]);

      const update = state.lastUpdate;
      const formulaMap = {
        lda: ldaParams
          ? [
              {
                title: "Discriminant score",
                description:
                  "LDA derives a linear score from class statistics rather than from iterative updates — change the table and the boundary jumps straight to the new optimum.",
                tex: "\\delta(\\mathbf{x}) = \\mathbf{x}^{\\top}\\Sigma^{-1}(\\mu_1 - \\mu_0) + b",
                derivation: [
                  {
                    tex: `\\mu_1 = (${U.texNum(ldaParams.mu1[0], 2)},\\; ${U.texNum(ldaParams.mu1[1], 2)})`,
                    result: "",
                    note: "Mean of every row currently labelled +1.",
                  },
                  { tex: `\\mu_0 = (${U.texNum(ldaParams.mu0[0], 2)},\\; ${U.texNum(ldaParams.mu0[1], 2)})`, result: "" },
                  {
                    tex: `\\Sigma = \\begin{bmatrix} ${U.texNum(ldaParams.covariance[0][0], 2)} & ${U.texNum(ldaParams.covariance[0][1], 2)} \\\\ ${U.texNum(ldaParams.covariance[1][0], 2)} & ${U.texNum(ldaParams.covariance[1][1], 2)} \\end{bmatrix}`,
                    result: "",
                    note: "One covariance pooled across both classes — that shared shape is what makes the boundary linear.",
                  },
                  {
                    tex: `\\mathbf{w} = \\Sigma^{-1}(\\mu_1 - \\mu_0) = \\begin{bmatrix} ${U.texNum(ldaParams.w[0], 2)} \\\\ ${U.texNum(ldaParams.w[1], 2)} \\end{bmatrix}`,
                    result: "",
                  },
                  { tex: `b = ${U.texNum(ldaParams.b, 3)}`, result: U.round(ldaParams.b, 3) },
                ],
                insight:
                  "<strong>Generative boundary:</strong> LDA assumes both classes are Gaussian with the same covariance. The boundary sits exactly where the two densities cross. Give one class a very different spread in the table and watch the assumption start to hurt.",
              },
            ]
          : [],
        logistic: [
          {
            title: "Probability model",
            description: "Logistic regression maps the linear score into a probability with the sigmoid.",
            tex: "p(y=1 \\mid \\mathbf{x}) = \\sigma(\\mathbf{w}^{\\top}\\mathbf{x}) = \\frac{1}{1 + e^{-\\mathbf{w}^{\\top}\\mathbf{x}}}",
            derivation: [
              { tex: `\\mathbf{w} = ${weightTex(state.weights)}`, result: "" },
              {
                tex: `\\lVert \\mathbf{w} \\rVert = ${U.texNum(norm, 3)}`,
                result: U.round(norm, 3),
                note: "Larger weights produce a sharper transition — the probability surface approaches a hard step.",
              },
            ],
            insight:
              "<strong>Log-odds:</strong> the model asserts the log-odds of class 1 is a linear function of x, which keeps predictions inside (0, 1) automatically.",
          },
          {
            title: "Gradient step",
            description: "Each click nudges the weights along the cross-entropy gradient, scaled by the residual.",
            tex: "\\mathbf{w} \\leftarrow \\mathbf{w} + \\eta\\,(y - p)\\,\\mathbf{x}",
            derivation:
              update && update.probability !== undefined
                ? [
                    {
                      tex: `\\mathbf{x} = ${weightTex(update.x)},\\quad y = ${update.target}`,
                      result: "",
                      note: `Point ${update.row.id} from your table.`,
                    },
                    {
                      tex: `\\mathbf{w}^{\\top}\\mathbf{x} = ${U.texNum(update.score, 3)} \\;\\Rightarrow\\; p = \\sigma(${U.texNum(update.score, 3)}) = ${U.texNum(update.probability, 3)}`,
                      result: U.round(update.probability, 3),
                    },
                    {
                      tex: `y - p = ${update.target} - ${U.texNum(update.probability, 3)} = ${U.texNum(update.target - update.probability, 3)}`,
                      result: U.round(update.target - update.probability, 3),
                    },
                    {
                      tex: `\\mathbf{w} \\leftarrow ${weightTex(update.before)} + ${U.texNum(update.eta, 2)} \\cdot ${U.texNum(update.target - update.probability, 3)} \\cdot ${weightTex(update.x)}`,
                      result: "",
                    },
                    { tex: `\\mathbf{w} = ${weightTex(update.after)}`, result: "" },
                  ]
                : [{ tex: "\\text{press ``Train one step'' to see the arithmetic}", result: "" }],
            insight:
              "<strong>Soft errors:</strong> unlike the perceptron, logistic regression updates on every point — even correctly classified ones — because p is never exactly 0 or 1.",
          },
        ],
        perceptron: [
          {
            title: "Update rule",
            description: "The perceptron only reacts to mistakes. Correctly classified points leave the weights alone.",
            tex: "\\text{if } y(\\mathbf{w}^{\\top}\\mathbf{x}) \\le 0 \\;\\text{ then }\\; \\mathbf{w} \\leftarrow \\mathbf{w} + \\eta\\,y\\,\\mathbf{x}",
            derivation: update
              ? [
                  {
                    tex: `\\mathbf{x} = ${weightTex(update.x)},\\quad y = ${update.y}`,
                    result: "",
                    note: `Point ${update.row.id} from your table.`,
                  },
                  {
                    tex: `y(\\mathbf{w}^{\\top}\\mathbf{x}) = ${update.y} \\times ${U.texNum(update.score, 3)} = ${U.texNum(update.y * update.score, 3)}`,
                    result: U.round(update.y * update.score, 3),
                  },
                  update.kind === "update"
                    ? {
                        tex: `\\mathbf{w} \\leftarrow ${weightTex(update.before)} + ${U.texNum(update.eta, 2)} \\cdot ${update.y} \\cdot ${weightTex(update.x)} = ${weightTex(update.after)}`,
                        result: "",
                        note: "The margin was non-positive, so the point pulled the boundary toward itself.",
                      }
                    : {
                        tex: `\\mathbf{w} \\text{ unchanged} = ${weightTex(update.after)}`,
                        result: "",
                        note: "The margin was already positive, so this point contributes nothing.",
                      },
                ]
              : [{ tex: "\\text{press ``Train one step'' to begin}", result: "" }],
            insight:
              "<strong>Sparse updates:</strong> the perceptron has zero loss on correct predictions. It converges if the data is linearly separable — load the XOR preset and watch it oscillate forever when it is not.",
          },
        ],
        svm: [
          {
            title: "Margin score",
            description: "Large-margin methods are not satisfied by a correct answer; they want y(wᵀx) ≥ 1.",
            tex: "\\text{margin} = y\\,(\\mathbf{w}^{\\top}\\mathbf{x}), \\qquad \\text{width} = \\frac{1}{\\lVert \\mathbf{w} \\rVert}",
            derivation: [
              { tex: `\\mathbf{w} = ${weightTex(state.weights)}`, result: "" },
              {
                tex: `\\lVert \\mathbf{w} \\rVert = ${U.texNum(norm, 3)} \\;\\Rightarrow\\; \\text{width} = ${U.texNum(norm > 1e-6 ? 1 / norm : 0, 3)}`,
                result: U.round(norm, 3),
                note: "Penalising ‖w‖² is exactly what forces a wider margin.",
              },
            ],
            insight:
              featureMapInput.value === "rbf"
                ? "<strong>Kernels:</strong> by measuring distance to fixed centres, a linear boundary in the lifted space becomes a curved boundary back in the original 2D plane. Load the XOR preset — impossible for a line, easy here."
                : "<strong>Support vectors:</strong> points with margin > 1 have zero loss and exert no pull. The boundary is held in place only by the closest points.",
          },
          {
            title: "Hinge update with shrinkage",
            description: "Every step shrinks the weights, and points inside the margin push back.",
            tex: "\\mathbf{w} \\leftarrow (1 - \\eta\\lambda)\\,\\mathbf{w} + \\eta\\,y\\,\\mathbf{x}\\;\\mathbb{1}[\\,y\\mathbf{w}^{\\top}\\mathbf{x} < 1\\,]",
            derivation: update
              ? [
                  {
                    tex: `\\text{margin} = ${U.texNum(update.margin, 3)} \\;${update.margin < 1 ? "<" : "\\ge"}\\; 1`,
                    result: U.round(update.margin, 3),
                    note: `Point ${update.row.id}; η = ${U.round(update.eta, 2)}, λ = ${U.round(update.reg, 3)}.`,
                  },
                  {
                    tex: `\\mathbf{w} = ${weightTex(update.after)}`,
                    result: "",
                    note:
                      update.kind === "update"
                        ? "Inside the margin, so the hinge term fired as well as the shrinkage."
                        : "Outside the margin, so only shrinkage applied.",
                  },
                ]
              : [{ tex: "\\text{press ``Train one step'' to begin}", result: "" }],
            insight:
              "<strong>λ is the bias–variance dial:</strong> raise it and the margin widens at the cost of training accuracy; drop it to zero and the boundary chases every point.",
          },
        ],
      };
      renderFormulaCards(mathNode, formulaMap[forcedAlgorithm]);

      if (forcedAlgorithm === "lda") {
        U.renderSteps(
          stepsNode,
          [
            "<strong>Estimate class means.</strong> Positive and negative classes get separate Gaussian centres.",
            "<strong>Estimate shared covariance.</strong> LDA pools covariance across classes instead of fitting one per class.",
            "<strong>Build the linear discriminant.</strong> The boundary comes from <span class=\"mono\">Σ⁻¹(μ₁−μ₀)</span> plus the prior-adjusted bias.",
          ],
          state.step
        );
      } else {
        U.renderSteps(stepsNode, state.history, state.history.length);
      }

      callout.innerHTML =
        forcedAlgorithm === "lda"
          ? "LDA is derived from a Gaussian generative model with shared covariance — there is nothing to iterate."
          : forcedAlgorithm === "logistic"
          ? "Logistic regression optimises probability fit directly through cross-entropy."
          : forcedAlgorithm === "perceptron"
          ? "The perceptron reacts only to mistakes, which is why its update rule is so sparse."
          : featureMapInput.value === "rbf"
          ? "The radial-basis lift behaves like a simple kernelised feature space."
          : "Large-margin updates focus on the examples sitting too close to the boundary.";

      const chart = U.makeChart(plot, {
        xDomain: [0, 10],
        yDomain: [0, 10],
        title: "Decision regions and training points",
      });
      for (let gx = 0; gx < 18; gx += 1) {
        for (let gy = 0; gy < 12; gy += 1) {
          const x = 0.4 + (gx / 17) * 9.2;
          const y = 0.4 + (gy / 11) * 9.2;
          const score = scorePoint({ x, y }, params);
          const fill =
            score >= 0
              ? U.tint(C().a, 0.08 + Math.min(Math.abs(score) / 6, 0.22))
              : U.tint(C().b, 0.08 + Math.min(Math.abs(score) / 6, 0.22));
          const x1 = chart.xScale(x - 0.28);
          const y1 = chart.yScale(y + 0.42);
          const x2 = chart.xScale(x + 0.28);
          const y2 = chart.yScale(y - 0.42);
          plot.appendChild(
            U.svgEl("rect", {
              x: Math.min(x1, x2),
              y: Math.min(y1, y2),
              width: Math.abs(x2 - x1),
              height: Math.abs(y2 - y1),
              fill,
            })
          );
        }
      }

      const wy = forcedAlgorithm === "lda" ? ldaParams.w[1] : state.weights[2] || 0;
      if ((forcedAlgorithm === "lda" || featureMapInput.value === "linear") && Math.abs(wy) > 1e-6) {
        const b = forcedAlgorithm === "lda" ? ldaParams.b : state.weights[0];
        const w0 = forcedAlgorithm === "lda" ? ldaParams.w[0] : state.weights[1];
        const x1 = 0.5;
        const y1 = -(b + w0 * x1) / wy;
        const x2 = 9.5;
        const y2 = -(b + w0 * x2) / wy;
        plot.appendChild(
          U.svgEl("line", {
            x1: chart.xScale(x1),
            y1: chart.yScale(y1),
            x2: chart.xScale(x2),
            y2: chart.yScale(y2),
            class: "decision-line",
          })
        );
      }

      points.forEach((row, index) => {
        drawScatterPoint(
          plot,
          chart,
          row,
          row.label === 1 ? C().a : C().b,
          state.lastId === row.id ? 10 : 7,
          (target, position) => {
            target.x = Number(U.round(position.x, 2));
            target.y = Number(U.round(position.y, 2));
            editor.setCell(index, "x", target.x);
            editor.setCell(index, "y", target.y);
            scheduleRender();
          }
        );
      });
    }

    const scheduleRender = U.rafThrottle(render);

    stepButton.addEventListener("click", () => {
      takeStep();
      render();
    });
    resetButton.addEventListener("click", () => {
      resetState();
      render();
    });
    if (runButton) {
      runButton.addEventListener("click", () => {
        for (let index = 0; index < 20; index += 1) takeStep();
        render();
      });
    }
    lrInput.addEventListener("input", () => {
      lrValue.textContent = Number(lrInput.value).toFixed(2);
      render();
    });
    if (lambdaInput) {
      lambdaInput.addEventListener("input", () => {
        lambdaValue.textContent = Number(lambdaInput.value).toFixed(2);
        render();
      });
    }
    datasetInput.addEventListener("change", () => {
      points = linearDatasets[datasetInput.value].map((point) => ({ ...point }));
      editor.setRows(points, true);
      resetState();
      render();
    });
    featureMapInput.addEventListener("change", () => {
      resetState();
      render();
    });

    resetState();
    U.onRedraw(render);
    render();
  }

  function mountBayesNet(rootNode) {
    const focus = definition.options.focus;
    mountSection(
      rootNode,
      "Graph structure and conditional independence",
      "Choose a motif, mark observed nodes, and inspect whether the path between the endpoints is active. This is the structural logic behind Bayesian networks.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-group">
              <label for="ci-structure">Structure</label>
              <select id="ci-structure">
                <option value="chain">Chain: A → B → C</option>
                <option value="fork">Fork: A ← B → C</option>
                <option value="collider">Collider: A → B ← C</option>
                <option value="mrf">MRF chain: A — B — C</option>
              </select>
            </div>
            <div class="control-group">
              <label for="ci-observe">Observed nodes</label>
              <select id="ci-observe">
                <option value="none">None</option>
                <option value="b">Observe B</option>
                <option value="a">Observe A</option>
                <option value="c">Observe C</option>
              </select>
            </div>
            <div class="step-controls">
              <button class="button primary" id="ci-evaluate">Evaluate path</button>
            </div>
            <div class="callout" id="ci-callout"></div>
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg id="ci-plot" viewBox="0 0 560 280" aria-label="Conditional independence graph"></svg>
            </div>
            <div class="equation-card" id="ci-math"></div>
            <div class="steps"><ol id="ci-steps"></ol></div>
          </div>
        </div>
      `
    );

    const structureInput = rootNode.querySelector("#ci-structure");
    const observeInput = rootNode.querySelector("#ci-observe");
    const button = rootNode.querySelector("#ci-evaluate");
    const plot = rootNode.querySelector("#ci-plot");
    const mathNode = rootNode.querySelector("#ci-math");
    const stepsNode = rootNode.querySelector("#ci-steps");
    const callout = rootNode.querySelector("#ci-callout");

    function render() {
      const structure = structureInput.value;
      const observed = observeInput.value;
      let active = true;
      let explanation = "";
      if (structure === "chain") {
        active = observed !== "b";
        explanation = observed === "b" ? "Conditioning on the middle node blocks the chain." : "Without observing the middle node, information can flow through the chain.";
      } else if (structure === "fork") {
        active = observed !== "b";
        explanation = observed === "b" ? "Conditioning on the common cause removes dependence between the children." : "The common cause induces dependence between the children.";
      } else if (structure === "collider") {
        active = observed === "b";
        explanation = observed === "b" ? "Conditioning on the collider opens explaining-away dependence." : "An unobserved collider blocks the path.";
      } else {
        active = observed !== "b";
        explanation = observed === "b" ? "In an undirected chain the separator node blocks the path when conditioned on." : "In an undirected chain the path stays active until the separator is fixed.";
      }

      callout.innerHTML = active ? `<strong>A and C are dependent.</strong> ${explanation}` : `<strong>A and C are conditionally independent.</strong> ${explanation}`;
      renderFormulaCards(mathNode, [
        {
          title: focus === "network" ? "Factorization" : "Separation rule",
          description:
            focus === "network"
              ? "A Bayesian network factors the joint distribution into one local conditional per node, given only that node's parents."
              : "Whether information flows between A and C depends on the motif and on what you have conditioned on.",
          tex:
            structure === "chain"
              ? "p(a, b, c) = p(a)\\,p(b \\mid a)\\,p(c \\mid b)"
              : structure === "fork"
              ? "p(a, b, c) = p(b)\\,p(a \\mid b)\\,p(c \\mid b)"
              : structure === "collider"
              ? "p(a, b, c) = p(a)\\,p(c)\\,p(b \\mid a, c)"
              : "p(a, b, c) \\propto \\psi(a, b)\\,\\psi(b, c)",
          insight:
            structure === "collider"
              ? "<strong>Colliders invert the usual rule.</strong> A and C start out independent; conditioning on B <em>creates</em> a dependence between them. This is Berkson's paradox — and the reason conditioning on a common effect introduces selection bias."
              : "<strong>Chains and forks behave the same way.</strong> Conditioning on the middle node blocks the path, making A and C conditionally independent.",
        },
        {
          title: "Current result",
          description: "The structural condition determines whether the path transmits information.",
          tex: `\\text{path}(A, C) = \\text{${active ? "active" : "blocked"}} \\;\\Longrightarrow\\; A ${active ? "\\not\\perp" : "\\perp"} C`,
        },
      ]);

      U.renderSteps(
        stepsNode,
        [
          `<strong>Choose the motif.</strong> Current structure: <strong>${structure}</strong>.`,
          `<strong>Mark observations.</strong> Current observed set: <strong>${observed === "none" ? "∅" : observed.toUpperCase()}</strong>.`,
          `<strong>Apply the separation rule.</strong> ${explanation}`,
        ],
        3
      );

      U.clear(plot);
      const positions = { A: { x: 110, y: 140 }, B: { x: 280, y: 140 }, C: { x: 450, y: 140 } };
      const stroke = active ? C().a : C().neutral;
      const opacity = active ? "0.95" : "0.35";
      function drawLine(from, to, directed) {
        plot.appendChild(U.svgEl("line", { x1: positions[from].x, y1: positions[from].y, x2: positions[to].x, y2: positions[to].y, stroke, "stroke-width": "4", opacity }));
        if (directed) {
          const dx = positions[to].x - positions[from].x;
          const dy = positions[to].y - positions[from].y;
          const len = Math.hypot(dx, dy);
          const ux = dx / len;
          const uy = dy / len;
          const tipX = positions[to].x - ux * 34;
          const tipY = positions[to].y - uy * 34;
          const leftX = tipX - ux * 14 + uy * 8;
          const leftY = tipY - uy * 14 - ux * 8;
          const rightX = tipX - ux * 14 - uy * 8;
          const rightY = tipY - uy * 14 + ux * 8;
          plot.appendChild(U.svgEl("polygon", { points: `${tipX},${tipY} ${leftX},${leftY} ${rightX},${rightY}`, fill: stroke, opacity }));
        }
      }
      if (structure === "chain") {
        drawLine("A", "B", true);
        drawLine("B", "C", true);
      } else if (structure === "fork") {
        drawLine("B", "A", true);
        drawLine("B", "C", true);
      } else if (structure === "collider") {
        drawLine("A", "B", true);
        drawLine("C", "B", true);
      } else {
        drawLine("A", "B", false);
        drawLine("B", "C", false);
      }
      Object.entries(positions).forEach(([name, pos]) => {
        const observedNode = observed.toUpperCase() === name;
        plot.appendChild(U.svgEl("circle", { cx: pos.x, cy: pos.y, r: 32, fill: observedNode ? C().b : C().plotBg, stroke: observedNode ? C().b : C().ink, "stroke-width": "3" }));
        plot.appendChild(U.svgEl("text", { x: pos.x, y: pos.y + 6, "text-anchor": "middle", class: "svg-title" })).textContent = name;
      });
    }

    button.addEventListener("click", render);
    [structureInput, observeInput].forEach((input) => input.addEventListener("input", render));
    U.onRedraw(render);
    render();
  }

  function mountMRF(rootNode) {
    mountSection(
      rootNode,
      "Local conditional in a Markov random field",
      "Set the neighbor states and the compatibility strength. The center node’s conditional distribution is computed from local factors only.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-grid">
              <div class="control-group">
                <label for="mrf-left">Left state X₁</label>
                <select id="mrf-left">
                  <option value="0">0</option>
                  <option value="1">1</option>
                </select>
              </div>
              <div class="control-group">
                <label for="mrf-right">Right state X₃</label>
                <select id="mrf-right">
                  <option value="0">0</option>
                  <option value="1">1</option>
                </select>
              </div>
              <div class="control-group">
                <label for="mrf-compatibility">Same-state compatibility</label>
                <input id="mrf-compatibility" type="number" min="1.1" max="5" step="0.1" value="2.3" />
              </div>
              <div class="control-group">
                <label for="mrf-bias">Unary bias toward state 1</label>
                <input id="mrf-bias" type="number" min="-2" max="2" step="0.1" value="0.4" />
              </div>
            </div>
            <div class="step-controls">
              <button class="button primary" id="mrf-step">Next step</button>
              <button class="button secondary" id="mrf-reset">Reset steps</button>
            </div>
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg id="mrf-plot" viewBox="0 0 560 280" aria-label="MRF plot"></svg>
            </div>
            <div class="readout"><div class="output-grid" id="mrf-output"></div></div>
            <div class="equation-card" id="mrf-math"></div>
            <div class="steps"><ol id="mrf-steps"></ol></div>
          </div>
        </div>
      `
    );

    const leftInput = rootNode.querySelector("#mrf-left");
    const rightInput = rootNode.querySelector("#mrf-right");
    const compatInput = rootNode.querySelector("#mrf-compatibility");
    const biasInput = rootNode.querySelector("#mrf-bias");
    const stepButton = rootNode.querySelector("#mrf-step");
    const resetButton = rootNode.querySelector("#mrf-reset");
    const plot = rootNode.querySelector("#mrf-plot");
    const output = rootNode.querySelector("#mrf-output");
    const mathNode = rootNode.querySelector("#mrf-math");
    const stepsNode = rootNode.querySelector("#mrf-steps");
    let revealed = 0;

    function render() {
      const left = Number(leftInput.value);
      const right = Number(rightInput.value);
      const same = Number(compatInput.value) || 2;
      const diff = 1;
      const bias = Number(biasInput.value) || 0;
      const unary0 = Math.exp(-bias);
      const unary1 = Math.exp(bias);
      const score0 = unary0 * (left === 0 ? same : diff) * (right === 0 ? same : diff);
      const score1 = unary1 * (left === 1 ? same : diff) * (right === 1 ? same : diff);
      const probs = normalize([score0, score1]);

      U.renderMetrics(output, [
        { label: "Unnorm score x₂=0", value: U.round(score0, 3) },
        { label: "Unnorm score x₂=1", value: U.round(score1, 3) },
        { label: "P(X₂=0)", value: U.round(probs[0], 3) },
        { label: "P(X₂=1)", value: U.round(probs[1], 3) },
      ]);
      renderFormulaCards(mathNode, [
        { title: "MRF distribution", description: "Undirected models multiply local compatibility factors, then divide by Z to make the result a probability.", tex: "p(\\mathbf{x}) = \\frac{1}{Z}\\prod_{c \\in \\mathcal{C}} \\psi_c(\\mathbf{x}_c), \\qquad Z = \\sum_{\\mathbf{x}} \\prod_{c} \\psi_c(\\mathbf{x}_c)" },
        {
          title: "Local conditional",
          description: "Only neighboring factors appear in the center-node conditional.",
          tex: "p(x_2 \\mid x_1, x_3) \\propto \\phi(x_2)\\,\\psi(x_1, x_2)\\,\\psi(x_2, x_3)",
        },
        {
          title: "Current normalized result",
          description: "The two local scores are normalized into a probability.",
          tex: `p(x_2 = 1 \\mid x_1, x_3) = \\frac{${U.texNum(score1, 3)}}{${U.texNum(score0 + score1, 3)}} = ${U.texNum(probs[1], 3)}`,
        },
      ]);
      U.renderSteps(
        stepsNode,
        [
          "<strong>Read the neighbor states.</strong> The center node only needs the current states of its Markov blanket.",
          `<strong>Compute local scores.</strong> For x₂=0 use score <strong>${U.round(score0, 3)}</strong>; for x₂=1 use score <strong>${U.round(score1, 3)}</strong>.`,
          "<strong>Normalize.</strong> Divide each score by the total score so the two values sum to 1.",
          `<strong>Interpret.</strong> The center node currently prefers state <strong>${probs[1] >= probs[0] ? 1 : 0}</strong>.`,
        ],
        revealed
      );

      U.clear(plot);
      const positions = [{ x: 120, y: 140 }, { x: 280, y: 140 }, { x: 440, y: 140 }];
      plot.appendChild(U.svgEl("line", { x1: 152, y1: 140, x2: 248, y2: 140, class: "cluster-line" }));
      plot.appendChild(U.svgEl("line", { x1: 312, y1: 140, x2: 408, y2: 140, class: "cluster-line" }));
      [left, probs[1] >= probs[0] ? 1 : 0, right].forEach((value, index) => {
        plot.appendChild(U.svgEl("circle", { cx: positions[index].x, cy: positions[index].y, r: 34, fill: index === 1 ? C().plotBg : value ? C().a : C().b, stroke: C().ink, "stroke-width": "3" }));
        plot.appendChild(U.svgEl("text", { x: positions[index].x, y: positions[index].y - 3, "text-anchor": "middle", class: "svg-title" })).textContent = `X${index + 1}`;
        plot.appendChild(U.svgEl("text", { x: positions[index].x, y: positions[index].y + 18, "text-anchor": "middle", class: "svg-label" })).textContent = index === 1 ? `P(1)=${U.round(probs[1], 2)}` : String(value);
      });
    }

    stepButton.addEventListener("click", () => {
      revealed = Math.min(revealed + 1, 4);
      render();
    });
    resetButton.addEventListener("click", () => {
      revealed = 0;
      render();
    });
    [leftInput, rightInput, compatInput, biasInput].forEach((input) =>
      input.addEventListener("input", () => {
        revealed = 0;
        render();
      })
    );
    U.onRedraw(render);
    render();
  }

  function mountBeliefPropagation(rootNode) {
    mountSection(
      rootNode,
      "Belief propagation on a three-node chain",
      "Pass messages one edge at a time and watch local evidence propagate into marginals.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-grid">
              <div class="control-group">
                <label for="bp-left-evidence">Left evidence P(x=1)</label>
                <input id="bp-left-evidence" type="number" min="0.05" max="0.95" step="0.05" value="0.8" />
              </div>
              <div class="control-group">
                <label for="bp-right-evidence">Right evidence P(x=1)</label>
                <input id="bp-right-evidence" type="number" min="0.05" max="0.95" step="0.05" value="0.3" />
              </div>
              <div class="control-group">
                <label for="bp-compatibility">Pairwise compatibility</label>
                <input id="bp-compatibility" type="number" min="0.55" max="0.95" step="0.05" value="0.85" />
              </div>
            </div>
            <div class="step-controls">
              <button class="button primary" id="bp-step">Pass next message</button>
              <button class="button secondary" id="bp-reset">Reset messages</button>
            </div>
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg id="bp-plot" viewBox="0 0 560 280" aria-label="Belief propagation plot"></svg>
            </div>
            <div class="readout"><div class="output-grid" id="bp-output"></div></div>
            <div class="equation-card" id="bp-math"></div>
            <div class="steps"><ol id="bp-steps"></ol></div>
          </div>
        </div>
      `
    );

    const leftEvidenceInput = rootNode.querySelector("#bp-left-evidence");
    const rightEvidenceInput = rootNode.querySelector("#bp-right-evidence");
    const compatibilityInput = rootNode.querySelector("#bp-compatibility");
    const stepButton = rootNode.querySelector("#bp-step");
    const resetButton = rootNode.querySelector("#bp-reset");
    const plot = rootNode.querySelector("#bp-plot");
    const output = rootNode.querySelector("#bp-output");
    const mathNode = rootNode.querySelector("#bp-math");
    const stepsNode = rootNode.querySelector("#bp-steps");
    let revealed = 0;

    function render() {
      const leftEvidence = Number(leftEvidenceInput.value) || 0.8;
      const rightEvidence = Number(rightEvidenceInput.value) || 0.3;
      const same = Number(compatibilityInput.value) || 0.85;
      const diff = 1 - same;
      const phi1 = [1 - leftEvidence, leftEvidence];
      const phi2 = [0.5, 0.5];
      const phi3 = [1 - rightEvidence, rightEvidence];
      const psi = [
        [same, diff],
        [diff, same],
      ];
      const messages = [];
      const m12 = normalize([phi1[0] * psi[0][0] + phi1[1] * psi[1][0], phi1[0] * psi[0][1] + phi1[1] * psi[1][1]]);
      messages.push({ name: "m₁→₂", value: m12, text: `Send <strong>m₁→₂ = [${U.round(m12[0], 3)}, ${U.round(m12[1], 3)}]</strong>.` });
      const m32 = normalize([phi3[0] * psi[0][0] + phi3[1] * psi[1][0], phi3[0] * psi[0][1] + phi3[1] * psi[1][1]]);
      messages.push({ name: "m₃→₂", value: m32, text: `Send <strong>m₃→₂ = [${U.round(m32[0], 3)}, ${U.round(m32[1], 3)}]</strong>.` });
      const m21 = normalize([psi[0][0] * phi2[0] * m32[0] + psi[0][1] * phi2[1] * m32[1], psi[1][0] * phi2[0] * m32[0] + psi[1][1] * phi2[1] * m32[1]]);
      messages.push({ name: "m₂→₁", value: m21, text: `Return <strong>m₂→₁ = [${U.round(m21[0], 3)}, ${U.round(m21[1], 3)}]</strong>.` });
      const m23 = normalize([psi[0][0] * phi2[0] * m12[0] + psi[0][1] * phi2[1] * m12[1], psi[1][0] * phi2[0] * m12[0] + psi[1][1] * phi2[1] * m12[1]]);
      messages.push({ name: "m₂→₃", value: m23, text: `Return <strong>m₂→₃ = [${U.round(m23[0], 3)}, ${U.round(m23[1], 3)}]</strong>.` });
      const belief1 = normalize([phi1[0] * (revealed >= 3 ? m21[0] : 1), phi1[1] * (revealed >= 3 ? m21[1] : 1)]);
      const belief2 = normalize([phi2[0] * (revealed >= 1 ? m12[0] : 1) * (revealed >= 2 ? m32[0] : 1), phi2[1] * (revealed >= 1 ? m12[1] : 1) * (revealed >= 2 ? m32[1] : 1)]);
      const belief3 = normalize([phi3[0] * (revealed >= 4 ? m23[0] : 1), phi3[1] * (revealed >= 4 ? m23[1] : 1)]);

      U.renderMetrics(output, [
        { label: "P(X1 = 1)", value: U.round(belief1[1], 3) },
        { label: "P(X2 = 1)", value: U.round(belief2[1], 3) },
        { label: "P(X3 = 1)", value: U.round(belief3[1], 3) },
      ]);
      renderFormulaCards(mathNode, [
        { title: "Message passing", description: "A message summarises everything one side of the graph has to say about a neighbour — note it excludes whatever that neighbour already told you, which is what stops evidence being double counted.", tex: "m_{i \\to j}(x_j) = \\sum_{x_i} \\phi_i(x_i)\\,\\psi_{ij}(x_i, x_j) \\prod_{k \\in N(i) \\setminus j} m_{k \\to i}(x_i)" },
        { title: "Belief", description: "A node's marginal is its own local evidence multiplied by every incoming message.", tex: "b_i(x_i) \\propto \\phi_i(x_i) \\prod_{k \\in N(i)} m_{k \\to i}(x_i)" },
      ]);
      U.renderSteps(stepsNode, messages.map((message) => message.text), revealed);

      U.clear(plot);
      const positions = [{ x: 110, y: 140 }, { x: 280, y: 140 }, { x: 450, y: 140 }];
      plot.appendChild(U.svgEl("text", { x: 280, y: 28, class: "svg-title", "text-anchor": "middle" })).textContent = "Message passing schedule";
      [[0, 1], [1, 2]].forEach(([from, to]) => {
        plot.appendChild(U.svgEl("line", { x1: positions[from].x + 32, y1: positions[from].y, x2: positions[to].x - 32, y2: positions[to].y, stroke: C().ink, "stroke-width": "3", opacity: "0.5" }));
      });
      [belief1, belief2, belief3].forEach((belief, index) => {
        plot.appendChild(U.svgEl("circle", { cx: positions[index].x, cy: positions[index].y, r: 34, fill: C().plotBg, stroke: C().ink, "stroke-width": "3" }));
        plot.appendChild(U.svgEl("text", { x: positions[index].x, y: positions[index].y - 4, class: "svg-title", "text-anchor": "middle" })).textContent = `X${index + 1}`;
        plot.appendChild(U.svgEl("text", { x: positions[index].x, y: positions[index].y + 18, class: "svg-label", "text-anchor": "middle" })).textContent = `P(1)=${U.round(belief[1], 2)}`;
      });
    }

    stepButton.addEventListener("click", () => {
      revealed = Math.min(revealed + 1, 4);
      render();
    });
    resetButton.addEventListener("click", () => {
      revealed = 0;
      render();
    });
    [leftEvidenceInput, rightEvidenceInput, compatibilityInput].forEach((input) =>
      input.addEventListener("input", () => {
        revealed = 0;
        render();
      })
    );
    U.onRedraw(render);
    render();
  }

  function mountMarkov(rootNode) {
    mountSection(
      rootNode,
      "Visible-state Markov chain",
      "Propagate a two-state weather distribution forward in time. Edit the transition matrix directly — every row is a conditional distribution and is re-normalised for you — then watch how quickly the chain forgets where it started.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-grid">
              <div class="control-group">
                <label for="markov-start">Initial P(Rain)</label>
                <div class="range-row">
                  <input id="markov-start" type="range" min="0" max="1" step="0.05" value="0.5" />
                  <span class="range-value" id="markov-start-value">0.50</span>
                </div>
              </div>
            </div>
            <div class="step-controls">
              <button class="button primary" id="markov-step">Next time step</button>
              <button class="button secondary" id="markov-run">Run 15 steps</button>
              <button class="button secondary" id="markov-reset">Reset chain</button>
            </div>
            ${editorSlot("markov-matrix")}
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg id="markov-plot" viewBox="0 0 560 280" aria-label="Markov chain plot"></svg>
              <div class="legend">
                <span><i style="background:#1a7a6e"></i> P(Rain)</span>
                <span><i style="background:#c96b28"></i> P(Sun)</span>
                <span><i style="background:#b03030"></i> Stationary distribution</span>
              </div>
            </div>
            <div class="readout"><div class="output-grid" id="markov-output"></div></div>
            <div class="equation-card" id="markov-math"></div>
            <div class="steps"><ol id="markov-steps"></ol></div>
          </div>
        </div>
      `
    );

    const startInput = rootNode.querySelector("#markov-start");
    const startValue = rootNode.querySelector("#markov-start-value");
    const stepButton = rootNode.querySelector("#markov-step");
    const runButton = rootNode.querySelector("#markov-run");
    const resetButton = rootNode.querySelector("#markov-reset");
    const plot = rootNode.querySelector("#markov-plot");
    const output = rootNode.querySelector("#markov-output");
    const mathNode = rootNode.querySelector("#markov-math");
    const stepsNode = rootNode.querySelector("#markov-steps");

    let transition = [
      [0.8, 0.2],
      [0.3, 0.7],
    ];
    let history = [];

    const matrixEditor = U.matrixEditor(rootNode.querySelector("#markov-matrix"), {
      title: "Transition matrix T",
      hint: "Row i, column j is P(next = j | now = i). Set the off-diagonals to 0 and the chain freezes; set a row to be identical to the other and the chain forgets its state in a single step.",
      rowLabels: ["from Rain", "from Sun"],
      colLabels: ["to Rain", "to Sun"],
      values: transition,
      rowStochastic: true,
      step: 0.05,
      min: 0,
      max: 1,
      onChange: (normalizedMatrix) => {
        transition = normalizedMatrix;
        reset();
        render();
      },
    });

    /* Solve for the stationary distribution of a 2-state chain in closed
       form: pi = pi T with pi summing to 1. */
    function stationary() {
      const a = transition[0][1]; // Rain -> Sun
      const b = transition[1][0]; // Sun  -> Rain
      if (a + b === 0) return null;
      return [b / (a + b), a / (a + b)];
    }

    function reset() {
      const rain = Number(startInput.value);
      history = [[rain, 1 - rain]];
    }

    function advance() {
      const previous = history[history.length - 1];
      history.push([
        previous[0] * transition[0][0] + previous[1] * transition[1][0],
        previous[0] * transition[0][1] + previous[1] * transition[1][1],
      ]);
    }

    function render() {
      startValue.textContent = Number(startInput.value).toFixed(2);
      const current = history[history.length - 1];
      const previous = history.length > 1 ? history[history.length - 2] : null;
      const pi = stationary();

      U.renderMetrics(output, [
        { label: "Time steps", value: String(history.length - 1) },
        { label: "P(Rain)", value: U.round(current[0], 4) },
        { label: "P(Sun)", value: U.round(current[1], 4) },
        { label: "Stationary P(Rain)", value: pi ? U.round(pi[0], 4) : "n/a" },
      ]);

      renderFormulaCards(mathNode, [
        {
          title: "Transition matrix",
          description:
            "The Markov property in one line: where you go next depends on where you are now, and on nothing that happened before that.",
          tex: `T = \\begin{bmatrix} ${U.texNum(transition[0][0], 2)} & ${U.texNum(transition[0][1], 2)} \\\\ ${U.texNum(transition[1][0], 2)} & ${U.texNum(transition[1][1], 2)} \\end{bmatrix}`,
          derivation: [
            {
              tex: `\\sum_j T_{1j} = ${U.texNum(transition[0][0] + transition[0][1], 2)}, \\quad \\sum_j T_{2j} = ${U.texNum(transition[1][0] + transition[1][1], 2)}`,
              result: "",
              note: "Every row must sum to 1 — each row is a full conditional distribution over where you land next.",
            },
          ],
          insight:
            "<strong>Memorylessness is a strong claim.</strong> Real weather depends on more than yesterday. When one step of history is not enough, you either expand the state or move to a higher-order model.",
        },
        {
          title: "Distribution update",
          description: "Propagating a belief forward is a single vector-matrix product, applied once per time step.",
          tex: "\\mathbf{p}_t = \\mathbf{p}_{t-1} T = \\mathbf{p}_0 T^{\\,t}",
          derivation: previous
            ? [
                {
                  tex: `\\mathbf{p}_{t-1} = \\begin{bmatrix} ${U.texNum(previous[0], 3)} & ${U.texNum(previous[1], 3)} \\end{bmatrix}`,
                  result: "",
                },
                {
                  tex: `p_t(\\text{Rain}) = ${U.texNum(previous[0], 3)}(${U.texNum(transition[0][0], 2)}) + ${U.texNum(previous[1], 3)}(${U.texNum(transition[1][0], 2)})`,
                  result: U.round(current[0], 4),
                },
                {
                  tex: `p_t(\\text{Sun}) = ${U.texNum(previous[0], 3)}(${U.texNum(transition[0][1], 2)}) + ${U.texNum(previous[1], 3)}(${U.texNum(transition[1][1], 2)})`,
                  result: U.round(current[1], 4),
                  note: "Each entry is a weighted average of the ways you could have arrived in that state.",
                },
              ]
            : [{ tex: "\\text{press ``Next time step'' to propagate the distribution}", result: "" }],
          insight:
            "<strong>Nothing else is happening here.</strong> Every forward algorithm in this category — filtering, Viterbi, Baum–Welch — is this same product with extra bookkeeping attached.",
        },
        {
          title: "Stationary distribution",
          description:
            "The distribution that stops changing. Run the chain long enough from any starting point and it converges here.",
          tex: "\\boldsymbol{\\pi} T = \\boldsymbol{\\pi}, \\qquad \\sum_i \\pi_i = 1",
          derivation: pi
            ? [
                {
                  tex: `\\pi_{\\text{Rain}} = \\frac{T_{21}}{T_{12} + T_{21}} = \\frac{${U.texNum(transition[1][0], 3)}}{${U.texNum(transition[0][1] + transition[1][0], 3)}}`,
                  result: U.round(pi[0], 4),
                },
                {
                  tex: `\\lVert \\mathbf{p}_t - \\boldsymbol{\\pi} \\rVert = ${U.texNum(Math.abs(current[0] - pi[0]) + Math.abs(current[1] - pi[1]), 5)}`,
                  result: U.round(Math.abs(current[0] - pi[0]) + Math.abs(current[1] - pi[1]), 5),
                  note: "Distance from the stationary distribution — keep stepping and watch it shrink geometrically.",
                },
              ]
            : [{ tex: "\\text{this chain has no unique stationary distribution}", result: "" }],
          insight:
            "<strong>The starting point washes out.</strong> Drag the initial probability anywhere and run 15 steps — you land in the same place. That forgetting rate is set by the second eigenvalue of T.",
        },
      ]);

      U.renderSteps(
        stepsNode,
        history.map(
          (distribution, index) =>
            `<strong>t = ${index}.</strong> Distribution = [Rain ${U.round(distribution[0], 4)}, Sun ${U.round(distribution[1], 4)}].`
        ),
        history.length
      );

      const chart = U.makeChart(plot, {
        xDomain: [0, Math.max(history.length - 1, 1)],
        yDomain: [0, 1],
        title: "State probability over time",
      });

      if (pi) {
        plot.appendChild(
          U.svgEl("line", {
            x1: chart.xScale(0),
            y1: chart.yScale(pi[0]),
            x2: chart.xScale(Math.max(history.length - 1, 1)),
            y2: chart.yScale(pi[0]),
            stroke: C().danger,
            "stroke-width": "2",
            "stroke-dasharray": "6 6",
            opacity: "0.75",
          })
        );
      }

      const rainPoints = history.map((distribution, index) => ({ x: index, y: distribution[0] }));
      const sunPoints = history.map((distribution, index) => ({ x: index, y: distribution[1] }));
      plot.appendChild(
        U.svgEl("path", { d: U.pathFromPoints(sunPoints, chart.xScale, chart.yScale), class: "curve-secondary" })
      );
      plot.appendChild(
        U.svgEl("path", { d: U.pathFromPoints(rainPoints, chart.xScale, chart.yScale), class: "curve-primary" })
      );
      rainPoints.forEach((point) => {
        plot.appendChild(
          U.svgEl("circle", { cx: chart.xScale(point.x), cy: chart.yScale(point.y), r: 5, fill: C().a })
        );
      });
    }

    stepButton.addEventListener("click", () => {
      advance();
      render();
    });
    runButton.addEventListener("click", () => {
      for (let index = 0; index < 15; index += 1) advance();
      render();
    });
    resetButton.addEventListener("click", () => {
      reset();
      render();
    });
    startInput.addEventListener("input", () => {
      reset();
      render();
    });

    reset();
    U.onRedraw(render);
    render();
  }

  function parseSequence(raw) {
    return raw
      .split(/[\s,]+/)
      .map((token) => token.trim().toUpperCase())
      .filter((token) => token === "U" || token === "N");
  }

  /* transition: 2x2 row-stochastic. emission: 2x2 where row = state,
     column = observation (U, N). Both come straight from matrix editors
     so a learner can build an asymmetric model, not just tune one dial. */
  function emissionLookup(emission) {
    return { U: [emission[0][0], emission[1][0]], N: [emission[0][1], emission[1][1]] };
  }

  function buildHmmModel(sequence, transition, emission) {
    const emit = emissionLookup(emission);
    const states = ["Rain", "Sun"];
    const alpha = [];
    const delta = [];
    const psi = [];
    if (!sequence.length) {
      return { transition, emit, states, alpha, delta, psi, path: [] };
    }
    alpha[0] = normalize([0.5 * emit[sequence[0]][0], 0.5 * emit[sequence[0]][1]]);
    delta[0] = normalize([0.5 * emit[sequence[0]][0], 0.5 * emit[sequence[0]][1]]);
    psi[0] = [0, 0];
    for (let time = 1; time < sequence.length; time += 1) {
      const obs = sequence[time];
      alpha[time] = normalize([
        emit[obs][0] * (alpha[time - 1][0] * transition[0][0] + alpha[time - 1][1] * transition[1][0]),
        emit[obs][1] * (alpha[time - 1][0] * transition[0][1] + alpha[time - 1][1] * transition[1][1]),
      ]);
      const rawDelta = [0, 0];
      const back = [0, 0];
      for (let j = 0; j < 2; j += 1) {
        const candidates = [
          delta[time - 1][0] * transition[0][j] * emit[obs][j],
          delta[time - 1][1] * transition[1][j] * emit[obs][j],
        ];
        back[j] = candidates[0] >= candidates[1] ? 0 : 1;
        rawDelta[j] = Math.max(candidates[0], candidates[1]);
      }
      delta[time] = normalize(rawDelta);
      psi[time] = back;
    }
    const path = Array(sequence.length).fill(0);
    path[sequence.length - 1] = delta[sequence.length - 1][0] >= delta[sequence.length - 1][1] ? 0 : 1;
    for (let time = sequence.length - 1; time > 0; time -= 1) {
      path[time - 1] = psi[time][path[time]];
    }
    return { transition, emit, states, alpha, delta, psi, path };
  }

  function forwardBackward(sequence, transition, emission) {
    const emit = emissionLookup(emission);
    const alpha = [];
    const beta = Array.from({ length: sequence.length }, () => [1, 1]);
    if (!sequence.length) {
      return { gamma: [], xi: [] };
    }
    alpha[0] = normalize([0.5 * emit[sequence[0]][0], 0.5 * emit[sequence[0]][1]]);
    for (let time = 1; time < sequence.length; time += 1) {
      const obs = sequence[time];
      alpha[time] = normalize([
        emit[obs][0] * (alpha[time - 1][0] * transition[0][0] + alpha[time - 1][1] * transition[1][0]),
        emit[obs][1] * (alpha[time - 1][0] * transition[0][1] + alpha[time - 1][1] * transition[1][1]),
      ]);
    }
    for (let time = sequence.length - 2; time >= 0; time -= 1) {
      beta[time] = normalize([
        transition[0][0] * emit[sequence[time + 1]][0] * beta[time + 1][0] +
          transition[0][1] * emit[sequence[time + 1]][1] * beta[time + 1][1],
        transition[1][0] * emit[sequence[time + 1]][0] * beta[time + 1][0] +
          transition[1][1] * emit[sequence[time + 1]][1] * beta[time + 1][1],
      ]);
    }
    const gamma = alpha.map((alphas, time) => normalize([alphas[0] * beta[time][0], alphas[1] * beta[time][1]]));
    const xi = [];
    for (let time = 0; time < sequence.length - 1; time += 1) {
      const obs = sequence[time + 1];
      const raw = [
        [
          alpha[time][0] * transition[0][0] * emit[obs][0] * beta[time + 1][0],
          alpha[time][0] * transition[0][1] * emit[obs][1] * beta[time + 1][1],
        ],
        [
          alpha[time][1] * transition[1][0] * emit[obs][0] * beta[time + 1][0],
          alpha[time][1] * transition[1][1] * emit[obs][1] * beta[time + 1][1],
        ],
      ];
      const total = raw.flat().reduce((acc, value) => acc + value, 0) || 1;
      xi.push(raw.map((row) => row.map((value) => value / total)));
    }
    return { gamma, xi };
  }

  function mountHmm(rootNode) {
    const focus = definition.options.focus;
    mountSection(
      rootNode,
      "Hidden Markov model explorer",
      "The state is hidden; only the observations are visible. Type your own observation sequence, edit the transition and emission matrices cell by cell, then step through filtering and Viterbi decoding.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-group">
              <label for="hmm-sequence">Observation sequence (U = umbrella seen, N = none)</label>
              <input id="hmm-sequence" type="text" value="U,U,N,U,N" />
              <p class="caption">Try a long run of the same observation and watch the belief saturate, then flip one symbol in the middle and see how far the correction propagates.</p>
            </div>
            <div class="step-controls">
              <button class="button primary" id="hmm-step">${focus === "decoding" ? "Next Viterbi / forward step" : "Next forward step"}</button>
              <button class="button secondary" id="hmm-all">Show all steps</button>
              <button class="button secondary" id="hmm-reset">Reset sequence</button>
              <button class="button ghost" id="hmm-learn">Run one learning update</button>
            </div>
            <div class="callout" id="hmm-callout"></div>
            ${editorSlot("hmm-transition-matrix")}
            ${editorSlot("hmm-emission-matrix")}
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg id="hmm-plot" viewBox="0 0 560 280" aria-label="Hidden Markov model plot"></svg>
              <div class="legend">
                <span><i style="background:#1a7a6e"></i> Filtered P(Rain)</span>
                <span><i style="background:#c96b28"></i> Viterbi path state</span>
              </div>
            </div>
            <div class="readout"><div class="output-grid" id="hmm-output"></div></div>
            <div class="equation-card" id="hmm-math"></div>
            <div class="steps"><ol id="hmm-steps"></ol></div>
            <div class="table-panel">
              <table>
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Observation</th>
                    <th>P(Rain)</th>
                    <th>P(Sun)</th>
                    <th>Best state</th>
                  </tr>
                </thead>
                <tbody id="hmm-table"></tbody>
              </table>
            </div>
          </div>
        </div>
      `
    );

    const sequenceInput = rootNode.querySelector("#hmm-sequence");
    const stepButton = rootNode.querySelector("#hmm-step");
    const allButton = rootNode.querySelector("#hmm-all");
    const resetButton = rootNode.querySelector("#hmm-reset");
    const learnButton = rootNode.querySelector("#hmm-learn");
    const callout = rootNode.querySelector("#hmm-callout");
    const plot = rootNode.querySelector("#hmm-plot");
    const output = rootNode.querySelector("#hmm-output");
    const mathNode = rootNode.querySelector("#hmm-math");
    const stepsNode = rootNode.querySelector("#hmm-steps");
    const table = rootNode.querySelector("#hmm-table");

    let transition = [
      [0.8, 0.2],
      [0.2, 0.8],
    ];
    let emission = [
      [0.75, 0.25],
      [0.25, 0.75],
    ];
    let revealed = 0;

    const transitionEditor = U.matrixEditor(rootNode.querySelector("#hmm-transition-matrix"), {
      title: "Transition matrix A — P(next state | current state)",
      hint: "How persistent is the weather? High diagonal values mean long runs of the same hidden state.",
      rowLabels: ["from Rain", "from Sun"],
      colLabels: ["to Rain", "to Sun"],
      values: transition,
      rowStochastic: true,
      step: 0.05,
      min: 0,
      max: 1,
      onChange: (matrix) => {
        transition = matrix;
        revealed = 0;
        render();
      },
    });

    const emissionEditor = U.matrixEditor(rootNode.querySelector("#hmm-emission-matrix"), {
      title: "Emission matrix B — P(observation | state)",
      hint: "How reliable is the sensor? Set both rows nearly equal and the observations stop carrying information about the hidden state.",
      rowLabels: ["Rain emits", "Sun emits"],
      colLabels: ["U (umbrella)", "N (none)"],
      values: emission,
      rowStochastic: true,
      step: 0.05,
      min: 0,
      max: 1,
      onChange: (matrix) => {
        emission = matrix;
        revealed = 0;
        render();
      },
    });

    function render(statusMessage) {
      const sequence = parseSequence(sequenceInput.value);
      const model = buildHmmModel(sequence, transition, emission);
      const shown = Math.min(revealed, sequence.length);
      const currentRain = shown ? model.alpha[shown - 1][0] : 0.5;
      const currentBest = shown ? model.states[model.path[shown - 1]] : "Start";

      callout.innerHTML =
        statusMessage ||
        (focus === "decoding"
          ? "This page emphasises Viterbi decoding alongside forward filtering — note where the two disagree."
          : focus === "learning"
          ? "This page emphasises parameter learning from soft hidden-state counts."
          : "This page emphasises the combined HMM picture: filtering, decoding, and learning.");

      U.renderMetrics(output, [
        { label: "Steps shown", value: `${shown} / ${sequence.length}` },
        { label: "Current P(Rain)", value: U.round(currentRain, 4) },
        { label: "Current best state", value: currentBest },
        { label: "Sequence length", value: String(sequence.length) },
      ]);

      /* Show the forward recursion with the actual numbers for the most
         recently revealed step. */
      const emit = emissionLookup(emission);
      const forwardDerivation = [];
      if (shown >= 2) {
        const t = shown - 1;
        const obs = sequence[t];
        const prev = model.alpha[t - 1];
        const predictRain = prev[0] * transition[0][0] + prev[1] * transition[1][0];
        const predictSun = prev[0] * transition[0][1] + prev[1] * transition[1][1];
        forwardDerivation.push({
          tex: `\\alpha_{${t}} = \\begin{bmatrix} ${U.texNum(prev[0], 3)} & ${U.texNum(prev[1], 3)} \\end{bmatrix}`,
          result: "",
          note: `Belief after ${t} observation${t !== 1 ? "s" : ""}.`,
        });
        forwardDerivation.push({
          tex: `\\text{predict: } \\begin{bmatrix} ${U.texNum(predictRain, 3)} & ${U.texNum(predictSun, 3)} \\end{bmatrix}`,
          result: "",
          note: "Push the belief through A, before looking at the new observation.",
        });
        forwardDerivation.push({
          tex: `\\text{correct by } b(o_{${t + 1}} = \\text{${obs}}) = \\begin{bmatrix} ${U.texNum(emit[obs][0], 2)} & ${U.texNum(emit[obs][1], 2)} \\end{bmatrix}`,
          result: "",
        });
        forwardDerivation.push({
          tex: `\\alpha_{${t + 1}} \\propto \\begin{bmatrix} ${U.texNum(predictRain * emit[obs][0], 4)} & ${U.texNum(predictSun * emit[obs][1], 4)} \\end{bmatrix} \\;\\to\\; \\begin{bmatrix} ${U.texNum(model.alpha[t][0], 3)} & ${U.texNum(model.alpha[t][1], 3)} \\end{bmatrix}`,
          result: U.round(model.alpha[t][0], 4),
          note: "Multiply elementwise, then normalise — that is the entire forward algorithm.",
        });
      } else if (shown === 1) {
        forwardDerivation.push({
          tex: `\\alpha_1 \\propto \\pi \\odot b(o_1 = \\text{${sequence[0]}}) = \\begin{bmatrix} ${U.texNum(model.alpha[0][0], 3)} & ${U.texNum(model.alpha[0][1], 3)} \\end{bmatrix}`,
          result: U.round(model.alpha[0][0], 4),
          note: "The first step starts from a uniform prior over the hidden state.",
        });
      } else {
        forwardDerivation.push({ tex: "\\text{press the step button to run the recursion}", result: "" });
      }

      renderFormulaCards(mathNode, [
        {
          title: "Forward update (filtering)",
          description:
            "Predict, then correct. Push the belief through the transition matrix, then reweight by how well each state explains what you just observed.",
          tex: "\\alpha_t(j) = \\eta\\; b_j(o_t) \\sum_i \\alpha_{t-1}(i)\\, a_{ij}",
          derivation: forwardDerivation,
          insight:
            "<strong>η is just normalisation.</strong> Without it the numbers shrink toward zero with every step and underflow. Real implementations either rescale like this or work in log space.",
        },
        {
          title: "Viterbi update (decoding)",
          description:
            "The same recursion with max in place of sum. Filtering asks “how likely is this state now?”; Viterbi asks “what is the single most likely sequence of states?”",
          tex: "\\delta_t(j) = b_j(o_t) \\max_i \\delta_{t-1}(i)\\, a_{ij}",
          derivation: shown
            ? [
                {
                  tex: `\\text{path} = (${model.path
                    .slice(0, shown)
                    .map((stateIndex) => `\\text{${model.states[stateIndex]}}`)
                    .join(",\\; ")})`,
                  result: "",
                  note: "Computed by backtracking through the stored argmax pointers, which is why the whole path can change when you reveal one more observation.",
                },
              ]
            : [{ tex: "\\text{step to decode the sequence}", result: "" }],
          insight:
            "<strong>The most likely path is not the sequence of most likely states.</strong> Taking the per-step argmax of the filtered belief can produce a path the model assigns probability zero — Viterbi cannot.",
        },
        {
          title: "Learning update (Baum–Welch)",
          description:
            "EM for HMMs: use current parameters to compute expected transition and emission counts, then re-estimate the parameters from those counts.",
          tex: "a_{ij} \\leftarrow \\frac{\\sum_t \\xi_t(i, j)}{\\sum_t \\gamma_t(i)}, \\qquad b_j(k) \\leftarrow \\frac{\\sum_{t: o_t = k} \\gamma_t(j)}{\\sum_t \\gamma_t(j)}",
          derivation: [
            {
              tex: `A = \\begin{bmatrix} ${U.texNum(transition[0][0], 2)} & ${U.texNum(transition[0][1], 2)} \\\\ ${U.texNum(transition[1][0], 2)} & ${U.texNum(transition[1][1], 2)} \\end{bmatrix}, \\quad B = \\begin{bmatrix} ${U.texNum(emission[0][0], 2)} & ${U.texNum(emission[0][1], 2)} \\\\ ${U.texNum(emission[1][0], 2)} & ${U.texNum(emission[1][1], 2)} \\end{bmatrix}`,
              result: "",
              note: "Press “Run one learning update” to re-estimate both matrices from the current sequence.",
            },
          ],
          insight:
            "<strong>EM only climbs.</strong> Each update is guaranteed not to decrease the likelihood, but it converges to a local optimum — different starting matrices reach different answers on the same sequence.",
        },
      ]);

      const steps = sequence.map(
        (obs, time) =>
          `<strong>t = ${time + 1}, obs = ${obs}.</strong> Forward belief = <span class="mono">[${U.round(model.alpha[time][0], 3)}, ${U.round(model.alpha[time][1], 3)}]</span>; Viterbi best state = <strong>${model.states[model.path[time]]}</strong>.`
      );
      U.renderSteps(stepsNode, steps, shown);

      table.innerHTML = sequence
        .map((obs, time) =>
          time >= shown
            ? `<tr style="opacity:.45"><td>${time + 1}</td><td>${obs}</td><td>—</td><td>—</td><td>—</td></tr>`
            : `<tr><td>${time + 1}</td><td>${obs}</td><td>${U.round(model.alpha[time][0], 3)}</td><td>${U.round(model.alpha[time][1], 3)}</td><td>${model.states[model.path[time]]}</td></tr>`
        )
        .join("");

      const chart = U.makeChart(plot, {
        xDomain: [1, Math.max(sequence.length, 2)],
        yDomain: [0, 1],
        title: "Filtered P(Rain) with the decoded path underneath",
      });
      plot.appendChild(
        U.svgEl("line", {
          x1: chart.xScale(1),
          y1: chart.yScale(0.5),
          x2: chart.xScale(Math.max(sequence.length, 2)),
          y2: chart.yScale(0.5),
          class: "grid-line",
        })
      );

      if (shown) {
        /* Viterbi path as a step function along the bottom. */
        model.path.slice(0, shown).forEach((stateIndex, index) => {
          plot.appendChild(
            U.svgEl("rect", {
              x: chart.xScale(index + 1) - 9,
              y: chart.yScale(0.06),
              width: 18,
              height: 10,
              rx: 3,
              fill: stateIndex === 0 ? C().a : C().b,
              opacity: "0.85",
            })
          );
        });

        const points = model.alpha.slice(0, shown).map((probability, index) => ({ x: index + 1, y: probability[0] }));
        plot.appendChild(
          U.svgEl("path", { d: U.pathFromPoints(points, chart.xScale, chart.yScale), class: "curve-primary" })
        );
        points.forEach((point, index) => {
          plot.appendChild(
            U.svgEl("circle", { cx: chart.xScale(point.x), cy: chart.yScale(point.y), r: 6, fill: C().a })
          );
          plot.appendChild(
            U.svgEl("text", {
              x: chart.xScale(point.x),
              y: chart.yScale(point.y) - 12,
              class: "svg-label",
              "text-anchor": "middle",
            })
          ).textContent = sequence[index];
        });
      }
    }

    stepButton.addEventListener("click", () => {
      revealed = Math.min(revealed + 1, parseSequence(sequenceInput.value).length);
      render();
    });
    allButton.addEventListener("click", () => {
      revealed = parseSequence(sequenceInput.value).length;
      render();
    });
    resetButton.addEventListener("click", () => {
      revealed = 0;
      render();
    });
    learnButton.addEventListener("click", () => {
      const sequence = parseSequence(sequenceInput.value);
      if (sequence.length < 2) {
        render("Add at least two observations to run a learning update.");
        return;
      }
      const { gamma, xi } = forwardBackward(sequence, transition, emission);

      /* Re-estimate A from expected transition counts. */
      const transitionCounts = [
        [0, 0],
        [0, 0],
      ];
      const occupancy = [0, 0];
      xi.forEach((item) => {
        for (let i = 0; i < 2; i += 1) {
          for (let j = 0; j < 2; j += 1) transitionCounts[i][j] += item[i][j];
          occupancy[i] += item[i][0] + item[i][1];
        }
      });
      const nextTransition = transitionCounts.map((row, i) =>
        occupancy[i] > 1e-9 ? row.map((value) => value / occupancy[i]) : transition[i].slice()
      );

      /* Re-estimate B from expected emission counts. */
      const emissionCounts = [
        [0, 0],
        [0, 0],
      ];
      const stateTotals = [0, 0];
      gamma.forEach((stateProbability, time) => {
        const column = sequence[time] === "U" ? 0 : 1;
        for (let i = 0; i < 2; i += 1) {
          emissionCounts[i][column] += stateProbability[i];
          stateTotals[i] += stateProbability[i];
        }
      });
      const nextEmission = emissionCounts.map((row, i) =>
        stateTotals[i] > 1e-9 ? row.map((value) => value / stateTotals[i]) : emission[i].slice()
      );

      transition = nextTransition;
      emission = nextEmission;
      transitionEditor.set(nextTransition, true);
      emissionEditor.set(nextEmission, true);
      revealed = sequence.length;
      render(
        `One Baum–Welch update re-estimated both matrices from the expected counts. Run it again — the likelihood cannot go down, but it can stall in a local optimum.`
      );
    });
    sequenceInput.addEventListener("input", () => {
      revealed = 0;
      render();
    });

    U.onRedraw(render);
    render();
  }

  function mountPartition(rootNode) {
    const lockedMethod =
      definition.options.focus === "kmeans"
        ? "kmeans"
        : definition.options.focus === "kmedoids"
        ? "kmedoids"
        : null;

    mountSection(
      rootNode,
      "Partition clustering loop",
      "Alternate between assignment and representative updates. Edit the point table — or drag the seeds themselves — to see how sensitive the result is to where the loop starts.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-grid">
              <div class="control-group">
                <label for="partition-method">Method</label>
                <select id="partition-method" ${lockedMethod ? "disabled" : ""}>
                  <option value="kmeans">K-means</option>
                  <option value="kmedoids">K-medoids</option>
                </select>
              </div>
              <div class="control-group">
                <label for="partition-k">Clusters k</label>
                <input id="partition-k" type="number" min="2" max="4" value="3" />
              </div>
              <div class="control-group">
                <label for="partition-metric">Distance</label>
                <select id="partition-metric">
                  <option value="euclidean">Euclidean</option>
                  <option value="manhattan">Manhattan</option>
                </select>
              </div>
              <div class="control-group">
                <label for="partition-seed">Seed placement</label>
                <button class="button secondary" id="partition-seed">Re-seed randomly</button>
              </div>
            </div>
            <div class="step-controls">
              <button class="button primary" id="partition-step">Next step</button>
              <button class="button secondary" id="partition-run">Run to convergence</button>
              <button class="button secondary" id="partition-reset">Reset</button>
            </div>
            <div class="callout" id="partition-callout"></div>
            ${editorSlot("partition-data")}
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg id="partition-plot" viewBox="0 0 560 280" aria-label="Partition clustering plot"></svg>
              <div class="drag-hint">✥ Drag points or the diamond-shaped representatives — bad seeds give bad clusters, which is the point.</div>
            </div>
            <div class="readout"><div class="output-grid" id="partition-output"></div></div>
            <div class="equation-card" id="partition-math"></div>
            <div class="steps"><ol id="partition-steps"></ol></div>
          </div>
        </div>
      `
    );

    const methodInput = rootNode.querySelector("#partition-method");
    const kInput = rootNode.querySelector("#partition-k");
    const metricInput = rootNode.querySelector("#partition-metric");
    const stepButton = rootNode.querySelector("#partition-step");
    const runButton = rootNode.querySelector("#partition-run");
    const resetButton = rootNode.querySelector("#partition-reset");
    const seedButton = rootNode.querySelector("#partition-seed");
    const callout = rootNode.querySelector("#partition-callout");
    const plot = rootNode.querySelector("#partition-plot");
    const output = rootNode.querySelector("#partition-output");
    const mathNode = rootNode.querySelector("#partition-math");
    const stepsNode = rootNode.querySelector("#partition-steps");

    let points = partitionPoints.map((point) => ({ ...point }));
    let state;

    if (lockedMethod) methodInput.value = lockedMethod;

    const editor = pointsEditor(
      rootNode.querySelector("#partition-data"),
      points,
      (rows) => {
        points = rows;
        resetState();
        render();
      },
      {
        title: "Points to cluster",
        hint: "Clustering is unsupervised, so this table has no labels — only positions. Try the “Two moons” preset with k-means and see the round clusters cut straight through the arcs.",
      }
    );

    function currentK() {
      return U.clamp(Math.round(Number(kInput.value) || 3), 2, Math.max(2, points.length));
    }

    function seedRepresentatives(k, random) {
      if (random) {
        const shuffled = [...points].sort(() => Math.random() - 0.5);
        return shuffled.slice(0, k).map((point, index) => ({ id: `r${index + 1}`, x: point.x, y: point.y }));
      }
      /* Spread the default seeds across the list so the very first
         assignment is not degenerate. */
      const stride = Math.max(1, Math.floor(points.length / k));
      return Array.from({ length: k }, (_, index) => {
        const source = points[Math.min(index * stride, points.length - 1)];
        return { id: `r${index + 1}`, x: source.x, y: source.y };
      });
    }

    function resetState(random) {
      state = {
        phase: "assign",
        assignments: Array(points.length).fill(-1),
        reps: seedRepresentatives(currentK(), random),
        history: [],
        step: 0,
        converged: false,
      };
    }

    function objective(metric, reps, assignments) {
      return points.reduce((acc, point, index) => {
        const cluster = assignments[index];
        if (cluster < 0 || !reps[cluster]) return acc;
        return acc + U.distance(point, reps[cluster], metric);
      }, 0);
    }

    function takeStep() {
      const method = methodInput.value;
      const metric = metricInput.value;

      if (state.phase === "assign") {
        const nextAssignments = points.map((point) => {
          const distances = state.reps.map((rep) => U.distance(point, rep, metric));
          return U.argMax(distances.map((value) => -value));
        });
        const changed = nextAssignments.filter((value, index) => value !== state.assignments[index]).length;
        state.assignments = nextAssignments;
        state.phase = "update";
        state.step += 1;
        state.converged = changed === 0 && state.step > 1;
        state.history.push(
          `<strong>Assignment ${state.step}.</strong> Each point joined its nearest representative — ${changed} membership${changed !== 1 ? "s" : ""} changed.`
        );
      } else {
        const grouped = state.reps.map((_, clusterIndex) =>
          points.filter((_, pointIndex) => state.assignments[pointIndex] === clusterIndex)
        );
        const nextReps = grouped.map((clusterPoints, clusterIndex) => {
          if (!clusterPoints.length) return state.reps[clusterIndex];
          if (method === "kmeans") {
            return {
              id: `r${clusterIndex + 1}`,
              x: U.mean(clusterPoints.map((point) => point.x)),
              y: U.mean(clusterPoints.map((point) => point.y)),
            };
          }
          return clusterPoints
            .map((candidate) => ({
              candidate,
              cost: clusterPoints.reduce((acc, point) => acc + U.distance(candidate, point, metric), 0),
            }))
            .sort((a, b) => a.cost - b.cost)[0].candidate;
        });
        const move = nextReps.reduce((acc, rep, index) => acc + U.distance(rep, state.reps[index], "euclidean"), 0);
        state.reps = nextReps.map((rep, index) => ({ id: `r${index + 1}`, x: rep.x, y: rep.y }));
        state.phase = "assign";
        state.converged = move < 1e-9;
        state.history.push(
          `<strong>${method === "kmeans" ? "Centroid" : "Medoid"} update ${state.step}.</strong> Total representative movement = ${U.round(move, 4)}.`
        );
      }
      if (state.history.length > 10) state.history = state.history.slice(-10);
    }

    function render() {
      if (!state || state.reps.length !== currentK() || state.assignments.length !== points.length) resetState();

      const method = methodInput.value;
      const metric = metricInput.value;
      const objectiveValue = objective(metric, state.reps, state.assignments);

      callout.innerHTML =
        method === "kmeans"
          ? "K-means uses arithmetic means as representatives, so a representative need not be a real data point."
          : "K-medoids restricts representatives to actual observed points, which makes it far more robust to outliers.";

      U.renderMetrics(output, [
        { label: "Phase", value: state.phase === "assign" ? "assign next" : "update next" },
        { label: "Objective", value: U.round(objectiveValue, 3) },
        { label: "Clusters", value: String(state.reps.length) },
        { label: "Status", value: state.converged ? "converged" : `step ${state.step}` },
      ]);

      const counts = state.reps.map((_, clusterIndex) => state.assignments.filter((value) => value === clusterIndex).length);

      /* Show the arithmetic for one concrete point so the argmin is not
         abstract — pick the first assigned point. */
      const sampleIndex = state.assignments.findIndex((value) => value >= 0);
      const sampleLines = [];
      if (sampleIndex >= 0) {
        const point = points[sampleIndex];
        state.reps.forEach((rep, clusterIndex) => {
          const d = U.distance(point, rep, metric);
          sampleLines.push({
            tex: `d(\\text{${point.id}},\\, r_${clusterIndex + 1}) = ${U.texNum(d, 3)}`,
            result: clusterIndex === state.assignments[sampleIndex] ? "← nearest" : "",
          });
        });
      }

      renderFormulaCards(mathNode, [
        {
          title: "Assignment step",
          description: "Every point joins the closest representative. Ties are broken arbitrarily.",
          tex: "c_i = \\operatorname*{arg\\,min}_{k} \\; d(\\mathbf{x}_i, \\mathbf{r}_k)",
          derivation: sampleLines.length
            ? [
                ...sampleLines,
                {
                  tex: `c_{\\text{${points[sampleIndex].id}}} = ${state.assignments[sampleIndex] + 1}`,
                  result: `cluster ${state.assignments[sampleIndex] + 1}`,
                  note: `Worked through for point ${points[sampleIndex].id}; every other point is decided the same way.`,
                },
                ...counts.map((count, clusterIndex) => ({
                  tex: `|C_${clusterIndex + 1}| = ${count}`,
                  result: String(count),
                })),
              ]
            : [{ tex: "\\text{press ``Next step'' to assign points}", result: "" }],
          insight:
            "<strong>The objective only ever decreases.</strong> Assignment cannot raise the total distance, and neither can the update — which is why the loop always converges, though not necessarily to the best solution.",
        },
        {
          title: method === "kmeans" ? "Centroid update" : "Medoid update",
          description:
            method === "kmeans"
              ? "The representative moves to the geometric average of the points assigned to it."
              : "The representative becomes the in-cluster point with the smallest total distance to its peers.",
          tex:
            method === "kmeans"
              ? "\\mathbf{r}_k = \\frac{1}{|C_k|}\\sum_{i \\in C_k} \\mathbf{x}_i"
              : "\\mathbf{r}_k = \\operatorname*{arg\\,min}_{\\mathbf{x}_j \\in C_k} \\sum_{i \\in C_k} d(\\mathbf{x}_i, \\mathbf{x}_j)",
          derivation: state.reps.map((rep, clusterIndex) => ({
            tex: `\\mathbf{r}_${clusterIndex + 1} = (${U.texNum(rep.x, 2)},\\; ${U.texNum(rep.y, 2)})`,
            result: `${counts[clusterIndex]} pts`,
          })),
          insight:
            method === "kmeans"
              ? "<strong>Means chase outliers.</strong> Add a single far-away point in the table and watch its centroid get dragged into empty space."
              : "<strong>Medoids resist outliers.</strong> Add the same far-away point and the medoid barely moves, because it must remain a real observation.",
        },
        {
          title: "Total objective",
          description: "The quantity the loop is actually minimising, summed over every point.",
          tex: "J = \\sum_{k=1}^{K}\\sum_{i \\in C_k} d(\\mathbf{x}_i, \\mathbf{r}_k)",
          derivation: [
            {
              tex: `J = ${U.texNum(objectiveValue, 4)}`,
              result: U.round(objectiveValue, 4),
              note: state.converged
                ? "No membership changed and no representative moved — this is a local optimum."
                : "Keep stepping; this number can only go down.",
            },
          ],
          insight:
            "<strong>Local optima are real:</strong> hit “Re-seed randomly” a few times on the same data. Different starts land on different J values — that is why k-means is run many times in practice.",
        },
      ]);

      U.renderSteps(stepsNode, state.history, state.history.length);

      const chart = U.makeChart(plot, { xDomain: [0, 10], yDomain: [0, 10], title: "Partition clustering state" });

      /* Connect each point to its representative so membership is legible
         even where clusters visually overlap. */
      points.forEach((point, index) => {
        const cluster = state.assignments[index];
        if (cluster < 0 || !state.reps[cluster]) return;
        plot.appendChild(
          U.svgEl("line", {
            x1: chart.xScale(point.x),
            y1: chart.yScale(point.y),
            x2: chart.xScale(state.reps[cluster].x),
            y2: chart.yScale(state.reps[cluster].y),
            stroke: clusterPaletteAt(cluster),
            "stroke-width": "1.2",
            opacity: "0.35",
          })
        );
      });

      points.forEach((point, index) => {
        const cluster = state.assignments[index];
        drawScatterPoint(
          plot,
          chart,
          point,
          cluster >= 0 ? clusterPaletteAt(cluster) : C().neutral,
          7,
          (target, position) => {
            target.x = Number(U.round(position.x, 2));
            target.y = Number(U.round(position.y, 2));
            editor.setCell(index, "x", target.x);
            editor.setCell(index, "y", target.y);
            scheduleRender();
          }
        );
      });

      state.reps.forEach((rep, index) => {
        const x = chart.xScale(rep.x);
        const y = chart.yScale(rep.y);
        const marker = U.svgEl("polygon", {
          points: `${x},${y - 12} ${x + 12},${y} ${x},${y + 12} ${x - 12},${y}`,
          fill: clusterPaletteAt(index),
          stroke: C().ring,
          "stroke-width": "2.5",
        });
        plot.appendChild(marker);
        U.draggable(marker, plot, chart, (position) => {
          rep.x = Number(U.round(position.x, 2));
          rep.y = Number(U.round(position.y, 2));
          scheduleRender();
        });
      });
    }

    const scheduleRender = U.rafThrottle(render);

    stepButton.addEventListener("click", () => {
      takeStep();
      render();
    });
    runButton.addEventListener("click", () => {
      for (let index = 0; index < 24 && !state.converged; index += 1) takeStep();
      render();
    });
    resetButton.addEventListener("click", () => {
      resetState();
      render();
    });
    seedButton.addEventListener("click", () => {
      resetState(true);
      render();
    });
    [methodInput, kInput, metricInput].forEach((input) =>
      input.addEventListener("input", () => {
        resetState();
        render();
      })
    );

    resetState();
    U.onRedraw(render);
    render();
  }

  function clusterDistance(clusterA, clusterB, linkage) {
    const distances = clusterA.members.flatMap((a) => clusterB.members.map((b) => U.distance(a, b, "euclidean")));
    if (linkage === "single") {
      return Math.min(...distances);
    }
    if (linkage === "complete") {
      return Math.max(...distances);
    }
    return U.mean(distances);
  }

  function clusterCentroid(points) {
    return { x: U.mean(points.map((point) => point.x)), y: U.mean(points.map((point) => point.y)) };
  }

  function clusterSSE(points) {
    if (points.length <= 1) {
      return 0;
    }
    const centroid = clusterCentroid(points);
    return points.reduce((acc, point) => acc + U.distance(point, centroid, "euclidean") ** 2, 0);
  }

  function splitCluster(points) {
    if (points.length <= 1) {
      return [points, []];
    }
    let bestPair = [points[0], points[1]];
    let bestDistance = -1;
    for (let i = 0; i < points.length; i += 1) {
      for (let j = i + 1; j < points.length; j += 1) {
        const d = U.distance(points[i], points[j], "euclidean");
        if (d > bestDistance) {
          bestDistance = d;
          bestPair = [points[i], points[j]];
        }
      }
    }
    const left = [];
    const right = [];
    points.forEach((point) => {
      const leftDistance = U.distance(point, bestPair[0], "euclidean");
      const rightDistance = U.distance(point, bestPair[1], "euclidean");
      if (leftDistance <= rightDistance) {
        left.push(point);
      } else {
        right.push(point);
      }
    });
    if (!left.length || !right.length) {
      const sorted = [...points].sort((a, b) => a.x - b.x);
      return [sorted.slice(0, Math.ceil(sorted.length / 2)), sorted.slice(Math.ceil(sorted.length / 2))];
    }
    return [left, right];
  }

  function mountHierarchy(rootNode) {
    const lockedFocus = definition.options.focus === "general" ? null : definition.options.focus;
    mountSection(
      rootNode,
      "Hierarchical clustering explorer",
      "Switch between agglomerative and divisive reasoning, reveal one merge or split at a time, and read the dendrogram that the merges build up. The pairwise distance matrix below shows exactly which pair is about to be joined.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-grid">
              <div class="control-group">
                <label for="hier-mode">Hierarchy mode</label>
                <select id="hier-mode" ${lockedFocus ? "disabled" : ""}>
                  <option value="agglomerative">Agglomerative (bottom-up)</option>
                  <option value="divisive">Divisive (top-down)</option>
                </select>
              </div>
              <div class="control-group">
                <label for="hier-linkage">Linkage</label>
                <select id="hier-linkage">
                  <option value="single">Single linkage</option>
                  <option value="complete">Complete linkage</option>
                  <option value="average">Average linkage</option>
                </select>
              </div>
            </div>
            <div class="step-controls">
              <button class="button primary" id="hier-step">Next hierarchy step</button>
              <button class="button secondary" id="hier-run">Run to the end</button>
              <button class="button secondary" id="hier-reset">Reset hierarchy</button>
            </div>
            <div class="callout" id="hier-callout"></div>
            ${editorSlot("hier-data")}
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg id="hier-plot" viewBox="0 0 560 280" aria-label="Hierarchical clustering plot"></svg>
              ${DRAG_HINT}
            </div>
            <div class="plot-card">
              <svg id="hier-dendrogram" viewBox="0 0 560 300" aria-label="Dendrogram"></svg>
            </div>
            <div class="mini-panel" id="hier-history"></div>
            <div class="equation-card" id="hier-math"></div>
            <div class="table-panel" id="hier-matrix"></div>
            <div class="steps"><ol id="hier-steps"></ol></div>
          </div>
        </div>
      `
    );

    const modeInput = rootNode.querySelector("#hier-mode");
    const linkageInput = rootNode.querySelector("#hier-linkage");
    const stepButton = rootNode.querySelector("#hier-step");
    const runButton = rootNode.querySelector("#hier-run");
    const resetButton = rootNode.querySelector("#hier-reset");
    const plot = rootNode.querySelector("#hier-plot");
    const dendro = rootNode.querySelector("#hier-dendrogram");
    const historyNode = rootNode.querySelector("#hier-history");
    const mathNode = rootNode.querySelector("#hier-math");
    const matrixNode = rootNode.querySelector("#hier-matrix");
    const stepsNode = rootNode.querySelector("#hier-steps");
    const callout = rootNode.querySelector("#hier-callout");

    let points = hierarchicalPoints.map((point) => ({ ...point }));
    let state;

    if (lockedFocus) modeInput.value = lockedFocus;

    const editor = pointsEditor(
      rootNode.querySelector("#hier-data"),
      points,
      (rows) => {
        points = rows;
        resetState();
        render();
      },
      {
        title: "Points to cluster",
        hint: "Six to ten points keeps the dendrogram readable. Move two points close together and they will be the first pair to merge.",
        minRows: 3,
        presets: {
          "Three pairs": hierarchicalPoints,
          "Chain": [
            { id: "A", x: 1.0, y: 2.0 }, { id: "B", x: 2.2, y: 2.4 }, { id: "C", x: 3.4, y: 2.8 },
            { id: "D", x: 4.6, y: 3.2 }, { id: "E", x: 5.8, y: 3.6 }, { id: "F", x: 7.0, y: 4.0 },
          ],
          "One far outlier": [
            { id: "A", x: 2.0, y: 2.0 }, { id: "B", x: 2.4, y: 2.6 }, { id: "C", x: 2.8, y: 2.2 },
            { id: "D", x: 3.2, y: 2.8 }, { id: "E", x: 2.6, y: 3.2 }, { id: "Z", x: 9.2, y: 8.8 },
          ],
        },
      }
    );

    function resetState() {
      if (modeInput.value === "agglomerative") {
        state = {
          mode: "agglomerative",
          clusters: points.map((point) => ({ id: point.id, members: [point], height: 0, children: [] })),
          history: [],
          nextId: 1,
          done: false,
        };
      } else {
        state = {
          mode: "divisive",
          clusters: [{ id: "C0", members: points.slice(), height: clusterSSE(points), children: [] }],
          history: [],
          nextId: 1,
          done: false,
        };
      }
      state.roots = state.clusters.slice();
    }

    /* The pair the algorithm would join next — computed separately from
       takeStep so the matrix can highlight it before you commit. */
    function nextMergePair() {
      const linkage = linkageInput.value;
      if (state.mode !== "agglomerative" || state.clusters.length <= 1) return null;
      let best = null;
      for (let i = 0; i < state.clusters.length; i += 1) {
        for (let j = i + 1; j < state.clusters.length; j += 1) {
          const distance = clusterDistance(state.clusters[i], state.clusters[j], linkage);
          if (!best || distance < best.distance) best = { i, j, distance };
        }
      }
      return best;
    }

    function takeStep() {
      if (state.mode === "agglomerative") {
        const best = nextMergePair();
        if (!best) {
          state.done = true;
          return;
        }
        const left = state.clusters[best.i];
        const right = state.clusters[best.j];
        const merged = {
          id: `M${state.nextId}`,
          members: [...left.members, ...right.members],
          height: best.distance,
          children: [left, right],
        };
        state.nextId += 1;
        state.history.push({ action: "merge", left: left.id, right: right.id, distance: best.distance, id: merged.id });
        state.clusters = state.clusters.filter((_, index) => index !== best.i && index !== best.j);
        state.clusters.push(merged);
        state.roots = state.clusters.slice();
        if (state.clusters.length <= 1) state.done = true;
      } else {
        const candidates = state.clusters.filter((cluster) => cluster.members.length > 1);
        if (!candidates.length) {
          state.done = true;
          return;
        }
        const target = candidates.sort((a, b) => clusterSSE(b.members) - clusterSSE(a.members))[0];
        const [leftMembers, rightMembers] = splitCluster(target.members);
        const leftId = `S${state.nextId}`;
        const rightId = `S${state.nextId + 1}`;
        state.nextId += 2;
        const leftNode = { id: leftId, members: leftMembers, height: clusterSSE(leftMembers), children: [] };
        const rightNode = { id: rightId, members: rightMembers, height: clusterSSE(rightMembers), children: [] };
        target.children = [leftNode, rightNode];
        state.history.push({
          action: "split",
          parent: target.id,
          left: leftId,
          right: rightId,
          score: clusterSSE(target.members),
        });
        state.clusters = state.clusters.filter((cluster) => cluster.id !== target.id).concat([leftNode, rightNode]);
      }
    }

    /* ── Dendrogram ─────────────────────────────────────────────
       Leaves are laid out left-to-right in the order an in-order walk
       of the merge tree visits them, which is exactly what stops the
       connector lines from crossing. */
    function drawDendrogram() {
      U.clear(dendro);
      const { width, height } = U.viewBoxSize(dendro);
      const padding = { top: 26, right: 20, bottom: 42, left: 46 };
      const roots = state.mode === "agglomerative" ? state.roots : [state.clusters[0] ? findRoot() : null].filter(Boolean);

      const leaves = [];
      function collectLeaves(node) {
        if (!node.children || !node.children.length) {
          leaves.push(node);
          return;
        }
        node.children.forEach(collectLeaves);
      }
      roots.forEach(collectLeaves);

      if (!leaves.length) return;

      const maxHeight = Math.max(
        0.001,
        ...roots.map(function maxOf(node) {
          const own = node.height || 0;
          const kids = (node.children || []).map(maxOf);
          return Math.max(own, ...(kids.length ? kids : [0]));
        })
      );

      const stepX = (width - padding.left - padding.right) / Math.max(leaves.length - 1, 1);
      const yFor = (h) => height - padding.bottom - (h / maxHeight) * (height - padding.top - padding.bottom);

      dendro.appendChild(
        U.svgEl("text", { x: padding.left, y: 16, class: "svg-title" })
      ).textContent = state.mode === "agglomerative" ? "Dendrogram — merge height = linkage distance" : "Split tree — height = cluster SSE";

      /* Axis for the merge heights, so the dendrogram is quantitative. */
      for (let index = 0; index <= 4; index += 1) {
        const value = (maxHeight * index) / 4;
        const y = yFor(value);
        dendro.appendChild(
          U.svgEl("line", { x1: padding.left, y1: y, x2: width - padding.right, y2: y, class: "grid-line" })
        );
        dendro.appendChild(U.svgEl("text", { x: 8, y: y + 4, class: "svg-label" })).textContent = U.round(value, 2);
      }

      const positions = new Map();
      leaves.forEach((leaf, index) => {
        positions.set(leaf.id, padding.left + index * stepX);
      });

      const colorFor = new Map();
      state.clusters.forEach((cluster, index) => {
        cluster.members.forEach((member) => colorFor.set(member.id, clusterPaletteAt(index)));
      });

      function draw(node) {
        if (!node.children || !node.children.length) {
          const x = positions.get(node.id);
          const y = yFor(0);
          dendro.appendChild(
            U.svgEl("circle", { cx: x, cy: y, r: 4, fill: colorFor.get(node.members[0].id) || C().neutral })
          );
          dendro.appendChild(
            U.svgEl("text", { x, y: y + 20, class: "svg-label", "text-anchor": "middle" })
          ).textContent = node.members.length === 1 ? node.members[0].id : node.id;
          return x;
        }
        const childX = node.children.map(draw);
        const x = U.mean(childX);
        positions.set(node.id, x);
        const y = yFor(node.height || 0);
        node.children.forEach((child, index) => {
          const cx = childX[index];
          const cy = yFor(child.children && child.children.length ? child.height || 0 : 0);
          dendro.appendChild(U.svgEl("line", { x1: cx, y1: cy, x2: cx, y2: y, class: "cluster-line" }));
        });
        dendro.appendChild(
          U.svgEl("line", { x1: Math.min(...childX), y1: y, x2: Math.max(...childX), y2: y, class: "cluster-line" })
        );
        dendro.appendChild(
          U.svgEl("text", { x: x + 6, y: y - 6, class: "svg-label" })
        ).textContent = U.round(node.height || 0, 2);
        return x;
      }

      roots.forEach(draw);
    }

    function findRoot() {
      /* Divisive mode keeps the original root around so the split tree
         can be drawn from the top. */
      return state.rootNode;
    }

    function render() {
      if (!state || state.mode !== modeInput.value) resetState();
      if (state.mode === "divisive" && !state.rootNode) state.rootNode = state.clusters[0];

      const linkage = linkageInput.value;
      callout.innerHTML =
        state.mode === "agglomerative"
          ? `Agglomerative mode with <strong>${linkage}</strong> linkage merges the closest pair of clusters first.`
          : "Divisive mode repeatedly splits whichever cluster has the largest internal spread.";

      const pending = nextMergePair();
      const linkageTex =
        linkage === "single"
          ? "d(A, B) = \\min_{a \\in A,\\, b \\in B} d(a, b)"
          : linkage === "complete"
          ? "d(A, B) = \\max_{a \\in A,\\, b \\in B} d(a, b)"
          : "d(A, B) = \\frac{1}{|A||B|}\\sum_{a \\in A}\\sum_{b \\in B} d(a, b)";

      renderFormulaCards(mathNode, [
        {
          title: state.mode === "agglomerative" ? "Linkage rule" : "Split heuristic",
          description:
            state.mode === "agglomerative"
              ? "The distance between two clusters depends entirely on the linkage definition — the same points can produce very different trees."
              : "Divisive clustering picks the cluster with the largest sum of squared error, then splits it around its two most distant members.",
          tex:
            state.mode === "agglomerative"
              ? linkageTex
              : "\\text{SSE}(C) = \\sum_{i \\in C} \\lVert \\mathbf{x}_i - \\bar{\\mathbf{x}}_C \\rVert^2",
          derivation:
            state.mode === "agglomerative" && pending
              ? [
                  {
                    tex: `\\text{closest pair} = (\\text{${state.clusters[pending.i].id}},\\, \\text{${state.clusters[pending.j].id}})`,
                    result: U.round(pending.distance, 3),
                    note: `Chosen from ${(state.clusters.length * (state.clusters.length - 1)) / 2} candidate pairs at this level.`,
                  },
                  {
                    tex: `d = ${U.texNum(pending.distance, 4)}`,
                    result: U.round(pending.distance, 4),
                    note: "This value becomes the height of the new branch in the dendrogram.",
                  },
                ]
              : state.mode === "divisive"
              ? state.clusters.map((cluster) => ({
                  tex: `\\text{SSE}(\\text{${cluster.id}}) = ${U.texNum(clusterSSE(cluster.members), 3)}`,
                  result: `${cluster.members.length} pts`,
                }))
              : [{ tex: "\\text{everything is merged into one cluster}", result: "" }],
          insight:
            linkage === "single"
              ? "<strong>Single linkage chains.</strong> Load the “Chain” preset — single linkage happily strings all six points into one long cluster, because it only ever looks at the closest pair."
              : linkage === "complete"
              ? "<strong>Complete linkage makes compact blobs.</strong> It is driven by the worst-case pair, so it resists chaining but breaks up elongated shapes."
              : "<strong>Average linkage sits in between</strong> — less prone to chaining than single, less aggressive at splitting elongated clusters than complete.",
        },
        {
          title: "Reading the dendrogram",
          description:
            "The height at which two branches join is the distance at which those clusters merged. Cutting the tree horizontally at any height gives you a clustering.",
          tex: "\\text{clusters}(h) = \\{\\, C : \\text{merge height}(C) \\le h \\,\\}",
          derivation: state.history.length
            ? [
                {
                  tex: `\\text{merges so far} = ${state.history.length}`,
                  result: String(state.clusters.length) + " clusters",
                  note: "A tall jump between consecutive merge heights is the usual signal for where to cut.",
                },
              ]
            : [{ tex: "\\text{step the algorithm to grow the tree}", result: "" }],
          insight:
            "<strong>No k needed up front.</strong> Unlike k-means you do not choose the number of clusters before running — you choose it afterwards by picking a cut height.",
        },
      ]);

      /* Live pairwise distance matrix with the next merge highlighted. */
      if (state.mode === "agglomerative" && state.clusters.length > 1 && state.clusters.length <= 12) {
        const ids = state.clusters.map((cluster) => cluster.id);
        matrixNode.innerHTML = `
          <table>
            <thead><tr><th>${linkage} linkage</th>${ids.map((id) => `<th>${id}</th>`).join("")}</tr></thead>
            <tbody>
              ${state.clusters
                .map(
                  (rowCluster, i) => `
                    <tr>
                      <th>${rowCluster.id}</th>
                      ${state.clusters
                        .map((colCluster, j) => {
                          if (i === j) return `<td style="opacity:.3">—</td>`;
                          const d = clusterDistance(rowCluster, colCluster, linkage);
                          const isNext =
                            pending && ((i === pending.i && j === pending.j) || (i === pending.j && j === pending.i));
                          return `<td style="${isNext ? "background:rgba(26,122,110,0.16);font-weight:700;color:var(--accent)" : ""}">${U.round(d, 2)}</td>`;
                        })
                        .join("")}
                    </tr>
                  `
                )
                .join("")}
            </tbody>
          </table>
        `;
      } else {
        matrixNode.innerHTML = `<p class="caption">The pairwise distance matrix appears in agglomerative mode while more than one cluster remains.</p>`;
      }

      U.renderSteps(
        stepsNode,
        state.history.map((entry, index) =>
          entry.action === "merge"
            ? `<strong>Merge ${index + 1}.</strong> Combine ${entry.left} and ${entry.right} at distance <strong>${U.round(entry.distance, 3)}</strong>.`
            : `<strong>Split ${index + 1}.</strong> Split ${entry.parent} into ${entry.left} and ${entry.right} from parent SSE <strong>${U.round(entry.score, 3)}</strong>.`
        ),
        state.history.length
      );

      historyNode.innerHTML = `
        <h3>${state.mode === "agglomerative" ? "Current clusters" : "Current split frontier"}</h3>
        <ul class="dense-list">
          ${state.clusters
            .map((cluster) => `<li><strong>${cluster.id}</strong>: ${cluster.members.map((point) => point.id).join(", ")}</li>`)
            .join("")}
        </ul>
      `;

      const colorMap = {};
      state.clusters.forEach((cluster, index) => {
        cluster.members.forEach((point) => {
          colorMap[point.id] = clusterPaletteAt(index);
        });
      });

      const xs = points.map((point) => point.x);
      const ys = points.map((point) => point.y);
      const chart = U.makeChart(plot, {
        xDomain: [Math.min(...xs) - 1, Math.max(...xs) + 1],
        yDomain: [Math.min(...ys) - 1, Math.max(...ys) + 1],
        title: "Current hierarchical clustering state",
      });

      /* Draw the convex-ish hull as connecting lines inside each cluster
         so groupings are visible on the scatter too. */
      state.clusters.forEach((cluster, index) => {
        if (cluster.members.length < 2) return;
        const centroid = clusterCentroid(cluster.members);
        cluster.members.forEach((member) => {
          plot.appendChild(
            U.svgEl("line", {
              x1: chart.xScale(centroid.x),
              y1: chart.yScale(centroid.y),
              x2: chart.xScale(member.x),
              y2: chart.yScale(member.y),
              stroke: clusterPaletteAt(index),
              "stroke-width": "1.4",
              opacity: "0.4",
            })
          );
        });
      });

      points.forEach((point, index) => {
        drawScatterPoint(plot, chart, point, colorMap[point.id] || C().neutral, 8, (target, position) => {
          target.x = Number(U.round(position.x, 2));
          target.y = Number(U.round(position.y, 2));
          editor.setCell(index, "x", target.x);
          editor.setCell(index, "y", target.y);
          resetState();
          scheduleRender();
        });
      });

      drawDendrogram();
    }

    const scheduleRender = U.rafThrottle(render);

    stepButton.addEventListener("click", () => {
      takeStep();
      render();
    });
    runButton.addEventListener("click", () => {
      for (let index = 0; index < 40 && !state.done; index += 1) takeStep();
      render();
    });
    resetButton.addEventListener("click", () => {
      resetState();
      render();
    });
    [modeInput, linkageInput].forEach((input) =>
      input.addEventListener("input", () => {
        resetState();
        render();
      })
    );

    resetState();
    U.onRedraw(render);
    render();
  }

  function buildDbscanEvents(points, eps, minPts) {
    const visited = new Set();
    const assigned = new Map();
    const events = [];
    let clusterId = 0;
    function neighbors(point) {
      return points.filter((candidate) => U.distance(point, candidate, "euclidean") <= eps);
    }
    points.forEach((point) => {
      if (visited.has(point.id)) {
        return;
      }
      visited.add(point.id);
      const seedNeighbors = neighbors(point);
      events.push({ type: "visit", pointId: point.id, neighbors: seedNeighbors.map((item) => item.id), core: seedNeighbors.length >= minPts });
      if (seedNeighbors.length < minPts) {
        events.push({ type: "noise", pointId: point.id });
        return;
      }
      clusterId += 1;
      assigned.set(point.id, clusterId);
      events.push({ type: "start", pointId: point.id, clusterId });
      const queue = seedNeighbors.filter((item) => item.id !== point.id);
      while (queue.length) {
        const current = queue.shift();
        if (!visited.has(current.id)) {
          visited.add(current.id);
          const currentNeighbors = neighbors(current);
          events.push({ type: "visit", pointId: current.id, neighbors: currentNeighbors.map((item) => item.id), core: currentNeighbors.length >= minPts, clusterId });
          if (currentNeighbors.length >= minPts) {
            currentNeighbors.forEach((candidate) => {
              if (!visited.has(candidate.id) && !queue.some((item) => item.id === candidate.id)) {
                queue.push(candidate);
              }
            });
          }
        }
        if (!assigned.has(current.id)) {
          assigned.set(current.id, clusterId);
          const currentNeighbors = neighbors(current);
          events.push({ type: "assign", pointId: current.id, clusterId, pointType: currentNeighbors.length >= minPts ? "core" : "border" });
        }
      }
    });
    return events;
  }

  function mountDbscan(rootNode) {
    mountSection(
      rootNode,
      "Density-based growth",
      "DBSCAN never asks how many clusters there are — it asks which points sit in dense neighbourhoods. Tune ε and minPts, edit the points, and step through the expansion to see clusters grow and outliers fall out as noise.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-grid">
              <div class="control-group">
                <label for="dbscan-eps">Radius ε</label>
                <div class="range-row">
                  <input id="dbscan-eps" type="range" min="0.3" max="4" step="0.1" value="1.2" />
                  <span class="range-value" id="dbscan-eps-value">1.2</span>
                </div>
              </div>
              <div class="control-group">
                <label for="dbscan-minpts">minPts</label>
                <input id="dbscan-minpts" type="number" min="2" max="8" value="3" />
              </div>
            </div>
            <div class="step-controls">
              <button class="button primary" id="dbscan-step">Next DBSCAN step</button>
              <button class="button secondary" id="dbscan-run">Run all steps</button>
              <button class="button secondary" id="dbscan-reset">Reset</button>
            </div>
            <div class="callout" id="dbscan-callout"></div>
            ${editorSlot("dbscan-data")}
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg id="dbscan-plot" viewBox="0 0 560 280" aria-label="DBSCAN plot"></svg>
              ${DRAG_HINT}
            </div>
            <div class="readout"><div class="output-grid" id="dbscan-output"></div></div>
            <div class="equation-card" id="dbscan-math"></div>
            <div class="steps"><ol id="dbscan-steps"></ol></div>
          </div>
        </div>
      `
    );

    const epsInput = rootNode.querySelector("#dbscan-eps");
    const epsValue = rootNode.querySelector("#dbscan-eps-value");
    const minPtsInput = rootNode.querySelector("#dbscan-minpts");
    const stepButton = rootNode.querySelector("#dbscan-step");
    const runButton = rootNode.querySelector("#dbscan-run");
    const resetButton = rootNode.querySelector("#dbscan-reset");
    const callout = rootNode.querySelector("#dbscan-callout");
    const plot = rootNode.querySelector("#dbscan-plot");
    const output = rootNode.querySelector("#dbscan-output");
    const mathNode = rootNode.querySelector("#dbscan-math");
    const stepsNode = rootNode.querySelector("#dbscan-steps");

    let points = dbscanPoints.map((point) => ({ ...point }));
    let revealed = 0;

    const editor = pointsEditor(
      rootNode.querySelector("#dbscan-data"),
      points,
      (rows) => {
        points = rows;
        revealed = 0;
        render();
      },
      {
        title: "Points to cluster",
        hint: "Drop a point far from everything and it will be labelled noise. Fill the gap between two clusters and DBSCAN will merge them into one — density is all that matters.",
      }
    );

    function replay(events, count) {
      const state = {};
      points.forEach((point) => {
        state[point.id] = { clusterId: null, type: "unvisited", visited: false };
      });
      let active = null;
      let lastText = "Use the step button to start DBSCAN.";
      events.slice(0, count).forEach((event) => {
        active = event.pointId;
        if (!state[event.pointId]) return;
        if (event.type === "visit") {
          state[event.pointId].visited = true;
          state[event.pointId].type = event.core ? "core" : "candidate";
          lastText = `${event.pointId} has ${event.neighbors.length} point${event.neighbors.length !== 1 ? "s" : ""} inside its ε-neighbourhood.`;
        } else if (event.type === "noise") {
          state[event.pointId].type = "noise";
          lastText = `${event.pointId} is currently marked as noise.`;
        } else if (event.type === "start") {
          state[event.pointId].clusterId = event.clusterId;
          state[event.pointId].type = "core";
          lastText = `${event.pointId} is dense enough to seed cluster ${event.clusterId}.`;
        } else if (event.type === "assign") {
          state[event.pointId].clusterId = event.clusterId;
          state[event.pointId].type = event.pointType;
          lastText = `${event.pointId} joins cluster ${event.clusterId} as a ${event.pointType} point.`;
        }
      });
      return { state, active, lastText };
    }

    function render() {
      const eps = Number(epsInput.value) || 1.2;
      const minPts = Math.max(Number(minPtsInput.value) || 3, 2);
      epsValue.textContent = eps.toFixed(1);

      const events = buildDbscanEvents(points, eps, minPts);
      const { state, active, lastText } = replay(events, revealed);
      callout.innerHTML = lastText;

      const clusterIds = new Set(
        Object.values(state)
          .map((item) => item.clusterId)
          .filter((value) => value !== null)
      );
      const noiseCount = Object.values(state).filter((item) => item.type === "noise").length;
      const coreCount = Object.values(state).filter((item) => item.type === "core").length;

      U.renderMetrics(output, [
        { label: "Events shown", value: `${revealed} / ${events.length}` },
        { label: "Clusters", value: String(clusterIds.size) },
        { label: "Core points", value: String(coreCount) },
        { label: "Noise", value: String(noiseCount) },
      ]);

      /* Show the neighbourhood count for the active point so the core
         test is arithmetic rather than assertion. */
      const activeEvent = revealed > 0 ? events[revealed - 1] : null;
      const activePoint = active ? points.find((point) => point.id === active) : null;
      const neighbourLines = [];
      if (activePoint) {
        const neighbours = points.filter(
          (point) => point.id !== activePoint.id && U.distance(point, activePoint, "euclidean") <= eps
        );
        neighbours.slice(0, 5).forEach((point) => {
          neighbourLines.push({
            tex: `d(\\text{${activePoint.id}}, \\text{${point.id}}) = ${U.texNum(U.distance(point, activePoint, "euclidean"), 3)} \\le ${U.texNum(eps, 2)}`,
            result: "in",
          });
        });
        neighbourLines.push({
          tex: `|N_\\varepsilon(\\text{${activePoint.id}})| = ${neighbours.length + 1}\\; ${neighbours.length + 1 >= minPts ? "\\ge" : "<"}\\; \\text{minPts} = ${minPts}`,
          result: neighbours.length + 1 >= minPts ? "core" : "not core",
          note:
            neighbours.length + 1 >= minPts
              ? "Dense enough, so this point can seed or extend a cluster."
              : "Too sparse to be a core point — it can still be a border point if a core point reaches it.",
        });
      }

      renderFormulaCards(mathNode, [
        {
          title: "ε-neighbourhood",
          description: "DBSCAN begins every decision by asking which points lie within ε of the point in hand.",
          tex: `N_\\varepsilon(p) = \\{\\, q \\in D : d(p, q) \\le ${U.texNum(eps, 2)} \\,\\}`,
          derivation: neighbourLines.length
            ? neighbourLines
            : [{ tex: "\\text{step the algorithm to inspect a neighbourhood}", result: "" }],
          insight:
            "<strong>ε is the whole ballgame.</strong> Slide it up and separate clusters fuse into one; slide it down and everything becomes noise. There is no k to choose, but ε is just as consequential.",
        },
        {
          title: "Core, border and noise",
          description:
            "A core point has at least minPts neighbours. A border point is not core but lies inside some core point's radius. Everything else is noise.",
          tex: "p \\text{ is core} \\iff |N_\\varepsilon(p)| \\ge \\text{minPts} = " + minPts,
          derivation: [
            { tex: `\\text{core points} = ${coreCount}`, result: String(coreCount) },
            { tex: `\\text{noise points} = ${noiseCount}`, result: String(noiseCount) },
            {
              tex: `\\text{clusters found} = ${clusterIds.size}`,
              result: String(clusterIds.size),
              note: "The number of clusters is an output here, never an input.",
            },
          ],
          insight:
            "<strong>Noise is a first-class answer.</strong> k-means must put every point somewhere; DBSCAN is allowed to say “this one belongs to nothing”, which is exactly what you want for outlier detection.",
        },
      ]);

      U.renderSteps(
        stepsNode,
        events.map((event) => {
          if (event.type === "visit") return `<strong>Visit ${event.pointId}.</strong> Query its ε-neighbourhood and test the core condition.`;
          if (event.type === "noise") return `<strong>Noise.</strong> ${event.pointId} does not have enough nearby points.`;
          if (event.type === "start") return `<strong>Start cluster ${event.clusterId}.</strong> ${event.pointId} is dense enough to seed a cluster.`;
          return `<strong>Expand cluster ${event.clusterId}.</strong> ${event.pointId} is density-reachable and gets assigned.`;
        }),
        revealed
      );

      const xs = points.map((point) => point.x);
      const ys = points.map((point) => point.y);
      const chart = U.makeChart(plot, {
        xDomain: [Math.min(...xs) - 1, Math.max(...xs) + 1],
        yDomain: [Math.min(...ys) - 1, Math.max(...ys) + 1],
        title: "DBSCAN growth",
      });

      /* Faint ε-discs on every core point make the density criterion
         visible across the whole plot, not just at the cursor. */
      points.forEach((point) => {
        if (state[point.id] && state[point.id].type === "core") {
          plot.appendChild(
            U.svgEl("ellipse", {
              cx: chart.xScale(point.x),
              cy: chart.yScale(point.y),
              rx: Math.abs(chart.xScale(point.x + eps) - chart.xScale(point.x)),
              ry: Math.abs(chart.yScale(point.y + eps) - chart.yScale(point.y)),
              fill: U.tint(C().a, 0.05),
              stroke: U.tint(C().a, 0.22),
              "stroke-width": "1",
            })
          );
        }
      });

      if (activePoint) {
        plot.appendChild(
          U.svgEl("ellipse", {
            cx: chart.xScale(activePoint.x),
            cy: chart.yScale(activePoint.y),
            rx: Math.abs(chart.xScale(activePoint.x + eps) - chart.xScale(activePoint.x)),
            ry: Math.abs(chart.yScale(activePoint.y + eps) - chart.yScale(activePoint.y)),
            fill: U.tint(C().a, 0.1),
            stroke: C().a,
            "stroke-width": "2",
            "stroke-dasharray": "6 6",
          })
        );
      }

      points.forEach((point, index) => {
        const item = state[point.id] || { clusterId: null, type: "unvisited" };
        const color =
          item.clusterId !== null
            ? clusterPaletteAt((item.clusterId - 1))
            : item.type === "noise"
            ? C().danger
            : C().neutral;
        drawScatterPoint(plot, chart, point, color, active === point.id ? 9 : 7, (target, position) => {
          target.x = Number(U.round(position.x, 2));
          target.y = Number(U.round(position.y, 2));
          editor.setCell(index, "x", target.x);
          editor.setCell(index, "y", target.y);
          revealed = 0;
          scheduleRender();
        });
      });
    }

    const scheduleRender = U.rafThrottle(render);

    stepButton.addEventListener("click", () => {
      const events = buildDbscanEvents(points, Number(epsInput.value) || 1.2, Math.max(Number(minPtsInput.value) || 3, 2));
      revealed = Math.min(revealed + 1, events.length);
      render();
    });
    runButton.addEventListener("click", () => {
      revealed = buildDbscanEvents(points, Number(epsInput.value) || 1.2, Math.max(Number(minPtsInput.value) || 3, 2)).length;
      render();
    });
    resetButton.addEventListener("click", () => {
      revealed = 0;
      render();
    });
    [epsInput, minPtsInput].forEach((input) =>
      input.addEventListener("input", () => {
        revealed = 0;
        render();
      })
    );
    U.onRedraw(render);
    render();
  }

  function mountDistance(rootNode) {
    mountSection(
      rootNode,
      "Distance measure comparer",
      "The same three candidates, ranked three different ways. Drag the query around and watch the winner change as you switch metrics — this is why the choice of distance is a modelling decision, not a detail.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-grid">
              <div class="control-group">
                <label for="distance-metric">Metric</label>
                <select id="distance-metric">
                  <option value="euclidean">Euclidean (L2)</option>
                  <option value="manhattan">Manhattan (L1)</option>
                  <option value="cosine">Cosine distance</option>
                </select>
              </div>
              <div class="control-group">
                <label for="distance-qx">Query x</label>
                <input id="distance-qx" type="number" step="0.1" value="3.2" />
              </div>
              <div class="control-group">
                <label for="distance-qy">Query y</label>
                <input id="distance-qy" type="number" step="0.1" value="4.1" />
              </div>
            </div>
            <div class="step-controls">
              <button class="button primary" id="distance-step">Next step</button>
              <button class="button secondary" id="distance-reset">Reset steps</button>
            </div>
            ${editorSlot("distance-data")}
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg id="distance-plot" viewBox="0 0 560 280" aria-label="Distance plot"></svg>
              ${DRAG_HINT}
            </div>
            <div class="readout"><div class="output-grid" id="distance-output"></div></div>
            <div class="equation-card" id="distance-math"></div>
            <div class="table-panel">
              <table>
                <thead><tr><th>Candidate</th><th>Euclidean</th><th>Manhattan</th><th>Cosine</th></tr></thead>
                <tbody id="distance-table"></tbody>
              </table>
            </div>
            <div class="steps"><ol id="distance-steps"></ol></div>
          </div>
        </div>
      `
    );

    const metricInput = rootNode.querySelector("#distance-metric");
    const qxInput = rootNode.querySelector("#distance-qx");
    const qyInput = rootNode.querySelector("#distance-qy");
    const stepButton = rootNode.querySelector("#distance-step");
    const resetButton = rootNode.querySelector("#distance-reset");
    const plot = rootNode.querySelector("#distance-plot");
    const output = rootNode.querySelector("#distance-output");
    const mathNode = rootNode.querySelector("#distance-math");
    const stepsNode = rootNode.querySelector("#distance-steps");
    const tableNode = rootNode.querySelector("#distance-table");

    let candidates = [
      { id: "A", x: 6.0, y: 6.2 },
      { id: "B", x: 7.5, y: 3.9 },
      { id: "C", x: 2.2, y: 7.4 },
    ];
    let revealed = 0;

    const editor = pointsEditor(
      rootNode.querySelector("#distance-data"),
      candidates,
      (rows) => {
        candidates = rows;
        revealed = 0;
        render();
      },
      {
        title: "Candidate points",
        hint: "Try placing two candidates along the same ray from the origin at different lengths — Euclidean will separate them, cosine will call them identical.",
        minRows: 2,
        presets: {
          "Spread out": [
            { id: "A", x: 6.0, y: 6.2 },
            { id: "B", x: 7.5, y: 3.9 },
            { id: "C", x: 2.2, y: 7.4 },
          ],
          "Same direction, different length": [
            { id: "A", x: 2.0, y: 2.0 },
            { id: "B", x: 5.0, y: 5.0 },
            { id: "C", x: 8.0, y: 8.0 },
          ],
          "Axis aligned": [
            { id: "A", x: 8.0, y: 4.1 },
            { id: "B", x: 3.2, y: 8.0 },
            { id: "C", x: 3.2, y: 1.0 },
          ],
        },
      }
    );

    function distanceMetric(a, b, metric) {
      if (metric === "cosine") {
        const denom = Math.hypot(a.x, a.y) * Math.hypot(b.x, b.y) || 1;
        return 1 - (a.x * b.x + a.y * b.y) / denom;
      }
      return U.distance(a, b, metric);
    }

    function render() {
      const metric = metricInput.value;
      const query = { id: "Q", x: Number(qxInput.value) || 0, y: Number(qyInput.value) || 0 };
      const scored = candidates
        .map((point) => ({ ...point, distance: distanceMetric(point, query, metric) }))
        .sort((a, b) => a.distance - b.distance);

      if (!scored.length) return;

      U.renderMetrics(output, [
        { label: "Metric", value: metric },
        { label: "Nearest", value: scored[0].id },
        { label: `${scored[0].id} score`, value: U.round(scored[0].distance, 3) },
        { label: "Ranking", value: scored.map((item) => item.id).join(" < ") },
      ]);

      tableNode.innerHTML = candidates
        .map((point) => {
          const isNearest = point.id === scored[0].id;
          return `
            <tr style="${isNearest ? "background:rgba(31,122,112,0.08);font-weight:600" : ""}">
              <td>${point.id}</td>
              <td>${U.round(distanceMetric(point, query, "euclidean"), 3)}</td>
              <td>${U.round(distanceMetric(point, query, "manhattan"), 3)}</td>
              <td>${U.round(distanceMetric(point, query, "cosine"), 3)}</td>
            </tr>
          `;
        })
        .join("");

      const nearest = scored[0];
      const dx = nearest.x - query.x;
      const dy = nearest.y - query.y;
      const metricTex =
        metric === "euclidean"
          ? "d(\\mathbf{x}, \\mathbf{q}) = \\sqrt{(x_1 - q_1)^2 + (x_2 - q_2)^2}"
          : metric === "manhattan"
          ? "d(\\mathbf{x}, \\mathbf{q}) = |x_1 - q_1| + |x_2 - q_2|"
          : "d_{\\cos}(\\mathbf{x}, \\mathbf{q}) = 1 - \\frac{\\mathbf{x} \\cdot \\mathbf{q}}{\\lVert \\mathbf{x}\\rVert\\,\\lVert \\mathbf{q}\\rVert}";

      const derivation = [];
      if (metric === "cosine") {
        const dotValue = nearest.x * query.x + nearest.y * query.y;
        const normX = Math.hypot(nearest.x, nearest.y);
        const normQ = Math.hypot(query.x, query.y);
        derivation.push({ tex: `\\mathbf{x} \\cdot \\mathbf{q} = ${U.texNum(dotValue, 3)}`, result: U.round(dotValue, 3) });
        derivation.push({
          tex: `\\lVert \\mathbf{x} \\rVert = ${U.texNum(normX, 3)}, \\quad \\lVert \\mathbf{q} \\rVert = ${U.texNum(normQ, 3)}`,
          result: "",
        });
        derivation.push({
          tex: `d_{\\cos} = 1 - \\frac{${U.texNum(dotValue, 3)}}{${U.texNum(normX, 3)} \\times ${U.texNum(normQ, 3)}}`,
          result: U.round(nearest.distance, 4),
          note: "Cosine ignores magnitude entirely — only the angle between the vectors matters.",
        });
      } else {
        derivation.push({
          tex: `\\Delta x = ${U.texNum(dx, 2)}, \\quad \\Delta y = ${U.texNum(dy, 2)}`,
          result: "",
          note: `Worked through for the winner, ${nearest.id}.`,
        });
        derivation.push({
          tex:
            metric === "euclidean"
              ? `d = \\sqrt{(${U.texNum(dx, 2)})^2 + (${U.texNum(dy, 2)})^2} = \\sqrt{${U.texNum(dx * dx + dy * dy, 3)}}`
              : `d = |${U.texNum(dx, 2)}| + |${U.texNum(dy, 2)}|`,
          result: U.round(nearest.distance, 4),
        });
      }

      renderFormulaCards(mathNode, [
        {
          title: "Metric definition",
          description:
            metric === "euclidean"
              ? "Straight-line distance. Circles are the level sets — every direction is treated identically."
              : metric === "manhattan"
              ? "Sum of axis-aligned moves. Level sets are diamonds, so it is more forgiving of a large error in one dimension."
              : "Angle between vectors, measured from the origin. Two points on the same ray have distance zero no matter how far apart they are.",
          tex: metricTex,
          derivation,
          insight:
            metric === "cosine"
              ? "<strong>Cosine ignores length.</strong> Load the “Same direction, different length” preset: all three candidates collapse to distance ≈ 0, even though they are far apart in the plane. This is why cosine is the default for text embeddings, where document length should not matter."
              : "<strong>Level sets tell the story.</strong> L2 draws circles, L1 draws diamonds. Switch between them with the query near a diagonal and watch the winner flip.",
        },
        {
          title: "Why the ranking changes",
          description: "All three metrics are computed for every candidate in the table on the right — compare the columns.",
          tex: "\\text{nearest} = \\operatorname*{arg\\,min}_{i} \\; d(\\mathbf{x}_i, \\mathbf{q})",
          derivation: scored.map((point, index) => ({
            tex: `\\text{rank } ${index + 1}: \\text{${point.id}} \\;\\to\\; ${U.texNum(point.distance, 3)}`,
            result: index === 0 ? "nearest" : "",
          })),
          insight:
            "<strong>Everything downstream inherits this choice.</strong> k-NN, k-means and DBSCAN all take a distance function as an argument — change it and every one of them gives different answers on the same data.",
        },
      ]);

      U.renderSteps(
        stepsNode,
        [
          `<strong>Compute coordinate differences.</strong> For candidate ${nearest.id}, Δx = ${U.round(dx, 2)}, Δy = ${U.round(dy, 2)}.`,
          `<strong>Aggregate according to the metric.</strong> The chosen metric is <strong>${metric}</strong>.`,
          `<strong>Rank the candidates.</strong> The current order is ${scored.map((item) => `${item.id} (${U.round(item.distance, 2)})`).join(" < ")}.`,
        ],
        revealed
      );

      const chart = U.makeChart(plot, { xDomain: [0, 10], yDomain: [0, 10], title: "Metric-dependent nearest neighbour" });

      /* Draw the level set through the winning candidate so the shape of
         the metric is literally visible. */
      const radius = nearest.distance;
      if (metric === "euclidean") {
        plot.appendChild(
          U.svgEl("ellipse", {
            cx: chart.xScale(query.x),
            cy: chart.yScale(query.y),
            rx: Math.abs(chart.xScale(radius) - chart.xScale(0)),
            ry: Math.abs(chart.yScale(0) - chart.yScale(radius)),
            fill: U.tint(C().ink, 0.05),
            stroke: U.tint(C().ink, 0.4),
            "stroke-dasharray": "5 5",
          })
        );
      } else if (metric === "manhattan") {
        const rx = Math.abs(chart.xScale(radius) - chart.xScale(0));
        const ry = Math.abs(chart.yScale(0) - chart.yScale(radius));
        const cx = chart.xScale(query.x);
        const cy = chart.yScale(query.y);
        plot.appendChild(
          U.svgEl("polygon", {
            points: `${cx},${cy - ry} ${cx + rx},${cy} ${cx},${cy + ry} ${cx - rx},${cy}`,
            fill: U.tint(C().ink, 0.05),
            stroke: U.tint(C().ink, 0.4),
            "stroke-dasharray": "5 5",
          })
        );
      } else {
        /* For cosine, the meaningful guide is the ray from the origin. */
        plot.appendChild(
          U.svgEl("line", {
            x1: chart.xScale(0),
            y1: chart.yScale(0),
            x2: chart.xScale(query.x * 3),
            y2: chart.yScale(query.y * 3),
            stroke: U.tint(C().ink, 0.4),
            "stroke-dasharray": "5 5",
          })
        );
      }

      candidates.forEach((point, index) => {
        drawScatterPoint(
          plot,
          chart,
          point,
          scored[0].id === point.id ? C().a : C().b,
          scored[0].id === point.id ? 9 : 7,
          (target, position) => {
            target.x = Number(U.round(position.x, 2));
            target.y = Number(U.round(position.y, 2));
            editor.setCell(index, "x", target.x);
            editor.setCell(index, "y", target.y);
            scheduleRender();
          }
        );
      });

      const qx = chart.xScale(query.x);
      const qy = chart.yScale(query.y);
      const marker = U.svgEl("polygon", {
        points: `${qx},${qy - 11} ${qx + 11},${qy} ${qx},${qy + 11} ${qx - 11},${qy}`,
        fill: C().ink,
      });
      plot.appendChild(marker);
      U.draggable(marker, plot, chart, (position) => {
        qxInput.value = U.round(position.x, 2);
        qyInput.value = U.round(position.y, 2);
        scheduleRender();
      });
      plot.appendChild(U.svgEl("text", { x: qx + 14, y: qy - 12, class: "svg-label" })).textContent = "Query (drag me)";
    }

    const scheduleRender = U.rafThrottle(render);

    stepButton.addEventListener("click", () => {
      revealed = Math.min(revealed + 1, 3);
      render();
    });
    resetButton.addEventListener("click", () => {
      revealed = 0;
      render();
    });
    [metricInput, qxInput, qyInput].forEach((input) =>
      input.addEventListener("input", () => {
        revealed = 0;
        render();
      })
    );
    U.onRedraw(render);
    render();
  }

  function mountSpectral(rootNode) {
    mountSection(
      rootNode,
      "Spectral clustering and graph Laplacians",
      "Turn the points into a weighted graph, build its Laplacian, and let the second-smallest eigenvector decide the cut. Edit the points to see a shape that defeats k-means get split cleanly here.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-grid">
              <div class="control-group">
                <label for="spectral-sigma">Similarity width σ</label>
                <div class="range-row">
                  <input id="spectral-sigma" type="range" min="0.3" max="4" step="0.1" value="1" />
                  <span class="range-value" id="spectral-sigma-value">1.0</span>
                </div>
              </div>
              <div class="control-group">
                <label for="spectral-threshold">Edge threshold</label>
                <div class="range-row">
                  <input id="spectral-threshold" type="range" min="0.01" max="0.8" step="0.01" value="0.2" />
                  <span class="range-value" id="spectral-threshold-value">0.20</span>
                </div>
              </div>
            </div>
            <div class="callout" id="spectral-callout"></div>
            ${editorSlot("spectral-data")}
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg id="spectral-plot" viewBox="0 0 560 280" aria-label="Spectral clustering plot"></svg>
              ${DRAG_HINT}
            </div>
            <div class="plot-card">
              <svg id="spectral-bars" viewBox="0 0 560 280" aria-label="Fiedler vector plot"></svg>
            </div>
            <div class="equation-card" id="spectral-math"></div>
            <div class="table-panel" id="spectral-matrix"></div>
            <div class="steps"><ol id="spectral-steps"></ol></div>
          </div>
        </div>
      `
    );

    const sigmaInput = rootNode.querySelector("#spectral-sigma");
    const sigmaValue = rootNode.querySelector("#spectral-sigma-value");
    const thresholdInput = rootNode.querySelector("#spectral-threshold");
    const thresholdValue = rootNode.querySelector("#spectral-threshold-value");
    const callout = rootNode.querySelector("#spectral-callout");
    const plot = rootNode.querySelector("#spectral-plot");
    const bars = rootNode.querySelector("#spectral-bars");
    const mathNode = rootNode.querySelector("#spectral-math");
    const matrixNode = rootNode.querySelector("#spectral-matrix");
    const stepsNode = rootNode.querySelector("#spectral-steps");

    let points = spectralPoints.map((point) => ({ ...point }));

    const editor = pointsEditor(
      rootNode.querySelector("#spectral-data"),
      points,
      (rows) => {
        points = rows;
        render();
      },
      {
        title: "Graph nodes",
        hint: "Keep this to roughly 6–10 nodes so the similarity matrix stays readable. Two well-separated groups produce a Fiedler vector with a clean sign split.",
        minRows: 3,
      }
    );

    function render() {
      const sigma = Math.max(Number(sigmaInput.value) || 1, 0.2);
      const threshold = Number(thresholdInput.value) || 0.2;
      sigmaValue.textContent = sigma.toFixed(1);
      thresholdValue.textContent = threshold.toFixed(2);

      const n = points.length;
      if (n < 2) return;

      const W = Array.from({ length: n }, () => Array(n).fill(0));
      for (let i = 0; i < n; i += 1) {
        for (let j = i + 1; j < n; j += 1) {
          const distance = U.distance(points[i], points[j], "euclidean");
          const weight = Math.exp(-(distance * distance) / (2 * sigma * sigma));
          if (weight >= threshold) {
            W[i][j] = weight;
            W[j][i] = weight;
          }
        }
      }
      const D = W.map((row) => row.reduce((acc, value) => acc + value, 0));
      const L = W.map((row, i) => row.map((value, j) => (i === j ? D[i] : 0) - value));
      const { eigenvalues, eigenvectors } = U.jacobiEigen(L);
      const order = eigenvalues.map((value, index) => ({ value, index })).sort((a, b) => a.value - b.value);
      const fiedler = eigenvectors[order[Math.min(1, order.length - 1)].index];
      const cut = fiedler.some((value) => value > 0) && fiedler.some((value) => value < 0) ? 0 : U.median(fiedler);
      const assignments = fiedler.map((value) => (value >= cut ? 1 : 0));
      const edgeCount = W.reduce((acc, row, i) => acc + row.filter((value, j) => j > i && value > 0).length, 0);

      /* The cut value: total weight of edges crossing the partition. */
      let cutWeight = 0;
      for (let i = 0; i < n; i += 1) {
        for (let j = i + 1; j < n; j += 1) {
          if (assignments[i] !== assignments[j]) cutWeight += W[i][j];
        }
      }

      callout.innerHTML = `Cut threshold <strong>${U.round(cut, 3)}</strong> — the sign pattern of the Fiedler vector defines the partition. λ₂ = <strong>${U.round(order[Math.min(1, order.length - 1)].value, 4)}</strong>.`;

      renderFormulaCards(mathNode, [
        {
          title: "Similarity graph",
          description: "Every pair of points gets an edge weight that decays with distance. Weights below the threshold are dropped, which is what makes the graph sparse.",
          tex: `W_{ij} = \\exp\\!\\left(-\\frac{\\lVert \\mathbf{x}_i - \\mathbf{x}_j \\rVert^2}{2\\sigma^2}\\right), \\quad \\sigma = ${U.texNum(sigma, 2)}`,
          derivation: [
            {
              tex: `\\text{edges kept} = ${edgeCount} \\;\\text{ of }\\; ${(n * (n - 1)) / 2}`,
              result: String(edgeCount),
              note: "Raise σ and distant points start to count as similar; the graph fills in and the split gets mushy.",
            },
          ],
          insight:
            "<strong>σ sets the notion of “near”.</strong> It plays the same role ε plays in DBSCAN — it is the scale at which you have decided the data lives.",
        },
        {
          title: "Graph Laplacian",
          description: "D is the diagonal matrix of node degrees. L = D − W encodes connectivity in a form whose eigenvalues reveal how hard the graph is to cut.",
          tex: "L = D - W, \\qquad D_{ii} = \\sum_j W_{ij}",
          derivation: [
            {
              tex: `\\lambda_1 = ${U.texNum(order[0].value, 4)} \\approx 0`,
              result: U.round(order[0].value, 4),
              note: "The smallest eigenvalue of a Laplacian is always 0 — its eigenvector is constant and carries no clustering information.",
            },
            {
              tex: `\\lambda_2 = ${U.texNum(order[Math.min(1, order.length - 1)].value, 4)}`,
              result: U.round(order[Math.min(1, order.length - 1)].value, 4),
              note: "λ₂ is the algebraic connectivity. Near zero means the graph is nearly disconnected — an easy, natural split.",
            },
          ],
          insight:
            "<strong>Eigenvalues count components.</strong> The number of zero eigenvalues equals the number of connected components. Drag the threshold up until a group detaches and watch λ₂ collapse toward 0.",
        },
        {
          title: "Fiedler split",
          description: "The eigenvector of the second-smallest eigenvalue assigns each node a coordinate; the sign of that coordinate is the cluster.",
          tex: "\\mathbf{v}_2 = \\operatorname*{arg\\,min}_{\\mathbf{v} \\perp \\mathbf{1}} \\frac{\\mathbf{v}^{\\top} L \\mathbf{v}}{\\mathbf{v}^{\\top}\\mathbf{v}}",
          derivation: [
            ...points.map((point, index) => ({
              tex: `v_2[\\text{${point.id}}] = ${U.texNum(fiedler[index], 3)}`,
              result: `cluster ${assignments[index] + 1}`,
            })),
            {
              tex: `\\text{cut weight} = ${U.texNum(cutWeight, 4)}`,
              result: U.round(cutWeight, 4),
              note: "Total weight of the edges the partition severs — the quantity the relaxation is minimising.",
            },
          ],
          insight:
            "<strong>This is why moons work.</strong> Spectral clustering never asks whether a cluster is round; it only asks whether the graph is hard to cut. Two interlocking arcs are trivially separable as a graph even though no straight line divides them.",
        },
      ]);

      /* Similarity matrix — the object the eigendecomposition acts on. */
      if (n <= 12) {
        matrixNode.innerHTML = `
          <table>
            <thead><tr><th>W</th>${points.map((point) => `<th>${point.id}</th>`).join("")}</tr></thead>
            <tbody>
              ${points
                .map(
                  (rowPoint, i) => `
                    <tr>
                      <th>${rowPoint.id}</th>
                      ${points
                        .map((_, j) => {
                          if (i === j) return `<td style="opacity:.3">—</td>`;
                          const value = W[i][j];
                          const intensity = Math.min(value, 1);
                          return `<td style="background:rgba(26,122,110,${0.25 * intensity})">${value > 0 ? U.round(value, 2) : "·"}</td>`;
                        })
                        .join("")}
                    </tr>
                  `
                )
                .join("")}
            </tbody>
          </table>
        `;
      } else {
        matrixNode.innerHTML = `<p class="caption">The similarity matrix is shown for up to 12 nodes.</p>`;
      }

      U.renderSteps(
        stepsNode,
        [
          `<strong>Build W.</strong> Keep graph edges whose weight is at least <strong>${U.round(threshold, 2)}</strong> — ${edgeCount} edge${edgeCount !== 1 ? "s" : ""} survive.`,
          "<strong>Build D and L.</strong> The degree matrix D and the Laplacian L = D − W summarise graph connectivity.",
          `<strong>Compute the Fiedler vector.</strong> The eigenvector for λ₂ = ${U.round(order[Math.min(1, order.length - 1)].value, 4)} captures the low-cut direction.`,
          "<strong>Split by sign.</strong> Positive and negative coordinates define the two spectral clusters.",
        ],
        4
      );

      const xs = points.map((point) => point.x);
      const ys = points.map((point) => point.y);
      const chart = U.makeChart(plot, {
        xDomain: [Math.min(...xs) - 1, Math.max(...xs) + 1],
        yDomain: [Math.min(...ys) - 1, Math.max(...ys) + 1],
        title: "Similarity graph partition",
      });
      for (let i = 0; i < n; i += 1) {
        for (let j = i + 1; j < n; j += 1) {
          if (W[i][j] > 0) {
            plot.appendChild(
              U.svgEl("line", {
                x1: chart.xScale(points[i].x),
                y1: chart.yScale(points[i].y),
                x2: chart.xScale(points[j].x),
                y2: chart.yScale(points[j].y),
                stroke: assignments[i] !== assignments[j] ? C().danger : C().ink,
                "stroke-width": 1 + 4 * W[i][j],
                opacity: assignments[i] !== assignments[j] ? "0.5" : "0.22",
              })
            );
          }
        }
      }
      points.forEach((point, index) => {
        drawScatterPoint(plot, chart, point, clusterPaletteAt(assignments[index]), 8, (target, position) => {
          target.x = Number(U.round(position.x, 2));
          target.y = Number(U.round(position.y, 2));
          editor.setCell(index, "x", target.x);
          editor.setCell(index, "y", target.y);
          scheduleRender();
        });
      });

      const yMin = Math.min(...fiedler) - 0.1;
      const yMax = Math.max(...fiedler) + 0.1;
      const barChart = U.makeChart(bars, {
        xDomain: [0, n + 1],
        yDomain: [yMin, yMax],
        title: "Fiedler vector coordinates (sign = cluster)",
      });
      bars.appendChild(
        U.svgEl("line", {
          x1: barChart.xScale(0),
          y1: barChart.yScale(cut),
          x2: barChart.xScale(n + 1),
          y2: barChart.yScale(cut),
          stroke: C().danger,
          "stroke-dasharray": "5 5",
        })
      );
      fiedler.forEach((value, index) => {
        const x1 = barChart.xScale(index + 0.45);
        const x2 = barChart.xScale(index + 1.05);
        const zeroY = barChart.yScale(cut);
        const y = barChart.yScale(value);
        bars.appendChild(
          U.svgEl("rect", {
            x: x1,
            y: Math.min(y, zeroY),
            width: x2 - x1,
            height: Math.abs(zeroY - y),
            fill: clusterPaletteAt(assignments[index]),
            rx: "6",
          })
        );
        bars.appendChild(
          U.svgEl("text", { x: 0.5 * (x1 + x2), y: zeroY + 18, class: "svg-label", "text-anchor": "middle" })
        ).textContent = points[index].id;
      });
    }

    const scheduleRender = U.rafThrottle(render);

    [sigmaInput, thresholdInput].forEach((input) => input.addEventListener("input", render));
    U.onRedraw(render);
    render();
  }


  /* ══════════════════════════════════════════════════════════════
     NEURAL NETWORK ENGINES
     ══════════════════════════════════════════════════════════════ */

  const activations = {
    sigmoid: {
      label: "Sigmoid",
      fn: (z) => U.sigmoid(z),
      dz: (z) => {
        const s = U.sigmoid(z);
        return s * (1 - s);
      },
      tex: "\\sigma(z) = \\dfrac{1}{1 + e^{-z}}",
      dTex: "\\sigma'(z) = \\sigma(z)\\,(1 - \\sigma(z))",
      range: "(0, 1)",
      peak: 0.25,
      note: "Squashes into (0, 1). The derivative peaks at 0.25, so every layer shrinks the gradient by at least a factor of four.",
    },
    tanh: {
      label: "Tanh",
      fn: (z) => Math.tanh(z),
      dz: (z) => 1 - Math.tanh(z) ** 2,
      tex: "\\tanh(z) = \\dfrac{e^{z} - e^{-z}}{e^{z} + e^{-z}}",
      dTex: "\\tanh'(z) = 1 - \\tanh^2(z)",
      range: "(−1, 1)",
      peak: 1,
      note: "Zero-centred, with a peak derivative of 1 — strictly better behaved than sigmoid for hidden layers, though it still saturates.",
    },
    relu: {
      label: "ReLU",
      fn: (z) => Math.max(0, z),
      dz: (z) => (z > 0 ? 1 : 0),
      tex: "\\mathrm{ReLU}(z) = \\max(0, z)",
      dTex: "\\mathrm{ReLU}'(z) = \\mathbb{1}[z > 0]",
      range: "[0, ∞)",
      peak: 1,
      note: "Passes positive gradients through untouched, which is what makes very deep networks trainable. Negative inputs give exactly zero — a unit stuck there is dead.",
    },
    leaky: {
      label: "Leaky ReLU",
      fn: (z) => (z > 0 ? z : 0.1 * z),
      dz: (z) => (z > 0 ? 1 : 0.1),
      tex: "f(z) = \\begin{cases} z & z > 0 \\\\ 0.1z & z \\le 0 \\end{cases}",
      dTex: "f'(z) = \\begin{cases} 1 & z > 0 \\\\ 0.1 & z \\le 0 \\end{cases}",
      range: "(−∞, ∞)",
      peak: 1,
      note: "A small negative slope keeps dead units alive — the gradient is never exactly zero, so a unit can always recover.",
    },
  };

  function mountActivation(rootNode) {
    mountSection(
      rootNode,
      "Activation functions and their gradients",
      "The top chart is the function; the bottom is its derivative — and the derivative is what actually decides whether a deep network trains. Move the input marker and read the exact values, then compare what happens to a gradient after passing through several layers.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-group">
              <label for="act-fn">Activation</label>
              <select id="act-fn">
                <option value="sigmoid">Sigmoid</option>
                <option value="tanh">Tanh</option>
                <option value="relu">ReLU</option>
                <option value="leaky">Leaky ReLU</option>
              </select>
            </div>
            <div class="control-group">
              <label for="act-z">Input z</label>
              <div class="range-row">
                <input id="act-z" type="range" min="-6" max="6" step="0.1" value="1.2" />
                <span class="range-value" id="act-z-value">1.20</span>
              </div>
            </div>
            <div class="control-group">
              <label for="act-layers">Depth to simulate (layers)</label>
              <div class="range-row">
                <input id="act-layers" type="range" min="1" max="12" step="1" value="6" />
                <span class="range-value" id="act-layers-value">6</span>
              </div>
            </div>
            <div class="control-group">
              <label for="act-compare">
                <input type="checkbox" id="act-compare" style="width:auto;margin-right:6px" checked /> Overlay all four
              </label>
            </div>
            <div class="callout" id="act-callout"></div>
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg id="act-plot" viewBox="0 0 560 260" aria-label="Activation function"></svg>
              <div class="legend" id="act-legend"></div>
            </div>
            <div class="plot-card">
              <svg id="act-deriv" viewBox="0 0 560 220" aria-label="Activation derivative"></svg>
            </div>
            <div class="readout"><div class="output-grid" id="act-output"></div></div>
            <div class="equation-card" id="act-math"></div>
            <div class="table-panel">
              <table>
                <thead><tr><th>Function</th><th>Range</th><th>f(z)</th><th>f′(z)</th><th>Max f′</th></tr></thead>
                <tbody id="act-table"></tbody>
              </table>
            </div>
            <div class="steps"><ol id="act-steps"></ol></div>
          </div>
        </div>
      `
    );

    const fnInput = rootNode.querySelector("#act-fn");
    const zInput = rootNode.querySelector("#act-z");
    const zValue = rootNode.querySelector("#act-z-value");
    const layersInput = rootNode.querySelector("#act-layers");
    const layersValue = rootNode.querySelector("#act-layers-value");
    const compareInput = rootNode.querySelector("#act-compare");
    const callout = rootNode.querySelector("#act-callout");
    const plot = rootNode.querySelector("#act-plot");
    const deriv = rootNode.querySelector("#act-deriv");
    const legend = rootNode.querySelector("#act-legend");
    const output = rootNode.querySelector("#act-output");
    const mathNode = rootNode.querySelector("#act-math");
    const tableNode = rootNode.querySelector("#act-table");
    const stepsNode = rootNode.querySelector("#act-steps");

    const colors = { sigmoid: C().a, tanh: C().c, relu: C().b, leaky: C().d };

    function render() {
      const key = fnInput.value;
      const active = activations[key];
      const z = Number(zInput.value);
      const layers = Number(layersInput.value);
      zValue.textContent = z.toFixed(2);
      layersValue.textContent = String(layers);

      const value = active.fn(z);
      const slope = active.dz(z);
      const compounded = Math.pow(slope, layers);

      /* Tiny values must not be rounded to a flat "0" — the whole point of
         this page is how small they get, so switch to scientific notation. */
      const fmtSmall = (value) =>
        value === 0 ? "0" : Math.abs(value) < 1e-4 ? value.toExponential(2) : U.round(value, 6);

      callout.innerHTML = `After ${layers} layer${layers !== 1 ? "s" : ""}, a gradient passing through this activation at z = ${U.round(z, 2)} is multiplied by <strong>${fmtSmall(compounded)}</strong>${compounded !== 0 && compounded < 1e-4 ? " — effectively nothing reaches the early layers" : ""}.`;

      U.renderMetrics(output, [
        { label: "f(z)", value: U.round(value, 4) },
        { label: "f′(z)", value: U.round(slope, 4) },
        { label: `f′(z)^${layers}`, value: fmtSmall(compounded) },
        { label: "Output range", value: active.range },
      ]);

      tableNode.innerHTML = Object.entries(activations)
        .map(
          ([id, entry]) => `
            <tr style="${id === key ? "background:var(--accent-soft);font-weight:600" : ""}">
              <td>${entry.label}</td>
              <td>${entry.range}</td>
              <td>${U.round(entry.fn(z), 4)}</td>
              <td>${U.round(entry.dz(z), 4)}</td>
              <td>${entry.peak}</td>
            </tr>
          `
        )
        .join("");

      renderFormulaCards(mathNode, [
        {
          title: `${active.label} — the function`,
          description: active.note,
          tex: active.tex,
          derivation: [
            {
              tex: `f(${U.texNum(z, 2)}) = ${U.texNum(value, 4)}`,
              result: U.round(value, 4),
              note: `Output range is ${active.range}.`,
            },
          ],
          insight:
            "<strong>Why any non-linearity at all?</strong> Two stacked linear layers give W₂(W₁x) = (W₂W₁)x — a single matrix. Without a non-linearity between them, a hundred layers have exactly the expressive power of one.",
        },
        {
          title: `${active.label} — the derivative`,
          description:
            "This is the number backpropagation multiplies by at every layer. Its size, compounded over depth, is the whole vanishing-gradient story.",
          tex: active.dTex,
          derivation: [
            { tex: `f'(${U.texNum(z, 2)}) = ${U.texNum(slope, 4)}`, result: U.round(slope, 4) },
            {
              tex: `\\left(${U.texNum(slope, 4)}\\right)^{${layers}} = ${fmtSmall(compounded)}`,
              result: fmtSmall(compounded),
              note: `The factor a gradient picks up crossing ${layers} such layers. Switch to sigmoid at a large |z| and watch this collapse.`,
            },
          ],
          insight:
            active.peak <= 0.25
              ? "<strong>This is why sigmoid fell out of favour for hidden layers.</strong> A maximum derivative of 0.25 means the gradient loses at least 75% of its magnitude per layer — after ten layers, less than one part in a million survives."
              : "<strong>A derivative of 1 is the ideal.</strong> The gradient crosses the layer unchanged, which is exactly why ReLU-family activations and residual connections made very deep networks trainable.",
        },
      ]);

      U.renderSteps(
        stepsNode,
        [
          `<strong>Pick the activation.</strong> Currently <strong>${active.label}</strong>, mapping into ${active.range}.`,
          `<strong>Apply it forward.</strong> f(${U.round(z, 2)}) = <strong>${U.round(value, 4)}</strong>.`,
          `<strong>Read the local gradient.</strong> f′(${U.round(z, 2)}) = <strong>${U.round(slope, 4)}</strong> — this is the factor backpropagation uses here.`,
          `<strong>Compound over depth.</strong> Across ${layers} layers the gradient is scaled by <strong>${fmtSmall(compounded)}</strong>.`,
        ],
        4
      );

      /* ── function chart ── */
      const xs = U.linspace(-6, 6, 220);
      const shown = compareInput.checked ? Object.keys(activations) : [key];
      const allValues = shown.flatMap((id) => xs.map((x) => activations[id].fn(x)));
      const yMin = Math.min(-1.2, ...allValues) - 0.2;
      const yMax = Math.max(1.2, ...allValues.filter((v) => v < 7)) + 0.4;

      const chart = U.makeChart(plot, { xDomain: [-6, 6], yDomain: [yMin, yMax], title: "Activation f(z)" });
      plot.appendChild(
        U.svgEl("line", { x1: chart.xScale(-6), y1: chart.yScale(0), x2: chart.xScale(6), y2: chart.yScale(0), class: "axis-line" })
      );
      shown.forEach((id) => {
        const curve = xs.map((x) => ({ x, y: U.clamp(activations[id].fn(x), yMin, yMax) }));
        plot.appendChild(
          U.svgEl("path", {
            d: U.pathFromPoints(curve, chart.xScale, chart.yScale),
            fill: "none",
            stroke: colors[id],
            "stroke-width": id === key ? "3" : "1.6",
            opacity: id === key ? "1" : "0.4",
          })
        );
      });
      const marker = U.svgEl("circle", {
        cx: chart.xScale(z),
        cy: chart.yScale(U.clamp(value, yMin, yMax)),
        r: 7,
        fill: colors[key],
        stroke: C().ring,
        "stroke-width": "2.5",
      });
      plot.appendChild(marker);
      U.draggable(marker, plot, chart, (position) => {
        zInput.value = U.round(position.x, 2);
        scheduleRender();
      });
      plot.appendChild(
        U.svgEl("line", {
          x1: chart.xScale(z),
          y1: chart.yScale(yMin),
          x2: chart.xScale(z),
          y2: chart.yScale(yMax),
          stroke: colors[key],
          "stroke-dasharray": "4 4",
          opacity: "0.35",
        })
      );

      legend.innerHTML = shown
        .map((id) => `<span><i style="background:${colors[id]}"></i> ${activations[id].label}</span>`)
        .join("");

      /* ── derivative chart ── */
      const derivChart = U.makeChart(deriv, {
        xDomain: [-6, 6],
        yDomain: [-0.1, 1.15],
        title: "Derivative f′(z) — the multiplier backpropagation applies",
      });
      shown.forEach((id) => {
        const curve = xs.map((x) => ({ x, y: U.clamp(activations[id].dz(x), -0.1, 1.15) }));
        deriv.appendChild(
          U.svgEl("path", {
            d: U.pathFromPoints(curve, derivChart.xScale, derivChart.yScale),
            fill: "none",
            stroke: colors[id],
            "stroke-width": id === key ? "3" : "1.6",
            opacity: id === key ? "1" : "0.4",
          })
        );
      });
      deriv.appendChild(
        U.svgEl("circle", {
          cx: derivChart.xScale(z),
          cy: derivChart.yScale(U.clamp(slope, -0.1, 1.15)),
          r: 6,
          fill: colors[key],
          stroke: C().ring,
          "stroke-width": "2.5",
        })
      );
      deriv.appendChild(
        U.svgEl("line", {
          x1: derivChart.xScale(-6),
          y1: derivChart.yScale(1),
          x2: derivChart.xScale(6),
          y2: derivChart.yScale(1),
          stroke: C().danger,
          "stroke-dasharray": "5 5",
          opacity: "0.5",
        })
      );
      deriv.appendChild(
        U.svgEl("text", { x: derivChart.xScale(-5.8), y: derivChart.yScale(1) - 6, class: "svg-label" })
      ).textContent = "f′ = 1 — gradient passes through unchanged";
    }

    const scheduleRender = U.rafThrottle(render);

    [fnInput, zInput, layersInput, compareInput].forEach((input) => input.addEventListener("input", render));
    U.onRedraw(render);
    render();
  }

  /* ── Gradient descent ─────────────────────────────────────────── */
  function mountGradientDescent(rootNode) {
    mountSection(
      rootNode,
      "Gradient descent on a loss surface",
      "Pick a surface, set the learning rate, and step downhill. The whole point of this page is the learning rate: nudge it past the stability threshold and watch the same run that was converging start climbing instead.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-grid">
              <div class="control-group">
                <label for="gd-surface">Loss surface</label>
                <select id="gd-surface">
                  <option value="bowl">Symmetric bowl</option>
                  <option value="valley">Narrow valley</option>
                  <option value="double">Two minima</option>
                </select>
              </div>
              <div class="control-group">
                <label for="gd-start">Start x₀</label>
                <input id="gd-start" type="number" step="0.1" value="-2.6" />
              </div>
            </div>
            <div class="control-group">
              <label for="gd-lr">Learning rate η</label>
              <div class="range-row">
                <input id="gd-lr" type="range" min="0.01" max="1.2" step="0.01" value="0.18" />
                <span class="range-value" id="gd-lr-value">0.18</span>
              </div>
            </div>
            <div class="control-group">
              <label for="gd-momentum">Momentum β</label>
              <div class="range-row">
                <input id="gd-momentum" type="range" min="0" max="0.95" step="0.05" value="0" />
                <span class="range-value" id="gd-momentum-value">0.00</span>
              </div>
            </div>
            <div class="step-controls">
              <button class="button primary" id="gd-step">Take one step</button>
              <button class="button secondary" id="gd-run">Run 25 steps</button>
              <button class="button secondary" id="gd-reset">Reset</button>
            </div>
            <div class="callout" id="gd-callout"></div>
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg id="gd-plot" viewBox="0 0 560 280" aria-label="Loss surface"></svg>
              <div class="drag-hint">✥ Drag the marker to start the descent from anywhere on the curve.</div>
            </div>
            <div class="plot-card">
              <svg id="gd-loss" viewBox="0 0 560 200" aria-label="Loss over steps"></svg>
            </div>
            <div class="readout"><div class="output-grid" id="gd-output"></div></div>
            <div class="equation-card" id="gd-math"></div>
            <div class="steps"><ol id="gd-steps"></ol></div>
          </div>
        </div>
      `
    );

    const surfaceInput = rootNode.querySelector("#gd-surface");
    const startInput = rootNode.querySelector("#gd-start");
    const lrInput = rootNode.querySelector("#gd-lr");
    const lrValue = rootNode.querySelector("#gd-lr-value");
    const momentumInput = rootNode.querySelector("#gd-momentum");
    const momentumValue = rootNode.querySelector("#gd-momentum-value");
    const stepButton = rootNode.querySelector("#gd-step");
    const runButton = rootNode.querySelector("#gd-run");
    const resetButton = rootNode.querySelector("#gd-reset");
    const callout = rootNode.querySelector("#gd-callout");
    const plot = rootNode.querySelector("#gd-plot");
    const lossPlot = rootNode.querySelector("#gd-loss");
    const output = rootNode.querySelector("#gd-output");
    const mathNode = rootNode.querySelector("#gd-math");
    const stepsNode = rootNode.querySelector("#gd-steps");

    const surfaces = {
      bowl: {
        label: "Symmetric bowl",
        loss: (x) => x * x,
        grad: (x) => 2 * x,
        tex: "L(x) = x^2",
        gradTex: "\\nabla L(x) = 2x",
        curvature: 2,
        note: "The simplest convex case. Gradient descent converges for any η below 2/curvature = 1.0, and diverges above it.",
      },
      valley: {
        label: "Narrow valley",
        loss: (x) => 6 * x * x,
        grad: (x) => 12 * x,
        tex: "L(x) = 6x^2",
        gradTex: "\\nabla L(x) = 12x",
        curvature: 12,
        note: "Six times steeper, so the stability threshold drops to 2/12 ≈ 0.167. A learning rate that was fine on the bowl now explodes here.",
      },
      double: {
        label: "Two minima",
        loss: (x) => 0.25 * Math.pow(x, 4) - 1.5 * x * x + 0.4 * x + 4,
        grad: (x) => Math.pow(x, 3) - 3 * x + 0.4,
        tex: "L(x) = \\tfrac{1}{4}x^4 - \\tfrac{3}{2}x^2 + 0.4x + 4",
        gradTex: "\\nabla L(x) = x^3 - 3x + 0.4",
        curvature: 6,
        note: "Non-convex. Where you start decides which minimum you land in — and one of them is worse than the other.",
      },
    };

    let path = [];
    let velocity = 0;
    let diverged = false;

    function reset() {
      path = [Number(startInput.value) || 0];
      velocity = 0;
      diverged = false;
    }

    function takeStep() {
      if (diverged) return;
      const surface = surfaces[surfaceInput.value];
      const eta = Number(lrInput.value);
      const beta = Number(momentumInput.value);
      const x = path[path.length - 1];
      const gradient = surface.grad(x);
      velocity = beta * velocity + gradient;
      const next = x - eta * velocity;
      if (!Number.isFinite(next) || Math.abs(next) > 1e6) {
        diverged = true;
        return;
      }
      path.push(next);
      if (path.length > 400) path = path.slice(-400);
    }

    function render() {
      const surface = surfaces[surfaceInput.value];
      const eta = Number(lrInput.value);
      const beta = Number(momentumInput.value);
      lrValue.textContent = eta.toFixed(2);
      momentumValue.textContent = beta.toFixed(2);

      const x = path[path.length - 1];
      const gradient = surface.grad(x);
      const loss = surface.loss(x);
      const threshold = 2 / surface.curvature;
      const unstable = eta > threshold;

      callout.innerHTML = diverged
        ? `<strong>Diverged.</strong> With η = ${eta.toFixed(2)} the steps overshoot and grow without bound. Lower the learning rate below ${U.round(threshold, 3)} and reset.`
        : unstable
        ? `<strong>η = ${eta.toFixed(2)} is above the stability threshold ${U.round(threshold, 3)}</strong> for this surface — steps will overshoot and the loss will climb.`
        : `η = ${eta.toFixed(2)} is below the stability threshold ${U.round(threshold, 3)} for this surface, so the loss decreases monotonically.`;

      U.renderMetrics(output, [
        { label: "Steps", value: String(path.length - 1) },
        { label: "x", value: diverged ? "diverged" : U.round(x, 5) },
        { label: "Loss", value: diverged ? "∞" : U.round(loss, 5) },
        { label: "Gradient", value: diverged ? "∞" : U.round(gradient, 5) },
      ]);

      const previous = path.length > 1 ? path[path.length - 2] : null;
      renderFormulaCards(mathNode, [
        {
          title: "The update rule",
          description:
            "The gradient points uphill, so subtract it. The learning rate is the only thing deciding how far along that direction you actually move.",
          tex: beta > 0
            ? "v_{t+1} = \\beta v_t + \\nabla L(x_t), \\qquad x_{t+1} = x_t - \\eta\\, v_{t+1}"
            : "x_{t+1} = x_t - \\eta \\, \\nabla L(x_t)",
          derivation: previous !== null && !diverged
            ? [
                { tex: `\\nabla L(${U.texNum(previous, 4)}) = ${U.texNum(surface.grad(previous), 4)}`, result: U.round(surface.grad(previous), 4) },
                beta > 0
                  ? {
                      tex: `v = ${U.texNum(beta, 2)}\\,v_{\\text{prev}} + ${U.texNum(surface.grad(previous), 4)} = ${U.texNum(velocity, 4)}`,
                      result: U.round(velocity, 4),
                      note: "Momentum accumulates past gradients, so consistent directions build up speed and oscillations cancel out.",
                    }
                  : null,
                {
                  tex: `x_{t+1} = ${U.texNum(previous, 4)} - ${U.texNum(eta, 2)} \\times ${U.texNum(beta > 0 ? velocity : surface.grad(previous), 4)}`,
                  result: U.round(x, 5),
                },
              ].filter(Boolean)
            : [{ tex: "\\text{press ``Take one step'' to move downhill}", result: "" }],
          insight:
            "<strong>The sign does all the work.</strong> A positive gradient means the loss rises as x rises, so the update moves x left. Nothing about this needs the loss to be a neural network — it is the same for every parameter of every model.",
        },
        {
          title: "This surface",
          description: surface.note,
          tex: `${surface.tex}, \\qquad ${surface.gradTex}`,
          derivation: [
            {
              tex: `\\eta_{\\max} = \\frac{2}{L''} = \\frac{2}{${surface.curvature}} = ${U.texNum(threshold, 4)}`,
              result: U.round(threshold, 4),
              note: `Your η is ${eta.toFixed(2)} — ${unstable ? "above" : "below"} the threshold.`,
            },
          ],
          insight:
            "<strong>Curvature sets the speed limit.</strong> For a quadratic, gradient descent converges only when η < 2/L″. That is why the “right” learning rate is not a universal number — it depends on the shape of the loss, which is why schedules and adaptive optimisers exist.",
        },
      ]);

      U.renderSteps(
        stepsNode,
        path.slice(1).map((value, index) => {
          const from = path[index];
          return `<strong>Step ${index + 1}.</strong> x: ${U.round(from, 4)} → <strong>${U.round(value, 4)}</strong>, loss ${U.round(surface.loss(from), 4)} → <strong>${U.round(surface.loss(value), 4)}</strong>.`;
        }),
        Math.min(path.length - 1, 12)
      );

      /* ── loss surface ── */
      const domain = surfaceInput.value === "double" ? [-3.4, 3.4] : [-3.2, 3.2];
      const xs = U.linspace(domain[0], domain[1], 220);
      const values = xs.map((value) => surface.loss(value));
      const yMax = Math.max(...values) * 1.08;
      const chart = U.makeChart(plot, {
        xDomain: domain,
        yDomain: [Math.min(0, ...values) - 0.2, yMax],
        title: `Loss surface — ${surface.label}`,
      });
      plot.appendChild(
        U.svgEl("path", {
          d: U.pathFromPoints(xs.map((value) => ({ x: value, y: surface.loss(value) })), chart.xScale, chart.yScale),
          class: "curve-primary",
        })
      );

      if (!diverged) {
        /* Draw the descent trajectory as arrows along the curve. */
        path.forEach((value, index) => {
          if (index === 0) return;
          const from = path[index - 1];
          plot.appendChild(
            U.svgEl("line", {
              x1: chart.xScale(from),
              y1: chart.yScale(surface.loss(from)),
              x2: chart.xScale(value),
              y2: chart.yScale(surface.loss(value)),
              stroke: C().b,
              "stroke-width": "1.8",
              opacity: "0.5",
            })
          );
        });
        path.forEach((value, index) => {
          plot.appendChild(
            U.svgEl("circle", {
              cx: chart.xScale(value),
              cy: chart.yScale(surface.loss(value)),
              r: index === path.length - 1 ? 8 : 3.5,
              fill: index === path.length - 1 ? C().b : U.tint(C().b, 0.45),
              stroke: index === path.length - 1 ? "#fff" : "none",
              "stroke-width": "2.5",
            })
          );
        });

        /* Tangent line at the current point — the gradient made visible. */
        const tangentSpan = 0.7;
        plot.appendChild(
          U.svgEl("line", {
            x1: chart.xScale(x - tangentSpan),
            y1: chart.yScale(loss - gradient * tangentSpan),
            x2: chart.xScale(x + tangentSpan),
            y2: chart.yScale(loss + gradient * tangentSpan),
            stroke: C().a,
            "stroke-width": "2.4",
            "stroke-dasharray": "6 4",
          })
        );

        const handle = U.svgEl("circle", {
          cx: chart.xScale(x),
          cy: chart.yScale(loss),
          r: 9,
          fill: "transparent",
          stroke: "transparent",
        });
        plot.appendChild(handle);
        U.draggable(handle, plot, chart, (position) => {
          startInput.value = U.round(position.x, 2);
          reset();
          scheduleRender();
        });
      }

      /* ── loss curve over steps ── */
      const lossValues = path.map((value) => surface.loss(value));
      const lossChart = U.makeChart(lossPlot, {
        xDomain: [0, Math.max(path.length - 1, 1)],
        yDomain: [0, Math.max(...lossValues, 0.1) * 1.1],
        title: "Loss per step",
      });
      lossPlot.appendChild(
        U.svgEl("path", {
          d: U.pathFromPoints(lossValues.map((value, index) => ({ x: index, y: value })), lossChart.xScale, lossChart.yScale),
          fill: "none",
          stroke: C().b,
          "stroke-width": "2.6",
        })
      );
      lossValues.forEach((value, index) => {
        lossPlot.appendChild(
          U.svgEl("circle", { cx: lossChart.xScale(index), cy: lossChart.yScale(value), r: 3.5, fill: C().b })
        );
      });
    }

    const scheduleRender = U.rafThrottle(render);

    stepButton.addEventListener("click", () => {
      takeStep();
      render();
    });
    runButton.addEventListener("click", () => {
      for (let index = 0; index < 25; index += 1) takeStep();
      render();
    });
    resetButton.addEventListener("click", () => {
      reset();
      render();
    });
    [surfaceInput, startInput, lrInput, momentumInput].forEach((input) =>
      input.addEventListener("input", () => {
        reset();
        render();
      })
    );

    reset();
    U.onRedraw(render);
    render();
  }



  /* ── Backpropagation ──────────────────────────────────────────
     Built on window.NNEngine (ported from the standalone Vite app), so
     the architecture, the activations, the loss function and every
     weight and bias are the learner's to change. Nothing on this page
     is a stored constant — the step narration is generated from the
     numbers the engine just produced. */
  function mountBackprop(rootNode) {
    const E = window.NNEngine;

    mountSection(
      rootNode,
      "Backpropagation, computed from your network",
      "Design the network, set the inputs and targets, then walk the forward pass, the backward pass and the weight update one step at a time. Add layers, switch activations, change the loss — every value in the diagram and every line of the derivation is recomputed from what you typed.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-grid">
              <div class="control-group">
                <label for="bp-loss">Loss function</label>
                <select id="bp-loss">
                  <option value="mse">Mean squared error</option>
                  <option value="bce">Binary cross-entropy</option>
                  <option value="cce">Categorical cross-entropy</option>
                </select>
              </div>
              <div class="control-group">
                <label for="bp-lr">Learning rate η</label>
                <div class="range-row">
                  <input id="bp-lr" type="range" min="0.01" max="3" step="0.01" value="0.5" />
                  <span class="range-value" id="bp-lr-value">0.50</span>
                </div>
              </div>
            </div>
            <div class="step-controls">
              <button class="button primary" id="bp-next">Next step</button>
              <button class="button secondary" id="bp-prev">Previous</button>
              <button class="button secondary" id="bp-all">Show all</button>
            </div>
            <div class="step-controls">
              <button class="button secondary" id="bp-epoch">Train 1 epoch</button>
              <button class="button secondary" id="bp-train">Train 100</button>
              <button class="button ghost" id="bp-reset">Reset weights</button>
            </div>
            <div class="callout" id="bp-callout"></div>
            ${editorSlot("bp-arch")}
            ${editorSlot("bp-io")}
            <div id="bp-params"></div>
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg id="bp-net" viewBox="0 0 660 360" aria-label="Network diagram"></svg>
              <div class="legend">
                <span><i style="background:#2563a8"></i> Input</span>
                <span><i style="background:#7c3aed"></i> Hidden</span>
                <span><i style="background:#c2410c"></i> Output</span>
                <span><i style="background:#b42318"></i> Loss</span>
              </div>
            </div>
            <div class="readout"><div class="output-grid" id="bp-output"></div></div>
            <div class="equation-card" id="bp-math"></div>
            <div class="two-up">
              <div class="plot-card">
                <svg id="bp-grads" viewBox="0 0 560 210" aria-label="Gradient magnitude per layer"></svg>
              </div>
              <div class="plot-card">
                <svg id="bp-loss-plot" viewBox="0 0 560 210" aria-label="Loss over epochs"></svg>
              </div>
            </div>
            <div class="steps"><ol id="bp-steps"></ol></div>
          </div>
        </div>
      `
    );

    const lossInput = rootNode.querySelector("#bp-loss");
    const lrInput = rootNode.querySelector("#bp-lr");
    const lrValue = rootNode.querySelector("#bp-lr-value");
    const nextButton = rootNode.querySelector("#bp-next");
    const prevButton = rootNode.querySelector("#bp-prev");
    const allButton = rootNode.querySelector("#bp-all");
    const epochButton = rootNode.querySelector("#bp-epoch");
    const trainButton = rootNode.querySelector("#bp-train");
    const resetButton = rootNode.querySelector("#bp-reset");
    const callout = rootNode.querySelector("#bp-callout");
    const net = rootNode.querySelector("#bp-net");
    const output = rootNode.querySelector("#bp-output");
    const mathNode = rootNode.querySelector("#bp-math");
    const gradPlot = rootNode.querySelector("#bp-grads");
    const lossPlot = rootNode.querySelector("#bp-loss-plot");
    const stepsNode = rootNode.querySelector("#bp-steps");
    const paramsNode = rootNode.querySelector("#bp-params");

    /* The classic textbook example, so the page opens on numbers that
       can be checked by hand. */
    const defaultArch = [
      { neurons: 3, activation: "sigmoid" },
      { neurons: 3, activation: "sigmoid" },
      { neurons: 2, activation: "sigmoid" },
    ];
    const defaultIO = [
      { name: "unit 1", input: 0.5, target: 1 },
      { name: "unit 2", input: 0.3, target: 0 },
    ];
    const textbookWeights = [
      [[0.2, 0.4], [0.3, 0.1], [0.5, 0.2]],
      [[0.3, 0.2, 0.4], [0.1, 0.5, 0.3], [0.4, 0.2, 0.1]],
      [[0.6, 0.1, 0.3], [0.2, 0.4, 0.5]],
    ];

    let arch = defaultArch.map((layer) => ({ ...layer }));
    let io = defaultIO.map((row) => ({ ...row }));
    let network = null;
    let cursor = -1;
    let lossHistory = [];
    let weightEditors = [];

    function config() {
      return {
        inputSize: io.length,
        layers: arch.map((layer) => ({
          neurons: Math.max(1, Math.round(Number(layer.neurons) || 1)),
          activation: layer.activation,
        })),
        loss: lossInput.value,
        learningRate: Number(lrInput.value),
      };
    }

    function buildNetwork(useTextbook) {
      const cfg = config();
      network = E.initializeNetwork(cfg, 1);
      if (useTextbook && matchesTextbookShape(cfg)) {
        network.layers.forEach((layer, li) => {
          layer.neurons.forEach((neuron, ni) => {
            neuron.weights = [...textbookWeights[li][ni]];
            neuron.bias = 0;
          });
        });
      }
      lossHistory = [];
      cursor = -1;
      syncIO();
    }

    function matchesTextbookShape(cfg) {
      return (
        cfg.inputSize === 2 &&
        cfg.layers.length === 3 &&
        cfg.layers[0].neurons === 3 &&
        cfg.layers[1].neurons === 3 &&
        cfg.layers[2].neurons === 2
      );
    }

    function syncIO() {
      network.inputs = io.map((row) => Number(row.input));
      network.target = io.map((row) => Number(row.target));
      /* The output layer decides how many targets matter; pad or trim so
         the loss never reads undefined. */
      const outputSize = network.layers[network.layers.length - 1].neurons.length;
      while (network.target.length < outputSize) network.target.push(0);
      network.target = network.target.slice(0, outputSize);
    }

    /* ── Editors ──────────────────────────────────────────────── */
    const archEditor = U.dataEditor(rootNode.querySelector("#bp-arch"), {
      title: "Architecture",
      hint: "One row per layer, in order; the last row is the output layer. Add a row to make the network deeper and watch the first layer's gradient shrink.",
      columns: [
        { key: "neurons", label: "Neurons", min: 1, max: 6, step: 1 },
        {
          key: "activation",
          label: "Activation",
          type: "select",
          options: Object.keys(E.ACTIVATIONS).map((key) => ({ value: key, label: E.ACTIVATIONS[key].label })),
        },
      ],
      rows: arch,
      minRows: 1,
      maxRows: 5,
      newRow: () => ({ neurons: 3, activation: "sigmoid" }),
      onChange: (rows) => {
        arch = rows.map((row) => ({ neurons: Number(row.neurons), activation: row.activation }));
        buildNetwork(true);
        buildParamEditors();
        render();
      },
    });

    const ioEditor = U.dataEditor(rootNode.querySelector("#bp-io"), {
      title: "Inputs and targets",
      hint: "One row per input feature. The target column is matched against the output layer — extra rows are ignored, missing ones count as 0.",
      columns: [
        { key: "name", label: "Unit", type: "text" },
        { key: "input", label: "x", step: 0.05 },
        { key: "target", label: "y", step: 0.05 },
      ],
      rows: io,
      minRows: 1,
      maxRows: 5,
      newRow: (current) => ({ name: `unit ${current.length + 1}`, input: 0.5, target: 0 }),
      onChange: (rows) => {
        io = rows;
        buildNetwork(true);
        buildParamEditors();
        render();
      },
    });

    /* One matrix editor per layer, rebuilt whenever the shape changes. */
    function buildParamEditors() {
      paramsNode.innerHTML = "";
      weightEditors = [];
      network.layers.forEach((layer, layerIndex) => {
        const slot = document.createElement("div");
        slot.className = "editor-slot";
        paramsNode.appendChild(slot);

        const isOutput = layerIndex === network.layers.length - 1;
        const inputLabels =
          layerIndex === 0
            ? io.map((row) => row.name || `x${io.indexOf(row) + 1}`)
            : network.layers[layerIndex - 1].neurons.map((_, i) => `a${i + 1}`);

        /* Bias is shown as an extra column so it is visibly part of the
           same parameter vector, not a footnote. */
        const values = layer.neurons.map((neuron) => [...neuron.weights, neuron.bias]);

        const editor = U.matrixEditor(slot, {
          title: `${isOutput ? "Output layer" : `Hidden layer ${layerIndex + 1}`} — weights and bias`,
          hint:
            layerIndex === 0
              ? "Row j is the incoming weight vector for neuron j, with its bias in the last column."
              : undefined,
          rowLabels: layer.neurons.map((_, i) => `n${i + 1}`),
          colLabels: [...inputLabels, "bias"],
          values,
          step: 0.05,
          min: -8,
          max: 8,
          onChange: (_, raw) => {
            layer.neurons.forEach((neuron, i) => {
              neuron.weights = raw[i].slice(0, neuron.weights.length);
              neuron.bias = raw[i][raw[i].length - 1];
            });
            cursor = -1;
            render();
          },
        });
        weightEditors.push(editor);
      });
    }

    function refreshParamEditors() {
      network.layers.forEach((layer, layerIndex) => {
        const editor = weightEditors[layerIndex];
        if (!editor) return;
        editor.set(layer.neurons.map((neuron) => [...neuron.weights, neuron.bias]), true);
      });
    }

    /* ── Step narration ───────────────────────────────────────── */
    const fmt = (value, digits = 4) => U.texNum(value, digits);
    const vecTex = (values, digits = 4) =>
      `\\begin{bmatrix} ${values.map((v) => fmt(v, digits)).join(" & ")} \\end{bmatrix}`;

    /* Recompute the whole pass, capturing a snapshot after each stage so
       the step list can show intermediate state faithfully. */
    function buildSteps() {
      const cfg = config();
      const steps = [];
      const work = E.clone(network);
      work.inputs = network.inputs;
      work.target = network.target;

      const layerName = (index) =>
        index === work.layers.length - 1 ? "output layer" : `hidden layer ${index + 1}`;

      work.layers.forEach((_, layerIndex) => {
        E.forwardLayer(work, layerIndex, cfg);
        const layer = work.layers[layerIndex];
        const incoming = E.activationsInto(work, layerIndex);
        const activation = E.ACTIVATIONS[cfg.layers[layerIndex].activation];
        const isSoftmax = cfg.layers[layerIndex].activation === "softmax";

        const derivation = layer.neurons.slice(0, 3).map((neuron, j) => ({
          tex: `z_{${j + 1}} = ${neuron.weights
            .map((w, k) => `(${fmt(w, 2)})(${fmt(incoming[k], 3)})`)
            .join(" + ")}${neuron.bias !== 0 ? ` + ${fmt(neuron.bias, 2)}` : ""} = ${fmt(neuron.z)}`,
          result: U.round(neuron.a, 4),
          note: j === 0 ? `Then a = f(z) with f = ${activation.label}.` : undefined,
        }));
        if (isSoftmax) {
          derivation.push({
            tex: `\\sum_j a_j = ${fmt(layer.neurons.reduce((acc, n) => acc + n.a, 0), 6)}`,
            result: "1",
            note: "Softmax normalises across the layer, so the outputs form a probability distribution.",
          });
        }

        steps.push({
          phase: "forward",
          title: `Forward — ${layerName(layerIndex)}`,
          activeLayer: layerIndex,
          tex: isSoftmax
            ? "\\mathbf{z}^{(l)} = W^{(l)}\\mathbf{a}^{(l-1)} + \\mathbf{b}^{(l)}, \\qquad a_i = \\frac{e^{z_i}}{\\sum_j e^{z_j}}"
            : "\\mathbf{z}^{(l)} = W^{(l)}\\mathbf{a}^{(l-1)} + \\mathbf{b}^{(l)}, \\qquad \\mathbf{a}^{(l)} = f\\!\\left(\\mathbf{z}^{(l)}\\right)",
          derivation,
          insight:
            layerIndex === 0
              ? "<strong>Store z, not just a.</strong> The backward pass needs the pre-activation to evaluate f′(z) — that is exactly what an autograd tape keeps for you."
              : "<strong>Each layer is a matrix multiply plus a non-linearity.</strong> Remove the non-linearity and the whole stack collapses into a single matrix.",
          snapshot: E.clone(work),
        });
      });

      const loss = E.computeLoss(work, cfg);
      const outputs = work.layers[work.layers.length - 1].neurons.map((n) => n.a);
      steps.push({
        phase: "forward",
        title: "Compute the loss",
        activeLayer: work.layers.length - 1,
        showLoss: true,
        tex: E.LOSSES[cfg.loss].tex,
        derivation: [
          {
            tex: `\\mathbf{a} = ${vecTex(outputs)}, \\quad \\mathbf{y} = ${vecTex(work.target, 2)}`,
            result: "",
          },
          {
            tex: `L = ${fmt(loss, 6)}`,
            result: U.round(loss, 6),
            note: E.LOSSES[cfg.loss].suits,
          },
        ],
        insight:
          "<strong>The loss is a single number.</strong> Every gradient on this page is the derivative of that one scalar with respect to one parameter — which is what makes reverse-mode differentiation so much cheaper than the forward alternative.",
        snapshot: E.clone(work),
      });

      /* Backward pass, output layer first. */
      const outIndex = work.layers.length - 1;
      E.backpropOutput(work, cfg);
      const outLayer = work.layers[outIndex];
      const shortcut = E.outputShortcut(cfg.layers[outIndex].activation, cfg.loss);

      steps.push({
        phase: "backward",
        title: "Backward — output error signal δ",
        activeLayer: outIndex,
        tex: shortcut
          ? "\\delta^{(L)} = \\mathbf{a}^{(L)} - \\mathbf{y}"
          : "\\delta^{(L)} = \\frac{\\partial L}{\\partial \\mathbf{a}} \\odot f'\\!\\left(\\mathbf{z}^{(L)}\\right)",
        derivation: outLayer.neurons.slice(0, 4).map((neuron, j) => ({
          tex: shortcut
            ? `\\delta_{${j + 1}} = ${fmt(neuron.a)} - ${fmt(work.target[j], 2)} = ${fmt(neuron.dLoss_dZ, 5)}`
            : `\\delta_{${j + 1}} = (${fmt(neuron.dLoss_dA, 4)})\\,(${fmt(E.ACTIVATIONS[cfg.layers[outIndex].activation].df(neuron.z), 4)}) = ${fmt(neuron.dLoss_dZ, 5)}`,
          result: U.round(neuron.dLoss_dZ, 6),
          note:
            j === 0 && shortcut
              ? "This pairing collapses to a plain (a − y): the activation derivative and the loss derivative cancel exactly. Computing them separately would be numerically unstable near a = 0 or 1."
              : j === 0
              ? "Two factors: how wrong the output is, and how responsive the activation is there."
              : undefined,
        })),
        insight: shortcut
          ? `<strong>${cfg.layers[outIndex].activation === "softmax" ? "Softmax with categorical cross-entropy" : "Sigmoid with binary cross-entropy"} is a deliberate pairing.</strong> The gradient simplifies to a − y, which is both faster and better conditioned. Switch the loss to squared error and watch this line grow an extra factor.`
          : "<strong>Try pairing sigmoid with binary cross-entropy</strong> (or softmax with categorical) — the two derivatives cancel and this expression collapses to a plain a − y.",
        snapshot: E.clone(work),
      });

      steps.push({
        phase: "backward",
        title: "Backward — gradients for the output weights",
        activeLayer: outIndex,
        tex: "\\frac{\\partial L}{\\partial W^{(l)}_{jk}} = \\delta^{(l)}_j \\, a^{(l-1)}_k, \\qquad \\frac{\\partial L}{\\partial b^{(l)}_j} = \\delta^{(l)}_j",
        derivation: outLayer.neurons.slice(0, 3).map((neuron, j) => ({
          tex: `\\nabla W_{${j + 1}} = ${fmt(neuron.dLoss_dZ, 5)} \\times ${vecTex(E.activationsInto(work, outIndex), 3)} = ${vecTex(neuron.dLoss_dW, 5)}`,
          result: "",
          note: j === 0 ? "Error at the destination times activation at the source — that is the entire rule." : undefined,
        })),
        insight:
          "<strong>The bias gradient is just δ.</strong> A bias has a constant input of 1, so the same rule gives it the error signal unmodified.",
        snapshot: E.clone(work),
      });

      for (let layerIndex = work.layers.length - 2; layerIndex >= 0; layerIndex -= 1) {
        E.backpropHidden(work, layerIndex, cfg);
        const layer = work.layers[layerIndex];
        const nextLayer = work.layers[layerIndex + 1];
        steps.push({
          phase: "backward",
          title: `Backward — ${layerName(layerIndex)} error signal δ`,
          activeLayer: layerIndex,
          tex: "\\delta^{(l)} = \\left(\\left(W^{(l+1)}\\right)^{\\top}\\delta^{(l+1)}\\right) \\odot f'\\!\\left(\\mathbf{z}^{(l)}\\right)",
          derivation: [
            ...layer.neurons.slice(0, 3).map((neuron, j) => ({
              tex: `\\delta_{${j + 1}} = \\left[${nextLayer.neurons
                .slice(0, 3)
                .map((nn) => `(${fmt(nn.dLoss_dZ, 5)})(${fmt(nn.weights[j], 2)})`)
                .join(" + ")}${nextLayer.neurons.length > 3 ? " + \\cdots" : ""}\\right](${fmt(E.ACTIVATIONS[cfg.layers[layerIndex].activation].df(neuron.z), 4)})`,
              result: U.round(neuron.dLoss_dZ, 7),
              note:
                j === 0
                  ? "This neuron feeds every neuron in the next layer, so its share of the blame is the sum over all of those paths — the multivariable chain rule."
                  : undefined,
            })),
            {
              tex: `\\frac{\\lVert \\nabla W^{(${outIndex + 1})} \\rVert}{\\lVert \\nabla W^{(${layerIndex + 1})} \\rVert} = ${fmt(
                E.gradientNorm(work, outIndex) / Math.max(E.gradientNorm(work, layerIndex), 1e-15),
                1
              )}`,
              result: U.round(
                E.gradientNorm(work, outIndex) / Math.max(E.gradientNorm(work, layerIndex), 1e-15),
                1
              ),
              note: "The vanishing gradient, measured. Switch every activation to ReLU and watch this ratio collapse toward 1.",
            },
          ],
          insight:
            "<strong>Nothing is recomputed.</strong> Each δ is built from the δ immediately to its right, which is why one backward sweep costs about as much as one forward pass — no matter how many parameters there are.",
          snapshot: E.clone(work),
        });
      }

      const eta = cfg.learningRate;
      const firstNeuron = work.layers[0].neurons[0];
      const lastNeuron = outLayer.neurons[0];
      steps.push({
        phase: "update",
        title: "Apply the gradient descent update",
        activeLayer: null,
        tex: "W^{(l)} \\leftarrow W^{(l)} - \\eta \\, \\nabla_{W^{(l)}} L, \\qquad \\mathbf{b}^{(l)} \\leftarrow \\mathbf{b}^{(l)} - \\eta \\, \\delta^{(l)}",
        derivation: [
          {
            tex: `W^{(${outIndex + 1})}_{11} \\leftarrow ${fmt(lastNeuron.weights[0], 4)} - ${fmt(eta, 2)}(${fmt(lastNeuron.dLoss_dW[0], 5)}) = ${fmt(lastNeuron.weights[0] - eta * lastNeuron.dLoss_dW[0], 5)}`,
            result: U.round(lastNeuron.weights[0] - eta * lastNeuron.dLoss_dW[0], 5),
          },
          {
            tex: `W^{(1)}_{11} \\leftarrow ${fmt(firstNeuron.weights[0], 4)} - ${fmt(eta, 2)}(${fmt(firstNeuron.dLoss_dW[0], 7)}) = ${fmt(firstNeuron.weights[0] - eta * firstNeuron.dLoss_dW[0], 7)}`,
            result: U.round(firstNeuron.weights[0] - eta * firstNeuron.dLoss_dW[0], 7),
            note: "The first layer barely moves — its gradient is orders of magnitude smaller. Press “Train 1 epoch” to commit the update and watch the loss fall.",
          },
        ],
        insight:
          "<strong>Every parameter moves at once.</strong> The update is applied to all layers simultaneously using gradients computed from the same forward pass — not layer by layer in sequence.",
        snapshot: E.clone(work),
      });

      return steps;
    }

    /* ── Rendering ────────────────────────────────────────────── */
    function drawNetwork(state, step) {
      U.clear(net);
      const { width, height } = U.viewBoxSize(net);
      const columns = [
        { label: "input", values: state.inputs, color: C().c, ids: state.inputs.map((_, i) => `in-${i}`) },
        ...state.layers.map((layer, index) => ({
          label: index === state.layers.length - 1 ? "output" : `hidden ${index + 1}`,
          values: layer.neurons.map((n) => n.a),
          color: index === state.layers.length - 1 ? C().b : C().d,
          ids: layer.neurons.map((_, i) => `l${index}-${i}`),
          layerIndex: index,
        })),
      ];

      const left = 60;
      const right = width - 110;
      const gapX = columns.length > 1 ? (right - left) / (columns.length - 1) : 0;
      const top = 44;
      const bottom = height - 42;
      const radius = Math.max(12, Math.min(24, 150 / Math.max(...columns.map((c) => c.values.length))));

      const positions = {};
      columns.forEach((column, ci) => {
        const x = left + ci * gapX;
        const span = bottom - top;
        column.values.forEach((_, ni) => {
          const y = top + (span * (ni + 1)) / (column.values.length + 1);
          positions[column.ids[ni]] = { x, y };
        });
      });

      const activeLayer = step ? step.activeLayer : null;

      /* Edges */
      state.layers.forEach((layer, li) => {
        const fromIds = li === 0 ? columns[0].ids : columns[li].ids;
        const toIds = columns[li + 1].ids;
        const isActive = activeLayer === li;
        layer.neurons.forEach((neuron, ni) => {
          neuron.weights.forEach((weight, ki) => {
            const a = positions[fromIds[ki]];
            const b = positions[toIds[ni]];
            if (!a || !b) return;
            net.appendChild(
              U.svgEl("line", {
                x1: a.x,
                y1: a.y,
                x2: b.x,
                y2: b.y,
                stroke: isActive ? C().a : weight >= 0 ? C().faint : C().faint,
                "stroke-width": isActive ? 2.4 : Math.min(0.5 + Math.abs(weight) * 1.4, 3),
                opacity: isActive ? 0.95 : 0.6,
              })
            );
          });
        });
      });

      /* Nodes */
      columns.forEach((column, ci) => {
        net.appendChild(
          U.svgEl("text", { x: left + ci * gapX, y: 22, class: "svg-label", "text-anchor": "middle" })
        ).textContent = column.label;
        column.values.forEach((value, ni) => {
          const pos = positions[column.ids[ni]];
          const isActive = ci > 0 && activeLayer === ci - 1;
          net.appendChild(
            U.svgEl("circle", {
              cx: pos.x,
              cy: pos.y,
              r: radius,
              fill: column.color,
              opacity: isActive ? 1 : 0.85,
              stroke: isActive ? C().a : "#fff",
              "stroke-width": isActive ? 4 : 2,
            })
          );
          net.appendChild(
            U.svgEl("text", {
              x: pos.x,
              y: pos.y + 4,
              "text-anchor": "middle",
              fill: C().onFill,
              "font-size": radius > 18 ? "11" : "9",
              "font-family": "var(--mono)",
              "font-weight": "600",
            })
          ).textContent = U.round(value, radius > 18 ? 3 : 2);
        });
      });

      /* Loss node */
      const showLoss = step && (step.showLoss || step.phase !== "forward");
      const lx = width - 46;
      const ly = height / 2;
      net.appendChild(
        U.svgEl("circle", {
          cx: lx,
          cy: ly,
          r: 24,
          fill: C().danger,
          opacity: showLoss ? 1 : 0.35,
          stroke: C().ring,
          "stroke-width": 2,
        })
      );
      net.appendChild(
        U.svgEl("text", {
          x: lx,
          y: ly + 4,
          "text-anchor": "middle",
          fill: C().onFill,
          "font-size": "10",
          "font-family": "var(--mono)",
          "font-weight": "600",
        })
      ).textContent = showLoss && state.totalLoss !== undefined ? U.round(state.totalLoss, 3) : "L";
      net.appendChild(
        U.svgEl("text", { x: lx, y: ly + 40, class: "svg-label", "text-anchor": "middle" })
      ).textContent = "loss";

      if (step && step.phase === "backward") {
        net.appendChild(
          U.svgEl("text", { x: left, y: height - 12, class: "svg-label" })
        ).textContent = "← error signals flow backwards through the transposed weights";
      }
    }

    function drawGradients(state) {
      const magnitudes = state.layers.map((_, index) => ({
        label: index === state.layers.length - 1 ? "output" : `h${index + 1}`,
        value: E.gradientNorm(state, index),
      }));
      const max = Math.max(...magnitudes.map((m) => m.value), 1e-12);
      const chart = U.makeChart(gradPlot, {
        xDomain: [0, magnitudes.length + 1],
        yDomain: [0, max * 1.25],
        title: "Gradient magnitude per layer",
      });
      magnitudes.forEach((item, index) => {
        const x1 = chart.xScale(index + 0.6);
        const x2 = chart.xScale(index + 1.4);
        const y = chart.yScale(item.value);
        gradPlot.appendChild(
          U.svgEl("rect", {
            x: x1,
            y,
            width: x2 - x1,
            height: Math.max(chart.yScale(0) - y, 1),
            fill: index === magnitudes.length - 1 ? C().b : C().d,
            rx: 5,
            opacity: 0.9,
          })
        );
        gradPlot.appendChild(
          U.svgEl("text", { x: (x1 + x2) / 2, y: chart.yScale(0) + 16, class: "svg-label", "text-anchor": "middle" })
        ).textContent = item.label;
        gradPlot.appendChild(
          U.svgEl("text", { x: (x1 + x2) / 2, y: y - 5, class: "svg-label", "text-anchor": "middle" })
        ).textContent = item.value < 1e-4 ? item.value.toExponential(1) : U.round(item.value, 4);
      });
    }

    function drawLossHistory() {
      if (!lossHistory.length) {
        U.clear(lossPlot);
        const { width } = U.viewBoxSize(lossPlot);
        lossPlot.appendChild(U.svgEl("text", { x: 16, y: 24, class: "svg-title" })).textContent =
          "Loss per epoch";
        lossPlot.appendChild(
          U.svgEl("text", { x: width / 2, y: 110, class: "svg-label", "text-anchor": "middle" })
        ).textContent = "Press “Train 1 epoch” to start recording";
        return;
      }
      const chart = U.makeChart(lossPlot, {
        xDomain: [0, Math.max(lossHistory.length - 1, 1)],
        yDomain: [0, Math.max(...lossHistory) * 1.15],
        title: "Loss per epoch",
      });
      lossPlot.appendChild(
        U.svgEl("path", {
          d: U.pathFromPoints(lossHistory.map((value, index) => ({ x: index, y: value })), chart.xScale, chart.yScale),
          fill: "none",
          stroke: C().b,
          "stroke-width": 2.6,
        })
      );
      if (lossHistory.length <= 60) {
        lossHistory.forEach((value, index) => {
          lossPlot.appendChild(
            U.svgEl("circle", { cx: chart.xScale(index), cy: chart.yScale(value), r: 3, fill: C().b })
          );
        });
      }
    }

    function render() {
      lrValue.textContent = Number(lrInput.value).toFixed(2);
      const cfg = config();

      /* Always show the current forward state, even before stepping. */
      const preview = E.clone(network);
      preview.inputs = network.inputs;
      preview.target = network.target;
      E.forwardAll(preview, cfg);
      E.computeLoss(preview, cfg);
      E.backpropAll(preview, cfg);

      const steps = buildSteps();
      cursor = U.clamp(cursor, -1, steps.length - 1);
      const step = cursor >= 0 ? steps[cursor] : null;
      const shown = step ? step.snapshot : preview;

      const outIndex = preview.layers.length - 1;
      const ratio = E.gradientNorm(preview, outIndex) / Math.max(E.gradientNorm(preview, 0), 1e-15);
      const shape = [cfg.inputSize, ...cfg.layers.map((l) => l.neurons)].join(" → ");
      const paramCount = preview.layers.reduce(
        (acc, layer) => acc + layer.neurons.reduce((a, n) => a + n.weights.length + 1, 0),
        0
      );

      callout.innerHTML = step
        ? `<strong>${step.title}</strong> — step ${cursor + 1} of ${steps.length} (${step.phase} pass).`
        : `Network <strong>${shape}</strong> with ${paramCount} parameters. Press <strong>Next step</strong> to walk the forward pass, the backward pass, and the update.`;

      U.renderMetrics(output, [
        { label: "Loss", value: U.round(preview.totalLoss, 6) },
        { label: "Prediction", value: preview.layers[outIndex].neurons.map((n) => U.round(n.a, 3)).join(", ") },
        { label: "Target", value: preview.target.map((v) => U.round(v, 2)).join(", ") },
        { label: "Grad ratio out/in", value: `${U.round(ratio, 1)}×` },
      ]);

      if (step) {
        renderFormulaCards(mathNode, [
          {
            title: step.title,
            description:
              step.phase === "forward"
                ? "Forward pass — data flows left to right; each layer is a matrix multiply, a bias, then the activation."
                : step.phase === "backward"
                ? "Backward pass — the error signal flows right to left, picking up a transposed weight matrix and a local activation derivative at every layer."
                : "Weight update — every parameter takes one small step against its own gradient.",
            tex: step.tex,
            derivation: step.derivation,
            insight: step.insight,
          },
        ]);
      } else {
        renderFormulaCards(mathNode, [
          {
            title: "The four equations of backpropagation",
            description:
              "Everything on this page is these four lines, applied layer by layer. Step through to watch each one evaluated with your numbers.",
            tex: "\\delta^{(L)} = \\nabla_{\\mathbf{a}} L \\odot f'\\!\\left(\\mathbf{z}^{(L)}\\right)",
            derivation: [
              { tex: "\\delta^{(l)} = \\left(\\left(W^{(l+1)}\\right)^{\\top}\\delta^{(l+1)}\\right) \\odot f'\\!\\left(\\mathbf{z}^{(l)}\\right)", result: "" },
              { tex: "\\frac{\\partial L}{\\partial W^{(l)}_{jk}} = \\delta^{(l)}_j \\, a^{(l-1)}_k", result: "" },
              {
                tex: "\\frac{\\partial L}{\\partial b^{(l)}_j} = \\delta^{(l)}_j",
                result: "",
                note: "The transpose in the second line is the only structurally interesting part: it routes each neuron's blame back along the very edges that carried its activation forward.",
              },
            ],
            insight:
              "<strong>Why it is called back-propagation.</strong> δ is literally propagated backwards through the same graph using the transposed weights. Forward moves activations right; backward moves error signals left.",
          },
          {
            title: `Loss — ${E.LOSSES[cfg.loss].label}`,
            description: E.LOSSES[cfg.loss].suits,
            tex: E.LOSSES[cfg.loss].tex,
            derivation: [
              {
                tex: `L = ${fmt(preview.totalLoss, 6)}`,
                result: U.round(preview.totalLoss, 6),
                note: E.outputShortcut(cfg.layers[outIndex].activation, cfg.loss)
                  ? "Paired with this output activation, the gradient simplifies exactly to a − y."
                  : "Pair this loss with its matching output activation and the gradient simplifies to a − y.",
              },
            ],
            insight:
              "<strong>The loss defines the problem.</strong> Squared error assumes a continuous target; cross-entropy assumes a probability. Choosing the wrong one is a modelling error no amount of training fixes.",
          },
        ]);
      }

      U.renderSteps(
        stepsNode,
        steps.map((entry, index) => `<strong>${index + 1}. ${entry.title}</strong> <span class="mono">[${entry.phase}]</span>`),
        cursor + 1
      );

      drawNetwork(shown, step);
      drawGradients(preview);
      drawLossHistory();

      prevButton.disabled = cursor < 0;
      nextButton.disabled = cursor >= steps.length - 1;
    }

    /* ── Events ───────────────────────────────────────────────── */
    nextButton.addEventListener("click", () => {
      cursor += 1;
      render();
    });
    prevButton.addEventListener("click", () => {
      cursor -= 1;
      render();
    });
    allButton.addEventListener("click", () => {
      cursor = buildSteps().length - 1;
      render();
    });
    epochButton.addEventListener("click", () => {
      const cfg = config();
      syncIO();
      E.trainEpoch(network, cfg);
      E.forwardAll(network, cfg);
      lossHistory.push(E.computeLoss(network, cfg));
      refreshParamEditors();
      cursor = -1;
      render();
    });
    trainButton.addEventListener("click", () => {
      const cfg = config();
      syncIO();
      for (let index = 0; index < 100; index += 1) {
        E.trainEpoch(network, cfg);
        E.forwardAll(network, cfg);
        lossHistory.push(E.computeLoss(network, cfg));
      }
      if (lossHistory.length > 400) lossHistory = lossHistory.slice(-400);
      refreshParamEditors();
      cursor = -1;
      render();
    });
    resetButton.addEventListener("click", () => {
      buildNetwork(true);
      buildParamEditors();
      render();
    });
    lossInput.addEventListener("change", () => {
      cursor = -1;
      render();
    });
    lrInput.addEventListener("input", render);

    buildNetwork(true);
    buildParamEditors();
    U.onRedraw(render);
    render();
  }

  /* ── Geometric transformations ────────────────────────────────
     Ported from an earlier standalone React prototype into the site's
     vanilla engine. The point of the page is the contrast between
     forward mapping (which leaves holes) and inverse mapping (which
     does not), so both are rendered from the same matrix. */
  function mountTransform(rootNode) {
    mountSection(
      rootNode,
      "Affine transformations, forward and inverse",
      "One 3×3 matrix moves, stretches, spins and slants an image. Use the presets to see which entries do what, or type the matrix directly. Then switch the mapping mode: forward mapping tears holes in the result, inverse mapping does not, and the reason is worth understanding.",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-group">
              <label for="tf-preset">Transformation</label>
              <select id="tf-preset">
                <option value="identity">Identity</option>
                <option value="translation">Translation</option>
                <option value="scaling">Scaling</option>
                <option value="rotation">Rotation</option>
                <option value="shearX">Shear X</option>
                <option value="shearY">Shear Y</option>
                <option value="custom">Custom (edit matrix)</option>
              </select>
            </div>
            <div class="control-group" id="tf-param-wrap">
              <label for="tf-param">Amount</label>
              <div class="range-row">
                <input id="tf-param" type="range" min="-180" max="180" step="1" value="35" />
                <span class="range-value" id="tf-param-value">35</span>
              </div>
            </div>
            <div class="control-group">
              <label for="tf-mode">Pixel mapping mode</label>
              <select id="tf-mode">
                <option value="inverse">Inverse mapping (pull) — production standard</option>
                <option value="forward">Forward mapping (push) — shows holes</option>
              </select>
            </div>
            <div class="control-group">
              <label for="tf-shape">Test image</label>
              <select id="tf-shape">
                <option value="letter">Letter F on a checkerboard</option>
                <option value="grid">Fine grid</option>
                <option value="circles">Coloured circles</option>
              </select>
            </div>
            <div class="callout" id="tf-callout"></div>
            ${editorSlot("tf-matrix")}
          </div>
          <div class="two-column">
            <div class="plot-card" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;align-items:start">
              <div>
                <p class="caption" style="margin:0 0 6px;font-weight:600;color:var(--ink)">Source</p>
                <canvas id="tf-source" width="240" height="240" style="width:100%;border-radius:9px;border:1px solid var(--line);image-rendering:pixelated"></canvas>
              </div>
              <div>
                <p class="caption" style="margin:0 0 6px;font-weight:600;color:var(--ink)">Transformed</p>
                <canvas id="tf-dest" width="240" height="240" style="width:100%;border-radius:9px;border:1px solid var(--line);image-rendering:pixelated"></canvas>
              </div>
            </div>
            <div class="plot-card">
              <svg id="tf-plot" viewBox="0 0 560 260" aria-label="Unit square under the transformation"></svg>
              <div class="legend">
                <span><i style="background:#8895a3"></i> Original unit square</span>
                <span><i style="background:#0d7a72"></i> Transformed</span>
              </div>
            </div>
            <div class="readout"><div class="output-grid" id="tf-output"></div></div>
            <div class="equation-card" id="tf-math"></div>
            <div class="steps"><ol id="tf-steps"></ol></div>
          </div>
        </div>
      `
    );

    const presetInput = rootNode.querySelector("#tf-preset");
    const paramInput = rootNode.querySelector("#tf-param");
    const paramValue = rootNode.querySelector("#tf-param-value");
    const paramWrap = rootNode.querySelector("#tf-param-wrap");
    const paramLabel = paramWrap.querySelector("label");
    const modeInput = rootNode.querySelector("#tf-mode");
    const shapeInput = rootNode.querySelector("#tf-shape");
    const callout = rootNode.querySelector("#tf-callout");
    const sourceCanvas = rootNode.querySelector("#tf-source");
    const destCanvas = rootNode.querySelector("#tf-dest");
    const plot = rootNode.querySelector("#tf-plot");
    const output = rootNode.querySelector("#tf-output");
    const mathNode = rootNode.querySelector("#tf-math");
    const stepsNode = rootNode.querySelector("#tf-steps");

    const SIZE = 240;
    let matrix = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    let sourceData = null;
    let suppressEditorSync = false;

    const matrixEditorHandle = U.matrixEditor(rootNode.querySelector("#tf-matrix"), {
      title: "Transformation matrix M",
      hint: "The top-left 2×2 block rotates, scales and shears. The right-hand column translates — that is the whole reason for homogeneous coordinates. The bottom row stays [0 0 1] for an affine map.",
      rowLabels: ["x′", "y′", "w"],
      colLabels: ["x", "y", "1"],
      values: matrix,
      step: 0.1,
      min: -6,
      max: 6,
      onChange: (_, raw) => {
        if (suppressEditorSync) return;
        matrix = raw;
        presetInput.value = "custom";
        syncParamVisibility();
        render();
      },
    });

    const paramConfig = {
      identity: null,
      translation: { min: -80, max: 80, step: 1, value: 40, label: "Translate x and y (pixels)" },
      scaling: { min: 0.2, max: 3, step: 0.1, value: 1.6, label: "Scale factor" },
      rotation: { min: -180, max: 180, step: 1, value: 35, label: "Angle θ (degrees)" },
      shearX: { min: -2, max: 2, step: 0.05, value: 0.6, label: "Horizontal shear" },
      shearY: { min: -2, max: 2, step: 0.05, value: 0.6, label: "Vertical shear" },
      custom: null,
    };

    function syncParamVisibility() {
      const config = paramConfig[presetInput.value];
      paramWrap.style.display = config ? "" : "none";
      if (config) paramLabel.textContent = config.label;
    }

    function applyPreset() {
      const preset = presetInput.value;
      const config = paramConfig[preset];
      if (!config) {
        if (preset === "identity") {
          matrix = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
          ];
        }
      } else {
        const amount = Number(paramInput.value);
        if (preset === "translation") {
          matrix = [
            [1, 0, amount],
            [0, 1, amount * 0.6],
            [0, 0, 1],
          ];
        } else if (preset === "scaling") {
          matrix = [
            [amount, 0, 0],
            [0, amount * 0.7, 0],
            [0, 0, 1],
          ];
        } else if (preset === "rotation") {
          const radians = (amount * Math.PI) / 180;
          matrix = [
            [Math.cos(radians), -Math.sin(radians), 0],
            [Math.sin(radians), Math.cos(radians), 0],
            [0, 0, 1],
          ];
        } else if (preset === "shearX") {
          matrix = [
            [1, amount, 0],
            [0, 1, 0],
            [0, 0, 1],
          ];
        } else {
          matrix = [
            [1, 0, 0],
            [amount, 1, 0],
            [0, 0, 1],
          ];
        }
      }
      suppressEditorSync = true;
      matrixEditorHandle.set(matrix, true);
      suppressEditorSync = false;
    }

    /* Build the source image once per shape choice. */
    function buildSource() {
      const canvas = document.createElement("canvas");
      canvas.width = SIZE;
      canvas.height = SIZE;
      const context = canvas.getContext("2d");
      const shape = shapeInput.value;

      for (let i = 0; i < SIZE; i += 24) {
        for (let j = 0; j < SIZE; j += 24) {
          context.fillStyle = (i + j) % 48 === 0 ? C().faint : "#ffffff";
          context.fillRect(i, j, 24, 24);
        }
      }

      if (shape === "letter") {
        context.fillStyle = C().c;
        context.fillRect(60, 48, 120, 24);
        context.fillRect(60, 48, 24, 144);
        context.fillRect(60, 108, 96, 24);
        context.fillStyle = C().b;
        context.beginPath();
        context.arc(168, 168, 24, 0, Math.PI * 2);
        context.fill();
      } else if (shape === "grid") {
        context.strokeStyle = C().a;
        context.lineWidth = 2;
        for (let i = 12; i < SIZE; i += 24) {
          context.beginPath();
          context.moveTo(i, 0);
          context.lineTo(i, SIZE);
          context.stroke();
          context.beginPath();
          context.moveTo(0, i);
          context.lineTo(SIZE, i);
          context.stroke();
        }
        context.fillStyle = C().b;
        context.fillRect(SIZE / 2 - 6, SIZE / 2 - 6, 12, 12);
      } else {
        const palette = [C().a, C().c, C().b, C().d];
        palette.forEach((color, index) => {
          context.fillStyle = color;
          context.beginPath();
          context.arc(60 + (index % 2) * 120, 60 + Math.floor(index / 2) * 120, 40, 0, Math.PI * 2);
          context.fill();
        });
      }

      sourceData = context.getImageData(0, 0, SIZE, SIZE);
      sourceCanvas.getContext("2d").putImageData(sourceData, 0, 0);
    }

    function invert3x3(m) {
      const det =
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
        m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
        m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
      if (Math.abs(det) < 1e-9) return null;
      const inv = [
        [
          (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det,
          (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det,
          (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det,
        ],
        [
          (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det,
          (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det,
          (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det,
        ],
        [
          (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det,
          (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det,
          (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det,
        ],
      ];
      return { inv, det };
    }

    function renderCanvas() {
      const context = destCanvas.getContext("2d");
      const destImage = context.createImageData(SIZE, SIZE);
      const source = sourceData.data;
      const dest = destImage.data;
      const inverse = invert3x3(matrix);
      const centre = SIZE / 2;
      let holes = 0;

      if (modeInput.value === "forward") {
        /* Push each source pixel to its destination. Rounding collisions
           are exactly what leave gaps. */
        const covered = new Uint8Array(SIZE * SIZE);
        for (let y = 0; y < SIZE; y += 1) {
          for (let x = 0; x < SIZE; x += 1) {
            const cx = x - centre;
            const cy = centre - y;
            const nx = matrix[0][0] * cx + matrix[0][1] * cy + matrix[0][2];
            const ny = matrix[1][0] * cx + matrix[1][1] * cy + matrix[1][2];
            const destX = Math.round(nx + centre);
            const destY = Math.round(centre - ny);
            if (destX < 0 || destX >= SIZE || destY < 0 || destY >= SIZE) continue;
            const sourceIndex = (y * SIZE + x) * 4;
            const destIndex = (destY * SIZE + destX) * 4;
            dest[destIndex] = source[sourceIndex];
            dest[destIndex + 1] = source[sourceIndex + 1];
            dest[destIndex + 2] = source[sourceIndex + 2];
            dest[destIndex + 3] = 255;
            covered[destY * SIZE + destX] = 1;
          }
        }
        /* Count holes only inside the transformed footprint. */
        if (inverse) {
          for (let y = 0; y < SIZE; y += 1) {
            for (let x = 0; x < SIZE; x += 1) {
              if (covered[y * SIZE + x]) continue;
              const cx = x - centre;
              const cy = centre - y;
              const sx = inverse.inv[0][0] * cx + inverse.inv[0][1] * cy + inverse.inv[0][2];
              const sy = inverse.inv[1][0] * cx + inverse.inv[1][1] * cy + inverse.inv[1][2];
              if (Math.abs(sx) < centre && Math.abs(sy) < centre) holes += 1;
            }
          }
        }
      } else if (inverse) {
        /* Pull from the source for every destination pixel. */
        for (let y = 0; y < SIZE; y += 1) {
          for (let x = 0; x < SIZE; x += 1) {
            const cx = x - centre;
            const cy = centre - y;
            const sx = inverse.inv[0][0] * cx + inverse.inv[0][1] * cy + inverse.inv[0][2];
            const sy = inverse.inv[1][0] * cx + inverse.inv[1][1] * cy + inverse.inv[1][2];
            const sourceX = Math.round(sx + centre);
            const sourceY = Math.round(centre - sy);
            if (sourceX < 0 || sourceX >= SIZE || sourceY < 0 || sourceY >= SIZE) continue;
            const sourceIndex = (sourceY * SIZE + sourceX) * 4;
            const destIndex = (y * SIZE + x) * 4;
            dest[destIndex] = source[sourceIndex];
            dest[destIndex + 1] = source[sourceIndex + 1];
            dest[destIndex + 2] = source[sourceIndex + 2];
            dest[destIndex + 3] = 255;
          }
        }
      }

      context.putImageData(destImage, 0, 0);
      return { holes, inverse };
    }

    function render() {
      const config = paramConfig[presetInput.value];
      if (config) paramValue.textContent = String(Number(paramInput.value));

      const { holes, inverse } = renderCanvas();
      const det = inverse ? inverse.det : 0;
      const invertible = inverse !== null;
      const mode = modeInput.value;

      callout.innerHTML = !invertible
        ? `<strong>det M = 0.</strong> This matrix collapses the plane onto a line, so it has no inverse and inverse mapping is impossible. Change a value to make the 2×2 block non-singular.`
        : mode === "forward"
        ? `Forward mapping left <strong>${holes}</strong> unfilled pixel${holes !== 1 ? "s" : ""} inside the transformed region. Scale up or rotate to make the tearing obvious.`
        : `Inverse mapping fills every destination pixel by construction — there are no holes, whatever the matrix.`;

      U.renderMetrics(output, [
        { label: "det M", value: U.round(det, 4) },
        { label: "Area scale", value: `${U.round(Math.abs(det), 3)}×` },
        { label: "Mode", value: mode === "forward" ? "forward (push)" : "inverse (pull)" },
        { label: "Holes", value: mode === "forward" ? String(holes) : "0" },
      ]);

      const m = matrix;
      const texMatrix = (values) =>
        `\\begin{bmatrix} ${values.map((row) => row.map((value) => U.texNum(value, 2)).join(" & ")).join(" \\\\ ")} \\end{bmatrix}`;

      renderFormulaCards(mathNode, [
        {
          title: "Homogeneous coordinates",
          description:
            "Translation is an addition and rotation is a multiplication, which normally cannot live in the same matrix. Appending a 1 to every point fixes that — now a single 3×3 matrix expresses both, and composing transformations becomes matrix multiplication.",
          tex: `\\begin{bmatrix} x' \\\\ y' \\\\ 1 \\end{bmatrix} = ${texMatrix(m)} \\begin{bmatrix} x \\\\ y \\\\ 1 \\end{bmatrix}`,
          derivation: [
            {
              tex: `x' = ${U.texNum(m[0][0], 2)}x + ${U.texNum(m[0][1], 2)}y + ${U.texNum(m[0][2], 2)}`,
              result: "",
            },
            {
              tex: `y' = ${U.texNum(m[1][0], 2)}x + ${U.texNum(m[1][1], 2)}y + ${U.texNum(m[1][2], 2)}`,
              result: "",
              note: "The third column is the translation; the top-left 2×2 block is everything else.",
            },
            {
              tex: `\\text{point } (1, 0) \\mapsto (${U.texNum(m[0][0] + m[0][2], 2)},\\; ${U.texNum(m[1][0] + m[1][2], 2)})`,
              result: "",
              note: "Watch the green square in the chart below — its corners are exactly these mapped points.",
            },
          ],
          insight:
            "<strong>Order matters.</strong> Matrix multiplication does not commute, so rotating then translating is a different transformation from translating then rotating. This is the source of an enormous number of graphics bugs.",
        },
        {
          title: "Determinant and invertibility",
          description:
            "The determinant of the 2×2 block is the factor by which areas change. A negative determinant means the image was flipped; zero means it was flattened onto a line and cannot be recovered.",
          tex: `\\det M = ad - bc = (${U.texNum(m[0][0], 2)})(${U.texNum(m[1][1], 2)}) - (${U.texNum(m[0][1], 2)})(${U.texNum(m[1][0], 2)}) = ${U.texNum(m[0][0] * m[1][1] - m[0][1] * m[1][0], 4)}`,
          derivation: invertible
            ? [
                { tex: `M^{-1} = ${texMatrix(inverse.inv)}`, result: "" },
                {
                  tex: `|\\det M| = ${U.texNum(Math.abs(det), 3)}`,
                  result: U.round(Math.abs(det), 3),
                  note:
                    Math.abs(det) > 1
                      ? "Greater than 1 — the image is enlarged, which is exactly when forward mapping starts leaving gaps."
                      : "At most 1 — the image is shrunk or preserved, so forward mapping has fewer holes to leave.",
                },
              ]
            : [{ tex: "M^{-1} \\text{ does not exist}", result: "" }],
          insight:
            "<strong>Rotation matrices are special.</strong> Their determinant is always exactly 1 and their inverse is simply their transpose — rotation preserves both area and orientation, which is why undoing one is so cheap.",
        },
        {
          title: mode === "forward" ? "Forward mapping (and why it tears)" : "Inverse mapping (and why it is used)",
          description:
            mode === "forward"
              ? "Iterate over source pixels and push each one to its computed destination. When the image is enlarged or rotated, the rounded destinations skip over positions and nothing ever writes to them."
              : "Iterate over destination pixels and pull from the source. Every destination pixel asks 'which source pixel am I?' and gets an answer, so nothing is ever left blank.",
          tex:
            mode === "forward"
              ? "\\mathbf{p}_{\\text{dest}} = \\operatorname{round}(M\\,\\mathbf{p}_{\\text{src}})"
              : "\\mathbf{p}_{\\text{src}} = \\operatorname{round}(M^{-1}\\mathbf{p}_{\\text{dest}})",
          derivation: [
            {
              tex:
                mode === "forward"
                  ? `\\text{unfilled pixels} = ${holes}`
                  : `\\text{unfilled pixels} = 0`,
              result: mode === "forward" ? String(holes) : "0",
              note:
                mode === "forward"
                  ? "Switch the mode selector to inverse mapping and this drops to zero for the same matrix."
                  : "This is why every real image library maps inversely, even though the forward direction is the intuitive one.",
            },
          ],
          insight:
            "<strong>Rounding is the culprit, not the matrix.</strong> Production resamplers go further and interpolate between the four neighbouring source pixels rather than rounding, which is the difference between nearest-neighbour and bilinear filtering.",
        },
      ]);

      U.renderSteps(
        stepsNode,
        [
          `<strong>Write the point in homogeneous form.</strong> (x, y) becomes the column [x, y, 1]ᵀ so translation fits in the matrix.`,
          `<strong>Multiply by M.</strong> The current determinant is <strong>${U.round(det, 4)}</strong>, so areas change by a factor of ${U.round(Math.abs(det), 3)}.`,
          `<strong>Choose a mapping direction.</strong> You are using <strong>${mode === "forward" ? "forward (push)" : "inverse (pull)"}</strong> mapping.`,
          mode === "forward"
            ? `<strong>Observe the artefacts.</strong> ${holes} destination pixel${holes !== 1 ? "s were" : " was"} never written to, leaving visible tears.`
            : `<strong>Every pixel is filled.</strong> Working backwards from the destination guarantees complete coverage.`,
        ],
        4
      );

      /* Unit square before and after — the clearest picture of what a
         matrix actually does to space. */
      const corners = [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
      ];
      const scaleFactor = 60;
      const mapped = corners.map(([x, y]) => [
        m[0][0] * x + m[0][1] * y + m[0][2] / scaleFactor,
        m[1][0] * x + m[1][1] * y + m[1][2] / scaleFactor,
      ]);
      const allX = [...corners.map((c) => c[0]), ...mapped.map((c) => c[0])];
      const allY = [...corners.map((c) => c[1]), ...mapped.map((c) => c[1])];

      /* Equal aspect ratio. On a geometry page a square must look square,
         so the x-range is widened to match the plot area's proportions
         rather than being fitted independently. */
      const pad = 0.6;
      const halfY = Math.max(1.6, ...allY.map(Math.abs), ...allX.map(Math.abs)) + pad;
      const plotW = 560 - 46 - 22;
      const plotH = 260 - 28 - 36;
      const halfX = halfY * (plotW / plotH);
      const chart = U.makeChart(plot, {
        xDomain: [-halfX, halfX],
        yDomain: [-halfY, halfY],
        title: "The unit square under M — corners map to the green shape",
      });
      /* Axes through the origin so the transformation is readable. */
      plot.appendChild(
        U.svgEl("line", { x1: chart.xScale(-halfX), y1: chart.yScale(0), x2: chart.xScale(halfX), y2: chart.yScale(0), class: "axis-line" })
      );
      plot.appendChild(
        U.svgEl("line", { x1: chart.xScale(0), y1: chart.yScale(-halfY), x2: chart.xScale(0), y2: chart.yScale(halfY), class: "axis-line" })
      );

      plot.appendChild(
        U.svgEl("polygon", {
          points: corners.map(([x, y]) => `${chart.xScale(x)},${chart.yScale(y)}`).join(" "),
          fill: U.tint(C().neutral, 0.16),
          stroke: C().neutral,
          "stroke-width": "2",
          "stroke-dasharray": "5 4",
        })
      );
      plot.appendChild(
        U.svgEl("polygon", {
          points: mapped.map(([x, y]) => `${chart.xScale(x)},${chart.yScale(y)}`).join(" "),
          fill: U.tint(C().a, 0.16),
          stroke: C().a,
          "stroke-width": "2.6",
        })
      );
      /* Basis vectors: the columns of M are where î and ĵ land. */
      [
        { from: [0, 0], to: mapped[1], color: C().b, label: "î → column 1" },
        { from: [0, 0], to: mapped[3], color: C().c, label: "ĵ → column 2" },
      ].forEach((arrow) => {
        plot.appendChild(
          U.svgEl("line", {
            x1: chart.xScale(arrow.from[0]),
            y1: chart.yScale(arrow.from[1]),
            x2: chart.xScale(arrow.to[0]),
            y2: chart.yScale(arrow.to[1]),
            stroke: arrow.color,
            "stroke-width": "3",
          })
        );
        plot.appendChild(
          U.svgEl("text", {
            x: chart.xScale(arrow.to[0]) + 8,
            y: chart.yScale(arrow.to[1]) - 6,
            class: "svg-label",
          })
        ).textContent = arrow.label;
      });
    }

    presetInput.addEventListener("change", () => {
      const config = paramConfig[presetInput.value];
      if (config) {
        paramInput.min = config.min;
        paramInput.max = config.max;
        paramInput.step = config.step;
        paramInput.value = config.value;
      }
      syncParamVisibility();
      applyPreset();
      render();
    });
    paramInput.addEventListener("input", () => {
      applyPreset();
      render();
    });
    modeInput.addEventListener("change", render);
    shapeInput.addEventListener("change", () => {
      buildSource();
      render();
    });

    presetInput.value = "rotation";
    const initial = paramConfig.rotation;
    paramInput.min = initial.min;
    paramInput.max = initial.max;
    paramInput.step = initial.step;
    paramInput.value = initial.value;
    syncParamVisibility();
    applyPreset();
    buildSource();
    U.onRedraw(render);
    render();
  }


  renderShell();
  const workspace = root.querySelector("#algorithm-workspace");
  const engines = {
    "concept-overview": mountConceptOverview,
    "concept-forms": mountConceptForms,
    "concept-tools": mountConceptTools,
    gaussian: mountGaussian,
    "bias-variance": mountBiasVariance,
    cleanup: mountCleanup,
    kde: mountKde,
    knn: mountKnn,
    tree: mountTree,
    linear: mountLinear,
    "bayes-net": mountBayesNet,
    mrf: mountMRF,
    "belief-propagation": mountBeliefPropagation,
    markov: mountMarkov,
    hmm: mountHmm,
    partition: mountPartition,
    hierarchy: mountHierarchy,
    distance: mountDistance,
    dbscan: mountDbscan,
    spectral: mountSpectral,
    activation: mountActivation,
    "gradient-descent": mountGradientDescent,
    backprop: mountBackprop,
    transform: mountTransform,
  };
  /* Labs that live in their own files (transformer-labs.js) register on
     window.MLExtraLabs and receive the shared rendering helpers. */
  Object.entries(window.MLExtraLabs || {}).forEach(([name, mount]) => {
    if (!engines[name]) engines[name] = (node) => mount(node, { renderFormulaCards });
  });
  if (engines[definition.engine]) {
    engines[definition.engine](workspace);
  } else {
    workspace.innerHTML = `<section class="lab"><p class="section-intro">The renderer for <strong>${definition.engine}</strong> has not been attached yet.</p></section>`;
  }
})();
