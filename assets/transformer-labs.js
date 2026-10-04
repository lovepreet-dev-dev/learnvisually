/* ── Attention and Transformer labs ───────────────────────────────
   Two concept pages that follow the original paper, "Attention Is All
   You Need" (Vaswani et al., 2017).

   These labs work differently from the rest of the site. A transformer
   has too many matrices for "type your own data" to stay readable, so
   the reader explores instead of edits: click a word, step through a
   stage, move one slider. Underneath is a real (tiny) model — 4-number
   word vectors, 2 attention heads — with weights chosen by hand so the
   patterns mean something ("it" really does look at "cat"). Every
   number on screen is still computed live from those weights.

   The maths and the toy model are plain functions with no DOM access,
   exposed as window.MLTransformerMath so they can be checked in Node. */
(function () {
  const root = typeof window !== "undefined" ? window : globalThis;

  /* ── Linear algebra on plain arrays ───────────────────────────── */

  const matmul = (A, B) =>
    A.map((row) => B[0].map((_, j) => row.reduce((acc, value, k) => acc + value * B[k][j], 0)));
  const transpose = (A) => A[0].map((_, j) => A.map((row) => row[j]));
  const addM = (A, B) => A.map((row, i) => row.map((value, j) => value + B[i][j]));
  const mapM = (A, fn) => A.map((row, i) => row.map((value, j) => fn(value, i, j)));
  const dotV = (a, b) => a.reduce((acc, value, i) => acc + value * b[i], 0);
  const relu = (value) => Math.max(0, value);

  /* -Infinity entries are masked positions: they get exactly zero weight. */
  function softmax(row) {
    const finite = row.filter(Number.isFinite);
    const peak = finite.length ? Math.max(...finite) : 0;
    const exps = row.map((value) => (Number.isFinite(value) ? Math.exp(value - peak) : 0));
    const total = exps.reduce((acc, value) => acc + value, 0) || 1;
    return exps.map((value) => value / total);
  }

  function layerNorm(row, eps = 1e-5) {
    const mean = row.reduce((acc, value) => acc + value, 0) / row.length;
    const variance = row.reduce((acc, value) => acc + (value - mean) ** 2, 0) / row.length;
    return row.map((value) => (value - mean) / Math.sqrt(variance + eps));
  }

  /* Section 3.5 of the paper: even dimensions get a sine, odd ones a
     cosine, and each pair turns at its own speed. */
  function positionalEncoding(pos, dModel) {
    const out = [];
    for (let i = 0; i < dModel; i += 1) {
      const pair = Math.floor(i / 2);
      const angle = pos / Math.pow(10000, (2 * pair) / dModel);
      out.push(i % 2 === 0 ? Math.sin(angle) : Math.cos(angle));
    }
    return out;
  }

  function seeded(seed) {
    let state = seed >>> 0;
    return () => {
      state = (state + 0x6d2b79f5) >>> 0;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussian(rng) {
    const u = Math.max(rng(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  }

  function randomMatrix(rows, cols, rng, scale) {
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => Math.round(gaussian(rng) * scale * 100) / 100)
    );
  }

  /* ── The toy vocabulary ───────────────────────────────────────
     Real embeddings have hundreds of unnamed dimensions. Ours have four,
     and each one is given a meaning so the reader can watch information
     move: after attention, "it" should have picked up "alive" from "cat". */

  const FEATURES = ["thing", "alive", "action/state", "refers back"];

  const VOCAB = {
    the: [0.2, 0, 0, 0],
    cat: [1, 1, 0, 0],
    dog: [1, 1, 0, 0],
    ball: [1, 0, 0, 0],
    sat: [0, 0, 1, 0],
    rolled: [0, 0, 1, 0],
    chased: [0, 0, 1, 0],
    because: [0, 0, 0.2, 0],
    it: [0.3, 0, 0, 1],
    was: [0, 0, 0.6, 0],
    tired: [0, 0.3, 0.8, 0],
    round: [0, 0, 0.6, 0],
    hungry: [0, 0.3, 0.8, 0],
  };

  const SENTENCES = [
    {
      id: "cat",
      text: "the cat sat because it was tired",
      focus: 4,
      lesson: "“it” has to find out what it refers to. Only one living thing is in the sentence, so the answer is clear.",
    },
    {
      id: "ball",
      text: "the ball rolled because it was round",
      focus: 4,
      lesson: "“it” still finds the noun, but less confidently: this head was built to look for <em>living</em> things, and a ball is not alive.",
    },
    {
      id: "dog",
      text: "the dog chased the cat because it was hungry",
      focus: 6,
      lesson: "Two living things. Head 1 cannot decide between them and splits its attention. Real models handle this with more heads, more layers and word order.",
    },
  ];

  const tokensOf = (sentence) => sentence.text.split(" ");
  const embed = (tokens) => tokens.map((token) => [...VOCAB[token]]);

  /* Two hand-built heads. Weight matrices are 4×2 (d_model = 4, d_k = 2),
     matching the paper's rule d_k = d_model / h with h = 2. Each row of a
     W matrix says how much one input feature feeds each output number. */
  const HEADS = [
    {
      name: "Head 1 · “who is involved?”",
      short: "who?",
      WQ: [[0, 0], [0, 0], [1.2, 0.4], [2.2, 1.0]],
      WK: [[0.3, 1.0], [1.6, 0.2], [0, 0], [0, 0]],
      WV: [[1, 0], [0, 1], [0, 0], [0, 0]],
      kAxes: ["key 1 ≈ “I am alive”", "key 2 ≈ “I am a thing”"],
      vLabels: ["thing", "alive"],
      idea: "Words that refer back (“it”) or describe a state (“sat”, “tired”) ask for a living thing. Nouns answer, and living nouns answer loudest.",
    },
    {
      name: "Head 2 · “what happened?”",
      short: "what?",
      WQ: [[3, 0], [0, 0], [0, 0], [3, 0]],
      WK: [[0, 0], [0, 0], [1, 0], [0, 0]],
      WV: [[0, 0], [0, 0], [1, 0], [0, 1]],
      kAxes: ["key 1 ≈ “I am an action”", "key 2 (unused)"],
      vLabels: ["action/state", "refers back"],
      idea: "Nouns and pronouns ask for actions. Every verb answers, and this head has no way to tell which action belongs to which noun.",
    },
  ];

  /* Output projection. A trained W_O mixes the heads; identity keeps the
     four named features readable after concatenation. */
  const W_O = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];

  function attentionHead(X, head) {
    const dk = head.WK[0].length;
    const Q = matmul(X, head.WQ);
    const K = matmul(X, head.WK);
    const V = matmul(X, head.WV);
    const S = matmul(Q, transpose(K));
    const scaled = mapM(S, (value) => value / Math.sqrt(dk));
    const A = scaled.map(softmax);
    const O = matmul(A, V);
    return { Q, K, V, S, scaled, A, O, dk };
  }

  function multiHead(X) {
    const heads = HEADS.map((head) => attentionHead(X, head));
    const concat = X.map((_, i) => heads.flatMap((result) => result.O[i]));
    const out = matmul(concat, W_O);
    return { heads, concat, out };
  }

  /* Step 1's idea before Q/K/V exist: score words by raw similarity. */
  function rawAttention(X, i, sharpness = 1) {
    const scores = X.map((row) => dotV(X[i], row) * sharpness);
    const weights = softmax(scores);
    const out = X[0].map((_, d) => X.reduce((acc, row, j) => acc + weights[j] * row[d], 0));
    return { scores, weights, out };
  }

  /* ── One full encoder layer (section 3.1) ──────────────────────
     Attention → add → norm → feed-forward → add → norm. The FFN weights
     are random but seeded so the page is identical on every visit. */
  const D_FF = 8;
  const ffnRng = seeded(11);
  const FFN = {
    W1: randomMatrix(4, D_FF, ffnRng, 0.8),
    b1: Array.from({ length: D_FF }, () => Math.round(gaussian(ffnRng) * 20) / 100),
    W2: randomMatrix(D_FF, 4, ffnRng, 0.5),
    b2: [0, 0, 0, 0],
  };

  function encoderLayer(X) {
    const mha = multiHead(X);
    const added1 = addM(X, mha.out);
    const norm1 = added1.map((row) => layerNorm(row));
    const hidden = matmul(norm1, FFN.W1).map((row) => row.map((value, j) => relu(value + FFN.b1[j])));
    const ffn = matmul(hidden, FFN.W2).map((row) => row.map((value, j) => value + FFN.b2[j]));
    const added2 = addM(norm1, ffn);
    const out = added2.map((row) => layerNorm(row));
    return { X, mha, added1, norm1, hidden, ffn, added2, out };
  }

  /* ── Word-order demo (permutation equivariance) ──────────────────
     Plain self-attention treats a sentence as a bag of words. Feed it the
     same words in a different order and every word gets exactly the same
     output; adding positional encodings breaks the tie. */
  const ORDER_VOCAB = {
    dog: [0.9, 0.1, 0.6, 0.2],
    bites: [0.1, 0.9, 0.1, 0.7],
    man: [0.8, 0.2, 0.1, 0.9],
  };
  const orderRng = seeded(5);
  const ORDER_W = {
    WQ: randomMatrix(4, 4, orderRng, 0.9),
    WK: randomMatrix(4, 4, orderRng, 0.9),
    WV: randomMatrix(4, 4, orderRng, 0.9),
  };

  function orderDemo(tokens, withPositions) {
    const X = tokens.map((token, pos) => {
      const base = ORDER_VOCAB[token];
      return withPositions ? addM([base], [positionalEncoding(pos, 4)])[0] : [...base];
    });
    const head = attentionHead(X, ORDER_W);
    return { X, A: head.A, O: head.O };
  }

  /* ── Context demo (2 features, so it can be drawn) ─────────────── */
  const CONTEXT_VOCAB = {
    bank: [0.7, 0.7],
    river: [0.1, 1.0],
    muddy: [0.2, 0.8],
    loan: [1.0, 0.12],
    gave: [0.45, 0.2],
    a: [0.1, 0.1],
    the: [0.15, 0.12],
    was: [0.18, 0.2],
    i: [0.2, 0.25],
    sat: [0.25, 0.4],
    on: [0.12, 0.18],
  };

  const CONTEXT_SENTENCES = [
    { id: "river", text: "the river bank was muddy", label: "river bank" },
    { id: "loan", text: "the bank gave a loan", label: "bank loan" },
    { id: "vague", text: "i sat on the bank", label: "no clue" },
  ];

  /* ── Decoder demo: English → German, as in the paper's WMT task ──── */
  const SOURCE = ["I", "love", "cats"];
  const TARGET = ["ich", "liebe", "Katzen"];
  const DECODER_INPUT = ["<s>", "ich", "liebe", "Katzen"];

  /* Masked self-attention scores, hand-set so that without the mask each
     row leans on the word it is supposed to predict next. */
  const SELF_SCORES = [
    [1.0, 2.2, 0.6, 0.4],
    [0.6, 1.0, 2.4, 0.5],
    [0.4, 0.9, 1.0, 2.5],
    [0.3, 0.7, 1.2, 1.4],
  ];

  /* Cross-attention scores: decoder position (rows) against English words. */
  const CROSS_SCORES = [
    [3.0, 0.5, 0.2],
    [0.4, 3.0, 0.6],
    [0.2, 0.7, 3.0],
    [0.6, 0.6, 1.6],
  ];

  const OUT_VOCAB = ["ich", "du", "liebe", "hasse", "Katzen", "Hunde", "</s>"];
  /* Illustrative next-word probabilities for each generation step. */
  const NEXT_PROBS = [
    [0.82, 0.07, 0.02, 0.02, 0.03, 0.02, 0.02],
    [0.02, 0.03, 0.76, 0.12, 0.03, 0.02, 0.02],
    [0.02, 0.01, 0.02, 0.01, 0.79, 0.13, 0.02],
    [0.01, 0.01, 0.02, 0.01, 0.03, 0.02, 0.9],
  ];

  function maskedScores(scores, masked) {
    return scores.map((row, i) => row.map((value, j) => (masked && j > i ? -Infinity : value)));
  }

  /* Equation 3: linear warm-up, then decay with the inverse square root. */
  function learningRate(step, dModel, warmup) {
    const s = Math.max(step, 1);
    return Math.pow(dModel, -0.5) * Math.min(Math.pow(s, -0.5), s * Math.pow(warmup, -1.5));
  }

  const math = {
    matmul,
    transpose,
    softmax,
    layerNorm,
    positionalEncoding,
    dotV,
    seeded,
    gaussian,
    attentionHead,
    multiHead,
    rawAttention,
    encoderLayer,
    orderDemo,
    maskedScores,
    learningRate,
    embed,
    tokensOf,
    FEATURES,
    VOCAB,
    SENTENCES,
    HEADS,
    CONTEXT_VOCAB,
    CONTEXT_SENTENCES,
    CROSS_SCORES,
    SELF_SCORES,
    NEXT_PROBS,
    OUT_VOCAB,
    FFN,
  };
  root.MLTransformerMath = math;

  if (typeof document === "undefined") return;

  /* ── Shared rendering helpers ─────────────────────────────────── */

  const U = () => root.MLUtils;
  const C = () => root.MLUtils.chartColors();

  /* Format with a real minus sign, and never print "-0.00". */
  function fmt(value, digits = 2) {
    if (value === -Infinity) return "−∞";
    if (!Number.isFinite(value)) return "—";
    const limit = 0.5 * Math.pow(10, -digits);
    if (Math.abs(value) < limit) return (0).toFixed(digits);
    return (value < 0 ? "−" : "") + Math.abs(value).toFixed(digits);
  }

  const pct = (value) => `${Math.round(value * 100)}%`;

  /* Colours come from CSS variables, which may be hex or rgb(); resolve
     each to its channels once per colour string. */
  const rgbCache = {};
  function rgbOf(color) {
    if (rgbCache[color]) return rgbCache[color];
    let channels = null;
    const hex = String(color).trim().match(/^#([0-9a-f]{6})$/i);
    if (hex) {
      const n = parseInt(hex[1], 16);
      channels = [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    } else {
      const digits = U().tint(color, 1).match(/\d+/g);
      channels = digits ? digits.slice(0, 3).map(Number) : [13, 122, 114];
    }
    rgbCache[color] = channels;
    return channels;
  }

  const rgba = (color, alpha) => `rgba(${rgbOf(color).join(", ")}, ${alpha})`;

  /* Fill for a matrix cell: teal for positive, orange for negative,
     opacity by magnitude. */
  function cellStyle(value, maxAbs) {
    if (!Number.isFinite(value)) return "";
    const strength = Math.min(1, Math.abs(value) / (maxAbs || 1));
    const alpha = 0.06 + strength * 0.82;
    const color = value >= 0 ? C().a : C().b;
    const ink = alpha > 0.55 ? `color:${C().onFill};` : "";
    return `background:${rgba(color, alpha)};${ink}`;
  }

  function maxAbsOf(matrix) {
    let peak = 0;
    matrix.forEach((row) =>
      row.forEach((value) => {
        if (Number.isFinite(value)) peak = Math.max(peak, Math.abs(value));
      })
    );
    return peak || 1;
  }

  /* An n×m matrix as a coloured grid of numbers. Row labels are buttons
     when the caller wants rows to be selectable. */
  function heatmap(matrix, options = {}) {
    const {
      rows = matrix.map((_, i) => String(i + 1)),
      cols = matrix[0].map((_, j) => String(j + 1)),
      focusRow = -1,
      digits = 2,
      maxAbs = maxAbsOf(matrix),
      title = "",
      shape = "",
      rowAttr = "",
      colHint = "",
    } = options;
    const head = cols.map((label) => `<div class="tl-heat-col">${label}</div>`).join("");
    const body = matrix
      .map((row, i) => {
        const focused = i === focusRow ? " is-focus" : "";
        const label = rowAttr
          ? `<button type="button" class="tl-heat-row${focused}" ${rowAttr}="${i}">${rows[i]}</button>`
          : `<div class="tl-heat-row${focused}">${rows[i]}</div>`;
        const cells = row
          .map((value) => {
            const masked = value === -Infinity ? " is-masked" : "";
            return `<div class="tl-heat-cell${focused}${masked}" style="${cellStyle(value, maxAbs)}">${fmt(value, digits)}</div>`;
          })
          .join("");
        return label + cells;
      })
      .join("");
    return `
      <figure class="tl-heat-wrap">
        ${title ? `<figcaption><span>${title}</span>${shape ? `<span class="tl-shape-tag">${shape}</span>` : ""}</figcaption>` : ""}
        ${colHint ? `<div class="tl-heat-hint">${colHint}</div>` : ""}
        <div class="tl-heat" style="grid-template-columns: max-content repeat(${cols.length}, minmax(34px, 1fr))">
          <div></div>${head}${body}
        </div>
      </figure>`;
  }

  /* Horizontal bars, one per label, for weights and probabilities. */
  function barList(values, labels, options = {}) {
    const { max = Math.max(...values, 1e-9), highlight = -1, color = C().a, digits = 2, asPercent = false } = options;
    return `<div class="tl-bars">${values
      .map((value, i) => {
        const width = Math.max(0, Math.min(100, (value / max) * 100));
        const hot = i === highlight ? " is-hot" : "";
        return `
          <div class="tl-bar-row${hot}">
            <span class="tl-bar-label">${labels[i]}</span>
            <span class="tl-bar-track"><span class="tl-bar-fill" style="width:${width}%;background:${color}"></span></span>
            <span class="tl-bar-value">${asPercent ? pct(value) : fmt(value, digits)}</span>
          </div>`;
      })
      .join("")}</div>`;
  }

  /* Signed bars centred on zero, used to compare a word's vector before
     and after a stage. `series` is [{name, values, color}]. */
  function featureCompare(labels, series, maxAbs = 1.2) {
    return `<div class="tl-features">${labels
      .map(
        (label, d) => `
          <div class="tl-feature">
            <span class="tl-feature-name">${label}</span>
            <div class="tl-feature-bars">${series
              .map((entry) => {
                const value = entry.values[d];
                const width = Math.min(50, (Math.abs(value) / maxAbs) * 50);
                const left = value >= 0 ? 50 : 50 - width;
                return `
                  <div class="tl-feature-line" title="${entry.name}: ${fmt(value)}">
                    <span class="tl-feature-track"><span class="tl-feature-fill" style="left:${left}%;width:${width}%;background:${entry.color}"></span></span>
                    <span class="tl-feature-value">${fmt(value)}</span>
                  </div>`;
              })
              .join("")}</div>
          </div>`
      )
      .join("")}
      <div class="legend">${series.map((entry) => `<span><i style="background:${entry.color}"></i> ${entry.name}</span>`).join("")}</div>
    </div>`;
  }

  /* The sentence itself, each word tinted by how much attention the
     focus word gives it. Clicking a word makes it the focus. */
  function sentenceStrip(tokens, focus, weights, attr, options = {}) {
    const { label = "", showPct = true } = options;
    return `<div class="tl-strip">${label ? `<span class="tl-strip-label">${label}</span>` : ""}${tokens
      .map((token, i) => {
        const w = weights ? weights[i] : 0;
        const style = weights ? `style="${cellStyle(w, 1)}"` : "";
        const focused = i === focus ? " is-focus" : "";
        const badge = weights && showPct ? `<small>${pct(w)}</small>` : "";
        const tag = attr ? "button" : "span";
        const extra = attr ? `type="button" ${attr}="${i}"` : "";
        return `<${tag} class="tl-token${focused}" ${extra} ${style}>${token}${badge}</${tag}>`;
      })
      .join("")}</div>`;
  }

  function quiz(items) {
    return `<div class="tl-quiz">${items
      .map((item) => `<details><summary>${item.q}</summary><div>${item.a}</div></details>`)
      .join("")}</div>`;
  }

  /* Delegated click handling: a single listener per section, keyed on a
     data attribute, so re-rendered markup needs no rebinding. */
  function onClickAttr(node, attr, handler) {
    node.addEventListener("click", (event) => {
      const target = event.target.closest(`[${attr}]`);
      if (target && node.contains(target)) handler(Number(target.getAttribute(attr)), target);
    });
  }

  function svgText(svg, x, y, text, attrs = {}) {
    const node = U().svgEl("text", Object.assign({ x, y, class: "svg-label" }, attrs));
    node.textContent = text;
    svg.appendChild(node);
    return node;
  }

  function arrowMarker(svg, id, color) {
    const defs = U().svgEl("defs");
    const marker = U().svgEl("marker", {
      id,
      viewBox: "0 0 10 10",
      refX: 9,
      refY: 5,
      markerWidth: 7,
      markerHeight: 7,
      orient: "auto-start-reverse",
    });
    marker.appendChild(U().svgEl("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: color }));
    defs.appendChild(marker);
    svg.appendChild(defs);
  }

  function clipToChart(svg, chart, id) {
    const defs = U().svgEl("defs");
    const clip = U().svgEl("clipPath", { id });
    clip.appendChild(
      U().svgEl("rect", {
        x: chart.padding.left,
        y: chart.padding.top,
        width: chart.width - chart.padding.left - chart.padding.right,
        height: chart.height - chart.padding.top - chart.padding.bottom,
      })
    );
    defs.appendChild(clip);
    svg.appendChild(defs);
  }

  const tex = (latex, display = false) => U().tex(latex, display);

  /* ── Figures reused from the first version of these pages ── */
  function paintSwatches(node) {
    node.querySelectorAll("[data-swatch]").forEach((swatch) => {
      swatch.style.background = C()[swatch.getAttribute("data-swatch")];
    });
  }

  function drawContextPlot(svg, tokens, X, focus, weights, after) {
    const chart = U().makeChart(svg, {
      xDomain: [0, 1.25],
      yDomain: [0, 1.25],
      title: "Each word as a 2-number vector",
    });
    svgText(svg, chart.xScale(1.25) - 4, chart.yScale(0) - 8, "money-ness →", { "text-anchor": "end" });
    svgText(svg, chart.padding.left + 6, chart.padding.top + 12, "↑ nature-ness");
    arrowMarker(svg, "tl-ctx-arrow", C().b);

    const origin = X[focus];
    tokens.forEach((token, j) => {
      if (j === focus) return;
      svg.appendChild(
        U().svgEl("line", {
          x1: chart.xScale(origin[0]),
          y1: chart.yScale(origin[1]),
          x2: chart.xScale(X[j][0]),
          y2: chart.yScale(X[j][1]),
          stroke: C().a,
          "stroke-width": 1 + weights[j] * 14,
          opacity: 0.18 + weights[j] * 0.7,
          "stroke-linecap": "round",
        })
      );
    });

    const placed = {};
    tokens.forEach((token, j) => {
      if (j === focus) return;
      const key = `${X[j][0]},${X[j][1]}`;
      const offset = (placed[key] = (placed[key] || 0) + 1) - 1;
      svg.appendChild(
        U().svgEl("circle", { cx: chart.xScale(X[j][0]), cy: chart.yScale(X[j][1]), r: 5, fill: C().neutral })
      );
      svgText(svg, chart.xScale(X[j][0]) + 8, chart.yScale(X[j][1]) - 6 + offset * 13, token);
    });

    svg.appendChild(
      U().svgEl("line", {
        x1: chart.xScale(origin[0]),
        y1: chart.yScale(origin[1]),
        x2: chart.xScale(after[0]),
        y2: chart.yScale(after[1]),
        stroke: C().b,
        "stroke-width": 2.5,
        "marker-end": "url(#tl-ctx-arrow)",
      })
    );
    svg.appendChild(
      U().svgEl("circle", {
        cx: chart.xScale(origin[0]),
        cy: chart.yScale(origin[1]),
        r: 8,
        fill: C().plotBg,
        stroke: C().c,
        "stroke-width": 3,
      })
    );
    svgText(svg, chart.xScale(origin[0]) + 11, chart.yScale(origin[1]) + 16, `${tokens[focus]} (before)`, {
      class: "svg-title",
    });
    svg.appendChild(
      U().svgEl("circle", { cx: chart.xScale(after[0]), cy: chart.yScale(after[1]), r: 6, fill: C().b })
    );
    svgText(svg, chart.xScale(after[0]), chart.yScale(after[1]) + 22, `${tokens[focus]} (after)`, {
      "text-anchor": "middle",
      class: "svg-title",
    });
  }

  function drawQueryKeyPlot(svg, tokens, result, focus) {
    const chart = U().makeChart(svg, {
      xDomain: [-0.4, 2.6],
      yDomain: [-0.4, 2.6],
      title: "Query (arrow) and keys (dots), head 1",
      padding: { left: 40, bottom: 34 },
    });
    svgText(svg, chart.xScale(2.6) - 4, chart.yScale(-0.4) - 8, "key 1 ≈ “I am alive” →", { "text-anchor": "end" });
    svgText(svg, chart.padding.left + 6, chart.padding.top + 12, "↑ key 2 ≈ “I am a thing”");
    clipToChart(svg, chart, "tl-qk-clip");
    arrowMarker(svg, "tl-qk-arrow", C().b);

    const q = result.Q[focus];
    const weights = result.A[focus];
    const qLen = Math.hypot(q[0], q[1]);
    const best = weights.indexOf(Math.max(...weights));

    /* Every point on a line perpendicular to the query gets the same
       score, which is a geometric way to read a dot product. */
    if (qLen > 1e-9) {
      const k = result.K[best];
      const score = q[0] * k[0] + q[1] * k[1];
      const base = [(q[0] * score) / (qLen * qLen), (q[1] * score) / (qLen * qLen)];
      const dir = [-q[1] / qLen, q[0] / qLen];
      svg.appendChild(
        U().svgEl("line", {
          x1: chart.xScale(base[0] - dir[0] * 5),
          y1: chart.yScale(base[1] - dir[1] * 5),
          x2: chart.xScale(base[0] + dir[0] * 5),
          y2: chart.yScale(base[1] + dir[1] * 5),
          stroke: C().b,
          "stroke-dasharray": "5 5",
          opacity: 0.45,
          "clip-path": "url(#tl-qk-clip)",
        })
      );
    }

    const groups = {};
    tokens.forEach((token, j) => {
      const k = result.K[j];
      const key = `${k[0].toFixed(2)},${k[1].toFixed(2)}`;
      if (!groups[key]) groups[key] = { k, names: [], weight: 0 };
      if (!groups[key].names.includes(token)) groups[key].names.push(token);
      groups[key].weight += weights[j];
    });
    Object.values(groups).forEach((group) => {
      const x = chart.xScale(group.k[0]);
      const y = chart.yScale(group.k[1]);
      svg.appendChild(
        U().svgEl("circle", { cx: x, cy: y, r: 4 + group.weight * 10, fill: C().a, opacity: 0.85 })
      );
      svgText(svg, x + 8 + group.weight * 8, y + 4, group.names.join(", "));
    });

    if (qLen > 1e-9) {
      const scale = Math.min(1, 2.4 / Math.max(Math.abs(q[0]), Math.abs(q[1])));
      svg.appendChild(
        U().svgEl("line", {
          x1: chart.xScale(0),
          y1: chart.yScale(0),
          x2: chart.xScale(q[0] * scale),
          y2: chart.yScale(q[1] * scale),
          stroke: C().b,
          "stroke-width": 3,
          "marker-end": "url(#tl-qk-arrow)",
        })
      );
      /* Label below the middle of the arrow, clear of the keys it points at. */
      svgText(svg, chart.xScale((q[0] * scale) / 2) + 6, chart.yScale((q[1] * scale) / 2) + 20, `query of “${tokens[focus]}”`, {
        class: "svg-title",
      });
    } else {
      svgText(svg, chart.xScale(1.2), chart.yScale(2.2), `“${tokens[focus]}” has no query in this head`, {
        "text-anchor": "middle",
        class: "svg-title",
      });
    }
  }

  function drawScaleBars(svg, wRaw, wScaled) {
    U().clear(svg);
    const width = 560;
    const height = 250;
    const pad = { left: 44, right: 16, top: 26, bottom: 30 };
    const plotH = height - pad.top - pad.bottom;
    const slotW = (width - pad.left - pad.right) / wRaw.length;
    const y = (value) => pad.top + plotH * (1 - value);
    svgText(svg, pad.left, 16, "Attention weight given to each of 8 random keys", { class: "svg-title" });
    [0, 0.25, 0.5, 0.75, 1].forEach((tick) => {
      svg.appendChild(U().svgEl("line", { x1: pad.left, x2: width - pad.right, y1: y(tick), y2: y(tick), class: "grid-line" }));
      svgText(svg, pad.left - 8, y(tick) + 4, `${Math.round(tick * 100)}%`, { "text-anchor": "end" });
    });
    wRaw.forEach((value, i) => {
      const x0 = pad.left + i * slotW + slotW * 0.14;
      const barW = slotW * 0.34;
      svg.appendChild(U().svgEl("rect", { x: x0, y: y(value), width: barW, height: Math.max(0.5, plotH * value), fill: C().b, rx: 2 }));
      svg.appendChild(
        U().svgEl("rect", { x: x0 + barW + 2, y: y(wScaled[i]), width: barW, height: Math.max(0.5, plotH * wScaled[i]), fill: C().a, rx: 2 })
      );
      svgText(svg, pad.left + i * slotW + slotW / 2, height - 10, `key ${i + 1}`, { "text-anchor": "middle" });
    });
  }

  function drawArchitecture(svg, selected) {
    U().clear(svg);
    const c = C();
    const kindColor = {
      embed: c.neutral,
      pe: c.neutral,
      mha: c.b,
      masked: c.b,
      cross: c.b,
      addnorm: c.d,
      ffn: c.c,
      linear: c.a,
      softmax: c.a,
      stack: c.faint,
    };
    arrowMarker(svg, "tl-map-arrow", c.axis);

    const line = (points, extra = {}) =>
      svg.appendChild(
        U().svgEl("polyline", Object.assign({
          points: points.map((p) => p.join(",")).join(" "),
          fill: "none",
          stroke: c.axis,
          "stroke-width": 1.6,
          "marker-end": "url(#tl-map-arrow)",
        }, extra))
      );

    /* Frames (the N× stacks) go first so boxes sit on top of them. */
    const frame = (x, y, w, h, labelX) => {
      const group = U().svgEl("g", { "data-part": "stack", tabindex: 0, role: "button", class: "tl-map-part", "aria-label": "N× stack" });
      group.appendChild(
        U().svgEl("rect", {
          x, y, width: w, height: h, rx: 16,
          fill: rgba(c.faint, selected === "stack" ? 0.28 : 0.14),
          stroke: selected === "stack" ? c.ink : c.faint,
          "stroke-width": selected === "stack" ? 2.5 : 1.4,
        })
      );
      const label = U().svgEl("text", { x: labelX, y: y + h / 2 + 5, class: "svg-title", "text-anchor": "middle" });
      label.textContent = "N×";
      group.appendChild(label);
      svg.appendChild(group);
    };
    frame(70, 268, 240, 232, 46);
    frame(410, 176, 240, 324, 676);

    const box = (part, x, y, label, w = 190, h = 34) => {
      const color = kindColor[part];
      const isOn = selected === part;
      const group = U().svgEl("g", { "data-part": part, tabindex: 0, role: "button", class: "tl-map-part", "aria-label": label });
      group.appendChild(
        U().svgEl("rect", {
          x, y, width: w, height: h, rx: 8,
          fill: rgba(color, isOn ? 0.42 : 0.16),
          stroke: isOn ? c.ink : color,
          "stroke-width": isOn ? 2.6 : 1.4,
        })
      );
      const text = U().svgEl("text", { x: x + w / 2, y: y + h / 2 + 4.5, "text-anchor": "middle", class: "tl-map-label" });
      text.textContent = label;
      group.appendChild(text);
      svg.appendChild(group);
    };

    const plus = (cx, cy) => {
      const isOn = selected === "pe";
      const group = U().svgEl("g", { "data-part": "pe", tabindex: 0, role: "button", class: "tl-map-part", "aria-label": "Positional encoding" });
      group.appendChild(U().svgEl("circle", { cx, cy, r: 12, fill: rgba(c.neutral, isOn ? 0.42 : 0.12), stroke: isOn ? c.ink : c.neutral, "stroke-width": isOn ? 2.6 : 1.4 }));
      const t = U().svgEl("text", { x: cx, y: cy + 5, "text-anchor": "middle", class: "tl-map-label" });
      t.textContent = "+";
      group.appendChild(t);
      svg.appendChild(group);
    };

    /* ── Encoder (centre x = 190) ── */
    line([[190, 612], [190, 602]]);
    line([[190, 566], [190, 544]]);
    line([[190, 518], [190, 484]]);
    line([[190, 448], [190, 440]]);
    line([[190, 404], [190, 376]]);
    line([[190, 340], [190, 332]]);
    line([[190, 505], [84, 505], [84, 421], [93, 421]]);
    line([[190, 392], [84, 392], [84, 313], [93, 313]]);
    /* Encoder output → every decoder cross-attention. */
    line([[190, 296], [190, 236], [370, 236], [370, 357], [423, 357]], { stroke: c.b, "stroke-width": 2.2 });
    svgText(svg, 280, 228, "encoder output → keys & values", { "text-anchor": "middle" });

    box("embed", 95, 566, "Input Embedding");
    plus(190, 531);
    svgText(svg, 172, 527, "Positional", { "text-anchor": "end" });
    svgText(svg, 172, 540, "Encoding", { "text-anchor": "end" });
    box("mha", 95, 448, "Multi-Head Attention");
    box("addnorm", 95, 404, "Add & Norm");
    box("ffn", 95, 340, "Feed Forward");
    box("addnorm", 95, 296, "Add & Norm");
    svgText(svg, 190, 630, "Inputs: “I love cats”", { "text-anchor": "middle", class: "svg-title" });
    svgText(svg, 76, 262, "ENCODER", { class: "tl-map-caption" });

    /* ── Decoder (centre x = 520) ── */
    line([[520, 612], [520, 602]]);
    line([[520, 566], [520, 544]]);
    line([[520, 518], [520, 484]]);
    line([[520, 448], [520, 440]]);
    line([[520, 404], [520, 376]]);
    line([[520, 340], [520, 332]]);
    line([[520, 296], [520, 288]]);
    line([[520, 252], [520, 244]]);
    line([[520, 208], [520, 164]]);
    line([[520, 128], [520, 114]]);
    line([[520, 78], [520, 62]]);
    line([[520, 505], [636, 505], [636, 421], [617, 421]]);
    line([[520, 392], [636, 392], [636, 313], [617, 313]]);
    line([[520, 292], [636, 292], [636, 225], [617, 225]]);

    box("embed", 425, 566, "Output Embedding");
    plus(520, 531);
    svgText(svg, 538, 527, "Positional");
    svgText(svg, 538, 540, "Encoding");
    box("masked", 425, 448, "Masked Multi-Head Attention");
    box("addnorm", 425, 404, "Add & Norm");
    box("cross", 425, 340, "Multi-Head Attention (cross)");
    box("addnorm", 425, 296, "Add & Norm");
    box("ffn", 425, 252, "Feed Forward");
    box("addnorm", 425, 208, "Add & Norm");
    box("linear", 425, 128, "Linear");
    box("softmax", 425, 78, "Softmax");
    svgText(svg, 520, 50, "Output probabilities", { "text-anchor": "middle", class: "svg-title" });
    svgText(svg, 520, 630, "Outputs, shifted right: “<s> ich liebe”", { "text-anchor": "middle", class: "svg-title" });
    svgText(svg, 416, 170, "DECODER", { class: "tl-map-caption" });

    const legend = [
      ["attention", c.b],
      ["feed-forward", c.c],
      ["add & norm", c.d],
      ["output head", c.a],
    ];
    legend.forEach(([label, color], i) => {
      svg.appendChild(U().svgEl("rect", { x: 20, y: 22 + i * 20, width: 12, height: 12, rx: 3, fill: rgba(color, 0.4), stroke: color }));
      svgText(svg, 38, 32 + i * 20, label);
    });
  }

  function drawPaths(svg, n) {
    U().clear(svg);
    const c = C();
    const shown = Math.min(n, 9);
    const labels = n <= 9 ? Array.from({ length: n }, (_, i) => `w${i + 1}`) : ["w1", "w2", "w3", "w4", "…", `w${n - 3}`, `w${n - 2}`, `w${n - 1}`, `w${n}`];
    const xs = labels.map((_, i) => 50 + (i * 460) / Math.max(1, shown - 1));
    arrowMarker(svg, "tl-why-arrow-a", c.a);
    arrowMarker(svg, "tl-why-arrow-b", c.b);

    svgText(svg, 20, 24, `RNN: word 1 reaches word ${n} through ${n - 1} steps, one after another`, { class: "svg-title" });
    labels.forEach((label, i) => {
      if (i < labels.length - 1) {
        svg.appendChild(
          U().svgEl("line", {
            x1: xs[i] + 15, y1: 70, x2: xs[i + 1] - 17, y2: 70,
            stroke: c.b, "stroke-width": 2, "marker-end": "url(#tl-why-arrow-b)",
            "stroke-dasharray": label === "…" || labels[i + 1] === "…" ? "3 4" : "none",
          })
        );
      }
      if (label !== "…") svg.appendChild(U().svgEl("circle", { cx: xs[i], cy: 70, r: 14, fill: rgba(c.b, 0.15), stroke: c.b }));
      svgText(svg, xs[i], 74, label, { "text-anchor": "middle" });
    });

    svgText(svg, 20, 130, `Self-attention: word ${n} reaches every word in 1 step, all at the same time`, { class: "svg-title" });
    labels.forEach((label, i) => {
      if (label !== "…") svg.appendChild(U().svgEl("circle", { cx: xs[i], cy: 222, r: 14, fill: rgba(c.a, 0.15), stroke: c.a }));
      svgText(svg, xs[i], 226, label, { "text-anchor": "middle" });
    });
    const last = xs[xs.length - 1];
    labels.slice(0, -1).forEach((label, i) => {
      if (label === "…") return;
      const mid = (xs[i] + last) / 2;
      const lift = 28 + (last - xs[i]) * 0.14;
      svg.appendChild(
        U().svgEl("path", {
          d: `M ${last} 207 Q ${mid} ${207 - lift} ${xs[i] + 4} 208`,
          fill: "none",
          stroke: c.a,
          "stroke-width": 1.8,
          opacity: 0.8,
          "marker-end": "url(#tl-why-arrow-a)",
        })
      );
    });
  }

  /* ════════════════════════════════════════════════════════════════
     GUIDE BUILDING BLOCKS
     These pages read top to bottom like a book chapter: plain words
     first, then a figure, then the maths, then one takeaway. Figures
     are static on purpose; the reader scrolls instead of fiddling.
     ════════════════════════════════════════════════════════════════ */

  const para = (html) => `<p>${html}</p>`;
  const plain = (html, label = "In plain words") =>
    `<aside class="g-plain"><div class="g-label">💬 ${label}</div>${html}</aside>`;
  const deeper = (title, html) =>
    `<div class="g-deeper"><div class="g-label">∑ ${title}</div>${html}</div>`;
  const takeaway = (html) => `<div class="g-takeaway"><div class="g-label">✓ Key takeaway</div><p>${html}</p></div>`;
  const ref = (text) => `<span class="g-ref">📄 ${text}</span>`;
  const mathBlock = (latex) => `<div class="g-math">${tex(latex, true)}</div>`;
  const svgSlot = (name, viewBox, label) =>
    `<svg data-fig="${name}" viewBox="${viewBox}" role="img" aria-label="${label}"></svg>`;

  function figure(number, caption, body, wide = false) {
    return `
      <figure class="g-fig${wide ? " g-wide" : ""}">
        <div class="g-fig-body">${body}</div>
        <figcaption><b>Figure ${number}.</b> ${caption}</figcaption>
      </figure>`;
  }

  function chapter(id, number, kicker, title, body) {
    return `
      <section class="g-chapter" id="${id}">
        <div class="g-marker" aria-hidden="true">${number}</div>
        <header class="g-head">
          <div class="g-kicker">${kicker}</div>
          <h2>${title}</h2>
        </header>
        ${body}
      </section>`;
  }

  function guideToc(items) {
    return `
      <nav class="g-toc" aria-label="Chapters">
        <div class="g-label">In this guide</div>
        <ol>${items
          .map((item) => `<li><a href="#${item.id}">${item.title}</a><span>${item.blurb}</span></li>`)
          .join("")}</ol>
      </nav>`;
  }

  function flow(items) {
    return `<div class="tl-flow g-flow">${items
      .map((item) => (item === "→" ? '<span class="tl-flow-arrow">→</span>' : `<span class="tl-flow-box">${item}</span>`))
      .join("")}</div>`;
  }

  /* A thin bar under the top of the window that fills as you read. */
  function startProgressBar() {
    if (document.querySelector(".g-progress")) return;
    const bar = document.createElement("div");
    bar.className = "g-progress";
    bar.innerHTML = "<i></i>";
    document.body.appendChild(bar);
    const fill = bar.firstChild;
    const update = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      fill.style.width = `${max > 0 ? Math.min(100, (window.scrollY / max) * 100) : 0}%`;
    };
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    update();
  }

  /* ── New figures ──────────────────────────────────────────────── */

  /* A sentence with one curved arrow from the word that asks to the word
     it needs. Width follows the sentence, so the viewBox is set here. */
  function drawArcSentence(svg, tokens, from, to, label) {
    U().clear(svg);
    const c = C();
    const widths = tokens.map((token) => token.length * 9.4 + 22);
    const gap = 10;
    const centers = [];
    let x = 12;
    widths.forEach((w) => {
      centers.push(x + w / 2);
      x += w + gap;
    });
    const total = x - gap + 12;
    svg.setAttribute("viewBox", `0 0 ${total} 128`);
    svg.style.maxWidth = `${total * 1.35}px`;
    arrowMarker(svg, `g-arc-${from}-${to}-${tokens.length}`, c.b);
    tokens.forEach((token, i) => {
      const hot = i === from ? c.b : i === to ? c.a : null;
      svg.appendChild(
        U().svgEl("rect", {
          x: centers[i] - widths[i] / 2, y: 78, width: widths[i], height: 34, rx: 8,
          fill: hot ? rgba(hot, 0.18) : c.plotBg, stroke: hot || c.faint, "stroke-width": hot ? 2 : 1,
        })
      );
      svgText(svg, centers[i], 100, token, { "text-anchor": "middle", class: "g-svg-word" });
    });
    const a = centers[from];
    const b = centers[to];
    const lift = Math.min(56, 22 + Math.abs(a - b) * 0.18);
    svg.appendChild(
      U().svgEl("path", {
        d: `M ${a} 74 Q ${(a + b) / 2} ${74 - lift * 1.6} ${b} 76`,
        fill: "none", stroke: c.b, "stroke-width": 2.4, "marker-end": `url(#g-arc-${from}-${to}-${tokens.length})`,
      })
    );
    svgText(svg, (a + b) / 2, 74 - lift * 0.9, label, { "text-anchor": "middle", class: "svg-title" });
  }

  /* Three small panels: same direction, right angle, opposite. */
  function drawDotPanels(svg) {
    U().clear(svg);
    const c = C();
    const cases = [
      { title: "Same direction", angle: 20, value: "large and positive" },
      { title: "At right angles", angle: 90, value: "zero" },
      { title: "Opposite", angle: 180, value: "negative" },
    ];
    cases.forEach((entry, i) => {
      const ox = 40 + i * 200;
      const oy = 120;
      const id = `g-dot-${i}`;
      arrowMarker(svg, `${id}-a`, c.c);
      arrowMarker(svg, `${id}-b`, c.b);
      svgText(svg, ox + 60, 24, entry.title, { "text-anchor": "middle", class: "svg-title" });
      svg.appendChild(U().svgEl("line", { x1: ox + 40, y1: oy, x2: ox + 120, y2: oy, stroke: c.c, "stroke-width": 3, "marker-end": `url(#${id}-a)` }));
      const rad = (entry.angle * Math.PI) / 180;
      svg.appendChild(
        U().svgEl("line", {
          x1: ox + 40, y1: oy, x2: ox + 40 + 76 * Math.cos(rad), y2: oy - 76 * Math.sin(rad),
          stroke: c.b, "stroke-width": 3, "marker-end": `url(#${id}-b)`,
        })
      );
      svg.appendChild(U().svgEl("circle", { cx: ox + 40, cy: oy, r: 3.5, fill: c.ink }));
      svgText(svg, ox + 60, 160, `a · b is ${entry.value}`, { "text-anchor": "middle" });
    });
  }

  function drawPeHeatmap(svg, pos, D, P) {
    U().clear(svg);
    const pad = { left: 44, top: 26, right: 12, bottom: 24 };
    const cellW = (560 - pad.left - pad.right) / D;
    const cellH = (300 - pad.top - pad.bottom) / P;
    svgText(svg, pad.left, 16, `${P} positions (rows) × ${D} dimensions (columns)`, { class: "svg-title" });
    for (let p = 0; p < P; p += 1) {
      positionalEncoding(p, D).forEach((value, d) => {
        svg.appendChild(
          U().svgEl("rect", {
            x: pad.left + d * cellW, y: pad.top + p * cellH, width: cellW + 0.4, height: cellH + 0.4,
            fill: rgba(value >= 0 ? C().a : C().b, Math.abs(value) * 0.9),
          })
        );
      });
    }
    svg.appendChild(
      U().svgEl("rect", {
        x: pad.left - 2, y: pad.top + pos * cellH - 1, width: D * cellW + 4, height: cellH + 2,
        fill: "none", stroke: C().ink, "stroke-width": 2,
      })
    );
    svgText(svg, pad.left - 6, pad.top + pos * cellH + cellH, `pos ${pos}`, { "text-anchor": "end" });
    svgText(svg, pad.left, 300 - 8, "← fast-turning dimensions");
    svgText(svg, 560 - pad.right, 300 - 8, "slow-turning dimensions →", { "text-anchor": "end" });
  }

  function drawPeSimilarity(svg, pos, D, P) {
    const codes = Array.from({ length: P }, (_, p) => positionalEncoding(p, D));
    const sims = codes.map((code) => math.dotV(codes[pos], code));
    const chart = U().makeChart(svg, {
      xDomain: [0, P - 1],
      yDomain: [Math.floor(Math.min(...sims)) - 1, D / 2 + 1],
      title: `How similar position ${pos}'s code is to every other position (dot product)`,
    });
    svg.appendChild(
      U().svgEl("path", {
        d: U().pathFromPoints(sims.map((value, p) => ({ x: p, y: value })), chart.xScale, chart.yScale),
        fill: "none", stroke: C().a, "stroke-width": 2.5,
      })
    );
    svg.appendChild(U().svgEl("circle", { cx: chart.xScale(pos), cy: chart.yScale(sims[pos]), r: 6, fill: C().b, stroke: C().ring, "stroke-width": 2 }));
  }

  function peClocks(pos, D) {
    const omega = (pair) => 1 / Math.pow(10000, (2 * pair) / D);
    return `<div class="tl-dials">${[0, 1, 3, 7]
      .map((pair) => {
        const angle = pos * omega(pair);
        const period = (2 * Math.PI) / omega(pair);
        const x = 40 + 30 * Math.sin(angle);
        const y = 40 - 30 * Math.cos(angle);
        return `
          <div class="tl-dial">
            <svg viewBox="0 0 80 80" aria-hidden="true">
              <circle cx="40" cy="40" r="34" fill="none" stroke="${C().faint}" stroke-width="2" />
              <line x1="40" y1="40" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="${C().b}" stroke-width="3.5" stroke-linecap="round" />
              <circle cx="40" cy="40" r="3.5" fill="${C().ink}" />
            </svg>
            <div><b>dimensions ${2 * pair} and ${2 * pair + 1}</b><br /><span class="caption">one full turn every ${period < 100 ? fmt(period, 1) : Math.round(period)} positions</span></div>
          </div>`;
      })
      .join("")}</div>`;
  }

  function drawLrSchedule(svg) {
    const warmup = 4000;
    const curve = Array.from({ length: 201 }, (_, i) => 1 + i * 500).map((s) => ({ x: s / 1000, y: learningRate(s, 512, warmup) * 1000 }));
    const peak = learningRate(warmup, 512, warmup) * 1000;
    const chart = U().makeChart(svg, { xDomain: [0, 100], yDomain: [0, 1], title: "Learning rate (× 10⁻³) over 100,000 training steps (thousands)" });
    svg.appendChild(U().svgEl("path", { d: U().pathFromPoints(curve, chart.xScale, chart.yScale), fill: "none", stroke: C().a, "stroke-width": 2.6 }));
    svg.appendChild(U().svgEl("circle", { cx: chart.xScale(4), cy: chart.yScale(peak), r: 6, fill: C().b, stroke: C().ring, "stroke-width": 2 }));
    svgText(svg, chart.xScale(4) + 10, chart.yScale(peak) - 6, `peak ${peak.toFixed(2)} × 10⁻³ at step 4,000: warm-up ends, decay begins`);
  }

  /* Numbered badges on top of Figure 1, each linking to its chapter. */
  const MAP_BADGES = [
    { n: 2, x: 290, y: 583, id: "tf-embedding" },
    { n: 3, x: 222, y: 531, id: "tf-positions" },
    { n: 4, x: 290, y: 465, id: "tf-encoder" },
    { n: 5, x: 52, y: 300, id: "tf-stack" },
    { n: 6, x: 620, y: 465, id: "tf-masking" },
    { n: 7, x: 620, y: 357, id: "tf-cross" },
    { n: 8, x: 620, y: 128, id: "tf-generate" },
  ];

  function drawGuideMap(svg) {
    drawArchitecture(svg, null);
    MAP_BADGES.forEach((badge) => {
      const group = U().svgEl("g", { class: "g-badge", "data-goto": badge.id, tabindex: 0, role: "link", "aria-label": `Chapter ${badge.n}` });
      group.appendChild(U().svgEl("circle", { cx: badge.x, cy: badge.y, r: 13, fill: C().ink }));
      const t = U().svgEl("text", { x: badge.x, y: badge.y + 4.5, "text-anchor": "middle", class: "g-badge-num" });
      t.textContent = String(badge.n);
      group.appendChild(t);
      svg.appendChild(group);
    });
  }

  /* Draw every [data-fig] SVG in a guide; `drawers` maps names to functions. */
  function drawFigures(rootNode, drawers) {
    rootNode.querySelectorAll("[data-fig]").forEach((svg) => {
      const draw = drawers[svg.getAttribute("data-fig")];
      if (draw) draw(svg);
    });
  }

  /* ════════════════════════════════════════════════════════════════
     ATTENTION GUIDE
     ════════════════════════════════════════════════════════════════ */

  function mountAttentionGuide(rootNode) {
    const sentence = SENTENCES[0];
    const tokens = math.tokensOf(sentence);
    const X = math.embed(tokens);
    const IT = 4;
    const head = HEADS[0];
    const h1 = attentionHead(X, head);
    const mh = multiHead(X);
    const raw = rawAttention(X, IT, 1);
    const n = tokens.length;

    const ctxSentence = CONTEXT_SENTENCES[0];
    const ctxTokens = math.tokensOf(ctxSentence);
    const ctxX = ctxTokens.map((token) => CONTEXT_VOCAB[token]);
    const ctxFocus = ctxTokens.indexOf("bank");
    const ctx = rawAttention(ctxX, ctxFocus, 3);

    const dotWords = ["cat", "dog", "ball", "it", "sat"];
    const dotMatrix = dotWords.map((a) => dotWords.map((b) => math.dotV(VOCAB[a], VOCAB[b])));

    const softScores = [2.0, 1.0, 0.5, -1.0];
    const softLabels = ["cat", "tired", "sat", "the"];

    function scaleSample(dk) {
      const rng = seeded(7919 + dk);
      const vec = () => Array.from({ length: dk }, () => gaussian(rng));
      const q = vec();
      const raws = Array.from({ length: 8 }, vec).map((k) => math.dotV(q, k));
      return { raw: softmax(raws), scaled: softmax(raws.map((value) => value / Math.sqrt(dk))) };
    }

    const order = [orderDemo(["dog", "bites", "man"], false), orderDemo(["man", "bites", "dog"], false)];

    const chapters = [
      { id: "att-why", title: "Why a word needs its neighbours", blurb: "the problem attention solves" },
      { id: "att-numbers", title: "How a computer stores a word", blurb: "embeddings" },
      { id: "att-dot", title: "Measuring relatedness", blurb: "the dot product" },
      { id: "att-softmax", title: "Turning scores into shares", blurb: "softmax" },
      { id: "att-first", title: "A first attempt, and why it fails", blurb: "blend in similar words" },
      { id: "att-qkv", title: "Query, key and value", blurb: "the real idea" },
      { id: "att-one", title: "One word, number by number", blurb: "a full worked example" },
      { id: "att-matrix", title: "Every word at once", blurb: "Equation 1 of the paper" },
      { id: "att-scale", title: "Why divide by √dₖ", blurb: "keeping softmax healthy" },
      { id: "att-heads", title: "Many questions at once", blurb: "multi-head attention" },
      { id: "att-limits", title: "What attention can't do alone", blurb: "what the Transformer adds" },
      { id: "att-recap", title: "Recap and self-check", blurb: "cheat sheet and quiz" },
    ];

    let pickFocus = IT;

    function body() {
      const qIt = h1.Q[IT];
      const best = 1;
      return `
        <div class="g-intro">
          ${para("This guide explains <strong>attention</strong>, the idea at the heart of the Transformer and of every modern language model, starting from nothing more than a weighted sum. Each chapter starts in plain words, then shows a picture, then the maths. You can stop after the plain-words parts and still understand the idea, or read everything and be ready for the original paper.")}
          ${para("One sentence runs through the whole guide: <strong>“the cat sat because it was tired”</strong>. By the end you will see, with real numbers, how the word “it” works out that it means the cat.")}
          ${guideToc(chapters)}
        </div>

        ${chapter("att-why", 1, "The problem", "Why a word needs its neighbours", `
          ${plain(para("Read the word <strong>“bank”</strong> on its own. Is it a riverbank or a place that keeps money? You can't tell. Now read “I sat by the river <strong>bank</strong>”: instantly you know. The meaning came from a <em>neighbour</em>, “river”.") + para("Words like <strong>“it”</strong> are even more extreme: alone, “it” means nothing at all. In “the cat sat because <strong>it</strong> was tired”, “it” only gets its meaning by pointing back to “cat”."))}
          ${figure("1.1", "To understand “it”, a reader looks back at “cat”. Attention is a way for a neural network to draw this arrow by itself.", svgSlot("arc-it", "0 0 600 128", "it points back to cat"))}
          ${para("A plain neural network, the kind on the earlier pages, takes each input on its own. Feed it the word “it” and it has no way to look at the rest of the sentence. We need a mechanism that lets every word <strong>look at the other words and pull in what it needs</strong>. That mechanism is attention.")}
          ${takeaway("Meaning depends on context. Attention lets each word gather information from the other words in its sentence.")}
        `)}

        ${chapter("att-numbers", 2, "Background", "How a computer stores a word", `
          ${plain(para("A network can only do arithmetic, so each word becomes a short list of numbers, called a <strong>vector</strong> or <strong>embedding</strong>. Think of it as a profile card: how much is this word a thing? Is it alive? Is it an action?"))}
          ${figure("2.1", "The toy vocabulary used throughout this guide. Each word has 4 numbers, and each number has a readable meaning. In a real model the numbers are learned, there are 512 or more of them, and they have no names, but the principle is the same.",
            heatmap(["cat", "dog", "ball", "sat", "it", "tired", "the"].map((w) => VOCAB[w]), { rows: ["cat", "dog", "ball", "sat", "it", "tired", "the"], cols: FEATURES, maxAbs: 1 }))}
          ${para("Notice two things already. <strong>“cat” and “dog” have identical cards</strong>, so the model treats them as the same kind of word. And <strong>“it” has alive = 0</strong>. On its own, “it” does not know it refers to something alive. Keep an eye on that zero; attention will fill it in.")}
          ${deeper("The math: an embedding is a lookup, which is a matrix multiply", `
            ${para("Number the vocabulary, write a word as a <em>one-hot</em> vector (all zeros except a 1 at its position), and multiply by the embedding matrix E. The 1 selects exactly one row: that word's vector. So an embedding layer is just a dense layer with no bias, applied to a one-hot input. It is trained by backpropagation like any other layer.")}
            ${mathBlock(String.raw`\underbrace{[0\;1\;0\;0\;0]}_{\text{one-hot “cat”}}\;\times\; E_{\,5\times 4} \;=\; \text{row 2 of } E \;=\; [1,\,1,\,0,\,0]`)}
          `)}
          ${takeaway("Every word becomes a vector of numbers. Similar words get similar vectors.")}
        `)}

        ${chapter("att-dot", 3, "Background", "Measuring relatedness: the dot product", `
          ${plain(para("To decide which words matter to each other, we need a number that says “how related are these two vectors?”. The simplest one is the <strong>dot product</strong>: multiply the two lists position by position and add up the results. If two vectors point the same way, the result is large. If they have nothing in common, it is zero."))}
          ${figure("3.1", "The dot product compares directions. Vectors pointing the same way give a big positive number, unrelated (perpendicular) vectors give zero, opposite vectors give a negative number.", svgSlot("dot-panels", "0 0 600 175", "Dot product of vectors in three directions"))}
          ${figure("3.2", "Dot products between toy words. “cat · dog” = 2 (closely related). “cat · sat” = 0 (nothing in common). Note “it · cat” is only 0.3: by this measure, “it” and “cat” look almost unrelated. That will be a problem in Chapter 5.",
            heatmap(dotMatrix, { rows: dotWords, cols: dotWords, maxAbs: 2 }))}
          ${deeper("The math", `
            ${mathBlock(String.raw`a \cdot b = \sum_i a_i b_i = |a|\,|b|\cos\theta`)}
            ${para("Example: cat · dog = 1·1 + 1·1 + 0·0 + 0·0 = 2. A single neuron computes exactly this: a dot product between its weights and its input. Attention will use dot products between <em>two inputs</em> instead.")}
          `)}
          ${takeaway("The dot product turns “how related are these two words?” into a single number.")}
        `)}

        ${chapter("att-softmax", 4, "Background", "Turning scores into shares: softmax", `
          ${plain(para("Suppose “it” has given every word a relevance score. We want to turn those scores into <strong>shares that add up to 100%</strong>: how much of its attention “it” gives to each word. <strong>Softmax</strong> does exactly that. Bigger scores get bigger shares, every share is positive, and they always add up to 100%."))}
          ${figure("4.1", "Raw scores on the left (any numbers, even negative) become shares on the right. The ordering is kept, and the biggest score gets most of the share.",
            `<div class="g-two">
              <div>${heatmap([softScores], { rows: ["score"], cols: softLabels, maxAbs: 2 })}</div>
              <div><div class="tl-subhead">After softmax</div>${barList(softmax(softScores), softLabels, { max: 1, asPercent: true, highlight: 0 })}</div>
            </div>`)}
          ${figure("4.2", "Multiplying all scores by a constant changes how “decisive” softmax is. Divided by 3: shares even out. Multiplied by 3: the top word takes almost everything. Chapter 9 shows why this matters.",
            `<div class="g-three">${[
              ["scores ÷ 3 (soft)", 1 / 3],
              ["scores × 1", 1],
              ["scores × 3 (sharp)", 3],
            ]
              .map(([title, k]) => `<div><div class="tl-subhead">${title}</div>${barList(softmax(softScores.map((v) => v * k)), softLabels, { max: 1, asPercent: true })}</div>`)
              .join("")}</div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`\operatorname{softmax}(z)_j = \frac{e^{z_j}}{\sum_k e^{z_k}}`)}
            ${para(`Here: e² = 7.39, e¹ = 2.72, e⁰·⁵ = 1.65, e⁻¹ = 0.37; total 12.13; so “cat” gets 7.39 / 12.13 = ${pct(softmax(softScores)[0])}. The exponential makes every value positive; dividing by the total makes them add up to 1.`)}
          `)}
          ${takeaway("Softmax turns any list of scores into positive shares that add up to 1.")}
        `)}

        ${chapter("att-first", 5, "First attempt", "Blend in similar words, and why it isn't enough", `
          ${plain(para("With dot products and softmax we can build a first version of attention: for each word, score every word by similarity, turn the scores into shares, and build a <strong>new vector</strong> for the word as the share-weighted average of all the words. “bank” in “the river bank was muddy” should drift toward “river”."))}
          ${figure("5.1", `Words as 2-number vectors (money-ness, nature-ness). “bank” starts in the middle. After blending in its neighbours it moves toward nature (orange arrow): it now means a riverbank. Line thickness shows each word's share.`,
            `<div class="g-two g-two-wide">
              <div>${svgSlot("ctx-plot", "0 0 560 360", "bank moves toward river")}</div>
              <div><div class="tl-subhead">Shares for “bank”</div>${barList(ctx.weights, ctxTokens, { max: 1, asPercent: true, highlight: ctxFocus })}</div>
            </div>`, true)}
          ${para("It works for “bank”. But try the same recipe on <strong>“it”</strong>:")}
          ${figure("5.2", "Plain similarity for “it”: it gives its largest share to itself, and only a small share to “cat”, the word it actually needs.",
            sentenceStrip(tokens, IT, raw.weights, "", { label: "Where “it” looks, using plain similarity" }))}
          ${para("Two problems show up:")}
          <ol class="g-list">
            <li><strong>Every word is most similar to itself</strong>, so it mostly looks at itself.</li>
            <li><strong>Relevant is not the same as similar.</strong> “it” needs “cat” because a pronoun needs a noun, not because the two words look alike. We need a way to score <em>what a word is looking for</em> against <em>what another word offers</em>.</li>
          </ol>
          ${takeaway("Averaging over similar words adds context, but similarity is the wrong test. We need a learned notion of relevance.")}
        `)}

        ${chapter("att-qkv", 6, "The key idea", "Query, key and value", `
          ${plain(para("Give every word <strong>three</strong> different vectors, each made from the word by its own learned matrix:") + `
            <ul class="g-list">
              <li><strong>Query</strong>: what am I looking for? (“it”: <em>I'm looking for a living thing</em>.)</li>
              <li><strong>Key</strong>: what do I offer, so others can find me? (“cat”: <em>I'm a living thing</em>.)</li>
              <li><strong>Value</strong>: what do I hand over if someone picks me? (“cat”: <em>thing, alive</em>.)</li>
            </ul>` + para("A word's score for another word is now <em>its query · the other's key</em>. Like searching a library: you type a query, it is matched against the labels (keys) on the spines, and you take home the contents (values). The difference: you take a little of <em>every</em> book, in proportion to how well its label matched."))}
          ${figure("6.1", "One word, three roles. The same vector for “it” is multiplied by three different weight matrices (learned during training) to produce its query, key and value.",
            `<div class="g-qkv">
              <div class="tl-flow-box"><span>“it”</span><small>x = (${X[IT].map((v) => fmt(v, 1)).join(", ")})</small></div>
              <div class="g-qkv-arrows">
                <div><span class="g-pill">× W<sup>Q</sup></span> → <b>query</b> (${h1.Q[IT].map((v) => fmt(v)).join(", ")}) <em>“I want a living thing”</em></div>
                <div><span class="g-pill">× W<sup>K</sup></span> → <b>key</b> (${h1.K[IT].map((v) => fmt(v)).join(", ")}) <em>“I offer very little”</em></div>
                <div><span class="g-pill">× W<sup>V</sup></span> → <b>value</b> (${h1.V[IT].map((v) => fmt(v)).join(", ")}) <em>“thing 0.3, alive 0”</em></div>
              </div>
            </div>`)}
          ${figure("6.2", "The query of “it” (arrow) and the key of every word (dots), in the 2-number space where they meet. “cat”'s key lies far along the query's direction, so it gets the highest score. Keys at the origin (sat, because, was) offer nothing to this question.",
            svgSlot("qk-plot", "0 0 420 340", "query of it and all keys"))}
          ${figure("6.3", "Try it: click any word to see where it looks. Top row: plain similarity (Chapter 5). Bottom row: query · key.", `<div data-widget="pick"></div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`q_i = x_i W^Q,\qquad k_j = x_j W^K,\qquad v_j = x_j W^V,\qquad \text{score}(i,j) = q_i \cdot k_j`)}
            ${para("W<sup>Q</sup>, W<sup>K</sup> and W<sup>V</sup> are ordinary dense layers with no bias and no activation, shared by every word. Here they are 4 × 2 (d<sub>model</sub> = 4 → d<sub>k</sub> = 2); in the paper, 512 × 64.")}
          `)}
          ${takeaway("Queries ask, keys answer, values are delivered. Because they are separate, “relevant” no longer has to mean “similar”.")}
        `)}

        ${chapter("att-one", 7, "Worked example", "Following “it”, number by number", `
          ${para("Here is the full computation for one word, with the actual numbers from our toy model. Every later formula is just this, done for all words at once.")}
          <ol class="g-steps">
            <li><b>Make the query.</b> q<sub>it</sub> = x<sub>it</sub> W<sup>Q</sup> = (${qIt.map((v) => fmt(v)).join(", ")}).</li>
            <li><b>Score every word</b> with q · k, then <b>divide by √d<sub>k</sub> = √2</b> (Chapter 9 explains why).</li>
            <li><b>Softmax</b> the scaled scores into shares.</li>
          </ol>
          ${figure("7.1", "Steps 2 and 3 for every word. “cat” scores 5.38; after scaling and softmax it receives 85% of the attention.",
            `<div class="table-panel"><table>
              <thead><tr><th>Word</th><th>Key k</th><th>q · k</th><th>÷ √2</th><th>Share</th></tr></thead>
              <tbody>${tokens
                .map((token, j) => `<tr${j === best ? ' class="g-hot"' : ""}><td>${token}</td><td class="mono">(${h1.K[j].map((v) => fmt(v)).join(", ")})</td><td class="mono">${fmt(h1.S[IT][j])}</td><td class="mono">${fmt(h1.scaled[IT][j])}</td><td>${barList([h1.A[IT][j]], [""], { max: 1, asPercent: true })}</td></tr>`)
                .join("")}</tbody>
            </table></div>`)}
          <ol class="g-steps" start="4">
            <li><b>Blend the values.</b> The output for “it” is the share-weighted sum of every word's value: 85% of cat's value, plus small amounts of the rest.</li>
          </ol>
          ${figure("7.2", "Before and after. “it” started with alive = 0. After attention it carries alive = 0.86, borrowed from “cat”. This is context entering a word.",
            featureCompare(head.vLabels, [
              { name: "“it” before (its own value)", values: X[IT].slice(0, 2), color: C().c },
              { name: "“it” after attention", values: h1.O[IT], color: C().b },
            ]))}
          ${mathBlock(String.raw`\text{out}_{\text{it}} = \sum_j a_{\text{it},j}\, v_j = ${fmt(h1.A[IT][1])}\cdot v_{\text{cat}} + \dots = (${h1.O[IT].map((v) => fmt(v)).join(",\\;")})`)}
          ${takeaway("Attention for one word: query, scores against every key, softmax, weighted sum of values.")}
        `)}

        ${chapter("att-matrix", 8, "Scaling up", "Every word at once: the matrix form", `
          ${plain(para("Every word asks its question at the same time. Stack the words as rows of a matrix and the whole computation becomes a few matrix multiplications. There is <strong>no loop over the words</strong>, so a GPU can do all of them in parallel. This is a big reason Transformers train so much faster than older models (RNNs) that read one word at a time."))}
          <div class="g-film">
            ${[
              ["1. Stack the words", "One row per word.", heatmap(X, { rows: tokens, cols: FEATURES, focusRow: IT, title: "X", shape: `${n} × 4` })],
              ["2. Make Q, K, V", "Three multiplications. Each row is one word's query, key, value.",
                `<div class="tl-mats-row">${heatmap(h1.Q, { rows: tokens, cols: ["q₁", "q₂"], focusRow: IT, title: "Q", shape: `${n} × 2` })}${heatmap(h1.K, { rows: tokens, cols: ["k₁", "k₂"], focusRow: IT, title: "K", shape: `${n} × 2` })}${heatmap(h1.V, { rows: tokens, cols: head.vLabels, focusRow: IT, title: "V", shape: `${n} × 2` })}</div>`],
              ["3. Score every pair: QKᵀ", "Row i, column j: how well word i's query matches word j's key.", heatmap(h1.S, { rows: tokens, cols: tokens, focusRow: IT, title: "QKᵀ", shape: `${n} × ${n}`, colHint: "keys →" })],
              ["4. Scale and softmax each row", "Divide by √dₖ, then turn each row into shares that add up to 1. This is the famous attention map.", heatmap(h1.A, { rows: tokens, cols: tokens, focusRow: IT, maxAbs: 1, title: "A = softmax(QKᵀ / √dₖ)", shape: `${n} × ${n}`, colHint: "keys →" })],
              ["5. Blend the values: AV", "Each output row is that word's share-weighted mix of all values.", heatmap(h1.O, { rows: tokens, cols: head.vLabels, focusRow: IT, title: "Output = AV", shape: `${n} × 2` })],
            ]
              .map(([title, text, html]) => `<div class="g-film-frame"><div class="g-film-text"><h4>${title}</h4><p>${text}</p></div><div class="g-film-fig">${html}</div></div>`)
              .join('<div class="g-film-arrow" aria-hidden="true">↓</div>')}
          </div>
          <p class="caption">The highlighted row is “it” in every frame: the same numbers as Chapter 7.</p>
          ${deeper("The math: Equation 1 of the paper", `
            ${mathBlock(String.raw`\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\Big(\frac{QK^{\top}}{\sqrt{d_k}}\Big)V`)}
            <ul class="g-list g-legend">
              <li><b>QKᵀ</b>: compare every query with every key.</li>
              <li><b>÷ √dₖ</b>: keep the scores at a sensible size.</li>
              <li><b>softmax</b>: turn each row into shares.</li>
              <li><b>× V</b>: deliver the blended values.</li>
            </ul>
            ${ref("Vaswani et al. 2017, §3.2.1")}
          `)}
          ${takeaway("Attention for a whole sentence is one formula: softmax(QKᵀ/√dₖ)V, computed for all words in parallel.")}
        `)}

        ${chapter("att-scale", 9, "A detail that matters", "Why divide by √dₖ?", `
          ${plain(para("A dot product adds up many small products. With 64 numbers per vector (the paper's d<sub>k</sub>), the total swings much more widely than with 2. Big scores make softmax <strong>winner-take-all</strong>: one word gets ~100%, the rest ~0%. A softmax stuck like that barely changes when its inputs change, so the network can't learn which words to attend to. Dividing by √d<sub>k</sub> brings the scores back to a normal size."))}
          ${figure("9.1", "Softmax over 8 random keys. Orange: raw scores. Teal: scores divided by √dₖ. With small dₖ (left) the two barely differ. With dₖ = 512 (right), the unscaled version puts nearly everything on one key, while the scaled one stays spread out and learnable.",
            `<div class="g-two">
              <div><div class="tl-subhead">dₖ = 4</div>${svgSlot("scale-4", "0 0 560 250", "softmax with d_k 4")}</div>
              <div><div class="tl-subhead">dₖ = 512</div>${svgSlot("scale-512", "0 0 560 250", "softmax with d_k 512")}</div>
            </div>
            <div class="legend"><span><i data-swatch="b"></i> without scaling</span><span><i data-swatch="a"></i> with ÷ √dₖ</span></div>`, true)}
          ${deeper("The math: where √dₖ comes from (paper footnote 4)", `
            ${para("If the entries of q and k are independent with mean 0 and variance 1 (roughly true at the start of training), each product q<sub>i</sub>k<sub>i</sub> has variance 1, and a sum of d<sub>k</sub> of them has variance d<sub>k</sub>:")}
            ${mathBlock(String.raw`\operatorname{Var}(q\cdot k) = \sum_{i=1}^{d_k}\operatorname{Var}(q_i k_i) = d_k \quad\Rightarrow\quad \operatorname{Var}\!\Big(\frac{q\cdot k}{\sqrt{d_k}}\Big) = 1`)}
            ${para("This is the same problem as a saturated sigmoid on the Activation Functions page: a flat function has a near-zero gradient and stops learning.")}
          `)}
          ${takeaway("Dividing by √dₖ keeps the scores at unit size, so softmax stays soft enough to learn from.")}
        `)}

        ${chapter("att-heads", 10, "More power", "Many questions at once: multi-head attention", `
          ${plain(para("One attention pattern answers one kind of question. Language has many: <em>who does “it” refer to? What did the cat do? Which adjective goes with which noun?</em> So the Transformer runs several smaller attentions side by side, called <strong>heads</strong>, each with its own W<sup>Q</sup>, W<sup>K</sup>, W<sup>V</sup>. It then joins their outputs and mixes them with one more matrix, W<sup>O</sup>."))}
          ${figure("10.1", "Our two heads look at the same sentence and ask different questions. Head 1 (“who is involved?”) sends “it” to “cat”. Head 2 (“what happened?”) sends nouns and “it” toward the action words.",
            `<div class="g-two">${HEADS.map((entry, h) => `<div><div class="tl-subhead">${entry.name}</div>${heatmap(mh.heads[h].A, { rows: tokens, cols: tokens, focusRow: IT, maxAbs: 1, colHint: "keys →" })}</div>`).join("")}</div>`, true)}
          ${figure("10.2", "Shapes. Each head works in a smaller space; joining the heads restores the original width, so the output can be added back to the input (next guide).",
            flow([`X<small>${n} × 4</small>`, "→", `<span>head 1</span><small>${n} × 2</small>`, `<span>head 2</span><small>${n} × 2</small>`, "→", `<span>concat</span><small>${n} × 4</small>`, "→", `<span>× W<sup>O</sup></span><small>${n} × 4</small>`]))}
          ${figure("10.3", "What multi-head attention brings back to “it”: head 1 fills in thing and alive, and head 2 fills in action/state. (W<sup>O</sup> is the identity here so the feature names survive.)",
            featureCompare(FEATURES, [
              { name: "“it” itself", values: X[IT], color: C().c },
              { name: "multi-head output for “it”", values: mh.out[IT], color: C().b },
            ]))}
          ${deeper("The math", `
            ${mathBlock(String.raw`\operatorname{MultiHead}(X) = \operatorname{Concat}(\text{head}_1,\ldots,\text{head}_h)\,W^O,\qquad \text{head}_i = \operatorname{Attention}(XW_i^Q,\,XW_i^K,\,XW_i^V)`)}
            ${para("The paper uses h = 8 heads with d<sub>k</sub> = 512 / 8 = 64. Eight 64-wide heads cost about the same as one 512-wide head, so you get eight different attention patterns for the price of one.")}
            ${ref("Vaswani et al. 2017, §3.2.2")}
          `)}
          ${takeaway("Several heads ask different questions in parallel; their answers are joined and mixed.")}
        `)}

        ${chapter("att-limits", 11, "Looking ahead", "What attention can't do alone", `
          ${para("Attention is powerful, but on its own it has three gaps. The Transformer is attention plus the parts that fill them.")}
          <div class="g-cards">
            <div class="soft-box"><h4>1. It ignores word order</h4><p>Scores depend only on <em>which</em> words are present, not <em>where</em>. Below, “dog bites man” and “man bites dog” give “dog” exactly the same output. Fix: <a href="./algorithm.html?id=transformer#tf-positions">positional encoding</a>.</p>
              <table class="g-mini"><tr><th></th><th>output for “dog”</th></tr><tr><td>dog bites man</td><td class="mono">(${order[0].O[0].map((v) => fmt(v)).join(", ")})</td></tr><tr><td>man bites dog</td><td class="mono">(${order[1].O[2].map((v) => fmt(v)).join(", ")})</td></tr></table></div>
            <div class="soft-box"><h4>2. It only mixes</h4><p>Attention moves information <em>between</em> words, but does no real processing of each word afterwards. Fix: a small <a href="./algorithm.html?id=transformer#tf-encoder">feed-forward network</a> after every attention layer (the ANN you already know).</p></div>
            <div class="soft-box"><h4>3. It costs n² scores</h4><p>Every word scores every word: ${n} words → ${n * n} scores; 1,000 words → 1,000,000. Fine for sentences, expensive for books. This is the main cost of the Transformer, discussed in the <a href="./algorithm.html?id=transformer#tf-why">last chapter</a> of the next guide.</p></div>
          </div>
          ${takeaway("Attention needs help with word order, per-word processing and depth. That is what the Transformer adds.")}
        `)}

        ${chapter("att-recap", 12, "Recap", "Recap and self-check", `
          <ol class="g-recap">
            <li>A word's meaning depends on its neighbours.</li>
            <li>Words are vectors; dot products measure relatedness; softmax turns scores into shares.</li>
            <li>Each word makes a <b>query</b>, a <b>key</b> and a <b>value</b> with learned matrices.</li>
            <li>Shares = softmax(query · keys / √dₖ); output = shares × values.</li>
            <li>For a whole sentence: <b>softmax(QKᵀ/√dₖ)V</b>, all words in parallel.</li>
            <li>Several heads ask different questions; their outputs are joined.</li>
          </ol>
          ${quiz([
            { q: "Why not just use the word vectors themselves as queries and keys?", a: "Every word would score highest with itself, and relevance would require similarity. “it” needs “cat” although the two are not alike (Chapter 5). Separate W<sup>Q</sup> and W<sup>K</sup> let “what I look for” differ from “what I offer”." },
            { q: "A word's query is all zeros. Where does it look?", a: "Every score is 0, so softmax gives every word the same share (1/n): the output is a plain average. In head 1, “cat” has a zero query; check its row in Figure 10.1." },
            { q: "What goes wrong without the √dₖ?", a: "With large dₖ the scores grow like √dₖ, softmax becomes winner-take-all and its gradient vanishes, so learning stalls (Chapter 9)." },
            { q: "Why are 8 heads of size 64 about as costly as 1 head of size 512?", a: "The projection matrices have the same total size: 8 × (512 × 64) = 512 × 512." },
            { q: "Shuffle the words of a sentence. What happens to each word's attention output?", a: "Nothing, apart from moving with its word. Attention has no notion of position, which is why the Transformer adds positional encodings." },
          ])}
          <div class="tl-next">
            <div><div class="eyebrow">Next guide</div><strong>The Transformer, box by box</strong><p class="caption">Word order, the encoder block, the decoder, training, and why it replaced RNNs.</p></div>
            <a class="button primary" href="./algorithm.html?id=transformer">Continue to the Transformer →</a>
          </div>
        `)}
      `;
    }

    function renderPick() {
      const host = rootNode.querySelector('[data-widget="pick"]');
      if (!host) return;
      const rawPick = rawAttention(X, pickFocus, 1);
      host.innerHTML = `
        ${sentenceStrip(tokens, pickFocus, rawPick.weights, "data-pick", { label: "Plain similarity" })}
        ${sentenceStrip(tokens, pickFocus, h1.A[pickFocus], "data-pick", { label: "Query · key (head 1)" })}`;
    }

    function render() {
      rootNode.innerHTML = body();
      paintSwatches(rootNode);
      drawFigures(rootNode, {
        "arc-it": (svg) => drawArcSentence(svg, tokens, IT, 1, "it → cat"),
        "dot-panels": drawDotPanels,
        "ctx-plot": (svg) => drawContextPlot(svg, ctxTokens, ctxX, ctxFocus, ctx.weights, ctx.out),
        "qk-plot": (svg) => drawQueryKeyPlot(svg, tokens, h1, IT),
        "scale-4": (svg) => { const s = scaleSample(4); drawScaleBars(svg, s.raw, s.scaled); },
        "scale-512": (svg) => { const s = scaleSample(512); drawScaleBars(svg, s.raw, s.scaled); },
      });
      renderPick();
    }

    onClickAttr(rootNode, "data-pick", (i) => {
      pickFocus = i;
      renderPick();
    });
    render();
    U().onRedraw(render);
    startProgressBar();
  }

  /* ════════════════════════════════════════════════════════════════
     TRANSFORMER GUIDE
     ════════════════════════════════════════════════════════════════ */

  function mountTransformerGuide(rootNode) {
    const tokens = math.tokensOf(SENTENCES[0]);
    const X = math.embed(tokens);
    const IT = 4;
    const run = encoderLayer(X);
    const n = tokens.length;
    const hiddenCols = Array.from({ length: math.FFN.W1[0].length }, (_, j) => `h${j + 1}`);
    const order = [orderDemo(["dog", "bites", "man"], false), orderDemo(["man", "bites", "dog"], false)];
    const orderPe = [orderDemo(["dog", "bites", "man"], true), orderDemo(["man", "bites", "dog"], true)];
    const esc = (token) => token.replace("<", "&lt;").replace(">", "&gt;");
    const decRows = DECODER_INPUT.map(esc);
    const crossA = CROSS_SCORES.map(softmax);
    const ln = (() => {
      const row = run.added1[IT];
      const mean = row.reduce((a, b) => a + b, 0) / row.length;
      const std = Math.sqrt(row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length);
      return { mean, std };
    })();
    const PE_D = 32;
    const PE_P = 50;
    const PE_POS = 7;

    /* Parameter count of the base model, from the paper's sizes. */
    const d = 512;
    const dff = 2048;
    const vocab = 37000;
    const params = [
      ["Embeddings (shared)", vocab * d],
      ["Encoder: attention × 6", 6 * 4 * d * d],
      ["Encoder: feed-forward × 6", 6 * 2 * d * dff],
      ["Decoder: 2 attentions × 6", 6 * 8 * d * d],
      ["Decoder: feed-forward × 6", 6 * 2 * d * dff],
    ];
    const paramTotal = params.reduce((acc, [, value]) => acc + value, 0);

    const chapters = [
      { id: "tf-big", title: "The big picture", blurb: "a reader and a writer" },
      { id: "tf-embedding", title: "Words in", blurb: "tokens and embeddings" },
      { id: "tf-positions", title: "Where is each word?", blurb: "positional encoding" },
      { id: "tf-encoder", title: "The encoder block", blurb: "talk, then think" },
      { id: "tf-stack", title: "Stacking six blocks", blurb: "depth, and where the parameters live" },
      { id: "tf-masking", title: "The decoder must not peek", blurb: "masked self-attention" },
      { id: "tf-cross", title: "Reading the source", blurb: "cross-attention" },
      { id: "tf-generate", title: "Writing the translation", blurb: "one word at a time" },
      { id: "tf-training", title: "How it learns", blurb: "the paper's training recipe" },
      { id: "tf-why", title: "Why it replaced RNNs", blurb: "the paper's Table 1" },
      { id: "tf-recap", title: "Recap and self-check", blurb: "one sentence through the whole model" },
    ];

    const genFrames = DECODER_INPUT.map((_, t) => {
      const probs = NEXT_PROBS[t];
      const order3 = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
      const looked = crossA[t].indexOf(Math.max(...crossA[t]));
      return { t, probs, order3, looked };
    });

    function body() {
      return `
        <div class="g-intro">
          ${para("The <a href=\"./algorithm.html?id=attention\">Attention guide</a> built the core operation. This guide assembles the complete model from the 2017 paper “Attention Is All You Need”, one box of its famous diagram at a time. Each chapter starts in plain words, then a figure, then the maths, then a takeaway.")}
          ${para("Two running examples: the <strong>encoder</strong> reads “the cat sat because it was tired” (the same toy model as the Attention guide), and the <strong>decoder</strong> translates <strong>“I love cats” → “ich liebe Katzen”</strong>, English to German, as in the paper.")}
          ${guideToc(chapters)}
        </div>

        ${chapter("tf-big", 1, "Overview", "The big picture: a reader and a writer", `
          ${plain(para("Think of a human translator. First they <strong>read</strong> the whole English sentence and understand it. Then they <strong>write</strong> the German sentence one word at a time, glancing back at the English as they go. The Transformer is built the same way: an <strong>encoder</strong> (the reader) and a <strong>decoder</strong> (the writer)."))}
          ${figure("1.1", "The architecture from the paper (its Figure 1). Left: the encoder. Right: the decoder. The numbered badges match the chapters of this guide; click one to jump there.", svgSlot("map", "0 0 720 640", "Transformer architecture"), true)}
          ${para("Both halves are built from the same few parts: attention, a small feed-forward network, and “Add & Norm” wrappers. Each half is a stack of N = 6 identical blocks. There is no recurrence: every word is processed at the same time.")}
          ${takeaway("Encoder reads the whole input at once; decoder writes the output word by word while looking back at the encoder.")}
        `)}

        ${chapter("tf-embedding", 2, "Encoder · input", "Words in: tokens and embeddings", `
          ${plain(para("Text is first cut into <strong>tokens</strong>: whole words or common pieces of words (“trans” + “former”), so the model can handle words it has never seen. Each token is then turned into a vector by looking it up in a learned table, the <strong>embedding</strong>, exactly as in Chapter 2 of the Attention guide."))}
          ${figure("2.1", "From text to vectors. The token IDs here are illustrative. The paper used a shared vocabulary of about 37,000 sub-word tokens (byte-pair encoding) for English and German.",
            flow(["“I love cats”", "→", "<span>tokens</span><small>I · love · cats</small>", "→", "<span>IDs</span><small>(e.g. 41 · 2093 · 7311)</small>", "→", "<span>vectors</span><small>3 × 512</small>"]))}
          ${deeper("Two details from the paper", `
            <ul class="g-list">
              <li>The embedding vectors are <strong>multiplied by √d<sub>model</sub></strong> (√512 ≈ 22.6) before positions are added, so the position signal (next chapter) doesn't drown out the word's meaning.</li>
              <li>The same matrix is <strong>shared</strong> by the input embedding, the output embedding and the final Linear layer that turns vectors back into word scores. One table serves all three jobs.</li>
            </ul>
            ${ref("Vaswani et al. 2017, §3.4")}
          `)}
          ${takeaway("Tokens become vectors via a learned lookup table, shared between input and output.")}
        `)}

        ${chapter("tf-positions", 3, "Encoder · input", "Where is each word? Positional encoding", `
          ${plain(para("Attention looks only at <em>which</em> words are present, never at <em>where</em> they are. To attention, “dog bites man” and “man bites dog” are the same bag of words, which is clearly a problem. The fix is to <strong>stamp each word with its position</strong> by adding a position vector to it before the first layer."))}
          ${figure("3.1", "The problem, with real numbers. Without positions, “dog” gets exactly the same output whether it bites or is bitten. With positions added, the outputs differ.",
            `<div class="table-panel"><table>
              <thead><tr><th>Output for “dog”</th><th>in “dog bites man”</th><th>in “man bites dog”</th></tr></thead>
              <tbody>
                <tr><td>without positions</td><td class="mono">(${order[0].O[0].map((v) => fmt(v)).join(", ")})</td><td class="mono">(${order[1].O[2].map((v) => fmt(v)).join(", ")})</td></tr>
                <tr><td>with positions</td><td class="mono">(${orderPe[0].O[0].map((v) => fmt(v)).join(", ")})</td><td class="mono">(${orderPe[1].O[2].map((v) => fmt(v)).join(", ")})</td></tr>
              </tbody></table></div>`)}
          ${para("The paper's position vector works like a <strong>set of clock hands turning at different speeds</strong>. The first hand spins fast (a full turn about every 6 positions), the next more slowly, and the last takes thousands of positions per turn. Together, the hands give every position a unique reading, like the digits of a counter, but smooth.")}
          ${figure("3.2", `The whole positional-encoding table for 50 positions and 32 dimensions (teal = positive, orange = negative). Each row is one position's “stamp”. Left columns change quickly from row to row; right columns change slowly. Position ${PE_POS} is outlined.`, svgSlot("pe-heat", "0 0 560 300", "positional encoding matrix"), true)}
          ${figure("3.3", `Four of the 16 clock hands at position ${PE_POS}.`, peClocks(PE_POS, PE_D))}
          ${figure("3.4", `Nearby positions get similar stamps: position ${PE_POS}'s stamp matches itself best and its neighbours next best. This gives attention an easy way to notice “these words are close together”.`, svgSlot("pe-sim", "0 0 560 220", "similarity of position codes"))}
          ${deeper("The math, and why the paper chose sines and cosines", `
            ${mathBlock(String.raw`PE_{(pos,\,2i)} = \sin(pos\cdot\omega_i),\quad PE_{(pos,\,2i+1)} = \cos(pos\cdot\omega_i),\quad \omega_i = \frac{1}{10000^{2i/d_{\text{model}}}}`)}
            ${para("Moving forward k positions turns every clock hand by a fixed angle, whatever the starting position. A turn is a linear operation (a rotation matrix), so <em>relative</em> positions are easy for the model to use, which is the paper's stated reason:")}
            ${mathBlock(String.raw`\begin{pmatrix}\sin\omega(p{+}k)\\ \cos\omega(p{+}k)\end{pmatrix} = \begin{pmatrix}\cos\omega k & \sin\omega k\\ -\sin\omega k & \cos\omega k\end{pmatrix}\begin{pmatrix}\sin\omega p\\ \cos\omega p\end{pmatrix}`)}
            ${para("The authors also tried <em>learned</em> position vectors and got nearly identical results (Table 3, row E); they kept sinusoids because they might work for sentences longer than any seen in training.")}
            ${ref("Vaswani et al. 2017, §3.5")}
          `)}
          ${takeaway("Positional encoding adds a unique, smoothly varying stamp to each position, the only place word order enters the model.")}
        `)}

        ${chapter("tf-encoder", 4, "Encoder", "The encoder block: talk, then think", `
          ${plain(para("One encoder block does two jobs in turn. First the words <strong>talk</strong>: attention lets each word gather information from the others. Then each word <strong>thinks</strong>: a small neural network processes it on its own. Around each job is a safety wrapper, called <strong>Add & Norm</strong>."))}
          ${figure("4.1", "Inside one encoder block. We now follow the word “it” through each stage with real numbers from the toy model.",
            flow(["input", "→", "<span>multi-head attention</span><small>talk</small>", "→", "add & norm", "→", "<span>feed-forward</span><small>think</small>", "→", "add & norm", "→", "output"]))}

          <h3 class="g-sub">4a · Attention: the words talk</h3>
          ${para("Exactly the multi-head attention from the Attention guide. “it” comes back carrying <em>alive</em> from “cat” and <em>action</em> from the verbs.")}
          ${figure("4.2", "Attention output for every word (Z). The highlighted row is “it”.", heatmap(run.mha.out, { rows: tokens, cols: FEATURES, focusRow: IT, title: "Z = MultiHead(X)", shape: `${n} × 4` }))}

          <h3 class="g-sub">4b · Add: keep the original, add the update</h3>
          ${plain(para("Instead of replacing a word with the attention output, we <strong>add</strong> the two. The attention output is treated as a <em>correction</em> on top of the original word. “it” keeps “refers back” and gains “alive”. This shortcut is called a <strong>residual connection</strong>."))}
          ${figure("4.3", "“it” before the block (x) and after adding the attention output (x + z).",
            featureCompare(FEATURES, [
              { name: "x (input)", values: run.X[IT], color: C().c },
              { name: "x + z (after add)", values: run.added1[IT], color: C().b },
            ], 2))}
          ${deeper("Why the residual matters for training", `
            ${mathBlock(String.raw`\frac{\partial\,(x + F(x))}{\partial x} = I + \frac{\partial F}{\partial x}`)}
            ${para("Even if a sub-layer's own gradient is tiny, the identity <b>I</b> passes the gradient back unchanged. That fixes the vanishing gradients from the Backpropagation page and lets the model stack many layers.")}
          `)}

          <h3 class="g-sub">4c · Norm: keep every word at a steady size</h3>
          ${plain(para("After adding, some numbers can grow large. <strong>Layer normalisation</strong> re-centres each word's vector to average 0 and rescales it to spread 1, using only that word's own numbers. Values stay in a comfortable range however many layers are stacked."))}
          ${figure("4.4", `For “it”: mean μ = ${fmt(ln.mean, 3)}, spread σ = ${fmt(ln.std, 3)} before; mean 0, spread 1 after.`,
            featureCompare(FEATURES, [
              { name: "before norm", values: run.added1[IT], color: C().c },
              { name: "after norm", values: run.norm1[IT], color: C().b },
            ], 2))}
          ${mathBlock(String.raw`\operatorname{LayerNorm}(x) = \gamma\,\frac{x-\mu}{\sigma} + \beta`)}

          <h3 class="g-sub">4d · Feed-forward: each word thinks on its own</h3>
          ${plain(para("Now a small ANN, the very kind from the earlier pages, processes <strong>each word separately</strong>, with the same weights for every word: expand to a wider hidden layer with ReLU, then project back. Attention moved information <em>between</em> words; this step works <em>within</em> each word on what it just gathered."))}
          ${figure("4.5", "Hidden layer of the feed-forward network (4 → 8 here; 512 → 2048 in the paper). Zeros are units switched off by ReLU. (The toy's feed-forward weights are random but fixed.)",
            heatmap(run.hidden, { rows: tokens, cols: hiddenCols, focusRow: IT, title: "ReLU(x W₁ + b₁)", shape: `${n} × 8` }))}
          ${mathBlock(String.raw`\operatorname{FFN}(x) = \max(0,\; xW_1 + b_1)\,W_2 + b_2`)}

          <h3 class="g-sub">4e · Add & norm again: the block's output</h3>
          ${figure("4.6", "The block's output has exactly the same shape as its input (7 words × 4 numbers), which is what lets blocks be stacked.",
            heatmap(run.out, { rows: tokens, cols: FEATURES, focusRow: IT, title: "Encoder block output", shape: `${n} × 4` }))}
          ${deeper("The whole block in two lines", `
            ${mathBlock(String.raw`h = \operatorname{LayerNorm}\big(x + \operatorname{MultiHead}(x)\big),\qquad y = \operatorname{LayerNorm}\big(h + \operatorname{FFN}(h)\big)`)}
            ${para("The paper also applies dropout (rate 0.1) to each sub-layer's output just before the add.")}
            ${ref("Vaswani et al. 2017, §3.1, §3.3")}
          `)}
          ${takeaway("Encoder block = attention (talk) + feed-forward (think), each wrapped in a residual add and a layer norm.")}
        `)}

        ${chapter("tf-stack", 5, "Encoder & decoder", "Stacking six blocks", `
          ${plain(para("One block lets each word gather information once. Stacking blocks lets it happen repeatedly: early layers might link “it” to “cat”, later layers can build on that (“the cat was tired, so it sat”). The paper stacks <strong>N = 6</strong> blocks in the encoder and 6 in the decoder, each with its own weights."))}
          ${figure("5.1", `Where the base model's parameters live, counted from the paper's sizes (d<sub>model</sub> = 512, d<sub>ff</sub> = 2048, vocabulary ≈ 37,000). This count gives ≈ ${fmt(paramTotal / 1e6, 0)} M; the paper reports 65 M once biases and norms are included.`,
            barList(params.map(([, v]) => v / 1e6), params.map(([label]) => label), { digits: 1, color: C().c }) + '<p class="caption">Millions of parameters. Feed-forward layers hold more than attention.</p>')}
          ${takeaway("Six identical blocks, each refining the previous one's output. Same shape in and out makes stacking trivial.")}
        `)}

        ${chapter("tf-masking", 6, "Decoder", "The decoder must not peek: masked self-attention", `
          ${plain(para("The decoder writes one word at a time. But to train fast, the whole correct German sentence is fed in at once, shifted one step right, and every position learns to predict its next word in parallel. Position “ich” must predict “liebe”. If it could simply <em>look</em> at “liebe” in its input, it would learn to cheat. So the decoder's self-attention is <strong>masked</strong>: each position may only look at itself and earlier positions."))}
          ${figure("6.1", "Left: without the mask, the row for “ich” puts most of its attention on “liebe”, the very word it must predict. Right: with the mask, every score above the diagonal is set to −∞ before softmax, so future words get exactly 0%. (Scores hand-set for illustration.)",
            `<div class="g-two">
              <div><div class="tl-subhead">Without mask: cheating</div>${heatmap(SELF_SCORES.map(softmax), { rows: decRows, cols: decRows, maxAbs: 1, colHint: "keys →" })}</div>
              <div><div class="tl-subhead">With causal mask</div>${heatmap(maskedScores(SELF_SCORES, true).map(softmax), { rows: decRows, cols: decRows, maxAbs: 1, colHint: "keys →" })}</div>
            </div>`, true)}
          ${deeper("The math", `
            ${mathBlock(String.raw`\operatorname{softmax}\!\Big(\frac{QK^{\top}}{\sqrt{d_k}} + M\Big),\qquad M_{ij} = \begin{cases}0 & j \le i\\ -\infty & j > i\end{cases}`)}
            ${para("Because e<sup>−∞</sup> = 0, masked positions get exactly zero weight. Training with the true previous words like this is called <strong>teacher forcing</strong>.")}
            ${ref("Vaswani et al. 2017, §3.2.3")}
          `)}
          ${takeaway("The causal mask lets the decoder train on whole sentences in parallel without seeing the future.")}
        `)}

        ${chapter("tf-cross", 7, "Decoder", "Reading the source: cross-attention", `
          ${plain(para("The decoder's middle layer is where translation happens. Its <strong>queries come from the German side</strong> (“what do I need to write next?”), while its <strong>keys and values come from the encoder's output</strong> for the English sentence. Each German position can look directly at any English word."))}
          ${figure("7.1", "Cross-attention: rows are decoder positions, columns are English words. Each position looks mostly at the English word it is about to translate. Nobody tells the model this alignment; it emerges from training. (Weights hand-set to show the typical pattern.)",
            heatmap(crossA, { rows: decRows, cols: SOURCE, maxAbs: 1, colHint: "English (keys & values from the encoder) →" }))}
          ${figure("7.2", "Where the three inputs of cross-attention come from.",
            flow(["<span>Q</span><small>from decoder</small>", "<span>K, V</span><small>from encoder output</small>", "→", "<span>Attention</span><small>softmax(QKᵀ/√dₖ)V</small>"]))}
          ${takeaway("Cross-attention is the bridge: decoder queries, encoder keys and values.")}
        `)}

        ${chapter("tf-generate", 8, "Decoder · output", "Writing the translation, one word at a time", `
          ${plain(para("At the top of the decoder, a <strong>Linear</strong> layer gives a score to every word in the vocabulary, and <strong>softmax</strong> turns the scores into probabilities. The model picks a word, appends it to its input, and runs again, until it produces the end-of-sentence token. The encoder runs only once; its output is reused at every step."))}
          <div class="g-film">
            ${genFrames
              .map((frame) => `
                <div class="g-film-frame">
                  <div class="g-film-text">
                    <h4>Step ${frame.t + 1}</h4>
                    <p>Decoder input: <b>${DECODER_INPUT.slice(0, frame.t + 1).map(esc).join(" ")}</b><br />Looks mostly at: <b>“${SOURCE[frame.looked]}”</b> (${pct(crossA[frame.t][frame.looked])})<br />Picks: <b>${esc(OUT_VOCAB[frame.order3[0][1]])}</b></p>
                  </div>
                  <div class="g-film-fig">${barList(frame.order3.map(([p]) => p), frame.order3.map(([, i]) => esc(OUT_VOCAB[i])), { max: 1, asPercent: true, highlight: 0, color: C().b })}</div>
                </div>`)
              .join('<div class="g-film-arrow" aria-hidden="true">↓</div>')}
          </div>
          <p class="caption">Top-3 next-word probabilities at each step (illustrative). Result: “ich liebe Katzen”.</p>
          ${deeper("How the paper decoded", `
            ${para("Instead of always taking the single most likely word (greedy decoding), the paper used <strong>beam search</strong> with beam size 4: it keeps the 4 best partial translations at each step and picks the best complete one, with a length penalty (α = 0.6) so short outputs aren't unfairly favoured.")}
            ${ref("Vaswani et al. 2017, §6.1")}
          `)}
          ${takeaway("Generation is a loop: predict a word, append it, repeat. Training is parallel; generation is not.")}
        `)}

        ${chapter("tf-training", 9, "Training", "How it learns: the training recipe", `
          ${plain(para("Training shows the model millions of sentence pairs. At every position it predicts the next word, and the <strong>loss</strong> (cross-entropy) measures how much probability it gave the correct word. Backpropagation then nudges every weight to do better. The paper adds a few tricks to make this stable and fast."))}
          ${figure("9.1", "The learning-rate schedule (Equation 3): it rises for the first 4,000 steps (warm-up), then slowly decays. Early on the model is random and its gradient estimates are unreliable, so it starts gently.", svgSlot("lr", "0 0 560 260", "learning rate schedule"))}
          ${figure("9.2", "Label smoothing (ε = 0.1): instead of demanding 100% on the correct word, the target keeps 90% there and spreads the rest over the other words. This makes the model less over-confident and slightly improved translation quality.",
            `<div class="g-two">
              <div><div class="tl-subhead">Hard target</div>${barList([1, 0, 0, 0, 0], ["liebe", "hasse", "mag", "habe", "bin"], { max: 1, asPercent: true, highlight: 0 })}</div>
              <div><div class="tl-subhead">Smoothed target</div>${barList([0.92, 0.02, 0.02, 0.02, 0.02], ["liebe", "hasse", "mag", "habe", "bin"], { max: 1, asPercent: true, highlight: 0 })}</div>
            </div>`)}
          ${figure("9.3", "The paper's settings next to our toy model.",
            `<div class="table-panel"><table>
              <thead><tr><th>Setting</th><th>Toy model</th><th>Paper: base</th><th>Paper: big</th></tr></thead>
              <tbody>
                <tr><td>Layers N</td><td>1</td><td>6</td><td>6</td></tr>
                <tr><td>d<sub>model</sub> / d<sub>ff</sub></td><td>4 / 8</td><td>512 / 2048</td><td>1024 / 4096</td></tr>
                <tr><td>Heads h / d<sub>k</sub></td><td>2 / 2</td><td>8 / 64</td><td>16 / 64</td></tr>
                <tr><td>Dropout / label smoothing</td><td>—</td><td>0.1 / 0.1</td><td>0.3 / 0.1</td></tr>
                <tr><td>Optimizer</td><td>—</td><td colspan="2">Adam, β₁ = 0.9, β₂ = 0.98, ε = 10⁻⁹</td></tr>
                <tr><td>Training</td><td>—</td><td>100K steps, 12 h on 8 GPUs</td><td>300K steps, 3.5 days</td></tr>
                <tr><td>Parameters</td><td>140 (one encoder layer)</td><td>65 M</td><td>213 M</td></tr>
                <tr><td>BLEU, English→German</td><td>—</td><td>27.3</td><td>28.4</td></tr>
              </tbody></table></div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`\text{lrate} = d_{\text{model}}^{-0.5}\cdot\min\!\big(\text{step}^{-0.5},\; \text{step}\cdot\text{warmup}^{-1.5}\big),\qquad \text{warmup} = 4000`)}
            ${ref("Vaswani et al. 2017, §5")}
          `)}
          ${takeaway("Next-word cross-entropy, teacher forcing, warm-up, dropout and label smoothing: a careful recipe matters as much as the architecture.")}
        `)}

        ${chapter("tf-why", 10, "The argument", "Why it replaced RNNs", `
          ${plain(para("Before Transformers, sequence models were mostly <strong>RNNs</strong>: they read one word at a time, passing a memory vector along. That has two costs. Information from word 1 must survive many hand-offs to reach word 50, and fades on the way. And word 50 can't be processed until words 1–49 are done, which wastes a GPU's parallel power. Self-attention fixes both: every word reaches every other in <strong>one step</strong>, all at once."))}
          ${figure("10.1", "Top: in an RNN, word 1 reaches word 8 through 7 hand-offs, one after another. Bottom: with self-attention, word 8 reaches every word directly, in parallel.", svgSlot("paths", "0 0 560 260", "RNN chain versus direct attention"))}
          ${figure("10.2", "The paper's Table 1, with numbers for a 50-word sentence and d = 512. Self-attention wins on all three, as long as sentences are shorter than d. Its weak spot: cost grows with n², so very long inputs get expensive.",
            `<div class="table-panel"><table>
              <thead><tr><th>Layer</th><th>Cost per layer</th><th>Sequential steps</th><th>Longest path</th></tr></thead>
              <tbody>
                <tr><td><b>Self-attention</b></td><td class="mono">O(n²·d) ≈ 1.3 M</td><td class="mono">O(1) = 1</td><td class="mono">O(1) = 1</td></tr>
                <tr><td><b>Recurrent</b></td><td class="mono">O(n·d²) ≈ 13.1 M</td><td class="mono">O(n) = 50</td><td class="mono">O(n) = 50</td></tr>
              </tbody></table></div>`)}
          ${ref("Vaswani et al. 2017, §4, Table 1")}
          ${takeaway("Shorter paths and full parallelism made Transformers both better and much faster to train.")}
        `)}

        ${chapter("tf-recap", 11, "Recap", "One sentence through the whole model", `
          <ol class="g-recap">
            <li><b>Tokens → vectors</b> via the embedding table (× √d<sub>model</sub>).</li>
            <li><b>+ positional encoding</b>, so word order is visible.</li>
            <li><b>Encoder × 6</b>: attention (talk) → add & norm → feed-forward (think) → add & norm.</li>
            <li><b>Decoder × 6</b>: masked self-attention → add & norm → cross-attention to the encoder → add & norm → feed-forward → add & norm.</li>
            <li><b>Linear + softmax</b>: probabilities for the next word; pick one, append, repeat.</li>
            <li><b>Training</b>: next-word cross-entropy with teacher forcing, warm-up schedule, dropout, label smoothing.</li>
          </ol>
          ${quiz([
            { q: "Remove positional encoding. What can the model still do, and what not?", a: "It still knows which words are present and how they relate by content, but it can't distinguish “dog bites man” from “man bites dog” (Chapter 3)." },
            { q: "What is the difference between the decoder's two attention layers?", a: "Masked self-attention: Q, K and V all come from the German words so far, with the future masked. Cross-attention: Q from the decoder, K and V from the encoder output (Chapters 6–7)." },
            { q: "Generation is word by word. How can training be parallel?", a: "In training the whole true target is fed in at once (teacher forcing) and the causal mask hides the future, so all positions are predicted in one pass (Chapter 6)." },
            { q: "Why x + Sublayer(x) instead of Sublayer(x)?", a: "The identity path keeps gradients flowing through deep stacks, and each layer only needs to learn a correction (Chapter 4b)." },
            { q: "When is an RNN layer cheaper than self-attention?", a: "When the sequence is longer than the width (n > d): O(n·d²) < O(n²·d). The RNN still needs n sequential steps (Chapter 10)." },
          ])}
          <div class="tl-next">
            <div><div class="eyebrow">Read the original</div><strong>Vaswani et al., “Attention Is All You Need” (2017)</strong><p class="caption">§3.1–3.5 → chapters 1–8 · §4 → chapter 10 · §5 → chapter 9 · §6 → results and ablations.</p></div>
            <a class="button primary" href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener">Open on arXiv ↗</a>
          </div>
        `)}
      `;
    }

    function render() {
      rootNode.innerHTML = body();
      drawFigures(rootNode, {
        map: drawGuideMap,
        "pe-heat": (svg) => drawPeHeatmap(svg, PE_POS, PE_D, PE_P),
        "pe-sim": (svg) => drawPeSimilarity(svg, PE_POS, PE_D, PE_P),
        lr: drawLrSchedule,
        paths: (svg) => drawPaths(svg, 8),
      });
    }

    /* Badges and boxes on Figure 1 jump to their chapter. */
    const PART_CHAPTER = {
      embed: "tf-embedding", pe: "tf-positions", mha: "tf-encoder", addnorm: "tf-encoder", ffn: "tf-encoder",
      stack: "tf-stack", masked: "tf-masking", cross: "tf-cross", linear: "tf-generate", softmax: "tf-generate",
    };
    const jump = (target) => {
      const id = target.getAttribute("data-goto") || PART_CHAPTER[target.getAttribute("data-part")];
      const node = id && document.getElementById(id);
      if (node) node.scrollIntoView({ behavior: "smooth", block: "start" });
    };
    rootNode.addEventListener("click", (event) => {
      const target = event.target.closest("[data-goto], [data-part]");
      if (target) jump(target);
    });
    rootNode.addEventListener("keydown", (event) => {
      const target = event.target.closest("[data-goto], [data-part]");
      if (target && (event.key === "Enter" || event.key === " ")) {
        event.preventDefault();
        jump(target);
      }
    });

    render();
    U().onRedraw(render);
    startProgressBar();
  }

  root.MLExtraLabs = Object.assign(root.MLExtraLabs || {}, {
    attention: mountAttentionGuide,
    transformer: mountTransformerGuide,
  });

  /* Shared building blocks for other sequence-model pages. */
  root.MLSeqUI = {
    fmt, pct, rgba, cellStyle, heatmap, barList, featureCompare, sentenceStrip, quiz, onClickAttr,
    svgText, arrowMarker, tex, paintSwatches, para, plain, deeper, takeaway, figure, chapter, guideToc,
    flow, mathBlock, svgSlot, startProgressBar, drawFigures,
  };
})();
