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

  function chips(items, active, attr) {
    return `<div class="tl-chips">${items
      .map(
        (item, i) =>
          `<button type="button" class="tl-chip${i === active ? " is-active" : ""}" ${attr}="${i}">${item}</button>`
      )
      .join("")}</div>`;
  }

  function lesson(id, number, title, intro, paper, body) {
    return `
      <section class="lab tl-lesson" id="${id}">
        <div class="section-header">
          <div>
            <div class="eyebrow">Step ${number}</div>
            <h2>${title}</h2>
          </div>
          ${paper ? `<span class="tl-paper" title="Where this appears in “Attention Is All You Need”">📄 Paper ${paper}</span>` : ""}
        </div>
        <p class="section-intro">${intro}</p>
        ${body}
      </section>`;
  }

  function lessonNav(items) {
    return `
      <nav class="tl-nav" aria-label="Lessons on this page">
        ${items.map((item, i) => `<a href="#${item.id}"><b>${i + 1}</b>${item.label}</a>`).join("")}
      </nav>`;
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

  /* ════════════════════════════════════════════════════════════════
     ATTENTION PAGE
     ════════════════════════════════════════════════════════════════ */

  function mountAttention(rootNode, helpers) {
    const nav = [
      { id: "att-context", label: "Why words need context" },
      { id: "att-qkv", label: "Query, key and value" },
      { id: "att-matrix", label: "Every word at once" },
      { id: "att-scale", label: "Why divide by √dₖ" },
      { id: "att-heads", label: "Multi-head attention" },
      { id: "att-check", label: "Check yourself" },
    ];

    rootNode.innerHTML = `
      <section class="lab tl-intro">
        <div class="section-header">
          <div>
            <div class="eyebrow">Start here</div>
            <h2>A neuron with weights that change for every sentence</h2>
          </div>
        </div>
        <p class="section-intro">
          A neuron is a weighted sum: <strong>Σ wⱼ·xⱼ</strong>. Attention is a weighted sum too. What's new is
          where the weights come from. A neuron's weights are learned once and then fixed. Attention computes
          its weights <em>on the fly, from the words themselves</em>, so they are different for every sentence.
          That is the whole idea. The steps below build it one piece at a time.
        </p>
        <div class="tl-intro-grid">
          <div class="soft-box">
            <strong>What you need to know</strong>
            <ul class="tl-list">
              <li><b>Dot product</b> a·b: large when two vectors point the same way.</li>
              <li><b>Softmax</b>: turns any list of scores into positive weights that add up to 1.</li>
              <li><b>Matrix multiply</b>: the same as a dense layer applied to every row.</li>
            </ul>
          </div>
          <div class="soft-box">
            <strong>The toy model used on this page</strong>
            <p class="caption">
              Each word is 4 numbers with names you can read: <b>thing</b>, <b>alive</b>, <b>action/state</b>,
              <b>refers back</b>. The weights are set by hand so the patterns make sense. The paper uses 512
              unnamed numbers learned from data, but the mechanics are identical, and every number here is computed live.
            </p>
          </div>
        </div>
        ${lessonNav(nav)}
      </section>
      <div id="att-context"></div>
      <div id="att-qkv"></div>
      <div id="att-matrix"></div>
      <div id="att-scale"></div>
      <div id="att-heads"></div>
      <div id="att-check"></div>
    `;

    const host = (id) => rootNode.querySelector(`#${id}`);
    /* The wrapper divs carry the ids for the nav links; the sections
       inside them get distinct ids. */
    mountContextLesson(host("att-context"), helpers);
    mountQkvLesson(host("att-qkv"), helpers);
    mountMatrixLesson(host("att-matrix"));
    mountScaleLesson(host("att-scale"), helpers);
    mountHeadsLesson(host("att-heads"));
    mountAttentionQuiz(host("att-check"));
  }

  /* ── Step 1: context, with raw similarity ─────────────────────── */
  function mountContextLesson(node, helpers) {
    node.innerHTML = lesson(
      "att-context-lab",
      1,
      "Why words need context",
      "An embedding gives each word one fixed vector. So “bank” gets the same vector next to “river” as next to “loan”, even though it means something different. Here is the simplest fix: build a new vector for each word as a <strong>weighted average of the words around it</strong>, giving more weight to words that are more similar. The two numbers per word below are “money-ness” and “nature-ness”.",
      "§3.2 (motivation)",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-group">
              <label>Sentence</label>
              <div data-slot="sentences"></div>
            </div>
            <div class="control-group">
              <label>Focus word (click one)</label>
              <div data-slot="strip"></div>
            </div>
            <div class="control-group">
              <label for="ctx-sharp">Sharpness β (multiplies every score)</label>
              <div class="range-row">
                <input id="ctx-sharp" type="range" min="0" max="12" step="0.5" value="3" />
                <span class="range-value" id="ctx-sharp-value">3.0</span>
              </div>
            </div>
            <div class="callout" data-slot="callout"></div>
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg data-slot="plot" viewBox="0 0 560 360" aria-label="Words as 2D vectors and the focus word's new position"></svg>
              <div class="legend">
                <span><i data-swatch="c"></i> Original vector</span>
                <span><i data-swatch="b"></i> New vector (after mixing)</span>
                <span><i data-swatch="a"></i> Line thickness = weight</span>
              </div>
            </div>
            <div class="plot-card tl-pad">
              <div class="tl-subhead">How much each word contributes</div>
              <div data-slot="bars"></div>
            </div>
            <div class="equation-card" data-slot="math"></div>
          </div>
        </div>
      `
    );

    const sharpInput = node.querySelector("#ctx-sharp");
    const sharpValue = node.querySelector("#ctx-sharp-value");
    const slot = (name) => node.querySelector(`[data-slot="${name}"]`);
    let sentenceIndex = 0;
    let focus = 2;

    onClickAttr(slot("sentences"), "data-ctx-sentence", (i) => {
      sentenceIndex = i;
      focus = math.tokensOf(CONTEXT_SENTENCES[i]).indexOf("bank");
      render();
    });
    onClickAttr(slot("strip"), "data-ctx-token", (i) => {
      focus = i;
      render();
    });
    sharpInput.addEventListener("input", render);

    function render() {
      const sentence = CONTEXT_SENTENCES[sentenceIndex];
      const tokens = math.tokensOf(sentence);
      const X = tokens.map((token) => CONTEXT_VOCAB[token]);
      const beta = Number(sharpInput.value);
      sharpValue.textContent = beta.toFixed(1);
      const result = rawAttention(X, focus, beta);
      const before = X[focus];
      const after = result.out;

      paintSwatches(node);
      slot("sentences").innerHTML = chips(
        CONTEXT_SENTENCES.map((entry) => entry.label),
        sentenceIndex,
        "data-ctx-sentence"
      );
      slot("strip").innerHTML = sentenceStrip(tokens, focus, result.weights, "data-ctx-token");

      const self = result.weights[focus];
      const dx = after[0] - before[0];
      const dy = after[1] - before[1];
      let direction = "barely moved: the words around it give no clear direction";
      if (Math.hypot(dx, dy) > 0.04) direction = dy > dx ? "moved toward <strong>nature</strong>" : "moved toward <strong>money</strong>";
      slot("callout").innerHTML = `
        “${tokens[focus]}” keeps <strong>${pct(self)}</strong> of itself and takes <strong>${pct(1 - self)}</strong>
        from the other words. Its vector ${direction}:
        (${fmt(before[0])}, ${fmt(before[1])}) → (${fmt(after[0])}, ${fmt(after[1])}).
        ${beta >= 8 ? "<br><br><strong>Notice:</strong> at high sharpness the word attends almost only to <em>itself</em>, because a vector is always most similar to itself. Context disappears." : ""}
        ${beta === 0 ? "<br><br>At β = 0 every score becomes 0, so every word gets the same weight: a plain average." : ""}`;

      slot("bars").innerHTML = barList(result.weights, tokens, { max: 1, highlight: focus, asPercent: true });

      drawContextPlot(slot("plot"), tokens, X, focus, result.weights, after);

      const order = tokens.map((_, j) => j).sort((a, b) => result.weights[b] - result.weights[a]).slice(0, 3);
      helpers.renderFormulaCards(slot("math"), [
        {
          title: "Score, then weight, then mix",
          description: "Three lines, which are the same three lines the paper uses. Only the vectors that go into them change in step 2.",
          tex: String.raw`x'_i = \sum_j \operatorname{softmax}_j\!\big(\beta \, x_i \cdot x_j\big)\, x_j`,
          derivation: [
            ...order.map((j) => ({
              tex: String.raw`\text{score}(\text{${tokens[focus]}}, \text{${tokens[j]}}) = ${fmt(before[0])}\cdot${fmt(X[j][0])} + ${fmt(before[1])}\cdot${fmt(X[j][1])} = ${fmt(math.dotV(before, X[j]), 3)}`,
              result: `w = ${fmt(result.weights[j], 3)}`,
            })),
            {
              tex: String.raw`x'_{\text{${tokens[focus]}}} = (${fmt(after[0], 3)},\; ${fmt(after[1], 3)})`,
              result: "new vector",
              note: "The weights add up to 1, so the new vector is a blend that stays among the original words.",
            },
          ],
          insight:
            "<strong>Two problems remain, and step 2 fixes both.</strong> (1) Raise the sharpness and the word only looks at itself. (2) Similar is not the same as relevant: “it” needs “cat”, but the two words are not alike at all.",
        },
      ]);
    }

    render();
    U().onRedraw(render);
  }

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

  /* ── Step 2: query, key, value ────────────────────────────────── */
  function mountQkvLesson(node, helpers) {
    const head = HEADS[0];
    node.innerHTML = lesson(
      "att-qkv-lab",
      2,
      "Query, key and value: asking the right question",
      "The fix is to give every word <strong>three different vectors</strong>, each made by its own learned weight matrix. The <strong>query</strong> is what the word is looking for. The <strong>key</strong> is what the word advertises to others. The <strong>value</strong> is what the word hands over if it is picked. A word's score for another word is <em>its query · their key</em>, so relevance no longer needs the two words to be similar.",
      "§3.2.1",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-group">
              <label>Sentence</label>
              <div data-slot="sentences"></div>
            </div>
            <div class="callout" data-slot="callout"></div>
            <div class="soft-box tl-analogy">
              <strong>Analogy: searching a library</strong>
              <dl>
                <dt>Query</dt><dd>What you type into the search box: “a living thing”.</dd>
                <dt>Key</dt><dd>The label on each book's spine: “I'm about a cat”.</dd>
                <dt>Value</dt><dd>What's inside the book: what you actually take home.</dd>
              </dl>
              <p class="caption">Unlike a real library, you don't take one book. You take a little of every book, in proportion to how well its label matches your search.</p>
            </div>
          </div>
          <div class="two-column">
            <div class="plot-card tl-pad">
              <div class="tl-subhead">Click any word to see where it looks</div>
              <div data-slot="strip-raw"></div>
              <div data-slot="strip-qk"></div>
            </div>
            <div class="two-up">
              <div class="plot-card">
                <svg data-slot="plot" viewBox="0 0 420 340" aria-label="The query and every key, in 2D"></svg>
                <p class="caption tl-svg-note">Bigger dot = more weight. Every key on the dashed line gets the same score as the winner: a dot product measures how far a key reaches <em>along</em> the query's direction.</p>
              </div>
              <div class="plot-card tl-pad">
                <div class="tl-subhead">What the word carries away (its value mix)</div>
                <div data-slot="features"></div>
              </div>
            </div>
            <div class="table-panel">
              <table>
                <thead><tr><th>Word</th><th>Key k</th><th>q · k</th><th>÷ √dₖ</th><th>Weight</th></tr></thead>
                <tbody data-slot="table"></tbody>
              </table>
            </div>
            <div class="equation-card" data-slot="math"></div>
          </div>
        </div>
      `
    );

    const slot = (name) => node.querySelector(`[data-slot="${name}"]`);
    let sentenceIndex = 0;
    let focus = SENTENCES[0].focus;

    onClickAttr(slot("sentences"), "data-qkv-sentence", (i) => {
      sentenceIndex = i;
      focus = SENTENCES[i].focus;
      render();
    });
    [slot("strip-raw"), slot("strip-qk")].forEach((strip) =>
      onClickAttr(strip, "data-qkv-token", (i) => {
        focus = i;
        render();
      })
    );

    function render() {
      const sentence = SENTENCES[sentenceIndex];
      const tokens = math.tokensOf(sentence);
      const X = math.embed(tokens);
      const result = attentionHead(X, head);
      const raw = rawAttention(X, focus, 1);
      const q = result.Q[focus];
      const weights = result.A[focus];
      const word = tokens[focus];

      slot("sentences").innerHTML = chips(
        SENTENCES.map((entry) => `“${entry.text}”`),
        sentenceIndex,
        "data-qkv-sentence"
      );
      slot("strip-raw").innerHTML = sentenceStrip(tokens, focus, raw.weights, "data-qkv-token", {
        label: "Plain similarity (step 1)",
      });
      slot("strip-qk").innerHTML = sentenceStrip(tokens, focus, weights, "data-qkv-token", {
        label: "Query · key (this step)",
      });

      const best = weights.indexOf(Math.max(...weights));
      const qZero = Math.hypot(q[0], q[1]) < 1e-9;
      slot("callout").innerHTML = qZero
        ? `“${word}” has a <strong>zero query</strong>: it isn't asking this head anything. Every score is 0, so every word gets the same weight, 1/${tokens.length} = ${pct(1 / tokens.length)}. A head can choose to ignore some words.`
        : `With plain similarity, “${word}” gives <strong>${pct(raw.weights[focus])}</strong> to itself. With query · key it gives
           <strong>${pct(weights[best])}</strong> to “${tokens[best]}”.<br><br>${sentence.lesson}`;

      const before = X[focus].slice(0, 2);
      slot("features").innerHTML = `
        ${featureCompare(head.vLabels, [
          { name: `“${word}” before`, values: before, color: C().c },
          { name: `“${word}” after attention`, values: result.O[focus], color: C().b },
        ])}
        <p class="caption">${
          word === "it"
            ? "“it” started with <b>alive = 0</b>. After attention it carries the “alive” it borrowed from the noun it found. That is how context gets into a word."
            : "Every word's output is a weighted mix of the value vectors. Click “it” to see context being borrowed."
        }</p>`;

      slot("table").innerHTML = tokens
        .map((token, j) => {
          const k = result.K[j];
          const hot = j === best ? ' style="background:var(--accent-soft);font-weight:600"' : "";
          return `<tr${hot}><td>${token}</td><td class="mono">(${fmt(k[0])}, ${fmt(k[1])})</td><td class="mono">${fmt(result.S[focus][j])}</td><td class="mono">${fmt(result.scaled[focus][j])}</td><td class="mono">${pct(weights[j])}</td></tr>`;
        })
        .join("");

      drawQueryKeyPlot(slot("plot"), tokens, result, focus);

      const k = result.K[best];
      helpers.renderFormulaCards(slot("math"), [
        {
          title: "Three projections of the same word",
          description: `Each projection is a dense layer with no bias and no activation. The same three matrices are used for every word. For “${word}”:`,
          tex: String.raw`q = x W^Q,\quad k = x W^K,\quad v = x W^V`,
          derivation: [
            {
              tex: String.raw`q_{\text{${word}}} = (${X[focus].map((v) => fmt(v)).join(",\\,")})\,W^Q = (${fmt(q[0])},\; ${fmt(q[1])})`,
              result: "query",
            },
            {
              tex: String.raw`q \cdot k_{\text{${tokens[best]}}} = ${fmt(q[0])}\cdot${fmt(k[0])} + ${fmt(q[1])}\cdot${fmt(k[1])} = ${fmt(result.S[focus][best], 3)}`,
              result: "score",
            },
            {
              tex: String.raw`\tfrac{${fmt(result.S[focus][best], 3)}}{\sqrt{2}} = ${fmt(result.scaled[focus][best], 3)} \;\xrightarrow{\text{softmax}}\; ${fmt(weights[best], 3)}`,
              result: "weight",
              note: "√dₖ with dₖ = 2 here. Step 4 explains why this division matters.",
            },
          ],
          insight:
            "<strong>Why three matrices instead of one?</strong> What a word looks for, how it is found and what it contributes are different jobs. “it” looks for a living thing but has nothing useful to offer by itself. Separate W<sup>Q</sup>, W<sup>K</sup> and W<sup>V</sup> let one word play all three roles differently.",
        },
      ]);
    }

    render();
    U().onRedraw(render);
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

  /* ── Step 3: the matrix form, stage by stage ──────────────────── */
  function mountMatrixLesson(node) {
    const head = HEADS[0];
    const sentence = SENTENCES[0];
    const tokens = math.tokensOf(sentence);
    const X = math.embed(tokens);
    const result = attentionHead(X, head);
    const n = tokens.length;

    const stages = [
      {
        name: "Stack the words",
        text: "Put the sentence into a matrix, <strong>one row per word</strong>. Everything after this is matrix arithmetic on all rows at once.",
        tex: String.raw`X \in \mathbb{R}^{n \times d_{\text{model}}}`,
        shape: `${n} × 4`,
      },
      {
        name: "Project to Q, K, V",
        text: "Multiply by three learned weight matrices. Each row of <b>Q</b>, <b>K</b> and <b>V</b> is that word's query, key and value. All words share the same weights, just as one dense layer is shared by every example in a batch.",
        tex: String.raw`Q = XW^Q,\quad K = XW^K,\quad V = XW^V`,
        shape: `${n} × 2 each`,
      },
      {
        name: "Score every pair",
        text: "<b>QKᵀ</b> takes the dot product of every query with every key in a single multiply. Row <em>i</em>, column <em>j</em> answers: how well does word <em>i</em>'s question match word <em>j</em>'s label?",
        tex: String.raw`S = QK^{\top}`,
        shape: `${n} × ${n}`,
      },
      {
        name: "Scale",
        text: "Divide every score by <b>√dₖ</b> (here √2). This keeps the scores in a range where softmax still has useful gradients. Step 4 shows why.",
        tex: String.raw`S' = \frac{QK^{\top}}{\sqrt{d_k}}`,
        shape: `${n} × ${n}`,
      },
      {
        name: "Softmax each row",
        text: "Turn each row into weights that are positive and <strong>add up to 1</strong>. This is the <em>attention matrix</em>, the heatmap you see in papers and blog posts. Row <em>i</em> is where word <em>i</em> looks.",
        tex: String.raw`A = \operatorname{softmax}_{\text{row}}\!\Big(\frac{QK^{\top}}{\sqrt{d_k}}\Big)`,
        shape: `${n} × ${n}`,
      },
      {
        name: "Mix the values",
        text: "Multiply by <b>V</b>: each output row is the weighted average of all value rows, using that word's attention weights. That gives Equation 1 of the paper.",
        tex: String.raw`\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\Big(\frac{QK^{\top}}{\sqrt{d_k}}\Big)V`,
        shape: `${n} × 2`,
      },
    ];

    node.innerHTML = lesson(
      "att-matrix-lab",
      3,
      "Every word at once: the matrix form",
      "Step 2 followed one word. In practice <strong>every word asks its question at the same time</strong>, and the whole computation becomes a few matrix multiplies. No loop runs over the words, so a GPU can do all of them in parallel. An RNN cannot: it must finish word 1 before starting word 2.",
      "§3.2.1 · Eq. 1",
      `
        <div class="tl-stepper">
          <div class="tl-stepper-bar">
            <div data-slot="dots"></div>
            <div class="step-controls">
              <button type="button" class="button secondary" data-step="-1">← Previous</button>
              <button type="button" class="button primary" data-step="1">Next stage →</button>
            </div>
          </div>
          <div class="tl-stage">
            <div class="tl-stage-text">
              <h3 data-slot="title"></h3>
              <p data-slot="text"></p>
              <div class="formula-tex" data-slot="tex"></div>
              <p class="caption">Click a word on the left of any matrix to follow its row through every stage.</p>
            </div>
            <div class="tl-stage-mats" data-slot="mats"></div>
          </div>
        </div>
      `
    );

    const slot = (name) => node.querySelector(`[data-slot="${name}"]`);
    let stage = 0;
    let focus = sentence.focus;

    node.querySelectorAll("[data-step]").forEach((button) =>
      button.addEventListener("click", () => {
        stage = Math.max(0, Math.min(stages.length - 1, stage + Number(button.getAttribute("data-step"))));
        render();
      })
    );
    onClickAttr(slot("dots"), "data-stage", (i) => {
      stage = i;
      render();
    });
    onClickAttr(slot("mats"), "data-mat-row", (i) => {
      focus = i;
      render();
    });

    function render() {
      const current = stages[stage];
      slot("dots").innerHTML = chips(
        stages.map((entry, i) => `${i + 1}. ${entry.name}`),
        stage,
        "data-stage"
      );
      slot("title").textContent = `${stage + 1}. ${current.name}`;
      slot("text").innerHTML = current.text;
      slot("tex").innerHTML = tex(current.tex, true);

      const common = { rows: tokens, focusRow: focus, rowAttr: "data-mat-row" };
      const square = { ...common, cols: tokens, colHint: "keys →" };
      let html = "";
      if (stage === 0) {
        html = heatmap(X, { ...common, cols: FEATURES, title: "X: word vectors", shape: current.shape });
      } else if (stage === 1) {
        html = `
          <div class="tl-mats-row">
            ${heatmap(result.Q, { ...common, cols: ["q₁", "q₂"], title: "Q", shape: `${n} × 2` })}
            ${heatmap(result.K, { ...common, cols: ["k₁", "k₂"], title: "K", shape: `${n} × 2` })}
            ${heatmap(result.V, { ...common, cols: head.vLabels, title: "V", shape: `${n} × 2` })}
          </div>
          <details class="tl-weights">
            <summary>Show the learned weight matrices W<sup>Q</sup>, W<sup>K</sup>, W<sup>V</sup> (4 × 2 each)</summary>
            <div class="tl-mats-row">
              ${heatmap(head.WQ, { rows: FEATURES, cols: ["q₁", "q₂"], title: "W<sup>Q</sup>" })}
              ${heatmap(head.WK, { rows: FEATURES, cols: ["k₁", "k₂"], title: "W<sup>K</sup>" })}
              ${heatmap(head.WV, { rows: FEATURES, cols: head.vLabels, title: "W<sup>V</sup>" })}
            </div>
            <p class="caption">Read a row as: “this input feature feeds these outputs”. For example, “refers back” feeds the query strongly, so pronouns ask the most insistent questions.</p>
          </details>`;
      } else if (stage === 2) {
        html = heatmap(result.S, { ...square, title: "S = QKᵀ (raw scores)", shape: current.shape });
      } else if (stage === 3) {
        html = heatmap(result.scaled, { ...square, title: "S ÷ √dₖ", shape: current.shape });
      } else if (stage === 4) {
        html = `${heatmap(result.A, { ...square, maxAbs: 1, title: "A: attention weights (each row sums to 1)", shape: current.shape })}
          <p class="caption">Row “${tokens[focus]}” sums to ${fmt(result.A[focus].reduce((a, b) => a + b, 0), 3)}.</p>`;
      } else {
        html = `
          <div class="tl-mats-row">
            ${heatmap(result.A, { ...square, maxAbs: 1, title: "A", shape: `${n} × ${n}`, digits: 2 })}
            <div class="tl-times">×</div>
            ${heatmap(result.V, { ...common, cols: head.vLabels, title: "V", shape: `${n} × 2` })}
            <div class="tl-times">=</div>
            ${heatmap(result.O, { ...common, cols: head.vLabels, title: "Output", shape: `${n} × 2` })}
          </div>
          <p class="caption">Output row “${tokens[focus]}” = Σⱼ A[${tokens[focus]}, j] · V[j] = (${fmt(result.O[focus][0])}, ${fmt(result.O[focus][1])}).</p>`;
      }
      slot("mats").innerHTML = html;
      node.querySelector('[data-step="-1"]').disabled = stage === 0;
      node.querySelector('[data-step="1"]').disabled = stage === stages.length - 1;
    }

    render();
    U().onRedraw(render);
  }

  /* ── Step 4: why divide by √dk ────────────────────────────────── */
  function mountScaleLesson(node, helpers) {
    const sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    node.innerHTML = lesson(
      "att-scale-lab",
      4,
      "Why divide by √dₖ?",
      "The paper's footnote 4 gives the reason. A dot product adds up dₖ products. If each product is random noise of size about 1, the total grows like <strong>√dₖ</strong>. In the paper dₖ = 64, so raw scores are about 8× larger than they should be. Softmax over very large scores puts almost all the weight on one key and gives near-zero gradient to the rest. Here random queries and keys play the role of an untrained network.",
      "§3.2.1 · footnote 4",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-group">
              <label for="scale-dk">Key size dₖ</label>
              <div class="range-row">
                <input id="scale-dk" type="range" min="0" max="${sizes.length - 1}" step="1" value="6" />
                <span class="range-value" id="scale-dk-value">64</span>
              </div>
            </div>
            <div class="step-controls">
              <button type="button" class="button secondary" data-slot="resample">New random query &amp; keys</button>
            </div>
            <div class="callout" data-slot="callout"></div>
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg data-slot="plot" viewBox="0 0 560 250" aria-label="Softmax weights over 8 keys, with and without scaling"></svg>
              <div class="legend">
                <span><i data-swatch="b"></i> Without scaling: softmax(q·k)</span>
                <span><i data-swatch="a"></i> With scaling: softmax(q·k / √dₖ)</span>
              </div>
            </div>
            <div class="readout"><div class="output-grid" data-slot="metrics"></div></div>
            <div class="equation-card" data-slot="math"></div>
          </div>
        </div>
      `
    );

    const slot = (name) => node.querySelector(`[data-slot="${name}"]`);
    const dkInput = node.querySelector("#scale-dk");
    const dkValue = node.querySelector("#scale-dk-value");
    let seed = 1;
    dkInput.addEventListener("input", render);
    slot("resample").addEventListener("click", () => {
      seed += 1;
      render();
    });

    function render() {
      const dk = sizes[Number(dkInput.value)];
      dkValue.textContent = String(dk);
      const rng = seeded(seed * 7919 + dk);
      const vec = () => Array.from({ length: dk }, () => gaussian(rng));
      const q = vec();
      const keys = Array.from({ length: 8 }, vec);
      const raw = keys.map((k) => math.dotV(q, k));
      const scaled = raw.map((value) => value / Math.sqrt(dk));
      const wRaw = softmax(raw);
      const wScaled = softmax(scaled);

      /* Empirical spread of q·k over many fresh pairs. */
      const spreadRng = seeded(seed * 104729 + dk * 31);
      const samples = Array.from({ length: 400 }, () => {
        let total = 0;
        for (let i = 0; i < dk; i += 1) total += gaussian(spreadRng) * gaussian(spreadRng);
        return total;
      });
      const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
      const std = Math.sqrt(samples.reduce((a, b) => a + (b - mean) ** 2, 0) / samples.length);
      const gradient = (weights) => weights.reduce((acc, w) => acc + w * (1 - w), 0);

      paintSwatches(node);
      U().renderMetrics(slot("metrics"), [
        { label: "Spread of q·k (std)", value: `${fmt(std)} ≈ √${dk} = ${fmt(Math.sqrt(dk))}` },
        { label: "Spread after ÷ √dₖ", value: fmt(std / Math.sqrt(dk)) },
        { label: "Top weight, unscaled", value: pct(Math.max(...wRaw)) },
        { label: "Top weight, scaled", value: pct(Math.max(...wScaled)) },
        { label: "Gradient signal Σw(1−w), unscaled", value: fmt(gradient(wRaw), 3) },
        { label: "Gradient signal Σw(1−w), scaled", value: fmt(gradient(wScaled), 3) },
      ]);

      const top = Math.max(...wRaw);
      slot("callout").innerHTML =
        dk <= 2
          ? "With a tiny dₖ the two versions barely differ, because √dₖ is close to 1."
          : top > 0.9
            ? `Unscaled, one key takes <strong>${pct(top)}</strong> of the weight. Softmax is <strong>saturated</strong>: it acts like a hard max, and the gradient that would teach the other keys is nearly zero. Scaling keeps the distribution soft enough to learn from.`
            : `At dₖ = ${dk} the unscaled scores are already ${fmt(Math.sqrt(dk), 1)}× too spread out. Drag dₖ higher, or resample: the orange bars keep collapsing onto a single key.`;

      drawScaleBars(slot("plot"), wRaw, wScaled);

      helpers.renderFormulaCards(slot("math"), [
        {
          title: "Where the √dₖ comes from",
          description: "Assume the entries of q and k are independent, with mean 0 and variance 1. This is roughly true at initialisation.",
          tex: String.raw`q \cdot k = \sum_{i=1}^{d_k} q_i k_i, \qquad \operatorname{Var}(q_i k_i) = 1`,
          derivation: [
            { tex: String.raw`\operatorname{Var}(q\cdot k) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i) = d_k = ${dk}`, result: `std = ${fmt(Math.sqrt(dk))}` },
            { tex: String.raw`\operatorname{Var}\!\Big(\frac{q\cdot k}{\sqrt{d_k}}\Big) = \frac{d_k}{d_k} = 1`, result: "std = 1", note: "Dividing by the standard deviation brings the scores back to unit size, whatever dₖ is." },
          ],
          insight:
            "<strong>The same disease as sigmoid.</strong> On the activation-functions page, a saturated sigmoid has a near-zero slope and stops learning. A saturated softmax fails the same way. The softmax Jacobian is diag(w) − wwᵀ, and every entry goes to 0 as one weight goes to 1.",
        },
      ]);
    }

    render();
    U().onRedraw(render);
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

  /* ── Step 5: multi-head ───────────────────────────────────────── */
  function mountHeadsLesson(node) {
    node.innerHTML = lesson(
      "att-heads-lab",
      5,
      "Multi-head attention: several questions at once",
      "One attention pattern can capture only one kind of relationship. Language has many: who a pronoun means, what a noun did, which adjective goes with which noun. So the transformer runs <strong>h smaller attentions side by side</strong>, each with its own W<sup>Q</sup>, W<sup>K</sup>, W<sup>V</sup>. It then glues their outputs back together and mixes them with one more matrix, W<sup>O</sup>.",
      "§3.2.2",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-group">
              <label>Sentence</label>
              <div data-slot="sentences"></div>
            </div>
            <div class="control-group">
              <label>Focus word (click one)</label>
              <div data-slot="strip"></div>
            </div>
            <div class="callout" data-slot="callout"></div>
            <div class="soft-box">
              <strong>Sizes in the paper</strong>
              <p class="caption">d<sub>model</sub> = 512 is split into <b>h = 8</b> heads of d<sub>k</sub> = 512 / 8 = <b>64</b>.
              Eight 64-wide heads cost about the same as one 512-wide head, so you get eight different patterns for the price of one.
              Here: d<sub>model</sub> = 4, h = 2, d<sub>k</sub> = 2.</p>
            </div>
          </div>
          <div class="two-column">
            <div class="two-up" data-slot="heads"></div>
            <div class="plot-card tl-pad">
              <div class="tl-subhead">Shapes, from input to output</div>
              <div data-slot="shapes"></div>
            </div>
            <div class="plot-card tl-pad">
              <div class="tl-subhead" data-slot="feature-title"></div>
              <div data-slot="features"></div>
            </div>
            <div class="equation-card">
              <div class="formula-tex">${tex(String.raw`\operatorname{MultiHead}(Q,K,V) = \operatorname{Concat}(\text{head}_1,\ldots,\text{head}_h)\,W^O`, true)}</div>
              <div class="formula-tex">${tex(String.raw`\text{head}_i = \operatorname{Attention}(XW_i^Q,\; XW_i^K,\; XW_i^V)`, true)}</div>
            </div>
          </div>
        </div>
      `
    );

    const slot = (name) => node.querySelector(`[data-slot="${name}"]`);
    let sentenceIndex = 0;
    let focus = SENTENCES[0].focus;

    onClickAttr(slot("sentences"), "data-mh-sentence", (i) => {
      sentenceIndex = i;
      focus = SENTENCES[i].focus;
      render();
    });
    onClickAttr(node, "data-mh-token", (i) => {
      focus = i;
      render();
    });

    function render() {
      const tokens = math.tokensOf(SENTENCES[sentenceIndex]);
      const X = math.embed(tokens);
      const result = multiHead(X);
      const word = tokens[focus];

      slot("sentences").innerHTML = chips(SENTENCES.map((entry) => `“${entry.text}”`), sentenceIndex, "data-mh-sentence");
      slot("strip").innerHTML = sentenceStrip(tokens, focus, null, "data-mh-token");

      slot("heads").innerHTML = HEADS.map((head, h) => {
        const A = result.heads[h].A;
        return `
          <div class="plot-card tl-pad">
            <div class="tl-subhead">${head.name}</div>
            <p class="caption">${head.idea}</p>
            ${sentenceStrip(tokens, focus, A[focus], "data-mh-token", { label: `“${word}” looks at` })}
            ${heatmap(A, { rows: tokens, cols: tokens, focusRow: focus, maxAbs: 1, rowAttr: "data-mh-token", colHint: "keys →" })}
          </div>`;
      }).join("");

      const n = tokens.length;
      slot("shapes").innerHTML = `
        <div class="tl-flow">
          <span class="tl-flow-box">X<small>${n} × 4</small></span><span class="tl-flow-arrow">→</span>
          <span class="tl-flow-stack">
            <span class="tl-flow-box">head 1<small>${n} × 2</small></span>
            <span class="tl-flow-box">head 2<small>${n} × 2</small></span>
          </span><span class="tl-flow-arrow">→</span>
          <span class="tl-flow-box">concat<small>${n} × 4</small></span><span class="tl-flow-arrow">→</span>
          <span class="tl-flow-box"><span>× W<sup>O</sup></span><small>${n} × 4</small></span>
        </div>
        <p class="caption">The output has the same shape as the input. That is what lets the paper stack the block 6 times and add the input back in (next page).</p>`;

      slot("feature-title").textContent = `“${word}”: its own vector vs. what multi-head attention brings back`;
      slot("features").innerHTML = `${featureCompare(FEATURES, [
        { name: "its own vector x", values: X[focus], color: C().c },
        { name: "multi-head output", values: result.out[focus], color: C().b },
      ])}
        <p class="caption">Head 1 fills in <b>thing / alive</b> and head 2 fills in <b>action / refers back</b>. W<sup>O</sup> is the identity here so the names survive. A trained W<sup>O</sup> mixes the heads together.</p>`;

      const h2 = result.heads[1].A[focus];
      const actions = tokens
        .map((token, j) => ({ token, w: h2[j] }))
        .filter((entry) => VOCAB[entry.token][2] > 0.5)
        .sort((a, b) => b.w - a.w);
      slot("callout").innerHTML =
        actions.length > 1 && h2[focus] < 0.5 && VOCAB[word][2] < 0.5
          ? `Head 2 spreads “${word}” across <strong>${actions.map((entry) => `${entry.token} ${pct(entry.w)}`).join(", ")}</strong>. It can't tell which action belongs to “${word}”: the words could be in any order and the scores would be the same. <strong>Attention has no sense of word order.</strong> The <a href="./algorithm.html?id=transformer#tf-order">Transformer page</a> fixes that next.`
          : `Two heads, two different questions about the same sentence. Click a noun or “it” to see the heads disagree about where to look.`;
    }

    render();
    U().onRedraw(render);
  }

  function mountAttentionQuiz(node) {
    node.innerHTML = lesson(
      "att-check-lab",
      6,
      "Check yourself",
      "Try to answer each question before you open it. If one surprises you, go back to the step it comes from.",
      "",
      `${quiz([
        {
          q: "Why can't attention just use the raw word vectors (Q = K = V = X)?",
          a: "Two reasons (step 1). Every word would score highest with itself, and relevance would require similarity. “it” needs “cat” even though the two words aren't alike. Separate W<sup>Q</sup> and W<sup>K</sup> let “looking for” differ from “offering”.",
        },
        {
          q: "A word's query is all zeros. What does its attention look like?",
          a: "Every score q·kⱼ is 0, so softmax gives every word the same weight 1/n and the output is the plain average of the values. A head can effectively opt out for some words (step 2, try “cat” in head 1).",
        },
        {
          q: "What goes wrong if you double dₖ but forget the √dₖ?",
          a: "The spread of the scores grows by √2. Softmax gets peakier, moves toward a one-hot pick, and its gradient shrinks. Training slows or stalls (step 4).",
        },
        {
          q: "Why are 8 heads of size 64 about as costly as 1 head of size 512?",
          a: "The projections have the same total size: 8 × (512 × 64) = 512 × 512. The score matrices are n × n per head either way. You get 8 independent patterns for roughly the same compute (step 5).",
        },
        {
          q: "Swap the order of the words. What happens to each word's attention output?",
          a: "Nothing, apart from moving with the word. The outputs are the same vectors, just reordered, because nothing in QKᵀ depends on position. That is the problem positional encoding solves on the Transformer page.",
        },
      ])}
      <div class="tl-next">
        <div>
          <div class="eyebrow">Next</div>
          <strong>Build the full Transformer from this block</strong>
          <p class="caption">Word order, residual connections, the decoder, masking and the training recipe from the paper.</p>
        </div>
        <a class="button primary" href="./algorithm.html?id=transformer">Go to the Transformer →</a>
      </div>`
    );
  }

  /* ════════════════════════════════════════════════════════════════
     TRANSFORMER PAGE
     ════════════════════════════════════════════════════════════════ */

  function mountTransformer(rootNode, helpers) {
    const nav = [
      { id: "tf-map", label: "The whole map" },
      { id: "tf-order", label: "Word order is lost" },
      { id: "tf-pe", label: "Positional encoding" },
      { id: "tf-block", label: "The encoder block" },
      { id: "tf-mask", label: "Decoder: no peeking" },
      { id: "tf-generate", label: "Decoder: translating" },
      { id: "tf-why", label: "Why it beat RNNs" },
      { id: "tf-train", label: "Training recipe" },
      { id: "tf-check", label: "Check yourself" },
    ];

    rootNode.innerHTML = `
      <section class="lab tl-intro">
        <div class="section-header">
          <div>
            <div class="eyebrow">Start here</div>
            <h2>Attention, plus four things that make it work</h2>
          </div>
        </div>
        <p class="section-intro">
          The <a href="./algorithm.html?id=attention">Attention page</a> built the core operation: every word gathers
          information from every other word. A Transformer wraps that operation with four things it can't do alone:
          a sense of <strong>word order</strong>, a <strong>per-word feed-forward network</strong> (your ANN),
          <strong>residual connections with normalisation</strong> so dozens of layers still train, and a
          <strong>decoder</strong> that writes the output one word at a time. The paper's title is literal. There is
          no recurrence and no convolution, only attention and these supporting parts.
        </p>
        <div class="tl-intro-grid">
          <div class="soft-box">
            <strong>Running examples</strong>
            <p class="caption">The encoder reads “the cat sat because it was tired” (the same sentence and the same toy weights as the Attention page).
            The decoder translates “I love cats” → “ich liebe Katzen”, English to German, which was the paper's main task.</p>
          </div>
          <div class="soft-box">
            <strong>Toy vs. paper</strong>
            <p class="caption">Here: d<sub>model</sub> = 4, h = 2, d<sub>ff</sub> = 8, one layer. Paper (base): d<sub>model</sub> = 512, h = 8, d<sub>ff</sub> = 2048, N = 6 layers.
            The decoder's weights are set by hand to illustrate the ideas.</p>
          </div>
        </div>
        ${lessonNav(nav)}
      </section>
      <div id="tf-map"></div>
      <div id="tf-order"></div>
      <div id="tf-pe"></div>
      <div id="tf-block"></div>
      <div id="tf-mask"></div>
      <div id="tf-generate"></div>
      <div id="tf-why"></div>
      <div id="tf-train"></div>
      <div id="tf-check"></div>
    `;

    const host = (id) => rootNode.querySelector(`#${id}`);
    mountMapLesson(host("tf-map"));
    mountOrderLesson(host("tf-order"));
    mountPositionLesson(host("tf-pe"), helpers);
    mountBlockLesson(host("tf-block"), helpers);
    mountMaskLesson(host("tf-mask"));
    mountGenerateLesson(host("tf-generate"));
    mountWhyLesson(host("tf-why"));
    mountTrainLesson(host("tf-train"));
    mountTransformerQuiz(host("tf-check"));
  }

  /* ── Step 1: the architecture map (the paper's Figure 1) ──────── */

  const PARTS = {
    embed: {
      title: "Input / output embedding",
      what: "Looks up a learned vector for each token. Mathematically it is a one-hot vector times a matrix: a dense layer with no bias.",
      why: "The network needs numbers, and words with similar meanings should get similar vectors.",
      paper: "§3.4",
      sizes: "≈37,000 shared sub-word tokens → 512 numbers each. The paper multiplies embeddings by √d<sub>model</sub> and shares one matrix between both embeddings and the final Linear layer.",
      link: null,
    },
    pe: {
      title: "Positional encoding ⊕",
      what: "Adds a fixed pattern of sines and cosines, different for every position, to each word vector.",
      why: "Attention ignores order. This addition is the <em>only</em> place where word order enters the model.",
      paper: "§3.5",
      sizes: "Same width as the embedding (512), so it can simply be added.",
      link: "tf-order",
    },
    mha: {
      title: "Multi-head self-attention",
      what: "Every source word looks at every source word, in 8 heads at once.",
      why: "Moves information <em>between</em> words, so each word's vector picks up its context.",
      paper: "§3.2",
      sizes: "8 heads × d<sub>k</sub> = 64. Score matrix n × n per head.",
      link: "attention",
    },
    addnorm: {
      title: "Add & Norm",
      what: "Adds the sub-layer's input back to its output (x + Sublayer(x)), then applies layer normalisation.",
      why: "The residual “add” gives gradients a direct path through deep stacks. The norm keeps each word vector at a steady scale.",
      paper: "§3.1, §5.4",
      sizes: "Dropout (0.1) is applied to the sub-layer output just before the add.",
      link: "tf-block",
    },
    ffn: {
      title: "Position-wise feed-forward",
      what: "One small ANN, 512 → 2048 → 512 with ReLU, applied to <em>each word separately</em> using the same weights.",
      why: "Attention mixes information across words. The FFN then processes each word with what it gathered. About two thirds of the parameters live here.",
      paper: "§3.3",
      sizes: "d<sub>ff</sub> = 2048 (4 × d<sub>model</sub>).",
      link: "tf-block",
    },
    masked: {
      title: "Masked multi-head self-attention",
      what: "Each target word attends only to itself and the words <em>before</em> it.",
      why: "During training the whole target sentence is fed in at once. The mask stops position i from peeking at the word it is supposed to predict.",
      paper: "§3.2.3",
      sizes: "Same as encoder attention, with −∞ added above the diagonal.",
      link: "tf-mask",
    },
    cross: {
      title: "Encoder–decoder (cross) attention",
      what: "Queries come from the decoder. Keys and values come from the encoder's final output.",
      why: "This is where the translation reads the source sentence. Every decoder layer can look at any source word directly.",
      paper: "§3.2.3",
      sizes: "Score matrix: target length × source length.",
      link: "tf-generate",
    },
    linear: {
      title: "Linear",
      what: "Maps each 512-number vector to one score per vocabulary word.",
      why: "Turns “a vector that means something” into “a score for every possible next word”.",
      paper: "§3.4",
      sizes: "512 → ≈37,000, with weights shared with the embeddings.",
      link: "tf-generate",
    },
    softmax: {
      title: "Softmax",
      what: "Turns the scores into next-word probabilities.",
      why: "Training pushes up the probability of the correct next word (cross-entropy with label smoothing).",
      paper: "§3.4, §5.4",
      sizes: "One probability distribution per position.",
      link: "tf-generate",
    },
    stack: {
      title: "N× (the stack)",
      what: "The block inside the frame is repeated N = 6 times, each copy with its own weights.",
      why: "Each layer refines the previous one. Because every block outputs the same shape it takes in, stacking is trivial.",
      paper: "§3.1",
      sizes: "N = 6 in both encoder and decoder.",
      link: "tf-block",
    },
  };

  function mountMapLesson(node) {
    node.innerHTML = lesson(
      "tf-map-lab",
      1,
      "The whole map: the paper's Figure 1",
      "This is the diagram everyone recognises from the paper. On the left, the <strong>encoder</strong> reads the source sentence. On the right, the <strong>decoder</strong> writes the translation. <strong>Click any box</strong> to see what it does, why it is there, and which step on this page covers it. The rest of the page works through the boxes bottom to top.",
      "§3 · Figure 1",
      `
        <div class="tl-map">
          <div class="plot-card">
            <svg data-slot="svg" viewBox="0 0 720 640" role="img" aria-label="Transformer architecture: encoder on the left, decoder on the right"></svg>
          </div>
          <div class="tl-map-info" data-slot="info" aria-live="polite"></div>
        </div>
      `
    );

    const svg = node.querySelector('[data-slot="svg"]');
    const info = node.querySelector('[data-slot="info"]');
    let selected = "mha";

    function select(part) {
      selected = part;
      render();
    }

    svg.addEventListener("click", (event) => {
      const target = event.target.closest("[data-part]");
      if (target) select(target.getAttribute("data-part"));
    });
    svg.addEventListener("keydown", (event) => {
      const target = event.target.closest("[data-part]");
      if (target && (event.key === "Enter" || event.key === " ")) {
        event.preventDefault();
        select(target.getAttribute("data-part"));
      }
    });

    function render() {
      drawArchitecture(svg, selected);
      const part = PARTS[selected];
      const link = part.link === "attention"
        ? `<a class="button secondary" href="./algorithm.html?id=attention">Covered on the Attention page →</a>`
        : part.link
          ? `<a class="button secondary" href="#${part.link}">Go to that step ↓</a>`
          : "";
      info.innerHTML = `
        <div class="eyebrow">Selected</div>
        <h3>${part.title}</h3>
        <dl class="tl-def">
          <dt>What it does</dt><dd>${part.what}</dd>
          <dt>Why it's there</dt><dd>${part.why}</dd>
          <dt>Sizes in the paper</dt><dd>${part.sizes}</dd>
        </dl>
        <div class="tl-map-foot"><span class="tl-paper">📄 Paper ${part.paper}</span>${link}</div>`;
    }

    render();
    U().onRedraw(render);
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

  /* ── Step 2: attention can't see word order ───────────────────── */
  function mountOrderLesson(node) {
    node.innerHTML = lesson(
      "tf-order-lab",
      2,
      "The problem: attention can't see word order",
      "Nothing in QKᵀ depends on <em>where</em> a word sits. Each score depends only on the two words involved. So feed in the same words in a different order and every word gets <strong>exactly the same output</strong>, just in a different row. To plain attention, “dog bites man” and “man bites dog” are the same sentence. An RNN never had this problem because it reads in order. The transformer needs another way to know where each word is.",
      "§3.5",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-group">
              <label class="tl-toggle"><input type="checkbox" data-slot="pe" /> Add positional encoding to the inputs</label>
            </div>
            <div class="callout" data-slot="callout"></div>
            <p class="caption">This demo uses its own small 4-number vectors and random (seeded) weights. The conclusion holds for <em>any</em> weights.</p>
          </div>
          <div class="two-column">
            <div class="two-up" data-slot="maps"></div>
            <div class="table-panel">
              <table>
                <thead><tr><th>Word</th><th>In “dog bites man”</th><th>In “man bites dog”</th><th>Diff.</th></tr></thead>
                <tbody data-slot="table"></tbody>
              </table>
            </div>
          </div>
        </div>
      `
    );

    const slot = (name) => node.querySelector(`[data-slot="${name}"]`);
    slot("pe").addEventListener("change", render);

    function render() {
      const withPe = slot("pe").checked;
      const a = ["dog", "bites", "man"];
      const b = ["man", "bites", "dog"];
      const runA = orderDemo(a, withPe);
      const runB = orderDemo(b, withPe);
      slot("maps").innerHTML = [
        [a, runA],
        [b, runB],
      ]
        .map(
          ([tokens, run]) => `
          <div class="plot-card tl-pad">
            <div class="tl-subhead">“${tokens.join(" ")}”: attention weights</div>
            ${heatmap(run.A, { rows: tokens, cols: tokens, maxAbs: 1, colHint: "keys →" })}
          </div>`
        )
        .join("");

      let biggest = 0;
      slot("table").innerHTML = a
        .map((word, i) => {
          const j = b.indexOf(word);
          const oa = runA.O[i];
          const ob = runB.O[j];
          const diff = Math.hypot(...oa.map((v, d) => v - ob[d]));
          biggest = Math.max(biggest, diff);
          const vec = (v) => `(${v.map((x) => fmt(x)).join(", ")})`;
          return `<tr><td><strong>${word}</strong></td><td class="mono">${vec(oa)}</td><td class="mono">${vec(ob)}</td><td class="mono">${fmt(diff, 3)}</td></tr>`;
        })
        .join("");

      slot("callout").innerHTML = withPe
        ? `With positions added, “dog” as the biter and “dog” as the bitten now get <strong>different</strong> outputs (difference up to ${fmt(biggest, 3)}). The model can finally tell who did what.`
        : `Every difference is <strong>0.000</strong>. The second heatmap is the first one with its rows and columns shuffled. The model can't tell who bit whom. Tick the box to add positions.`;
    }

    render();
    U().onRedraw(render);
  }

  /* ── Step 3: positional encoding ──────────────────────────────── */
  function mountPositionLesson(node, helpers) {
    const D = 32;
    const P = 50;
    const pairsShown = [0, 1, 3, 7];
    node.innerHTML = lesson(
      "tf-pe-lab",
      3,
      "Positional encoding: a clock for every position",
      "The fix is to <strong>add</strong> a position signal to each word vector before the first layer. The paper's signal works like a set of clock hands that turn at different speeds. The first pair of numbers spins fast (about one turn every 6 positions), the next more slowly, and the last takes thousands of positions per turn. Read together, the hands give every position a unique time, much like the digits of a binary counter, but smooth. Nothing here is learned: the pattern is a fixed formula.",
      "§3.5",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-group">
              <label for="pe-pos">Position</label>
              <div class="range-row">
                <input id="pe-pos" type="range" min="0" max="${P - 1}" step="1" value="7" />
                <span class="range-value" id="pe-pos-value">7</span>
              </div>
            </div>
            <div class="callout" data-slot="callout"></div>
            <div class="soft-box">
              <strong>Why add it, rather than append it?</strong>
              <p class="caption">Adding keeps the width at d<sub>model</sub>, so nothing downstream changes. The network can learn to keep meaning and position in different directions of the same space. The paper multiplies embeddings by √d<sub>model</sub> first, so position doesn't drown out meaning.</p>
            </div>
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg data-slot="heat" viewBox="0 0 560 300" aria-label="Positional encoding matrix: positions by dimensions"></svg>
            </div>
            <div class="plot-card tl-pad">
              <div class="tl-subhead">Four of the 16 “clock hands” at this position</div>
              <div class="tl-dials" data-slot="dials"></div>
            </div>
            <div class="plot-card">
              <svg data-slot="sim" viewBox="0 0 560 220" aria-label="Similarity of this position's code to every other position"></svg>
            </div>
            <div class="equation-card" data-slot="math"></div>
          </div>
        </div>
      `
    );

    const slot = (name) => node.querySelector(`[data-slot="${name}"]`);
    const posInput = node.querySelector("#pe-pos");
    const posValue = node.querySelector("#pe-pos-value");
    posInput.addEventListener("input", render);
    const omega = (pair) => 1 / Math.pow(10000, (2 * pair) / D);

    function render() {
      const pos = Number(posInput.value);
      posValue.textContent = String(pos);
      const codes = Array.from({ length: P }, (_, p) => positionalEncoding(p, D));

      /* Heatmap of the whole matrix. */
      const heat = slot("heat");
      U().clear(heat);
      const pad = { left: 44, top: 26, right: 12, bottom: 24 };
      const cellW = (560 - pad.left - pad.right) / D;
      const cellH = (300 - pad.top - pad.bottom) / P;
      svgText(heat, pad.left, 16, `PE matrix: ${P} positions (rows) × ${D} dimensions (columns)`, { class: "svg-title" });
      codes.forEach((row, p) =>
        row.forEach((value, d) => {
          heat.appendChild(
            U().svgEl("rect", {
              x: pad.left + d * cellW,
              y: pad.top + p * cellH,
              width: cellW + 0.4,
              height: cellH + 0.4,
              fill: rgba(value >= 0 ? C().a : C().b, Math.abs(value) * 0.9),
            })
          );
        })
      );
      heat.appendChild(
        U().svgEl("rect", {
          x: pad.left - 2,
          y: pad.top + pos * cellH - 1,
          width: D * cellW + 4,
          height: cellH + 2,
          fill: "none",
          stroke: C().ink,
          "stroke-width": 2,
        })
      );
      svgText(heat, pad.left - 6, pad.top + pos * cellH + cellH, `pos ${pos}`, { "text-anchor": "end" });
      svgText(heat, pad.left, 300 - 8, "← fast-turning dims");
      svgText(heat, 560 - pad.right, 300 - 8, "slow-turning dims →", { "text-anchor": "end" });

      /* Clock dials for a few frequency pairs. */
      slot("dials").innerHTML = pairsShown
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
              <div><b>dims ${2 * pair}, ${2 * pair + 1}</b><br /><span class="caption">sin = ${fmt(Math.sin(angle))}, cos = ${fmt(Math.cos(angle))}</span><br /><span class="caption">one turn per ${period < 100 ? fmt(period, 1) : Math.round(period)} positions</span></div>
            </div>`;
        })
        .join("");

      /* Similarity of this code to every other position. */
      const sims = codes.map((code) => math.dotV(codes[pos], code));
      const sim = slot("sim");
      const chart = U().makeChart(sim, {
        xDomain: [0, P - 1],
        yDomain: [Math.min(...sims) - 0.5, D / 2 + 1],
        title: `Dot product of position ${pos}'s code with every position`,
      });
      sim.appendChild(
        U().svgEl("path", {
          d: U().pathFromPoints(sims.map((value, p) => ({ x: p, y: value })), chart.xScale, chart.yScale),
          fill: "none",
          stroke: C().a,
          "stroke-width": 2.5,
        })
      );
      sim.appendChild(
        U().svgEl("circle", { cx: chart.xScale(pos), cy: chart.yScale(sims[pos]), r: 6, fill: C().b, stroke: C().ring, "stroke-width": 2 })
      );

      const near = sims[Math.min(P - 1, pos + 1)];
      const far = sims[pos + 20 < P ? pos + 20 : Math.max(0, pos - 20)];
      slot("callout").innerHTML = `Position ${pos} is matched best by itself (dot product ${fmt(sims[pos], 1)}), then its neighbours (${fmt(near, 1)}), with far positions lower (${fmt(far, 1)}). Nearby positions get similar codes, which gives attention a way to prefer nearby words.`;

      const k = 3;
      const theta = k * omega(0);
      helpers.renderFormulaCards(slot("math"), [
        {
          title: "The formula, and why the paper chose it",
          description: "Each pair of dimensions (2i, 2i+1) is one clock hand: a sine and a cosine of the same angle, turning at its own speed ωᵢ.",
          tex: String.raw`\begin{aligned} PE_{(pos,\,2i)} &= \sin\!\big(pos\cdot\omega_i\big) \\ PE_{(pos,\,2i+1)} &= \cos\!\big(pos\cdot\omega_i\big) \end{aligned} \qquad \omega_i = \frac{1}{10000^{2i/d_{\text{model}}}}`,
          derivation: [
            {
              tex: String.raw`\begin{pmatrix}\sin\omega(p{+}k)\\ \cos\omega(p{+}k)\end{pmatrix} = R(\omega k)\begin{pmatrix}\sin\omega p\\ \cos\omega p\end{pmatrix}`,
              result: "shift = rotate",
            },
            {
              tex: String.raw`R(\theta) = \begin{pmatrix}\cos\theta & \sin\theta\\ -\sin\theta & \cos\theta\end{pmatrix}`,
              result: "rotation",
              note: "Moving k positions forward is the same fixed rotation, whatever p is. The paper's reasoning: “PE<sub>pos+k</sub> can be represented as a linear function of PE<sub>pos</sub>”, so relative offsets are easy for attention to learn.",
            },
            {
              tex: String.raw`\text{e.g. } k = ${k},\ \omega_0 = 1:\ \text{rotate by } ${fmt(theta, 2)} \text{ rad at every position}`,
              result: "same for all p",
            },
          ],
          insight:
            "<strong>Learned or fixed?</strong> The paper also tried learned position vectors and got nearly identical results (Table 3, row E). It kept the sinusoids because they might extend to sentences longer than any seen in training.",
        },
      ]);
    }

    render();
    U().onRedraw(render);
  }

  /* ── Step 4: one encoder block, stage by stage ────────────────── */
  function mountBlockLesson(node, helpers) {
    const sentence = SENTENCES[0];
    const tokens = math.tokensOf(sentence);
    const X = math.embed(tokens);
    const run = encoderLayer(X);
    const n = tokens.length;
    const hiddenCols = Array.from({ length: math.FFN.W1[0].length }, (_, j) => `h${j + 1}`);

    const stages = [
      {
        name: "Input",
        flow: 0,
        matrix: run.X,
        cols: FEATURES,
        title: "X: word vectors",
        text: "One row per word, the same four named features as the Attention page. (We leave positional encoding out here so the feature names stay readable. In the real model it has already been added.)",
        tex: String.raw`X \in \mathbb{R}^{${n}\times 4}`,
      },
      {
        name: "Multi-head attention",
        flow: 1,
        matrix: run.mha.out,
        cols: FEATURES,
        title: "Z = MultiHead(X)",
        text: "<strong>Communication between words.</strong> Every word gathers information from the others, exactly as on the Attention page. “it” comes back carrying <em>alive</em> from “cat”.",
        tex: String.raw`Z = \operatorname{MultiHead}(X, X, X)`,
        compare: () => [run.X, "x", run.mha.out, "attention output z"],
      },
      {
        name: "Add (residual)",
        flow: 2,
        matrix: run.added1,
        cols: FEATURES,
        title: "X + Z",
        text: "<strong>Add the input back.</strong> The attention output is an <em>update</em> on top of the original word, not a replacement. “it” keeps “refers back” and gains “alive”.",
        tex: String.raw`X + Z`,
        compare: () => [run.X, "x", run.added1, "x + z"],
        residual: true,
      },
      {
        name: "Layer norm",
        flow: 3,
        matrix: run.norm1,
        cols: FEATURES,
        title: "LayerNorm(X + Z)",
        text: "<strong>Re-centre and re-scale each word on its own</strong>: subtract the row's mean and divide by its standard deviation. Each word vector stays at a steady size however many layers are stacked. (The learned scale γ and shift β are 1 and 0 here.)",
        tex: String.raw`\operatorname{LN}(x) = \gamma\,\frac{x - \mu}{\sigma} + \beta`,
        compare: () => [run.added1, "before norm", run.norm1, "after norm"],
        norm: true,
      },
      {
        name: "Feed-forward: hidden",
        flow: 4,
        matrix: run.hidden,
        cols: hiddenCols,
        title: "ReLU(x W₁ + b₁)",
        text: "<strong>Computation within each word.</strong> This is a plain ANN hidden layer, 4 → 8 here (512 → 2048 in the paper), applied to <em>each row separately with the same weights</em>. Zeros are ReLU switching units off. (FFN weights here are random but fixed.)",
        tex: String.raw`H = \max(0,\; xW_1 + b_1)`,
      },
      {
        name: "Feed-forward: output",
        flow: 4,
        matrix: run.ffn,
        cols: FEATURES,
        title: "H W₂ + b₂",
        text: "The second layer projects back to d<sub>model</sub> = 4, so the block's output shape matches its input.",
        tex: String.raw`\operatorname{FFN}(x) = \max(0,\; xW_1 + b_1)\,W_2 + b_2`,
      },
      {
        name: "Add & norm → output",
        flow: 5,
        matrix: run.out,
        cols: FEATURES,
        title: "Block output",
        text: "Add the FFN's input back and normalise again. The result has the same shape as X, which is what lets the paper stack <strong>6</strong> of these blocks, each feeding the next.",
        tex: String.raw`\operatorname{LN}\big(h + \operatorname{FFN}(h)\big),\quad h = \operatorname{LN}(X + Z)`,
        compare: () => [run.X, "block input", run.out, "block output"],
        residual: true,
      },
    ];

    const flowNames = ["Input", "Attention", "Add", "Norm", "Feed-forward", "Add & Norm"];

    node.innerHTML = lesson(
      "tf-block-lab",
      4,
      "The encoder block: talk, then think",
      "One encoder layer alternates two jobs. <strong>Attention</strong> lets words exchange information (communication). The <strong>feed-forward network</strong> then processes each word on its own (computation). Each job is wrapped in a residual “add” and a layer norm. Step through it and click any word to follow its row.",
      "§3.1 · §3.3",
      `
        <div class="tl-stepper">
          <div class="tl-flow" data-slot="flow"></div>
          <div class="tl-stepper-bar">
            <div data-slot="dots"></div>
            <div class="step-controls">
              <button type="button" class="button secondary" data-step="-1">← Previous</button>
              <button type="button" class="button primary" data-step="1">Next stage →</button>
            </div>
          </div>
          <div class="tl-stage">
            <div class="tl-stage-text">
              <h3 data-slot="title"></h3>
              <p data-slot="text"></p>
              <div class="formula-tex" data-slot="tex"></div>
              <div data-slot="extra"></div>
            </div>
            <div class="tl-stage-mats">
              <div data-slot="mat"></div>
              <div data-slot="compare"></div>
            </div>
          </div>
        </div>
      `
    );

    const slot = (name) => node.querySelector(`[data-slot="${name}"]`);
    let stage = 0;
    let focus = sentence.focus;

    node.querySelectorAll("[data-step]").forEach((button) =>
      button.addEventListener("click", () => {
        stage = Math.max(0, Math.min(stages.length - 1, stage + Number(button.getAttribute("data-step"))));
        render();
      })
    );
    onClickAttr(slot("dots"), "data-block-stage", (i) => {
      stage = i;
      render();
    });
    onClickAttr(slot("mat"), "data-block-row", (i) => {
      focus = i;
      render();
    });

    function render() {
      const current = stages[stage];
      slot("flow").innerHTML = flowNames
        .map((name, i) => `<span class="tl-flow-box${i === current.flow ? " is-on" : ""}">${name}</span>`)
        .join('<span class="tl-flow-arrow">→</span>');
      slot("dots").innerHTML = chips(stages.map((entry, i) => `${i + 1}. ${entry.name}`), stage, "data-block-stage");
      slot("title").textContent = `${stage + 1}. ${current.name}`;
      slot("text").innerHTML = current.text;
      slot("tex").innerHTML = tex(current.tex, true);
      slot("mat").innerHTML = heatmap(current.matrix, {
        rows: tokens,
        cols: current.cols,
        focusRow: focus,
        rowAttr: "data-block-row",
        title: current.title,
        shape: `${n} × ${current.cols.length}`,
      });

      if (current.compare) {
        const [a, aName, b, bName] = current.compare();
        slot("compare").innerHTML = `
          <div class="plot-card tl-pad">
            <div class="tl-subhead">“${tokens[focus]}”: ${aName} vs ${bName}</div>
            ${featureCompare(FEATURES, [
              { name: aName, values: a[focus], color: C().c },
              { name: bName, values: b[focus], color: C().b },
            ], Math.max(1.2, maxAbsOf([a[focus], b[focus]])))}
          </div>`;
      } else {
        slot("compare").innerHTML = "";
      }

      let extra = "";
      if (current.residual) {
        extra = `<div class="insight-box"><strong>Why the residual “add” matters.</strong> ${tex(String.raw`\frac{\partial\,(x + F(x))}{\partial x} = I + \frac{\partial F}{\partial x}`)}. Even if a sub-layer's own gradient is tiny, the identity term <b>I</b> passes the gradient back unchanged. That fixes the vanishing gradients you saw on the <a href="./algorithm.html?id=backpropagation">Backpropagation</a> page and makes a stack of 6 (or 100) layers trainable.</div>`;
      } else if (current.norm) {
        const row = run.added1[focus];
        const mean = row.reduce((a, b) => a + b, 0) / row.length;
        const std = Math.sqrt(row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length);
        extra = `<div class="insight-box">For “${tokens[focus]}”: μ = ${fmt(mean, 3)}, σ = ${fmt(std, 3)}. After normalising, this row has mean 0 and standard deviation 1. Unlike batch norm, it uses only this word's own numbers, so it works for one sentence at a time and for any length.</div>`;
      } else if (stage === 4) {
        const zeros = run.hidden[focus].filter((v) => v === 0).length;
        extra = `<div class="insight-box">“${tokens[focus]}” switches off <strong>${zeros} of ${run.hidden[focus].length}</strong> hidden units. This is the ANN you already know. The only new detail is that it runs once per word, and every word shares its weights.</div>`;
      }
      slot("extra").innerHTML = extra;

      node.querySelector('[data-step="-1"]').disabled = stage === 0;
      node.querySelector('[data-step="1"]').disabled = stage === stages.length - 1;
    }

    render();
    U().onRedraw(render);
  }

  /* ── Step 5: the causal mask ──────────────────────────────────── */
  function mountMaskLesson(node) {
    node.innerHTML = lesson(
      "tf-mask-lab",
      5,
      "The decoder, part 1: no peeking",
      "When translating, the decoder writes one word at a time. In training, though, the whole target sentence is fed in at once, shifted right by one, so every position learns in parallel. Row “ich” must predict the next word, “liebe”. Without protection, it could simply <em>look at</em> “liebe” in its own input. The fix is to set every score above the diagonal to <strong>−∞</strong> before the softmax. e<sup>−∞</sup> = 0, so future words get exactly zero weight.",
      "§3.2.3",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-group">
              <label class="tl-toggle"><input type="checkbox" data-slot="mask" checked /> Apply the causal mask</label>
            </div>
            <div class="callout" data-slot="callout"></div>
            <div class="soft-box">
              <strong>What each row must predict</strong>
              <ul class="tl-list">
                ${DECODER_INPUT.map((token, i) => `<li><b>${token.replace("<", "&lt;").replace(">", "&gt;")}</b> → ${i < TARGET.length ? TARGET[i] : "&lt;/s&gt; (end)"}</li>`).join("")}
              </ul>
            </div>
          </div>
          <div class="two-column">
            <div class="two-up" data-slot="maps"></div>
            <p class="caption">Scores are hand-set for this demo so that, unmasked, each row leans on exactly the word it should be predicting, which is the shortcut a real model would find.</p>
          </div>
        </div>
      `
    );

    const slot = (name) => node.querySelector(`[data-slot="${name}"]`);
    slot("mask").addEventListener("change", render);
    const rows = DECODER_INPUT.map((token) => token.replace("<", "&lt;").replace(">", "&gt;"));

    function render() {
      const masked = slot("mask").checked;
      const scores = maskedScores(SELF_SCORES, masked);
      const weights = scores.map(softmax);
      slot("maps").innerHTML = `
        <div class="plot-card tl-pad">${heatmap(scores, { rows, cols: rows, maxAbs: 3, title: masked ? "Scores + mask" : "Scores (no mask)", colHint: "keys →" })}</div>
        <div class="plot-card tl-pad">${heatmap(weights, { rows, cols: rows, maxAbs: 1, title: "Softmax weights", colHint: "keys →" })}</div>`;
      const cheat = weights[1][2];
      slot("callout").innerHTML = masked
        ? `Everything above the diagonal is <strong>0</strong>. Row “ich” sees only “&lt;s&gt;” and “ich”, exactly what it will have when generating for real. Training and generation now match.`
        : `Row “ich” puts <strong>${pct(cheat)}</strong> of its attention on “liebe”, <em>the very word it is meant to predict</em>. The model would learn to copy, get near-perfect training loss, and fail at generation time, when that word doesn't exist yet.`;
    }

    render();
    U().onRedraw(render);
  }

  /* ── Step 6: generation with cross-attention ──────────────────── */
  function mountGenerateLesson(node) {
    node.innerHTML = lesson(
      "tf-generate-lab",
      6,
      "The decoder, part 2: reading the source and writing the translation",
      "The encoder runs <strong>once</strong> and turns “I love cats” into a set of vectors. The decoder then loops: given the words written so far, it predicts the next one. In each decoder layer, <strong>cross-attention</strong> sends a query from the current position to the encoder's keys and values, which is how the translation looks at the source. Press “Next word” to watch it translate.",
      "§3.2.3 · §3.4",
      `
        <div class="tl-stepper">
          <div class="tl-stepper-bar">
            <div class="tl-gen-status" data-slot="status"></div>
            <div class="step-controls">
              <button type="button" class="button secondary" data-gen="-1">← Back</button>
              <button type="button" class="button primary" data-gen="1">Next word →</button>
              <button type="button" class="button ghost" data-gen="0">Restart</button>
            </div>
          </div>
          <div class="tl-gen">
            <div class="plot-card tl-pad">
              <div class="tl-subhead">Encoder input (read once)</div>
              <div data-slot="source"></div>
              <div class="tl-subhead">Decoder input so far</div>
              <div data-slot="target"></div>
              <div class="tl-subhead">Cross-attention: where the newest position looks in the source</div>
              <div data-slot="cross"></div>
            </div>
            <div class="plot-card tl-pad">
              <div class="tl-subhead">Next-word probabilities (Linear → Softmax)</div>
              <div data-slot="probs"></div>
              <div class="callout" data-slot="callout"></div>
            </div>
          </div>
          <div class="plot-card tl-pad">
            <div class="tl-subhead">Full cross-attention so far (rows: decoder positions · columns: English words)</div>
            <div data-slot="matrix"></div>
            <p class="caption">This alignment pattern emerges from training. Nobody tells the model that “Katzen” means “cats”. The weights here are hand-set to show the kind of pattern trained models learn.</p>
          </div>
        </div>
      `
    );

    const slot = (name) => node.querySelector(`[data-slot="${name}"]`);
    let t = 1;
    node.querySelectorAll("[data-gen]").forEach((button) =>
      button.addEventListener("click", () => {
        const move = Number(button.getAttribute("data-gen"));
        t = move === 0 ? 1 : Math.max(1, Math.min(DECODER_INPUT.length, t + move));
        render();
      })
    );
    const esc = (token) => token.replace("<", "&lt;").replace(">", "&gt;");

    function render() {
      const position = t - 1;
      const crossWeights = CROSS_SCORES.slice(0, t).map(softmax);
      const probs = NEXT_PROBS[position];
      const pick = probs.indexOf(Math.max(...probs));
      const written = DECODER_INPUT.slice(0, t).map(esc);
      const outputs = [...TARGET, "</s>"].slice(0, t).map(esc);

      slot("status").innerHTML = `Step ${t} of ${DECODER_INPUT.length} · translation so far: <strong>${outputs.join(" ")}</strong>`;
      slot("source").innerHTML = sentenceStrip(SOURCE, -1, null, "");
      slot("target").innerHTML = sentenceStrip(written, position, null, "");
      slot("cross").innerHTML = sentenceStrip(SOURCE, -1, crossWeights[position], "", { label: `“${written[position]}” →` });
      slot("probs").innerHTML = barList(probs, OUT_VOCAB.map(esc), { max: 1, highlight: pick, asPercent: true, color: C().b });
      slot("matrix").innerHTML = heatmap(crossWeights, {
        rows: written,
        cols: SOURCE,
        focusRow: position,
        maxAbs: 1,
        colHint: "English (keys & values from the encoder) →",
      });

      const looked = crossWeights[position].indexOf(Math.max(...crossWeights[position]));
      slot("callout").innerHTML =
        pick === OUT_VOCAB.length - 1
          ? `The model predicts <strong>&lt;/s&gt;</strong> (end of sentence) with ${pct(probs[pick])}. Generation stops: “${TARGET.join(" ")}”.`
          : `Looking mostly at “${SOURCE[looked]}” (${pct(crossWeights[position][looked])}), the model picks <strong>${OUT_VOCAB[pick]}</strong> (${pct(probs[pick])}). It is appended to the decoder input and the loop runs again.`;

      node.querySelector('[data-gen="-1"]').disabled = t === 1;
      node.querySelector('[data-gen="1"]').disabled = t === DECODER_INPUT.length;
    }

    render();
    U().onRedraw(render);
  }

  /* ── Step 7: why self-attention won (Table 1) ─────────────────── */
  function mountWhyLesson(node) {
    node.innerHTML = lesson(
      "tf-why-lab",
      7,
      "Why it beat RNNs: the paper's Table 1",
      "Section 4 of the paper argues the case with three numbers. <strong>Path length</strong>: how many steps information takes between two distant words, where fewer steps means long-range links are easier to learn. <strong>Sequential steps</strong>: how much work must wait for earlier work, which decides how well a GPU can parallelise. <strong>Cost per layer</strong>: the total amount of arithmetic. Drag the sentence length and watch the trade-off.",
      "§4 · Table 1",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-group">
              <label for="why-n">Sentence length n</label>
              <div class="range-row">
                <input id="why-n" type="range" min="1" max="12" step="1" value="5" />
                <span class="range-value" id="why-n-value">32</span>
              </div>
            </div>
            <p class="caption">Width d = 512, as in the paper.</p>
            <div class="callout" data-slot="callout"></div>
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg data-slot="svg" viewBox="0 0 560 260" aria-label="Path from first to last word: RNN chain versus direct attention"></svg>
            </div>
            <div class="table-panel">
              <table>
                <thead><tr><th>Layer type</th><th>Cost per layer</th><th>Sequential steps</th><th>Max path length</th></tr></thead>
                <tbody data-slot="table"></tbody>
              </table>
            </div>
          </div>
        </div>
      `
    );

    const slot = (name) => node.querySelector(`[data-slot="${name}"]`);
    const nInput = node.querySelector("#why-n");
    const nValue = node.querySelector("#why-n-value");
    nInput.addEventListener("input", render);
    const d = 512;
    const big = (value) =>
      value >= 1e9 ? `${fmt(value / 1e9, 1)} B` : value >= 1e6 ? `${fmt(value / 1e6, 1)} M` : value >= 1e3 ? `${fmt(value / 1e3, 1)} K` : String(value);

    function render() {
      const n = Math.pow(2, Number(nInput.value));
      nValue.textContent = String(n);
      const rnnCost = n * d * d;
      const attCost = n * n * d;

      slot("table").innerHTML = `
        <tr><td><strong>Self-attention</strong></td><td class="mono">O(n²·d) = ${big(attCost)}</td><td class="mono">O(1) = 1</td><td class="mono">O(1) = 1</td></tr>
        <tr><td><strong>Recurrent (RNN)</strong></td><td class="mono">O(n·d²) = ${big(rnnCost)}</td><td class="mono">O(n) = ${n}</td><td class="mono">O(n) = ${n}</td></tr>`;

      slot("callout").innerHTML =
        n < d
          ? `With n = ${n} < d = ${d}, self-attention is also <strong>cheaper</strong> (${fmt(rnnCost / attCost, 1)}× less arithmetic), and every word reaches every other in <strong>1 step instead of ${n - 1}</strong>. Typical sentences are much shorter than 512 tokens, which is the paper's argument.`
          : n === d
            ? `At n = d the costs are equal. Above this, the n² term starts to bite.`
            : `With n = ${n} > d, self-attention costs <strong>${fmt(attCost / rnnCost, 1)}× more</strong> arithmetic. This quadratic cost in sentence length is the transformer's main weakness, and why so much later research targets “efficient attention”.`;

      drawPaths(slot("svg"), n);
    }

    render();
    U().onRedraw(render);
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

  /* ── Step 8: the training recipe ──────────────────────────────── */
  function mountTrainLesson(node) {
    node.innerHTML = lesson(
      "tf-train-lab",
      8,
      "Training it: the paper's recipe",
      "The architecture is half the story. The paper also needed a careful training setup. The most distinctive part is the <strong>learning-rate schedule</strong> (Equation 3). The rate rises linearly for the first <em>warmup</em> steps, then decays with the inverse square root of the step. Early on the model is random and Adam's running estimates are poor, so big steps would knock training off course. Warm-up waits until the gradients can be trusted.",
      "§5 · Eq. 3",
      `
        <div class="lab-grid">
          <div class="controls">
            <div class="control-group">
              <label for="lr-warm">Warm-up steps</label>
              <div class="range-row">
                <input id="lr-warm" type="range" min="500" max="16000" step="500" value="4000" />
                <span class="range-value" id="lr-warm-value">4000</span>
              </div>
            </div>
            <div class="control-group">
              <label for="lr-model">Model width d<sub>model</sub></label>
              <select id="lr-model">
                <option value="512">512 (base model)</option>
                <option value="1024">1024 (big model)</option>
              </select>
            </div>
            <div class="callout" data-slot="callout"></div>
          </div>
          <div class="two-column">
            <div class="plot-card">
              <svg data-slot="svg" viewBox="0 0 560 260" aria-label="Learning rate over training steps"></svg>
            </div>
            <div class="equation-card">
              <div class="formula-tex">${tex(String.raw`\text{lrate} = d_{\text{model}}^{-0.5}\cdot\min\!\big(\text{step}^{-0.5},\; \text{step}\cdot\text{warmup}^{-1.5}\big)`, true)}</div>
            </div>
            <div class="table-panel">
              <table>
                <thead><tr><th>Setting</th><th>This page</th><th>Paper: base</th><th>Paper: big</th></tr></thead>
                <tbody>
                  <tr><td>Layers N (encoder and decoder)</td><td>1</td><td>6</td><td>6</td></tr>
                  <tr><td>d<sub>model</sub></td><td>4</td><td>512</td><td>1024</td></tr>
                  <tr><td>d<sub>ff</sub></td><td>8</td><td>2048</td><td>4096</td></tr>
                  <tr><td>Heads h</td><td>2</td><td>8</td><td>16</td></tr>
                  <tr><td>d<sub>k</sub> = d<sub>v</sub></td><td>2</td><td>64</td><td>64</td></tr>
                  <tr><td>Dropout</td><td>—</td><td>0.1</td><td>0.3</td></tr>
                  <tr><td>Label smoothing ε</td><td>—</td><td>0.1</td><td>0.1</td></tr>
                  <tr><td>Training steps</td><td>—</td><td>100K (12 h, 8 GPUs)</td><td>300K (3.5 days)</td></tr>
                  <tr><td>Parameters</td><td>140 (one encoder layer)</td><td>65 M</td><td>213 M</td></tr>
                  <tr><td>BLEU, English→German</td><td>—</td><td>27.3</td><td>28.4</td></tr>
                </tbody>
              </table>
            </div>
            <div class="tl-cards">
              <article class="soft-box">
                <h3>Optimizer</h3>
                <p>Adam with β₁ = 0.9, β₂ = 0.98, ε = 10⁻⁹. β₂ is lower than the usual 0.999, so the running variance estimate adapts faster.</p>
              </article>
              <article class="soft-box">
                <h3>Dropout</h3>
                <p>Applied to every sub-layer output before the residual add, and to the embeddings + positional encodings. Rate 0.1 for the base model.</p>
              </article>
              <article class="soft-box">
                <h3>Label smoothing (ε = 0.1)</h3>
                <p>The target is 90% on the correct word and 10% spread over the rest, instead of 100% on one word. It makes the model less over-confident: perplexity gets slightly worse, but BLEU improves.</p>
              </article>
              <article class="soft-box">
                <h3>Teacher forcing</h3>
                <p>While training, the decoder always gets the <em>true</em> previous words, not its own guesses. With the causal mask, all positions then train in a single parallel pass.</p>
              </article>
            </div>
          </div>
        </div>
      `
    );

    const slot = (name) => node.querySelector(`[data-slot="${name}"]`);
    const warmInput = node.querySelector("#lr-warm");
    const warmValue = node.querySelector("#lr-warm-value");
    const modelInput = node.querySelector("#lr-model");
    [warmInput, modelInput].forEach((input) => input.addEventListener("input", render));

    function render() {
      const warmup = Number(warmInput.value);
      const dModel = Number(modelInput.value);
      warmValue.textContent = String(warmup);
      const steps = Array.from({ length: 201 }, (_, i) => 1 + i * 500);
      const peak = learningRate(warmup, dModel, warmup);
      const curve = steps.map((s) => ({ x: s / 1000, y: learningRate(s, dModel, warmup) * 1000 }));
      const svg = slot("svg");
      const yMax = Math.max(2.5, peak * 1000 * 1.15);
      const chart = U().makeChart(svg, {
        xDomain: [0, 100],
        yDomain: [0, yMax],
        title: "Learning rate (× 10⁻³) vs. training step (thousands)",
      });
      svg.appendChild(
        U().svgEl("path", { d: U().pathFromPoints(curve, chart.xScale, chart.yScale), fill: "none", stroke: C().a, "stroke-width": 2.6 })
      );
      svg.appendChild(
        U().svgEl("line", {
          x1: chart.xScale(warmup / 1000), x2: chart.xScale(warmup / 1000),
          y1: chart.yScale(0), y2: chart.yScale(yMax),
          stroke: C().b, "stroke-dasharray": "5 5", opacity: 0.6,
        })
      );
      svg.appendChild(U().svgEl("circle", { cx: chart.xScale(warmup / 1000), cy: chart.yScale(peak * 1000), r: 6, fill: C().b, stroke: C().ring, "stroke-width": 2 }));
      svgText(svg, chart.xScale(warmup / 1000) + 8, chart.yScale(peak * 1000) - 8, `peak ${(peak * 1000).toFixed(2)}×10⁻³ at step ${warmup}`);

      slot("callout").innerHTML = `The rate peaks at step <strong>${warmup}</strong> at <strong>${(peak * 1000).toFixed(2)} × 10⁻³</strong>, then decays. A shorter warm-up gives a higher, sharper peak (more risk of early divergence). A wider model gets a smaller rate overall, because of the d<sub>model</sub><sup>−0.5</sup> factor.`;
    }

    render();
    U().onRedraw(render);
  }

  function mountTransformerQuiz(node) {
    node.innerHTML = lesson(
      "tf-check-lab",
      9,
      "Check yourself",
      "Try each one before you open it.",
      "",
      `${quiz([
        {
          q: "If you removed positional encoding, what could the model still do, and what couldn't it?",
          a: "It could still tell which words are present and how they relate by content. It could not tell order apart: “dog bites man” and “man bites dog” would give the same set of outputs (step 2).",
        },
        {
          q: "What is the difference between the decoder's two attention layers?",
          a: "Masked self-attention: queries, keys and values all come from the target words so far, with the future masked out. Cross-attention: queries come from the decoder, keys and values come from the encoder output. That is where the source sentence is read (steps 5–6).",
        },
        {
          q: "Generation is one word at a time. So how is training parallel?",
          a: "In training the whole (true) target is fed in at once (teacher forcing). The causal mask makes position i see only positions ≤ i, so all positions compute their predictions in one pass with no information leak (step 5).",
        },
        {
          q: "Why does every sub-layer use x + Sublayer(x) instead of just Sublayer(x)?",
          a: "The identity path keeps gradients flowing through deep stacks (∂/∂x = I + ∂F/∂x). Each sub-layer then only has to learn a correction, not rebuild the whole representation (step 4).",
        },
        {
          q: "When would an RNN layer be cheaper than self-attention?",
          a: "When the sequence is longer than the width (n > d). Self-attention costs O(n²·d) against the RNN's O(n·d²). The RNN still has O(n) sequential steps and O(n) path length (step 7).",
        },
        {
          q: "Where does the FFN fit into “attention is all you need”, if it isn't attention?",
          a: "The title means no recurrence and no convolution. Between words, information moves only through attention. The FFN works within each word, and it holds most of the parameters.",
        },
      ])}
      <div class="tl-next">
        <div>
          <div class="eyebrow">Read the paper next</div>
          <strong>Vaswani et al., “Attention Is All You Need” (2017)</strong>
          <p class="caption">Sections 3.1–3.5 match steps 1–6 above, Section 4 is step 7, and Section 5 is step 8. Section 6 covers results and the ablations in Table 3.</p>
        </div>
        <a class="button primary" href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener">Open on arXiv ↗</a>
      </div>`
    );
  }

  root.MLExtraLabs = Object.assign(root.MLExtraLabs || {}, {
    attention: mountAttention,
    transformer: mountTransformer,
  });
})();
