/* ── Small LLMs and distillation ────────────────────────────────
   Part 5 of the sequence-model series: how small language models are
   made good. It reuses the guide building blocks (window.MLSeqUI) and the
   widget helpers from sequence-guides.js (window.MLGuideWidgets).

   Every widget computes live: memory and speed from published model
   sizes, the Chinchilla scaling law, parameter budgets of real small
   models, softened teacher targets, the distillation loss and its
   gradient, two student networks trained in the browser, forward vs
   reverse KL, pruning with a least-squares "heal", and quantization. */
(function () {
  const root = window;
  const UI = () => root.MLSeqUI;
  const W = () => root.MLGuideWidgets;
  const M = () => root.MLTransformerMath;
  const U = () => root.MLUtils;
  const C = () => root.MLUtils.chartColors();

  const sigmoid = (z) => 1 / (1 + Math.exp(-z));
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const entropyBits = (p) => -p.reduce((acc, v) => acc + (v > 0 ? v * Math.log2(v) : 0), 0);
  const klDiv = (p, q) => p.reduce((acc, v, i) => acc + (v > 0 ? v * Math.log(v / Math.max(q[i], 1e-300)) : 0), 0);

  /* ── Shared drawing helpers ──────────────────────────────────── */

  /* A plot area with hand-picked ticks, so log axes can read "1B"
     instead of "9.0". Returns scales and an append helper. */
  function frame(svg, o) {
    U().clear(svg);
    const [, , width, height] = svg.getAttribute("viewBox").split(/\s+/).map(Number);
    const pad = Object.assign({ top: 30, right: 16, bottom: 38, left: 50 }, o.pad || {});
    const [x0, x1] = o.xDomain;
    const [y0, y1] = o.yDomain;
    const xs = (x) => pad.left + ((x - x0) / (x1 - x0)) * (width - pad.left - pad.right);
    const ys = (y) => height - pad.bottom - ((y - y0) / (y1 - y0)) * (height - pad.top - pad.bottom);
    const el = (tag, attrs) => svg.appendChild(U().svgEl(tag, attrs));
    (o.xTicks || []).forEach(([v, label]) => {
      el("line", { x1: xs(v), y1: pad.top, x2: xs(v), y2: height - pad.bottom, class: "grid-line" });
      UI().svgText(svg, xs(v), height - pad.bottom + 16, label, { "text-anchor": "middle" });
    });
    (o.yTicks || []).forEach(([v, label]) => {
      el("line", { x1: pad.left, y1: ys(v), x2: width - pad.right, y2: ys(v), class: "grid-line" });
      UI().svgText(svg, pad.left - 6, ys(v) + 4, label, { "text-anchor": "end" });
    });
    el("line", { x1: pad.left, y1: height - pad.bottom, x2: width - pad.right, y2: height - pad.bottom, class: "axis-line" });
    el("line", { x1: pad.left, y1: pad.top, x2: pad.left, y2: height - pad.bottom, class: "axis-line" });
    if (o.title) UI().svgText(svg, pad.left, 18, o.title, { class: "svg-title" });
    if (o.xLabel) UI().svgText(svg, width - pad.right, height - 4, o.xLabel, { "text-anchor": "end" });
    let clip = null;
    if (o.clipId) {
      const defs = el("defs", {});
      const path = U().svgEl("clipPath", { id: o.clipId });
      path.appendChild(U().svgEl("rect", { x: pad.left, y: pad.top, width: width - pad.left - pad.right, height: height - pad.top - pad.bottom }));
      defs.appendChild(path);
      clip = `url(#${o.clipId})`;
    }
    return { xs, ys, width, height, pad, el, clip };
  }

  const linePath = (points, xs, ys) =>
    points.map(([x, y], i) => `${i ? "L" : "M"}${xs(x).toFixed(1)} ${ys(y).toFixed(1)}`).join(" ");

  function niceTicks(lo, hi, count = 5) {
    const raw = (hi - lo) / count;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
    const ticks = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(Number(v.toFixed(6)));
    return ticks;
  }

  function sparkline(values, color) {
    if (values.length < 2) return "";
    const w = 320;
    const h = 70;
    const max = Math.max(...values);
    const min = Math.min(...values, 0);
    const d = values
      .map((v, i) => `${i ? "L" : "M"}${(8 + (i / (values.length - 1)) * (w - 16)).toFixed(1)} ${(h - 10 - ((v - min) / (max - min || 1)) * (h - 22)).toFixed(1)}`)
      .join(" ");
    return `<svg class="sl-spark" viewBox="0 0 ${w} ${h}" role="img" aria-label="loss over training steps"><path d="${d}" fill="none" stroke="${color}" stroke-width="2.2" /></svg>`;
  }

  /* Put state values back into a widget's inputs after a re-render. */
  function restoreInputs(host, s) {
    host.querySelectorAll("[data-input]").forEach((node) => {
      const key = node.getAttribute("data-input");
      if (key in s) node.value = s[key];
    });
  }

  const row = (label, html) => `<div class="sl-row"><span>${label}</span>${html}</div>`;

  /* ── Data: devices and published model configurations ────────── */

  /* Typical, rounded figures. Decoding one token reads every weight once,
     so memory bandwidth caps the speed. */
  const DEVICES = [
    { id: "phone", label: "Phone", mem: 8, bw: 60 },
    { id: "laptop", label: "Laptop", mem: 16, bw: 100 },
    { id: "gpu", label: "Gaming GPU", mem: 24, bw: 1000 },
    { id: "dc", label: "Data-centre GPU", mem: 80, bw: 3350 },
  ];
  const USABLE = 0.75;

  const SIZES = [
    { id: "135m", label: "135 M · SmolLM2", P: 0.135e9 },
    { id: "05b", label: "0.5 B · Qwen2.5", P: 0.49e9 },
    { id: "1b", label: "1.2 B · Llama 3.2 1B", P: 1.24e9 },
    { id: "4b", label: "3.8 B · Phi-3 mini", P: 3.8e9 },
    { id: "8b", label: "8 B · Llama 3.1 8B", P: 8.03e9 },
    { id: "70b", label: "70 B · Llama 3.1 70B", P: 70.6e9 },
    { id: "405b", label: "405 B · Llama 3.1 405B", P: 405e9 },
  ];
  const PRECISIONS = [["4", "32-bit"], ["2", "16-bit"], ["1", "8-bit"], ["0.5", "4-bit"]];

  /* Hoffmann et al. (2022), "Chinchilla", the fitted loss law. */
  const CH = { E: 1.69, A: 406.4, B: 410.7, a: 0.34, b: 0.28 };
  const lossND = (N, D) => CH.E + CH.A / N ** CH.a + CH.B / D ** CH.b;

  /* Published configurations (config.json of each model). */
  const ARCH = {
    smol135: { label: "SmolLM2-135M", L: 30, d: 576, heads: 9, kv: 3, hd: 64, ff: 1536, V: 49152, tied: "tied", real: "135 M" },
    qwen05: { label: "Qwen2.5-0.5B", L: 24, d: 896, heads: 14, kv: 2, hd: 64, ff: 4864, V: 151936, tied: "tied", real: "0.49 B" },
    llama1b: { label: "Llama 3.2 1B", L: 16, d: 2048, heads: 32, kv: 8, hd: 64, ff: 8192, V: 128256, tied: "tied", real: "1.24 B" },
    gemma2b: { label: "Gemma 2 2B", L: 26, d: 2304, heads: 8, kv: 4, hd: 256, ff: 9216, V: 256000, tied: "tied", real: "2.6 B" },
    phi3: { label: "Phi-3 mini", L: 32, d: 3072, heads: 32, kv: 32, hd: 96, ff: 8192, V: 32064, tied: "untied", real: "3.8 B" },
    llama8b: { label: "Llama 3.1 8B", L: 32, d: 4096, heads: 32, kv: 8, hd: 128, ff: 14336, V: 128256, tied: "untied", real: "8.0 B" },
  };
  const ARCH_KEYS = ["L", "d", "heads", "kv", "hd", "ff", "V"];

  const CLASSES = ["cat", "tiger", "dog", "fox", "car", "banana"];

  /* ── A tiny network trained in the browser ───────────────────── */

  /* The "big model": a wavy border plus a small island. */
  function teacherLogit(x, y) {
    const wave = y - 0.45 * Math.sin(3 * x) + 0.1;
    const island = 0.3 - Math.hypot(x - 0.5, y + 0.55);
    return 7 * Math.max(wave, island * 2.2);
  }

  /* 2 → H (tanh) → 1 (sigmoid). Parameters packed as
     [W1 (2H), b1 (H), w2 (H), b2], trained with Adam. */
  function netInit(H, seed) {
    const rng = M().seeded(seed);
    const p = new Float64Array(4 * H + 1);
    for (let i = 0; i < 2 * H; i += 1) p[i] = M().gaussian(rng) * 1.6;
    for (let j = 0; j < H; j += 1) p[2 * H + j] = M().gaussian(rng) * 0.6;
    for (let j = 0; j < H; j += 1) p[3 * H + j] = M().gaussian(rng) * 0.5;
    return { H, p, m: new Float64Array(p.length), v: new Float64Array(p.length), t: 0 };
  }

  function netLogit(net, x, y, h) {
    const { H, p } = net;
    let z = p[4 * H];
    for (let j = 0; j < H; j += 1) {
      const a = Math.tanh(p[2 * j] * x + p[2 * j + 1] * y + p[2 * H + j]);
      if (h) h[j] = a;
      z += p[3 * H + j] * a;
    }
    return z;
  }

  /* One full-batch step of binary cross-entropy. Targets may be soft
     (any probability), which is all distillation needs. */
  function netStep(net, points, targets, lr) {
    const { H, p } = net;
    const g = new Float64Array(p.length);
    const h = new Float64Array(H);
    const n = points.length;
    for (let i = 0; i < n; i += 1) {
      const { x, y } = points[i];
      const dz = (sigmoid(netLogit(net, x, y, h)) - targets[i]) / n;
      g[4 * H] += dz;
      for (let j = 0; j < H; j += 1) {
        g[3 * H + j] += dz * h[j];
        const dh = dz * p[3 * H + j] * (1 - h[j] * h[j]);
        g[2 * j] += dh * x;
        g[2 * j + 1] += dh * y;
        g[2 * H + j] += dh;
      }
    }
    net.t += 1;
    const b1 = 0.9;
    const b2 = 0.999;
    for (let k = 0; k < p.length; k += 1) {
      net.m[k] = b1 * net.m[k] + (1 - b1) * g[k];
      net.v[k] = b2 * net.v[k] + (1 - b2) * g[k] * g[k];
      const mh = net.m[k] / (1 - b1 ** net.t);
      const vh = net.v[k] / (1 - b2 ** net.t);
      p[k] -= (lr * mh) / (Math.sqrt(vh) + 1e-8);
    }
  }

  function makeDemoData(nLab) {
    const rng = M().seeded(7 + nLab * 13);
    const labelled = Array.from({ length: nLab }, () => {
      const x = rng() * 2 - 1;
      const y = rng() * 2 - 1;
      return { x, y, t: rng() < sigmoid(teacherLogit(x, y)) ? 1 : 0 };
    });
    const rng2 = M().seeded(99);
    const transfer = Array.from({ length: 400 }, () => {
      const x = rng2() * 2 - 1;
      const y = rng2() * 2 - 1;
      return { x, y, p: sigmoid(teacherLogit(x, y)) };
    });
    return { labelled, transfer };
  }

  /* ── Forward vs reverse KL on a 1-D example ──────────────────── */

  const KL_X = Array.from({ length: 481 }, (_, i) => -6 + i * 0.025);
  const KL_DX = 0.025;
  const normal = (x, mu, s) => Math.exp(-0.5 * ((x - mu) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
  const KL_P = KL_X.map((x) => 0.6 * normal(x, -2.2, 0.5) + 0.4 * normal(x, 2.2, 0.6));

  function klPair(mu, sigma) {
    let fwd = 0;
    let rev = 0;
    KL_X.forEach((x, i) => {
      const p = Math.max(KL_P[i], 1e-300);
      const q = Math.max(normal(x, mu, sigma), 1e-300);
      fwd += p * Math.log(p / q) * KL_DX;
      rev += q * Math.log(q / p) * KL_DX;
    });
    return { fwd, rev };
  }

  /* ── Pruning data: a 10×10 layer and correlated inputs ───────── */

  let pruneData = null;
  function getPruneData() {
    if (pruneData) return pruneData;
    const rng = M().seeded(2024);
    const g = () => M().gaussian(rng);
    const n = 10;
    const rowScale = Array.from({ length: n }, () => 0.2 + 1.5 * rng() ** 2);
    const Wm = Array.from({ length: n }, (_, i) => Array.from({ length: n }, () => Math.round(g() * rowScale[i] * 100) / 100));
    /* Real activations are strongly correlated; three hidden factors
       plus a little noise imitate that. */
    const B = Array.from({ length: n }, () => [g(), g(), g()]);
    const X = Array.from({ length: 80 }, () => {
      const z = [g(), g(), g()];
      return B.map((b) => b[0] * z[0] + b[1] * z[1] + b[2] * z[2] + 0.3 * g());
    });
    const order = Array.from({ length: n * n }, (_, k) => k);
    for (let k = order.length - 1; k > 0; k -= 1) {
      const j = Math.floor(rng() * (k + 1));
      [order[k], order[j]] = [order[j], order[k]];
    }
    pruneData = { n, Wm, X, randomOrder: order };
    return pruneData;
  }

  function solve(A, b) {
    const n = b.length;
    const a = A.map((r, i) => [...r, b[i]]);
    for (let c = 0; c < n; c += 1) {
      let best = c;
      for (let r = c + 1; r < n; r += 1) if (Math.abs(a[r][c]) > Math.abs(a[best][c])) best = r;
      [a[c], a[best]] = [a[best], a[c]];
      const piv = a[c][c] || 1e-12;
      for (let r = 0; r < n; r += 1) {
        if (r === c) continue;
        const f = a[r][c] / piv;
        for (let k = c; k <= n; k += 1) a[r][k] -= f * a[c][k];
      }
    }
    return a.map((r, i) => r[n] / (r[i] || 1e-12));
  }

  /* Which weights survive: a boolean n×n mask. */
  function pruneMask(method, sparsity) {
    const { n, Wm, randomOrder } = getPruneData();
    const mask = Wm.map((r) => r.map(() => true));
    if (method === "structured") {
      const drop = Math.round((sparsity / 100) * n);
      Wm.map((r, i) => [Math.hypot(...r), i])
        .sort((a, b) => a[0] - b[0])
        .slice(0, drop)
        .forEach(([, i]) => mask[i].fill(false));
      return mask;
    }
    const k = Math.round((sparsity / 100) * n * n);
    const order = method === "random"
      ? randomOrder
      : Array.from({ length: n * n }, (_, idx) => idx).sort((a, b) => Math.abs(Wm[Math.floor(a / n)][a % n]) - Math.abs(Wm[Math.floor(b / n)][b % n]));
    order.slice(0, k).forEach((idx) => {
      mask[Math.floor(idx / n)][idx % n] = false;
    });
    return mask;
  }

  /* Pruned weights, optionally "healed": each output's surviving
     weights re-fitted by least squares to reproduce the original
     outputs on the sample inputs. */
  function prunedWeights(mask, heal) {
    const { n, Wm, X } = getPruneData();
    return Wm.map((r, i) => {
      const keep = r.map((_, j) => j).filter((j) => mask[i][j]);
      const out = r.map((v, j) => (mask[i][j] ? v : 0));
      if (!heal || !keep.length || keep.length === n) return out;
      const target = X.map((x) => r.reduce((acc, v, j) => acc + v * x[j], 0));
      const A = keep.map((a) => keep.map((b) => X.reduce((acc, x) => acc + x[a] * x[b], 0) + (a === b ? 1e-6 : 0)));
      const rhs = keep.map((a) => X.reduce((acc, x, s) => acc + x[a] * target[s], 0));
      const wS = solve(A, rhs);
      keep.forEach((j, k) => {
        out[j] = wS[k];
      });
      return out;
    });
  }

  function relError(Wp) {
    const { Wm, X } = getPruneData();
    let num = 0;
    let den = 0;
    X.forEach((x) => {
      Wm.forEach((r, i) => {
        const y = r.reduce((acc, v, j) => acc + v * x[j], 0);
        const yp = Wp[i].reduce((acc, v, j) => acc + v * x[j], 0);
        num += (y - yp) ** 2;
        den += y * y;
      });
    });
    return Math.sqrt(num / (den || 1));
  }

  /* ── Quantization ────────────────────────────────────────────── */

  let quantBase = null;
  function quantWeights(outlier) {
    if (!quantBase) {
      const rng = M().seeded(77);
      quantBase = Array.from({ length: 48 }, () => Math.round(M().gaussian(rng) * 0.42 * 1000) / 1000);
    }
    const ws = quantBase.slice();
    if (outlier) ws[29] = 3.1;
    return ws;
  }

  /* Symmetric "absmax" quantization, one scale per group. */
  function quantize(ws, bits, group) {
    const qmax = 2 ** (bits - 1) - 1;
    const scales = [];
    const q = [];
    const deq = [];
    for (let start = 0; start < ws.length; start += group) {
      const chunk = ws.slice(start, start + group);
      const s = Math.max(...chunk.map(Math.abs)) / qmax || 1;
      scales.push(s);
      chunk.forEach((v) => {
        const k = clamp(Math.round(v / s), -qmax, qmax);
        q.push(k);
        deq.push(k * s);
      });
    }
    return { qmax, scales, q, deq };
  }

  /* ════════════════════════════════════════════════════════════════
     THE GUIDE
     ════════════════════════════════════════════════════════════════ */

  function mountSmallLlmGuide(rootNode) {
    const ui = UI();
    const { esc, widget, range, segmented, mountWidget, fmtInt, fmtBig } = W();
    const { para, plain, deeper, takeaway, figure, chapter, guideToc, flow, mathBlock, svgSlot, barList, featureCompare, quiz, fmt } = ui;

    const chapters = [
      { id: "slm-why", title: "Why go small?", blurb: "memory, speed and cost" },
      { id: "slm-scaling", title: "Train small models for longer", blurb: "scaling laws, past Chinchilla" },
      { id: "slm-params", title: "Where a small model's parameters go", blurb: "embeddings, tying, GQA" },
      { id: "slm-soft", title: "Distillation: learning from a teacher", blurb: "soft targets and temperature" },
      { id: "slm-loss", title: "The distillation loss", blurb: "KL, cross-entropy and the T² factor" },
      { id: "slm-demo", title: "Watch a student learn", blurb: "two small networks, trained live" },
      { id: "slm-llm", title: "Distilling an LLM", blurb: "logits, synthetic data, on-policy" },
      { id: "slm-prune", title: "Pruning: cut, then heal", blurb: "unstructured vs structured" },
      { id: "slm-quant", title: "Quantization: fewer bits per weight", blurb: "scales, outliers, groups" },
      { id: "slm-recipes", title: "How real small models are made", blurb: "putting the techniques together" },
      { id: "slm-recap", title: "Recap and self-check", blurb: "" },
    ];

    function body() {
      return `
        <div class="g-intro">
          ${para("The biggest language models have hundreds of billions of parameters and run in data centres. Yet a 1-billion-parameter model on a phone can now summarise an email, and a 3-billion-parameter model can beat the 175-billion-parameter GPT-3 on many benchmarks. This guide shows <strong>how small models get that good</strong>: training them for longer, spending their parameters carefully, learning from a big <strong>teacher</strong> (distillation), cutting a big model down (pruning) and storing each weight in fewer bits (quantization).")}
          ${para("Every widget computes its numbers live. It builds on <a href=\"./algorithm.html?id=llm\">From Transformer to LLM</a>; the cross-entropy and softmax from the <a href=\"./algorithm.html?id=sequence-primer\">Primer</a> are all the maths you need.")}
          ${guideToc(chapters)}
        </div>

        ${chapter("slm-why", 1, "Motivation", "Why go small?", `
          ${plain(para("A model is mostly a huge list of numbers, and those numbers must sit in memory while it runs. At 16 bits (2 bytes) each, a 70-billion-parameter model needs about 140 GB just for its weights: more than any single consumer device has. Worse, writing <em>each</em> token means reading <em>every</em> weight from memory once, so a model's size also sets its speed. Small models fit on laptops and phones, answer faster, cost less per token, and keep private data on the device."))}
          ${widget("will it fit, and how fast can it go?", `
            <label class="g-select"><span>Model size</span><select data-input="size">${SIZES.map((m) => `<option value="${m.id}">${m.label}</option>`).join("")}</select></label>
            ${row("Each weight stored in", segmented("bytes", PRECISIONS, "2"))}
            ${row("Device", segmented("device", DEVICES.map((d) => [d.id, d.label]), "laptop"))}
            <div data-out="fit"></div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`\text{memory} \approx N_{\text{params}} \times \text{bytes per weight},\qquad \text{tokens/s} \lesssim \frac{\text{memory bandwidth}}{\text{memory}}`)}
            ${para("The speed bound comes from decoding being <em>memory-bound</em>: for each new token the processor must stream all the weights in, and the arithmetic is fast by comparison. Real speeds are lower (the KV cache must be read too), but the ratio between models holds. Device figures are typical, rounded values; a quarter of memory is kept free for the system, activations and the KV cache.")}
          `)}
          ${takeaway("Size sets both memory and speed. Halving the bytes per weight, or the number of weights, roughly doubles the speed limit.")}
        `)}

        ${chapter("slm-scaling", 2, "Scaling", "Train small models for longer", `
          ${plain(para("For a fixed training budget, should you train a big model on a little data or a small model on a lot? The <strong>Chinchilla</strong> paper (Hoffmann et al., 2022) fitted a formula for the loss and found the sweet spot at about <strong>20 training tokens per parameter</strong>. But that only counts training. A model that will then answer billions of requests is cheaper overall if it is <em>smaller</em> and trained far past that point, because every request costs compute proportional to its size. That is why today's small models see thousands of tokens per parameter."))}
          ${widget("the Chinchilla loss formula", `
            ${range("cPow", "Training compute (10ˣ FLOPs)", 19, 25, 0.25, 21)}
            ${range("nPow", "Model size (10ˣ parameters)", 7, 11.5, 0.05, 9)}
            <div class="g-two g-two-wide">
              <div>${svgSlot("iso", "0 0 520 300", "loss against model size for a fixed compute budget")}</div>
              <div data-out="scale"></div>
            </div>
            <div class="tl-subhead">Make it smaller, but just as good</div>
            ${row("Shrink the model by", segmented("shrink", [["1.5", "1.5×"], ["2", "2×"], ["4", "4×"]], "2"))}
            <div data-out="shrink"></div>`)}
          ${figure("2.1", "Training tokens per parameter in published models. Small models are trained far beyond the compute-optimal ratio.",
            `<div class="table-panel"><table>
              <thead><tr><th>Model</th><th>Year</th><th>Parameters</th><th>Training tokens</th><th>Tokens per parameter</th></tr></thead>
              <tbody>
                <tr><td>GPT-3</td><td>2020</td><td>175 B</td><td>300 B</td><td>≈ 2</td></tr>
                <tr><td>Chinchilla</td><td>2022</td><td>70 B</td><td>1.4 T</td><td>20</td></tr>
                <tr><td>Llama 2 7B</td><td>2023</td><td>7 B</td><td>2 T</td><td>≈ 290</td></tr>
                <tr><td>Llama 3 8B</td><td>2024</td><td>8 B</td><td>15 T</td><td>≈ 1,900</td></tr>
                <tr><td>SmolLM2 1.7B</td><td>2025</td><td>1.7 B</td><td>11 T</td><td>≈ 6,500</td></tr>
              </tbody></table></div>`, true)}
          ${deeper("The math", `
            ${mathBlock(String.raw`L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},\qquad E = 1.69,\ A = 406.4,\ B = 410.7,\ \alpha = 0.34,\ \beta = 0.28`)}
            ${mathBlock(String.raw`\text{training FLOPs} \approx 6ND,\qquad \text{inference FLOPs per token} \approx 2N`)}
            ${para("N is parameters and D is training tokens. E is the irreducible loss of text itself; the other two terms shrink as the model or the data grows. Because a smaller N leaves a bigger A/N<sup>α</sup> term, a smaller model needs disproportionately more data to reach the same loss, and below some size it can never get there. The widget solves L(N/s, D′) = L(N, D) for D′.")}
          `)}
          ${takeaway("Compute-optimal is ~20 tokens per parameter, but if a model will be used a lot, it pays to make it smaller and train it much longer.")}
        `)}

        ${chapter("slm-params", 3, "Architecture", "Where a small model's parameters go", `
          ${plain(para("In a big model almost all parameters live in the stacked layers. In a small one, the <strong>vocabulary table</strong> becomes a large share: Qwen2.5-0.5B has 151,936 tokens × 896 numbers = 136 million parameters in its embedding table, more than a quarter of the model. So small models <strong>tie</strong> the input embedding and the output layer (one shared matrix instead of two), use <strong>grouped-query attention</strong> to shrink W<sup>K</sup> and W<sup>V</sup>, and often go <strong>deep and thin</strong>: more layers of a narrower width (MobileLLM, 2024)."))}
          ${widget("a small model's parameter budget", `
            ${segmented("preset", Object.entries(ARCH).map(([key, a]) => [key, a.label]), "qwen05")}
            <div class="g-inputs">
              <label><span>layers L</span><input type="number" data-input="L" min="1" step="1" /></label>
              <label><span>width d</span><input type="number" data-input="d" min="8" step="64" /></label>
              <label><span>query heads</span><input type="number" data-input="heads" min="1" step="1" /></label>
              <label><span>K/V heads</span><input type="number" data-input="kv" min="1" step="1" /></label>
              <label><span>head size</span><input type="number" data-input="hd" min="8" step="8" /></label>
              <label><span>FFN width</span><input type="number" data-input="ff" min="8" step="64" /></label>
              <label><span>vocabulary V</span><input type="number" data-input="V" min="100" step="1000" /></label>
            </div>
            ${row("Input and output embeddings", segmented("tied", [["tied", "tied (shared)"], ["untied", "separate"]], "tied"))}
            <div data-out="arch"></div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`\underbrace{L\,d\,d_h\,(2h + 2h_{kv})}_{\text{attention}} + \underbrace{3\,L\,d\,d_{ff}}_{\text{SwiGLU FFN}} + \underbrace{V d\,(1\text{ or }2)}_{\text{embeddings}}`)}
            ${para("h is the number of query heads, h<sub>kv</sub> the number of key/value heads and d<sub>h</sub> the head size. SwiGLU uses three matrices, hence the 3. Norms and biases add well under 1%. With the presets, this reproduces each model's published size.")}
          `)}
          ${takeaway("In small models the vocabulary is expensive: tie the embeddings, share keys and values, and prefer depth over width.")}
        `)}

        ${chapter("slm-soft", 4, "Distillation", "Distillation: learning from a teacher", `
          ${plain(para("<strong>Knowledge distillation</strong> (Hinton, Vinyals &amp; Dean, 2015) trains a small <strong>student</strong> to imitate a big <strong>teacher</strong>. The trick is in what it imitates. A normal training label is <em>hard</em>: “this photo is a cat”, full stop. The teacher's output is a whole probability distribution: “cat, a bit like a tiger, slightly like a dog, nothing like a car”. Those small probabilities, which Hinton called <em>dark knowledge</em>, tell the student how the classes relate. Dividing the teacher's scores by a <strong>temperature</strong> T &gt; 1 before the softmax makes them visible."))}
          ${widget("soften the teacher's answer for a photo of a cat", `
            <div class="g-inputs">${CLASSES.map((c, i) => `<label><span>score for “${c}”</span><input type="number" data-input="z${i}" step="0.5" /></label>`).join("")}</div>
            ${range("T", "Temperature T", 1, 10, 0.5, 1)}
            <div data-out="soft"></div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`p_i^{(T)} = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}},\qquad \frac{p_i^{(T)}}{p_j^{(T)}} = e^{(z_i - z_j)/T}`)}
            ${para("The ratio between two classes depends only on their score difference divided by T. At T = 1 a gap of 7 makes one class about 1,100× likelier; at T = 4 only about 6×. The ranking never changes, only how loudly the small differences are spoken.")}
          `)}
          ${takeaway("A teacher's soft probabilities carry far more information per example than a hard label; temperature turns up the volume on the small ones.")}
        `)}

        ${chapter("slm-loss", 5, "Distillation", "The distillation loss", `
          ${plain(para("The student is trained to make its own softened distribution match the teacher's, measured by the <strong>KL divergence</strong> (how many extra nats it costs to describe the teacher's answer using the student's). Usually this is mixed with ordinary cross-entropy on the true label, weighted by α. Below, the student's six scores start at zero; each gradient step moves them the way the loss says. The teacher's scores come from the widget above, so change them there and see the student chase a new target."))}
          ${widget("train a student's scores by hand", `
            ${range("T", "Temperature T", 1, 10, 0.5, 4)}
            ${range("alpha", "Weight on the hard label α", 0, 1, 0.05, 0.1)}
            ${range("lr", "Learning rate", 0.05, 2, 0.05, 0.5)}
            <div class="g-actions">
              <button type="button" class="button primary" data-action="step1">1 gradient step</button>
              <button type="button" class="button secondary" data-action="step25">25 steps</button>
              <button type="button" class="button secondary" data-action="reset">Reset student</button>
            </div>
            <div data-out="loss"></div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`\mathcal{L} = \alpha\,\underbrace{\big(-\log p^{s}_{y}\big)}_{\text{hard label}} + (1-\alpha)\,T^{2}\,\underbrace{\operatorname{KL}\!\big(p^{t,(T)}\,\big\|\,p^{s,(T)}\big)}_{\text{match the teacher}}`)}
            ${mathBlock(String.raw`\frac{\partial \mathcal{L}}{\partial z^{s}_i} = \alpha\,\big(p^{s}_i - \mathbb{1}[i = y]\big) + (1-\alpha)\,T\,\big(p^{s,(T)}_i - p^{t,(T)}_i\big)`)}
            ${para("Softening by T shrinks the soft term's gradients by about 1/T², so the loss multiplies it by T² to keep both terms in balance whatever T you pick. At very high T, matching soft targets becomes the same as matching the teacher's raw scores with squared error.")}
          `)}
          ${takeaway("Distillation loss = KL to the teacher's softened distribution (×T²), optionally plus cross-entropy on the true label.")}
        `)}

        ${chapter("slm-demo", 6, "Experiment", "Watch a student learn", `
          ${plain(para("Here distillation runs for real. The <strong>teacher</strong> is a fixed “big model” that splits the plane into two classes with a wavy border and a small island. Two identical small networks start from the same random weights. <strong>Student A</strong> learns the usual way, from a handful of human-labelled points (sampled, so a few labels near the border are noisy). <strong>Student B</strong> learns from the teacher's answers on 400 unlabelled points. Press Train and watch."))}
          ${widget("two students, trained in your browser", `
            ${row("Student size", segmented("H", [["3", "3 hidden units"], ["8", "8"], ["16", "16"]], "8"))}
            ${row("Human-labelled points (A)", segmented("nLab", [["10", "10"], ["20", "20"], ["60", "60"]], "20"))}
            ${row("Student B learns from", segmented("target", [["soft", "teacher's probabilities"], ["hard", "teacher's hard labels"]], "soft"))}
            <div class="g-actions">
              <button type="button" class="button primary" data-action="train">Train 600 steps</button>
              <button type="button" class="button secondary" data-action="reset">Reset</button>
            </div>
            <div class="g-three">
              <div><div class="tl-subhead">Teacher</div>${svgSlot("dTeacher", "0 0 240 240", "teacher decision map")}<p class="caption" data-out="dT"></p></div>
              <div><div class="tl-subhead">Student A · human labels</div>${svgSlot("dA", "0 0 240 240", "student trained on labels")}<p class="caption" data-out="dA"></p></div>
              <div><div class="tl-subhead">Student B · distilled</div>${svgSlot("dB", "0 0 240 240", "student trained on teacher outputs")}<p class="caption" data-out="dB"></p></div>
            </div>`)}
          <ul class="g-list">
            <li><strong>Few labels.</strong> With 10 or 20 points, student A guesses a border that fits its points and misses the island; student B, fed by the teacher, finds the island.</li>
            <li><strong>Capacity gap.</strong> With 3 hidden units even the distilled student cannot draw the island: a student can only copy what it is big enough to represent.</li>
            <li><strong>Soft vs hard.</strong> Switch B to the teacher's hard labels. It still beats A, because the teacher can label as much data as you like, but its map loses the teacher's smooth confidence near the border.</li>
          </ul>
          ${takeaway("A teacher turns unlabelled data into training signal, and its soft answers carry its uncertainty; the student's own size caps how much it can absorb.")}
        `)}

        ${chapter("slm-llm", 7, "LLMs", "Distilling an LLM: three recipes", `
          ${plain(para("For a language model, the “classes” are the next token, so the teacher gives a probability for every token in the vocabulary at every position. There are three main ways to pass that knowledge on."))}
          ${figure("7.1", "Three ways to distil a language model.",
            `<div class="g-cards g-cards-3">
              <div class="soft-box"><h4>Logit distillation</h4><p class="caption">Word-level KD. DistilBERT, Gemma 2 2B/9B, Llama 3.2 1B/3B.</p>${flow(["text", "→", "teacher + student", "→", "match every token's distribution"])}<p>The richest signal: the full distribution at every position. Needs the teacher's raw outputs and the same tokenizer for both.</p></div>
              <div class="soft-box"><h4>Synthetic data</h4><p class="caption">Sequence-level KD. Alpaca, Phi, DeepSeek-R1-Distill.</p>${flow(["prompts", "→", "teacher writes answers", "→", "student fine-tunes"])}<p>The teacher writes text and the student trains on it with the ordinary loss. Works even through an API that returns only text.</p></div>
              <div class="soft-box"><h4>On-policy distillation</h4><p class="caption">GKD, MiniLLM.</p>${flow(["student writes", "→", "teacher scores each token", "→", "student updates"])}<p>Trains on the student's own attempts, so it learns to recover from its own mistakes, not only to copy perfect text.</p></div>
            </div>`, true)}
          ${para("Which way the KL divergence points matters. Ordinary distillation uses <strong>forward KL</strong>, KL(teacher ‖ student): the student is punished wherever the teacher has probability and it has none, so it spreads out to <em>cover</em> everything the teacher might say. On-policy methods use <strong>reverse KL</strong>, KL(student ‖ teacher): the student is punished for saying anything the teacher wouldn't, so it <em>picks one</em> safe mode. A small student can't represent everything a big teacher can, so the choice decides how it fails.")}
          ${widget("fit a one-peak student to a two-peak teacher", `
            ${row("Divergence", segmented("mode", [["forward", "forward KL (mode-covering)"], ["reverse", "reverse KL (mode-seeking)"]], "forward"))}
            ${range("mu", "Student mean μ", -4, 4, 0.05, 0.5)}
            ${range("sigma", "Student spread σ", 0.2, 3, 0.05, 0.6)}
            <div class="g-actions">
              <button type="button" class="button primary" data-action="fit">Fit the student</button>
              <button type="button" class="button secondary" data-action="reset">Reset</button>
            </div>
            ${svgSlot("kl", "0 0 560 260", "teacher and student distributions")}
            <div data-out="kl"></div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`\operatorname{KL}(p\,\|\,q) = \sum_x p(x)\log\frac{p(x)}{q(x)}\qquad\text{forward: } p = \text{teacher},\ q = \text{student}`)}
            ${para("Forward KL blows up where p &gt; 0 but q ≈ 0, so q must cover all of p (it ends up matching p's mean and variance). Reverse KL blows up where q &gt; 0 but p ≈ 0, so q avoids the gap between peaks. For text, a mode-covering student may blend two plausible answers into an implausible one; a mode-seeking student gives one confident answer but less variety.")}
          `)}
          ${takeaway("LLM distillation matches token distributions (logits), trains on teacher-written text, or scores the student's own text; forward KL covers, reverse KL commits.")}
        `)}

        ${chapter("slm-prune", 8, "Compression", "Pruning: cut, then heal", `
          ${plain(para("<strong>Pruning</strong> removes the parts of a trained network that matter least. <strong>Unstructured</strong> pruning zeroes individual small weights; it keeps accuracy best, but the matrix keeps its shape, so it only gets faster with special sparse hardware or kernels. <strong>Structured</strong> pruning removes whole neurons, attention heads or layers, giving a genuinely smaller dense model that is faster everywhere. Either way the model gets worse, so it is <strong>healed</strong> with more training, ideally by distilling from the original model."))}
          ${widget("prune one layer of 100 weights", `
            ${range("sparsity", "Weights removed (%)", 0, 90, 10, 50)}
            ${row("Which ones", segmented("method", [["magnitude", "smallest |w| (unstructured)"], ["structured", "whole neurons (structured)"], ["random", "random"]], "magnitude"))}
            ${row("After pruning", segmented("heal", [["no", "leave the rest"], ["yes", "heal: re-fit the rest"]], "no"))}
            <div data-out="prune"></div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`\text{keep } w_{ij} \text{ if } |w_{ij}| \text{ is large};\qquad \text{heal: } w_S = \big(X_S^{\top}X_S\big)^{-1}X_S^{\top}\,(X w)`)}
            ${para("For each output neuron, S is the set of inputs it kept and X holds sample inputs. The heal re-fits the surviving weights so the neuron's output on real inputs stays as close as possible to the original. Because inputs are correlated, surviving weights can take over the job of removed ones. Methods like SparseGPT apply this idea layer by layer to billion-parameter models; a removed neuron, however, can only be compensated by later layers, which is what distillation-based healing trains.")}
          `)}
          ${takeaway("Unstructured pruning keeps accuracy, structured pruning gives real speed; both need healing, and distillation from the original is the best healer.")}
        `)}

        ${chapter("slm-quant", 9, "Compression", "Quantization: fewer bits per weight", `
          ${plain(para("<strong>Quantization</strong> stores each weight as a small integer instead of a 16-bit number. With 4 bits there are only 15 levels (−7 … 7), so each group of weights gets a <strong>scale</strong>: the largest weight maps to 7 and everything else is rounded to the nearest level. The enemy is the <strong>outlier</strong>: one huge weight stretches the scale so far that all the ordinary weights round to 0. Giving every small group its own scale contains the damage, which is what practical methods (GPTQ, AWQ, llama.cpp's formats, QLoRA's NF4) do."))}
          ${widget("quantize 48 weights", `
            ${range("bits", "Bits per weight", 2, 8, 1, 4)}
            ${row("Scales", segmented("scheme", [["tensor", "one for all 48"], ["group", "one per group of 16"]], "tensor"))}
            ${row("Outlier", segmented("outlier", [["no", "none"], ["yes", "one weight = 3.1"]], "no"))}
            ${svgSlot("quant", "0 0 560 260", "original and quantized weights")}
            <div data-out="quant"></div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`s = \frac{\max_i |w_i|}{2^{b-1}-1},\qquad q_i = \operatorname{round}\!\Big(\frac{w_i}{s}\Big),\qquad \hat w_i = q_i\,s`)}
            ${para("Each group stores b-bit integers plus one 16-bit scale, so a group of g weights costs b + 16/g bits per weight. Real formats use groups of 32–128. <em>Post-training</em> quantization (PTQ) rounds a finished model, with clever corrections in GPTQ and AWQ; <em>quantization-aware training</em> (QAT) simulates the rounding during training so the model learns to tolerate it.")}
          `)}
          ${takeaway("4-bit quantization cuts memory 4× versus 16-bit with small losses, as long as outliers are handled with small groups or special care.")}
        `)}

        ${chapter("slm-recipes", 10, "In practice", "How real small models are made", `
          ${plain(para("Production small models stack these techniques. A typical modern pipeline:"))}
          ${figure("10.1", "A common recipe for a strong small model.",
            flow(["<span>Big teacher</span><small>trained first</small>", "→", "<span>Prune</span><small>width / depth</small>", "→", "<span>Distil</span><small>teacher logits, trillions of tokens</small>", "→", "<span>Fine-tune</span><small>teacher-written chats</small>", "→", "<span>Quantize</span><small>4–8 bit</small>", "→", "<span>Ship</span><small>laptop, phone</small>"]))}
          ${figure("10.2", "Published recipes. Most strong small models use at least one of the techniques in this guide.",
            `<div class="table-panel"><table>
              <thead><tr><th>Model</th><th>Year</th><th>How it was made small and good</th></tr></thead>
              <tbody>
                <tr><td>DistilBERT</td><td>2019</td><td>Half of BERT-base's layers, initialised from the teacher, trained with a distillation loss: 40% smaller, 60% faster, about 97% of BERT's language-understanding score.</td></tr>
                <tr><td>Alpaca 7B</td><td>2023</td><td>LLaMA 7B fine-tuned on 52,000 instructions and answers written by a larger OpenAI model: sequence-level distillation.</td></tr>
                <tr><td>Phi-1 / Phi-3</td><td>2023–24</td><td>Small models trained on heavily filtered “textbook-quality” web data plus synthetic text written by larger models.</td></tr>
                <tr><td>Sheared-LLaMA</td><td>2023</td><td>Llama 2 7B structurally pruned to 1.3 B and 2.7 B, then trained further to recover.</td></tr>
                <tr><td>Gemma 2 2B / 9B</td><td>2024</td><td>Pretrained with logit distillation from a larger model instead of plain next-token prediction.</td></tr>
                <tr><td>Minitron 4B / 8B</td><td>2024</td><td>Nemotron-4 15B pruned in width and depth, healed with distillation, using up to 40× fewer training tokens than training from scratch.</td></tr>
                <tr><td>Llama 3.2 1B / 3B</td><td>2024</td><td>Pruned from Llama 3.1 8B, with logits from Llama 3.1 8B and 70B used as targets during pretraining.</td></tr>
                <tr><td>DeepSeek-R1-Distill</td><td>2025</td><td>Qwen2.5 and Llama models (1.5 B–70 B) fine-tuned on about 800,000 reasoning samples generated with DeepSeek-R1.</td></tr>
              </tbody></table></div>`, true)}
          <h3 class="g-sub">Two more ways small models earn their place</h3>
          <ul class="g-list">
            <li><strong>Speculative decoding.</strong> A small <em>draft</em> model guesses the next few tokens; the big model checks them all in one parallel pass and keeps the ones it agrees with. With the right acceptance rule the output is exactly what the big model would have written, just faster (Leviathan et al., 2023; Chen et al., 2023).</li>
            <li><strong>Cheap fine-tuning with LoRA.</strong> Instead of updating all weights, train two thin matrices whose product is added to each weight matrix. QLoRA does this on top of a 4-bit model, so a small model can be specialised on a single consumer GPU.</li>
          </ul>
          ${takeaway("Strong small models = good data, long training, a teacher to learn from, careful pruning, and quantization at the end.")}
        `)}

        ${chapter("slm-recap", 11, "Recap", "Recap and self-check", `
          <ol class="g-recap">
            <li>A model's <b>size sets its memory and its speed</b>: bytes ≈ parameters × bytes per weight, and tokens/s ≲ bandwidth ÷ bytes.</li>
            <li>Small models are trained <b>far past 20 tokens per parameter</b>, because inference cost grows with size.</li>
            <li>In small models the <b>vocabulary is expensive</b>: tie embeddings, use grouped-query attention, go deep and thin.</li>
            <li><b>Distillation</b> trains a student on a teacher's <b>softened probabilities</b> with loss T²·KL, often plus cross-entropy on true labels.</li>
            <li>For LLMs: <b>logit</b> distillation, <b>synthetic data</b>, or <b>on-policy</b> with reverse KL.</li>
            <li><b>Pruning</b> removes weights or whole structures and needs <b>healing</b>; <b>quantization</b> stores weights in 4–8 bits with per-group scales.</li>
          </ol>
          ${quiz([
            { q: "Roughly how much memory do the weights of an 8-billion-parameter model need at 4 bits, and could it run on a 16 GB laptop?", a: "8 × 10⁹ × 0.5 bytes ≈ 4 GB (a little more with the scales). Yes, comfortably, with room left for the KV cache." },
            { q: "Why would you train a 1 B model on 10 T tokens when Chinchilla says ~20 B tokens is compute-optimal?", a: "Chinchilla minimises training compute only. If the model will generate trillions of tokens in use, a smaller model trained longer reaches similar quality and every generated token is cheaper." },
            { q: "What does a teacher's output at T = 4 tell a student that the hard label “cat” doesn't?", a: "How similar the other classes are: for example that a cat looks much more like a tiger or a dog than like a car. That relational information is the “dark knowledge”." },
            { q: "Why is the soft part of the distillation loss multiplied by T²?", a: "Softening by T shrinks the soft term's gradients by about 1/T²; multiplying by T² keeps its influence comparable to the hard-label term for any temperature." },
            { q: "A small student keeps blending two different correct answers into one muddled answer. Which divergence is it probably trained with, and what could help?", a: "Forward KL, which is mode-covering. Reverse KL (as in on-policy distillation) makes the student commit to one mode." },
            { q: "Why does one large outlier hurt per-tensor 4-bit quantization so much?", a: "The scale is set by the largest absolute weight, so all ordinary weights fall within a tiny fraction of the 15 levels and many round to zero. Per-group scales limit the damage to the outlier's own group." },
          ])}
          <div class="tl-next">
            <div><div class="eyebrow">Previous guide</div><strong>From the Transformer to today's LLMs</strong><p class="caption">Decoder-only models, sampling, RoPE, the KV cache, mixture of experts and instruction tuning.</p></div>
            <a class="button secondary" href="./algorithm.html?id=llm">Back to LLMs</a>
          </div>
          <div class="tl-next">
            <div><div class="eyebrow">Back to the start</div><strong>The whole series</strong><p class="caption">Primer → Attention → Transformer → LLMs → Small LLMs.</p></div>
            <a class="button primary" href="./algorithm.html?id=sequence-primer">Go to the Primer</a>
          </div>
        `)}
      `;
    }

    const state = {
      fit: { size: "1b", bytes: "2", device: "laptop" },
      scale: { cPow: 21, nPow: 9, shrink: "2" },
      arch: Object.assign({ preset: "qwen05" }, ARCH.qwen05),
      soft: { T: 1, z0: 9, z1: 6.5, z2: 5.5, z3: 4, z4: -1, z5: -3 },
      loss: { T: 4, alpha: 0.1, lr: 0.5, zs: [0, 0, 0, 0, 0, 0], steps: 0, history: [] },
      demo: { H: "8", nLab: "20", target: "soft", sig: "", nets: null, data: null },
      kl: { mode: "forward", mu: 0.5, sigma: 0.6 },
      prune: { sparsity: 50, method: "magnitude", heal: "no" },
      quant: { bits: 4, scheme: "tensor", outlier: "no" },
    };

    let demoRun = null;
    let klRun = null;
    let renderLoss = null;

    const hostOf = (name) => rootNode.querySelector(`[data-out="${name}"]`).closest(".g-widget");

    /* ── 1. Fit and speed ── */
    function mountFit() {
      const host = hostOf("fit");
      restoreInputs(host, state.fit);
      mountWidget(host, state.fit, (h, s) => {
        const model = SIZES.find((m) => m.id === s.size);
        const bytes = Number(s.bytes);
        const need = (model.P * bytes) / 1e9;
        const device = DEVICES.find((d) => d.id === s.device);
        const usable = device.mem * USABLE;
        const fits = need <= usable;
        const precisionLabel = PRECISIONS.find(([b]) => b === s.bytes)[1];
        const sizes = PRECISIONS.map(([b]) => (model.P * Number(b)) / 1e9);
        h.querySelector('[data-out="fit"]').innerHTML = `
          <div class="g-metrics">
            <div><span>weights</span><b>${need < 10 ? need.toFixed(2) : fmtInt(need)} GB</b></div>
            <div><span>on a ${device.label.toLowerCase()}</span><b>${fits ? "fits ✓" : "too big ✗"}</b></div>
            <div><span>speed limit</span><b>${fits ? `≈ ${fmtInt(device.bw / need)} tokens/s` : "—"}</b></div>
            <div><span>largest that fits here</span><b>${fmtBig((usable * 1e9) / bytes)} params</b></div>
          </div>
          <div class="g-two">
            <div>
              <div class="tl-subhead">This model at each precision (GB)</div>
              ${barList(sizes, PRECISIONS.map(([, label]) => label), { digits: sizes[0] < 10 ? 2 : 0, highlight: PRECISIONS.findIndex(([b]) => b === s.bytes), color: C().a })}
            </div>
            <div>
              <div class="tl-subhead">At ${precisionLabel}, on each device</div>
              <table class="g-mini"><tr><th>device</th><th>memory</th><th>fits?</th><th>≈ tokens/s</th></tr>${DEVICES.map((d) => {
                const ok = need <= d.mem * USABLE;
                return `<tr><td>${d.label}</td><td class="mono">${d.mem} GB</td><td>${ok ? "✓" : "✗"}</td><td class="mono">${ok ? fmtInt(d.bw / need) : "—"}</td></tr>`;
              }).join("")}</table>
            </div>
          </div>
          <p class="caption">Device memory and bandwidth (${DEVICES.map((d) => `${d.label.toLowerCase()} ≈ ${fmtInt(d.bw)} GB/s`).join(", ")}) are typical round figures. Speeds are upper bounds for generating one reply.</p>`;
      });
    }

    /* ── 2. Scaling law ── */
    function mountScale() {
      const host = hostOf("scale");
      restoreInputs(host, state.scale);
      mountWidget(host, state.scale, (h, s) => {
        const Cf = 10 ** s.cPow;
        const N = 10 ** s.nPow;
        const D = Cf / (6 * N);
        const L = lossND(N, D);
        const grid = [];
        for (let x = 7; x <= 11.5 + 1e-9; x += 0.02) grid.push([x, lossND(10 ** x, Cf / (6 * 10 ** x))]);
        const best = grid.reduce((a, b) => (b[1] < a[1] ? b : a));
        const yLo = Math.floor((best[1] - 0.08) * 10) / 10;
        const yHi = yLo + 1.2;

        const svg = h.querySelector('[data-fig="iso"]');
        const f = frame(svg, {
          xDomain: [7, 11.5],
          yDomain: [yLo, yHi],
          xTicks: [[7, "10M"], [8, "100M"], [9, "1B"], [10, "10B"], [11, "100B"]],
          yTicks: niceTicks(yLo, yHi, 5).map((v) => [v, v.toFixed(1)]),
          title: `Loss for a budget of 10^${s.cPow} FLOPs`,
          xLabel: "model size (parameters)",
          clipId: "slm-iso-clip",
        });
        f.el("path", { d: linePath(grid, f.xs, f.ys), fill: "none", stroke: C().c, "stroke-width": 2.6, "clip-path": f.clip });
        f.el("circle", { cx: f.xs(best[0]), cy: f.ys(best[1]), r: 6, fill: C().a });
        ui.svgText(svg, f.xs(best[0]), f.ys(best[1]) + 20, "compute-optimal", { "text-anchor": "middle", fill: C().a });
        if (L <= yHi) {
          f.el("circle", { cx: f.xs(s.nPow), cy: f.ys(L), r: 6, fill: C().b, stroke: C().ring, "stroke-width": 2 });
          ui.svgText(svg, f.xs(s.nPow), f.ys(L) - 12, "your model", { "text-anchor": "middle", fill: C().b });
        }
        const Nb = 10 ** best[0];
        h.querySelector('[data-out="scale"]').innerHTML = `
          <div class="g-metrics">
            <div><span>your model</span><b>${fmtBig(N)}</b></div>
            <div><span>training tokens</span><b>${fmtBig(D)}</b></div>
            <div><span>tokens / param</span><b>${fmtInt(D / N)}</b></div>
            <div><span>predicted loss</span><b>${L.toFixed(3)}</b></div>
          </div>
          <p class="caption">Same budget, best split: <b>${fmtBig(Nb)}</b> parameters on <b>${fmtBig(Cf / (6 * Nb))}</b> tokens (${fmtInt(Cf / (6 * Nb) / Nb)} per parameter), loss ${best[1].toFixed(3)}. Too small and it can't hold the knowledge; too big and it sees too little data.</p>`;

        const shrink = Number(s.shrink);
        const Ns = N / shrink;
        const rest = L - CH.E - CH.A / Ns ** CH.a;
        let html;
        if (rest <= 0) {
          html = `<p class="caption"><b>Impossible.</b> A model of ${fmtBig(Ns)} parameters can never reach loss ${L.toFixed(3)}, however much data it sees: its A/N<sup>α</sup> term alone is too large. Pick a bigger model or a smaller shrink.</p>`;
        } else {
          const Ds = (CH.B / rest) ** (1 / CH.b);
          const trainRatio = (Ns * Ds) / (N * D);
          const extra = 6 * Ns * Ds - 6 * N * D;
          const save = 2 * N - 2 * Ns;
          html = `
            <div class="g-metrics">
              <div><span>smaller model</span><b>${fmtBig(Ns)}</b></div>
              <div><span>tokens needed</span><b>${fmtBig(Ds)} (×${(Ds / D).toFixed(1)})</b></div>
              <div><span>training compute</span><b>×${trainRatio.toFixed(2)}</b></div>
              <div><span>cost per generated token</span><b>÷${shrink}</b></div>
            </div>
            <p class="caption">${extra > 0 ? `The extra training pays for itself after about <b>${fmtBig(extra / save)} generated tokens</b>: a popular model serves that quickly.` : "It even costs less to train: your model was bigger than compute-optimal."}</p>`;
        }
        h.querySelector('[data-out="shrink"]').innerHTML = html;
      }, {
        formatOutput: (key, value) => (key === "nPow" ? fmtBig(10 ** value) : key === "cPow" ? `10^${value}` : value),
      });
    }

    /* ── 3. Parameter budget ── */
    function mountArch() {
      const host = hostOf("arch");
      restoreInputs(host, state.arch);
      const sync = mountWidget(host, state.arch, (h, s) => {
        const v = Object.fromEntries(ARCH_KEYS.map((k) => [k, Math.max(1, Number(s[k]) || 1)]));
        const attn = v.L * v.d * v.hd * (2 * v.heads + 2 * v.kv);
        const ffn = 3 * v.L * v.d * v.ff;
        const emb = v.V * v.d * (s.tied === "tied" ? 1 : 2);
        const total = attn + ffn + emb;
        const preset = ARCH[s.preset];
        const matches = preset && ARCH_KEYS.every((k) => preset[k] === v[k]) && preset.tied === s.tied;
        h.querySelector('[data-out="arch"]').innerHTML = `
          ${barList([attn, ffn, emb].map((x) => x / 1e6), ["attention", "feed-forward", `embeddings (${s.tied === "tied" ? "shared" : "in + out"})`], { digits: 1, color: C().a })}
          <div class="g-metrics">
            <div><span>estimated total</span><b>${fmtBig(total)}</b></div>
            ${matches ? `<div><span>published size</span><b>${preset.real}</b></div>` : ""}
            <div><span>embedding share</span><b>${((emb / total) * 100).toFixed(0)}%</b></div>
            <div><span>K/V sharing</span><b>${v.heads > v.kv ? `${(v.heads / v.kv).toFixed(v.heads % v.kv ? 1 : 0)} query heads per K/V head` : "none (multi-head)"}</b></div>
          </div>
          <p class="caption">Millions of parameters. ${s.tied === "tied" ? `Untying would add another ${fmtBig(v.V * v.d)}.` : `Tying would save ${fmtBig(v.V * v.d)} (${((v.V * v.d / total) * 100).toFixed(0)}%).`} Compare Llama 3.1 8B, where the embeddings are a much smaller share.</p>`;
      });
      host.addEventListener("click", (event) => {
        const seg = event.target.closest('[data-seg="preset"]');
        if (!seg) return;
        Object.assign(state.arch, ARCH[seg.getAttribute("data-value")]);
        restoreInputs(host, state.arch);
        if (sync) sync();
      });
    }

    /* ── 4. Soft targets ── */
    const teacherZ = () => [0, 1, 2, 3, 4, 5].map((i) => Number(state.soft[`z${i}`]) || 0);

    function mountSoft() {
      const host = hostOf("soft");
      restoreInputs(host, state.soft);
      mountWidget(host, state.soft, (h, s) => {
        const z = teacherZ();
        const p1 = M().softmax(z);
        const pT = M().softmax(z.map((v) => v / s.T));
        const hard = CLASSES.map((_, i) => (i === 0 ? 1 : 0));
        const odds = (p) => {
          const r = p[1] / Math.max(p[4], 1e-300);
          return r < 100 ? r.toFixed(1) : fmtBig(r);
        };
        h.querySelector('[data-out="soft"]').innerHTML = `
          <div class="g-three">
            <div><div class="tl-subhead">Hard label</div>${barList(hard, CLASSES, { max: 1, asPercent: true, color: C().neutral })}</div>
            <div><div class="tl-subhead">Teacher at T = 1</div>${barList(p1, CLASSES, { max: 1, asPercent: true, color: C().c })}</div>
            <div><div class="tl-subhead">Teacher at T = ${s.T}</div>${barList(pT, CLASSES, { max: 1, asPercent: true, color: C().a })}</div>
          </div>
          <div class="g-metrics">
            <div><span>information (entropy)</span><b>0 → ${entropyBits(p1).toFixed(2)} → ${entropyBits(pT).toFixed(2)} bits</b></div>
            <div><span>tiger : car at T = 1</span><b>${odds(p1)} : 1</b></div>
            <div><span>tiger : car at T = ${s.T}</span><b>${odds(pT)} : 1</b></div>
          </div>
          <p class="caption">Percentages are rounded, so tiny probabilities show as 0% even when they carry information. Raise T and watch them surface.</p>`;
        if (renderLoss) renderLoss();
      }, { formatOutput: (key, value) => Number(value).toFixed(1) });
    }

    /* ── 5. Distillation loss ── */
    function lossParts(s) {
      const zt = teacherZ();
      const ps = M().softmax(s.zs);
      const psT = M().softmax(s.zs.map((v) => v / s.T));
      const ptT = M().softmax(zt.map((v) => v / s.T));
      const ce = -Math.log(Math.max(ps[0], 1e-300));
      const kl = klDiv(ptT, psT);
      const total = s.alpha * ce + (1 - s.alpha) * s.T * s.T * kl;
      const grad = ps.map((p, i) => s.alpha * (p - (i === 0 ? 1 : 0)) + (1 - s.alpha) * s.T * (psT[i] - ptT[i]));
      return { ps, psT, ptT, ce, kl, total, grad };
    }

    function lossSteps(s, count) {
      for (let k = 0; k < count; k += 1) {
        const { total, grad } = lossParts(s);
        if (s.history.length === 0) s.history.push(total);
        s.zs = s.zs.map((v, i) => v - s.lr * grad[i]);
        s.steps += 1;
        s.history.push(lossParts(s).total);
      }
      if (s.history.length > 400) s.history = s.history.slice(-400);
    }

    function mountLoss() {
      const host = hostOf("loss");
      restoreInputs(host, state.loss);
      const draw = (h, s) => {
        const parts = lossParts(s);
        const maxG = Math.max(0.25, ...parts.grad.map(Math.abs));
        const top = parts.ps.indexOf(Math.max(...parts.ps));
        h.querySelector('[data-out="loss"]').innerHTML = `
          <div class="g-two">
            <div><div class="tl-subhead">Teacher, softened at T = ${s.T}</div>${barList(parts.ptT, CLASSES, { max: 1, asPercent: true, color: C().a })}</div>
            <div><div class="tl-subhead">Student, softened at T = ${s.T}</div>${barList(parts.psT, CLASSES, { max: 1, asPercent: true, color: C().b })}</div>
          </div>
          <div class="g-metrics">
            <div><span>hard-label loss</span><b>${fmt(parts.ce, 3)}</b></div>
            <div><span>KL to teacher</span><b>${fmt(parts.kl, 4)}</b></div>
            <div><span>total loss</span><b>${fmt(parts.total, 3)}</b></div>
            <div><span>student's answer</span><b>${CLASSES[top]} (${Math.round(parts.ps[top] * 100)}%)</b></div>
            <div><span>steps</span><b>${s.steps}</b></div>
          </div>
          <div class="tl-subhead">Gradient for each score (the next step moves the opposite way)</div>
          ${featureCompare(CLASSES, [{ name: "∂L/∂z", values: parts.grad, color: C().c }], maxG)}
          ${s.history.length > 1 ? `<div class="tl-subhead">Total loss over the steps so far</div>${sparkline(s.history, C().b)}` : ""}`;
      };
      mountWidget(host, state.loss, draw, {
        formatOutput: (key, value) => Number(value).toFixed(2),
        step1: (s) => lossSteps(s, 1),
        step25: (s) => lossSteps(s, 25),
        reset: (s) => {
          s.zs = [0, 0, 0, 0, 0, 0];
          s.steps = 0;
          s.history = [];
        },
      });
      renderLoss = () => draw(host, state.loss);
    }

    /* ── 6. Live distillation demo ── */
    const R = 30;
    let demoCells = null;
    let demoPointsSig = null;

    function cellColor(p) {
      const strength = Math.abs(p - 0.5) * 2;
      return ui.rgba(p >= 0.5 ? C().a : C().b, 0.08 + 0.7 * strength);
    }

    function buildDemo(host) {
      demoCells = {};
      demoPointsSig = null;
      ["dTeacher", "dA", "dB"].forEach((name) => {
        const svg = host.querySelector(`[data-fig="${name}"]`);
        U().clear(svg);
        const size = 240 / R;
        const cells = [];
        for (let j = 0; j < R; j += 1) {
          for (let i = 0; i < R; i += 1) {
            const rect = U().svgEl("rect", { x: i * size, y: j * size, width: size, height: size, "shape-rendering": "crispEdges" });
            svg.appendChild(rect);
            cells.push({ rect, x: -1 + ((i + 0.5) * 2) / R, y: 1 - ((j + 0.5) * 2) / R });
          }
        }
        const layer = U().svgEl("g", {});
        svg.appendChild(layer);
        demoCells[name] = { svg, cells, layer };
      });
      demoCells.dTeacher.cells.forEach((c) => c.rect.setAttribute("fill", cellColor(sigmoid(teacherLogit(c.x, c.y)))));
    }

    const toPx = (v) => ((v + 1) / 2) * 240;

    function paintDemo(host, s) {
      if (!demoCells) return;
      const { nets, data } = s;
      const agree = { dA: 0, dB: 0 };
      const gap = { dA: 0, dB: 0 };
      [["dA", nets.A], ["dB", nets.B]].forEach(([name, net]) => {
        demoCells[name].cells.forEach((c) => {
          const p = sigmoid(netLogit(net, c.x, c.y));
          const pt = sigmoid(teacherLogit(c.x, c.y));
          c.rect.setAttribute("fill", cellColor(p));
          if ((p >= 0.5) === (pt >= 0.5)) agree[name] += 1;
          gap[name] += Math.abs(p - pt);
        });
      });
      if (demoPointsSig !== s.sig) {
        demoPointsSig = s.sig;
        const ring = C().ring;
        const { dA, dB } = demoCells;
        U().clear(dA.layer);
        U().clear(dB.layer);
        data.labelled.forEach((pt) => {
          dA.layer.appendChild(U().svgEl("circle", { cx: toPx(pt.x), cy: 240 - toPx(pt.y), r: 5, fill: pt.t ? C().a : C().b, stroke: ring, "stroke-width": 1.8 }));
        });
        data.transfer.forEach((pt) => {
          const p = s.target === "soft" ? pt.p : pt.p >= 0.5 ? 1 : 0;
          dB.layer.appendChild(U().svgEl("circle", { cx: toPx(pt.x), cy: 240 - toPx(pt.y), r: 2.2, fill: p >= 0.5 ? C().a : C().b, opacity: 0.35 + 0.65 * Math.abs(p - 0.5) * 2 }));
        });
      }
      const total = R * R;
      const line = (name) => `${((agree[name] / total) * 100).toFixed(1)}% agreement with the teacher · mean gap ${(gap[name] / total).toFixed(2)}`;
      host.querySelector('[data-out="dT"]').textContent = "Teal and orange are the two classes; pale means unsure.";
      host.querySelector('[data-out="dA"]').textContent = `${data.labelled.length} labelled points · ${nets.steps} steps · ${line("dA")}`;
      host.querySelector('[data-out="dB"]').textContent = `400 points, ${s.target === "soft" ? "soft probabilities" : "hard labels"} from the teacher · ${nets.steps} steps · ${line("dB")}`;
    }

    function ensureNets(s) {
      const sig = `${s.H}|${s.nLab}|${s.target}`;
      if (sig === s.sig && s.nets) return;
      demoRun = null;
      s.data = makeDemoData(Number(s.nLab));
      s.nets = { A: netInit(Number(s.H), 11), B: netInit(Number(s.H), 11), steps: 0 };
      s.sig = sig;
    }

    function mountDemo() {
      const host = hostOf("dT");
      buildDemo(host);
      const step = (s) => {
        const { data, nets } = s;
        const tA = data.labelled.map((p) => p.t);
        const tB = data.transfer.map((p) => (s.target === "soft" ? p.p : p.p >= 0.5 ? 1 : 0));
        netStep(nets.A, data.labelled, tA, 0.03);
        netStep(nets.B, data.transfer, tB, 0.03);
        nets.steps += 1;
      };
      mountWidget(host, state.demo, (h, s) => {
        ensureNets(s);
        paintDemo(h, s);
      }, {
        reset: (s) => {
          s.sig = "";
          demoRun = null;
        },
        train: (s) => {
          ensureNets(s);
          const token = {};
          demoRun = token;
          let done = 0;
          const tick = () => {
            if (demoRun !== token || !host.isConnected) return;
            const n = Math.min(5, 600 - done);
            for (let k = 0; k < n; k += 1) step(s);
            done += n;
            paintDemo(host, s);
            if (done < 600) requestAnimationFrame(tick);
            else demoRun = null;
          };
          requestAnimationFrame(tick);
        },
      });
    }

    /* ── 7. Forward vs reverse KL ── */
    function mountKl() {
      const host = hostOf("kl");
      restoreInputs(host, state.kl);
      const draw = (h, s) => {
        const svg = h.querySelector('[data-fig="kl"]');
        const f = frame(svg, {
          xDomain: [-6, 6],
          yDomain: [0, 0.6],
          xTicks: [-6, -4, -2, 0, 2, 4, 6].map((v) => [v, String(v)]),
          yTicks: [0, 0.2, 0.4].map((v) => [v, v.toFixed(1)]),
          title: "Teacher (two peaks) and student (one peak)",
          clipId: "slm-kl-clip",
        });
        const teacherPts = KL_X.map((x, i) => [x, KL_P[i]]);
        const studentPts = KL_X.map((x) => [x, normal(x, s.mu, s.sigma)]);
        f.el("path", { d: `${linePath(teacherPts, f.xs, f.ys)} L${f.xs(6)} ${f.ys(0)} L${f.xs(-6)} ${f.ys(0)} Z`, fill: ui.rgba(C().a, 0.22), stroke: "none" });
        f.el("path", { d: linePath(teacherPts, f.xs, f.ys), fill: "none", stroke: C().a, "stroke-width": 2.4 });
        f.el("path", { d: linePath(studentPts, f.xs, f.ys), fill: "none", stroke: C().b, "stroke-width": 2.8, "clip-path": f.clip });
        [[C().a, "teacher"], [C().b, "student"]].forEach(([color, label], i) => {
          f.el("rect", { x: f.width - 120, y: 40 + i * 18, width: 14, height: 4, fill: color });
          ui.svgText(svg, f.width - 100, 45 + i * 18, label);
        });
        const { fwd, rev } = klPair(s.mu, s.sigma);
        const gapMass = (fn) => KL_X.reduce((acc, x, i) => acc + (Math.abs(x) < 0.8 ? fn(x, i) * KL_DX : 0), 0);
        const studentGap = gapMass((x) => normal(x, s.mu, s.sigma));
        const teacherGap = gapMass((x, i) => KL_P[i]);
        h.querySelector('[data-out="kl"]').innerHTML = `
          <div class="g-metrics">
            <div><span>forward KL(teacher‖student)</span><b>${fmt(fwd, 3)}</b></div>
            <div><span>reverse KL(student‖teacher)</span><b>${fmt(rev, 3)}</b></div>
            <div><span>probability in the gap (|x| &lt; 0.8)</span><b>teacher ${(teacherGap * 100).toFixed(1)}% · student ${(studentGap * 100).toFixed(1)}%</b></div>
          </div>
          <p class="caption">Press “Fit the student” with each divergence. Forward KL settles between the peaks with a wide spread, putting lots of probability where the teacher has almost none; reverse KL locks onto one peak. Start reverse KL from a narrow student (σ below 1) and move μ first to choose which peak it finds.</p>`;
      };
      const sync = mountWidget(host, state.kl, draw, {
        formatOutput: (key, value) => (key === "mode" ? value : Number(value).toFixed(2)),
        reset: (s) => {
          klRun = null;
          Object.assign(s, { mu: 0.5, sigma: 0.6 });
          restoreInputs(host, s);
        },
        fit: (s) => {
          const token = {};
          klRun = token;
          let theta = [s.mu, Math.log(s.sigma)];
          const m = [0, 0];
          const v = [0, 0];
          let t = 0;
          const objective = ([mu, ls]) => {
            const kl = klPair(mu, Math.exp(ls));
            return s.mode === "forward" ? kl.fwd : kl.rev;
          };
          const tick = () => {
            if (klRun !== token || !host.isConnected) return;
            for (let k = 0; k < 3; k += 1) {
              const g = [0, 1].map((idx) => {
                const up = theta.slice();
                const down = theta.slice();
                up[idx] += 1e-3;
                down[idx] -= 1e-3;
                return (objective(up) - objective(down)) / 2e-3;
              });
              t += 1;
              theta = theta.map((value, idx) => {
                m[idx] = 0.9 * m[idx] + 0.1 * g[idx];
                v[idx] = 0.999 * v[idx] + 0.001 * g[idx] * g[idx];
                return value - (0.05 * (m[idx] / (1 - 0.9 ** t))) / (Math.sqrt(v[idx] / (1 - 0.999 ** t)) + 1e-8);
              });
              theta[0] = clamp(theta[0], -4, 4);
              theta[1] = clamp(theta[1], Math.log(0.2), Math.log(3));
            }
            s.mu = theta[0];
            s.sigma = Math.exp(theta[1]);
            restoreInputs(host, s);
            if (sync) sync();
            if (t < 450) requestAnimationFrame(tick);
            else klRun = null;
          };
          requestAnimationFrame(tick);
        },
      });
      host.addEventListener("input", () => {
        klRun = null;
      });
    }

    /* ── 8. Pruning ── */
    function mountPrune() {
      const host = hostOf("prune");
      restoreInputs(host, state.prune);
      const methods = [["magnitude", "smallest |w|"], ["structured", "whole neurons"], ["random", "random"]];
      mountWidget(host, state.prune, (h, s) => {
        const heal = s.heal === "yes";
        const { n, Wm } = getPruneData();
        const mask = pruneMask(s.method, s.sparsity);
        const Wp = prunedWeights(mask, heal);
        const maxAbs = Math.max(...Wm.flat().map(Math.abs));
        const cells = Wp.map((r, i) => `<div class="tl-heat-row">n${i + 1}</div>${r
          .map((value, j) => (mask[i][j]
            ? `<div class="tl-heat-cell" style="${ui.cellStyle(value, maxAbs)}">${fmt(value, 1)}</div>`
            : '<div class="tl-heat-cell sl-pruned">0</div>'))
          .join("")}`).join("");
        const errors = methods.map(([m]) => relError(prunedWeights(pruneMask(m, s.sparsity), heal)));
        const kept = mask.flat().filter(Boolean).length;
        h.querySelector('[data-out="prune"]').innerHTML = `
          <div class="g-two">
            <div>
              <div class="tl-subhead">The layer's weights (rows = output neurons)</div>
              <div class="tl-heat" style="grid-template-columns: max-content repeat(${n}, minmax(30px, 1fr))"><div></div>${Array.from({ length: n }, (_, j) => `<div class="tl-heat-col">x${j + 1}</div>`).join("")}${cells}</div>
            </div>
            <div>
              ${svgSlot("pruneCurve", "0 0 360 250", "output error against sparsity")}
            </div>
          </div>
          <div class="tl-subhead">Output error at ${s.sparsity}% removed${heal ? ", after healing" : ""}</div>
          ${barList(errors.map((e) => e * 100), methods.map(([, label]) => label), { max: 100, digits: 1, highlight: methods.findIndex(([m]) => m === s.method), color: C().c })}
          <p class="caption">${kept} of ${n * n} weights kept. Error is ‖Wx − W′x‖ ÷ ‖Wx‖ in %, averaged over 80 correlated sample inputs.${heal ? " Healing helps unstructured pruning a lot; a removed neuron outputs nothing, so the layer itself can't heal it." : ""}</p>`;

        const svg = h.querySelector('[data-fig="pruneCurve"]');
        const curves = methods.map(([m]) => [0, 10, 20, 30, 40, 50, 60, 70, 80, 90].map((sp) => [sp, relError(prunedWeights(pruneMask(m, sp), heal)) * 100]));
        const yMax = Math.max(100, ...curves.flat().map(([, e]) => e));
        const f = frame(svg, {
          xDomain: [0, 90],
          yDomain: [0, yMax],
          xTicks: [0, 30, 60, 90].map((v) => [v, `${v}%`]),
          yTicks: niceTicks(0, yMax, 4).map((v) => [v, `${v}%`]),
          title: "Output error vs weights removed",
          pad: { left: 48 },
        });
        f.el("line", { x1: f.xs(s.sparsity), y1: f.pad.top, x2: f.xs(s.sparsity), y2: f.height - f.pad.bottom, stroke: C().ink, "stroke-dasharray": "4 4", opacity: 0.5 });
        const colors = [C().a, C().c, C().b];
        curves.forEach((pts, k) => {
          f.el("path", { d: linePath(pts, f.xs, f.ys), fill: "none", stroke: colors[k], "stroke-width": methods[k][0] === s.method ? 3.2 : 2 });
          f.el("rect", { x: 60, y: 38 + k * 16, width: 12, height: 4, fill: colors[k] });
          ui.svgText(svg, 78, 43 + k * 16, methods[k][1]);
        });
      }, { formatOutput: (key, value) => (key === "sparsity" ? `${value}%` : value) });
    }

    /* ── 9. Quantization ── */
    function mountQuant() {
      const host = hostOf("quant");
      restoreInputs(host, state.quant);
      mountWidget(host, state.quant, (h, s) => {
        const ws = quantWeights(s.outlier === "yes");
        const group = s.scheme === "group" ? 16 : ws.length;
        const { qmax, scales, q, deq } = quantize(ws, s.bits, group);
        const yMax = s.outlier === "yes" ? 3.4 : 1.4;
        const svg = h.querySelector('[data-fig="quant"]');
        const f = frame(svg, {
          xDomain: [-0.5, ws.length - 0.5],
          yDomain: [-yMax, yMax],
          xTicks: [0, 16, 32, 47].map((v) => [v, `w${v}`]),
          yTicks: niceTicks(-yMax, yMax, 4).map((v) => [v, fmt(v, 1)]),
          title: `Original weights (dots) and their ${s.bits}-bit versions (bars)`,
          clipId: "slm-q-clip",
        });
        if (qmax <= 15) {
          scales.forEach((sc, g) => {
            const x0 = f.xs(g * group - 0.5);
            const x1 = f.xs(Math.min(ws.length, (g + 1) * group) - 0.5);
            for (let k = -qmax; k <= qmax; k += 1) {
              if (Math.abs(k * sc) > yMax) continue;
              f.el("line", { x1: x0, y1: f.ys(k * sc), x2: x1, y2: f.ys(k * sc), stroke: C().faint, "stroke-width": k === 0 ? 1.4 : 0.8, opacity: 0.8 });
            }
          });
        }
        if (group < ws.length) {
          for (let g = group; g < ws.length; g += group) {
            f.el("line", { x1: f.xs(g - 0.5), y1: f.pad.top, x2: f.xs(g - 0.5), y2: f.height - f.pad.bottom, stroke: C().ink, "stroke-dasharray": "4 4", opacity: 0.35 });
          }
        }
        ws.forEach((wv, i) => {
          const x = f.xs(i);
          f.el("line", { x1: x, y1: f.ys(wv), x2: x, y2: f.ys(deq[i]), stroke: C().danger, "stroke-width": 1.4, "clip-path": f.clip });
          f.el("line", { x1: x - 4, y1: f.ys(deq[i]), x2: x + 4, y2: f.ys(deq[i]), stroke: C().b, "stroke-width": 3, "clip-path": f.clip });
          f.el("circle", { cx: x, cy: f.ys(wv), r: 3, fill: C().a, "clip-path": f.clip });
        });
        const err = ws.map((wv, i) => wv - deq[i]);
        const rmse = Math.sqrt(err.reduce((acc, e) => acc + e * e, 0) / ws.length);
        const rms = Math.sqrt(ws.reduce((acc, wv) => acc + wv * wv, 0) / ws.length);
        const zeros = ws.filter((wv, i) => q[i] === 0 && Math.abs(wv) > 1e-9).length;
        const bpw = s.bits + (16 * scales.length) / ws.length;
        const sample = [24, 25, 26, 27, 28, 29, 30, 31];
        h.querySelector('[data-out="quant"]').innerHTML = `
          <div class="g-metrics">
            <div><span>levels</span><b>${2 * qmax + 1} (−${qmax} … ${qmax})</b></div>
            <div><span>error (RMS)</span><b>${fmt(rmse, 3)} (${((rmse / rms) * 100).toFixed(1)}%)</b></div>
            <div><span>weights rounded to 0</span><b>${zeros} of ${ws.length}</b></div>
            <div><span>bits per weight incl. scales</span><b>${bpw.toFixed(2)}</b></div>
            <div><span>an 8 B model at this rate</span><b>${((8e9 * bpw) / 8 / 1e9).toFixed(1)} GB</b></div>
          </div>
          <div class="tl-subhead">The arithmetic for weights w24–w31${s.outlier === "yes" ? " (w29 is the outlier)" : ""}</div>
          <table class="g-mini"><tr><th>weight</th><th>w</th><th>w ÷ s</th><th>q (integer)</th><th>q · s</th><th>error</th></tr>${sample
            .map((i) => {
              const sc = scales[Math.floor(i / group)];
              return `<tr><td>w${i}</td><td class="mono">${fmt(ws[i], 3)}</td><td class="mono">${fmt(ws[i] / sc, 2)}</td><td class="mono">${q[i]}</td><td class="mono">${fmt(deq[i], 3)}</td><td class="mono">${fmt(ws[i] - deq[i], 3)}</td></tr>`;
            })
            .join("")}</table>
          <p class="caption">Scale${scales.length > 1 ? "s" : ""} s = ${scales.map((sc) => fmt(sc, 3)).join(", ")}. Groups of 16 keep this small example readable; real formats use 32–128.</p>`;
      });
    }

    function render() {
      rootNode.innerHTML = body();
      renderLoss = null;
      mountFit();
      mountScale();
      mountArch();
      mountLoss();
      mountSoft();
      mountDemo();
      mountKl();
      mountPrune();
      mountQuant();
    }

    render();
    U().onRedraw(render);
    ui.startProgressBar();
  }

  /* Plot helpers shared with nn-guides.js. */
  root.MLPlot = { frame, linePath, niceTicks, sparkline, restoreInputs, row };

  root.MLExtraLabs = Object.assign(root.MLExtraLabs || {}, {
    "small-llm": mountSmallLlmGuide,
  });
})();
