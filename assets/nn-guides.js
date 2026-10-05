/* ── Neural network basics as long-form guides ──────────────────
   Activation functions, gradient descent and backpropagation, in the
   same chapter-by-chapter layout as the Attention, Transformer and LLM
   guides. Building blocks come from window.MLSeqUI (transformer-labs.js),
   window.MLGuideWidgets (sequence-guides.js) and window.MLPlot
   (small-llm-guide.js). Every widget computes its numbers live. */
(function () {
  const root = window;
  const UI = () => root.MLSeqUI;
  const W = () => root.MLGuideWidgets;
  const P = () => root.MLPlot;
  const M = () => root.MLTransformerMath;
  const U = () => root.MLUtils;
  const C = () => root.MLUtils.chartColors();

  const sig = (z) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  /* Plain ASCII numbers for KaTeX (it dislikes the Unicode minus). */
  const tn = (v, d = 3) => {
    const s = Number(v).toFixed(d);
    return /^-0\.?0*$/.test(s) ? s.slice(1) : s;
  };
  const tp = (v, d = 3) => (v < 0 ? `(${tn(v, d)})` : tn(v, d));

  const gelu = (z) => 0.5 * z * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (z + 0.044715 * z ** 3)));
  const numDf = (f) => (z) => (f(z + 1e-4) - f(z - 1e-4)) / 2e-4;

  const ACTS = {
    sigmoid: { label: "Sigmoid", f: sig, df: (z) => sig(z) * (1 - sig(z)) },
    tanh: { label: "Tanh", f: Math.tanh, df: (z) => 1 - Math.tanh(z) ** 2 },
    relu: { label: "ReLU", f: (z) => Math.max(0, z), df: (z) => (z > 0 ? 1 : 0) },
    leaky: { label: "Leaky ReLU", f: (z) => (z > 0 ? z : 0.1 * z), df: (z) => (z > 0 ? 1 : 0.1) },
    gelu: { label: "GELU", f: gelu, df: numDf(gelu) },
    silu: { label: "SiLU (Swish)", f: (z) => z * sig(z), df: (z) => sig(z) * (1 + z * (1 - sig(z))) },
    linear: { label: "None (linear)", f: (z) => z, df: () => 1 },
  };

  /* E[f'(z)²] for z ~ N(0, s²), by simple quadrature. */
  function meanSqDeriv(df, s) {
    let total = 0;
    let weight = 0;
    for (let k = -400; k <= 400; k += 1) {
      const u = k / 100;
      const w = Math.exp(-0.5 * u * u);
      total += w * df(u * s) ** 2;
      weight += w;
    }
    return total / weight;
  }

  const hostOf = (rootNode, name) => rootNode.querySelector(`[data-out="${name}"]`).closest(".g-widget");

  function nextCard(eyebrow, title, caption, href, label, primary = true) {
    return `
      <div class="tl-next">
        <div><div class="eyebrow">${eyebrow}</div><strong>${title}</strong><p class="caption">${caption}</p></div>
        <a class="button ${primary ? "primary" : "secondary"}" href="${href}">${label}</a>
      </div>`;
  }

  /* Plot a set of functions of one variable into an svg. */
  function plotCurves(svg, curves, o) {
    const { frame, linePath, niceTicks } = P();
    const f = frame(svg, Object.assign({
      xTicks: niceTicks(o.xDomain[0], o.xDomain[1], 6).map((v) => [v, String(v)]),
      yTicks: niceTicks(o.yDomain[0], o.yDomain[1], 4).map((v) => [v, String(v)]),
      clipId: o.clipId,
    }, o));
    curves.forEach((c) => {
      const pts = [];
      for (let i = 0; i <= 240; i += 1) {
        const x = o.xDomain[0] + ((o.xDomain[1] - o.xDomain[0]) * i) / 240;
        pts.push([x, c.fn(x)]);
      }
      f.el("path", { d: linePath(pts, f.xs, f.ys), fill: "none", stroke: c.color, "stroke-width": c.width || 2.6, "stroke-dasharray": c.dash || "none", "clip-path": f.clip, opacity: c.opacity || 1 });
    });
    if (o.legend) {
      o.legend.forEach(([color, label, dash], i) => {
        f.el("line", { x1: f.pad.left + 12, y1: f.pad.top + 10 + i * 17, x2: f.pad.left + 30, y2: f.pad.top + 10 + i * 17, stroke: color, "stroke-width": 3, "stroke-dasharray": dash || "none" });
        UI().svgText(svg, f.pad.left + 36, f.pad.top + 14 + i * 17, label);
      });
    }
    return f;
  }

  /* ════════════════════════════════════════════════════════════════
     GUIDE 1 · ACTIVATION FUNCTIONS
     ════════════════════════════════════════════════════════════════ */

  function mountActivationGuide(rootNode) {
    const ui = UI();
    const { widget, range, segmented, mountWidget } = W();
    const { row, restoreInputs } = P();
    const { para, plain, deeper, takeaway, figure, chapter, guideToc, mathBlock, svgSlot, barList, quiz, fmt } = ui;

    const chapters = [
      { id: "act-neuron", title: "A neuron: sum, then squash", blurb: "weighted sum plus an activation" },
      { id: "act-collapse", title: "Why stacking linear layers is pointless", blurb: "matrices multiply into one matrix" },
      { id: "act-squash", title: "Sigmoid and tanh", blurb: "the classic S-curves and their slopes" },
      { id: "act-vanish", title: "Vanishing and exploding gradients", blurb: "what deep stacks do to a slope" },
      { id: "act-relu", title: "ReLU, and neurons that die", blurb: "the simple fix and its failure mode" },
      { id: "act-modern", title: "Smooth modern activations", blurb: "GELU and SiLU in today's models" },
      { id: "act-output", title: "Activations at the output", blurb: "sigmoid, softmax or nothing" },
      { id: "act-universal", title: "Bending lines into any curve", blurb: "universal approximation with ReLUs" },
      { id: "act-recap", title: "Recap and self-check", blurb: "" },
    ];

    const DEAD_X = (() => {
      const rng = M().seeded(31);
      return Array.from({ length: 60 }, () => M().gaussian(rng));
    })();

    const GRID_LINES = (() => {
      const lines = [];
      const ticks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1];
      ticks.forEach((t) => {
        const h = [];
        const v = [];
        for (let i = 0; i <= 30; i += 1) {
          const s = -1 + (2 * i) / 30;
          h.push([s, t]);
          v.push([t, s]);
        }
        lines.push({ pts: h, kind: "h" }, { pts: v, kind: "v" });
      });
      return lines;
    })();

    const TARGETS = {
      sine: { label: "a wave", fn: (x) => Math.sin(1.5 * x) },
      bump: { label: "a bump", fn: (x) => 1.5 * Math.exp(-x * x) },
      step: { label: "a step", fn: (x) => sig(5 * x) },
    };

    function body() {
      return `
        <div class="g-intro">
          ${para("A neural network is layers of simple units, and each unit does two things: it adds up its inputs with weights, then passes the total through an <strong>activation function</strong>. That second step looks like a detail, but without it a 100-layer network is no more powerful than a single layer. This guide shows why, and how the choice of activation decides whether a deep network can learn at all.")}
          ${para("Each chapter starts in plain words, then gives you something to play with, then the maths.")}
          ${guideToc(chapters)}
        </div>

        ${chapter("act-neuron", 1, "Basics", "A neuron: sum, then squash", `
          ${plain(para("A neuron takes some inputs x<sub>1</sub>, x<sub>2</sub>, …, multiplies each by a <strong>weight</strong> (how much it cares about that input), adds a <strong>bias</strong> (its own baseline), and gets one number z. Then it applies an activation function f to get its output a = f(z). Learning means adjusting the weights and bias; the activation stays fixed."))}
          ${widget("one neuron with two inputs", `
            ${range("x1", "Input x₁", -2, 2, 0.1, 1)}
            ${range("x2", "Input x₂", -2, 2, 0.1, -0.5)}
            ${range("w1", "Weight w₁", -2, 2, 0.1, 0.8)}
            ${range("w2", "Weight w₂", -2, 2, 0.1, 1.2)}
            ${range("b", "Bias b", -2, 2, 0.1, 0.3)}
            ${row("Activation f", segmented("act", [["sigmoid", "sigmoid"], ["tanh", "tanh"], ["relu", "ReLU"], ["linear", "none"]], "sigmoid"))}
            <div class="g-two g-two-wide">
              <div>${svgSlot("neuron", "0 0 460 260", "activation curve with the neuron's operating point")}</div>
              <div data-out="neuron"></div>
            </div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`z = \sum_i w_i x_i + b = \mathbf{w}\cdot\mathbf{x} + b,\qquad a = f(z)`)}
            ${para("A whole layer does this for many neurons at once, which is a matrix–vector product: <b>z</b> = W<b>x</b> + <b>b</b>, then f is applied to every entry.")}
          `)}
          ${takeaway("A neuron is a weighted sum plus a bias, passed through a fixed function f.")}
        `)}

        ${chapter("act-collapse", 2, "Why it matters", "Why stacking linear layers is pointless", `
          ${plain(para("Without an activation, a layer is just a matrix multiply, and two matrix multiplies in a row are the same as one (with the product matrix). So a deep network of purely linear layers can only ever stretch, rotate and shear space: straight lines stay straight. Put an activation between the layers and the network can <em>bend</em> space, which is what lets it separate tangled data."))}
          ${widget("push a grid through two layers", `
            <div class="g-inputs">
              <label><span>W₁ row 1</span><input type="number" data-input="a11" step="0.1" /></label>
              <label><span>&nbsp;</span><input type="number" data-input="a12" step="0.1" /></label>
              <label><span>W₁ row 2</span><input type="number" data-input="a21" step="0.1" /></label>
              <label><span>&nbsp;</span><input type="number" data-input="a22" step="0.1" /></label>
            </div>
            ${row("Between the layers", segmented("act", [["linear", "nothing (linear)"], ["relu", "ReLU"], ["tanh", "tanh"]], "linear"))}
            <div class="g-three">
              <div><div class="tl-subhead">Input</div>${svgSlot("gridIn", "0 0 220 220", "input grid")}</div>
              <div><div class="tl-subhead">After layer 1 + f</div>${svgSlot("gridMid", "0 0 220 220", "grid after the first layer")}</div>
              <div><div class="tl-subhead">After layer 2</div>${svgSlot("gridOut", "0 0 220 220", "grid after the second layer")}</div>
            </div>
            <div data-out="grid"></div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`W_2\big(W_1\mathbf{x} + \mathbf{b}_1\big) + \mathbf{b}_2 = \underbrace{(W_2W_1)}_{\text{one matrix}}\mathbf{x} + \underbrace{(W_2\mathbf{b}_1 + \mathbf{b}_2)}_{\text{one bias}}`)}
            ${para("With f in between, W<sub>2</sub> f(W<sub>1</sub><b>x</b> + <b>b</b><sub>1</sub>) cannot be rewritten that way. ReLU folds space along lines where a neuron switches on; tanh squeezes it into a box.")}
          `)}
          ${takeaway("Activations are the only reason depth adds power: without them, any number of layers equals one.")}
        `)}

        ${chapter("act-squash", 3, "Classics", "Sigmoid and tanh", `
          ${plain(para("The first networks used smooth S-shaped functions. <strong>Sigmoid</strong> squashes any number into (0, 1), like a probability. <strong>Tanh</strong> squashes into (−1, 1) and is centred on zero, which makes the next layer's job easier. Their weakness is in the <em>slope</em>: far from zero both curves go flat, and a flat function passes almost no gradient back during training. That is called <strong>saturation</strong>."))}
          ${widget("a function and its slope", `
            ${row("Function", segmented("act", [["sigmoid", "sigmoid"], ["tanh", "tanh"]], "sigmoid"))}
            ${range("z", "Input z", -6, 6, 0.1, 1.5)}
            <div class="g-two g-two-wide">
              <div>${svgSlot("squash", "0 0 460 270", "function, derivative and saturated regions")}</div>
              <div data-out="squash"></div>
            </div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`\sigma(z) = \frac{1}{1+e^{-z}},\quad \sigma'(z) = \sigma(z)\big(1-\sigma(z)\big) \le \tfrac14;\qquad \tanh'(z) = 1 - \tanh^2(z) \le 1`)}
            ${para("The sigmoid's slope is at most 0.25, reached only at z = 0. That single number explains the next chapter.")}
          `)}
          ${takeaway("Sigmoid and tanh are smooth and bounded, but flat at the ends: saturated neurons barely learn.")}
        `)}

        ${chapter("act-vanish", 4, "Depth", "Vanishing and exploding gradients", `
          ${plain(para("Training sends a signal backwards from the loss to every weight (that's backpropagation). At every layer it gets multiplied by a weight and by the activation's slope. Multiply 20 numbers that are each about 0.2 and you get 10<sup>−14</sup>: the early layers receive essentially nothing and stop learning. That's the <strong>vanishing gradient</strong>. Numbers bigger than 1 do the opposite and the signal <strong>explodes</strong>. The goal is a per-layer factor of about 1."))}
          ${widget("how a gradient shrinks or grows with depth", `
            ${row("Activation", segmented("act", [["sigmoid", "sigmoid"], ["tanh", "tanh"], ["relu", "ReLU"]], "sigmoid"))}
            ${range("depth", "Layers", 2, 30, 1, 12)}
            ${range("w", "Weight scale", 0.5, 3, 0.05, 1)}
            ${range("s", "Spread of pre-activations z", 0.25, 3, 0.05, 1)}
            ${svgSlot("vanish", "0 0 560 260", "gradient size at each layer, log scale")}
            <div data-out="vanish"></div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`\Big|\frac{\partial L}{\partial \mathbf{h}_{\ell}}\Big| \approx r^{\,L-\ell}\,\Big|\frac{\partial L}{\partial \mathbf{h}_{L}}\Big|,\qquad r = w\sqrt{\mathbb{E}\big[f'(z)^2\big]}`)}
            ${para("This is the standard back-of-the-envelope analysis, assuming each layer's weights have typical size w and pre-activations z are spread with standard deviation s. For ReLU, E[f′²] = ½, so a weight scale of √2 ≈ 1.41 gives r = 1 exactly: that is <strong>He initialisation</strong> (He et al., 2015). Residual connections and normalisation layers, used in every Transformer, attack the same problem.")}
          `)}
          ${takeaway("Gradients get multiplied once per layer; keep the factor near 1 with the right activation and initialisation, or deep networks won't train.")}
        `)}

        ${chapter("act-relu", 5, "Modern default", "ReLU, and neurons that die", `
          ${plain(para("<strong>ReLU</strong> (rectified linear unit) is almost embarrassingly simple: f(z) = max(0, z). For positive inputs its slope is exactly 1, so gradients pass through undiminished; that's why it made deep networks practical around 2012. The catch: for negative inputs the slope is exactly 0. If a neuron's bias drifts so negative that <em>no</em> input ever turns it on, it outputs 0 forever, gets zero gradient forever, and never recovers. It is <strong>dead</strong>. <strong>Leaky ReLU</strong> keeps a small slope (0.1 here) on the negative side so it can come back."))}
          ${widget("one ReLU neuron, 60 different inputs", `
            ${row("Activation", segmented("act", [["relu", "ReLU"], ["leaky", "Leaky ReLU"]], "relu"))}
            ${range("w", "Weight w", -2, 2, 0.05, 1)}
            ${range("b", "Bias b", -5, 2, 0.05, 0)}
            <div class="g-two g-two-wide">
              <div>${svgSlot("dead", "0 0 460 260", "each input's pre-activation and output")}</div>
              <div data-out="dead"></div>
            </div>`)}
          ${takeaway("ReLU passes gradients perfectly when on and not at all when off; a neuron that is never on is dead, which leaky variants prevent.")}
        `)}

        ${chapter("act-modern", 6, "Modern", "Smooth modern activations", `
          ${plain(para("Large language models mostly use smooth cousins of ReLU. <strong>GELU</strong> (used by BERT and GPT-2) and <strong>SiLU</strong>, also called Swish (used inside the SwiGLU layers of Llama and others), look like ReLU for large inputs but curve gently through zero and let small negative values through. The smooth corner and the tiny negative dip consistently train a little better at scale."))}
          ${figure("6.1", "Four activations (left) and their slopes (right). Only ReLU has a jump in its slope.", `
            <div class="g-two">
              <div>${svgSlot("modernF", "0 0 380 260", "ReLU, leaky ReLU, GELU and SiLU")}</div>
              <div>${svgSlot("modernD", "0 0 380 260", "their derivatives")}</div>
            </div>`, true)}
          ${deeper("The math", `
            ${mathBlock(String.raw`\operatorname{GELU}(z) = z\,\Phi(z),\qquad \operatorname{SiLU}(z) = z\,\sigma(z)`)}
            ${para("Φ is the standard normal CDF. Both multiply z by a smooth “gate” between 0 and 1, so they act like a soft version of ReLU's on/off switch. See the <a href=\"./algorithm.html?id=llm#llm-block\">LLM guide</a> for how SwiGLU uses SiLU.")}
          `)}
          ${takeaway("Today's large models use smooth gates like GELU and SiLU: ReLU's behaviour without its sharp corner.")}
        `)}

        ${chapter("act-output", 7, "Output layer", "Activations at the output", `
          ${plain(para("The last layer's activation depends on what the network must output. A number such as a price uses <strong>no activation</strong>. A yes/no answer uses <strong>sigmoid</strong>, which gives a probability. A choice of exactly one class among many uses <strong>softmax</strong>, which turns a list of scores into probabilities that sum to 1. Several independent yes/no labels (a photo can contain both a cat <em>and</em> a dog) use one sigmoid per label."))}
          ${widget("softmax vs independent sigmoids", `
            <div class="g-inputs">${["cat", "dog", "bird", "fish"].map((c, i) => `<label><span>score for “${c}”</span><input type="number" data-input="z${i}" step="0.5" /></label>`).join("")}</div>
            <div data-out="output"></div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`\operatorname{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}\quad\text{vs}\quad \sigma(z_i) = \frac{1}{1+e^{-z_i}}`)}
            ${para("Softmax couples the outputs: raising one score lowers every other probability. Sigmoids are independent. Each pairs with a matching loss: softmax with categorical cross-entropy, sigmoid with binary cross-entropy, and no activation with squared error.")}
          `)}
          ${takeaway("Output activation = output type: none for numbers, sigmoid for yes/no, softmax for one-of-many.")}
        `)}

        ${chapter("act-universal", 8, "Power", "Bending lines into any curve", `
          ${plain(para("How can such simple pieces represent complicated functions? Each ReLU unit adds one <em>bend</em> to an otherwise straight line. With enough units, a single hidden layer can follow any smooth curve as closely as you like: this is the <strong>universal approximation theorem</strong>. Deeper networks can do the same job with far fewer units, because each layer can reuse and fold the bends of the previous one."))}
          ${widget("build a curve out of ReLU units", `
            ${row("Target", segmented("target", Object.entries(TARGETS).map(([k, t]) => [k, t.label]), "sine"))}
            ${range("n", "Hidden ReLU units", 1, 24, 1, 4)}
            ${svgSlot("universal", "0 0 560 280", "target curve and the ReLU approximation")}
            <div data-out="universal"></div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`g(x) = c + \sum_{k=1}^{n} a_k \operatorname{ReLU}(x - t_k)`)}
            ${para("Unit k switches on at x = t<sub>k</sub> and changes the slope by a<sub>k</sub>. Here the bends are evenly spaced and the slopes are chosen to join the target's values, so the error is that of joining dots with straight lines; a trained network would also move the bends to where the curve needs them.")}
          `)}
          ${takeaway("Each ReLU adds a bend; enough bends can follow any curve, which is why networks can learn almost anything.")}
        `)}

        ${chapter("act-recap", 9, "Recap", "Recap and self-check", `
          <ol class="g-recap">
            <li>A neuron computes <b>z = w·x + b</b> and outputs <b>a = f(z)</b>.</li>
            <li>Without activations, any stack of layers <b>collapses to one</b> linear layer.</li>
            <li>Sigmoid and tanh <b>saturate</b>; the sigmoid's slope is at most <b>0.25</b>.</li>
            <li>Gradients are multiplied once per layer: keep the factor near 1 (ReLU + <b>He initialisation</b>, residuals, normalisation).</li>
            <li>ReLU neurons can <b>die</b>; leaky ReLU, GELU and SiLU avoid hard zeros.</li>
            <li>Output layer: <b>none</b> for numbers, <b>sigmoid</b> for yes/no, <b>softmax</b> for one-of-many.</li>
          </ol>
          ${quiz([
            { q: "Two layers with weights W₁ (3×2) and W₂ (2×3) and no activation. What single layer is equivalent?", a: "One layer with weight matrix W₂W₁ (2×2) and bias W₂b₁ + b₂." },
            { q: "Why do 10 sigmoid layers train much worse than 10 ReLU layers?", a: "The backward signal is multiplied by σ′ ≤ 0.25 at each layer, so after 10 layers it is at most 0.25¹⁰ ≈ 10⁻⁶ of its size; ReLU's slope is 1 wherever it is active." },
            { q: "A ReLU neuron has bias −10 and its inputs are all small. What happens in training?", a: "z is always negative, so the output and the gradient are always 0. The neuron is dead and its weights never change." },
            { q: "Which output activation for predicting tomorrow's temperature? For tagging a photo with every animal in it?", a: "None (linear) for the temperature; one sigmoid per animal for the multi-label tagging." },
          ])}
          ${nextCard("Next guide", "Gradient descent", "How a network uses the slope of the loss to improve its weights, step by step.", "./algorithm.html?id=gradient-descent", "Continue to gradient descent →")}
        `)}
      `;
    }

    const state = {
      neuron: { x1: 1, x2: -0.5, w1: 0.8, w2: 1.2, b: 0.3, act: "sigmoid" },
      grid: { a11: 1.2, a12: -0.6, a21: 0.5, a22: 1.0, act: "linear" },
      squash: { act: "sigmoid", z: 1.5 },
      vanish: { act: "sigmoid", depth: 12, w: 1, s: 1 },
      dead: { act: "relu", w: 1, b: 0 },
      output: { z0: 2, z1: 1, z2: 0.1, z3: -1 },
      universal: { target: "sine", n: 4 },
    };
    const fmtOut = (key, value) => (typeof value === "number" ? Number(value).toFixed(2).replace(/\.00$/, "") : value);

    function mountNeuron() {
      const host = hostOf(rootNode, "neuron");
      restoreInputs(host, state.neuron);
      mountWidget(host, state.neuron, (h, s) => {
        const act = ACTS[s.act];
        const z = s.w1 * s.x1 + s.w2 * s.x2 + s.b;
        const a = act.f(z);
        const svg = h.querySelector('[data-fig="neuron"]');
        const yDom = s.act === "sigmoid" ? [-0.2, 1.2] : s.act === "tanh" ? [-1.2, 1.2] : [-3, 6];
        const f = plotCurves(svg, [{ fn: act.f, color: C().a }], { xDomain: [-6, 6], yDomain: yDom, title: `a = ${act.label === "None (linear)" ? "z" : act.label}(z)`, clipId: "ag-neuron-clip" });
        const zc = clamp(z, -6, 6);
        f.el("line", { x1: f.xs(zc), y1: f.ys(yDom[0]), x2: f.xs(zc), y2: f.ys(clamp(a, yDom[0], yDom[1])), stroke: C().b, "stroke-dasharray": "4 4" });
        f.el("circle", { cx: f.xs(zc), cy: f.ys(clamp(a, yDom[0], yDom[1])), r: 6, fill: C().b, stroke: C().ring, "stroke-width": 2 });
        h.querySelector('[data-out="neuron"]').innerHTML = `
          ${mathBlock(String.raw`z = ${tn(s.w1, 1)}\cdot${tp(s.x1, 1)} + ${tn(s.w2, 1)}\cdot${tp(s.x2, 1)} + ${tp(s.b, 1)} = ${tn(z, 2)}`)}
          ${mathBlock(String.raw`a = f(${tn(z, 2)}) = ${tn(a, 3)}`)}
          <p class="caption">The orange dot is where this neuron sits on its activation curve. Push z far from 0 with sigmoid or tanh and the output stops changing: the neuron is saturated.</p>`;
      }, { formatOutput: fmtOut });
    }

    function mountGrid() {
      const host = hostOf(rootNode, "grid");
      restoreInputs(host, state.grid);
      const W2 = [[0.9, 0.7], [-0.8, 1.1]];
      const b1 = [0.2, -0.3];
      const draw = (svg, lines) => {
        U().clear(svg);
        let m = 0.2;
        lines.forEach((l) => l.pts.forEach(([x, y]) => {
          m = Math.max(m, Math.abs(x), Math.abs(y));
        }));
        m *= 1.08;
        const sc = (v) => 110 + (v / m) * 104;
        svg.appendChild(U().svgEl("line", { x1: 0, y1: 110, x2: 220, y2: 110, class: "axis-line" }));
        svg.appendChild(U().svgEl("line", { x1: 110, y1: 0, x2: 110, y2: 220, class: "axis-line" }));
        lines.forEach((l) => {
          svg.appendChild(U().svgEl("path", {
            d: l.pts.map(([x, y], i) => `${i ? "L" : "M"}${sc(x).toFixed(1)} ${(220 - sc(y)).toFixed(1)}`).join(" "),
            fill: "none",
            stroke: l.kind === "h" ? C().c : C().a,
            "stroke-width": 1.6,
            opacity: 0.9,
          }));
        });
      };
      mountWidget(host, state.grid, (h, s) => {
        const W1 = [[Number(s.a11) || 0, Number(s.a12) || 0], [Number(s.a21) || 0, Number(s.a22) || 0]];
        const f = ACTS[s.act].f;
        const layer = (Wm, b, p) => [Wm[0][0] * p[0] + Wm[0][1] * p[1] + (b ? b[0] : 0), Wm[1][0] * p[0] + Wm[1][1] * p[1] + (b ? b[1] : 0)];
        const mid = GRID_LINES.map((l) => ({ kind: l.kind, pts: l.pts.map((p) => layer(W1, b1, p).map(f)) }));
        const out = mid.map((l) => ({ kind: l.kind, pts: l.pts.map((p) => layer(W2, null, p)) }));
        draw(h.querySelector('[data-fig="gridIn"]'), GRID_LINES);
        draw(h.querySelector('[data-fig="gridMid"]'), mid);
        draw(h.querySelector('[data-fig="gridOut"]'), out);
        const prod = [[W2[0][0] * W1[0][0] + W2[0][1] * W1[1][0], W2[0][0] * W1[0][1] + W2[0][1] * W1[1][1]], [W2[1][0] * W1[0][0] + W2[1][1] * W1[1][0], W2[1][0] * W1[0][1] + W2[1][1] * W1[1][1]]];
        h.querySelector('[data-out="grid"]').innerHTML = s.act === "linear"
          ? `${mathBlock(String.raw`W_2W_1 = \begin{bmatrix}${tn(W2[0][0], 1)} & ${tn(W2[0][1], 1)}\\ ${tn(W2[1][0], 1)} & ${tn(W2[1][1], 1)}\end{bmatrix}\begin{bmatrix}${tn(W1[0][0], 1)} & ${tn(W1[0][1], 1)}\\ ${tn(W1[1][0], 1)} & ${tn(W1[1][1], 1)}\end{bmatrix} = \begin{bmatrix}${tn(prod[0][0], 2)} & ${tn(prod[0][1], 2)}\\ ${tn(prod[1][0], 2)} & ${tn(prod[1][1], 2)}\end{bmatrix}`)}
             <p class="caption">Straight lines stay straight and parallel lines stay parallel: the two layers are exactly one matrix (plus a shift). Edit W₁ and they still collapse.</p>`
          : `<p class="caption">With ${ACTS[s.act].label} in between, grid lines ${s.act === "relu" ? "fold where a neuron switches off (everything negative is flattened onto an axis)" : "curve and squeeze into a box"}. No single matrix can do that, so the second layer now sees a genuinely new arrangement of the data.</p>`;
      }, { formatOutput: fmtOut });
    }

    function mountSquash() {
      const host = hostOf(rootNode, "squash");
      restoreInputs(host, state.squash);
      mountWidget(host, state.squash, (h, s) => {
        const act = ACTS[s.act];
        const yDom = s.act === "sigmoid" ? [-0.1, 1.1] : [-1.1, 1.1];
        const svg = h.querySelector('[data-fig="squash"]');
        const f = P().frame(svg, {
          xDomain: [-6, 6],
          yDomain: yDom,
          xTicks: [-6, -4, -2, 0, 2, 4, 6].map((v) => [v, String(v)]),
          yTicks: P().niceTicks(yDom[0], yDom[1], 4).map((v) => [v, String(v)]),
          title: `${act.label} (solid) and its slope (dashed)`,
        });
        for (let x = -6; x < 6; x += 0.05) {
          if (act.df(x) < 0.05) f.el("rect", { x: f.xs(x), y: f.pad.top, width: f.xs(x + 0.05) - f.xs(x) + 0.3, height: f.ys(yDom[0]) - f.pad.top, fill: ui.rgba(C().danger, 0.08) });
        }
        [[act.f, C().a, "none"], [act.df, C().c, "6 4"]].forEach(([fn, color, dash]) => {
          const pts = [];
          for (let i = 0; i <= 240; i += 1) pts.push([-6 + i * 0.05, fn(-6 + i * 0.05)]);
          f.el("path", { d: P().linePath(pts, f.xs, f.ys), fill: "none", stroke: color, "stroke-width": 2.6, "stroke-dasharray": dash });
        });
        f.el("circle", { cx: f.xs(s.z), cy: f.ys(act.f(s.z)), r: 6, fill: C().b, stroke: C().ring, "stroke-width": 2 });
        f.el("circle", { cx: f.xs(s.z), cy: f.ys(act.df(s.z)), r: 5, fill: C().c, stroke: C().ring, "stroke-width": 2 });
        const d = act.df(s.z);
        h.querySelector('[data-out="squash"]').innerHTML = `
          <div class="g-metrics">
            <div><span>f(z)</span><b>${fmt(act.f(s.z), 4)}</b></div>
            <div><span>slope f′(z)</span><b>${fmt(d, 4)}</b></div>
            <div><span>largest possible slope</span><b>${s.act === "sigmoid" ? "0.25" : "1"}</b></div>
            <div><span>state</span><b>${d < 0.05 ? "saturated" : "responsive"}</b></div>
          </div>
          <p class="caption">Shaded: where the slope is below 0.05. A neuron there changes its output by less than a twentieth of any change in its input, so learning signals barely pass.</p>`;
      }, { formatOutput: fmtOut });
    }

    function mountVanish() {
      const host = hostOf(rootNode, "vanish");
      restoreInputs(host, state.vanish);
      mountWidget(host, state.vanish, (h, s) => {
        const r = s.w * Math.sqrt(meanSqDeriv(ACTS[s.act].df, s.s));
        const L = s.depth;
        const logs = Array.from({ length: L }, (_, i) => (L - 1 - i) * Math.log10(r));
        const lo = Math.max(-16, Math.floor(Math.min(0, ...logs)) - 1);
        const hi = Math.min(16, Math.ceil(Math.max(0, ...logs)) + 1);
        const svg = h.querySelector('[data-fig="vanish"]');
        const ticks = P().niceTicks(lo, hi, 5).filter((v) => Number.isInteger(v));
        const f = P().frame(svg, {
          xDomain: [0.4, L + 0.6],
          yDomain: [lo, hi],
          xTicks: Array.from({ length: L }, (_, i) => i + 1).filter((v) => L <= 15 || v % 5 === 0 || v === 1).map((v) => [v, String(v)]),
          yTicks: ticks.map((v) => [v, v === 0 ? "1" : `1e${v}`]),
          title: "Gradient size reaching each layer (log scale; the output is the last layer)",
          xLabel: "layer",
          pad: { left: 52 },
        });
        f.el("line", { x1: f.pad.left, y1: f.ys(0), x2: f.width - f.pad.right, y2: f.ys(0), stroke: C().ink, opacity: 0.4 });
        const bw = Math.max(3, (f.xs(2) - f.xs(1)) * 0.7);
        logs.forEach((v, i) => {
          const y0 = f.ys(0);
          const y1 = f.ys(clamp(v, lo, hi));
          f.el("rect", { x: f.xs(i + 1) - bw / 2, y: Math.min(y0, y1), width: bw, height: Math.max(1, Math.abs(y1 - y0)), fill: v < -3 ? C().b : v > 3 ? C().danger : C().a, rx: 2 });
        });
        const first = 10 ** logs[0];
        const verdict = r < 0.8 ? "vanishing" : r > 1.25 ? "exploding" : "healthy";
        h.querySelector('[data-out="vanish"]').innerHTML = `
          <div class="g-metrics">
            <div><span>factor per layer r</span><b>${fmt(r, 3)}</b></div>
            <div><span>gradient at layer 1</span><b>${first < 1e-3 || first > 1e3 ? first.toExponential(1) : fmt(first, 3)}</b></div>
            <div><span>verdict</span><b>${verdict}</b></div>
          </div>
          <p class="caption">${s.act === "sigmoid" ? "Sigmoid's slope never exceeds 0.25, so with ordinary weights r is far below 1. " : ""}${s.act === "relu" ? "Try weight scale 1.41 (= √2): r becomes 1 and every layer gets the same signal. That is He initialisation. " : ""}Bars are relative to the gradient at the output layer.</p>`;
      }, { formatOutput: fmtOut });
    }

    function mountDead() {
      const host = hostOf(rootNode, "dead");
      restoreInputs(host, state.dead);
      mountWidget(host, state.dead, (h, s) => {
        const act = ACTS[s.act];
        const zs = DEAD_X.map((x) => s.w * x + s.b);
        const active = zs.filter((z) => z > 0).length;
        const meanGrad = zs.reduce((acc, z) => acc + act.df(z), 0) / zs.length;
        const svg = h.querySelector('[data-fig="dead"]');
        const f = plotCurves(svg, [{ fn: act.f, color: C().faint }], { xDomain: [-8, 6], yDomain: [-1, 5], title: "Each dot: one input's z and the neuron's output", clipId: "ag-dead-clip" });
        zs.forEach((z) => {
          if (z < -8 || z > 6) return;
          f.el("circle", { cx: f.xs(z), cy: f.ys(clamp(act.f(z), -1, 5)), r: 4, fill: z > 0 ? C().a : C().b, opacity: 0.85 });
        });
        const dead = active === 0 && s.act === "relu";
        h.querySelector('[data-out="dead"]').innerHTML = `
          <div class="g-metrics">
            <div><span>inputs that switch it on</span><b>${active} of ${zs.length}</b></div>
            <div><span>average slope (learning signal)</span><b>${fmt(meanGrad, 3)}</b></div>
            <div><span>status</span><b>${dead ? "dead ✗" : active === 0 ? "off, but leaking" : "alive ✓"}</b></div>
          </div>
          <p class="caption">${dead
            ? "Every input lands on the flat part. The output is 0 and so is every gradient, so gradient descent will never move w or b again."
            : s.act === "leaky" && active === 0
              ? "Every input is negative, but leaky ReLU still passes 10% of the slope, so training can push the bias back up."
              : "Orange dots are inputs where the neuron is off and passes no gradient. Drag the bias down to about −4 to kill it."}</p>`;
      }, { formatOutput: fmtOut });
    }

    function mountOutput() {
      const host = hostOf(rootNode, "output");
      restoreInputs(host, state.output);
      const labels = ["cat", "dog", "bird", "fish"];
      mountWidget(host, state.output, (h, s) => {
        const z = [0, 1, 2, 3].map((i) => Number(s[`z${i}`]) || 0);
        const soft = M().softmax(z);
        const sg = z.map(sig);
        const sum = (v) => v.reduce((a, b) => a + b, 0);
        h.querySelector('[data-out="output"]').innerHTML = `
          <div class="g-two">
            <div><div class="tl-subhead">Softmax (pick one)</div>${barList(soft, labels, { max: 1, asPercent: true, color: C().a })}<p class="caption">Sum: <b>${(sum(soft) * 100).toFixed(0)}%</b></p></div>
            <div><div class="tl-subhead">Independent sigmoids (any number)</div>${barList(sg, labels, { max: 1, asPercent: true, color: C().c })}<p class="caption">Sum: <b>${(sum(sg) * 100).toFixed(0)}%</b></p></div>
          </div>
          <p class="caption">Raise one score: with softmax every other probability drops; with sigmoids the others don't move.</p>`;
      }, { formatOutput: fmtOut });
    }

    function mountUniversal() {
      const host = hostOf(rootNode, "universal");
      restoreInputs(host, state.universal);
      mountWidget(host, state.universal, (h, s) => {
        const target = TARGETS[s.target].fn;
        const n = s.n;
        const knots = Array.from({ length: n + 1 }, (_, k) => -3 + (6 * k) / n);
        const vals = knots.map(target);
        const slopes = Array.from({ length: n }, (_, k) => (vals[k + 1] - vals[k]) / (knots[k + 1] - knots[k]));
        const units = slopes.map((sl, k) => ({ t: knots[k], a: k === 0 ? sl : sl - slopes[k - 1] }));
        const g = (x) => vals[0] + units.reduce((acc, u) => acc + u.a * Math.max(0, x - u.t), 0);
        const svg = h.querySelector('[data-fig="universal"]');
        const curves = units.map((u) => ({ fn: (x) => u.a * Math.max(0, x - u.t), color: C().neutral, width: 1.2, opacity: 0.45 }));
        curves.push({ fn: target, color: C().a, width: 5, opacity: 0.45 }, { fn: g, color: C().b, width: 2.6 });
        const f = plotCurves(svg, curves, { xDomain: [-3, 3], yDomain: [-1.6, 1.8], title: `${n} ReLU unit${n > 1 ? "s" : ""} approximating ${TARGETS[s.target].label}`, clipId: "ag-uni-clip", legend: [[C().a, "target"], [C().b, "network output"], [C().neutral, "individual units"]] });
        knots.slice(0, n).forEach((t) => f.el("circle", { cx: f.xs(t), cy: f.ys(g(t)), r: 3.5, fill: C().b }));
        let maxErr = 0;
        for (let x = -3; x <= 3; x += 0.01) maxErr = Math.max(maxErr, Math.abs(g(x) - target(x)));
        h.querySelector('[data-out="universal"]').innerHTML = `
          <div class="g-metrics">
            <div><span>hidden units</span><b>${n}</b></div>
            <div><span>parameters</span><b>${3 * n + 1}</b></div>
            <div><span>worst error</span><b>${fmt(maxErr, 3)}</b></div>
          </div>
          <p class="caption">Each faint line is one unit: flat, then a ramp starting at its bend (dots). Their sum is the orange curve. Double the units and the worst error drops by about 4×.</p>`;
      }, { formatOutput: fmtOut });
    }

    function drawModern() {
      const names = ["relu", "leaky", "gelu", "silu"];
      const colors = [C().neutral, C().c, C().a, C().b];
      const legend = names.map((n, i) => [colors[i], ACTS[n].label]);
      const fsvg = rootNode.querySelector('[data-fig="modernF"]');
      const dsvg = rootNode.querySelector('[data-fig="modernD"]');
      plotCurves(fsvg, names.map((n, i) => ({ fn: ACTS[n].f, color: colors[i] })), { xDomain: [-4, 3], yDomain: [-1, 3], title: "f(z)", legend, clipId: "ag-mf" });
      plotCurves(dsvg, names.map((n, i) => ({ fn: ACTS[n].df, color: colors[i] })), { xDomain: [-4, 3], yDomain: [-0.3, 1.3], title: "f′(z)", clipId: "ag-md" });
    }

    function render() {
      rootNode.innerHTML = body();
      drawModern();
      mountNeuron();
      mountGrid();
      mountSquash();
      mountVanish();
      mountDead();
      mountOutput();
      mountUniversal();
    }

    render();
    U().onRedraw(render);
    ui.startProgressBar();
  }

  /* ════════════════════════════════════════════════════════════════
     GUIDE 2 · GRADIENT DESCENT
     ════════════════════════════════════════════════════════════════ */

  /* Points for fitting y ≈ w·x (one parameter). */
  const FIT = (() => {
    const xs = [0.5, 0.9, 1.3, 1.6, 2.0, 2.4, 2.9, 3.3, 3.7, 4.0];
    const noise = [0.3, -0.4, 0.2, 0.5, -0.3, 0.1, -0.5, 0.4, -0.2, 0.3];
    return xs.map((x, i) => ({ x, y: 1.6 * x + noise[i] }));
  })();
  const fitLoss = (w) => FIT.reduce((acc, p) => acc + (w * p.x - p.y) ** 2, 0) / FIT.length;
  const fitGrad = (w) => (2 * FIT.reduce((acc, p) => acc + p.x * (w * p.x - p.y), 0)) / FIT.length;
  const FIT_BEST = FIT.reduce((a, p) => a + p.x * p.y, 0) / FIT.reduce((a, p) => a + p.x * p.x, 0);

  const SURFACES = {
    bowl: { label: "a bowl", f: (x) => 0.5 * x * x, df: (x) => x, yDomain: [-0.3, 5], note: "Curvature 1, so steps converge for any η below 2." },
    double: { label: "two valleys", f: (x) => 0.25 * x ** 4 - x * x + 0.35 * x + 1.2, df: (x) => x ** 3 - 2 * x + 0.35, yDomain: [-0.5, 5], note: "The left valley is deeper. Where you start decides which one you end in." },
    plateau: { label: "a plateau", f: (x) => 2 - 2 * Math.exp(-0.5 * x * x) + 0.02 * x * x, df: (x) => 2 * x * Math.exp(-0.5 * x * x) + 0.04 * x, yDomain: [-0.2, 2.6], note: "Far from the minimum the slope is almost zero, so plain gradient descent crawls." },
  };

  function mountGradientGuide(rootNode) {
    const ui = UI();
    const { widget, range, segmented, mountWidget, fmtInt } = W();
    const { row, restoreInputs, frame, linePath, sparkline, niceTicks } = P();
    const { para, plain, deeper, takeaway, figure, chapter, guideToc, mathBlock, svgSlot, quiz, fmt } = ui;

    const chapters = [
      { id: "gd-loss", title: "Learning means making a number small", blurb: "the loss function" },
      { id: "gd-slope", title: "The slope points uphill", blurb: "derivatives and gradients" },
      { id: "gd-step", title: "Take a step downhill", blurb: "the update rule" },
      { id: "gd-rate", title: "The learning rate", blurb: "crawl, converge, oscillate, diverge" },
      { id: "gd-2d", title: "Many parameters: valleys", blurb: "gradients as arrows, zig-zags" },
      { id: "gd-momentum", title: "Momentum and Adam", blurb: "better ways to step" },
      { id: "gd-sgd", title: "Mini-batches", blurb: "stochastic gradient descent" },
      { id: "gd-practice", title: "In practice", blurb: "schedules, minima and saddles" },
      { id: "gd-recap", title: "Recap and self-check", blurb: "" },
    ];

    function body() {
      return `
        <div class="g-intro">
          ${para("Every neural network, from a two-neuron toy to a large language model, is trained by the same idea: measure how wrong the model is, work out which direction makes it less wrong, and take a small step that way. Repeat millions of times. That is <strong>gradient descent</strong>.")}
          ${para("This guide builds it up from fitting a single line to the optimisers used for real networks. It assumes only the idea of a slope; the <a href=\"./algorithm.html?id=activation-functions\">activation functions</a> guide comes before it and <a href=\"./algorithm.html?id=backpropagation\">backpropagation</a> after.")}
          ${guideToc(chapters)}
        </div>

        ${chapter("gd-loss", 1, "Basics", "Learning means making a number small", `
          ${plain(para("To train a model we first need a single number that says how bad it is: the <strong>loss</strong>. Here the model is a line through the origin, y = w·x, with one knob w. For each data point, the error is the vertical gap between the point and the line; square the gaps (so negatives don't cancel) and average them. A good w makes this <em>mean squared error</em> small. Learning is now a search problem: find the w at the bottom of the loss curve."))}
          ${widget("fit a line by hand", `
            ${range("w", "Slope w", -1, 4, 0.05, 0.6)}
            <div class="g-two">
              <div>${svgSlot("fitData", "0 0 360 260", "data points and the line")}</div>
              <div>${svgSlot("fitLoss", "0 0 360 260", "loss as a function of w")}</div>
            </div>
            <div data-out="fit"></div>`)}
          ${deeper("The math", `${mathBlock(String.raw`L(w) = \frac{1}{n}\sum_{i=1}^{n}\big(w\,x_i - y_i\big)^2`)}`)}
          ${takeaway("Training = choosing parameters that minimise a loss, a single number measuring how wrong the model is.")}
        `)}

        ${chapter("gd-slope", 2, "Basics", "The slope points uphill", `
          ${plain(para("Standing at some w, we can't see the whole curve (real models have billions of knobs). But we can measure the <strong>slope</strong> right where we stand: nudge w a tiny bit and see how the loss changes. A positive slope means the loss rises to the right, so we should go left; a negative slope means go right. With many parameters, the slopes form a vector called the <strong>gradient</strong>, which points in the steepest uphill direction."))}
          ${widget("measure the slope two ways", `
            ${range("w", "Slope w", -1, 4, 0.05, 0.6)}
            ${row("Nudge size h", segmented("h", [["1", "1"], ["0.1", "0.1"], ["0.01", "0.01"], ["0.001", "0.001"]], "0.1"))}
            <div class="g-two g-two-wide">
              <div>${svgSlot("slope", "0 0 460 260", "loss curve with the tangent line")}</div>
              <div data-out="slope"></div>
            </div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`\frac{dL}{dw} = \frac{2}{n}\sum_i x_i\,(w x_i - y_i)\;\approx\;\frac{L(w+h) - L(w-h)}{2h}`)}
            ${para("The formula on the left is exact; the nudge on the right is how you'd check it. Nudging costs two loss evaluations <em>per parameter</em>, which is hopeless for millions of parameters. Backpropagation gets the exact gradient for all of them at once.")}
          `)}
          ${takeaway("The derivative tells you which way is uphill and how steep it is; the gradient is that for every parameter at once.")}
        `)}

        ${chapter("gd-step", 3, "The algorithm", "Take a step downhill", `
          ${plain(para("Gradient descent is one line: <strong>new w = old w − η × slope</strong>. The minus sign walks downhill. The <strong>learning rate</strong> η scales the step. Steep places give big steps and flat places small ones, so the steps shrink automatically as you approach the bottom. Try it on three differently shaped curves."))}
          ${widget("gradient descent, one step at a time", `
            ${row("Curve", segmented("surface", Object.entries(SURFACES).map(([k, v]) => [k, v.label]), "bowl"))}
            ${range("x0", "Start x₀", -3, 3, 0.1, 2.6)}
            ${range("eta", "Learning rate η", 0.01, 2.2, 0.01, 0.3)}
            <div class="g-actions">
              <button type="button" class="button primary" data-action="step1">Take one step</button>
              <button type="button" class="button secondary" data-action="step10">10 steps</button>
              <button type="button" class="button secondary" data-action="reset">Reset</button>
            </div>
            <div class="g-two g-two-wide">
              <div>${svgSlot("descent", "0 0 460 280", "the curve and the path taken")}</div>
              <div data-out="descent"></div>
            </div>`)}
          ${deeper("The math", `${mathBlock(String.raw`x_{t+1} = x_t - \eta\, f'(x_t)`)}`)}
          ${takeaway("Gradient descent repeats x ← x − η·f′(x): a step against the slope, scaled by the learning rate.")}
        `)}

        ${chapter("gd-rate", 4, "The key knob", "The learning rate", `
          ${plain(para("The learning rate is the most important setting in training. Too small and you crawl. Just right and you glide in. A bit too big and you overshoot the bottom and bounce from side to side. Bigger still and every bounce lands higher than the last: training <strong>diverges</strong>. On a bowl with curvature k the boundary is exactly η = 2/k."))}
          ${figure("4.1", "Twelve steps on the bowl f(x) = ½x² from x = 2.6, for four learning rates.", `<div class="sl-multiples">${[0.1, 0.9, 1.8, 2.05].map((eta, i) => `<div><div class="tl-subhead">η = ${eta} · ${["crawls", "converges", "oscillates", "diverges"][i]}</div>${svgSlot(`rate${i}`, "0 0 240 180", `learning rate ${eta}`)}</div>`).join("")}</div>`, true)}
          ${deeper("The math", `
            ${mathBlock(String.raw`f(x) = \tfrac{k}{2}x^2 \;\Rightarrow\; x_{t+1} = (1 - \eta k)\,x_t,\qquad \text{converges} \iff |1-\eta k| < 1 \iff 0 < \eta < \tfrac{2}{k}`)}
            ${para("Each step multiplies the distance to the minimum by (1 − ηk). Between 1/k and 2/k that factor is negative: you jump across the bottom each step but still get closer. Above 2/k its size exceeds 1 and the distance grows.")}
          `)}
          ${takeaway("Below 2/curvature you converge (oscillating above 1/curvature); above it you diverge.")}
        `)}

        ${chapter("gd-2d", 5, "More parameters", "Many parameters: valleys", `
          ${plain(para("With two parameters the loss is a landscape, drawn here as contour lines (like a hiking map). The gradient is an arrow pointing straight uphill, at right angles to the contours. In a long narrow valley it points mostly across the valley, not along it, so gradient descent <strong>zig-zags</strong>. And the learning rate must suit the <em>steepest</em> direction, which makes progress along the gentle direction painfully slow. Real loss landscapes are like this in millions of dimensions."))}
          ${widget("gradient descent in a valley", `
            ${range("kappa", "Valley narrowness (steep ÷ gentle curvature)", 1, 25, 0.5, 10)}
            ${range("eta", "Learning rate η", 0.01, 0.25, 0.005, 0.17)}
            ${range("steps", "Steps", 1, 100, 1, 40)}
            <div class="g-two g-two-wide">
              <div>${svgSlot("valley", "0 0 460 320", "contours of the valley and the path")}</div>
              <div data-out="valley"></div>
            </div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`f(x,y) = \tfrac12\big(x^2 + \kappa\,y^2\big),\qquad \nabla f = \begin{bmatrix} x \\ \kappa y\end{bmatrix},\qquad \text{stable} \iff \eta < \tfrac{2}{\kappa}`)}
            ${para("Along x each step shrinks the distance by (1 − η); along y by (1 − ηκ). The largest safe η is set by κ, and then the x direction shrinks by only about 1 − 2/κ per step. The ratio κ is the <em>condition number</em>; the bigger it is, the slower plain gradient descent gets.")}
          `)}
          ${takeaway("In narrow valleys the gradient points across, not along, so plain gradient descent zig-zags and slows down.")}
        `)}

        ${chapter("gd-momentum", 6, "Better optimisers", "Momentum and Adam", `
          ${plain(para("Two fixes are used everywhere. <strong>Momentum</strong> treats the parameters like a heavy ball: each step adds the new gradient to a running velocity, so back-and-forth components cancel out while the steady downhill direction builds up speed. <strong>Adam</strong> gives every parameter its own step size, dividing each gradient by a running average of its recent size: directions with big gradients take careful steps, directions with small gradients take bold ones. Adam (and its variant AdamW) trains almost every large language model."))}
          ${widget("three optimisers, one valley", `
            ${range("kappa", "Valley narrowness", 1, 25, 0.5, 25)}
            ${range("eta", "Learning rate η (plain and momentum)", 0.01, 0.13, 0.005, 0.05)}
            ${range("beta", "Momentum β", 0, 0.95, 0.05, 0.6)}
            ${range("alpha", "Adam step size α", 0.01, 0.5, 0.01, 0.2)}
            <div class="g-two g-two-wide">
              <div>${svgSlot("optim", "0 0 460 320", "paths of three optimisers")}</div>
              <div data-out="optim"></div>
            </div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`\text{momentum: } v \leftarrow \beta v + \nabla f,\quad \theta \leftarrow \theta - \eta v`)}
            ${mathBlock(String.raw`\text{Adam: } m \leftarrow \beta_1 m + (1-\beta_1)g,\;\; s \leftarrow \beta_2 s + (1-\beta_2)g^2,\;\; \theta \leftarrow \theta - \alpha\,\frac{\hat m}{\sqrt{\hat s} + \epsilon}`)}
            ${para("m̂ and ŝ are m and s corrected for starting at zero (divided by 1 − β<sup>t</sup>). Adam's step is roughly α in every direction regardless of the gradient's size, which is why it needs little tuning (Kingma &amp; Ba, 2014).")}
          `)}
          ${takeaway("Momentum smooths the zig-zag and speeds up along the valley; Adam rescales each parameter's step so one setting works across very different scales.")}
        `)}

        ${chapter("gd-sgd", 7, "Scale", "Mini-batches: stochastic gradient descent", `
          ${plain(para("The true gradient averages over every training example, which for a large dataset means reading all of it before taking one step. Instead, networks use a <strong>mini-batch</strong>: a random handful of examples gives a noisy but cheap estimate of the gradient. Many cheap, noisy steps beat a few perfect ones, and the noise even helps escape poor spots. Here a line y = w·x + b is fitted to 32 points."))}
          ${widget("batch size vs noise", `
            ${row("Batch size", segmented("batch", [["1", "1 example"], ["4", "4"], ["32", "all 32 (full batch)"]], "4"))}
            ${range("eta", "Learning rate η", 0.01, 0.3, 0.01, 0.1)}
            ${range("steps", "Steps", 5, 150, 5, 60)}
            <div class="g-actions"><button type="button" class="button secondary" data-action="shuffle">Draw new random batches</button></div>
            <div class="g-two">
              <div>${svgSlot("sgdMap", "0 0 340 300", "loss landscape over w and b with the path")}</div>
              <div>${svgSlot("sgdLoss", "0 0 340 300", "loss over steps")}</div>
            </div>
            <div data-out="sgd"></div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`\theta \leftarrow \theta - \eta\,\frac{1}{|B|}\sum_{i\in B}\nabla_\theta \ell_i(\theta),\qquad B \subset \{1,\dots,n\}\ \text{random}`)}
            ${para("The mini-batch gradient is an unbiased estimate of the full gradient; its noise shrinks like 1/√|B|. One pass through the data is an <em>epoch</em>. Batches also suit GPUs, which process many examples in parallel.")}
          `)}
          ${takeaway("Real training uses noisy gradients from random mini-batches: far cheaper per step, and good enough.")}
        `)}

        ${chapter("gd-practice", 8, "In practice", "Schedules, minima and saddles", `
          ${plain(para("Two more facts from real training. First, the learning rate is rarely constant: it usually <strong>warms up</strong> from zero over the first steps (large early gradients are unreliable) and then <strong>decays</strong>, often along a cosine curve, so the final steps settle precisely. Second, in millions of dimensions, bad local minima turn out to be rare. The common trouble spots are <strong>saddle points</strong> and plateaus where the gradient is tiny in every direction; momentum and gradient noise help push through them."))}
          ${figure("8.1", "A typical learning-rate schedule: linear warm-up for 5% of training, then cosine decay to 10% of the peak.", svgSlot("schedule", "0 0 560 220", "learning rate over training"), true)}
          ${takeaway("Warm up, then decay the learning rate; in high dimensions saddles and plateaus matter more than local minima.")}
        `)}

        ${chapter("gd-recap", 9, "Recap", "Recap and self-check", `
          <ol class="g-recap">
            <li>Training minimises a <b>loss</b>; the <b>gradient</b> points uphill.</li>
            <li>Gradient descent: <b>θ ← θ − η∇L</b>.</li>
            <li>The <b>learning rate</b> must be below 2/curvature in the steepest direction.</li>
            <li>Narrow valleys make plain descent <b>zig-zag</b>; <b>momentum</b> and <b>Adam</b> fix that.</li>
            <li>Real training uses <b>mini-batches</b>, a <b>warm-up</b> and a <b>decaying</b> learning rate.</li>
          </ol>
          ${quiz([
            { q: "The loss went up and then shot to infinity after a few steps. What is the most likely cause?", a: "The learning rate is too large: above 2/curvature each step overshoots by more than it gains, so the error grows geometrically." },
            { q: "On f(x) = 2x², what is the largest learning rate that still converges?", a: "The curvature is k = 4, so η must be below 2/4 = 0.5." },
            { q: "Why does momentum help in a narrow valley?", a: "The across-valley components of successive gradients alternate in sign and cancel in the velocity, while the along-valley component keeps the same sign and accumulates." },
            { q: "What's the trade-off when you shrink the batch size?", a: "Each step is cheaper but its gradient is noisier (noise ∝ 1/√batch), so you usually need a smaller learning rate or more steps." },
          ])}
          ${nextCard("Next guide", "Backpropagation", "Where the gradient comes from: the chain rule run backwards through a network.", "./algorithm.html?id=backpropagation", "Continue to backpropagation →")}
        `)}
      `;
    }

    const state = {
      fit: { w: 0.6 },
      slope: { w: 0.6, h: "0.1" },
      descent: { surface: "bowl", x0: 2.6, eta: 0.3, path: null },
      valley: { kappa: 10, eta: 0.17, steps: 40 },
      optim: { kappa: 25, eta: 0.05, beta: 0.6, alpha: 0.2 },
      sgd: { batch: "4", eta: 0.1, steps: 60, seed: 1 },
    };
    const fmtOut = (key, value) => (typeof value === "number" ? String(Math.round(value * 1000) / 1000) : value);

    function drawLossCurve(svg, w, title) {
      const f = frame(svg, {
        xDomain: [-1, 4],
        yDomain: [0, 40],
        xTicks: [-1, 0, 1, 2, 3, 4].map((v) => [v, String(v)]),
        yTicks: [0, 10, 20, 30, 40].map((v) => [v, String(v)]),
        title,
        xLabel: "w",
        clipId: `gd-lc-${svg.getAttribute("data-fig")}`,
      });
      const pts = [];
      for (let i = 0; i <= 200; i += 1) pts.push([-1 + i * 0.025, fitLoss(-1 + i * 0.025)]);
      f.el("path", { d: linePath(pts, f.xs, f.ys), fill: "none", stroke: C().c, "stroke-width": 2.6, "clip-path": f.clip });
      f.el("circle", { cx: f.xs(FIT_BEST), cy: f.ys(fitLoss(FIT_BEST)), r: 4, fill: C().a });
      if (w !== null) f.el("circle", { cx: f.xs(w), cy: f.ys(Math.min(40, fitLoss(w))), r: 6, fill: C().b, stroke: C().ring, "stroke-width": 2 });
      return f;
    }

    function mountFit() {
      const host = hostOf(rootNode, "fit");
      restoreInputs(host, state.fit);
      mountWidget(host, state.fit, (h, s) => {
        const svg = h.querySelector('[data-fig="fitData"]');
        const f = frame(svg, { xDomain: [0, 4.5], yDomain: [-1, 9], xTicks: [0, 1, 2, 3, 4].map((v) => [v, String(v)]), yTicks: [0, 4, 8].map((v) => [v, String(v)]), title: `y = ${tn(s.w, 2)}·x`, clipId: "gd-fit-clip" });
        FIT.forEach((p) => {
          f.el("line", { x1: f.xs(p.x), y1: f.ys(p.y), x2: f.xs(p.x), y2: f.ys(s.w * p.x), stroke: C().danger, "stroke-width": 1.5, opacity: 0.7, "clip-path": f.clip });
        });
        f.el("line", { x1: f.xs(0), y1: f.ys(0), x2: f.xs(4.5), y2: f.ys(4.5 * s.w), stroke: C().b, "stroke-width": 2.6, "clip-path": f.clip });
        FIT.forEach((p) => f.el("circle", { cx: f.xs(p.x), cy: f.ys(p.y), r: 4.5, fill: C().a }));
        drawLossCurve(h.querySelector('[data-fig="fitLoss"]'), s.w, "Loss L(w)");
        h.querySelector('[data-out="fit"]').innerHTML = `
          <div class="g-metrics">
            <div><span>your w</span><b>${fmt(s.w, 2)}</b></div>
            <div><span>loss L(w)</span><b>${fmt(fitLoss(s.w), 3)}</b></div>
            <div><span>best possible w</span><b>${fmt(FIT_BEST, 3)}</b></div>
            <div><span>best loss</span><b>${fmt(fitLoss(FIT_BEST), 3)}</b></div>
          </div>
          <p class="caption">Red lines are the errors. The teal dot on the right marks the bottom of the loss curve; your orange dot moves as you change w.</p>`;
      }, { formatOutput: fmtOut });
    }

    function mountSlope() {
      const host = hostOf(rootNode, "slope");
      restoreInputs(host, state.slope);
      mountWidget(host, state.slope, (h, s) => {
        const hh = Number(s.h);
        const exact = fitGrad(s.w);
        const approx = (fitLoss(s.w + hh) - fitLoss(s.w - hh)) / (2 * hh);
        const f = drawLossCurve(h.querySelector('[data-fig="slope"]'), s.w, "Loss with the tangent at your w");
        const L0 = fitLoss(s.w);
        f.el("line", { x1: f.xs(s.w - 1), y1: f.ys(L0 - exact), x2: f.xs(s.w + 1), y2: f.ys(L0 + exact), stroke: C().b, "stroke-width": 2, "stroke-dasharray": "6 4", "clip-path": f.clip });
        [s.w - hh, s.w + hh].forEach((x) => f.el("circle", { cx: f.xs(x), cy: f.ys(Math.min(40, fitLoss(x))), r: 3.5, fill: C().neutral, "clip-path": f.clip }));
        ui.arrowMarker(h.querySelector('[data-fig="slope"]'), "gd-dir", C().a);
        const dir = exact > 0 ? -1 : 1;
        f.el("line", { x1: f.xs(s.w), y1: f.ys(0) - 12, x2: f.xs(s.w + dir * 0.6), y2: f.ys(0) - 12, stroke: C().a, "stroke-width": 3, "marker-end": "url(#gd-dir)" });
        h.querySelector('[data-out="slope"]').innerHTML = `
          <div class="g-metrics">
            <div><span>exact slope dL/dw</span><b>${fmt(exact, 4)}</b></div>
            <div><span>nudge estimate (h = ${s.h})</span><b>${fmt(approx, 4)}</b></div>
            <div><span>difference</span><b>${Math.abs(exact - approx) < 1e-9 ? "≈ 0" : Math.abs(exact - approx).toExponential(1)}</b></div>
          </div>
          <p class="caption">The slope is ${exact > 0 ? "positive: the loss rises to the right, so step <b>left</b>" : exact < 0 ? "negative: the loss falls to the right, so step <b>right</b>" : "zero: you're at the bottom"} (teal arrow). Grey dots are the two nudged points. For this quadratic loss the two-sided nudge is exact for any h; for curvier functions smaller h is more accurate, until rounding errors take over.</p>`;
      }, { formatOutput: fmtOut });
    }

    function mountDescent() {
      const host = hostOf(rootNode, "descent");
      restoreInputs(host, state.descent);
      const run = (s, n) => {
        const surf = SURFACES[s.surface];
        if (!s.path) s.path = [s.x0];
        for (let k = 0; k < n; k += 1) {
          const x = s.path[s.path.length - 1];
          if (!Number.isFinite(x) || Math.abs(x) > 1e6) break;
          s.path.push(x - s.eta * surf.df(x));
        }
      };
      mountWidget(host, state.descent, (h, s) => {
        const sig2 = `${s.surface}|${s.x0}|${s.eta}`;
        if (s.sig !== sig2) {
          s.sig = sig2;
          s.path = [s.x0];
        }
        const surf = SURFACES[s.surface];
        const svg = h.querySelector('[data-fig="descent"]');
        const f = plotCurves(svg, [{ fn: surf.f, color: C().c }], { xDomain: [-3.2, 3.2], yDomain: surf.yDomain, title: `f(x) on ${surf.label}`, clipId: "gd-desc-clip" });
        ui.arrowMarker(svg, "gd-step-arrow", C().b);
        const path = s.path;
        for (let i = 1; i < path.length; i += 1) {
          const a = path[i - 1];
          const b = path[i];
          if (!Number.isFinite(b) || Math.abs(b) > 50) break;
          f.el("line", { x1: f.xs(a), y1: f.ys(surf.f(a)), x2: f.xs(b), y2: f.ys(surf.f(b)), stroke: C().b, "stroke-width": 1.8, opacity: 0.8, "marker-end": "url(#gd-step-arrow)", "clip-path": f.clip });
        }
        path.forEach((x, i) => {
          if (!Number.isFinite(x) || Math.abs(x) > 50) return;
          f.el("circle", { cx: f.xs(x), cy: f.ys(surf.f(x)), r: i === path.length - 1 ? 6 : 3.5, fill: i === 0 ? C().neutral : C().b, "clip-path": f.clip });
        });
        const x = path[path.length - 1];
        const ok = Number.isFinite(x) && Math.abs(x) < 1e6;
        const losses = path.filter((v) => Number.isFinite(v) && Math.abs(v) < 1e6).map(surf.f);
        const diverged = !ok || (path.length > 3 && Math.abs(x) > 10);
        h.querySelector('[data-out="descent"]').innerHTML = `
          <div class="g-metrics">
            <div><span>step</span><b>${path.length - 1}</b></div>
            <div><span>x</span><b>${ok ? fmt(x, 4) : "∞"}</b></div>
            <div><span>f(x)</span><b>${ok ? fmt(surf.f(x), 4) : "∞"}</b></div>
            <div><span>slope f′(x)</span><b>${ok ? fmt(surf.df(x), 4) : "—"}</b></div>
            <div><span>next step −η·f′(x)</span><b>${ok ? fmt(-s.eta * surf.df(x), 4) : "—"}</b></div>
          </div>
          ${losses.length > 1 ? `<div class="tl-subhead">f(x) after each step</div>${sparkline(losses.map((v) => Math.min(v, 50)), C().b)}` : ""}
          <p class="caption">${diverged ? "<b>Diverging:</b> every step overshoots further. Lower η. " : ""}${surf.note}</p>`;
      }, {
        formatOutput: fmtOut,
        step1: (s) => run(s, 1),
        step10: (s) => run(s, 10),
        reset: (s) => {
          s.path = [s.x0];
        },
      });
    }

    function drawRates() {
      [0.1, 0.9, 1.8, 2.05].forEach((eta, i) => {
        const svg = rootNode.querySelector(`[data-fig="rate${i}"]`);
        const f = frame(svg, { xDomain: [-4.2, 4.2], yDomain: [-0.3, 9], xTicks: [[-4, "−4"], [0, "0"], [4, "4"]], yTicks: [], pad: { left: 14, right: 10, top: 12, bottom: 28 }, clipId: `gd-rate-${i}` });
        const pts = [];
        for (let k = 0; k <= 160; k += 1) pts.push([-4.2 + k * 0.0525, 0.5 * (-4.2 + k * 0.0525) ** 2]);
        f.el("path", { d: linePath(pts, f.xs, f.ys), fill: "none", stroke: C().c, "stroke-width": 2 });
        let x = 2.6;
        const xs = [x];
        for (let k = 0; k < 12; k += 1) {
          x -= eta * x;
          xs.push(x);
        }
        for (let k = 1; k < xs.length; k += 1) {
          f.el("line", { x1: f.xs(xs[k - 1]), y1: f.ys(0.5 * xs[k - 1] ** 2), x2: f.xs(xs[k]), y2: f.ys(0.5 * xs[k] ** 2), stroke: C().b, "stroke-width": 1.4, opacity: 0.85, "clip-path": f.clip });
        }
        xs.forEach((v) => f.el("circle", { cx: f.xs(v), cy: f.ys(0.5 * v * v), r: 2.8, fill: C().b, "clip-path": f.clip }));
      });
    }

    /* Optimiser paths on f = ½(x² + κy²). */
    function valleyPath(kind, o, steps) {
      let p = [-2.6, 1.6];
      const path = [p.slice()];
      let v = [0, 0];
      let m = [0, 0];
      let s2 = [0, 0];
      for (let t = 1; t <= steps; t += 1) {
        const g = [p[0], o.kappa * p[1]];
        if (kind === "gd") p = [p[0] - o.eta * g[0], p[1] - o.eta * g[1]];
        else if (kind === "momentum") {
          v = [o.beta * v[0] + g[0], o.beta * v[1] + g[1]];
          p = [p[0] - o.eta * v[0], p[1] - o.eta * v[1]];
        } else {
          m = m.map((mi, i) => 0.9 * mi + 0.1 * g[i]);
          s2 = s2.map((si, i) => 0.999 * si + 0.001 * g[i] * g[i]);
          p = p.map((pi, i) => pi - (o.alpha * (m[i] / (1 - 0.9 ** t))) / (Math.sqrt(s2[i] / (1 - 0.999 ** t)) + 1e-8));
        }
        path.push(p.slice());
        if (Math.abs(p[0]) > 1e4 || Math.abs(p[1]) > 1e4) break;
      }
      return path;
    }
    const valleyF = (p, k) => 0.5 * (p[0] ** 2 + k * p[1] ** 2);
    const stepsTo = (path, k, tol = 1e-3) => {
      const i = path.findIndex((p) => valleyF(p, k) < tol);
      return i < 0 ? null : i;
    };

    function drawValley(svg, kappa, paths) {
      const f = frame(svg, { xDomain: [-3, 3], yDomain: [-2, 2], xTicks: [-3, -2, -1, 0, 1, 2, 3].map((v) => [v, String(v)]), yTicks: [-2, -1, 0, 1, 2].map((v) => [v, String(v)]), pad: { left: 36, right: 14, top: 14, bottom: 30 }, clipId: `gd-v-${svg.getAttribute("data-fig")}` });
      [0.02, 0.1, 0.3, 0.7, 1.4, 2.4, 3.6].forEach((c) => {
        const rx = Math.sqrt(2 * c);
        const ry = Math.sqrt((2 * c) / kappa);
        f.el("ellipse", { cx: f.xs(0), cy: f.ys(0), rx: f.xs(rx) - f.xs(0), ry: f.ys(0) - f.ys(ry), fill: "none", stroke: C().faint, "stroke-width": 1.2, "clip-path": f.clip });
      });
      f.el("circle", { cx: f.xs(0), cy: f.ys(0), r: 4, fill: C().ink, opacity: 0.6 });
      paths.forEach(({ path, color }) => {
        const pts = path.map(([x, y]) => [clamp(x, -50, 50), clamp(y, -50, 50)]);
        f.el("path", { d: linePath(pts, f.xs, f.ys), fill: "none", stroke: color, "stroke-width": 2, "clip-path": f.clip, opacity: 0.9 });
        pts.forEach((p, i) => {
          if (i % 2 === 0 || i === pts.length - 1) f.el("circle", { cx: f.xs(p[0]), cy: f.ys(p[1]), r: 2.5, fill: color, "clip-path": f.clip });
        });
      });
      f.el("circle", { cx: f.xs(-2.6), cy: f.ys(1.6), r: 5, fill: C().neutral });
      return f;
    }

    function mountValley() {
      const host = hostOf(rootNode, "valley");
      restoreInputs(host, state.valley);
      mountWidget(host, state.valley, (h, s) => {
        const path = valleyPath("gd", s, s.steps);
        drawValley(h.querySelector('[data-fig="valley"]'), s.kappa, [{ path, color: C().b }]);
        const last = path[path.length - 1];
        const limit = 2 / s.kappa;
        const n = stepsTo(valleyPath("gd", s, 2000), s.kappa);
        h.querySelector('[data-out="valley"]').innerHTML = `
          <div class="g-metrics">
            <div><span>loss after ${path.length - 1} steps</span><b>${Number.isFinite(valleyF(last, s.kappa)) && valleyF(last, s.kappa) < 1e6 ? fmt(valleyF(last, s.kappa), 4) : "∞"}</b></div>
            <div><span>largest stable η = 2/κ</span><b>${fmt(limit, 3)}</b></div>
            <div><span>steps to loss &lt; 0.001</span><b>${s.eta >= limit ? "never (diverges)" : n === null ? "> 2,000" : fmtInt(n)}</b></div>
          </div>
          <p class="caption">${s.eta >= limit ? "η is above 2/κ: the steep direction explodes. " : s.eta > 1 / s.kappa ? "η is between 1/κ and 2/κ: the path zig-zags across the valley but still converges. " : "η is small enough not to zig-zag, but now progress along the valley floor is slow. "}Make the valley narrower and watch the number of steps climb.</p>`;
      }, { formatOutput: fmtOut });
    }

    function mountOptim() {
      const host = hostOf(rootNode, "optim");
      restoreInputs(host, state.optim);
      mountWidget(host, state.optim, (h, s) => {
        const kinds = [["gd", "plain", C().neutral], ["momentum", "momentum", C().b], ["adam", "Adam", C().a]];
        drawValley(h.querySelector('[data-fig="optim"]'), s.kappa, kinds.map(([k, , color]) => ({ path: valleyPath(k, s, 40), color })));
        const counts = kinds.map(([k]) => stepsTo(valleyPath(k, s, 3000), s.kappa));
        const best = Math.min(...counts.filter((c) => c !== null));
        h.querySelector('[data-out="optim"]').innerHTML = `
          <div class="legend">${kinds.map(([, label, color]) => `<span><i style="background:${color}"></i> ${label}</span>`).join("")}</div>
          <table class="g-mini"><tr><th>optimiser</th><th>steps to loss &lt; 0.001</th></tr>${kinds
            .map(([, label], i) => `<tr><td>${label}</td><td class="mono">${counts[i] === null ? (label === "plain" && s.eta >= 2 / s.kappa ? "diverges" : "> 3,000") : `${fmtInt(counts[i])}${counts[i] === best ? " ✓" : ""}`}</td></tr>`)
            .join("")}</table>
          <p class="caption">The first 40 steps of each are drawn. Momentum builds speed along the valley floor and needs far fewer steps; set β = 0 and it becomes plain gradient descent, push β towards 0.95 and it overshoots and circles. Adam's steps are about α long in every direction, so it heads diagonally at first. On a simple bowl like this it isn't the fastest; its strength is working well with little tuning when different parameters have wildly different scales, as in large networks.</p>`;
      }, { formatOutput: fmtOut });
    }

    /* Line fit y = w·x + b on 32 points, for the mini-batch chapter. */
    const LINE = (() => {
      const rng = M().seeded(5);
      return Array.from({ length: 32 }, () => {
        const x = -1 + 3 * rng();
        return { x, y: 2 * x + 1 + 0.6 * M().gaussian(rng) };
      });
    })();
    const lineLoss = (w, b, pts = LINE) => pts.reduce((acc, p) => acc + (w * p.x + b - p.y) ** 2, 0) / pts.length;

    function mountSgd() {
      const host = hostOf(rootNode, "sgd");
      restoreInputs(host, state.sgd);
      mountWidget(host, state.sgd, (h, s) => {
        const bs = Number(s.batch);
        const rng = M().seeded(100 + s.seed);
        let w = -0.5;
        let b = 3.5;
        const path = [[w, b]];
        const losses = [lineLoss(w, b)];
        let order = [];
        for (let t = 0; t < s.steps; t += 1) {
          if (order.length < bs) {
            const idx = LINE.map((_, i) => i);
            for (let k = idx.length - 1; k > 0; k -= 1) {
              const j = Math.floor(rng() * (k + 1));
              [idx[k], idx[j]] = [idx[j], idx[k]];
            }
            order = order.concat(idx);
          }
          const batch = order.splice(0, bs).map((i) => LINE[i]);
          let gw = 0;
          let gb = 0;
          batch.forEach((p) => {
            const r = w * p.x + b - p.y;
            gw += (2 * r * p.x) / batch.length;
            gb += (2 * r) / batch.length;
          });
          w -= s.eta * gw;
          b -= s.eta * gb;
          path.push([w, b]);
          losses.push(lineLoss(w, b));
        }
        /* Loss map. */
        const svg = h.querySelector('[data-fig="sgdMap"]');
        const f = frame(svg, { xDomain: [-1, 4], yDomain: [-1, 4], xTicks: [-1, 0, 1, 2, 3, 4].map((v) => [v, String(v)]), yTicks: [-1, 0, 1, 2, 3, 4].map((v) => [v, String(v)]), title: "Loss over (w, b)", xLabel: "w", pad: { left: 34, right: 10 }, clipId: "gd-sgd-clip" });
        const R = 34;
        const vals = [];
        for (let j = 0; j < R; j += 1) for (let i = 0; i < R; i += 1) vals.push(Math.log(lineLoss(-1 + (5 * (i + 0.5)) / R, -1 + (5 * (j + 0.5)) / R)));
        const lo = Math.min(...vals);
        const hi = Math.max(...vals);
        for (let j = 0; j < R; j += 1) {
          for (let i = 0; i < R; i += 1) {
            const v = (vals[j * R + i] - lo) / (hi - lo);
            const x0 = f.xs(-1 + (5 * i) / R);
            const y1 = f.ys(-1 + (5 * (j + 1)) / R);
            f.el("rect", { x: x0, y: y1, width: f.xs(-1 + (5 * (i + 1)) / R) - x0, height: f.ys(-1 + (5 * j) / R) - y1, fill: ui.rgba(C().c, 0.55 * (1 - v) + 0.03), "shape-rendering": "crispEdges" });
          }
        }
        f.el("path", { d: linePath(path.map(([pw, pb]) => [clamp(pw, -5, 8), clamp(pb, -5, 8)]), f.xs, f.ys), fill: "none", stroke: C().b, "stroke-width": 1.8, "clip-path": f.clip });
        f.el("circle", { cx: f.xs(path[0][0]), cy: f.ys(path[0][1]), r: 5, fill: C().neutral });
        f.el("circle", { cx: f.xs(clamp(w, -5, 8)), cy: f.ys(clamp(b, -5, 8)), r: 5, fill: C().b, stroke: C().ring, "stroke-width": 2, "clip-path": f.clip });
        /* Loss curve. */
        const lsvg = h.querySelector('[data-fig="sgdLoss"]');
        const top = Math.min(30, Math.max(...losses.filter(Number.isFinite)));
        const lf = frame(lsvg, { xDomain: [0, s.steps], yDomain: [0, top], xTicks: niceTicks(0, s.steps, 4).map((v) => [v, String(v)]), yTicks: niceTicks(0, top, 4).map((v) => [v, String(v)]), title: "Full-data loss after each step", xLabel: "step", pad: { left: 40, right: 10 }, clipId: "gd-sgd-l" });
        lf.el("path", { d: linePath(losses.map((v, i) => [i, Math.min(v, top * 2)]), lf.xs, lf.ys), fill: "none", stroke: C().b, "stroke-width": 2, "clip-path": lf.clip });
        const final = losses[losses.length - 1];
        h.querySelector('[data-out="sgd"]').innerHTML = `
          <div class="g-metrics">
            <div><span>final loss</span><b>${Number.isFinite(final) && final < 1e6 ? fmt(final, 3) : "∞"}</b></div>
            <div><span>examples read</span><b>${fmtInt(s.steps * bs)}</b></div>
            <div><span>epochs</span><b>${fmt((s.steps * bs) / LINE.length, 1)}</b></div>
          </div>
          <p class="caption">With batch size 1 the path wanders and the loss jitters, but each step reads only one example. The full batch is smooth but reads all 32 every step. Compare final losses for the same number of <em>examples read</em>, not steps.</p>`;
      }, {
        formatOutput: fmtOut,
        shuffle: (s) => {
          s.seed += 1;
        },
      });
    }

    function drawSchedule() {
      const svg = rootNode.querySelector('[data-fig="schedule"]');
      const f = frame(svg, { xDomain: [0, 100], yDomain: [0, 1.1], xTicks: [0, 25, 50, 75, 100].map((v) => [v, `${v}%`]), yTicks: [0, 0.5, 1].map((v) => [v, v === 1 ? "peak" : v === 0 ? "0" : "½"]), title: "Learning rate during training", xLabel: "training progress", pad: { left: 50 } });
      const lr = (p) => (p < 5 ? p / 5 : 0.1 + 0.9 * 0.5 * (1 + Math.cos((Math.PI * (p - 5)) / 95)));
      const pts = [];
      for (let i = 0; i <= 400; i += 1) pts.push([i / 4, lr(i / 4)]);
      f.el("path", { d: linePath(pts, f.xs, f.ys), fill: "none", stroke: C().a, "stroke-width": 2.6 });
      ui.svgText(svg, f.xs(5) + 6, f.ys(1) - 6, "end of warm-up");
    }

    function render() {
      rootNode.innerHTML = body();
      drawRates();
      drawSchedule();
      mountFit();
      mountSlope();
      mountDescent();
      mountValley();
      mountOptim();
      mountSgd();
    }

    render();
    U().onRedraw(render);
    ui.startProgressBar();
  }

  /* ════════════════════════════════════════════════════════════════
     GUIDE 3 · BACKPROPAGATION
     ════════════════════════════════════════════════════════════════ */

  /* A 2-2-1 sigmoid network with fixed starting weights. */
  const TINY_START = { W1: [[0.15, 0.2], [0.25, 0.3]], b1: [0.35, 0.35], W2: [0.4, 0.45], b2: 0.6 };
  const cloneNet = (n) => JSON.parse(JSON.stringify(n));

  function tinyForward(n, x) {
    const z1 = [0, 1].map((j) => n.W1[j][0] * x[0] + n.W1[j][1] * x[1] + n.b1[j]);
    const a1 = z1.map(sig);
    const z2 = n.W2[0] * a1[0] + n.W2[1] * a1[1] + n.b2;
    const a2 = sig(z2);
    return { z1, a1, z2, a2 };
  }

  function tinyBackward(n, x, y, f, bug = "none") {
    const L = 0.5 * (f.a2 - y) ** 2;
    const d2 = (f.a2 - y) * f.a2 * (1 - f.a2);
    const gW2 = f.a1.map((a) => d2 * a);
    const gb2 = d2;
    const d1 = [0, 1].map((j) => {
      const v = bug === "swap" ? n.W2[1 - j] : n.W2[j];
      return v * d2 * (bug === "noderiv" ? 1 : f.a1[j] * (1 - f.a1[j]));
    });
    const gW1 = d1.map((d) => [d * x[0], d * x[1]]);
    return { L, d2, gW2, gb2, d1, gW1, gb1: d1.slice() };
  }

  const tinyLoss = (n, x, y) => 0.5 * (tinyForward(n, x).a2 - y) ** 2;

  function tinyUpdate(n, g, lr) {
    return {
      W1: n.W1.map((r, j) => r.map((w, k) => w - lr * g.gW1[j][k])),
      b1: n.b1.map((b, j) => b - lr * g.gb1[j]),
      W2: n.W2.map((w, j) => w - lr * g.gW2[j]),
      b2: n.b2 - lr * g.gb2,
    };
  }

  /* Every parameter of the tiny net, as [label, get, set]. */
  const TINY_PARAMS = [
    ["w₁₁", (n) => n.W1[0][0], (n, v) => { n.W1[0][0] = v; }, (g) => g.gW1[0][0]],
    ["w₁₂", (n) => n.W1[0][1], (n, v) => { n.W1[0][1] = v; }, (g) => g.gW1[0][1]],
    ["w₂₁", (n) => n.W1[1][0], (n, v) => { n.W1[1][0] = v; }, (g) => g.gW1[1][0]],
    ["w₂₂", (n) => n.W1[1][1], (n, v) => { n.W1[1][1] = v; }, (g) => g.gW1[1][1]],
    ["b₁", (n) => n.b1[0], (n, v) => { n.b1[0] = v; }, (g) => g.gb1[0]],
    ["b₂", (n) => n.b1[1], (n, v) => { n.b1[1] = v; }, (g) => g.gb1[1]],
    ["v₁", (n) => n.W2[0], (n, v) => { n.W2[0] = v; }, (g) => g.gW2[0]],
    ["v₂", (n) => n.W2[1], (n, v) => { n.W2[1] = v; }, (g) => g.gW2[1]],
    ["c", (n) => n.b2, (n, v) => { n.b2 = v; }, (g) => g.gb2],
  ];

  /* XOR with a 2 → H → 1 network. */
  const XOR = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];

  function xorInit(H, seed) {
    const rng = M().seeded(seed);
    const g = () => M().gaussian(rng);
    return {
      H,
      W1: Array.from({ length: H }, () => [g() * 1.4, g() * 1.4]),
      b1: Array.from({ length: H }, () => g() * 0.6),
      w2: Array.from({ length: H }, () => g() * 1.0),
      b2: 0,
      epochs: 0,
      history: [],
    };
  }

  function xorForward(net, x, act) {
    const z = net.W1.map((r, j) => r[0] * x[0] + r[1] * x[1] + net.b1[j]);
    const hdn = z.map(act.f);
    const out = sig(hdn.reduce((acc, v, j) => acc + v * net.w2[j], net.b2));
    return { z, hdn, out };
  }

  function xorEpoch(net, act, lr) {
    const H = net.H;
    const gW1 = Array.from({ length: H }, () => [0, 0]);
    const gb1 = new Array(H).fill(0);
    const gw2 = new Array(H).fill(0);
    let gb2 = 0;
    let loss = 0;
    XOR.forEach(([x1, x2, t]) => {
      const { z, hdn, out } = xorForward(net, [x1, x2], act);
      const p = clamp(out, 1e-12, 1 - 1e-12);
      loss -= (t * Math.log(p) + (1 - t) * Math.log(1 - p)) / 4;
      const dz = (out - t) / 4;
      gb2 += dz;
      for (let j = 0; j < H; j += 1) {
        gw2[j] += dz * hdn[j];
        const dh = dz * net.w2[j] * act.df(z[j]);
        gW1[j][0] += dh * x1;
        gW1[j][1] += dh * x2;
        gb1[j] += dh;
      }
    });
    for (let j = 0; j < H; j += 1) {
      net.W1[j][0] -= lr * gW1[j][0];
      net.W1[j][1] -= lr * gW1[j][1];
      net.b1[j] -= lr * gb1[j];
      net.w2[j] -= lr * gw2[j];
    }
    net.b2 -= lr * gb2;
    net.epochs += 1;
    return loss;
  }

  /* Average weight-gradient norm per layer for a deep, narrow network. */
  function depthGradients(depth, actKey, init) {
    const act = ACTS[actKey];
    const n = 8;
    const rng = M().seeded(17);
    const std = init === "small" ? 0.1 : init === "xavier" ? Math.sqrt(1 / n) : Math.sqrt(2 / n);
    const Ws = Array.from({ length: depth }, () => Array.from({ length: n }, () => Array.from({ length: n }, () => M().gaussian(rng) * std)));
    const norms = new Array(depth).fill(0);
    const samples = 24;
    for (let s = 0; s < samples; s += 1) {
      const x = Array.from({ length: n }, () => M().gaussian(rng));
      const as = [x];
      const zs = [];
      Ws.forEach((Wl) => {
        const prev = as[as.length - 1];
        const z = Wl.map((r) => r.reduce((acc, w, k) => acc + w * prev[k], 0));
        zs.push(z);
        as.push(z.map(act.f));
      });
      let up = new Array(n).fill(1);
      for (let l = depth - 1; l >= 0; l -= 1) {
        const delta = up.map((u, i) => u * act.df(zs[l][i]));
        const prev = as[l];
        let sq = 0;
        delta.forEach((d) => prev.forEach((a) => {
          sq += (d * a) ** 2;
        }));
        norms[l] += Math.sqrt(sq) / samples;
        up = prev.map((_, k) => delta.reduce((acc, d, i) => acc + d * Ws[l][i][k], 0));
      }
    }
    return norms;
  }

  function mountBackpropGuide(rootNode, ctx) {
    const ui = UI();
    const { widget, range, segmented, mountWidget, fmtInt, fmtBig } = W();
    const { row, restoreInputs, frame, linePath, sparkline, niceTicks } = P();
    const { para, plain, deeper, takeaway, figure, chapter, guideToc, mathBlock, svgSlot, quiz, fmt, heatmap } = ui;

    const chapters = [
      { id: "bp-why", title: "The question backprop answers", blurb: "and why nudging every weight is too slow" },
      { id: "bp-chain", title: "The chain rule on one neuron", blurb: "multiply local slopes along the path" },
      { id: "bp-graph", title: "Computation graphs", blurb: "forward values, backward gradients" },
      { id: "bp-tiny", title: "A whole network, step by step", blurb: "every number of one training step" },
      { id: "bp-matrix", title: "The matrix form", blurb: "how libraries actually compute it" },
      { id: "bp-train", title: "Training: repeat", blurb: "a network learns XOR" },
      { id: "bp-depth", title: "Gradients through depth", blurb: "vanishing gradients, measured" },
      { id: "bp-check", title: "Checking your gradients", blurb: "catch a buggy backward pass" },
      { id: "bp-play", title: "Playground", blurb: "design your own network" },
      { id: "bp-recap", title: "Recap and self-check", blurb: "" },
    ];

    function body() {
      return `
        <div class="g-intro">
          ${para("Gradient descent needs the slope of the loss with respect to <em>every</em> weight. A modern network has millions or billions of weights. <strong>Backpropagation</strong> computes all of those slopes in a single backward sweep that costs about as much as running the network forwards twice. It is nothing more than the chain rule from school calculus, applied in a clever order.")}
          ${para("It follows <a href=\"./algorithm.html?id=gradient-descent\">gradient descent</a>. One small network runs through the middle of this guide, and you'll see every number of a full training step.")}
          ${guideToc(chapters)}
        </div>

        ${chapter("bp-why", 1, "Motivation", "The question backprop answers", `
          ${plain(para("For each weight we want to know: <em>if I nudge this weight a little, how much does the loss change?</em> The obvious way is to actually nudge it and rerun the network, once per weight. With a billion weights that's a billion forward passes for a single training step. Backpropagation gets the exact answer for all weights with one forward pass and one backward pass."))}
          ${widget("nudging vs backpropagation", `
            ${row("Network size", segmented("size", [["1e4", "10 K weights"], ["1e6", "1 M"], ["1.24e8", "124 M (GPT-2)"], ["8e9", "8 B (Llama 3 8B)"]], "1e6"))}
            <div data-out="why"></div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`\text{nudging: } \frac{\partial L}{\partial w_i} \approx \frac{L(\mathbf{w} + h\mathbf{e}_i) - L(\mathbf{w})}{h}\ \text{ for each } i\quad(N+1 \text{ passes});\qquad \text{backprop: } \approx 3 \text{ passes' worth of work}`)}
            ${para("The backward pass costs roughly twice the forward pass (each weight is used once to pass the signal back and once to form its own gradient), so a full step is about three forward passes, whatever N is.")}
          `)}
          ${takeaway("Backprop gives every weight's gradient for the price of about three forward passes, instead of one pass per weight.")}
        `)}

        ${chapter("bp-chain", 2, "The idea", "The chain rule on one neuron", `
          ${plain(para("Follow one neuron: the weight w multiplies the input, the bias is added to give z, the sigmoid turns z into the output a, and the loss compares a with the target y. A change in w ripples through each stage. The <strong>chain rule</strong> says the total effect is the product of each stage's local slope. Backprop computes these products starting from the loss and walking <em>backwards</em>, so each one reuses the one before."))}
          ${widget("one neuron, forwards and backwards", `
            ${range("x", "Input x", -2, 2, 0.1, 1.5)}
            ${range("w", "Weight w", -2, 2, 0.1, 0.8)}
            ${range("b", "Bias b", -2, 2, 0.1, -0.2)}
            ${range("y", "Target y", 0, 1, 0.05, 0.2)}
            ${svgSlot("chain", "0 0 640 210", "computation path of one neuron")}
            <div data-out="chain"></div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`\frac{\partial L}{\partial w} = \underbrace{\frac{\partial L}{\partial a}}_{a - y}\cdot\underbrace{\frac{\partial a}{\partial z}}_{a(1-a)}\cdot\underbrace{\frac{\partial z}{\partial w}}_{x},\qquad \frac{\partial L}{\partial b} = \frac{\partial L}{\partial a}\cdot\frac{\partial a}{\partial z}\cdot 1`)}
            ${para("The shared part ∂L/∂a · ∂a/∂z = ∂L/∂z is called the neuron's <strong>error signal δ</strong>. Every gradient at this neuron is δ times something local. Computing δ once and reusing it is the whole trick.")}
          `)}
          ${takeaway("The chain rule multiplies local slopes along the path; backprop computes the shared part δ once and reuses it.")}
        `)}

        ${chapter("bp-graph", 3, "The idea", "Computation graphs", `
          ${plain(para("Any calculation can be drawn as a graph of simple operations. Run it <strong>forwards</strong> to get the value, storing each intermediate result. Then run it <strong>backwards</strong>: each node receives “how much the output cares about me” from above and passes it to its inputs, multiplied by its own local slope. Two rules cover a lot: an <strong>add</strong> node passes the gradient unchanged to both inputs; a <strong>multiply</strong> node sends each input the gradient times the <em>other</em> input. Libraries like PyTorch build this graph automatically (<em>autograd</em>)."))}
          ${widget("f = (x + y) · z", `
            <div class="g-inputs">
              <label><span>x</span><input type="number" data-input="x" step="1" /></label>
              <label><span>y</span><input type="number" data-input="y" step="1" /></label>
              <label><span>z</span><input type="number" data-input="z" step="1" /></label>
            </div>
            ${svgSlot("gates", "0 0 560 230", "add and multiply gates with forward and backward values")}
            <div data-out="gates"></div>`)}
          ${takeaway("Forward: compute and store. Backward: each node multiplies the incoming gradient by its local slope and passes it on.")}
        `)}

        ${chapter("bp-tiny", 4, "Worked example", "A whole network, step by step", `
          ${plain(para("Now a real network: 2 inputs, 2 hidden sigmoid neurons, 1 sigmoid output, squared-error loss. Step through one complete training step: the forward pass, the loss, the error signals flowing back, the gradients, and the update. Every number is computed from the inputs and target you choose. Teal values flow forwards; orange values flow backwards."))}
          ${widget("one training step of a 2-2-1 network", `
            <div class="g-inputs">
              <label><span>input x₁</span><input type="number" data-input="x1" step="0.05" /></label>
              <label><span>input x₂</span><input type="number" data-input="x2" step="0.05" /></label>
              <label><span>target y</span><input type="number" data-input="y" step="0.05" min="0" max="1" /></label>
              <label><span>learning rate η</span><input type="number" data-input="lr" step="0.1" /></label>
            </div>
            <div class="g-actions">
              <button type="button" class="button secondary" data-action="prev">← Previous</button>
              <button type="button" class="button primary" data-action="next">Next step →</button>
              <button type="button" class="button secondary" data-action="commit">Apply update, start again</button>
              <button type="button" class="button secondary" data-action="reset">Reset weights</button>
            </div>
            ${svgSlot("tiny", "0 0 680 330", "the network with forward and backward values")}
            <div data-out="tiny"></div>`)}
          ${takeaway("A training step is: forward pass, loss, δ at the output, δ pulled back through the weights, gradient = δ × input, update.")}
        `)}

        ${chapter("bp-matrix", 5, "Real code", "The matrix form", `
          ${plain(para("Libraries don't loop over neurons; they use matrices. The backward step for a whole layer is one line: take the next layer's δ vector, multiply by the <em>transpose</em> of the weights between them (the forward weights, run backwards), then multiply element-wise by the activation's slope. The gradient for the weight matrix is an outer product of δ and the layer's input. Here are the live numbers from the network above."))}
          <div data-out="matrix"></div>
          ${deeper("The math", `
            ${mathBlock(String.raw`\boldsymbol\delta^{(L)} = \nabla_{\mathbf a}L \odot f'(\mathbf z^{(L)}),\qquad \boldsymbol\delta^{(\ell)} = \big(W^{(\ell+1)}\big)^{\!\top}\boldsymbol\delta^{(\ell+1)} \odot f'(\mathbf z^{(\ell)}),\qquad \frac{\partial L}{\partial W^{(\ell)}} = \boldsymbol\delta^{(\ell)}\,\mathbf a^{(\ell-1)\top}`)}
            ${para("Three lines, any number of layers. With a batch of examples, the vectors become matrices with one column per example and the gradient sums over them.")}
          `)}
          ${takeaway("δ for a layer = Wᵀ·δ of the next layer ⊙ f′(z); weight gradient = δ · (input)ᵀ.")}
        `)}

        ${chapter("bp-train", 6, "Training", "Training: repeat", `
          ${plain(para("One training step barely changes anything. Training repeats it thousands of times. Here a small network learns <strong>XOR</strong> (output 1 when exactly one input is 1), the classic problem a single neuron can't solve because no straight line separates the classes. The hidden layer has to learn a new representation in which they <em>can</em> be separated."))}
          ${widget("a network learns XOR", `
            ${row("Hidden neurons", segmented("H", [["2", "2"], ["3", "3"], ["4", "4"], ["8", "8"]], "3"))}
            ${row("Hidden activation", segmented("act", [["tanh", "tanh"], ["sigmoid", "sigmoid"], ["relu", "ReLU"]], "tanh"))}
            ${range("lr", "Learning rate η", 0.05, 3, 0.05, 1)}
            <div class="g-actions">
              <button type="button" class="button primary" data-action="train">Train 400 epochs</button>
              <button type="button" class="button secondary" data-action="reset">New random start</button>
            </div>
            <div class="g-two">
              <div>${svgSlot("xor", "0 0 300 300", "network output over the input square")}</div>
              <div data-out="xor"></div>
            </div>`)}
          ${takeaway("Training repeats forward, backward and update; the hidden layer learns features that make the problem solvable.")}
        `)}

        ${chapter("bp-depth", 7, "Depth", "Gradients through depth", `
          ${plain(para("Because δ is multiplied by a weight matrix and an activation slope at every layer on the way back, deep networks can lose (or blow up) their gradient. The <a href=\"./algorithm.html?id=activation-functions#act-vanish\">activation guide</a> estimated this on paper. Here it is measured: a real backward pass through a deep stack of 8-neuron layers, averaged over 24 random inputs."))}
          ${widget("measure the gradient at every layer", `
            ${range("depth", "Layers", 2, 30, 1, 14)}
            ${row("Activation", segmented("act", [["sigmoid", "sigmoid"], ["tanh", "tanh"], ["relu", "ReLU"]], "sigmoid"))}
            ${row("Weight initialisation", segmented("init", [["small", "small (std 0.1)"], ["xavier", "Xavier (1/√n)"], ["he", "He (√(2/n))"]], "xavier"))}
            ${svgSlot("depth", "0 0 560 260", "weight-gradient size per layer, log scale")}
            <div data-out="depth"></div>`)}
          ${takeaway("Measured gradients shrink layer by layer with sigmoid or tiny weights; ReLU with He initialisation keeps them steady.")}
        `)}

        ${chapter("bp-check", 8, "Debugging", "Checking your gradients", `
          ${plain(para("A backward pass with a bug still produces numbers, and training may even seem to work, just worse. The standard check is to compare each backprop gradient with a slow nudge estimate. They should agree to many decimal places. Pick a bug below and see which gradients it breaks."))}
          ${widget("gradient check on the 2-2-1 network", `
            ${row("Backward pass", segmented("bug", [["none", "correct"], ["noderiv", "bug: forgot σ′ in hidden layer"], ["swap", "bug: used the wrong weight"]], "none"))}
            <div data-out="check"></div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`\text{relative error} = \frac{|g_{\text{backprop}} - g_{\text{nudge}}|}{\max\big(|g_{\text{backprop}}| + |g_{\text{nudge}}|,\ 10^{-12}\big)},\qquad g_{\text{nudge}} = \frac{L(w+h) - L(w-h)}{2h}`)}
            ${para("Below about 10⁻⁷ is a pass. Note that a bug in the hidden layer leaves the output layer's gradients correct, which tells you where to look.")}
          `)}
          ${takeaway("Always verify a hand-written backward pass against a numerical estimate; bugs show up as large relative errors.")}
        `)}

        ${chapter("bp-play", 9, "Playground", "Design your own network", `
          ${plain(para("The full lab: choose the number of layers and neurons, activations per layer, the loss, inputs and targets, then step through the forward pass, backward pass and update, or train for many epochs. Every value is recomputed from what you enter."))}
          <div class="g-playground" id="nn-playground"></div>
        `)}

        ${chapter("bp-recap", 10, "Recap", "Recap and self-check", `
          <ol class="g-recap">
            <li>Backprop computes <b>∂L/∂w for every weight</b> in one backward pass: about 3× the cost of a forward pass.</li>
            <li>It is the <b>chain rule</b>, evaluated from the loss backwards so shared factors are reused.</li>
            <li>Each neuron's <b>error signal δ = ∂L/∂z</b>; a weight's gradient is δ × the input it multiplies.</li>
            <li>Layer by layer: <b>δ⁽ˡ⁾ = W⁽ˡ⁺¹⁾ᵀ δ⁽ˡ⁺¹⁾ ⊙ f′(z⁽ˡ⁾)</b>.</li>
            <li>Check hand-written gradients against <b>numerical estimates</b>.</li>
          </ol>
          ${quiz([
            { q: "Why compute gradients from the loss backwards rather than from the inputs forwards?", a: "There is one loss and many weights. Going backwards, each δ is computed once and shared by every weight that feeds that neuron; going forwards you'd need a separate pass per weight." },
            { q: "A neuron's δ is 0.2 and the input on one of its weights is −0.5. What is that weight's gradient, and which way will gradient descent move it?", a: "∂L/∂w = 0.2 × (−0.5) = −0.1. Gradient descent subtracts η × (−0.1), so the weight increases." },
            { q: "An add node receives an upstream gradient of 3. What does it send to each input? And a multiply node computing p·q with p = 2, q = −4?", a: "Add: 3 to each input. Multiply: 3 × (−4) = −12 to p and 3 × 2 = 6 to q." },
            { q: "Your backprop gradients match the numerical ones in the output layer but not the hidden layer. Where is the bug?", a: "In the step that pulls δ back through the output weights to the hidden layer: the transpose, the choice of weight, or the activation derivative there." },
          ])}
          ${nextCard("Next in the series", "Before attention: the building blocks", "Tokens, embeddings, softmax and recurrent networks, leading to attention, Transformers and LLMs.", "./algorithm.html?id=sequence-primer", "Continue to the Primer →")}
        `)}
      `;
    }

    const state = {
      why: { size: "1e6" },
      chain: { x: 1.5, w: 0.8, b: -0.2, y: 0.2 },
      gates: { x: -2, y: 5, z: -4 },
      tiny: { x1: 0.05, x2: 0.1, y: 0.01, lr: 0.5, stage: 0, net: cloneNet(TINY_START), steps: 0 },
      xor: { H: "3", act: "tanh", lr: 1, seed: 3, sig: "", net: null },
      depth: { depth: 14, act: "sigmoid", init: "xavier" },
      check: { bug: "none" },
    };
    const fmtOut = (key, value) => (typeof value === "number" ? String(Math.round(value * 1000) / 1000) : value);
    let renderMatrix = null;
    let renderCheck = null;
    let xorRun = null;

    function mountWhy() {
      const host = hostOf(rootNode, "why");
      mountWidget(host, state.why, (h, s) => {
        const N = Number(s.size);
        h.querySelector('[data-out="why"]').innerHTML = `
          <div class="g-metrics">
            <div><span>nudging: forward passes per step</span><b>${fmtBig(N + 1)}</b></div>
            <div><span>backprop: forward-pass equivalents</span><b>≈ 3</b></div>
            <div><span>speed-up</span><b>≈ ${fmtBig((N + 1) / 3)}×</b></div>
          </div>
          <p class="caption">If one forward pass took a millisecond, a single nudging step would take ${(() => {
            const sec = (N + 1) / 1000;
            return sec < 120 ? `${fmt(sec, 0)} seconds` : sec < 7200 ? `${fmt(sec / 60, 0)} minutes` : sec < 172800 ? `${fmt(sec / 3600, 1)} hours` : `${fmt(sec / 86400, 0)} days`;
          })()}; backprop takes about 3 milliseconds. Training runs for hundreds of thousands of steps.</p>`;
      });
    }

    function mountChain() {
      const host = hostOf(rootNode, "chain");
      restoreInputs(host, state.chain);
      mountWidget(host, state.chain, (h, s) => {
        const wx = s.w * s.x;
        const z = wx + s.b;
        const a = sig(z);
        const L = 0.5 * (a - s.y) ** 2;
        const dLa = a - s.y;
        const daz = a * (1 - a);
        const delta = dLa * daz;
        const dw = delta * s.x;
        const db = delta;
        const lossAt = (w, b) => 0.5 * (sig(w * s.x + b) - s.y) ** 2;
        const numW = (lossAt(s.w + 1e-5, s.b) - lossAt(s.w - 1e-5, s.b)) / 2e-5;
        const svg = h.querySelector('[data-fig="chain"]');
        U().clear(svg);
        ui.arrowMarker(svg, "bp-ch-f", C().a);
        const nodes = [
          { x: 40, y: 60, label: "w", fwd: tn(s.w, 2), back: tn(dw, 4) },
          { x: 40, y: 150, label: "x", fwd: tn(s.x, 2), back: "" },
          { x: 170, y: 105, label: "×", fwd: tn(wx, 3), back: tn(delta, 4) },
          { x: 290, y: 105, label: "+ b", fwd: tn(z, 3), back: tn(delta, 4) },
          { x: 410, y: 105, label: "σ", fwd: tn(a, 4), back: tn(dLa, 4) },
          { x: 540, y: 105, label: "loss", fwd: tn(L, 4), back: "1" },
        ];
        const edges = [[0, 2], [1, 2], [2, 3], [3, 4], [4, 5]];
        edges.forEach(([i, j]) => {
          svg.appendChild(U().svgEl("line", { x1: nodes[i].x + 26, y1: nodes[i].y, x2: nodes[j].x - 30, y2: nodes[j].y, stroke: C().faint, "stroke-width": 2, "marker-end": "url(#bp-ch-f)" }));
        });
        nodes.forEach((n, i) => {
          svg.appendChild(U().svgEl("circle", { cx: n.x, cy: n.y, r: 26, fill: C().plotBg, stroke: i < 2 ? C().neutral : C().c, "stroke-width": 2 }));
          ui.svgText(svg, n.x, n.y + 5, n.label, { "text-anchor": "middle", class: "svg-title" });
          ui.svgText(svg, n.x, n.y - 34, n.fwd, { "text-anchor": "middle", fill: C().a });
          if (n.back) ui.svgText(svg, n.x, n.y + 46, n.back, { "text-anchor": "middle", fill: C().b });
        });
        ui.svgText(svg, 8, 200, "teal: forward value · orange: ∂L/∂(this node), flowing right to left", {});
        h.querySelector('[data-out="chain"]').innerHTML = `
          ${mathBlock(String.raw`\frac{\partial L}{\partial w} = \underbrace{${tn(dLa, 4)}}_{a-y}\times\underbrace{${tn(daz, 4)}}_{a(1-a)}\times\underbrace{${tp(s.x, 2)}}_{x} = ${tn(dw, 5)}`)}
          <div class="g-metrics">
            <div><span>δ = ∂L/∂z</span><b>${fmt(delta, 5)}</b></div>
            <div><span>∂L/∂w (backprop)</span><b>${fmt(dw, 6)}</b></div>
            <div><span>∂L/∂w (nudge check)</span><b>${fmt(numW, 6)}</b></div>
            <div><span>∂L/∂b</span><b>${fmt(db, 5)}</b></div>
          </div>
          <p class="caption">Set the target equal to the output and every gradient becomes 0: nothing to fix. Make |z| large and a(1 − a) shrinks, so the neuron learns slowly (saturation).</p>`;
      }, { formatOutput: fmtOut });
    }

    function mountGates() {
      const host = hostOf(rootNode, "gates");
      restoreInputs(host, state.gates);
      mountWidget(host, state.gates, (h, s) => {
        const x = Number(s.x) || 0;
        const y = Number(s.y) || 0;
        const z = Number(s.z) || 0;
        const q = x + y;
        const f = q * z;
        const dq = z;
        const dz = q;
        const svg = h.querySelector('[data-fig="gates"]');
        U().clear(svg);
        ui.arrowMarker(svg, "bp-g", C().faint);
        const N = {
          x: [50, 50, "x", x, dq],
          y: [50, 125, "y", y, dq],
          z: [50, 195, "z", z, dz],
          add: [230, 88, "+", q, dq],
          mul: [400, 140, "×", f, 1],
          out: [520, 140, "f", f, 1],
        };
        [["x", "add"], ["y", "add"], ["add", "mul"], ["z", "mul"], ["mul", "out"]].forEach(([a, b]) => {
          svg.appendChild(U().svgEl("line", { x1: N[a][0] + 22, y1: N[a][1], x2: N[b][0] - 26, y2: N[b][1], stroke: C().faint, "stroke-width": 2, "marker-end": "url(#bp-g)" }));
        });
        Object.entries(N).forEach(([key, [cx, cy, label, fwd, back]]) => {
          svg.appendChild(U().svgEl("circle", { cx, cy, r: 22, fill: C().plotBg, stroke: key === "add" || key === "mul" ? C().c : C().neutral, "stroke-width": 2 }));
          ui.svgText(svg, cx, cy + 5, label, { "text-anchor": "middle", class: "svg-title" });
          ui.svgText(svg, cx + (key === "out" ? 0 : 0), cy - 28, tn(fwd, 2).replace(/\.00$/, ""), { "text-anchor": "middle", fill: C().a });
          ui.svgText(svg, cx, cy + 40, tn(back, 2).replace(/\.00$/, ""), { "text-anchor": "middle", fill: C().b });
        });
        h.querySelector('[data-out="gates"]').innerHTML = `
          <ol class="g-pipeline">
            <li><b>Forward:</b> q = x + y = ${fmt(q, 2)}, then f = q · z = ${fmt(f, 2)}.</li>
            <li><b>Start backward</b> with ∂f/∂f = 1.</li>
            <li><b>Multiply node</b> swaps its inputs: ∂f/∂q = z = ${fmt(dq, 2)} and ∂f/∂z = q = ${fmt(dz, 2)}.</li>
            <li><b>Add node</b> copies the gradient to both inputs: ∂f/∂x = ∂f/∂y = ${fmt(dq, 2)}.</li>
          </ol>
          <p class="caption">Check: raising x by 1 changes f by z = ${fmt(z, 2)}, exactly as the backward pass predicts.</p>`;
      }, { formatOutput: fmtOut });
    }

    const TINY_STAGES = [
      "The inputs and target",
      "Hidden layer: weighted sums",
      "Hidden layer: activations",
      "Output neuron",
      "The loss",
      "Output error signal δ",
      "Gradients for the output weights",
      "Pull δ back to the hidden layer",
      "Gradients for the hidden weights",
      "Update every weight",
    ];

    function tinyData(s) {
      const x = [Number(s.x1) || 0, Number(s.x2) || 0];
      const y = clamp(Number(s.y) || 0, 0, 1);
      const lr = Number(s.lr) || 0;
      const f = tinyForward(s.net, x);
      const g = tinyBackward(s.net, x, y, f);
      const n2 = tinyUpdate(s.net, g, lr);
      const L2 = tinyLoss(n2, x, y);
      return { x, y, lr, f, g, n2, L2 };
    }

    function drawTiny(svg, s, d) {
      U().clear(svg);
      svg.classList.add("sl-halo");
      const st = s.stage;
      const n = s.net;
      const pos = { i0: [70, 90], i1: [70, 240], h0: [320, 90], h1: [320, 240], o: [560, 165] };
      const edgeW = (from, to, w, g, showG, active, t) => {
        const [x1, y1] = pos[from];
        const [x2, y2] = pos[to];
        svg.appendChild(U().svgEl("line", { x1: x1 + 30, y1, x2: x2 - 30, y2, stroke: active ? C().c : C().faint, "stroke-width": active ? 2.6 : 1.6 }));
        const lx = x1 + 30 + (x2 - x1 - 60) * t;
        const ly = y1 + (y2 - y1) * t;
        ui.svgText(svg, lx, ly - 6, tn(w, 3), { "text-anchor": "middle", class: "svg-label" });
        if (showG) ui.svgText(svg, lx, ly + 12, `∇ ${tn(g, 4)}`, { "text-anchor": "middle", fill: C().b });
      };
      edgeW("i0", "h0", n.W1[0][0], d.g.gW1[0][0], st >= 8, st === 1 || st === 8, 0.5);
      edgeW("i1", "h0", n.W1[0][1], d.g.gW1[0][1], st >= 8, st === 1 || st === 8, 0.25);
      edgeW("i0", "h1", n.W1[1][0], d.g.gW1[1][0], st >= 8, st === 1 || st === 8, 0.25);
      edgeW("i1", "h1", n.W1[1][1], d.g.gW1[1][1], st >= 8, st === 1 || st === 8, 0.5);
      edgeW("h0", "o", n.W2[0], d.g.gW2[0], st >= 6, st === 3 || st === 6 || st === 7, 0.5);
      edgeW("h1", "o", n.W2[1], d.g.gW2[1], st >= 6, st === 3 || st === 6 || st === 7, 0.5);
      const node = (key, label, top, bottom, back, active, color) => {
        const [cx, cy] = pos[key];
        svg.appendChild(U().svgEl("circle", { cx, cy, r: 30, fill: C().plotBg, stroke: active ? C().b : color, "stroke-width": active ? 3.2 : 2 }));
        ui.svgText(svg, cx, cy - 4, label, { "text-anchor": "middle", class: "svg-title" });
        if (top) ui.svgText(svg, cx, cy + 13, top, { "text-anchor": "middle", fill: C().a });
        if (bottom) ui.svgText(svg, cx, cy + 48, bottom, { "text-anchor": "middle", class: "svg-label" });
        if (back) ui.svgText(svg, cx, cy - 40, back, { "text-anchor": "middle", fill: C().b });
      };
      node("i0", "x₁", tn(d.x[0], 2), "", "", st === 0, C().c);
      node("i1", "x₂", tn(d.x[1], 2), "", "", st === 0, C().c);
      ["h0", "h1"].forEach((k, j) => {
        node(k, `h${j + 1}`, st >= 2 ? `a=${tn(d.f.a1[j], 3)}` : "", st >= 1 ? `z=${tn(d.f.z1[j], 3)} · b=${tn(n.b1[j], 2)}` : `b=${tn(n.b1[j], 2)}`, st >= 7 ? `δ=${tn(d.g.d1[j], 5)}` : "", st === 1 || st === 2 || st === 7, C().d);
      });
      node("o", "out", st >= 3 ? `a=${tn(d.f.a2, 3)}` : "", st >= 3 ? `z=${tn(d.f.z2, 3)} · c=${tn(n.b2, 2)}` : `c=${tn(n.b2, 2)}`, st >= 5 ? `δ=${tn(d.g.d2, 5)}` : "", st === 3 || st === 5, C().b);
      ui.svgText(svg, 560, 262, st >= 4 ? `L = ${tn(d.g.L, 5)}` : `y = ${tn(d.y, 2)}`, Object.assign({ "text-anchor": "middle", class: "svg-title" }, st === 4 ? { fill: C().b } : {}));
    }

    function tinyMath(s, d) {
      const n = s.net;
      const { x, y, f, g, lr } = d;
      switch (s.stage) {
        case 0:
          return `${para(`Inputs x = (${fmt(x[0], 2)}, ${fmt(x[1], 2)}), target y = ${fmt(y, 2)}. The weights start at fixed values${s.steps ? `, already updated ${s.steps} time${s.steps > 1 ? "s" : ""}` : ""}. Press <b>Next step</b>.`)}`;
        case 1:
          return mathBlock(String.raw`\begin{aligned} z_1 &= w_{11}x_1 + w_{12}x_2 + b_1 = ${tn(n.W1[0][0])}\cdot${tp(x[0], 2)} + ${tn(n.W1[0][1])}\cdot${tp(x[1], 2)} + ${tp(n.b1[0])} = ${tn(f.z1[0], 4)}\\ z_2 &= w_{21}x_1 + w_{22}x_2 + b_2 = ${tn(n.W1[1][0])}\cdot${tp(x[0], 2)} + ${tn(n.W1[1][1])}\cdot${tp(x[1], 2)} + ${tp(n.b1[1])} = ${tn(f.z1[1], 4)}\end{aligned}`);
        case 2:
          return mathBlock(String.raw`h_1 = \sigma(${tn(f.z1[0], 4)}) = ${tn(f.a1[0], 4)},\qquad h_2 = \sigma(${tn(f.z1[1], 4)}) = ${tn(f.a1[1], 4)}`);
        case 3:
          return mathBlock(String.raw`z_{\text{out}} = v_1h_1 + v_2h_2 + c = ${tn(n.W2[0])}\cdot${tn(f.a1[0], 4)} + ${tn(n.W2[1])}\cdot${tn(f.a1[1], 4)} + ${tp(n.b2)} = ${tn(f.z2, 4)},\quad a_{\text{out}} = \sigma(z_{\text{out}}) = ${tn(f.a2, 4)}`);
        case 4:
          return mathBlock(String.raw`L = \tfrac12\,(a_{\text{out}} - y)^2 = \tfrac12\,(${tn(f.a2, 4)} - ${tn(y, 2)})^2 = ${tn(g.L, 6)}`);
        case 5:
          return `${mathBlock(String.raw`\delta_{\text{out}} = \frac{\partial L}{\partial z_{\text{out}}} = (a_{\text{out}} - y)\cdot a_{\text{out}}(1 - a_{\text{out}}) = ${tn(f.a2 - y, 4)}\cdot${tn(f.a2 * (1 - f.a2), 4)} = ${tn(g.d2, 6)}`)}${para("The backward pass starts here. δ says how much the loss would change per unit change of the output neuron's z.")}`;
        case 6:
          return mathBlock(String.raw`\frac{\partial L}{\partial v_1} = \delta_{\text{out}}\,h_1 = ${tn(g.d2, 5)}\cdot${tn(f.a1[0], 4)} = ${tn(g.gW2[0], 6)},\quad \frac{\partial L}{\partial v_2} = ${tn(g.gW2[1], 6)},\quad \frac{\partial L}{\partial c} = \delta_{\text{out}} = ${tn(g.gb2, 6)}`);
        case 7:
          return `${mathBlock(String.raw`\delta_j = v_j\,\delta_{\text{out}}\cdot h_j(1-h_j):\quad \delta_1 = ${tn(n.W2[0])}\cdot${tp(g.d2, 5)}\cdot${tn(f.a1[0] * (1 - f.a1[0]), 4)} = ${tn(g.d1[0], 7)},\quad \delta_2 = ${tn(g.d1[1], 7)}`)}${para("Each hidden neuron gets its share of the blame: the output's δ, scaled by the weight connecting them (the old value, before any update), times its own slope.")}`;
        case 8:
          return mathBlock(String.raw`\frac{\partial L}{\partial w_{jk}} = \delta_j\,x_k:\quad \frac{\partial L}{\partial w_{11}} = ${tn(g.d1[0], 6)}\cdot${tp(x[0], 2)} = ${tn(g.gW1[0][0], 8)},\ \ \frac{\partial L}{\partial w_{22}} = ${tn(g.gW1[1][1], 8)},\ \ \ldots`);
        default:
          return `${mathBlock(String.raw`w \leftarrow w - \eta\,\frac{\partial L}{\partial w}:\quad v_1 = ${tn(n.W2[0], 4)} - ${tn(lr, 2)}\cdot${tp(g.gW2[0], 5)} = ${tn(d.n2.W2[0], 5)},\ \ldots`)}
            <div class="g-metrics">
              <div><span>loss before</span><b>${fmt(g.L, 6)}</b></div>
              <div><span>loss after</span><b>${fmt(d.L2, 6)}</b></div>
              <div><span>change</span><b>${fmt(((d.L2 - g.L) / (g.L || 1)) * 100, 2)}%</b></div>
            </div>
            ${para("Press <b>Apply update, start again</b> to keep the new weights and run the next step.")}`;
      }
    }

    function mountTiny() {
      const host = hostOf(rootNode, "tiny");
      restoreInputs(host, state.tiny);
      const draw = (h, s) => {
        const d = tinyData(s);
        drawTiny(h.querySelector('[data-fig="tiny"]'), s, d);
        h.querySelector('[data-out="tiny"]').innerHTML = `
          <div class="tl-subhead">Step ${s.stage + 1} of ${TINY_STAGES.length}: ${TINY_STAGES[s.stage]}</div>
          ${tinyMath(s, d)}`;
        if (renderMatrix) renderMatrix();
        if (renderCheck) renderCheck();
      };
      mountWidget(host, state.tiny, draw, {
        formatOutput: fmtOut,
        next: (s) => {
          s.stage = Math.min(TINY_STAGES.length - 1, s.stage + 1);
        },
        prev: (s) => {
          s.stage = Math.max(0, s.stage - 1);
        },
        commit: (s) => {
          s.net = tinyData(s).n2;
          s.steps += 1;
          s.stage = 0;
        },
        reset: (s) => {
          s.net = cloneNet(TINY_START);
          s.steps = 0;
          s.stage = 0;
        },
      });
    }

    function mountMatrix() {
      const out = rootNode.querySelector('[data-out="matrix"]');
      renderMatrix = () => {
        const s = state.tiny;
        const d = tinyData(s);
        const col = (v) => v.map((x) => [x]);
        out.innerHTML = `
          <div class="tl-mats-row">
            ${heatmap([[d.g.d2]], { rows: ["out"], cols: ["δ⁽²⁾"], digits: 4, title: "δ at the output", shape: "1" })}
            ${heatmap(col(s.net.W2), { rows: ["h1", "h2"], cols: ["W⁽²⁾ᵀ"], digits: 3, title: "transposed weights", shape: "2×1" })}
            ${heatmap(col(d.f.a1.map((a) => a * (1 - a))), { rows: ["h1", "h2"], cols: ["σ′(z)"], digits: 4, title: "local slopes", shape: "2" })}
            ${heatmap(col(d.g.d1), { rows: ["h1", "h2"], cols: ["δ⁽¹⁾"], digits: 6, title: "= δ at the hidden layer", shape: "2" })}
            ${heatmap(d.g.gW1, { rows: ["h1", "h2"], cols: ["x₁", "x₂"], digits: 6, title: "∂L/∂W⁽¹⁾ = δ⁽¹⁾ xᵀ", shape: "2×2" })}
          </div>
          <p class="caption">Live values from the network in chapter 4 (they change when you apply an update there).</p>`;
      };
      renderMatrix();
    }

    function drawXor(svg, net, act) {
      U().clear(svg);
      svg.classList.add("sl-halo");
      const R = 30;
      const size = 300 / R;
      for (let j = 0; j < R; j += 1) {
        for (let i = 0; i < R; i += 1) {
          const x = -0.25 + (1.5 * (i + 0.5)) / R;
          const y = 1.25 - (1.5 * (j + 0.5)) / R;
          const p = xorForward(net, [x, y], act).out;
          svg.appendChild(U().svgEl("rect", { x: i * size, y: j * size, width: size, height: size, fill: ui.rgba(p >= 0.5 ? C().a : C().b, 0.08 + 0.7 * Math.abs(p - 0.5) * 2), "shape-rendering": "crispEdges" }));
        }
      }
      XOR.forEach(([x, y, t]) => {
        svg.appendChild(U().svgEl("circle", { cx: ((x + 0.25) / 1.5) * 300, cy: ((1.25 - y) / 1.5) * 300, r: 11, fill: t ? C().a : C().b, stroke: C().ring, "stroke-width": 3 }));
        ui.svgText(svg, ((x + 0.25) / 1.5) * 300, ((1.25 - y) / 1.5) * 300 + 32, `(${x},${y})→${t}`, { "text-anchor": "middle", class: "svg-label" });
      });
    }

    function mountXor() {
      const host = hostOf(rootNode, "xor");
      restoreInputs(host, state.xor);
      const ensure = (s) => {
        const sg = `${s.H}|${s.act}|${s.seed}`;
        if (sg !== s.sig || !s.net) {
          xorRun = null;
          s.net = xorInit(Number(s.H), s.seed);
          s.sig = sg;
        }
      };
      const paint = (h, s) => {
        const act = ACTS[s.act];
        drawXor(h.querySelector('[data-fig="xor"]'), s.net, act);
        const outs = XOR.map(([x, y]) => xorForward(s.net, [x, y], act).out);
        const correct = outs.filter((p, i) => (p >= 0.5 ? 1 : 0) === XOR[i][2]).length;
        const last = s.net.history[s.net.history.length - 1];
        h.querySelector('[data-out="xor"]').innerHTML = `
          <div class="g-metrics">
            <div><span>epochs</span><b>${fmtInt(s.net.epochs)}</b></div>
            <div><span>loss</span><b>${last === undefined ? "—" : fmt(last, 4)}</b></div>
            <div><span>correct</span><b>${correct} of 4</b></div>
          </div>
          <table class="g-mini"><tr><th>input</th><th>target</th><th>output</th></tr>${XOR.map(([x, y, t], i) => `<tr><td>(${x}, ${y})</td><td>${t}</td><td class="mono">${fmt(outs[i], 3)}</td></tr>`).join("")}</table>
          ${s.net.history.length > 1 ? `<div class="tl-subhead">Loss over epochs</div>${sparkline(s.net.history.filter((_, i) => i % 4 === 0), C().b)}` : ""}
          <p class="caption">${s.H === "2" ? "With only 2 hidden neurons some random starts get stuck; press “New random start”. " : ""}${s.act === "relu" ? "With ReLU, hidden neurons can die and leave the network unable to finish. " : ""}Teal regions are classified 1, orange 0.</p>`;
      };
      mountWidget(host, state.xor, (h, s) => {
        ensure(s);
        paint(h, s);
      }, {
        formatOutput: fmtOut,
        reset: (s) => {
          s.seed += 1;
        },
        train: (s) => {
          ensure(s);
          const token = {};
          xorRun = token;
          let done = 0;
          const tick = () => {
            if (xorRun !== token || !host.isConnected) return;
            const act = ACTS[s.act];
            for (let k = 0; k < 10; k += 1) s.net.history.push(xorEpoch(s.net, act, s.lr));
            done += 10;
            paint(host, s);
            if (done < 400) requestAnimationFrame(tick);
            else xorRun = null;
          };
          requestAnimationFrame(tick);
        },
      });
    }

    function mountDepth() {
      const host = hostOf(rootNode, "depth");
      restoreInputs(host, state.depth);
      mountWidget(host, state.depth, (h, s) => {
        const norms = depthGradients(s.depth, s.act, s.init);
        const logs = norms.map((v) => Math.log10(Math.max(v, 1e-30)));
        const lo = Math.max(-20, Math.floor(Math.min(...logs)) - 1);
        const hi = Math.ceil(Math.max(...logs)) + 1;
        const svg = h.querySelector('[data-fig="depth"]');
        const L = s.depth;
        const f = frame(svg, {
          xDomain: [0.4, L + 0.6],
          yDomain: [lo, hi],
          xTicks: Array.from({ length: L }, (_, i) => i + 1).filter((v) => L <= 15 || v % 5 === 0 || v === 1).map((v) => [v, String(v)]),
          yTicks: niceTicks(lo, hi, 5).filter((v) => Number.isInteger(v)).map((v) => [v, v === 0 ? "1" : `1e${v}`]),
          title: "Size of ∂L/∂W at each layer (layer 1 is next to the input)",
          xLabel: "layer",
          pad: { left: 52 },
        });
        const bw = Math.max(3, (f.xs(2) - f.xs(1)) * 0.7);
        logs.forEach((v, i) => {
          f.el("rect", { x: f.xs(i + 1) - bw / 2, y: f.ys(v), width: bw, height: f.ys(lo) - f.ys(v), fill: C().c, rx: 2 });
        });
        const ratio = norms[0] / norms[L - 1];
        h.querySelector('[data-out="depth"]').innerHTML = `
          <div class="g-metrics">
            <div><span>gradient at layer 1</span><b>${norms[0].toExponential(1)}</b></div>
            <div><span>gradient at layer ${L}</span><b>${norms[L - 1].toExponential(1)}</b></div>
            <div><span>first ÷ last</span><b>${ratio < 0.01 || ratio > 100 ? ratio.toExponential(1) : fmt(ratio, 2)}</b></div>
          </div>
          <p class="caption">With a ratio far below 1, the first layers learn orders of magnitude more slowly than the last. Try ReLU with He initialisation, then sigmoid with small weights.</p>`;
      }, { formatOutput: fmtOut });
    }

    function mountCheck() {
      const host = hostOf(rootNode, "check");
      const draw = (h, s) => {
        const t = state.tiny;
        const x = [Number(t.x1) || 0, Number(t.x2) || 0];
        const y = clamp(Number(t.y) || 0, 0, 1);
        const net = t.net;
        const g = tinyBackward(net, x, y, tinyForward(net, x), s.bug);
        const rows = TINY_PARAMS.map(([label, get, set, grad]) => {
          const plus = cloneNet(net);
          const minus = cloneNet(net);
          set(plus, get(net) + 1e-5);
          set(minus, get(net) - 1e-5);
          const num = (tinyLoss(plus, x, y) - tinyLoss(minus, x, y)) / 2e-5;
          const bp = grad(g);
          const rel = Math.abs(bp - num) / Math.max(Math.abs(bp) + Math.abs(num), 1e-12);
          return { label, bp, num, rel };
        });
        const failed = rows.filter((r) => r.rel > 1e-5).length;
        h.querySelector('[data-out="check"]').innerHTML = `
          <table class="g-mini"><tr><th>parameter</th><th>backprop</th><th>nudge</th><th>relative error</th><th></th></tr>${rows
            .map((r) => `<tr><td>${r.label}</td><td class="mono">${r.bp.toExponential(4)}</td><td class="mono">${r.num.toExponential(4)}</td><td class="mono">${r.rel.toExponential(1)}</td><td>${r.rel > 1e-5 ? "✗" : "✓"}</td></tr>`)
            .join("")}</table>
          <p class="caption">${failed ? `<b>${failed} of ${rows.length} gradients are wrong.</b> All of them are hidden-layer parameters (w and b₁, b₂); the output layer (v₁, v₂, c) still passes.` : "All gradients agree to about 10⁻¹⁰: the backward pass is correct."} Uses the inputs, target and current weights from chapter 4.</p>`;
      };
      mountWidget(host, state.check, draw);
      renderCheck = () => draw(host, state.check);
    }

    function mountPlayground() {
      const node = rootNode.querySelector("#nn-playground");
      if (node && ctx && ctx.engines && ctx.engines.backprop) {
        try {
          ctx.engines.backprop(node);
        } catch (error) {
          node.innerHTML = '<p class="caption">The playground could not be loaded.</p>';
        }
      }
    }

    function render() {
      rootNode.innerHTML = body();
      renderMatrix = null;
      renderCheck = null;
      mountWhy();
      mountChain();
      mountGates();
      mountMatrix();
      mountTiny();
      mountXor();
      mountDepth();
      mountCheck();
      mountPlayground();
    }

    render();
    U().onRedraw(render);
    ui.startProgressBar();
  }

  root.MLExtraLabs = Object.assign(root.MLExtraLabs || {}, {
    "activation-guide": mountActivationGuide,
    "gradient-descent-guide": mountGradientGuide,
    "backprop-guide": mountBackpropGuide,
  });
})();
