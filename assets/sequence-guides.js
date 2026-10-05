/* ── Primer and LLM guides ──────────────────────────────────────
   Two more long-form guides that bracket the Attention and Transformer
   guides: a primer on the building blocks before them, and the road from
   the 2017 Transformer to today's large language models after them.

   They reuse the guide building blocks from transformer-labs.js
   (window.MLSeqUI) and add "Try it" widgets: small editable figures
   where the numbers are live. The tokenizer widgets run the real GPT-2
   and GPT-4 (cl100k) tokenizers, bundled from js-tiktoken (MIT) into
   assets/vendor and loaded only when the reader reaches them. */
(function () {
  const root = window;
  const UI = () => root.MLSeqUI;
  const M = () => root.MLTransformerMath;
  const U = () => root.MLUtils;
  const C = () => root.MLUtils.chartColors();

  const VENDOR = "../assets/vendor/";
  const VENDOR_VERSION = "20261004";

  const esc = (text) => String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  /* ── Shared widget helpers ───────────────────────────────────── */

  function widget(title, html) {
    return `<div class="g-widget"><div class="g-label">✎ Try it · ${title}</div>${html}</div>`;
  }

  function range(name, label, min, max, step, value) {
    return `
      <label class="g-range">
        <span>${label}</span>
        <input type="range" data-input="${name}" min="${min}" max="${max}" step="${step}" value="${value}" />
        <output data-output="${name}">${value}</output>
      </label>`;
  }

  function segmented(name, options, active) {
    return `<div class="g-seg" role="group">${options
      .map(([value, label]) => `<button type="button" data-seg="${name}" data-value="${value}" class="${value === active ? "is-on" : ""}">${label}</button>`)
      .join("")}</div>`;
  }

  const fmtInt = (value) => Math.round(value).toLocaleString("en-US");
  function fmtBig(value) {
    if (value >= 1e12) return `${(value / 1e12).toFixed(2)} T`;
    if (value >= 1e9) return `${(value / 1e9).toFixed(2)} B`;
    if (value >= 1e6) return `${(value / 1e6).toFixed(1)} M`;
    if (value >= 1e3) return `${(value / 1e3).toFixed(1)} K`;
    return String(Math.round(value));
  }
  function fmtBytes(bytes) {
    if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
    if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  /* A widget keeps its own state and re-renders only its own output, so
     typing in a textarea never loses focus. */
  function mountWidget(host, state, renderOutput, handlers = {}) {
    if (!host) return;
    const sync = () => {
      host.querySelectorAll("[data-output]").forEach((node) => {
        const key = node.getAttribute("data-output");
        if (key in state) node.textContent = handlers.formatOutput ? handlers.formatOutput(key, state[key]) : state[key];
      });
      host.querySelectorAll("[data-seg]").forEach((button) => {
        button.classList.toggle("is-on", String(state[button.getAttribute("data-seg")]) === button.getAttribute("data-value"));
      });
      renderOutput(host, state);
    };
    host.addEventListener("input", (event) => {
      const key = event.target.getAttribute("data-input");
      if (!key) return;
      state[key] = event.target.type === "range" || event.target.type === "number" ? Number(event.target.value) : event.target.value;
      sync();
    });
    host.addEventListener("click", (event) => {
      const seg = event.target.closest("[data-seg]");
      if (seg) {
        state[seg.getAttribute("data-seg")] = seg.getAttribute("data-value");
        sync();
      }
      const action = event.target.closest("[data-action]");
      if (action && handlers[action.getAttribute("data-action")]) {
        handlers[action.getAttribute("data-action")](state);
        sync();
      }
    });
    sync();
    return sync;
  }

  /* ── Real tokenizers, loaded on demand ───────────────────────── */

  const TOKENIZERS = {
    gpt2: { file: "tiktoken-gpt2.js", label: "GPT-2 (2019)", vocab: 50257 },
    cl100k: { file: "tiktoken-cl100k.js", label: "GPT-4 (2023)", vocab: 100277 },
  };
  const scripts = {};
  const encoders = {};

  function loadScript(src) {
    if (!scripts[src]) {
      scripts[src] = new Promise((resolve, reject) => {
        const node = document.createElement("script");
        node.src = `${src}?v=${VENDOR_VERSION}`;
        node.onload = resolve;
        node.onerror = () => {
          delete scripts[src];
          reject(new Error(`Could not load ${src}`));
        };
        document.head.appendChild(node);
      });
    }
    return scripts[src];
  }

  async function getEncoder(name) {
    if (encoders[name]) return encoders[name];
    await loadScript(`${VENDOR}tiktoken-core.js`);
    await loadScript(`${VENDOR}${TOKENIZERS[name].file}`);
    encoders[name] = new root.MLTiktoken(root.MLTiktokenRanks[name]);
    return encoders[name];
  }

  /* Special-token text is encoded as ordinary text instead of throwing. */
  const encodeText = (encoder, text) => encoder.encode(text, [], []);

  /* GPT-2 works on bytes, so one character (é, न, an emoji) can span
     several tokens that are not printable alone. Consecutive tokens that
     only decode together are shown as one chip with a token count. */
  function tokenChips(encoder, ids) {
    const groups = [];
    let i = 0;
    while (i < ids.length) {
      let j = i + 1;
      let text = encoder.decode(ids.slice(i, j));
      while (text.includes("\uFFFD") && j < ids.length && j - i < 8) {
        j += 1;
        text = encoder.decode(ids.slice(i, j));
      }
      if (text.includes("\uFFFD")) {
        j = i + 1;
        text = encoder.decode([ids[i]]);
      }
      groups.push({ text, ids: ids.slice(i, j) });
      i = j;
    }
    return groups
      .map((group, k) => {
        const piece = group.text.replace(/\n/g, "⏎").replace(/ /g, "·");
        const multi = group.ids.length > 1;
        return `<span class="g-tok g-tok-${k % 5}${multi ? " g-tok-multi" : ""}" title="token id${multi ? "s" : ""} ${group.ids.join(", ")}">${esc(piece) || "∅"}${multi ? `<sup>×${group.ids.length}</sup>` : ""}</span>`;
      })
      .join("");
  }

  /* Start loading a tokenizer only when its widget scrolls near view. */
  function whenNear(node, callback) {
    if (!node) return;
    if (!("IntersectionObserver" in root)) {
      callback();
      return;
    }
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        observer.disconnect();
        callback();
      }
    }, { rootMargin: "600px" });
    observer.observe(node);
  }

  /* ── Byte-pair encoding, run live on a small corpus ──────────── */

  function trainBpe(corpus, maxMerges) {
    const counts = {};
    corpus
      .toLowerCase()
      .split(/\s+/)
      .filter(Boolean)
      .forEach((word) => {
        counts[word] = (counts[word] || 0) + 1;
      });
    let words = Object.entries(counts).map(([word, count]) => ({ word, count, parts: [...word, "_"] }));
    const base = new Set(words.flatMap((entry) => entry.parts));
    const merges = [];
    for (let step = 0; step < maxMerges; step += 1) {
      const pairs = new Map();
      words.forEach(({ parts, count }) => {
        for (let i = 0; i < parts.length - 1; i += 1) {
          const key = `${parts[i]}\u0000${parts[i + 1]}`;
          pairs.set(key, (pairs.get(key) || 0) + count);
        }
      });
      let best = null;
      let bestCount = 1;
      pairs.forEach((count, key) => {
        if (count > bestCount) {
          best = key;
          bestCount = count;
        }
      });
      if (!best) break;
      const [a, b] = best.split("\u0000");
      words = words.map((entry) => {
        const out = [];
        for (let i = 0; i < entry.parts.length; i += 1) {
          if (i < entry.parts.length - 1 && entry.parts[i] === a && entry.parts[i + 1] === b) {
            out.push(a + b);
            i += 1;
          } else {
            out.push(entry.parts[i]);
          }
        }
        return { ...entry, parts: out };
      });
      merges.push({ a, b, count: bestCount });
    }
    const tokens = words.reduce((acc, entry) => acc + entry.parts.length * entry.count, 0);
    return { words, merges, vocab: base.size + merges.length, tokens };
  }

  /* ════════════════════════════════════════════════════════════════
     PRIMER GUIDE
     ════════════════════════════════════════════════════════════════ */

  function mountPrimerGuide(rootNode) {
    const ui = UI();
    const { para, plain, deeper, takeaway, figure, chapter, guideToc, flow, mathBlock, svgSlot, heatmap, barList, quiz, fmt, pct } = ui;
    const math = M();

    const chapters = [
      { id: "pr-tokens", title: "From text to tokens", blurb: "what a model actually reads" },
      { id: "pr-bpe", title: "How a tokenizer is built", blurb: "byte-pair encoding" },
      { id: "pr-embed", title: "From tokens to vectors", blurb: "embeddings" },
      { id: "pr-dot", title: "Measuring similarity", blurb: "dot product and cosine" },
      { id: "pr-softmax", title: "From scores to probabilities", blurb: "softmax and temperature" },
      { id: "pr-rnn", title: "Reading in order", blurb: "recurrent networks and their limits" },
      { id: "pr-bottleneck", title: "Translation and the bottleneck", blurb: "where attention was born" },
      { id: "pr-notation", title: "Notation you'll meet", blurb: "a cheat sheet for the paper" },
      { id: "pr-recap", title: "Recap and self-check", blurb: "ready for the Attention guide" },
    ];

    const embedWords = ["cat", "dog", "kitten", "car", "truck", "apple"];
    const embedVecs = { cat: [0.9, 0.8], dog: [0.8, 0.9], kitten: [0.95, 0.65], car: [-0.8, 0.5], truck: [-0.9, 0.35], apple: [0.1, -0.9] };

    function body() {
      return `
        <div class="g-intro">
          ${para("Before attention can make sense, a few building blocks have to be second nature: how text becomes numbers, how a network measures similarity, how scores become probabilities, and why the older way of reading text (one word at a time) ran into trouble. This guide covers exactly those, with live widgets you can edit.")}
          ${para("Already comfortable with all of them? Skip straight to the <a href=\"./algorithm.html?id=attention\">Attention guide</a>.")}
          ${guideToc(chapters)}
        </div>

        ${chapter("pr-tokens", 1, "Input", "From text to tokens", `
          ${plain(para("A language model never sees letters or words directly. Text is first cut into <strong>tokens</strong>: common words stay whole (“ the”, “ cat”), rarer words are split into familiar pieces (“unbeliev” + “ably”), and anything else falls back to single characters or bytes. Each token has an ID number in a fixed vocabulary; the model only ever sees those IDs."))}
          ${widget("a real tokenizer", `
            <textarea class="g-text" data-input="text" rows="3" spellcheck="false">Transformers are unbelievably powerful! Tokenization isn't always intuitive: 12345 + ChatGPT + नमस्ते.</textarea>
            ${segmented("model", Object.entries(TOKENIZERS).map(([key, entry]) => [key, entry.label]), "gpt2")}
            <div data-out="tok"></div>`)}
          ${para("A small <sup>×3</sup> on a chip means that one piece of text needed 3 tokens: GPT-2 works on raw bytes, and characters outside basic English (é, न, emoji) take several bytes each.")}
          ${para("Things to notice as you type: a leading space is part of the token (shown as ·), so “ cat” and “cat” are different tokens. Numbers are chopped into arbitrary chunks, which is one reason models are clumsy at arithmetic. Non-English text costs many more tokens per word, especially in GPT-2, whose vocabulary was built mostly from English web pages. GPT-4's tokenizer has twice the vocabulary and is more efficient almost everywhere.")}
          ${deeper("The math", `
            ${para("A tokenizer is a fixed function from text to a list of integers, decided <em>before</em> training: <code>text → [t₁, t₂, …, tₙ]</code> with every tᵢ ∈ {0, …, V−1}. V is the vocabulary size: about 37,000 in the Transformer paper, 50,257 in GPT-2, about 100,000 in GPT-4. Everything the model computes has length n, so the number of tokens directly sets the cost.")}
          `)}
          ${takeaway("Models read token IDs, not words. Common words are one token; rare words are split into pieces.")}
        `)}

        ${chapter("pr-bpe", 2, "Input", "How a tokenizer is built: byte-pair encoding", `
          ${plain(para("Where does the vocabulary come from? <strong>Byte-pair encoding (BPE)</strong> learns it from data with a very simple loop: start with single characters, find the pair of neighbouring symbols that appears most often, glue that pair into a new symbol, and repeat. Frequent words end up as single symbols; rare ones stay in pieces."))}
          ${widget("watch BPE learn", `
            <textarea class="g-text" data-input="corpus" rows="3" spellcheck="false">low low low low low lower lower newest newest newest newest newest newest widest widest widest</textarea>
            ${range("merges", "Number of merges", 0, 12, 1, 4)}
            <div data-out="bpe"></div>`)}
          ${para("The default corpus is the classic example from the paper that introduced BPE for translation (Sennrich et al., 2016). “_” marks the end of a word. After a few merges “est_” becomes one symbol, because “newest” and “widest” share it, which is exactly the kind of reusable piece a tokenizer wants.")}
          ${deeper("Details", `
            ${para("GPT-2 runs the same loop on <em>bytes</em> rather than characters, so any text, in any language and even emoji, can be encoded without an “unknown” token. It stopped after about 50,000 merges. The Transformer paper used a shared English–German BPE vocabulary of about 37,000 tokens.")}
          `)}
          ${takeaway("BPE builds a vocabulary by repeatedly merging the most frequent pair of symbols.")}
        `)}

        ${chapter("pr-embed", 3, "Input", "From tokens to vectors: embeddings", `
          ${plain(para("A token ID like 2093 is just a label: 2093 is not “bigger” than 41 in any meaningful way. So each ID is swapped for a learned list of numbers, its <strong>embedding</strong>. During training, tokens used in similar ways drift to similar vectors: “cat” ends up near “kitten” and far from “truck”."))}
          ${figure("3.1", "A toy 2-number embedding. Animals cluster together, vehicles cluster together, and “apple” sits apart. Real embeddings have hundreds of numbers, but the clustering idea is the same.", svgSlot("embed-map", "0 0 420 360", "word embeddings in 2D"))}
          ${figure("3.2", "An embedding lookup is a matrix multiply. The one-hot vector for token #2 (“dog”) selects row 2 of the embedding matrix E.",
            `<div class="tl-mats-row">
              ${heatmap([[0, 1, 0, 0, 0, 0]], { rows: ["one-hot"], cols: embedWords, maxAbs: 1, title: "one-hot(dog)", shape: "1 × 6" })}
              <div class="g-times">×</div>
              ${heatmap(embedWords.map((w) => embedVecs[w]), { rows: embedWords, cols: ["e₁", "e₂"], maxAbs: 1, focusRow: 1, title: "E", shape: "6 × 2" })}
              <div class="g-times">=</div>
              ${heatmap([embedVecs.dog], { rows: ["dog"], cols: ["e₁", "e₂"], maxAbs: 1, title: "embedding", shape: "1 × 2" })}
            </div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`x_t = \text{onehot}(t)\,E = E_{t,:},\qquad E \in \mathbb{R}^{V \times d_{\text{model}}}`)}
            ${para("So an embedding layer is a dense layer without bias, fed a one-hot input. Real implementations skip the multiply and just read the row. GPT-2's E is 50,257 × 768, about 38.6 million numbers: roughly a third of the whole model.")}
          `)}
          ${takeaway("Each token ID becomes a learned vector; tokens used alike end up with similar vectors.")}
        `)}

        ${chapter("pr-dot", 4, "Tool", "Measuring similarity: dot product and cosine", `
          ${plain(para("Attention constantly asks “how related are these two vectors?”. The answer is the <strong>dot product</strong>: multiply matching entries and add them up. It is large when the vectors point the same way, zero when they are at right angles, and negative when they point apart. The <strong>cosine similarity</strong> is the same thing with lengths divided out, so it only measures direction."))}
          ${widget("drag the two arrow tips", `
            <div class="g-two g-two-wide">
              <div>${svgSlot("dot-drag", "0 0 380 380", "two draggable vectors")}</div>
              <div data-out="dot"></div>
            </div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`a\cdot b = \sum_i a_i b_i = |a|\,|b|\cos\theta,\qquad \cos\theta = \frac{a\cdot b}{|a|\,|b|}`)}
            ${para("Attention uses the plain dot product (not the cosine), so <em>length matters</em>: a long query vector asks more “loudly”. That is one reason the scores need scaling, as the Attention guide explains in Chapter 9.")}
          `)}
          ${takeaway("Dot product = alignment × lengths. It is the similarity score at the heart of attention.")}
        `)}

        ${chapter("pr-softmax", 5, "Tool", "From scores to probabilities: softmax and temperature", `
          ${plain(para("Networks produce raw scores (any real numbers). <strong>Softmax</strong> turns a list of scores into probabilities: all positive, adding up to 1, with the biggest score getting the biggest share. Dividing the scores by a <strong>temperature</strong> first controls how decisive it is: low temperature makes it nearly pick one winner; high temperature spreads the probability out."))}
          ${widget("edit the scores and the temperature", `
            <div class="g-inputs">${["cat", "tired", "sat", "the"]
              .map((label, i) => `<label><span>${label}</span><input type="number" step="0.5" data-input="s${i}" value="${[2, 1, 0.5, -1][i]}" /></label>`)
              .join("")}</div>
            ${range("temp", "Temperature T", 0.1, 5, 0.1, 1)}
            <div data-out="soft"></div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`p_j = \frac{e^{z_j/T}}{\sum_k e^{z_k/T}}`)}
            ${para("Adding the same constant to every score changes nothing (it cancels), only differences matter. As T → 0 softmax becomes “pick the max”; as T → ∞ it becomes uniform. You will meet softmax twice more: inside attention (to turn match scores into weights) and at the output of a language model (to turn word scores into next-word probabilities), where temperature is a setting you can change in most chatbots.")}
          `)}
          ${takeaway("Softmax turns scores into probabilities; temperature sets how sharp they are.")}
        `)}

        ${chapter("pr-rnn", 6, "History", "Reading in order: recurrent networks", `
          ${plain(para("Before Transformers, the standard way to read a sentence was a <strong>recurrent neural network (RNN)</strong>: one small network applied to each word in turn, passing a “memory” vector from each step to the next. It is like reading a book while only allowed to keep a single sticky note of everything so far."))}
          ${figure("6.1", "An RNN unrolled over time. The same cell (same weights) runs at every step, taking the current word and the previous memory h, and producing a new memory.",
            flow(["h₀", "→", "<span>cell</span><small>+ “the”</small>", "→", "h₁", "→", "<span>cell</span><small>+ “cat”</small>", "→", "h₂", "→", "<span>cell</span><small>+ “sat”</small>", "→", "h₃", "→", "…"]))}
          ${para("This design has two built-in problems:")}
          <ol class="g-list">
            <li><strong>It forgets.</strong> Information from word 1 must survive being multiplied through every later step. If each step shrinks it, it fades away; if each step grows it, it explodes. Try it below.</li>
            <li><strong>It is slow.</strong> Step 50 can't start until step 49 is done, so a GPU, built to do thousands of things at once, mostly waits.</li>
          </ol>
          ${widget("how much of word 1 survives", `
            ${range("w", "Per-step factor w (recurrent weight × activation slope)", 0.5, 1.5, 0.05, 0.8)}
            <div data-out="rnn"></div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`h_t = \tanh\!\big(W_{hh}\,h_{t-1} + W_{xh}\,x_t + b\big),\qquad \frac{\partial h_t}{\partial h_1} = \prod_{k=2}^{t} W_{hh}^{\top}\operatorname{diag}\!\big(\tanh'(\cdot)\big)`)}
            ${para("The gradient that teaches step 1 is a product of t − 1 matrices. Its size behaves like wᵗ⁻¹ for a typical factor w: below 1 it vanishes, above 1 it explodes. LSTMs and GRUs (1997, 2014) added gates to protect the memory, which helped a lot, but did nothing about the step-by-step slowness.")}
          `)}
          ${takeaway("RNNs read in order but forget over long distances and can't be parallelised.")}
        `)}

        ${chapter("pr-bottleneck", 7, "History", "Translation and the bottleneck: where attention was born", `
          ${plain(para("For translation, an <strong>encoder</strong> RNN read the source sentence and a <strong>decoder</strong> RNN wrote the translation. The catch: the decoder only received the encoder's <em>final</em> memory, one vector meant to hold the meaning of the whole sentence. Long sentences didn't fit. In 2014, Bahdanau, Cho and Bengio let the decoder <strong>look back at every encoder step</strong> and take a weighted mix, choosing the weights fresh for each output word. That weighted look-back is attention."))}
          ${figure("7.1", "Before (top): everything squeezed through one vector. After (bottom): at each output step, the decoder attends to all encoder states.",
            `<div class="g-stack">
              ${flow(["I", "love", "cats", "→", "<span>one vector</span><small>the bottleneck</small>", "→", "ich", "liebe", "Katzen"])}
              ${flow(["I · love · cats", "→", "<span>all encoder states</span><small>kept</small>", "⇢", "<span>decoder looks back</span><small>weighted mix per word</small>", "→", "ich · liebe · Katzen"])}
            </div>`)}
          ${para("The 2017 Transformer paper asked a bold question: if attention is what makes this work, do we need the RNN at all? Its title is the answer.")}
          ${takeaway("Attention was invented to let a decoder look back at every input word instead of one summary vector.")}
        `)}

        ${chapter("pr-notation", 8, "Reference", "Notation you'll meet", `
          ${figure("8.1", "Symbols used in the Attention and Transformer guides and in the paper.",
            `<div class="table-panel"><table>
              <thead><tr><th>Symbol</th><th>Meaning</th><th>Paper (base)</th></tr></thead>
              <tbody>
                <tr><td>n</td><td>number of tokens in the sequence</td><td>varies</td></tr>
                <tr><td>V</td><td>vocabulary size</td><td>≈ 37,000</td></tr>
                <tr><td>d<sub>model</sub></td><td>width of every token vector</td><td>512</td></tr>
                <tr><td>X</td><td>input matrix, one row per token (n × d<sub>model</sub>)</td><td>—</td></tr>
                <tr><td>Q, K, V</td><td>queries, keys, values</td><td>—</td></tr>
                <tr><td>W<sup>Q</sup>, W<sup>K</sup>, W<sup>V</sup>, W<sup>O</sup></td><td>learned projection matrices</td><td>—</td></tr>
                <tr><td>h</td><td>number of attention heads</td><td>8</td></tr>
                <tr><td>d<sub>k</sub>, d<sub>v</sub></td><td>width of each head's keys and values</td><td>64</td></tr>
                <tr><td>d<sub>ff</sub></td><td>hidden width of the feed-forward network</td><td>2048</td></tr>
                <tr><td>N</td><td>number of stacked layers</td><td>6</td></tr>
                <tr><td>PE</td><td>positional encoding</td><td>sinusoidal</td></tr>
              </tbody></table></div>`)}
        `)}

        ${chapter("pr-recap", 9, "Recap", "Recap and self-check", `
          <ol class="g-recap">
            <li>Text → <b>tokens</b> (learned by BPE) → <b>IDs</b> → <b>embedding vectors</b>.</li>
            <li>The <b>dot product</b> measures how aligned two vectors are.</li>
            <li><b>Softmax</b> turns scores into probabilities; <b>temperature</b> sets their sharpness.</li>
            <li><b>RNNs</b> read in order, forget over distance and can't run in parallel.</li>
            <li><b>Attention</b> began as a decoder looking back at every encoder state.</li>
          </ol>
          ${quiz([
            { q: "Why are “ cat” and “cat” different tokens?", a: "The leading space is part of the token. Words in the middle of a sentence usually start with a space, so the tokenizer learns them as separate pieces." },
            { q: "Why is a one-hot vector times E the same as a table lookup?", a: "All entries are 0 except a single 1, so the product picks out exactly one row of E." },
            { q: "Add 10 to every score before softmax. What changes?", a: "Nothing: e^(z+10) = e^10 · e^z, and the common factor cancels in the division." },
            { q: "Name the two problems that made RNNs a poor fit for long text.", a: "Forgetting (vanishing gradients over many steps) and slowness (each step waits for the previous one)." },
          ])}
          <div class="tl-next">
            <div><div class="eyebrow">Next guide</div><strong>Attention, from the ground up</strong><p class="caption">Query, key and value, built step by step with real numbers.</p></div>
            <a class="button primary" href="./algorithm.html?id=attention">Continue to Attention →</a>
          </div>
        `)}
      `;
    }

    const state = {
      tok: { text: null, model: "gpt2" },
      bpe: { corpus: null, merges: 4 },
      dot: { a: { x: 0.9, y: 0.3 }, b: { x: 0.35, y: 0.85 } },
      soft: { s0: 2, s1: 1, s2: 0.5, s3: -1, temp: 1 },
      rnn: { w: 0.8 },
    };

    function drawEmbedMap(svg) {
      const chart = U().makeChart(svg, { xDomain: [-1.2, 1.2], yDomain: [-1.2, 1.2], title: "Toy embeddings (2 numbers per word)" });
      embedWords.forEach((word, i) => {
        const [x, y] = embedVecs[word];
        const color = i < 3 ? C().a : i < 5 ? C().c : C().b;
        svg.appendChild(U().svgEl("circle", { cx: chart.xScale(x), cy: chart.yScale(y), r: 6, fill: color }));
        ui.svgText(svg, chart.xScale(x) + 9, chart.yScale(y) + 4, word, { class: "svg-title" });
      });
    }

    function drawDot(svg, host) {
      const s = state.dot;
      const chart = U().makeChart(svg, { xDomain: [-1.2, 1.2], yDomain: [-1.2, 1.2], title: "Drag the dots" });
      ui.arrowMarker(svg, "pr-dot-a", C().c);
      ui.arrowMarker(svg, "pr-dot-b", C().b);
      const o = [chart.xScale(0), chart.yScale(0)];
      [["a", C().c], ["b", C().b]].forEach(([key, color]) => {
        const p = s[key];
        svg.appendChild(U().svgEl("line", { x1: o[0], y1: o[1], x2: chart.xScale(p.x), y2: chart.yScale(p.y), stroke: color, "stroke-width": 3, "marker-end": `url(#pr-dot-${key})` }));
        const handle = U().svgEl("circle", { cx: chart.xScale(p.x), cy: chart.yScale(p.y), r: 11, fill: color, opacity: 0.35, stroke: color, "stroke-width": 2 });
        svg.appendChild(handle);
        ui.svgText(svg, chart.xScale(p.x) + 13, chart.yScale(p.y) - 8, key, { class: "svg-title" });
        U().draggable(handle, svg, chart, (pos) => {
          s[key] = { x: Math.round(pos.x * 100) / 100, y: Math.round(pos.y * 100) / 100 };
          drawDot(svg, host);
        });
      });
      const a = [s.a.x, s.a.y];
      const b = [s.b.x, s.b.y];
      const dot = math.dotV(a, b);
      const la = Math.hypot(...a);
      const lb = Math.hypot(...b);
      const cos = la && lb ? dot / (la * lb) : 0;
      const angle = (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;
      const out = host.querySelector('[data-out="dot"]');
      out.innerHTML = `
        <div class="g-metrics">
          <div><span>a</span><b>(${fmt(a[0])}, ${fmt(a[1])})</b></div>
          <div><span>b</span><b>(${fmt(b[0])}, ${fmt(b[1])})</b></div>
          <div><span>a · b</span><b>${fmt(dot, 3)}</b></div>
          <div><span>cos θ</span><b>${fmt(cos, 3)}</b></div>
          <div><span>angle θ</span><b>${Math.round(angle)}°</b></div>
        </div>
        <p class="caption">a · b = ${fmt(a[0])}·${fmt(b[0])} + ${fmt(a[1])}·${fmt(b[1])} = ${fmt(dot, 3)}. ${
          angle < 30 ? "Pointing the same way: strongly related." : angle > 150 ? "Pointing apart: negative score." : Math.abs(angle - 90) < 15 ? "Close to a right angle: unrelated (score ≈ 0)." : "Partly aligned."
        } Try making one arrow longer without turning it: the dot product grows, the cosine doesn't.</p>`;
    }

    function render() {
      rootNode.innerHTML = body();
      ui.drawFigures(rootNode, { "embed-map": drawEmbedMap });

      const widgets = rootNode.querySelectorAll(".g-widget");
      const [tokHost, bpeHost, dotHost, softHost, rnnHost] = widgets;

      /* Tokenizer: real GPT-2 / GPT-4 encodings. */
      const tokArea = tokHost.querySelector("textarea");
      if (state.tok.text !== null) tokArea.value = state.tok.text;
      state.tok.text = tokArea.value;
      let tokReady = false;
      const renderTok = async (host, s) => {
        const out = host.querySelector('[data-out="tok"]');
        if (!tokReady) {
          out.innerHTML = '<p class="caption">The tokenizer loads when you scroll here.</p>';
          return;
        }
        out.innerHTML = `<p class="caption">Loading the ${TOKENIZERS[s.model].label} tokenizer…</p>`;
        try {
          const encoder = await getEncoder(s.model);
          if (s.model !== state.tok.model) return;
          const ids = encodeText(encoder, s.text);
          const words = s.text.trim() ? s.text.trim().split(/\s+/).length : 0;
          out.innerHTML = `
            <div class="g-toks">${tokenChips(encoder, ids) || '<span class="caption">Type something above.</span>'}</div>
            <div class="g-metrics">
              <div><span>characters</span><b>${fmtInt(s.text.length)}</b></div>
              <div><span>words</span><b>${fmtInt(words)}</b></div>
              <div><span>tokens</span><b>${fmtInt(ids.length)}</b></div>
              <div><span>characters per token</span><b>${ids.length ? (s.text.length / ids.length).toFixed(2) : "—"}</b></div>
              <div><span>vocabulary size</span><b>${fmtInt(TOKENIZERS[s.model].vocab)}</b></div>
            </div>
            <details class="g-ids"><summary>Show the token IDs the model actually receives</summary><code>[${ids.join(", ")}]</code></details>`;
        } catch (error) {
          out.innerHTML = '<p class="caption">The tokenizer could not be loaded. Check your connection and reload the page.</p>';
        }
      };
      mountWidget(tokHost, state.tok, renderTok);
      whenNear(tokHost, () => {
        tokReady = true;
        renderTok(tokHost, state.tok);
      });

      /* BPE trainer. */
      const corpusArea = bpeHost.querySelector("textarea");
      if (state.bpe.corpus !== null) corpusArea.value = state.bpe.corpus;
      state.bpe.corpus = corpusArea.value;
      bpeHost.querySelector('[data-input="merges"]').value = state.bpe.merges;
      mountWidget(bpeHost, state.bpe, (host, s) => {
        const result = trainBpe(s.corpus, s.merges);
        host.querySelector('[data-out="bpe"]').innerHTML = `
          <div class="g-two">
            <div>
              <div class="tl-subhead">Merges learned, in order</div>
              ${result.merges.length
                ? `<ol class="g-merges">${result.merges.map((m) => `<li><code>${esc(m.a)}</code> + <code>${esc(m.b)}</code> → <code>${esc(m.a + m.b)}</code> <span class="caption">seen ${m.count}×</span></li>`).join("")}</ol>`
                : '<p class="caption">No merges yet: every word is spelled out letter by letter.</p>'}
            </div>
            <div>
              <div class="tl-subhead">How each word is now split</div>
              <div class="g-bpe-words">${result.words
                .map((w) => `<div><span class="caption">${esc(w.word)} ×${w.count}</span><div class="g-toks">${w.parts.map((p, i) => `<span class="g-tok g-tok-${i % 5}">${esc(p)}</span>`).join("")}</div></div>`)
                .join("")}</div>
            </div>
          </div>
          <div class="g-metrics">
            <div><span>vocabulary size</span><b>${result.vocab}</b></div>
            <div><span>tokens to encode the corpus</span><b>${result.tokens}</b></div>
          </div>`;
      });

      /* Dot product. */
      drawDot(dotHost.querySelector('[data-fig="dot-drag"]'), dotHost);

      /* Softmax with temperature. */
      ["s0", "s1", "s2", "s3", "temp"].forEach((key) => {
        const input = softHost.querySelector(`[data-input="${key}"]`);
        if (input) input.value = state.soft[key];
      });
      mountWidget(softHost, state.soft, (host, s) => {
        const labels = ["cat", "tired", "sat", "the"];
        const z = [s.s0, s.s1, s.s2, s.s3].map((v) => (Number.isFinite(v) ? v : 0));
        const p = math.softmax(z.map((v) => v / s.temp));
        const exps = z.map((v) => Math.exp(v / s.temp));
        host.querySelector('[data-out="soft"]').innerHTML = `
          ${barList(p, labels, { max: 1, asPercent: true, highlight: p.indexOf(Math.max(...p)) })}
          <p class="caption">e<sup>z/T</sup>: ${exps.map((v, i) => `${labels[i]} ${v < 1000 ? v.toFixed(2) : v.toExponential(1)}`).join(" · ")}. Total ${exps.reduce((a, b) => a + b, 0).toFixed(2)}. ${
            s.temp < 0.4 ? "Low temperature: nearly all probability on the top score." : s.temp > 2.5 ? "High temperature: probabilities flatten toward equal." : ""
          }</p>`;
      }, { formatOutput: (key, value) => Number(value).toFixed(1) });

      /* RNN memory fade. */
      rnnHost.querySelector('[data-input="w"]').value = state.rnn.w;
      mountWidget(rnnHost, state.rnn, (host, s) => {
        const steps = [1, 2, 5, 10, 20, 50];
        const values = steps.map((k) => Math.pow(s.w, k - 1));
        const shown = values.map((v) => Math.min(v, 1e6));
        host.querySelector('[data-out="rnn"]').innerHTML = `
          ${barList(shown.map((v) => Math.log10(1 + v)), steps.map((k) => `step ${k}`), { max: Math.max(...shown.map((v) => Math.log10(1 + v)), 0.4), digits: 2 })}
          <div class="g-metrics">${steps.map((k, i) => `<div><span>signal at step ${k}</span><b>${values[i] < 1e-3 ? values[i].toExponential(1) : values[i] > 1e4 ? values[i].toExponential(1) : values[i].toFixed(3)}</b></div>`).join("")}</div>
          <p class="caption">Bars use a log scale. ${
            s.w < 0.97 ? `With w = ${s.w.toFixed(2)}, word 1's influence after 50 steps is ${values[5].toExponential(1)}: effectively forgotten (vanishing gradient).` : s.w > 1.03 ? `With w = ${s.w.toFixed(2)}, the signal grows without bound (exploding gradient), and training becomes unstable.` : "Only when w is almost exactly 1 does the signal survive. That knife-edge is why plain RNNs struggled."
          }</p>`;
      }, { formatOutput: (key, value) => Number(value).toFixed(2) });
    }

    render();
    U().onRedraw(render);
    ui.startProgressBar();
  }

  /* ════════════════════════════════════════════════════════════════
     LLM GUIDE
     ════════════════════════════════════════════════════════════════ */

  /* Published configurations (layers, query heads, key/value heads, head
     width). Used for the KV-cache calculator. */
  const KV_MODELS = {
    gpt2: { label: "GPT-2 small (2019)", layers: 12, heads: 12, kv: 12, dHead: 64 },
    llama2_7b: { label: "Llama 2 7B (2023)", layers: 32, heads: 32, kv: 32, dHead: 128 },
    mistral_7b: { label: "Mistral 7B (2023)", layers: 32, heads: 32, kv: 8, dHead: 128 },
    llama3_8b: { label: "Llama 3 8B (2024)", layers: 32, heads: 32, kv: 8, dHead: 128 },
    llama2_70b: { label: "Llama 2 70B (2023)", layers: 80, heads: 64, kv: 8, dHead: 128 },
  };

  const PARAM_PRESETS = {
    gpt2: { label: "GPT-2 small", L: 12, d: 768, V: 50257, ctx: 1024, real: "124 M" },
    gpt2xl: { label: "GPT-2 XL", L: 48, d: 1600, V: 50257, ctx: 1024, real: "1.5 B" },
    gpt3: { label: "GPT-3", L: 96, d: 12288, V: 50257, ctx: 2048, real: "175 B" },
  };

  const SAMPLE_WORDS = ["mat", "floor", "sofa", "bed", "roof", "keyboard", "moon", "banana"];
  const SAMPLE_LOGITS = [3.2, 2.5, 2.2, 1.9, 1.2, 0.6, -0.4, -1.5];

  function sampleDistribution(logits, temp, topK, topP) {
    const probs = M().softmax(logits.map((z) => z / temp));
    const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
    const keep = new Set();
    let cumulative = 0;
    order.forEach(([p, i], rank) => {
      if (rank < topK && cumulative < topP) {
        keep.add(i);
        cumulative += p;
      }
    });
    const kept = probs.map((p, i) => (keep.has(i) ? p : 0));
    const total = kept.reduce((a, b) => a + b, 0) || 1;
    return { before: probs, after: kept.map((p) => p / total), keep };
  }

  function mountLlmGuide(rootNode) {
    const ui = UI();
    const { para, plain, deeper, takeaway, figure, chapter, guideToc, flow, mathBlock, svgSlot, heatmap, barList, quiz, fmt, pct } = ui;

    const chapters = [
      { id: "llm-families", title: "Three families, one block", blurb: "encoder, decoder, or both" },
      { id: "llm-objective", title: "One job: predict the next token", blurb: "the training objective" },
      { id: "llm-forward", title: "A prompt's trip through GPT-2", blurb: "shapes, with a real tokenizer" },
      { id: "llm-sampling", title: "Choosing the next token", blurb: "temperature, top-k, top-p" },
      { id: "llm-block", title: "What changed inside the block", blurb: "pre-norm, RMSNorm, GELU, SwiGLU" },
      { id: "llm-rope", title: "Positions that rotate", blurb: "rotary position embeddings (RoPE)" },
      { id: "llm-kv", title: "Making generation fast", blurb: "KV cache, multi-query and grouped-query attention" },
      { id: "llm-moe", title: "Bigger but not slower", blurb: "mixture of experts" },
      { id: "llm-scale", title: "Scale", blurb: "counting the parameters" },
      { id: "llm-assistant", title: "From text predictor to assistant", blurb: "instruction tuning and RLHF" },
      { id: "llm-recap", title: "Recap and self-check", blurb: "" },
    ];

    const moeTokens = ["The", " cat", " sat", " on", " the", " mat"];
    const moeScores = [
      [2.1, 0.2, 0.1, 1.4, 0.0, 0.3, 0.2, 0.1],
      [0.1, 2.4, 0.3, 0.2, 1.6, 0.1, 0.0, 0.4],
      [0.2, 0.3, 2.2, 0.1, 0.2, 1.5, 0.3, 0.1],
      [1.8, 0.1, 0.2, 1.7, 0.1, 0.2, 0.4, 0.0],
      [2.0, 0.3, 0.1, 1.5, 0.2, 0.1, 0.1, 0.3],
      [0.1, 1.9, 0.2, 0.3, 2.0, 0.1, 0.5, 0.2],
    ];
    const moeProbs = moeScores.map((row) => M().softmax(row));
    const moeTop2 = moeProbs.map((row) => row.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 2).map(([, i]) => i));

    function body() {
      return `
        <div class="g-intro">
          ${para("ChatGPT, Claude, Gemini and Llama are all, at their core, the Transformer from the 2017 paper, with the encoder removed, a few parts upgraded, and everything made much bigger. This guide walks from the paper to a modern large language model (LLM), one change at a time, with live widgets where the numbers come from real tokenizers and published model sizes.")}
          ${para("It assumes the <a href=\"./algorithm.html?id=attention\">Attention</a> and <a href=\"./algorithm.html?id=transformer\">Transformer</a> guides.")}
          ${guideToc(chapters)}
        </div>

        ${chapter("llm-families", 1, "Architecture", "Three families, one block", `
          ${plain(para("The paper's model had two halves: an encoder that reads and a decoder that writes. Later models kept one half or both. <strong>Encoder-only</strong> models (BERT, 2018) read text in both directions and are good at understanding it. <strong>Decoder-only</strong> models (GPT, 2018 onward) can only look backwards, which makes them natural <em>writers</em>. Today's chat LLMs are almost all decoder-only."))}
          ${figure("1.1", "The same Transformer block, arranged three ways.",
            `<div class="g-cards g-cards-3">
              <div class="soft-box"><h4>Encoder–decoder</h4><p class="caption">The 2017 paper, T5, many translation systems.</p>${flow(["encoder ×N", "→", "decoder ×N"])}<p>Reads the input fully, then writes an output. Uses masked self-attention <em>and</em> cross-attention.</p></div>
              <div class="soft-box"><h4>Encoder-only</h4><p class="caption">BERT (2018), RoBERTa.</p>${flow(["encoder ×N"])}<p>Every token sees every token. Trained by hiding words and guessing them. Great for classification and search; can't write freely.</p></div>
              <div class="soft-box"><h4>Decoder-only</h4><p class="caption">GPT-1/2/3/4, Llama, Claude, Gemini.</p>${flow(["decoder ×N", "<small>(no cross-attention)</small>"])}<p>Causal mask: each token sees only the past. Trained to predict the next token. Any task can be phrased as “continue this text”.</p></div>
            </div>`, true)}
          ${para("Why did decoder-only win for general-purpose models? It is the simplest of the three (one stack, one mask), every position of every document is a training example, and translation, question answering and coding can all be expressed as continuing a piece of text.")}
          ${takeaway("A GPT-style LLM is the Transformer's decoder alone: masked self-attention + feed-forward, stacked many times.")}
        `)}

        ${chapter("llm-objective", 2, "Training", "One job: predict the next token", `
          ${plain(para("An LLM is trained on one task, repeated trillions of times: <strong>given the text so far, predict the next token</strong>. Thanks to the causal mask, one pass over a 1,000-token document produces 1,000 predictions at once, each position predicting its successor. Learning to do this well forces the model to absorb grammar, facts, and reasoning patterns, because all of them help predict what comes next."))}
          ${figure("2.1", "One sentence, five training examples, all computed in a single parallel pass.",
            `<div class="table-panel"><table>
              <thead><tr><th>Model sees</th><th>Must predict</th></tr></thead>
              <tbody>${[["The", " cat"], ["The cat", " sat"], ["The cat sat", " on"], ["The cat sat on", " the"], ["The cat sat on the", " mat"]]
                .map(([seen, next]) => `<tr><td>${esc(seen)}</td><td><b>${esc(next)}</b></td></tr>`)
                .join("")}</tbody></table></div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`\mathcal{L} = -\frac{1}{n}\sum_{t=1}^{n} \log p_\theta\big(x_t \mid x_{<t}\big)`)}
            ${para("This is plain cross-entropy, summed over positions. A model that gives the right next token probability 1 everywhere would have loss 0; guessing uniformly over GPT-2's 50,257 tokens would give log 50,257 ≈ 10.8. Pretraining runs this on enormous text collections: GPT-3 used about 300 billion tokens, Llama 3 about 15 trillion.")}
          `)}
          ${takeaway("Pretraining = next-token prediction at every position of a huge amount of text.")}
        `)}

        ${chapter("llm-forward", 3, "Architecture", "A prompt's trip through GPT-2", `
          ${plain(para("Let's follow a real prompt through GPT-2 small, the 2019 model whose sizes are public: 12 layers, vectors of 768 numbers, 12 heads, a 50,257-token vocabulary. Type anything; the tokens come from GPT-2's real tokenizer and every shape below is computed from them."))}
          ${widget("your prompt in GPT-2 small", `
            <textarea class="g-text" data-input="text" rows="2" spellcheck="false">The cat sat on the</textarea>
            <div data-out="forward"></div>`)}
          ${takeaway("Every layer keeps the shape n × 768; only the final Linear layer expands the last position to 50,257 word scores.")}
        `)}

        ${chapter("llm-sampling", 4, "Generation", "Choosing the next token: temperature, top-k, top-p", `
          ${plain(para("The model outputs a probability for every token in its vocabulary. How do we pick one? Always taking the top token (<strong>greedy</strong>) is safe but dull and repetitive. Instead, chatbots <em>sample</em>, with three dials: <strong>temperature</strong> (sharpen or flatten the probabilities), <strong>top-k</strong> (only consider the k most likely tokens) and <strong>top-p</strong> (only consider the smallest set of top tokens whose probabilities add up to p)."))}
          ${widget("tune the sampler for “The cat sat on the …”", `
            ${range("temp", "Temperature", 0.1, 2, 0.05, 0.8)}
            ${range("topK", "Top-k", 1, 8, 1, 8)}
            ${range("topP", "Top-p", 0.1, 1, 0.05, 0.9)}
            <div class="g-actions"><button type="button" class="button secondary" data-action="sample">Sample 20 times</button></div>
            <div data-out="sample"></div>`)}
          ${para("The word scores in this widget are illustrative; everything after them (temperature, filtering, renormalising, sampling) is computed live exactly as a real sampler does it.")}
          ${deeper("The math", `
            ${mathBlock(String.raw`p_i \propto \exp(z_i / T),\quad \text{then keep } i \in \text{top-}k \cap \text{top-}p,\ \text{renormalise, sample}`)}
            ${para("Top-p is also called <em>nucleus sampling</em> (Holtzman et al., 2019). Its advantage over top-k: when the model is confident, the nucleus is small; when it is unsure, the nucleus grows automatically.")}
          `)}
          ${takeaway("Temperature reshapes the distribution; top-k and top-p cut off its unlikely tail before sampling.")}
        `)}

        ${chapter("llm-block", 5, "Architecture", "What changed inside the block", `
          ${plain(para("The block of a 2024 LLM is recognisably the 2017 block: attention, then a feed-forward network, each with a residual connection. But several small parts were swapped for versions that train more stably or perform a little better at scale."))}
          ${figure("5.1", "Component by component, from the paper to a typical modern open model (Llama-style).",
            `<div class="table-panel"><table>
              <thead><tr><th>Part</th><th>2017 paper</th><th>Modern LLM</th><th>Why it changed</th></tr></thead>
              <tbody>
                <tr><td>Normalisation position</td><td>after each sub-layer (post-norm)</td><td>before each sub-layer (pre-norm)</td><td>Keeps a clean residual path; deep stacks train without careful warm-up.</td></tr>
                <tr><td>Normalisation type</td><td>LayerNorm</td><td>RMSNorm</td><td>Skips the mean subtraction: simpler and slightly faster, same quality.</td></tr>
                <tr><td>Activation</td><td>ReLU</td><td>GELU (GPT-2), SwiGLU (Llama, PaLM)</td><td>Smooth activations and gating gave consistently better results.</td></tr>
                <tr><td>Positions</td><td>sinusoidal, added once</td><td>learned (GPT-2), then RoPE</td><td>RoPE encodes <em>relative</em> position inside every attention layer.</td></tr>
                <tr><td>Attention heads</td><td>multi-head</td><td>grouped-query</td><td>Shares keys/values between heads to shrink the KV cache.</td></tr>
                <tr><td>Biases</td><td>yes</td><td>usually removed</td><td>Little benefit at scale; fewer parameters to tune.</td></tr>
                <tr><td>Feed-forward</td><td>dense</td><td>dense, or mixture of experts</td><td>More parameters at the same compute per token.</td></tr>
              </tbody></table></div>`, true)}
          ${figure("5.2", "Activation functions. ReLU has a sharp corner at 0; GELU and SiLU (used inside SwiGLU) are smooth and let small negative values through.", svgSlot("acts", "0 0 560 260", "ReLU, GELU and SiLU curves"))}
          ${deeper("The math", `
            ${mathBlock(String.raw`\text{pre-norm: } x \leftarrow x + \operatorname{Attn}(\operatorname{Norm}(x)),\quad x \leftarrow x + \operatorname{FFN}(\operatorname{Norm}(x))`)}
            ${mathBlock(String.raw`\operatorname{RMSNorm}(x) = \frac{x}{\sqrt{\tfrac{1}{d}\sum_i x_i^2 + \epsilon}}\odot g,\qquad \operatorname{SwiGLU}(x) = \big(\operatorname{SiLU}(xW_1)\odot xW_3\big)W_2`)}
            ${para("SwiGLU uses three matrices instead of two, so models shrink its hidden width to about 8/3 · d<sub>model</sub> to keep the parameter count the same.")}
          `)}
          ${takeaway("Same block, better parts: pre-norm with RMSNorm, smooth gated activations, RoPE, grouped-query attention.")}
        `)}

        ${chapter("llm-rope", 6, "Architecture", "Positions that rotate: RoPE", `
          ${plain(para("The paper <em>added</em> a position stamp to each word once, at the bottom. Most modern LLMs use <strong>rotary position embeddings (RoPE)</strong> instead: inside every attention layer, each query and key is <em>rotated</em> by an angle proportional to its position. When a query and a key are compared, the two rotations partly cancel, and what remains depends only on <strong>how far apart</strong> they are, not on where they sit in the text."))}
          ${widget("rotate a query and a key", `
            ${range("m", "Query position m", 0, 30, 1, 3)}
            ${range("n", "Key position n", 0, 30, 1, 7)}
            <div class="g-two g-two-wide">
              <div>${svgSlot("rope", "0 0 340 340", "rotated query and key")}</div>
              <div data-out="rope"></div>
            </div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`\tilde q_m = R(m\theta)\,q,\quad \tilde k_n = R(n\theta)\,k \;\Rightarrow\; \tilde q_m\cdot\tilde k_n = q^{\top}R\big((n-m)\theta\big)\,k`)}
            ${para("This is the same rotation property the paper used to justify sinusoids, applied directly to queries and keys. Real models rotate each pair of dimensions at its own frequency, exactly like the clock hands in the Transformer guide (Su et al., 2021).")}
          `)}
          ${takeaway("RoPE rotates queries and keys by position, so attention scores depend on relative distance.")}
        `)}

        ${chapter("llm-kv", 7, "Inference", "Making generation fast: the KV cache", `
          ${plain(para("When generating token 101, the keys and values of tokens 1–100 are exactly the same as they were a step earlier. So instead of recomputing them, the model <strong>stores</strong> them in a <strong>KV cache</strong> and only computes the new token's query, key and value. That turns generation from quadratic work per token into linear work, but the cache takes memory: two vectors per token, per layer, per key/value head."))}
          ${para("<strong>Multi-query attention</strong> (all heads share one key/value head) and <strong>grouped-query attention</strong> (groups of heads share one) shrink the cache dramatically with little loss in quality, which is why most recent open models use grouped-query attention.")}
          ${widget("KV-cache memory for real model configurations", `
            <label class="g-select"><span>Model</span><select data-input="model">${Object.entries(KV_MODELS).map(([key, m]) => `<option value="${key}">${m.label}</option>`).join("")}</select></label>
            ${range("ctxPow", "Context length (tokens)", 9, 17, 1, 13)}
            ${segmented("bytes", [["2", "16-bit"], ["1", "8-bit"]], "2")}
            <div data-out="kv"></div>`)}
          ${deeper("The math", `
            ${mathBlock(String.raw`\text{KV bytes} = 2 \times L \times n_{\text{kv heads}} \times d_{\text{head}} \times n_{\text{tokens}} \times \text{bytes per number}`)}
            ${para("The 2 counts keys and values. Model sizes are the published configurations; the widget computes the cache for one sequence.")}
          `)}
          ${takeaway("Caching keys and values makes generation fast; sharing them across heads (MQA/GQA) keeps the cache small.")}
        `)}

        ${chapter("llm-moe", 8, "Architecture", "Bigger but not slower: mixture of experts", `
          ${plain(para("A <strong>mixture-of-experts</strong> layer replaces the single feed-forward network with several (say 8) <strong>experts</strong> and a small <strong>router</strong>. For each token, the router scores the experts and only the top 2 run. The model can hold far more parameters than it uses for any one token, so it gets the knowledge of a big model at the speed of a smaller one."))}
          ${figure("8.1", "Router probabilities for each token over 8 experts (illustrative). Each token is sent to its top-2 experts (outlined); different kinds of tokens tend to prefer different experts.",
            `${heatmap(moeProbs, { rows: moeTokens.map((t) => esc(t.replace(" ", "·"))), cols: ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"], maxAbs: 1 })}
             <p class="caption">Top-2 per token: ${moeTokens.map((t, i) => `<b>${esc(t.trim())}</b> → E${moeTop2[i][0] + 1}, E${moeTop2[i][1] + 1}`).join(" · ")}</p>`)}
          ${para("A well-known open example is Mixtral 8×7B (2023): about 47 billion parameters in total, but only about 13 billion are used for each token, because each token runs through just 2 of its 8 experts per layer.")}
          ${takeaway("Mixture of experts adds parameters without adding compute per token: a router picks a few experts for each token.")}
        `)}

        ${chapter("llm-scale", 9, "Scale", "Counting the parameters", `
          ${plain(para("Where do the billions of parameters live? Each layer has about 4·d² in attention (W<sup>Q</sup>, W<sup>K</sup>, W<sup>V</sup>, W<sup>O</sup>) and about 8·d² in a feed-forward network with d<sub>ff</sub> = 4d. That's about <strong>12·d² per layer</strong>, plus the embedding table. The formula below reproduces the published sizes of the GPT models."))}
          ${widget("a parameter calculator", `
            ${segmented("preset", Object.entries(PARAM_PRESETS).map(([key, p]) => [key, p.label]), "gpt2")}
            <div class="g-inputs">
              <label><span>layers L</span><input type="number" data-input="L" min="1" step="1" value="12" /></label>
              <label><span>width d</span><input type="number" data-input="d" min="8" step="64" value="768" /></label>
              <label><span>vocabulary V</span><input type="number" data-input="V" min="100" step="1000" value="50257" /></label>
            </div>
            <div data-out="params"></div>`)}
          ${figure("9.1", "Published sizes of some milestone models.",
            `<div class="table-panel"><table>
              <thead><tr><th>Model</th><th>Year</th><th>Type</th><th>Parameters</th><th>Layers</th><th>d<sub>model</sub></th><th>Context</th></tr></thead>
              <tbody>
                <tr><td>Transformer (base)</td><td>2017</td><td>enc–dec</td><td>65 M</td><td>6 + 6</td><td>512</td><td>—</td></tr>
                <tr><td>GPT-1</td><td>2018</td><td>decoder</td><td>117 M</td><td>12</td><td>768</td><td>512</td></tr>
                <tr><td>BERT-base</td><td>2018</td><td>encoder</td><td>110 M</td><td>12</td><td>768</td><td>512</td></tr>
                <tr><td>GPT-2 XL</td><td>2019</td><td>decoder</td><td>1.5 B</td><td>48</td><td>1600</td><td>1024</td></tr>
                <tr><td>GPT-3</td><td>2020</td><td>decoder</td><td>175 B</td><td>96</td><td>12288</td><td>2048</td></tr>
                <tr><td>Llama 2 70B</td><td>2023</td><td>decoder</td><td>70 B</td><td>80</td><td>8192</td><td>4096</td></tr>
                <tr><td>Llama 3 8B</td><td>2024</td><td>decoder</td><td>8 B</td><td>32</td><td>4096</td><td>8192</td></tr>
              </tbody></table></div>`, true)}
          ${takeaway("Parameters ≈ 12·L·d² + V·d. Width counts twice: doubling d quadruples the per-layer size.")}
        `)}

        ${chapter("llm-assistant", 10, "Training", "From text predictor to assistant", `
          ${plain(para("A model trained only on next-token prediction is a brilliant <em>autocomplete</em>, not a helpful assistant: asked a question, it might continue with three more questions. Turning it into an assistant takes further training stages on much smaller, carefully made datasets."))}
          ${figure("10.1", "The typical training pipeline of a chat model.",
            flow(["<span>Pretraining</span><small>next token · trillions of tokens</small>", "→", "<span>Supervised fine-tuning</span><small>example conversations</small>", "→", "<span>Preference tuning</span><small>RLHF or DPO</small>", "→", "<span>Assistant</span>"]))}
          <ol class="g-list">
            <li><strong>Supervised fine-tuning (SFT).</strong> Continue training on example conversations written or checked by people: prompt → good answer. Same loss as pretraining, different data.</li>
            <li><strong>Preference tuning.</strong> People (or AI models following written principles) compare two answers and pick the better one. <em>RLHF</em> trains a reward model on these choices and then optimises the LLM against it with reinforcement learning (Ouyang et al., 2022). <em>DPO</em> skips the separate reward model and learns directly from the preference pairs (Rafailov et al., 2023).</li>
            <li><strong>Reasoning training</strong> (recent models). Reinforcement learning on problems with checkable answers, such as maths and code, rewarding correct final results; models learn to write out longer chains of reasoning before answering.</li>
          </ol>
          ${takeaway("Pretraining builds knowledge; fine-tuning and preference tuning shape behaviour.")}
        `)}

        ${chapter("llm-recap", 11, "Recap", "Recap and self-check", `
          <ol class="g-recap">
            <li>An LLM is a <b>decoder-only</b> Transformer trained to <b>predict the next token</b>.</li>
            <li>Generation repeats: run the model, <b>sample</b> a token (temperature, top-k, top-p), append, repeat, reusing the <b>KV cache</b>.</li>
            <li>Modern blocks use <b>pre-norm RMSNorm</b>, <b>SwiGLU</b>, <b>RoPE</b> and <b>grouped-query attention</b>; some use <b>mixture of experts</b>.</li>
            <li>Parameters ≈ <b>12·L·d² + V·d</b>.</li>
            <li><b>SFT</b> and <b>preference tuning</b> turn a text predictor into an assistant.</li>
          </ol>
          ${quiz([
            { q: "Why can one forward pass over a document give a training signal at every position?", a: "The causal mask means position t only sees tokens before it, so every position is a valid, independent next-token prediction." },
            { q: "Temperature 0.1 vs 1.5: which gives more varied text, and why?", a: "1.5. Dividing scores by a larger T flattens the softmax, so less likely tokens get picked more often." },
            { q: "Why does grouped-query attention matter more for long conversations?", a: "The KV cache grows linearly with context length; sharing keys and values across heads divides that memory by heads ÷ KV heads (4× for Llama 3 8B)." },
            { q: "With RoPE, does the score between a query at position 3 and a key at 7 equal the score between positions 13 and 17?", a: "Yes (for the same q and k vectors): the score depends only on the offset n − m = 4." },
            { q: "Roughly how many parameters does a 24-layer model of width 1024 have, with a 50,000-token vocabulary?", a: "12 · 24 · 1024² ≈ 302 M, plus 50,000 · 1024 ≈ 51 M, so about 353 M (close to GPT-2 medium's 355 M)." },
          ])}
          <div class="tl-next">
            <div><div class="eyebrow">Next guide</div><strong>Small LLMs: distillation, pruning and quantization</strong><p class="caption">How small models get good: training past Chinchilla, learning from a teacher, pruning and fewer bits per weight.</p></div>
            <a class="button primary" href="./algorithm.html?id=small-llm">Continue to Small LLMs →</a>
          </div>
        `)}
      `;
    }

    const state = {
      fwd: { text: null },
      samp: { temp: 0.8, topK: 8, topP: 0.9, draws: null },
      rope: { m: 3, n: 7 },
      kv: { model: "llama3_8b", ctxPow: 13, bytes: "2" },
      params: { preset: "gpt2", L: 12, d: 768, V: 50257 },
    };

    function drawActivations(svg) {
      const xs = U().linspace(-4, 4, 200);
      const phi = (x) => 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
      const curves = [
        ["ReLU", (x) => Math.max(0, x), C().c],
        ["GELU", (x) => x * phi(x), C().b],
        ["SiLU (Swish)", (x) => x / (1 + Math.exp(-x)), C().a],
      ];
      const chart = U().makeChart(svg, { xDomain: [-4, 4], yDomain: [-1, 4], title: "Three activation functions" });
      curves.forEach(([label, fn, color], i) => {
        svg.appendChild(U().svgEl("path", { d: U().pathFromPoints(xs.map((x) => ({ x, y: fn(x) })), chart.xScale, chart.yScale), fill: "none", stroke: color, "stroke-width": 2.6 }));
        svg.appendChild(U().svgEl("rect", { x: 70, y: 40 + i * 18, width: 12, height: 4, fill: color }));
        ui.svgText(svg, 88, 45 + i * 18, label);
      });
    }

    function drawRope(svg, s) {
      U().clear(svg);
      const theta = 0.35;
      const q = [1, 0.25];
      const k = [0.85, 0.55];
      const rot = (v, a) => [v[0] * Math.cos(a) - v[1] * Math.sin(a), v[0] * Math.sin(a) + v[1] * Math.cos(a)];
      const cx = 170;
      const cy = 170;
      const r = 120;
      svg.appendChild(U().svgEl("circle", { cx, cy, r, fill: "none", stroke: C().faint, "stroke-width": 1.5 }));
      ui.arrowMarker(svg, "rope-q", C().b);
      ui.arrowMarker(svg, "rope-k", C().c);
      [[rot(q, s.m * theta), C().b, "rope-q", `q at ${s.m}`], [rot(k, s.n * theta), C().c, "rope-k", `k at ${s.n}`]].forEach(([v, color, id, label]) => {
        const len = Math.hypot(...v);
        const x = cx + (v[0] / len) * r;
        const y = cy - (v[1] / len) * r;
        svg.appendChild(U().svgEl("line", { x1: cx, y1: cy, x2: x, y2: y, stroke: color, "stroke-width": 3.5, "marker-end": `url(#${id})` }));
        ui.svgText(svg, x + (x > cx ? 6 : -6), y + (y > cy ? 14 : -6), label, { "text-anchor": x > cx ? "start" : "end", class: "svg-title" });
      });
      svg.appendChild(U().svgEl("circle", { cx, cy, r: 4, fill: C().ink }));
    }

    function render() {
      rootNode.innerHTML = body();
      ui.drawFigures(rootNode, { acts: drawActivations });
      const [fwdHost, sampHost, ropeHost, kvHost, paramHost] = rootNode.querySelectorAll(".g-widget");

      /* Prompt through GPT-2: real tokens, published sizes. */
      const area = fwdHost.querySelector("textarea");
      if (state.fwd.text !== null) area.value = state.fwd.text;
      state.fwd.text = area.value;
      let ready = false;
      const renderForward = async (host, s) => {
        const out = host.querySelector('[data-out="forward"]');
        if (!ready) {
          out.innerHTML = '<p class="caption">The GPT-2 tokenizer loads when you scroll here.</p>';
          return;
        }
        try {
          const encoder = await getEncoder("gpt2");
          const ids = encodeText(encoder, s.text);
          const n = Math.max(ids.length, 1);
          out.innerHTML = `
            <div class="g-toks">${tokenChips(encoder, ids)}</div>
            ${ids.length > 1024 ? '<p class="caption"><b>Too long for GPT-2:</b> its context window is 1,024 tokens.</p>' : ""}
            <ol class="g-pipeline">
              <li><b>Token IDs</b> <code>[${ids.slice(0, 12).join(", ")}${ids.length > 12 ? ", …" : ""}]</code> <span class="tl-shape-tag">${n}</span></li>
              <li><b>Token embeddings + learned position embeddings</b> <span class="tl-shape-tag">${n} × 768</span></li>
              <li><b>12 decoder blocks</b>: each with 12 heads, so ${fmtInt(12 * n * n)} attention scores per layer (${n} × ${n} per head; the causal mask zeroes ${fmtInt((12 * n * (n - 1)) / 2)} of them) and a feed-forward layer 768 → 3072 → 768 <span class="tl-shape-tag">${n} × 768</span></li>
              <li><b>Final layer norm</b> <span class="tl-shape-tag">${n} × 768</span></li>
              <li><b>Linear to vocabulary</b> (shares the embedding matrix): one score for each of 50,257 tokens at each position <span class="tl-shape-tag">${n} × 50,257</span></li>
              <li><b>Softmax on the last row</b>: probabilities for the token after “${esc(encoder.decode(ids.slice(-1)) || "")}” <span class="tl-shape-tag">50,257</span></li>
            </ol>
            <p class="caption">Rough work for this prompt: about ${fmtBig(2 * 124e6 * n)} multiply-adds (≈ 2 × parameters × tokens).</p>`;
        } catch (error) {
          out.innerHTML = '<p class="caption">The tokenizer could not be loaded. Check your connection and reload the page.</p>';
        }
      };
      mountWidget(fwdHost, state.fwd, renderForward);
      whenNear(fwdHost, () => {
        ready = true;
        renderForward(fwdHost, state.fwd);
      });

      /* Sampler. */
      ["temp", "topK", "topP"].forEach((key) => {
        sampHost.querySelector(`[data-input="${key}"]`).value = state.samp[key];
      });
      mountWidget(sampHost, state.samp, (host, s) => {
        const dist = sampleDistribution(SAMPLE_LOGITS, s.temp, s.topK, s.topP);
        const counts = s.draws ? SAMPLE_WORDS.map((_, i) => s.draws.filter((d) => d === i).length) : null;
        host.querySelector('[data-out="sample"]').innerHTML = `
          <div class="g-two">
            <div><div class="tl-subhead">After temperature</div>${barList(dist.before, SAMPLE_WORDS, { max: 1, asPercent: true })}</div>
            <div><div class="tl-subhead">After top-k / top-p (what is sampled)</div>${barList(dist.after, SAMPLE_WORDS.map((w, i) => (dist.keep.has(i) ? w : `<s>${w}</s>`)), { max: 1, asPercent: true, color: C().b })}</div>
          </div>
          <p class="caption">${dist.keep.size} of ${SAMPLE_WORDS.length} candidates survive the filters.${counts ? ` Last 20 samples: ${SAMPLE_WORDS.map((w, i) => (counts[i] ? `${w} ×${counts[i]}` : "")).filter(Boolean).join(", ")}.` : ""}</p>`;
      }, {
        formatOutput: (key, value) => (key === "topK" ? String(value) : Number(value).toFixed(2)),
        sample: (s) => {
          const dist = sampleDistribution(SAMPLE_LOGITS, s.temp, s.topK, s.topP);
          s.draws = Array.from({ length: 20 }, () => {
            let r = Math.random();
            for (let i = 0; i < dist.after.length; i += 1) {
              r -= dist.after[i];
              if (r <= 0) return i;
            }
            return dist.after.length - 1;
          });
        },
      });

      /* RoPE. */
      ["m", "n"].forEach((key) => {
        ropeHost.querySelector(`[data-input="${key}"]`).value = state.rope[key];
      });
      mountWidget(ropeHost, state.rope, (host, s) => {
        const theta = 0.35;
        const q = [1, 0.25];
        const k = [0.85, 0.55];
        const rot = (v, a) => [v[0] * Math.cos(a) - v[1] * Math.sin(a), v[0] * Math.sin(a) + v[1] * Math.cos(a)];
        const score = (m, n) => M().dotV(rot(q, m * theta), rot(k, n * theta));
        drawRope(host.querySelector('[data-fig="rope"]'), s);
        const shifts = [0, 5, 10, 20];
        host.querySelector('[data-out="rope"]').innerHTML = `
          <div class="g-metrics">
            <div><span>offset n − m</span><b>${s.n - s.m}</b></div>
            <div><span>score q̃ · k̃</span><b>${fmt(score(s.m, s.n), 3)}</b></div>
          </div>
          <div class="tl-subhead">Shift both positions by the same amount</div>
          <table class="g-mini"><tr><th>positions (m, n)</th><th>score</th></tr>${shifts
            .map((d) => `<tr><td>(${s.m + d}, ${s.n + d})</td><td class="mono">${fmt(score(s.m + d, s.n + d), 3)}</td></tr>`)
            .join("")}</table>
          <p class="caption">Same offset → same score, wherever the pair sits. Change only one slider and the score changes.</p>`;
      });

      /* KV cache. */
      kvHost.querySelector('[data-input="model"]').value = state.kv.model;
      kvHost.querySelector('[data-input="ctxPow"]').value = state.kv.ctxPow;
      mountWidget(kvHost, state.kv, (host, s) => {
        const model = KV_MODELS[s.model];
        const tokens = 2 ** s.ctxPow;
        const bytes = Number(s.bytes);
        const cache = (kv) => 2 * model.layers * kv * model.dHead * tokens * bytes;
        const options = [
          ["multi-head (every head has its own K, V)", model.heads],
          ["grouped-query (8 K/V heads)", Math.min(8, model.heads)],
          ["multi-query (1 shared K/V head)", 1],
        ];
        const values = options.map(([, kv]) => cache(kv));
        const kind = model.kv === model.heads ? "multi-head" : model.kv === 1 ? "multi-query" : `grouped-query (${model.kv} K/V heads for ${model.heads} query heads)`;
        host.querySelector('[data-out="kv"]').innerHTML = `
          <div class="g-metrics">
            <div><span>context</span><b>${fmtInt(tokens)} tokens</b></div>
            <div><span>this model uses</span><b>${kind}</b></div>
            <div><span>its KV cache</span><b>${fmtBytes(cache(model.kv))}</b></div>
          </div>
          <div class="tl-subhead">The same model if it used…</div>
          ${barList(values.map((v) => v / 1024 ** 3), options.map(([label]) => label), { digits: 2, color: C().c })}
          <p class="caption">Gigabytes, for one conversation. ${model.layers} layers × ${model.dHead}-number heads. Without a cache, every new token would recompute keys and values for all ${fmtInt(tokens)} previous tokens in every layer.</p>`;
      }, { formatOutput: (key, value) => (key === "ctxPow" ? fmtInt(2 ** value) : value) });

      /* Parameter calculator. */
      ["L", "d", "V"].forEach((key) => {
        paramHost.querySelector(`[data-input="${key}"]`).value = state.params[key];
      });
      mountWidget(paramHost, state.params, (host, s) => {
        const L = Math.max(1, s.L || 1);
        const d = Math.max(1, s.d || 1);
        const V = Math.max(1, s.V || 1);
        const attn = 4 * d * d * L;
        const ffn = 8 * d * d * L;
        const emb = V * d;
        const total = attn + ffn + emb;
        const preset = PARAM_PRESETS[s.preset];
        const matches = preset && preset.L === L && preset.d === d && preset.V === V;
        host.querySelector('[data-out="params"]').innerHTML = `
          ${barList([attn, ffn, emb].map((v) => v / 1e6), ["attention 4·L·d²", "feed-forward 8·L·d²", "embeddings V·d"], { digits: 1, color: C().a })}
          <div class="g-metrics">
            <div><span>estimated total</span><b>${fmtBig(total)}</b></div>
            ${matches ? `<div><span>published size</span><b>${preset.real}</b></div>` : ""}
          </div>
          <p class="caption">Millions of parameters. Biases, layer norms and position embeddings add well under 1% for large models.</p>`;
      }, {
        formatOutput: (key, value) => value,
      });
      paramHost.addEventListener("click", (event) => {
        const seg = event.target.closest('[data-seg="preset"]');
        if (!seg) return;
        const preset = PARAM_PRESETS[seg.getAttribute("data-value")];
        Object.assign(state.params, { L: preset.L, d: preset.d, V: preset.V });
        ["L", "d", "V"].forEach((key) => {
          paramHost.querySelector(`[data-input="${key}"]`).value = state.params[key];
        });
        paramHost.querySelector('[data-input="L"]').dispatchEvent(new Event("input", { bubbles: true }));
      });
    }

    render();
    U().onRedraw(render);
    ui.startProgressBar();
  }

  /* Widget helpers shared with later guides (small-llm-guide.js). */
  root.MLGuideWidgets = { esc, widget, range, segmented, mountWidget, whenNear, fmtInt, fmtBig, fmtBytes };

  root.MLExtraLabs = Object.assign(root.MLExtraLabs || {}, {
    "sequence-primer": mountPrimerGuide,
    llm: mountLlmGuide,
  });
})();
