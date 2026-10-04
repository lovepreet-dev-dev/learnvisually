# Vendored files

`tiktoken-core.js`, `tiktoken-gpt2.js` and `tiktoken-cl100k.js` are browser bundles of
[js-tiktoken](https://www.npmjs.com/package/js-tiktoken) v1.0.21 (MIT licence), a JavaScript port of
OpenAI's tiktoken. They contain the real GPT-2 (`gpt2`) and GPT-4 (`cl100k_base`) BPE vocabularies.

They are used only by the Primer and LLM guides (`assets/sequence-guides.js`), which load them
on demand when the reader scrolls to a tokenizer widget.

Rebuild with esbuild:

```js
// core.js:   import { Tiktoken } from "js-tiktoken/lite"; window.MLTiktoken = Tiktoken;
// gpt2.js:   import r from "js-tiktoken/ranks/gpt2"; (window.MLTiktokenRanks ||= {}).gpt2 = r;
// cl100k.js: import r from "js-tiktoken/ranks/cl100k_base"; (window.MLTiktokenRanks ||= {}).cl100k = r;
// npx esbuild <file>.js --bundle --minify --format=iife --target=es2018 --charset=utf8 --outfile=tiktoken-<name>.js
```

The full licence text is in `js-tiktoken-LICENSE.txt`.
