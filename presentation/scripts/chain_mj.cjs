// Temporary: check that clip 3's TeX strings convert with MathJax.
const base = __dirname + '/../node_modules/mathjax-full/js/';
const {liteAdaptor} = require(base + 'adaptors/liteAdaptor');
const {RegisterHTMLHandler} = require(base + 'handlers/html');
const {TeX} = require(base + 'input/tex');
const {AllPackages} = require(base + 'input/tex/AllPackages');
const {mathjax} = require(base + 'mathjax');
const {SVG} = require(base + 'output/svg');
const a = liteAdaptor();
RegisterHTMLHandler(a);
const doc = mathjax.document('', {InputJax: new TeX({packages: AllPackages}), OutputJax: new SVG({fontCache: 'local'})});
for (const t of ['\\frac{\\color{#7FEE64}{R_i}}{\\sum_k R_k}', 'R_k \\mathrel{+}= \\Delta P_{ki}', 'j \\sim P_{i\\,\\cdot}']) {
  const s = a.innerHTML(doc.convert(t));
  console.log(t, s.length, /merror|data-mjx-error/.test(s) ? 'ERROR' : 'ok', (s.match(/fill="[^"]*"/g) || []).slice(0, 3));
}
