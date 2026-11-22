// Headless render: starts its own Vite dev server, drives the Motion Canvas editor with
// Puppeteer, then splits the MP4 into one file per beat (beats are logged by `beat()`
// in src/lib/ui.tsx) and writes a contact sheet of stills for review.
//
// usage: node scripts/render.mjs <project> [--sheet-only]
//   <project> is a name registered in vite.config.ts (e.g. clip1_wiring).
// Output: output/<project>.mp4, output/<project>/<NN>_<beat>.mp4, output/<project>_sheet.png
//
// Renders are serialised with a lock file (output/.render.lock) so parallel callers
// don't exhaust memory; each caller waits its turn.
import puppeteer from 'puppeteer';
import {createServer} from 'vite';
import {execFileSync} from 'node:child_process';
import {existsSync, mkdirSync, statSync, openSync, closeSync, unlinkSync, readFileSync, writeFileSync, renameSync, rmSync} from 'node:fs';
import ffmpegInstaller from '@ffmpeg-installer/ffmpeg';

const project = process.argv[2];
if (!project) throw new Error('usage: node scripts/render.mjs <project>');
const ffmpeg = ffmpegInstaller.path;
// With several projects the editor serves /src/<name> and writes output/src/<name>.mp4.
const outCandidates = [`output/src/${project}.mp4`, `output/${project}.mp4`];
const out = `output/${project}.mp4`;
mkdirSync('output', {recursive: true});
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ------------------------------------------------------------------ lock
const LOCK = 'output/.render.lock';
async function acquire() {
  for (;;) {
    try {
      const fd = openSync(LOCK, 'wx');
      writeFileSync(fd, String(process.pid));
      closeSync(fd);
      return;
    } catch {
      // stale lock (holder died)?
      try {
        const pid = parseInt(readFileSync(LOCK, 'utf8'), 10);
        process.kill(pid, 0);
      } catch {
        try { unlinkSync(LOCK); } catch {}
        continue;
      }
      await sleep(3000);
    }
  }
}
const release = () => { try { if (readFileSync(LOCK, 'utf8') === String(process.pid)) unlinkSync(LOCK); } catch {} };
process.on('exit', release);
for (const s of ['SIGINT', 'SIGTERM']) process.on(s, () => process.exit(1));

console.log(`waiting for render lock ...`);
await acquire();

// ------------------------------------------------------------------ server + browser
const server = await createServer({server: {port: 0, host: '127.0.0.1'}, logLevel: 'error'});
await server.listen();
const port = server.httpServer.address().port;
const browser = await puppeteer.launch({headless: true, args: ['--no-sandbox']});
async function shutdown(code) {
  try { await browser.close(); } catch {}
  try { await server.close(); } catch {}
  release();
  process.exit(code);
}
async function fail(msg, page) {
  console.error('ERROR:', msg);
  if (page) try { await page.screenshot({path: `output/${project}_error.png`}); console.error(`  screenshot: output/${project}_error.png`); } catch {}
  await shutdown(1);
}

const page = await browser.newPage();
page.setDefaultTimeout(120000);
await page.setViewport({width: 1600, height: 1000});
const beats = new Map();
let rendering = false;
page.on('console', m => {
  const t = m.text();
  const b = t.match(/^BEAT (\S+) ([\d.]+)/);
  if (b) { if (rendering) beats.set(b[1], parseFloat(b[2])); }
  else if (!t.startsWith('[vite]') && !t.includes('404')) console.log('  page:', t.slice(0, 300));
});
page.on('pageerror', e => console.log('  pageerror:', e.message.slice(0, 500)));

const mtime = f => (existsSync(f) ? statSync(f).mtimeMs : 0);
const before = Object.fromEntries(outCandidates.map(f => [f, mtime(f)]));
const hasButton = label => page.evaluate(l => [...document.querySelectorAll('button')].some(b => b.innerText.trim() === l), label);

// A frozen page (e.g. an infinite loop in a scene) makes every CDP call hang, so wrap
// them in a wall-clock timeout.
const within = (p, ms, what) => Promise.race([p, sleep(ms).then(() => { throw new Error(`timed out: ${what}`); })]);

try {
  for (const path of [`/src/${project}`, `/${project}`, `/`]) {
    const resp = await within(page.goto(`http://127.0.0.1:${port}${path}`, {waitUntil: 'load'}), 150000, 'page load');
    if (resp.status() !== 404) break;
  }
  const t1 = Date.now();
  while (!(await within(hasButton('RENDER'), 30000, 'editor UI (page frozen? check scene for infinite loops)'))) {
    if (Date.now() - t1 > 150000) throw new Error('editor did not show RENDER button');
    await sleep(1000);
  }
  await sleep(3000);
} catch (e) {
  await fail(e.message, page);
}

console.log(`rendering ${project} ...`);
rendering = true;
const t0 = Date.now();
try {
  await within(page.evaluate(() => [...document.querySelectorAll('button')].find(b => b.innerText.trim() === 'RENDER').click()), 30000, 'click');
  const t2 = Date.now();
  while (!(await within(hasButton('ABORT'), 30000, 'render start'))) {
    if (Date.now() - t2 > 60000) throw new Error('render did not start (ABORT button never appeared)');
    await sleep(500);
  }
  let last = 0;
  while (await within(hasButton('ABORT'), 60000, 'render progress (page frozen?)')) {
    if (Date.now() - last > 30000) {
      last = Date.now();
      const frame = await page.evaluate(() => document.body.innerText.match(/\[(\d+)\]/g)?.slice(0, 2).join(' '));
      console.log(`  ... ${((Date.now() - t0) / 1000).toFixed(0)} s, frames ${frame ?? '?'}`);
    }
    if (Date.now() - t0 > 30 * 60000) throw new Error('render took more than 30 min');
    await sleep(1000);
  }
  await sleep(2000);
} catch (e) {
  await fail(e.message, page);
}
try { await browser.close(); } catch {}
try { await server.close(); } catch {}
release();

const written = outCandidates.find(f => mtime(f) > before[f]);
if (!written) {
  console.error(`ERROR: none of ${outCandidates.join(', ')} was written`);
  process.exit(1);
}
if (written !== out) renameSync(written, out);
console.log(`rendered ${out} in ${((Date.now() - t0) / 1000).toFixed(0)} s`);

// ------------------------------------------------------------------ split per beat
const list = [...beats.entries()].sort((a, b) => a[1] - b[1]);
console.log('beats:', list.map(([n, t]) => `${n}@${t.toFixed(2)}`).join(' '));
rmSync(`output/${project}`, {recursive: true, force: true}); // drop stale beats from older renders
mkdirSync(`output/${project}`, {recursive: true});
for (let k = 0; k < list.length - 1; k++) {
  const [name, start] = list[k];
  const end = list[k + 1][1];
  if (end - start < 0.05) continue;
  const file = `output/${project}/${String(k + 1).padStart(2, '0')}_${name}.mp4`;
  execFileSync(ffmpeg, [
    '-y', '-loglevel', 'error', '-i', out, '-ss', start.toFixed(3), '-to', end.toFixed(3),
    '-an', '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', file,
  ]);
  console.log('  wrote', file);
}

// ------------------------------------------------------------------ contact sheet
execFileSync(ffmpeg, [
  '-y', '-loglevel', 'error', '-i', out, '-vf', 'fps=1,scale=480:-1,tile=6x10', '-frames:v', '1',
  `output/${project}_sheet.png`,
]);
console.log(`  wrote output/${project}_sheet.png`);
process.exit(0);
