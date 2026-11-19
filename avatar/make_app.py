"""Builds app.html (the clean BahuSanket presentation front end) from index.html's avatar engine.
Usage: python make_app.py <index.html> <out app.html>"""
import sys
src = open(sys.argv[1]).read().split("\n")
find = lambda s: next(i for i, l in enumerate(src) if l.startswith(s))
head_css = "\n".join(src[find("  :root {"):find("  html, body {")])                 # colour tokens (light + dark)
engine = "\n".join(src[find('<script type="importmap">'):find("// ---------- player ----------")])
conv = "\n".join(src[find("function paraToV2(d) {"):find("// No depth in model output")])
engine = engine.replace('const statusEl = $("status");', 'const statusEl = $("status");')

HTML = r'''<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BahuSanket Live Signer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Mono:wght@400;500&family=Noto+Sans+Devanagari:wght@400;600&family=Noto+Sans+Kannada:wght@400;600&display=swap">
<style>
''' + head_css + r'''
  * { box-sizing: border-box; }
  html, body { background: var(--bg); color: var(--ink); margin: 0; }
  body {
    font-family: var(--font-body), "Noto Sans Devanagari", "Noto Sans Kannada", sans-serif; font-size: 16px; line-height: 1.5;
    padding: 18px max(16px, env(safe-area-inset-right, 0px)) 28px max(16px, env(safe-area-inset-left, 0px));
  }
  .wrap { max-width: 1280px; margin: 0 auto; display: grid; gap: 18px; }
  .top { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  .brand { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  .brand h1 { font-family: var(--font-display); font-weight: 700; font-size: clamp(26px, 3.2vw, 36px); margin: 0; letter-spacing: -.01em; }
  .brand p { margin: 0; color: var(--muted); }
  .pill { margin-left: auto; font-family: var(--font-mono); font-size: 12px; padding: 5px 12px; border-radius: 999px; border: 1px solid var(--line); color: var(--muted); display: inline-flex; align-items: center; gap: 8px; }
  .pill::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
  .pill.on { color: var(--accent); border-color: var(--accent); }
  .pill.on::before { background: var(--accent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 25%, transparent); }
  .main { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.15fr); gap: 18px; align-items: start; }
  @media (max-width: 900px) { .main { grid-template-columns: minmax(0, 1fr); } }
  .card { background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 18px; display: grid; gap: 14px; min-width: 0; }
  .card h2 { font-family: var(--font-display); font-weight: 700; font-size: 18px; margin: 0; }
  .seg { display: inline-flex; border: 1px solid var(--accent); border-radius: 999px; padding: 2px; gap: 2px; flex-wrap: wrap; }
  .seg button { font: inherit; font-weight: 600; border: 0; background: transparent; color: var(--accent); padding: 6px 14px; border-radius: 999px; cursor: pointer; }
  .seg button[aria-pressed="true"] { background: var(--accent); color: var(--accent-ink); }
  textarea { width: 100%; font: inherit; font-size: 19px; line-height: 1.5; padding: 12px 14px; border-radius: 12px; border: 1px solid var(--line);
    background: var(--bg); color: var(--ink); resize: vertical; min-height: 150px; }
  .go { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  button.primary { font: inherit; font-weight: 700; font-size: 17px; padding: 11px 26px; border-radius: 999px; border: 1px solid var(--accent);
    background: var(--accent); color: var(--accent-ink); cursor: pointer; }
  button.ghost { font: inherit; font-size: 13px; padding: 5px 12px; border-radius: 999px; border: 1px solid var(--line); background: transparent; color: var(--ink); cursor: pointer; }
  button:disabled { opacity: .55; cursor: progress; }
  button:focus-visible, select:focus-visible, textarea:focus-visible { outline: 3px solid color-mix(in srgb, var(--accent) 45%, transparent); outline-offset: 2px; }
  .kbd { font-family: var(--font-mono); font-size: 12px; color: var(--muted); }
  .examples { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  .examples span { font-size: 13px; color: var(--muted); margin-right: 2px; }
  .steps { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; }
  @media (max-width: 520px) { .steps { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  .steps li { border: 1px solid var(--line); border-radius: 10px; padding: 8px 10px; font-size: 13px; line-height: 1.3; color: var(--muted); }
  .steps li b { display: block; color: var(--ink); font-size: 13px; }
  .steps li .t { font-family: var(--font-mono); font-size: 12px; }
  .steps.busy li { animation: pulse 1.4s ease-in-out infinite; }
  .steps.busy li:nth-child(2) { animation-delay: .2s; } .steps.busy li:nth-child(3) { animation-delay: .4s; } .steps.busy li:nth-child(4) { animation-delay: .6s; }
  .steps.done li { border-color: var(--accent); }
  @keyframes pulse { 50% { border-color: var(--accent); } }
  @media (prefers-reduced-motion: reduce) { .steps.busy li { animation: none; border-color: var(--accent); } }
  .msg { margin: 0; font-size: 14px; color: var(--muted); min-height: 1.4em; }
  .msg.err { color: #b3261e; }
  .hist { display: grid; gap: 6px; max-height: 260px; overflow: auto; }
  .hist button { text-align: left; font: inherit; font-size: 14px; padding: 8px 12px; border-radius: 10px; border: 1px solid var(--line); background: var(--bg); color: var(--ink); cursor: pointer; display: grid; gap: 2px; }
  .hist button[aria-current="true"] { border-color: var(--accent); box-shadow: inset 3px 0 0 var(--accent); }
  .hist small { color: var(--muted); font-family: var(--font-mono); font-size: 11px; }
  .stagecol { display: grid; gap: 10px; min-width: 0; }
  .stage { position: relative; border-radius: 16px; overflow: hidden; aspect-ratio: 4 / 4.4; width: 100%; max-height: calc(100vh - 170px); margin: 0 auto;
    background: linear-gradient(180deg, var(--stage-top), var(--stage-bottom)); border: 1px solid var(--line); }
  @supports (aspect-ratio: 1) { .stage { width: min(100%, calc((100vh - 170px) * 4 / 4.4)); } }
  .stage canvas { display: block; width: 100%; height: 100%; }
  .status { position: absolute; inset: 0; display: grid; place-items: center; text-align: center; padding: 24px; color: var(--muted); font-family: var(--font-mono); font-size: 13px; }
  .status[hidden], [hidden] { display: none !important; }
  .caption { position: absolute; left: 12px; right: 12px; bottom: 12px; padding: 10px 14px; border-radius: 12px;
    background: color-mix(in srgb, var(--surface) 90%, transparent); border: 1px solid var(--line); display: grid; gap: 2px; }
  .caption .h { font-size: 12px; color: var(--muted); font-family: var(--font-mono); }
  .caption .s { font-family: var(--font-display), "Noto Sans Devanagari", "Noto Sans Kannada", sans-serif; font-weight: 600; font-size: clamp(18px, 2.3vw, 24px); line-height: 1.3; }
  .caption .e { font-size: 15px; color: var(--muted); }
  .caption:empty { display: none; }
  .bar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: center; }
  .bar select { font: inherit; font-size: 14px; padding: 6px 10px; border-radius: 999px; border: 1px solid var(--line); background: var(--surface); color: var(--ink); }
  .time { font-family: var(--font-mono); font-size: 13px; color: var(--muted); font-variant-numeric: tabular-nums; min-width: 6ch; }
  .foot { font-size: 12px; color: var(--muted); margin: 0; }
  .foot a { color: var(--accent); }
</style>
<body>
<div class="wrap">
  <div class="top">
    <div class="brand">
      <h1>BahuSanket</h1>
      <p>Text to Indian Sign Language · English · हिन्दी · ಕನ್ನಡ</p>
    </div>
    <span class="pill" id="pill">Connecting…</span>
  </div>

  <div class="main">
    <div class="card">
      <h2>Type a paragraph</h2>
      <div class="seg" role="group" aria-label="Language" id="langSeg">
        <button type="button" data-lang="auto" aria-pressed="true">Detect</button>
        <button type="button" data-lang="en" aria-pressed="false">English</button>
        <button type="button" data-lang="hi" aria-pressed="false">हिन्दी</button>
        <button type="button" data-lang="kn" aria-pressed="false">ಕನ್ನಡ</button>
      </div>
      <textarea id="text" placeholder="Type or paste one or more sentences in English, Hindi or Kannada…" aria-label="Paragraph to sign"></textarea>
      <div class="go">
        <button class="primary" id="go" type="button">Sign it</button>
        <span class="kbd">or Ctrl + Enter</span>
      </div>
      <div class="examples"><span>Try:</span>
        <button type="button" class="ghost" data-ex="en">English</button>
        <button type="button" class="ghost" data-ex="hi">हिन्दी</button>
        <button type="button" class="ghost" data-ex="kn">ಕನ್ನಡ</button>
      </div>
      <ol class="steps" id="steps" aria-label="What happens">
        <li><b>1 · Translate</b>IndicTrans2 → English <span class="t" data-t="translate"></span></li>
        <li><b>2 · Generate</b>text → sign poses <span class="t" data-t="generate"></span></li>
        <li><b>3 · Blend</b>sentences joined <span class="t" data-t="blend"></span></li>
        <li><b>4 · Depth</b>2D → 3D <span class="t" data-t="depth"></span></li>
      </ol>
      <p class="msg" id="msg" role="status" aria-live="polite"></p>
      <h2 style="font-size:15px">Signed so far</h2>
      <div class="hist" id="hist"></div>
    </div>

    <div class="stagecol">
      <div class="stage" id="stage">
        <canvas id="three" aria-label="3D avatar signing"></canvas>
        <canvas id="stick" hidden aria-label="Stick figure of the same pose points"></canvas>
        <div class="status" id="status">Loading the avatar…</div>
        <div class="caption" id="caption"></div>
      </div>
      <div class="bar">
        <button class="ghost" id="play" type="button">Pause</button>
        <button class="ghost" id="replay" type="button">Replay</button>
        <span class="time" id="time">0.0 s</span>
        <select id="speed" aria-label="Signing speed">
          <option value="1">Speed 1×</option>
          <option value="0.75" selected>Speed 0.75×</option>
          <option value="0.5">Speed 0.5×</option>
        </select>
        <div class="seg" role="group" aria-label="Show">
          <button type="button" data-view="avatar" aria-pressed="true">Avatar</button>
          <button type="button" data-view="stick" aria-pressed="false">Pose points</button>
        </div>
      </div>
      <p class="foot">Generated by the BahuSanket model, not recorded signing. Avatar: <a href="https://github.com/vrm-c/vrm-specification/tree/master/samples/VRM1_Constraint_Twist_Sample" target="_blank" rel="noopener">VRM1 Constraint Twist Sample</a> © pixiv Inc. (VRM Public License 1.0). Translation: <a href="https://arxiv.org/abs/2305.16307" target="_blank" rel="noopener">IndicTrans2</a> (AI4Bharat). Training data: <a href="https://huggingface.co/datasets/Exploration-Lab/iSign" target="_blank" rel="noopener">iSign</a>.</p>
    </div>
  </div>
</div>
''' + engine + r'''
// ---------- app ----------
function addDepth() {}          // every clip here carries depth from the depth model
''' + conv + r'''
const LANG = { en: "English", hi: "Hindi", kn: "Kannada" };
const TOPIC = { rain: "Rain alert", scheme: "Scheme for deaf students" };
let items = [], cur = -1, frame = 0, playing = true, speed = 0.75, acc = 0, last = performance.now(), lang = "auto", live = false, busy = false;
const histEl = $("hist");

function addItem(c, pointIdx, fps, label, origin) {
  const v2 = paraToV2({ point_idx: pointIdx, fps, clips: [{ ...c, cut: null }] });
  const clip = fromV2(v2, "model")[0];
  items.push({ clip, label, origin, lang: c.lang });
  const i = items.length - 1;
  const b = document.createElement("button"); b.type = "button"; b.dataset.i = String(i);
  const s = document.createElement("span"); s.textContent = c.source.length > 90 ? c.source.slice(0, 90) + "…" : c.source; s.lang = c.lang;
  const sm = document.createElement("small"); sm.textContent = `${label} · ${LANG[c.lang] || c.lang} · ${(clip.n / fps).toFixed(1)} s`;
  b.append(s, sm); b.addEventListener("click", () => select(i));
  histEl.prepend(b);
  return i;
}
let segShown = -1;
function caption(c, t) {
  let k = c.segs.findIndex((g) => t >= g.start && t < g.end); if (k < 0) k = c.segs.length - 1;
  if (k === segShown) return; segShown = k;
  const g = c.segs[k], cap = $("caption"); cap.textContent = "";
  const h = document.createElement("div"); h.className = "h";
  h.textContent = `${LANG[c.lang] || c.lang} · sentence ${k + 1} of ${c.segs.length}`;
  const s = document.createElement("div"); s.className = "s"; s.lang = c.lang; s.textContent = g.src;
  cap.append(h, s);
  if (c.lang !== "en" || g.en !== g.src) { const e = document.createElement("div"); e.className = "e"; e.textContent = "→ " + g.en; cap.append(e); }
}
function select(i) {
  cur = i; frame = 0; acc = 0; segShown = -1;
  for (const b of histEl.children) b.setAttribute("aria-current", String(b.dataset.i === String(i)));
  show(true);
}
function show(snap) {
  if (cur < 0) return;
  const c = items[cur].clip;
  caption(c, frame);
  if (vrm) applyFrame(c, frame, snap ? 1 : 0.6);
  drawStick(c, frame);
  $("time").textContent = (frame / c.fps).toFixed(1) + " s";
}
function tick(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (cur >= 0 && playing) {
    const c = items[cur].clip; acc += dt * speed;
    const step = 1 / c.fps;
    if (acc >= step) {
      const n = Math.floor(acc / step); acc -= n * step; frame += n;
      if (frame >= c.n) frame = 0;
      show(frame < n);
    }
  }
  if (vrm) vrm.update(dt);
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
$("play").addEventListener("click", () => { playing = !playing; $("play").textContent = playing ? "Pause" : "Play"; });
$("replay").addEventListener("click", () => { frame = 0; acc = 0; segShown = -1; playing = true; $("play").textContent = "Pause"; show(true); });
$("speed").addEventListener("change", () => { speed = Number($("speed").value); });
function setView(v) {
  for (const b of document.querySelectorAll("[data-view]")) b.setAttribute("aria-pressed", String(b.dataset.view === v));
  canvas.hidden = v !== "avatar"; stick.hidden = v !== "stick"; show(true);
}
for (const b of document.querySelectorAll("[data-view]")) b.addEventListener("click", () => setView(b.dataset.view));
function setLang(v) {
  lang = v;
  for (const b of document.querySelectorAll("[data-lang]")) b.setAttribute("aria-pressed", String(b.dataset.lang === v));
}
for (const b of document.querySelectorAll("[data-lang]")) b.addEventListener("click", () => setLang(b.dataset.lang));
const EXAMPLES = {
  en: "Good morning everyone. Thank you for coming today. Please ask your questions at the end.",
  hi: "आज मौसम बहुत अच्छा है। बच्चे पार्क में खेल रहे हैं।",
  kn: "ನಮಸ್ಕಾರ. ಇಂದು ನಮ್ಮ ಕಾಲೇಜಿನಲ್ಲಿ ಒಂದು ಕಾರ್ಯಕ್ರಮವಿದೆ. ಎಲ್ಲರೂ ಬನ್ನಿ.",
};
for (const b of document.querySelectorAll("[data-ex]")) b.addEventListener("click", () => { $("text").value = EXAMPLES[b.dataset.ex]; setLang(b.dataset.ex); $("text").focus(); });
function msg(t, err) { $("msg").textContent = t; $("msg").classList.toggle("err", !!err); }
function stepTimes(tm) { for (const el of document.querySelectorAll("[data-t]")) { const v = tm ? tm[el.dataset.t] : null; el.textContent = v == null ? "" : `${v} s`; } }

async function signIt() {
  const text = $("text").value.trim();
  if (busy) return;
  if (!text) { msg("Type a sentence first."); $("text").focus(); return; }
  if (!live) { msg("Live signing needs the Colab server (notebook 20). The examples on the left still play.", true); return; }
  busy = true; $("go").disabled = true; stepTimes(null);
  $("steps").className = "steps busy";
  const t0 = performance.now();
  const timer = setInterval(() => msg(`Working… ${((performance.now() - t0) / 1000).toFixed(0)} s`), 300);
  try {
    const r = await fetch("api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, lang }) });
    const res = await r.json().catch(() => ({ ok: false, error: `the server answered ${r.status}` }));
    if (!res.ok) throw new Error(res.error || "unknown error");
    const i = addItem(res.clip, res.point_idx, res.fps, "Typed now", "live");
    select(i); playing = true; $("play").textContent = "Pause";
    const tm = res.timing || {};
    $("steps").className = "steps done";
    stepTimes(tm.cached ? null : { translate: res.clip.lang === "en" ? "–" : tm.translate, generate: tm.generate, blend: "<0.1", depth: tm.depth });
    msg(`Signed in ${((performance.now() - t0) / 1000).toFixed(1)} s` + (tm.cached ? " (typed before, so shown at once)" : "") + `: ${new Set(res.clip.segments.map((g) => g.src)).size} sentence${new Set(res.clip.segments.map((g) => g.src)).size > 1 ? "s" : ""}, ${(res.clip.n / res.fps).toFixed(1)} s of signing.`);
  } catch (err) {
    $("steps").className = "steps";
    msg("Could not sign it: " + (err && err.message ? err.message : err), true);
  } finally { clearInterval(timer); busy = false; $("go").disabled = false; }
}
$("go").addEventListener("click", signIt);
$("text").addEventListener("keydown", (ev) => { if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); signIt(); } });

let polls = 0;
function checkLive() {          // the server answers ok once its models are loaded; keep asking for a few minutes
  fetch("api/health").then((r) => (r.ok ? r.json() : null)).then((h) => {
    live = !!(h && h.ok);
    $("pill").textContent = live ? `Live · ${h.gpu}` : h ? "Loading the models…" : "Offline · prepared examples only";
    $("pill").classList.toggle("on", live);
    if (h && !live && ++polls < 120) setTimeout(checkLive, 3000);
  }).catch(() => { $("pill").textContent = "Offline · prepared examples only"; });
}
checkLive();

Promise.all([
  fetch("clips_paragraph.json").then((r) => (r.ok ? r.json() : null)).catch(() => null),
  fetch("avatar.b64.txt").then((r) => r.text()).then((b64) => {
    const bin = Uint8Array.from(atob(b64.trim()), (ch) => ch.charCodeAt(0));
    return loader.parseAsync(bin.buffer, "");
  }),
]).then(([para, gltf]) => {
  if (para) for (const c of para.clips.slice().reverse()) addItem(c, para.point_idx, para.fps, `Example: ${TOPIC[c.uid.split("_")[0]] || c.uid}`, "demo");
  vrm = gltf.userData.vrm;
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.combineSkeletons(gltf.scene);
  vrm.scene.traverse((o) => { o.frustumCulled = false; });
  measureRest(); tuckHair(); buildBodyShell();
  scene.add(vrm.scene);
  if (vrm.lookAt) vrm.lookAt.target = camera;
  statusEl.hidden = true;
  if (items.length) select(items.length - 1);
}).catch((err) => {
  statusEl.textContent = "The avatar could not load (" + (err && err.message ? err.message : err) + "). Reload the page to try again.";
  console.error(err);
});
if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { playing = false; $("play").textContent = "Play"; }
requestAnimationFrame(tick);
window.__bahu = { get ready() { return !!vrm; }, get items() { return items.length; }, pause() { playing = false; }, set frame(v) { frame = v; show(true); } };
</script>
'''
open(sys.argv[2], "w").write(HTML)
print("app.html", len(HTML))
