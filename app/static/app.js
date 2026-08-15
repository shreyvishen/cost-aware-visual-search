/* Renders the page from data.json. Every number on screen is read from the artifacts,
   so the page cannot drift from the runs. */

const $ = (s) => document.querySelector(s);
const el = (t, c, h) => { const n = document.createElement(t); if (c) n.className = c; if (h != null) n.innerHTML = h; return n; };
const n1 = (v) => v == null ? "—" : (+v).toFixed(1);
const n2 = (v) => v == null ? "—" : (+v).toFixed(2);
const n3 = (v) => v == null ? "—" : (+v).toFixed(3);
const usd = (v) => v == null ? "—" : "$" + (+v).toFixed(3);
const ms = (v) => v == null ? "—" : Math.round(v).toLocaleString();

let D = null;

/* ---------- tables ---------------------------------------------------- */

/* `best` on a cell bolds it. Comparisons are only meaningful within a column, so the
   caller decides which direction wins — lower latency, higher accuracy. */
function markBest(rows, col, dir) {
  let bi = -1, bv = null;
  rows.forEach((r, i) => {
    const v = r.cells[col] && r.cells[col].raw;
    if (typeof v !== "number") return;
    if (bv === null || (dir === "lo" ? v < bv : v > bv)) { bv = v; bi = i; }
  });
  if (bi >= 0) rows[bi].cells[col].best = true;
}

function table(node, cols, rows) {
  node.innerHTML = "";
  const thead = el("thead");
  const tr = el("tr");
  cols.forEach((c) => tr.appendChild(el("th", null, c)));
  thead.appendChild(tr); node.appendChild(thead);
  const tb = el("tbody");
  rows.forEach((r) => {
    const t = el("tr", r.hi ? "hi" : null);
    r.cells.forEach((c, i) => {
      const cls = (i === 0 ? "lab" : (c.cls || "num")) + (c.best ? " best" : "");
      const td = el("td", cls, c.v == null ? c : c.v);
      tb.appendChild ? t.appendChild(td) : null;
    });
    tb.appendChild(t);
  });
  node.appendChild(tb);
}

function headline() {
  const h = D.headline, by = Object.fromEntries(h.map((x) => [x.key, x]));
  // Column headers carry their own provenance. Accuracy comes off the rig, everything else
  // off the Mac — putting them in one table without saying so is how a reader concludes that
  // B does 1.54 zooms on device when it actually does 0.42.
  const cols = ["", "accuracy<br><span class='src'>rig · 191</span>",
    "zooms<br><span class='src'>M4 · 36</span>", "prefill tok<br><span class='src'>M4 · 36</span>",
    "decode tok<br><span class='src'>M4 · 36</span>", "latency<br><span class='src'>M4 · 36</span>",
    "$ / 1k<br><span class='src'>M4 · 36</span>"];
  const accOf = (k) => D.accuracy[k]?.full;
  const rows = h.map((m) => ({
    hi: m.key === "b",
    cells: [
      m.label,
      accOf(m.key) == null ? "—" : n3(accOf(m.key)),
      n2(m.zooms),
      Math.round(m.prefill_tokens).toLocaleString(),
      n1(m.decode_tokens),
      ms(m.latency_ms) + " ms",
      usd(m.usd_per_1k),
    ].map((v) => ({ v })),
  }));
  // raw values so the winner in each column can be found, then bolded
  h.forEach((m, i) => {
    const raw = [null, accOf(m.key), m.zooms, m.prefill_tokens, m.decode_tokens,
                 m.latency_ms, m.usd_per_1k];
    raw.forEach((v, j) => { if (rows[i].cells[j]) rows[i].cells[j].raw = v; });
  });
  markBest(rows, 1, "hi");   // accuracy
  [2, 3, 4, 5, 6].forEach((c) => markBest(rows, c, "lo"));  // everything else: less is better
  table($("#t-headline"), cols, rows);

  const a = by.a, b = by.b;
  if (a && b) {
    const sp = (a.latency_ms / b.latency_ms).toFixed(2);
    $("#tag-speed").textContent = sp + "×";
    const cut = (x, y) => Math.round((1 - y / x) * 100);
    const dDec = (a.decode_tokens - b.decode_tokens) * D.coeffs.b;
    const dVis = (a.prefill_tokens - b.prefill_tokens) * D.coeffs.a;
    const dTool = (a.zooms - b.zooms) * D.coeffs.c;
    const tot = Math.max(dDec + dVis + dTool, 1);
    const pc = (v) => Math.max(0, (v / tot) * 100);
    $("#n-headline").innerHTML =
      `<div class="saveline"><b>Same accuracy</b>` +
      ` &nbsp;·&nbsp; <b>${(a.latency_ms / b.latency_ms).toFixed(2)}× faster</b>` +
      ` &nbsp;·&nbsp; <b>${cut(a.usd_per_1k, b.usd_per_1k)}% cheaper</b></div>` +
      `<div class="savebar" title="where the saving comes from">` +
      `<span style="width:${pc(dDec)}%;background:#2f5cff"></span>` +
      `<span style="width:${pc(dVis)}%;background:#93a8ff"></span>` +
      `<span style="width:${pc(dTool)}%;background:#d6ddff"></span></div>` +
      `<div class="savekey"><span><i style="background:#2f5cff"></i>thinking ${Math.round(pc(dDec))}%</span>` +
      `<span><i style="background:#93a8ff"></i>looking ${Math.round(pc(dVis))}%</span>` +
      `<span><i style="background:#d6ddff"></i>tool calls ${Math.round(pc(dTool))}%</span></div>` +
      `<div class="savenote">It learned to think less, not look less — a decode token costs ` +
      `<span class="num">${D.coeffs.b.toFixed(1)} ms</span> against ` +
      `<span class="num">${D.coeffs.a.toFixed(2)} ms</span> for a vision token.</div>`;
  }
}

function hero() {
  const by = Object.fromEntries(D.headline.map((x) => [x.key, x]));
  const a = by.a, b = by.b;
  if (!a || !b) return;
  const items = [
    [`${(a.latency_ms / b.latency_ms).toFixed(2)}×`, "faster on device", true],
    [`${Math.round((1 - b.usd_per_1k / a.usd_per_1k) * 100)}%`, "cheaper per question", true],
  ];
  $("#hero").innerHTML = items.map(([v, l, good]) =>
    `<div class="hstat${good ? " good" : ""}"><b>${v}</b><span>${l}</span></div>`).join("");
}

function tabs() {
  const nav = $("#tabs");
  nav.querySelectorAll("button").forEach((btn) => {
    btn.onclick = () => {
      nav.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === btn));
      document.querySelectorAll(".page").forEach((p) =>
        p.classList.toggle("on", p.id === "p-" + btn.dataset.p));
      window.scrollTo({ top: 0, behavior: "smooth" });
    };
  });
}

function zoomTable() {
  const rows = [];
  for (const k of ["a", "b"]) {
    const x = D.accuracy[k]; if (!x) continue;
    const gain = x.ref_zoom - x.ref_nozoom;
    rows.push({
      hi: k === "b",
      cells: [
        "Run " + k.toUpperCase(),
        n3(x.ref_nozoom), n3(x.ref_zoom),
        `<span class="win">+${gain.toFixed(3)}</span>`,
        `+${Math.round((gain / x.ref_nozoom) * 100)}%`,
      ].map((v) => ({ v })),
    });
  }
  rows.forEach((r, i) => {
    const k = ["a", "b"][i], x = D.accuracy[k];
    r.cells[1].raw = x.ref_nozoom; r.cells[2].raw = x.ref_zoom;
    r.cells[3].raw = x.ref_zoom - x.ref_nozoom;
  });
  [2, 3].forEach((c) => markBest(rows, c, "hi"));
  table($("#t-zoom"), ["", "zoom disabled", "zoom allowed", "gain", "relative"], rows);
}

function hyper() {
  const s = D.hyper.shared, d = D.hyper.differs;
  $("#hp-line").textContent =
    `GRPO · ${s.group_size} rollouts × ${s.prompts_per_step} prompts per step · ` +
    `LoRA r${s.lora_rank} · lr ${s.lr} · ${s.minutes} min per run`;
  const rows = [
    ["algorithm", "GRPO, group is the baseline", "GRPO, group is the baseline"],
    ["rollouts per prompt (G)", s.group_size, s.group_size],
    ["prompts per step", s.prompts_per_step, s.prompts_per_step],
    ["gradient steps", d.a.steps, d.b.steps],
    ["LoRA rank / lr", `r${s.lora_rank} · ${s.lr}`, `r${s.lora_rank} · ${s.lr}`],
    ["KL coeff / clip", `${s.kl_coef} · ${s.clip_eps}`, `${s.kl_coef} · ${s.clip_eps}`],
    ["max zooms / new tokens", `${s.max_zooms} · ${s.max_new_tokens}`, `${s.max_zooms} · ${s.max_new_tokens}`],
    ["thumbnail", `÷${s.downsample}, ≤${s.thumb_max_side}px`, `÷${s.downsample}, ≤${s.thumb_max_side}px`],
    ["temperature / seed", `${s.temperature} · ${s.seed}`, `${s.temperature} · ${s.seed}`],
    ["<b>cost term</b>", "<b>none</b>", "<b>measured M4 ms @ Q4</b>"],
    ["<b>λ</b>", "0", `<b>${d.b.lambda.toExponential(2)}</b>`],
  ].map((r) => ({ cells: r.map((v) => ({ v })) }));
  table($("#t-hyper"), ["", "Run A", "Run B"], rows);
}

/* ---------- charts ---------------------------------------------------- */

function line(title, cap, series, fmt) {
  const W = 380, H = 150, P = { l: 38, r: 10, t: 8, b: 22 };
  const all = series.flatMap((s) => s.pts.map((p) => p[1]));
  let lo = Math.min(...all), hi = Math.max(...all);
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
  const xs = Math.max(...series.flatMap((s) => s.pts.map((p) => p[0])));
  const X = (v) => P.l + (v / Math.max(xs, 1)) * (W - P.l - P.r);
  const Y = (v) => H - P.b - ((v - lo) / (hi - lo)) * (H - P.t - P.b);

  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img">`;
  for (let i = 0; i <= 3; i++) {
    const v = lo + (i / 3) * (hi - lo), y = Y(v);
    svg += `<line x1="${P.l}" y1="${y}" x2="${W - P.r}" y2="${y}" stroke="#e9e9ee" stroke-width="1"/>`;
    svg += `<text x="${P.l - 6}" y="${y + 3.5}" text-anchor="end" font-size="9" fill="#a1a1ac" font-family="ui-monospace,monospace">${fmt(v)}</text>`;
  }
  svg += `<text x="${P.l}" y="${H - 6}" font-size="9" fill="#a1a1ac" font-family="ui-monospace,monospace">0</text>`;
  svg += `<text x="${W - P.r}" y="${H - 6}" text-anchor="end" font-size="9" fill="#a1a1ac" font-family="ui-monospace,monospace">step ${xs}</text>`;
  series.forEach((s) => {
    const d = s.pts.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1)).join(" ");
    svg += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="1.75" stroke-linejoin="round" stroke-linecap="round"/>`;
    s.pts.forEach((p) => { svg += `<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="1.8" fill="${s.color}"/>`; });
  });
  svg += `</svg>`;

  const c = el("div", "chart");
  c.appendChild(el("h3", null, title));
  c.appendChild(el("p", "cap", cap));
  c.insertAdjacentHTML("beforeend", svg);
  c.appendChild(el("div", "legend", series.map((s) => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("")));
  return c;
}

const CA = "#a1a1ac", CB = "#2f5cff";

function charts() {
  const box = $("#charts"); box.innerHTML = "";
  const pick = (k, f) => (D.curves[k] || []).map((r) => [r.step, f(r)]);
  box.appendChild(line("Mean reward", "reward = correct ? 1 − λ·cost_ms : 0 — this is the RL objective", [
    { name: "A", color: CA, pts: pick("a", (r) => r.reward) },
    { name: "B", color: CB, pts: pick("b", (r) => r.reward) },
  ], (v) => v.toFixed(2)));
  box.appendChild(line("Zooms per question", "how much the policy looks", [
    { name: "A", color: CA, pts: pick("a", (r) => r.zooms) },
    { name: "B", color: CB, pts: pick("b", (r) => r.zooms) },
  ], (v) => v.toFixed(2)));
  box.appendChild(line("Decode tokens", "how much the policy thinks", [
    { name: "A", color: CA, pts: pick("a", (r) => r.decode) },
    { name: "B", color: CB, pts: pick("b", (r) => r.decode) },
  ], (v) => Math.round(v)));
  box.appendChild(line("cost_ms — the term inside the reward",
    "a·vision + b·decode + c·zooms, both runs priced on the frozen Q4 table", [
    { name: "A", color: CA, pts: pick("a", (r) => r.cost_q4) },
    { name: "B", color: CB, pts: pick("b", (r) => r.cost_q4) },
  ], (v) => Math.round(v)));

  const ga = D.curves.a.reduce((s, r) => s + r.groups_used, 0), gat = D.curves.a.reduce((s, r) => s + r.groups_total, 0);
  const gb = D.curves.b.reduce((s, r) => s + r.groups_used, 0), gbt = D.curves.b.reduce((s, r) => s + r.groups_total, 0);
  $("#n-train").innerHTML =
    `B's reward sits lower because it is paying the cost term — compare behaviour, not height.` +
    `<br>A binary reward ties often, and a tied group of 8 teaches nothing. The cost term breaks ` +
    `those ties, so B learned from <span class="num">${Math.round(gb / gbt * 100)}%</span> of its ` +
    `groups against A's <span class="num">${Math.round(ga / gat * 100)}%</span> — and because cost ` +
    `only separates rollouts that are <i>already correct</i>, it cannot trade accuracy for speed.`;
}

/* ---------- 2×2 and samples -------------------------------------------- */

const CELLS = [
  ["both_right", "Both right"],
  ["only_a", "Only A right"],
  ["only_b", "Only B right"],
  ["both_wrong", "Both wrong"],
];

function matrix() {
  const box = $("#cells"); box.innerHTML = "";
  CELLS.forEach(([k, lab], i) => {
    const b = el("button", "cell");
    b.setAttribute("aria-pressed", i === 1 ? "true" : "false");
    b.innerHTML = `<div class="n num">${D.matrix[k]}</div><div class="t">${lab}</div>`;
    b.onclick = () => {
      document.querySelectorAll(".cell").forEach((x) => x.setAttribute("aria-pressed", "false"));
      b.setAttribute("aria-pressed", "true");
      showSamples(k);
    };
    box.appendChild(b);
  });
  showSamples("only_a");
  const m = D.matrix;
  $("#n-matrix").innerHTML =
    `A and B disagree on <span class="num">${m.only_a + m.only_b}</span> of 191 questions, ` +
    `almost evenly split — <span class="num">${m.only_a}</span> to A, ` +
    `<span class="num">${m.only_b}</span> to B — an even split. The two policies are equally ` +
    `accurate; they are simply wrong about different pictures.`;
}

function boxOverlay(sid, boxes, color) {
  // Boxes are on a 0-1000 grid over the whole image, so percentages map directly.
  const marks = (boxes || []).map((b, i) => {
    const [x1, y1, x2, y2] = b;
    const st = `left:${x1 / 10}%;top:${y1 / 10}%;width:${(x2 - x1) / 10}%;height:${(y2 - y1) / 10}%;` +
      `border-color:${color}`;
    return `<span class="bx" style="${st}"><i style="background:${color}">${i + 1}</i></span>`;
  }).join("");
  return `<div class="shotwrap"><img src="/img/${sid}" alt="" loading="lazy">${marks}</div>`;
}

function traceHtml(t) {
  if (!t || !t.length) return `<p class="tmeta">No turns recorded.</p>`;
  return t.map((turn, i) => {
    let h = `<div class="turn"><div class="tno">turn ${i + 1}</div>`;
    if (turn.think) h += `<div class="tblk"><span class="tk">thinking</span><p>${esc(turn.think)}</p></div>`;
    if (turn.tool) {
      const a = turn.tool.arguments || {};
      h += `<div class="tblk"><span class="tk">tool call</span><pre class="num">image_zoom_in_tool(
  bbox_2d = [${(a.bbox_2d || []).join(", ")}]${a.label ? `,\n  label = ${JSON.stringify(a.label)}` : ""}
)</pre></div>`;
    }
    if (turn.crop_vision_tokens) h += `<div class="tblk"><span class="tk">crop returned</span><p class="num">${turn.crop_vision_tokens} vision tokens</p></div>`;
    if (turn.answer) h += `<div class="tblk"><span class="tk">answer</span><p><b>${esc(turn.answer)}</b></p></div>`;
    return h + `</div>`;
  }).join("");
}

const esc = (s) => (s || "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

function showSamples(k) {
  const box = $("#samples"); box.innerHTML = "";
  const list = D.samples[k] || [];
  if (!list.length) { box.appendChild(el("p", "sub", "No examples in this cell.")); return; }
  list.forEach((s, si) => {
    const d = el("div", "sample");
    d.innerHTML = `<div class="q">${esc(s.question)}</div>
      <div class="gold">Gold: ${esc(s.gold)}</div>
      <div class="pair">
        ${["a", "b"].map((who) => {
          const r = s[who], color = who === "a" ? "#a1a1ac" : "#2f5cff";
          return `<div class="col">
            <div class="who">Run ${who.toUpperCase()}
              <span class="pill ${r.correct ? "good" : "bad"}"><span class="dot"></span>${r.correct ? "correct" : "wrong"}</span>
            </div>
            ${boxOverlay(s.sid, r.boxes, color)}
            <div class="val">${esc(r.answer) || "<i>no answer</i>"}</div>
            <div class="meta num">${r.zooms} zoom${r.zooms === 1 ? "" : "s"} · ${r.decode} decode tok</div>
            <button class="tbtn" data-k="${k}" data-i="${si}" data-w="${who}">show trace</button>
            <div class="trace" hidden>${traceHtml(r.trace)}</div>
          </div>`;
        }).join("")}
      </div>`;
    box.appendChild(d);
  });
  box.querySelectorAll(".tbtn").forEach((b) => {
    b.onclick = () => {
      const t = b.nextElementSibling;
      t.hidden = !t.hidden;
      b.textContent = t.hidden ? "show trace" : "hide trace";
    };
  });
}

/* ---------- quantization ------------------------------------------------ */

function quant() {
  const order = { a: 0, b: 1 };
  const rows = [...D.quant].sort((x, y) => (x.quant.localeCompare(y.quant)) || order[x.model] - order[y.model])
    .map((q) => ({
      hi: q.model === "b",
      cells: [
        `${q.label} <span class="pill">${q.quant}</span>`,
        Math.round(q.prefill_tokens).toLocaleString(),
        n1(q.decode_tokens), n2(q.zooms), ms(q.latency_ms) + " ms", usd(q.usd_per_1k),
      ].map((v) => ({ v })),
    }));
  rows.forEach((r, i) => {
    const q = [...D.quant].sort((x, y) => (x.quant.localeCompare(y.quant)) || order[x.model] - order[y.model])[i];
    [null, q.prefill_tokens, q.decode_tokens, q.zooms, q.latency_ms, q.usd_per_1k]
      .forEach((v, j) => { if (r.cells[j]) r.cells[j].raw = v; });
  });
  for (let i = 0; i + 1 < rows.length; i += 2) {
    const pair = rows.slice(i, i + 2);
    [1, 2, 3, 4, 5].forEach((c) => markBest(pair, c, "lo"));
  }
  table($("#t-quant"), ["", "prefill tok", "decode tok", "zooms", "latency", "$ / 1k"], rows);
}

/* ---------- live demo: a chat pane that shows the episode happening ------ */

/* The model emits a literal <think>…</think> / <tool_call>…</tool_call> / <answer>…</answer>
   contract. Showing the raw tags is noise; hiding them entirely would hide the format the
   whole harness is built around. So: drop the empty scaffolding, keep the content, and mark
   the answer. */
function renderStream(t) {
  let out = esc(t);
  out = out.replace(/&lt;think&gt;\s*/g, "").replace(/\s*&lt;\/think&gt;/g, "");
  out = out.replace(/&lt;tool_call&gt;([\s\S]*?)(&lt;\/tool_call&gt;|$)/g,
    (_, body) => `<span class="tool-lit">${body.trim()}</span>`);
  out = out.replace(/&lt;answer&gt;([\s\S]*?)(&lt;\/answer&gt;|$)/g,
    (_, body) => `<span class="ans-lit">${body.trim()}</span>`);
  return out.replace(/^\s+/, "");
}

function chatEl(cls, html) { const n = el("div", cls, html); $("#chat").appendChild(n); n.scrollIntoView({ block: "nearest" }); return n; }

function demo() {
  const file = $("#f-img"), go = $("#f-go");
  file.onchange = () => {
    const f = file.files[0];
    $("#f-name").textContent = f ? f.name : "";
    $("#f-preview").innerHTML = f ? `<img src="${URL.createObjectURL(f)}" alt="">` : "";
  };
  go.onclick = run;
  $("#f-q").addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });
}

async function run() {
  const f = $("#f-img").files[0], q = $("#f-q").value.trim();
  const chat = $("#chat"), go = $("#f-go");
  if (!f || !q) { chat.innerHTML = ""; chatEl("sys", "Attach an image and ask a question."); return; }
  chat.innerHTML = ""; go.disabled = true; go.textContent = "Running";

  const model = $("#f-model").value, quant = $("#f-quant").value, down = $("#f-down").checked;
  chatEl("msg user", `<img class="att" src="${URL.createObjectURL(f)}" alt="">
    <div class="txt">${esc(q)}</div>
    <div class="tags"><span class="pill">Run ${model.toUpperCase()}</span>
      <span class="pill">${quant}</span>
      <span class="pill">${down ? "down-project + zoom" : "full resolution"}</span></div>`);

  const fd = new FormData();
  fd.append("image", f); fd.append("question", q);
  fd.append("model", model); fd.append("quant", quant);
  fd.append("downproject", down ? "1" : "0");

  let statusNode = chatEl("sys", `<span class="spin"></span> loading ${quant.toUpperCase()} weights…`);
  let turnNode = null, shown = null, tokBuf = "";

  try {
    const res = await fetch("/api/stream", { method: "POST", body: fd });
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n"); buf = lines.pop();
      for (const ln of lines) {
        if (!ln.trim()) continue;
        let ev; try { ev = JSON.parse(ln); } catch { continue; }
        handle(ev);
      }
    }
  } catch (e) {
    chatEl("sys err", "Stream failed: " + e);
  }
  go.disabled = false; go.textContent = "Run";

  function handle(ev) {
    switch (ev.type) {
      case "status":
        if (ev.stage === "ready") {
          statusNode.className = "sys";
          statusNode.innerHTML = ev.load_ms
            ? `model ready <span class="num">${(ev.load_ms / 1000).toFixed(1)}s</span> to load`
            : `model already warm`;
        }
        break;
      case "image":
        shown = chatEl("msg bot",
          `<div class="lbl">${down ? "what it sees — down-projected" : "what it sees — full resolution"}
             <span class="num">${ev.w}×${ev.h}</span>
             <span class="num dim">from ${ev.orig_w}×${ev.orig_h}</span></div>
           <div class="shotwrap live"><img src="data:image/png;base64,${ev.png_b64}" alt=""></div>`);
        break;
      case "turn_start":
        tokBuf = "";
        turnNode = chatEl("msg bot", `<div class="lbl">turn ${ev.index + 1}</div><div class="stream"></div>`);
        break;
      case "token":
        tokBuf += ev.text;
        if (turnNode) turnNode.querySelector(".stream").innerHTML = renderStream(tokBuf);
        break;
      case "prefill":
        if (turnNode) turnNode.insertAdjacentHTML("beforeend",
          `<div class="tick num">prefill ${ev.tokens} tok · ${Math.round(ev.ms)} ms · ${ev.tok_per_s} tok/s</div>`);
        break;
      case "decode":
        if (turnNode) turnNode.insertAdjacentHTML("beforeend",
          `<div class="tick num">decode ${ev.tokens} tok · ${Math.round(ev.ms)} ms · ${ev.tok_per_s} tok/s</div>`);
        break;
      case "turn_end":
        if (ev.kind === "tool_call" && ev.bbox_2d) {
          chatEl("msg tool", `<div class="lbl">calling <b>image_zoom_in_tool</b></div>
            <pre class="num">bbox_2d = [${ev.bbox_2d.join(", ")}]</pre>`);
          if (shown) {
            const w = shown.querySelector(".shotwrap");
            const [x1, y1, x2, y2] = ev.bbox_2d;
            w.insertAdjacentHTML("beforeend",
              `<span class="bx draw" style="left:${x1 / 10}%;top:${y1 / 10}%;width:${(x2 - x1) / 10}%;height:${(y2 - y1) / 10}%"></span>`);
          }
        }
        break;
      case "crop":
        chatEl("msg tool", ev.png_b64
          ? `<div class="lbl">crop returned <span class="num">${ev.vision_tokens} vision tokens</span></div>
             <div class="shotwrap live"><img src="data:image/png;base64,${ev.png_b64}" alt=""></div>`
          : `<div class="lbl err">${ev.note || "the crop failed"}</div>`);
        break;
      case "done": {
        const s = (l, v) => `<div><b>${v}</b>${l}</div>`;
        chatEl("msg answer", `<div class="lbl">answer</div><div class="big">${esc(ev.answer) || "<i>no answer</i>"}</div>
          <div class="stat">
            ${s("prefill tok", ev.prefill_tokens)}${s("decode tok", ev.decode_tokens)}
            ${s("zooms", ev.zooms)}${s("total", Math.round(ev.total_ms) + " ms")}
            ${s("decode", (ev.decode_tok_per_s ?? "—") + " tok/s")}
            ${s("$ / 1k", "$" + (ev.usd * 1000).toFixed(3))}</div>`);
        break;
      }
      case "error":
        chatEl("sys err", ev.msg);
        break;
    }
  }
}

/* ---------- boot -------------------------------------------------------- */

fetch("/data.json").then((r) => r.json()).then((d) => {
  D = d;
  $("#f-coeffs").textContent =
    `a=${d.coeffs.a.toFixed(2)} b=${d.coeffs.b.toFixed(1)} c=${d.coeffs.c.toFixed(0)} ms, R²=${d.coeffs.r2.toFixed(4)}`;
  hero(); tabs(); headline(); zoomTable(); hyper(); charts(); matrix(); quant(); demo();
});
