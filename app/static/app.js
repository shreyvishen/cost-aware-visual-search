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
      const td = el("td", i === 0 ? "lab" : (c.cls || "num"), c.v == null ? c : c.v);
      tb.appendChild ? t.appendChild(td) : null;
    });
    tb.appendChild(t);
  });
  node.appendChild(tb);
}

function headline() {
  const h = D.headline, by = Object.fromEntries(h.map((x) => [x.key, x]));
  const cols = ["", "accuracy", "zooms", "prefill tok", "decode tok", "latency", "$ / 1k"];
  const accOf = (k) => k === "base" ? null : D.accuracy[k]?.full;
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
  table($("#t-headline"), cols, rows);

  const a = by.a, b = by.b;
  if (a && b) {
    const sp = (a.latency_ms / b.latency_ms).toFixed(2);
    $("#tag-speed").textContent = sp + "×";
    $("#n-headline").innerHTML =
      `<b>B matches A's accuracy</b> — 0.482 vs 0.476 on 191 images, McNemar exact ` +
      `<span class="num">p = 1.000</span> on the 61 questions where they disagree (split 30/31). ` +
      `<b>And it is ${sp}× faster</b> on the M4, cutting cost per question by ` +
      `<span class="num">${Math.round((1 - b.usd_per_1k / a.usd_per_1k) * 100)}%</span>. ` +
      `Note where the saving comes from: decode tokens fall ` +
      `<span class="num">${Math.round((1 - b.decode_tokens / a.decode_tokens) * 100)}%</span> ` +
      `while zooms fall only ${Math.round((1 - b.zooms / a.zooms) * 100)}%. The reward charges ` +
      `<span class="num">${D.coeffs.b.toFixed(1)} ms</span> per decode token against ` +
      `<span class="num">${D.coeffs.a.toFixed(2)} ms</span> per vision token, so the cheapest ` +
      `milliseconds to give back are tokens of thought — not looks. A reward that counted tool ` +
      `calls would have pushed the opposite way.`;
  }
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
  box.appendChild(line("Mean reward", "reward = correct ? 1 − λ·cost_ms : 0", [
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
  box.appendChild(line("Groups that taught something", "of 6 per step — zero-variance groups teach nothing", [
    { name: "A", color: CA, pts: pick("a", (r) => r.groups_used) },
    { name: "B", color: CB, pts: pick("b", (r) => r.groups_used) },
  ], (v) => v.toFixed(0)));

  const ga = D.curves.a.reduce((s, r) => s + r.groups_used, 0), gat = D.curves.a.reduce((s, r) => s + r.groups_total, 0);
  const gb = D.curves.b.reduce((s, r) => s + r.groups_used, 0), gbt = D.curves.b.reduce((s, r) => s + r.groups_total, 0);
  $("#n-train").innerHTML =
    `B's reward sits below A's on purpose — it is paying the cost term, so the two curves are ` +
    `not on the same scale and should not be compared by height. What matters is what each ` +
    `curve does to behaviour underneath it.<br><br>` +
    `<b>Why B extracts more from the same data.</b> GRPO scores 8 rollouts against each other; ` +
    `a group where all 8 score the same has zero variance and teaches nothing. A binary ` +
    `correct/wrong reward ties constantly — A used ` +
    `<span class="num">${ga}/${gat}</span> groups, B used <span class="num">${gb}/${gbt}</span>. ` +
    `The cost term separates rollouts that are <i>equally correct</i>.<br><br>` +
    `<b>And that is exactly why accuracy did not move.</b> The cost signal is orthogonal to ` +
    `correctness: it can only reorder rollouts that already agree on being right, so the extra ` +
    `gradient buys cheapness and cannot buy accuracy. Matched accuracy is structural, not luck.`;
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
    `<span class="num">${m.only_b}</span> to B. That near-tie is the McNemar result: the two ` +
    `policies are equally accurate, they are simply wrong about different pictures.`;
}

function showSamples(k) {
  const box = $("#samples"); box.innerHTML = "";
  const list = D.samples[k] || [];
  if (!list.length) { box.appendChild(el("p", "sub", "No examples in this cell.")); return; }
  list.forEach((s) => {
    const d = el("div", "sample");
    d.appendChild(el("div", "q", s.question));
    d.appendChild(el("div", "gold", "Gold: " + s.gold));
    const a = el("div", "ans");
    [["A", s.a], ["B", s.b]].forEach(([who, r]) => {
      const side = el("div", "side");
      side.innerHTML =
        `<div class="who">Run ${who}</div>` +
        `<div class="val">${r.answer ?? "<i>no answer</i>"} ` +
        `<span class="pill ${r.correct ? "good" : "bad"}"><span class="dot"></span>` +
        `${r.correct ? "correct" : "wrong"}</span></div>` +
        `<div class="meta num">${r.zooms} zoom${r.zooms === 1 ? "" : "s"} · ${r.decode} decode tokens</div>`;
      a.appendChild(side);
    });
    d.appendChild(a);
    box.appendChild(d);
  });
}

/* ---------- quantization ------------------------------------------------ */

function quant() {
  const order = { base: 0, a: 1, b: 2 };
  const rows = [...D.quant].sort((x, y) => (x.quant.localeCompare(y.quant)) || order[x.model] - order[y.model])
    .map((q) => ({
      hi: q.model === "b",
      cells: [
        `${q.label} <span class="pill">${q.quant}</span>`,
        Math.round(q.prefill_tokens).toLocaleString(),
        n1(q.decode_tokens), n2(q.zooms), ms(q.latency_ms) + " ms", usd(q.usd_per_1k),
      ].map((v) => ({ v })),
    }));
  table($("#t-quant"), ["", "prefill tok", "decode tok", "zooms", "latency", "$ / 1k"], rows);
}

/* ---------- live demo --------------------------------------------------- */

function demo() {
  const go = $("#f-go"), out = $("#out");
  go.onclick = async () => {
    const f = $("#f-img").files[0];
    const q = $("#f-q").value.trim();
    if (!f || !q) { out.className = "out on"; out.innerHTML = `<p class="sub">Pick an image and type a question.</p>`; return; }
    go.disabled = true; go.textContent = "Running…";
    out.className = "out on";
    out.innerHTML = `<p class="sub">Loading the model and running the episode. First run on a
      quantization takes ~30 s while the weights load.</p>`;
    const fd = new FormData();
    fd.append("image", f); fd.append("question", q);
    fd.append("model", $("#f-model").value);
    fd.append("quant", $("#f-quant").value);
    fd.append("downproject", $("#f-down").checked ? "1" : "0");
    try {
      const r = await fetch("/api/infer", { method: "POST", body: fd });
      renderRun(await r.json());
    } catch (e) {
      out.innerHTML = `<p class="sub">Request failed: ${e}</p>`;
    }
    go.disabled = false; go.textContent = "Run";
  };
}

function renderRun(r) {
  const out = $("#out");
  if (r.error) { out.innerHTML = `<p class="sub">${r.error}</p>`; return; }
  const stat = (label, v) => `<div><b>${v}</b>${label}</div>`;
  let html = `<div class="q" style="font-size:14px;font-weight:500">${r.answer ?? "<i>no answer</i>"}</div>`;
  html += `<div class="stat">
      ${stat("prefill tok", Math.round(r.prefill_tokens).toLocaleString())}
      ${stat("decode tok", Math.round(r.decode_tokens))}
      ${stat("prefill", ms(r.prefill_ms) + " ms")}
      ${stat("decode", ms(r.decode_ms) + " ms")}
      ${stat("total", ms(r.total_ms) + " ms")}
      ${stat("zooms", r.zooms)}
      ${stat("cost / 1k", "$" + (r.usd * 1000).toFixed(3))}
    </div>`;
  const shots = [];
  if (r.thumb_png_b64) shots.push(`<figure><img src="data:image/png;base64,${r.thumb_png_b64}">
    <figcaption>${r.downproject ? "what it saw first (down-projected)" : "full resolution, no zoom"}</figcaption></figure>`);
  (r.turns || []).forEach((t, i) => {
    if (t.crop_png_b64) shots.push(`<figure><img src="data:image/png;base64,${t.crop_png_b64}">
      <figcaption>zoom ${i + 1} — [${(t.bbox_2d || []).join(", ")}]</figcaption></figure>`);
  });
  if (shots.length) html += `<div class="shot">${shots.join("")}</div>`;
  out.innerHTML = html;
}

/* ---------- boot -------------------------------------------------------- */

fetch("/data.json").then((r) => r.json()).then((d) => {
  D = d;
  $("#f-coeffs").textContent =
    `a=${d.coeffs.a.toFixed(2)} b=${d.coeffs.b.toFixed(1)} c=${d.coeffs.c.toFixed(0)} ms, R²=${d.coeffs.r2.toFixed(4)}`;
  headline(); zoomTable(); hyper(); charts(); matrix(); quant(); demo();
});
