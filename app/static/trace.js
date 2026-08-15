/* One V*Bench question, both policies, end to end. Reads the same data.json the main page
   reads, so nothing here can drift from the runs. */

const $ = (s) => document.querySelector(s);
const esc = (s) => (s || "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

const WHO = { a: { name: "No cost", color: "#a1a1ac" }, b: { name: "Cost-aware", color: "#2f5cff" } };
const CELL = {
  both_right: "Both right",
  only_a: "Only no-cost got it",
  only_b: "Only cost-aware got it",
  both_wrong: "Both wrong",
};

function boxOverlay(sid, boxes, color) {
  // Boxes are on a 0-1000 grid over the whole image, so percentages map straight across.
  const marks = (boxes || []).map((b, i) => {
    const [x1, y1, x2, y2] = b;
    const st = `left:${x1 / 10}%;top:${y1 / 10}%;width:${(x2 - x1) / 10}%;height:${(y2 - y1) / 10}%;` +
      `border-color:${color}`;
    return `<span class="bx" style="${st}"><i style="background:${color}">${i + 1}</i></span>`;
  }).join("");
  return `<div class="shotwrap"><img src="/img/${esc(sid)}" alt="">${marks}</div>`;
}

function turnsHtml(trace, correct) {
  if (!trace || !trace.length) return `<p class="tmeta">No turns recorded.</p>`;
  return trace.map((turn, i) => {
    let h = `<div class="turn"><div class="tno">turn <span class="n">${i + 1}</span></div>`;
    if (turn.think) {
      h += `<div class="tblk"><span class="tk">thinking</span><p>${esc(turn.think)}</p></div>`;
    }
    if (turn.tool) {
      const a = turn.tool.arguments || {};
      const label = a.label ? `,\n  label = ${esc(JSON.stringify(a.label))}` : "";
      h += `<div class="tblk"><span class="tk">tool call</span><pre class="num">${esc(turn.tool.name || "image_zoom_in_tool")}(
  bbox_2d = [${(a.bbox_2d || []).join(", ")}]${label}
)</pre></div>`;
    }
    if (turn.crop_vision_tokens) {
      h += `<div class="tblk"><span class="tk">crop returned</span>` +
        `<p class="cost">the crop cost <b>${turn.crop_vision_tokens}</b> vision tokens</p></div>`;
    }
    if (turn.answer) {
      h += `<div class="tblk"><span class="tk">final answer</span>` +
        `<div class="fin${correct ? "" : " bad"}">${esc(turn.answer)}</div></div>`;
    }
    return h + `</div>`;
  }).join("");
}

function colHead(who, r) {
  return `<div class="head">
    <span class="name">${WHO[who].name}</span>
    <span class="spacer"></span>
    <span class="pill ${r.correct ? "good" : "bad"}"><span class="dot"></span>${r.correct ? "correct" : "wrong"}</span>
  </div>`;
}

function shotCol(who, s) {
  const r = s[who] || {};
  const n = (r.boxes || []).length;
  return `<div class="tcol ${who}">
    ${colHead(who, r)}
    ${boxOverlay(s.sid, r.boxes, WHO[who].color)}
    <div class="tans">
      <div class="k">Answered</div>
      <div class="v">${r.answer ? esc(r.answer) : "<i>no answer</i>"}</div>
    </div>
    <div class="tstat">
      <div><b>${r.zooms != null ? r.zooms : n}</b>zoom${(r.zooms != null ? r.zooms : n) === 1 ? "" : "s"}</div>
      <div><b>${r.vision != null ? r.vision : "—"}</b>vision tok</div>
      <div><b>${r.decode != null ? r.decode : "—"}</b>decode tok</div>
    </div>
  </div>`;
}

function traceCol(who, s) {
  const r = s[who] || {};
  return `<div class="tcol ${who}">
    ${colHead(who, r)}
    <div class="turns">${turnsHtml(r.trace, r.correct)}</div>
  </div>`;
}

function render(s, cellName) {
  document.title = s.question + " · trace";
  $("#t-eyebrow").textContent = cellName ? `Trace · ${cellName}` : "Trace";
  $("#t-q").textContent = s.question;
  $("#t-gold").innerHTML = `Gold answer: <b>${esc(s.gold)}</b>`;
  const sid = `<p class="sid num">${esc(s.sid)}</p>`;
  $("#thead").insertAdjacentHTML("beforeend", sid);
  $("#t-body").innerHTML =
    `<div class="tpair">${shotCol("a", s)}${shotCol("b", s)}</div>
     <section class="treel">
       <h2>Turn by turn</h2>
       <p class="sub">What each policy thought, where it chose to look, and what the crop cost it.</p>
       <div class="tpair">${traceCol("a", s)}${traceCol("b", s)}</div>
     </section>`;
}

function missing(msg) {
  $("#t-q").textContent = "Trace not found";
  $("#t-gold").textContent = "";
  $("#t-body").innerHTML = `<p class="miss">${esc(msg)} <a href="/">Back to the examples</a>.</p>`;
}

function boot() {
  const sid = new URLSearchParams(location.search).get("sid");
  if (!sid) return missing("No sid in the URL. This page needs /trace.html?sid=…");
  fetch("/data.json", { cache: "no-store" })
    .then((r) => r.json())
    .then((D) => {
      for (const [k, list] of Object.entries(D.samples || {})) {
        const hit = (list || []).find((s) => s.sid === sid);
        if (hit) return render(hit, CELL[k] || k);
      }
      missing(`No sample with sid ${sid}.`);
    })
    .catch((e) => missing(`Could not load data.json (${e}).`));
}

boot();
