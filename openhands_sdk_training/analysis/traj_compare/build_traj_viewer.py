#!/usr/bin/env python3
"""Build a standalone step-by-step trajectory reader from `export_trajectories.py`
output.

One HTML file, no network, no build step. Left rail lists every trajectory
grouped by source (base rollout / SFT-arm rollout / training demonstration);
the main pane walks the events in order -- reasoning, tool call with its full
arguments, observation -- with a compare mode that puts two of them side by side.

    python3 build_traj_viewer.py --in exported/ --out trajectories.html

Tool colours and the light/dark surface tokens are the same ones
`build_dashboard.py` uses, so a tool means the same colour across both views.
"""
import argparse, html, json, os, sys, collections

PAL = {
    "s1_light": "#2a78d6", "s1_dark": "#3987e5",
    "s2_light": "#eb6834", "s2_dark": "#d95926",
    "neg_light": "#e34948", "neg_dark": "#e66767",
    "pos_light": "#1baf7a", "pos_dark": "#199e70",
}
# Same fixed tool identity as the dashboard -- never cycled.
TOOL_COLORS = {"terminal": ("#2a78d6", "#3987e5"), "file_editor": ("#eb6834", "#d95926"),
               "think": ("#1baf7a", "#199e70"), "finish": ("#4a3aa7", "#9085e9")}

CSS = """
:root{color-scheme:light dark}
*{box-sizing:border-box}
body{margin:0;
  --surface-1:#fcfcfb; --surface-2:#f4f3f0; --surface-3:#ecebe6; --border:#e0dfda;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#77766f;
  --s1:%(s1_light)s; --s2:%(s2_light)s; --neg:%(neg_light)s; --pos:%(pos_light)s;
  --t-terminal:#2a78d6; --t-file_editor:#eb6834; --t-think:#1baf7a; --t-finish:#4a3aa7;
  background:var(--surface-1); color:var(--text-primary);
  font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;}
@media (prefers-color-scheme:dark){body:where(:not([data-theme="light"])){
  --surface-1:#1a1a19; --surface-2:#232322; --surface-3:#2b2b29; --border:#3a3a37;
  --text-primary:#fff; --text-secondary:#c3c2b7; --text-muted:#96958c;
  --s1:%(s1_dark)s; --s2:%(s2_dark)s; --neg:%(neg_dark)s; --pos:%(pos_dark)s;
  --t-terminal:#3987e5; --t-file_editor:#d95926; --t-think:#199e70; --t-finish:#9085e9;}}
body[data-theme="dark"]{
  --surface-1:#1a1a19; --surface-2:#232322; --surface-3:#2b2b29; --border:#3a3a37;
  --text-primary:#fff; --text-secondary:#c3c2b7; --text-muted:#96958c;
  --s1:%(s1_dark)s; --s2:%(s2_dark)s; --neg:%(neg_dark)s; --pos:%(pos_dark)s;
  --t-terminal:#3987e5; --t-file_editor:#d95926; --t-think:#199e70; --t-finish:#9085e9;}

.wrap{display:grid;grid-template-columns:290px 1fr;height:100vh}
aside{border-right:1px solid var(--border);background:var(--surface-2);overflow-y:auto;padding:14px 0}
aside h1{font-size:14px;margin:0 14px 2px;letter-spacing:-.01em}
aside .sub{font-size:11.5px;color:var(--text-muted);margin:0 14px 12px}
.grp{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--text-secondary);
     margin:14px 14px 4px;display:flex;align-items:center;gap:6px}
.grp .swatch{width:8px;height:8px;border-radius:2px;background:var(--text-muted)}
.grp.eval .swatch{background:var(--s1)} .grp.train .swatch{background:var(--s2)}
.item{padding:6px 14px;cursor:pointer;border-left:3px solid transparent;font-size:12.5px}
.item:hover{background:var(--surface-3)}
.item.sel{background:var(--surface-3);border-left-color:var(--s1)}
.item.selB{border-left-color:var(--s2)}
.item .iid{display:block;color:var(--text-primary);word-break:break-all;line-height:1.35}
.item .m{color:var(--text-muted);font-size:11px;font-variant-numeric:tabular-nums}
main{overflow-y:auto;padding:18px 24px 120px}
.cols{display:grid;gap:18px}
.cols.two{grid-template-columns:1fr 1fr}
header.th{position:sticky;top:0;background:var(--surface-1);padding-bottom:8px;z-index:2;
  border-bottom:1px solid var(--border);margin-bottom:10px}
header.th h2{font-size:15px;margin:0 0 3px;letter-spacing:-.005em;word-break:break-all}
.chips{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.chip{font-size:10.5px;border:1px solid var(--border);border-radius:5px;padding:1px 6px;
      color:var(--text-secondary);background:var(--surface-2);white-space:nowrap}
.chip.ok{color:var(--pos);border-color:var(--pos)}
.chip.bad{color:var(--neg);border-color:var(--neg)}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:8px 0;
     border-bottom:1px solid var(--border);margin-bottom:12px;font-size:12.5px}
.bar label{color:var(--text-secondary);display:flex;gap:4px;align-items:center;cursor:pointer}
.bar input[type=text]{background:var(--surface-2);border:1px solid var(--border);border-radius:6px;
  padding:4px 8px;color:var(--text-primary);font:inherit;font-size:12.5px;width:190px}
.bar select{background:var(--surface-2);border:1px solid var(--border);border-radius:6px;
  padding:3px 6px;color:var(--text-primary);font:inherit;font-size:12px;max-width:260px}
button.b{background:var(--surface-2);border:1px solid var(--border);border-radius:6px;
  padding:3px 9px;color:var(--text-secondary);font:inherit;font-size:12px;cursor:pointer}
button.b:hover{color:var(--text-primary)}

.evt{border:1px solid var(--border);border-radius:9px;margin:7px 0;background:var(--surface-2);
     overflow:hidden;scroll-margin-top:96px}
.evt.cur{outline:2px solid var(--s1);outline-offset:1px}
.eh{display:flex;gap:8px;align-items:baseline;padding:6px 10px;cursor:pointer;
    border-left:3px solid var(--border)}
.evt.think .eh{border-left-color:var(--t-think)}
/* only the tools we know get an identity colour; anything else (including a
   hallucinated tool name) stays neutral rather than masquerading as a shell */
.evt.call .eh{border-left-color:var(--text-muted)}
.evt.call.terminal .eh{border-left-color:var(--t-terminal)}
.evt.call.file_editor .eh{border-left-color:var(--t-file_editor)}
.evt.call.finish .eh{border-left-color:var(--t-finish)}
.evt.badtool .eh{border-left-color:var(--neg)} .evt.badtool .k{color:var(--neg)}
.evt.boundary{background:none;border:0;border-top:1px dashed var(--border);border-radius:0;
  margin:16px 0 6px}
.evt.boundary .eh{border-left:0;color:var(--text-muted);font-size:11.5px;cursor:default}
.evt.boundary .eb{display:none!important}
.evt.obs .eh{border-left-color:var(--text-muted)}
.evt.obs.err .eh{border-left-color:var(--neg)}
.evt.msg .eh{border-left-color:var(--text-muted)}
.evt.condense .eh{border-left-color:var(--t-finish)}
.n{font-size:11px;color:var(--text-muted);font-variant-numeric:tabular-nums;min-width:26px}
.k{font-size:11px;text-transform:uppercase;letter-spacing:.05em;font-weight:600}
.evt.think .k{color:var(--t-think)} .evt.call .k{color:var(--text-secondary)}
.evt.call.terminal .k{color:var(--t-terminal)}
.evt.call.file_editor .k{color:var(--t-file_editor)}
.evt.call.finish .k{color:var(--t-finish)}
.evt.obs .k{color:var(--text-muted)} .evt.obs.err .k{color:var(--neg)}
.evt.msg .k,.evt.condense .k{color:var(--text-muted)}
.peek{color:var(--text-secondary);font-size:12px;white-space:nowrap;overflow:hidden;
      text-overflow:ellipsis;flex:1;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.eb{display:none;padding:2px 10px 10px 13px}
.evt.open .eb{display:block}
.argk{font-size:11px;color:var(--text-muted);margin:6px 0 1px;font-family:ui-monospace,monospace}
pre{margin:0;padding:8px 10px;background:var(--surface-1);border:1px solid var(--border);
    border-radius:6px;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
    white-space:pre-wrap;word-break:break-word;max-height:340px;overflow:auto;color:var(--text-primary)}
pre.tall{max-height:none}
pre.think{white-space:pre-wrap;font-family:inherit;font-size:13px;line-height:1.6}
.more{font-size:11px;color:var(--text-muted);cursor:pointer;margin-top:3px;display:inline-block}
.hide{display:none}
details.instr{margin:0 0 12px} details.instr summary{cursor:pointer;font-size:12.5px;color:var(--text-secondary)}
.empty{color:var(--text-muted);font-size:13px;padding:30px 0}
kbd{font:11px ui-monospace,monospace;border:1px solid var(--border);border-bottom-width:2px;
    border-radius:4px;padding:0 4px;color:var(--text-secondary)}
"""

JS = r"""
const DATA = JSON.parse(document.getElementById("data").textContent);
const byKey = {};
DATA.trajectories.forEach(t => byKey[t.label + "||" + t.id] = t);
let selA = null, selB = null, cur = -1;

const esc = s => (s == null ? "" : String(s)).replace(/[&<>"]/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c]));
// Tool names are model output, not a trusted identifier: base rollouts contain
// hallucinated names and ones where the model broke its own tool-call syntax
// mid-name (embedded </think>, <tool_call>, ...). Never let that reach a class
// attribute raw.
const slug = s => String(s == null ? "" : s).replace(/[^A-Za-z0-9_-]/g, "").slice(0, 40);
const cleanTool = s => /^[a-z_][a-z0-9_]*$/i.test(String(s == null ? "" : s));

function peekOf(e){
  if (e.kind === "call"){
    const a = e.args || {};
    const k = ["command","path","file_text","old_str","new_str","thought","message"].find(x => x in a)
              || Object.keys(a)[0];
    return k ? (k === "command" ? a[k] : k + "=" + a[k]) : "";
  }
  if (e.kind === "obs") return e.text || "";
  return e.text || "";
}

function evtHTML(e, i){
  const cls = ["evt", e.kind];
  const bad = e.kind === "call" && !cleanTool(e.tool);
  if (e.kind === "call" && e.tool) cls.push(slug(e.tool));
  if (bad) cls.push("badtool");
  if (e.kind === "obs" && e.err) cls.push("err");
  let label = e.kind === "call" ? (e.tool || "call")
            : e.kind === "think" ? "think"
            : e.kind === "obs" ? "result"
            : e.kind === "msg" ? (e.role || "msg") : e.kind;
  let body = "";
  if (e.kind === "call"){
    for (const [k, v] of Object.entries(e.args || {}))
      body += `<div class="argk">${esc(k)}</div><pre>${esc(v)}</pre>`;
    if (e.cats && e.cats.length) body += `<div class="argk">categories: ${esc(e.cats.join(", "))}</div>`;
  } else if (e.kind === "obs"){
    body = `<pre>${esc(e.text) || "<em>(empty)</em>"}</pre>`;
  } else {
    body = `<pre class="think">${esc(e.text)}</pre>`;
  }
  const meta = (e.kind === "obs" && e.err ? ' <span class="chip bad">error</span>' : "")
             + (bad ? ' <span class="chip bad">malformed tool name</span>' : "");
  const src = e.kind === "think" && e.source ? ` <span class="chip">${esc(e.source)}</span>` : "";
  return `<div class="${cls.join(" ")}" data-i="${i}" id="e${i}">
    <div class="eh"><span class="n">${i}</span><span class="k">${esc(label)}</span>${meta}${src}
      <span class="peek">${esc(peekOf(e).slice(0, 300).replace(/\s+/g, " "))}</span>
      <span class="n">${(peekOf(e) || "").length.toLocaleString()}c</span></div>
    <div class="eb">${body}</div></div>`;
}

function visible(e){
  const box = document.getElementById("f-" + e.kind);
  if (box && !box.checked) return false;
  const q = document.getElementById("q").value.trim().toLowerCase();
  if (!q) return true;
  return JSON.stringify(e).toLowerCase().includes(q);
}

function colHTML(t, side){
  if (!t) return "";
  const m = t.meta || {};
  const chips = [];
  chips.push(`<span class="chip">${esc(t.kind)}</span>`);
  chips.push(`<span class="chip">${esc(t.label)}</span>`);
  if (t.resolved === true) chips.push('<span class="chip ok">resolved</span>');
  if (t.resolved === false) chips.push('<span class="chip bad">unresolved</span>');
  if (t.error) chips.push(`<span class="chip bad">error: ${esc(t.error.slice(0,60))}</span>`);
  chips.push(`<span class="chip">${m.n_events} events</span>`);
  chips.push(`<span class="chip">${m.n_calls} calls</span>`);
  chips.push(`<span class="chip">${m.n_think} think</span>`);
  if (m.n_condense) chips.push(`<span class="chip">${m.n_condense} condensations</span>`);
  if (m.record_type && m.record_type !== "trajectory")
    chips.push(`<span class="chip bad">record_type: ${esc(m.record_type)}</span>`);
  if (m.segments) chips.push(`<span class="chip">segments ${esc(m.segments.join(","))} (${m.n_rows} rows)</span>`);
  else if (m.segment != null) chips.push(`<span class="chip">segment ${m.segment}</span>`);
  if (m.forgotten_event_count != null)
    chips.push(`<span class="chip">forgot ${m.forgotten_event_count} events</span>`);
  if (m.source_dataset) chips.push(`<span class="chip">${esc(m.source_dataset)}</span>`);
  if (m.empty_history) chips.push('<span class="chip bad">empty history</span>');
  const nbad = t.events.filter(e => e.kind === "call" && !cleanTool(e.tool)).length;
  if (nbad) chips.push(`<span class="chip bad">${nbad} malformed tool calls</span>`);

  const evs = t.events.map((e, i) => [e, i]).filter(([e]) => visible(e));
  const shown = evs.length, total = t.events.length;
  return `<div class="col" data-side="${side}">
    <header class="th"><h2>${esc(t.title)}</h2><div class="chips">${chips.join("")}</div></header>
    ${t.instruction ? `<details class="instr"><summary>task statement (${t.instruction.length.toLocaleString()} chars)</summary><pre>${esc(t.instruction)}</pre></details>` : ""}
    ${t.patch ? `<details class="instr"><summary>final patch (${t.patch.length.toLocaleString()} chars)</summary><pre>${esc(t.patch)}</pre></details>` : ""}
    <div class="cnt chip">showing ${shown} of ${total} events</div>
    <div class="evts">${evs.map(([e, i]) => evtHTML(e, i)).join("") || '<div class="empty">no events match the filter</div>'}</div>
  </div>`;
}

function render(){
  const a = byKey[selA], b = selB ? byKey[selB] : null;
  const main = document.getElementById("cols");
  main.className = "cols" + (b ? " two" : "");
  main.innerHTML = colHTML(a, "A") + (b ? colHTML(b, "B") : "");
  main.querySelectorAll(".eh").forEach(h => h.onclick = () => h.parentElement.classList.toggle("open"));
  document.querySelectorAll(".item").forEach(el => {
    el.classList.toggle("sel", el.dataset.key === selA);
    el.classList.toggle("selB", el.dataset.key === selB);
  });
  cur = -1;
}

function step(d){
  const evs = [...document.querySelectorAll('.col[data-side="A"] .evt')];
  if (!evs.length) return;
  evs.forEach(e => e.classList.remove("cur"));
  cur = Math.max(0, Math.min(evs.length - 1, cur + d));
  const el = evs[cur];
  el.classList.add("cur"); el.classList.add("open");
  el.scrollIntoView({block: "center", behavior: "smooth"});
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".item").forEach(el => {
    el.onclick = ev => {
      if (ev.shiftKey){ selB = (selB === el.dataset.key) ? null : el.dataset.key; }
      else { selA = el.dataset.key; }
      render();
    };
  });
  document.querySelectorAll(".bar input").forEach(i => i.oninput = render);
  document.getElementById("expand").onclick = () =>
    document.querySelectorAll(".evt").forEach(e => e.classList.add("open"));
  document.getElementById("collapse").onclick = () =>
    document.querySelectorAll(".evt").forEach(e => e.classList.remove("open"));
  document.getElementById("untall").onclick = () =>
    document.querySelectorAll("pre").forEach(p => p.classList.toggle("tall"));
  document.getElementById("swap").onclick = () => { const t = selA; selA = selB || selA; selB = t === selA ? null : t; render(); };
  document.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT") { if (e.key === "Escape") e.target.blur(); return; }
    if (e.key === "j") { step(1); e.preventDefault(); }
    if (e.key === "k") { step(-1); e.preventDefault(); }
    if (e.key === "/") { document.getElementById("q").focus(); e.preventDefault(); }
  });
  selA = DATA.first;
  render();
});
"""


def build(indir, out, title):
    idx_path = os.path.join(indir, "index.json")
    if not os.path.exists(idx_path):
        sys.exit(f"no index.json in {indir} -- run export_trajectories.py first")
    index = json.load(open(idx_path))

    trajs = []
    for row in index:
        p = os.path.join(indir, "traj", row["label"], f"{row['id']}.json")
        if not os.path.exists(p):
            print(f"WARNING missing {p}", file=sys.stderr)
            continue
        trajs.append(json.load(open(p)))
    if not trajs:
        sys.exit("no trajectories found")

    # Sidebar: group by label, evals first (they are what you usually open).
    groups = collections.OrderedDict()
    for t in sorted(trajs, key=lambda t: (t["kind"] != "eval", t["label"], t["id"])):
        groups.setdefault((t["label"], t["kind"]), []).append(t)

    rail = []
    for (label, kind), items in groups.items():
        rail.append(f'<div class="grp {html.escape(kind)}"><span class="swatch"></span>'
                    f'{html.escape(label)} <span class="m">({len(items)})</span></div>')
        for t in items:
            m = t["meta"]
            badge = ""
            if t["resolved"] is True:
                badge = " &check;"
            elif t["resolved"] is False:
                badge = " &times;"
            key = f'{t["label"]}||{t["id"]}'
            rt = m.get("record_type")
            rt = f' &middot; {html.escape(str(rt))}' if rt and rt != "trajectory" else ""
            rail.append(
                f'<div class="item" data-key="{html.escape(key)}">'
                f'<span class="iid">{html.escape(t["title"])}{badge}</span>'
                f'<span class="m">{m["n_events"]} ev &middot; {m["n_calls"]} calls'
                f' &middot; {m["n_think"]} think{rt}</span></div>')

    payload = {"trajectories": trajs,
               "first": f'{trajs[0]["label"]}||{trajs[0]["id"]}'}
    blob = json.dumps(payload, default=str).replace("</", "<\\/")

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS % PAL}</style></head><body>
<div class="wrap">
<aside>
  <h1>{html.escape(title)}</h1>
  <div class="sub">click to open &middot; shift-click to compare &middot;
    <kbd>j</kbd>/<kbd>k</kbd> step &middot; <kbd>/</kbd> search</div>
  {''.join(rail)}
</aside>
<main>
  <div class="bar">
    <label><input type="checkbox" id="f-think" checked> think</label>
    <label><input type="checkbox" id="f-call" checked> tool calls</label>
    <label><input type="checkbox" id="f-obs" checked> observations</label>
    <label><input type="checkbox" id="f-msg" checked> messages</label>
    <label><input type="checkbox" id="f-condense" checked> condensations</label>
    <input type="text" id="q" placeholder="filter events&hellip;">
    <button class="b" id="expand">expand all</button>
    <button class="b" id="collapse">collapse all</button>
    <button class="b" id="untall">toggle full height</button>
    <button class="b" id="swap">swap A/B</button>
  </div>
  <div id="cols" class="cols"></div>
</main>
</div>
<script id="data" type="application/json">{blob}</script>
<script>{JS}</script>
</body></html>"""
    with open(out, "w") as g:
        g.write(doc)
    n_ev = sum(len(t["events"]) for t in trajs)
    print(f"wrote {out}  ({len(trajs)} trajectories, {n_ev:,} events, "
          f"{os.path.getsize(out) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="indir", required=True, help="export_trajectories.py output dir")
    ap.add_argument("--out", default="trajectories.html")
    ap.add_argument("--title", default="Trajectory reader")
    a = ap.parse_args()
    build(a.indir, a.out, a.title)


if __name__ == "__main__":
    main()
