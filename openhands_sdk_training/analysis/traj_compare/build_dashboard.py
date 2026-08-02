#!/usr/bin/env python3
"""Build a self-contained HTML dashboard comparing two models' SWE-bench
agent trajectories, instance-by-instance.

Inputs (all produced by extract_traj_stats.py + the eval's merged.report.json):
  --stats     stats.jsonl        one row per (model, instance_id)
  --report    NAME=path.json     merged.report.json per model (for resolved_ids)
  --digest    dir                optional; per-turn digests for the pair viewer
  --pairs     file.json          optional; [{"tag":..., "iid":...}] pairs to feature

Output: one standalone .html file (no network, no build step).

Design follows the house data-viz rules: categorical hues assigned in fixed
order and never cycled, one axis per plot, diverging encoding only for polarity,
legend always present for >=2 series, hover tooltips on every mark, light+dark
both stepped from the same ramps.
"""
import argparse, json, html, statistics as st
from collections import Counter, defaultdict

# ---------------------------------------------------------------- palette
# Validated with the house validator (scripts/validate_palette.js):
#   light "#2a78d6,#eb6834" and dark "#3987e5,#d95926" -> ALL CHECKS PASS.
#   diverging pair "#2a78d6,#e34948" light -> ALL CHECKS PASS.
PAL = {
    "s1_light": "#2a78d6", "s1_dark": "#3987e5",     # slot 1 - model A (base)
    "s2_light": "#eb6834", "s2_dark": "#d95926",     # slot 2 - model B (arm)
    "neg_light": "#e34948", "neg_dark": "#e66767",   # diverging: arm worse
    "pos_light": "#2a78d6", "pos_dark": "#3987e5",   # diverging: arm better
}
# tool identity: fixed order, never cycled (terminal, file_editor, think, finish)
TOOL_COLORS = [("terminal", "#2a78d6", "#3987e5"), ("file_editor", "#eb6834", "#d95926"),
               ("think", "#1baf7a", "#199e70"), ("finish", "#4a3aa7", "#9085e9")]


def esc(s):
    return html.escape(str(s), quote=True)


def median(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else 0


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0


def pct(part, whole):
    return (100.0 * part / whole) if whole else 0.0


def repo_of(iid):
    return iid.split("__")[0]


# ---------------------------------------------------------------- svg helpers
def svg_open(w, h, cls="plot"):
    return (f'<svg class="{cls}" viewBox="0 0 {w} {h}" width="100%" height="{h}" '
            f'role="img" preserveAspectRatio="xMidYMid meet">')


def tip(text):
    """Native SVG tooltip - works with no JS and survives copy/paste of the file."""
    return f"<title>{esc(text)}</title>"


# ---------------------------------------------------------------- panels
def panel_hero(A, B, rows):
    """Stat tiles. The number IS the chart - no one-bar bar charts."""
    a_ok = [r for r in rows if r["model"] == A and not r.get("error")]
    b_all = [r for r in rows if r["model"] == B]
    a_res = sum(1 for r in rows if r["model"] == A and r["resolved"])
    b_res = sum(1 for r in rows if r["model"] == B and r["resolved"])
    a_ver = pct(sum(1 for r in a_ok if r.get("verified_after_edit")), len(a_ok))
    b_ver = pct(sum(1 for r in b_all if r.get("verified_after_edit")), len(b_all))
    def unver(rs):
        return pct(sum(1 for r in rs if r.get("finish_claims_success")
                       and not r.get("verified_after_edit")), len(rs))
    a_un, b_un = unver(a_ok), unver(b_all)
    # `worse` states the VALENCE of the change explicitly: colouring a delta by its
    # arithmetic sign would paint "+90pp claims-success-without-verifying" as good.
    tiles = [
        ("resolved", f"{a_res}", f"{b_res}", f"{b_res - a_res:+d}", "of 500 instances", True),
        ("re-tested after last edit", f"{a_ver:.0f}%", f"{b_ver:.0f}%",
         f"{b_ver - a_ver:+.0f} pp", "the verification loop", True),
        ("claims success, never verified", f"{a_un:.0f}%", f"{b_un:.0f}%",
         f"{b_un - a_un:+.0f} pp", "verification-shaped text", True),
        ("median turns", f"{median([r['n_actions'] for r in a_ok]):.0f}",
         f"{median([r['n_actions'] for r in b_all]):.0f}", "", "agent steps per instance", None),
    ]
    out = ['<div class="tiles">']
    for label, av, bv, delta, sub, worse in tiles:
        dcls = "" if worse is None else ("down" if worse else "up")
        out.append(f'''<div class="tile">
          <div class="tile-label">{esc(label)}</div>
          <div class="tile-pair">
            <span class="tv a">{esc(av)}</span>
            <span class="arrow">&rarr;</span>
            <span class="tv b">{esc(bv)}</span>
          </div>
          <div class="tile-delta {dcls}">{esc(delta)}</div>
          <div class="tile-sub">{esc(sub)}</div>
        </div>''')
    out.append("</div>")
    return "".join(out)


def panel_repo_dumbbell(A, B, rows, nameA, nameB):
    """Per-repo resolve rate, paired. Dumbbell = two points + connector: shows
    level AND change on ONE axis (never a dual axis)."""
    per = defaultdict(lambda: {"n": 0, A: 0, B: 0})
    for r in rows:
        if r["model"] not in (A, B):
            continue
        rp = repo_of(r["instance_id"])
        if r["model"] == A:
            per[rp]["n"] += 1
        per[rp][r["model"]] += 1 if r["resolved"] else 0
    items = [(k, v) for k, v in per.items() if v["n"] >= 5]
    items.sort(key=lambda kv: pct(kv[1][A], kv[1]["n"]) - pct(kv[1][B], kv[1]["n"]), reverse=True)

    row_h, top, left, right = 26, 34, 132, 56
    w, h = 720, top + row_h * len(items) + 26
    xmax = 50.0
    def X(p):
        return left + (p / xmax) * (w - left - right)

    s = [svg_open(w, h)]
    for gx in range(0, int(xmax) + 1, 10):     # recessive hairline grid
        s.append(f'<line class="grid" x1="{X(gx):.1f}" y1="{top-14}" x2="{X(gx):.1f}" y2="{h-22}"/>')
        s.append(f'<text class="ax" x="{X(gx):.1f}" y="{h-8}" text-anchor="middle">{gx}%</text>')
    for i, (rp, v) in enumerate(items):
        y = top + i * row_h
        pa, pb = pct(v[A], v["n"]), pct(v[B], v["n"])
        s.append(f'<text class="rowlab" x="{left-12}" y="{y+4}" text-anchor="end">{esc(rp)}</text>')
        s.append(f'<text class="rown" x="{left-12}" y="{y+15}" text-anchor="end">n={v["n"]}</text>')
        n = v["n"]
        s.append(f'<line class="conn" x1="{X(pa):.1f}" y1="{y}" x2="{X(pb):.1f}" y2="{y}"/>')
        ta = tip(f"{rp}: {nameA} {v[A]}/{n} ({pa:.1f}%)")
        tb = tip(f"{rp}: {nameB} {v[B]}/{n} ({pb:.1f}%)")
        s.append(f'<g>{ta}<circle class="pt a" cx="{X(pa):.1f}" cy="{y}" r="5.5"/></g>')
        s.append(f'<g>{tb}<circle class="pt b" cx="{X(pb):.1f}" cy="{y}" r="5.5"/></g>')
        d = v[B] - v[A]
        dc = "delta-neg" if d < 0 else ("delta-pos" if d > 0 else "delta-zero")
        s.append(f'<text class="{dc}" x="{w-10}" y="{y+4}" text-anchor="end">{d:+d}</text>')
    s.append("</svg>")
    return "".join(s)


def panel_discordance(A, B, rows):
    """Diverging: instances LOST (base-only) vs GAINED (arm-only) per slice.
    Polarity -> diverging pair + neutral midpoint, never a hue at the middle."""
    res = {m: {r["instance_id"] for r in rows if r["model"] == m and r["resolved"]}
           for m in (A, B)}
    allids = {r["instance_id"] for r in rows if r["model"] == A}
    slices = []
    repos = Counter(repo_of(i) for i in allids)
    for rp, n in repos.most_common():
        if n < 15:
            continue
        ids = {i for i in allids if repo_of(i) == rp}
        slices.append((rp, len(ids), len((ids & res[A]) - res[B]), len((ids & res[B]) - res[A])))
    small = {i for i in allids if repos[repo_of(i)] < 15}
    if small:
        slices.append(("other (small repos)", len(small),
                       len((small & res[A]) - res[B]), len((small & res[B]) - res[A])))
    slices.sort(key=lambda t: t[3] - t[2])

    row_h, top, left, mid = 30, 40, 150, 380
    w, h = 720, top + row_h * len(slices) + 30
    scale = 7.0
    s = [svg_open(w, h)]
    s.append(f'<line class="axis" x1="{mid}" y1="{top-16}" x2="{mid}" y2="{h-24}"/>')
    s.append(f'<text class="ax" x="{mid-70}" y="{top-22}" text-anchor="middle">lost (base only)</text>')
    s.append(f'<text class="ax" x="{mid+78}" y="{top-22}" text-anchor="middle">gained (arm only)</text>')
    for i, (rp, n, lost, gain) in enumerate(slices):
        y = top + i * row_h
        lw, gw = lost * scale, gain * scale
        s.append(f'<text class="rowlab" x="{left-12}" y="{y+4}" text-anchor="end">{esc(rp)}</text>')
        # 2px surface gap at the midpoint keeps the two fills from touching
        s.append(f'<g>{tip(f"{rp}: {lost} resolved by base only")}'
                 f'<rect class="bar neg" x="{mid-1-lw:.1f}" y="{y-8}" width="{lw:.1f}" height="16" rx="4"/></g>')
        s.append(f'<g>{tip(f"{rp}: {gain} resolved by arm only")}'
                 f'<rect class="bar pos" x="{mid+1}" y="{y-8}" width="{gw:.1f}" height="16" rx="4"/></g>')
        if lost:
            s.append(f'<text class="barlab" x="{mid-6-lw:.1f}" y="{y+4}" text-anchor="end">{lost}</text>')
        if gain:
            s.append(f'<text class="barlab" x="{mid+6+gw:.1f}" y="{y+4}">{gain}</text>')
        net, disc = gain - lost, lost + gain
        if disc == 0:
            verdict = "no discordance"
        elif abs(net) <= max(2, 0.34 * disc):
            verdict = "symmetric churn"
        else:
            verdict = "systematic"
        s.append(f'<text class="rown" x="{w-8}" y="{y+4}" text-anchor="end">{esc(verdict)}</text>')
    s.append("</svg>")
    return "".join(s)


def panel_behaviour(A, B, rows, nameA, nameB):
    """Grouped bars: behavioural rates. One axis (percent of trajectories)."""
    a = [r for r in rows if r["model"] == A and not r.get("error")]
    b = [r for r in rows if r["model"] == B]
    metrics = [
        ("ran any test command", "ran_any_test"),
        ("created a repro / test script", "repro_created"),
        ("re-tested after last edit", "verified_after_edit"),
        ("activated the test env", "env_activated"),
        ("made at least one edit", lambda r: r.get("n_edits", 0) > 0),
        ("ended with finish", "used_finish"),
        ("produced an empty patch", "empty_patch"),
        ("finish claims success", "finish_claims_success"),
        ("claims success, never verified",
         lambda r: r.get("finish_claims_success") and not r.get("verified_after_edit")),
    ]
    def rate(rs, key):
        if callable(key):
            return pct(sum(1 for r in rs if key(r)), len(rs))
        if not any(key in r for r in rs):
            return None
        return pct(sum(1 for r in rs if r.get(key)), len(rs))

    rows_out, row_h, top, left = [], 40, 30, 224
    w = 720
    h = top + row_h * len(metrics) + 24
    barw = w - left - 68
    s = [svg_open(w, h)]
    for gx in (0, 25, 50, 75, 100):
        x = left + barw * gx / 100
        s.append(f'<line class="grid" x1="{x:.1f}" y1="{top-12}" x2="{x:.1f}" y2="{h-20}"/>')
        s.append(f'<text class="ax" x="{x:.1f}" y="{h-6}" text-anchor="middle">{gx}%</text>')
    for i, (label, key) in enumerate(metrics):
        y = top + i * row_h
        ra, rb = rate(a, key), rate(b, key)
        if ra is None or rb is None:
            continue
        s.append(f'<text class="rowlab" x="{left-12}" y="{y+8}" text-anchor="end">{esc(label)}</text>')
        for j, (val, cls, nm) in enumerate(((ra, "a", nameA), (rb, "b", nameB))):
            yy = y - 4 + j * 13
            bw = max(1.5, barw * val / 100)
            s.append(f'<g>{tip(f"{nm}: {val:.1f}%")}'
                     f'<rect class="bar {cls}" x="{left}" y="{yy}" width="{bw:.1f}" height="11" rx="4"/></g>')
            s.append(f'<text class="barlab" x="{left+bw+7:.1f}" y="{yy+9}">{val:.0f}%</text>')
    s.append("</svg>")
    return "".join(s)


def panel_turn_dist(A, B, rows, nameA, nameB):
    """Distribution of turns per trajectory - two overlaid step histograms."""
    a = [r["n_actions"] for r in rows if r["model"] == A and not r.get("error")]
    b = [r["n_actions"] for r in rows if r["model"] == B]
    hi = 500
    nb = 25
    step = hi / nb
    def hist(v):
        h = [0] * nb
        for x in v:
            h[min(nb - 1, int(x / step))] += 1
        return [100.0 * c / len(v) for c in h] if v else h
    ha, hb = hist(a), hist(b)
    top, left, w = 26, 52, 720
    h = 220
    ph = h - top - 34
    ymax = max(max(ha), max(hb)) * 1.12
    def X(i):
        return left + (w - left - 16) * i / nb
    def Y(v):
        return top + ph - (v / ymax) * ph
    s = [svg_open(w, h)]
    for gy in range(0, int(ymax) + 1, 10):
        s.append(f'<line class="grid" x1="{left}" y1="{Y(gy):.1f}" x2="{w-16}" y2="{Y(gy):.1f}"/>')
        s.append(f'<text class="ax" x="{left-8}" y="{Y(gy)+4:.1f}" text-anchor="end">{gy}%</text>')
    for hh, cls in ((ha, "a"), (hb, "b")):
        pts = []
        for i, v in enumerate(hh):
            pts.append(f"{X(i):.1f},{Y(v):.1f}")
            pts.append(f"{X(i+1):.1f},{Y(v):.1f}")
        s.append(f'<polyline class="step {cls}" points="{" ".join(pts)}"/>')
    for i in range(0, nb + 1, 5):
        s.append(f'<text class="ax" x="{X(i):.1f}" y="{h-12}" text-anchor="middle">{int(i*step)}</text>')
    s.append(f'<text class="ax" x="{(left+w)/2:.0f}" y="{h-1}" text-anchor="middle">turns per trajectory (agent steps)</text>')
    # direct labels instead of relying on colour alone
    s.append(f'<text class="dlab a" x="{X(1.4):.0f}" y="{Y(max(hb))-8:.0f}">{esc(nameB)}</text>')
    s.append(f'<text class="dlab b" x="{X(6.2):.0f}" y="{Y(max(ha))-6:.0f}">{esc(nameA)}</text>')
    s.append("</svg>")
    return "".join(s)


def fingerprint(turns, width=560, hgt=15):
    """One trajectory as a tool-sequence strip: the shape of a policy at a glance."""
    if not turns:
        return '<div class="fp-empty">no saved trajectory</div>'
    cmap = {t[0]: t[1] for t in TOOL_COLORS}
    n = len(turns)
    cw = max(0.9, width / n)
    s = [svg_open(width, hgt, "fp")]
    for i, t in enumerate(turns):
        c = cmap.get(t.get("tool"), "#8a8a86")
        err = t.get("obs_err")
        s.append(f'<rect x="{i*cw:.2f}" y="0" width="{max(0.9,cw-0.35):.2f}" height="{hgt}" '
                 f'fill="{c}" opacity="{0.45 if err else 1}">'
                 f'{tip(f"turn {i}: {t.get(chr(116)+chr(111)+chr(111)+chr(108))}" + (" (tool error)" if err else ""))}</rect>')
    s.append("</svg>")
    return "".join(s)


def panel_pairs(pairs, digest_dir, A, B, nameA, nameB, statmap):
    if not pairs or not digest_dir:
        return ""
    import os
    out = []
    for p in pairs:
        iid, tag = p["iid"], p.get("tag", "")
        blocks = []
        for m, nm, cls in ((A, nameA, "a"), (B, nameB, "b")):
            fp = os.path.join(digest_dir, m, f"{iid}.json")
            turns = []
            if os.path.exists(fp):
                try:
                    turns = json.load(open(fp)).get("turns", [])
                except Exception:
                    turns = []
            st_ = statmap.get((m, iid), {})
            badge = "resolved" if st_.get("resolved") else "failed"
            blocks.append(f'''<div class="fp-row">
              <div class="fp-meta"><span class="dot {cls}"></span>{esc(nm)}
                <span class="badge {badge}">{badge}</span></div>
              <div class="fp-strip">{fingerprint(turns)}</div>
              <div class="fp-num">{len(turns)} turns &middot; {st_.get("n_test_runs", 0)} test runs</div>
            </div>''')
        out.append(f'<div class="pair"><div class="pair-h"><code>{esc(iid)}</code>'
                   f'<span class="tag">{esc(tag)}</span></div>{"".join(blocks)}</div>')
    return "".join(out)


# ---------------------------------------------------------------- page
CSS = """
:root{color-scheme:light dark}
.viz-root{
  --surface-1:#fcfcfb; --surface-2:#f4f3f0; --border:#e0dfda;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#77766f;
  --s1:%(s1_light)s; --s2:%(s2_light)s; --neg:%(neg_light)s; --pos:%(pos_light)s;
  --grid:#e6e5e0;
  background:var(--surface-1); color:var(--text-primary);
  font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
  padding:30px 34px; max-width:1080px; margin:0 auto;
}
@media (prefers-color-scheme:dark){
 :root:where(:not([data-theme="light"])) .viz-root{
  --surface-1:#1a1a19; --surface-2:#232322; --border:#3a3a37;
  --text-primary:#fff; --text-secondary:#c3c2b7; --text-muted:#96958c;
  --s1:%(s1_dark)s; --s2:%(s2_dark)s; --neg:%(neg_dark)s; --pos:%(pos_dark)s;
  --grid:#333330;
 }}
:root[data-theme="dark"] .viz-root{
  --surface-1:#1a1a19; --surface-2:#232322; --border:#3a3a37;
  --text-primary:#fff; --text-secondary:#c3c2b7; --text-muted:#96958c;
  --s1:%(s1_dark)s; --s2:%(s2_dark)s; --neg:%(neg_dark)s; --pos:%(pos_dark)s;
  --grid:#333330;
}
h1{font-size:21px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:15px;margin:34px 0 4px;letter-spacing:-.005em}
.sub{color:var(--text-secondary);font-size:13px;margin:0 0 6px}
.note{color:var(--text-muted);font-size:12px;margin:6px 0 0}
section{border-top:1px solid var(--border);padding-top:6px}
.legend{display:flex;gap:16px;align-items:center;margin:8px 0 2px;font-size:12.5px;color:var(--text-secondary)}
.dot{width:9px;height:9px;border-radius:50%%;display:inline-block;margin-right:6px}
.dot.a{background:var(--s1)} .dot.b{background:var(--s2)}
.dot.neg{background:var(--neg)} .dot.pos{background:var(--pos)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:14px 0 4px}
.tile{background:var(--surface-2);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.tile-label{font-size:11.5px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.04em}
.tile-pair{display:flex;align-items:baseline;gap:8px;margin-top:6px}
.tv{font-size:25px;font-weight:600;font-variant-numeric:tabular-nums}
.tv.a{color:var(--s1)} .tv.b{color:var(--s2)}
.arrow{color:var(--text-muted);font-size:15px}
.tile-delta{font-size:13px;font-weight:600;font-variant-numeric:tabular-nums;color:var(--text-secondary)}
.tile-delta.down{color:var(--neg)} .tile-delta.up{color:var(--pos)}
.tile-sub{font-size:11.5px;color:var(--text-muted);margin-top:2px}
svg.plot{display:block;margin:6px 0 0;overflow:visible}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--border);stroke-width:1}
text{fill:var(--text-secondary);font-size:11.5px;font-family:inherit}
text.rowlab{fill:var(--text-primary);font-size:12.5px}
text.rown{fill:var(--text-muted);font-size:10.5px}
text.ax{fill:var(--text-muted);font-size:10.5px}
text.barlab{fill:var(--text-secondary);font-size:11px;font-variant-numeric:tabular-nums}
text.dlab{font-size:12px;font-weight:600}
text.dlab.a{fill:var(--s2)} text.dlab.b{fill:var(--s1)}
.delta-neg{fill:var(--neg);font-weight:600;font-variant-numeric:tabular-nums}
.delta-pos{fill:var(--pos);font-weight:600;font-variant-numeric:tabular-nums}
.delta-zero{fill:var(--text-muted);font-variant-numeric:tabular-nums}
.conn{stroke:var(--border);stroke-width:2}
circle.pt{stroke:var(--surface-1);stroke-width:2}
circle.pt.a{fill:var(--s1)} circle.pt.b{fill:var(--s2)}
rect.bar.a{fill:var(--s1)} rect.bar.b{fill:var(--s2)}
rect.bar.neg{fill:var(--neg)} rect.bar.pos{fill:var(--pos)}
polyline.step{fill:none;stroke-width:2;stroke-linejoin:round}
polyline.step.a{stroke:var(--s1)} polyline.step.b{stroke:var(--s2)}
.pair{border:1px solid var(--border);border-radius:10px;padding:10px 12px;margin:10px 0;background:var(--surface-2)}
.pair-h{display:flex;gap:10px;align-items:center;margin-bottom:6px}
.pair-h code{font-size:12.5px;color:var(--text-primary)}
.tag{font-size:10.5px;color:var(--text-muted);border:1px solid var(--border);border-radius:5px;padding:1px 6px}
.fp-row{display:grid;grid-template-columns:190px 1fr 128px;align-items:center;gap:10px;margin:3px 0}
.fp-meta{font-size:12px;color:var(--text-secondary);display:flex;align-items:center;gap:2px}
.fp-num{font-size:11px;color:var(--text-muted);text-align:right;font-variant-numeric:tabular-nums}
.badge{font-size:10px;border-radius:4px;padding:1px 5px;margin-left:6px;border:1px solid var(--border)}
.badge.resolved{color:var(--pos)} .badge.failed{color:var(--neg)}
.fp-empty{font-size:11px;color:var(--text-muted)}
table{border-collapse:collapse;font-size:12.5px;margin-top:8px;width:100%%}
th,td{text-align:right;padding:4px 9px;border-bottom:1px solid var(--border);font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}
th{color:var(--text-secondary);font-weight:600}
details{margin-top:10px} summary{cursor:pointer;font-size:12.5px;color:var(--text-secondary)}
"""


def build(args):
    rows = [json.loads(l) for l in open(args.stats)]
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    A, B = (args.a or models[0]), (args.b or models[1])
    nameA, nameB = args.name_a or A, args.name_b or B
    statmap = {(r["model"], r["instance_id"]): r for r in rows}
    pairs = json.load(open(args.pairs)) if args.pairs else []

    a_ok = [r for r in rows if r["model"] == A and not r.get("error")]
    n_a_err = sum(1 for r in rows if r["model"] == A and r.get("error"))

    # table view (accessibility: identity never by colour alone)
    tbl_fields = [("turns", "n_actions"), ("terminal calls", "n_terminal"),
                  ("file_editor calls", "n_file_editor"), ("think calls", "n_think"),
                  ("test runs", "n_test_runs"), ("context condensations", "n_condensation"),
                  ("completion tokens", "completion_tokens_total"),
                  ("reasoning chars (think)", "think_arg_chars_total"),
                  ("patch lines added", "patch_added"), ("wall seconds", "duration_s")]
    b_all = [r for r in rows if r["model"] == B]
    trs = []
    for lab, f in tbl_fields:
        trs.append(f"<tr><td>{esc(lab)}</td><td>{mean([r.get(f) for r in a_ok]):,.1f}</td>"
                   f"<td>{median([r.get(f) for r in a_ok]):,.0f}</td>"
                   f"<td>{mean([r.get(f) for r in b_all]):,.1f}</td>"
                   f"<td>{median([r.get(f) for r in b_all]):,.0f}</td></tr>")

    legend = (f'<div class="legend"><span><span class="dot a"></span>{esc(nameA)}</span>'
              f'<span><span class="dot b"></span>{esc(nameB)}</span></div>')

    body = f"""<div class="viz-root">
<h1>{esc(nameA)} vs {esc(nameB)} &mdash; SWE-bench Verified trajectories</h1>
<p class="sub">Paired comparison over the same 500 instances, same harness. {esc(nameA)} statistics are
computed over its {len(a_ok)} trajectories with saved history ({n_a_err} rollouts errored and stored no history).</p>
{panel_hero(A, B, rows)}

<section><h2>Where the loss happens</h2>
<p class="sub">Resolve rate per repository, paired. Repos with n&nbsp;&ge;&nbsp;5.</p>
{legend}{panel_repo_dumbbell(A, B, rows, nameA, nameB)}</section>

<section><h2>Discordant instances: churn vs systematic loss</h2>
<p class="sub">Instances only one model solved. Symmetric bars = reshuffling within a preserved
capability; one-sided bars = capability destroyed.</p>
<div class="legend"><span><span class="dot neg"></span>lost (only {esc(nameA)} solved)</span>
<span><span class="dot pos"></span>gained (only {esc(nameB)} solved)</span></div>
{panel_discordance(A, B, rows)}</section>

<section><h2>What the agents actually do</h2>
<p class="sub">Percent of trajectories exhibiting each behaviour.</p>
{legend}{panel_behaviour(A, B, rows, nameA, nameB)}</section>

<section><h2>Turn budget</h2>
<p class="sub">Share of trajectories by number of agent steps.</p>
{legend}{panel_turn_dist(A, B, rows, nameA, nameB)}</section>

<section><h2>Trajectory fingerprints</h2>
<p class="sub">Each strip is one run, left to right, one cell per turn, coloured by tool.
Faded cells are tool errors.</p>
<div class="legend">{"".join(f'<span><span class="dot" style="background:{c}"></span>{esc(t)}</span>' for t, c, _ in TOOL_COLORS)}</div>
{panel_pairs(pairs, args.digest, A, B, nameA, nameB, statmap)}</section>

<section><h2>Table view</h2>
<table><thead><tr><th>metric</th><th>{esc(nameA)} mean</th><th>{esc(nameA)} median</th>
<th>{esc(nameB)} mean</th><th>{esc(nameB)} median</th></tr></thead>
<tbody>{"".join(trs)}</tbody></table>
<p class="note">Generated by build_dashboard.py from stats.jsonl. Hover any mark for exact values.</p>
</section>
</div>"""

    page = (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{esc(nameA)} vs {esc(nameB)} — trajectory comparison</title>"
            f"<style>{CSS % PAL}</style></head><body>{body}</body></html>")
    with open(args.out, "w") as f:
        f.write(page)
    print(f"wrote {args.out}  ({len(page):,} bytes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--digest")
    ap.add_argument("--pairs")
    ap.add_argument("--a"), ap.add_argument("--b")
    ap.add_argument("--name-a"), ap.add_argument("--name-b")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
