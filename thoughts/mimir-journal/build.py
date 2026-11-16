#!/usr/bin/env python3
"""Build the self-contained MIMIR research-journal HTML from journal_data.json."""
import json, re, html, os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "journal_data.json"), encoding="utf-8"))
CSS = open(os.path.join(HERE, "styles.css"), encoding="utf-8").read()
JS = open(os.path.join(HERE, "app.js"), encoding="utf-8").read()
OUT = os.path.join(HERE, "mimir-journal.html")

def esc(s, quote=False):
    return html.escape("" if s is None else str(s), quote=quote)

# ---------- mini markdown ----------
def inline_md(t):
    t = esc(t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<!\w)\*([^*\n]+?)\*(?!\w)", r"<em>\1</em>", t)
    t = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"<em>\1</em>", t)
    return t

def md(text, dropcap=False):
    """Line-scanning markdown: headings (#,##,###), bullet lists (-,*), paragraphs.
    Handles a heading immediately followed by body text or a list (no blank line)."""
    if not text:
        return ""
    out, para, items, first = [], [], [], [True]
    def flush_para():
        if para:
            cls = ' class="dropcap"' if (dropcap and first[0]) else ""
            out.append("<p%s>%s</p>" % (cls, inline_md(" ".join(para))))
            first[0] = False
            para.clear()
    def flush_items():
        if items:
            out.append("<ul>" + "".join("<li>" + inline_md(x) + "</li>" for x in items) + "</ul>")
            items.clear()
    for raw in text.replace("\r", "").split("\n"):
        s = raw.strip()
        if s == "":
            flush_para(); flush_items(); continue
        mh = re.match(r"(#{1,3})\s+(.*)", s)
        if mh:
            flush_para(); flush_items()
            lvl = len(mh.group(1))
            out.append("<h%d>%s</h%d>" % (lvl, inline_md(mh.group(2)), lvl))
            continue
        mi = re.match(r"[-*]\s+(.*)", s)
        if mi:
            flush_para(); items.append(mi.group(1)); continue
        flush_items(); para.append(s)
    flush_para(); flush_items()
    return "\n".join(out)

# ---------- family bucket classifier ----------
BUCKETS = [
    ("agentic",     "Agentic / ReAct"),
    ("engram",      "Engram / Foundry"),
    ("kg",          "Knowledge-Graph / Structured"),
    ("extraction",  "Extraction & Ontology"),
    ("temporal",    "Temporal Reasoning"),
    ("episodic",    "Episodic / Event Binding"),
    ("dual",        "Dual Extraction & Observations"),
    ("interpretive","Interpretive · Oracle / Salience / Semiotic"),
    ("parametric",  "Continual Learning & Parametric"),
    ("worldmodel",  "World Models (JEPA)"),
    ("formal",      "Formal Proofs (Lean / Energy)"),
    ("method",      "Routing & Evaluation"),
]
BUCKET_NAME = dict(BUCKETS)
RULES = [
    ("formal",      ["lean", "energy", "hopfield", "nlcdm", "lyapunov", "banach", "sorry", "content-address", " proof"]),
    ("temporal",    ["temporal", "allen", "chronos", "chat time", "calendar", "interval"]),
    ("worldmodel",  ["jepa", "world model", "vicreg", "contrastive", "self-supervised", "architecture i", "sbert"]),
    ("parametric",  ["continual", "lora", "parametric", "fine-tune", "fine tune", "test-time", "ttt",
                     "titans", "mega", "weight", "knowledge edit", "representation engineering",
                     "neural memory", "residual"]),
    ("episodic",    ["episodic", "event bind", "event-bind", "eventframe", "event frame", "binding"]),
    ("dual",        ["dual extraction", "observation"]),
    ("interpretive",["oracle", "salience", "semiotic"]),
    ("engram",      ["engram", "foundry", "recall"]),
    ("kg",          ["knowledge graph", "knowledge-graph", "structured", "entity", "hyperedge", "subject index", " kg"]),
    ("extraction",  ["extraction", "ontology", "faceted", "adelie", "triple", "ingestion", "chunk", "msl"]),
    ("agentic",     ["agentic", "react", "mabench", "fc", "distillation"]),
]
def bucket_of(fam, name):
    t = (" " + (fam or "") + " " + (name or "") + " ").lower()
    for key, kws in RULES:
        for kw in kws:
            if kw in t:
                return key
    return "method"

# ---------- score parsing ----------
def parse_pct(score):
    if not score:
        return None
    s = str(score)
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if b:
            return a / b * 100.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    if m:
        return float(m.group(1))
    m = re.search(r"\b(\d+(?:\.\d+)?)\b", s)
    if m:
        v = float(m.group(1))
        if 0 <= v <= 100:
            return v
    return None

def is_mimir(sys, note):
    # MIMIR's own runs are explicitly prefixed "MIMIR " in the data; externals are named directly.
    return (sys or "").strip().lower().startswith("mimir")

def mimir_disp(sys):
    return re.sub(r"^\s*MIMIR\s+", "", sys or "")

# =====================================================================
# RENDER SECTIONS
# =====================================================================
st = DATA.get("stats", {})
parts = []

# ---- toprail ----
NAV = [("intro", "Overview"), ("timeline", "Timeline"), ("chapters", "Chapters"),
       ("archive", "Approaches"), ("leaderboard", "Leaderboard"),
       ("graveyard", "Dead Ends"), ("proofs", "Proofs"), ("reflection", "Reflection")]
nav_links = "".join('<a href="#%s">%s</a>' % (i, esc(l)) for i, l in NAV)
parts.append('<div class="progress"></div>')
parts.append('<div class="toprail"><div class="inner">'
             '<div class="brand">MIMIR<b>.</b></div>'
             '<nav>%s</nav></div></div>' % nav_links)

# ---- HERO ----
def big(n):  # split trailing % / unit for accent styling
    return esc(n)
stats_cells = [
    ("347", "Session logs mined"),
    ("94", "Approaches tried"),
    ("11", "Strategy families"),
    ("335", "Lean theorems · 0 sorry"),
    ("60.3<span class=\"accent\">%</span>", "Best honest SubEM · #1"),
    ("38", "Documented dead ends"),
]
statband = "".join(
    '<div class="st"><div class="n">%s</div><div class="l">%s</div></div>' % (n, esc(l))
    for n, l in stats_cells)
parts.append("""
<header class="hero"><div class="wrap">
  <div class="hero-grid">
    <div>
      <div class="kicker">A research journal · formal memory systems</div>
      <h1>Mi<span class="em">mir</span></h1>
      <p class="sub">%s</p>
      <div class="meta">
        <span class="dt">%s</span>
        <span class="dt">347 sessions</span>
        <span class="dt">~200 experiments</span>
        <span class="dt">2 disciplines: Lean&nbsp;4 · LongMemEval</span>
      </div>
    </div>
    <div class="statband reveal">%s</div>
  </div>
</div></header>
""" % (esc(DATA.get("subtitle", "")), esc(st.get("date_range", "")), statband))

# ---- INTRO + families ----
fam_rows = ""
for f in DATA.get("families", []):
    fam_rows += ('<div class="era-ev" style="grid-template-columns:1fr">'
                 '<div><b>%s.</b> <span>%s</span>%s</div></div>') % (
        esc(f.get("name", "")), esc(f.get("description", "")),
        (' <span style="color:var(--muted)">— %s</span>' % esc(f.get("arc", ""))) if f.get("arc") else "")
fam_block = ('<div class="era-events" style="margin-top:40px">'
             '<div class="eyebrow" style="margin-bottom:18px">The eleven families</div>%s</div>') % fam_rows
parts.append("""
<section id="intro"><div class="wrap">
  <div class="sec-head reveal">
    <div class="sec-num">00</div>
    <div><div class="eyebrow">Overview</div>
      <h2 class="sec-title">Can a system learn, or only retrieve?</h2></div>
  </div>
  <div class="prose lead reveal">%s</div>
  <div class="reveal">%s</div>
</div></section>
""" % (md(DATA.get("intro_md", ""), dropcap=True), fam_block))

# ---- TIMELINE ----
tl = ""
for ev in DATA.get("timeline", []):
    v = (ev.get("verdict") or "neutral").lower()
    if v not in ("worked","partial","failed","dead-end","neutral","in-progress"):
        v = "neutral"
    fam = ev.get("family")
    tl += ('<div class="tl-item reveal"><div class="tl-date">%s</div>'
           '<div class="tl-body"><span class="tl-dot" data-v="%s"></span>'
           '<h3 class="tl-title">%s</h3>%s'
           '<p class="tl-desc">%s</p></div></div>') % (
        esc(ev.get("date","")), v, esc(ev.get("title","")),
        ('<span class="tl-fam">%s</span>' % esc(fam)) if fam else "",
        esc(ev.get("desc","")))
parts.append("""
<section id="timeline"><div class="wrap">
  <div class="sec-head reveal"><div class="sec-num">01</div>
    <div><div class="eyebrow">The arc · Feb–May 2026</div>
      <h2 class="sec-title">Three months, in order</h2>
      <p class="sec-lede">Sixty-one moments that mattered — the wins, the regressions, and the deletions that kept the project honest.</p></div></div>
  <div class="tl">%s</div>
</div></section>
""" % tl)

# ---- ERA CHAPTERS ----
eras_html = ""
for i, e in enumerate(DATA.get("eras", []), 1):
    ev = ""
    for k in (e.get("key_events") or [])[:8]:
        vv = (k.get("verdict") or "neutral").lower()
        if vv not in ("worked","partial","failed","dead-end","neutral","in-progress"):
            vv = "neutral"
        ev += ('<div class="era-ev"><span class="vd dot" data-v="%s"></span>'
               '<div><b>%s.</b> <span>%s</span></div></div>') % (
            vv, esc(k.get("title","")), esc(k.get("desc","")))
    events_block = ('<div class="era-events">%s</div>' % ev) if ev else ""
    eras_html += """
    <article class="era reveal">
      <aside class="era-aside"><div class="era-no">%02d</div><div class="era-range">%s</div></aside>
      <div><h3 class="era-title">%s</h3><div class="prose">%s</div>%s</div>
    </article>""" % (i, esc(e.get("range","")), esc(e.get("title","")),
                     md(e.get("narrative_md","")), events_block)
parts.append("""
<section id="chapters"><div class="wrap">
  <div class="sec-head reveal"><div class="sec-num">02</div>
    <div><div class="eyebrow">Five chapters</div>
      <h2 class="sec-title">The story, told in eras</h2></div></div>
  %s
</div></section>
""" % eras_html)

# ---- ARCHIVE (approaches) ----
cards = DATA.get("approach_cards", [])
bucket_counts = {}
verdict_counts = {}
cards_html = ""
for a in cards:
    v = (a.get("verdict") or "neutral").lower()
    if v not in ("worked","partial","failed","dead-end","neutral","in-progress"):
        v = "neutral"
    bk = bucket_of(a.get("family",""), a.get("name",""))
    bucket_counts[bk] = bucket_counts.get(bk, 0) + 1
    verdict_counts[v] = verdict_counts.get(v, 0) + 1
    search_blob = " ".join(str(a.get(k,"")) for k in ("name","family","hypothesis","method","result"))
    nums = "".join("<span>%s</span>" % esc(n) for n in (a.get("key_numbers") or [])[:6] if n)
    rows = ""
    if a.get("hypothesis"):
        rows += '<p class="card-row"><span class="k">Bet</span>%s</p>' % esc(a["hypothesis"])
    if a.get("method"):
        rows += '<p class="card-row"><span class="k">Method</span>%s</p>' % esc(a["method"])
    result = ""
    if a.get("result"):
        result = '<div class="card-result"><span class="k">Result</span>%s%s</div>' % (
            esc(a["result"]), ('<div class="card-nums">%s</div>' % nums) if nums else "")
    cards_html += ('<article class="card reveal" data-bucket="%s" data-v="%s" data-search="%s">'
                   '<div class="card-top"><div class="card-fam">%s</div>'
                   '<span class="vd" data-v="%s">%s</span></div>'
                   '<h3 class="card-name">%s</h3>%s%s</article>') % (
        bk, v, esc(search_blob.lower(), quote=True),
        esc(a.get("family","")), v, esc(v),
        esc(a.get("name","")), rows, result)

fam_chips = '<button class="chip fchip on" data-fam="all">All<span class="ct">%d</span></button>' % len(cards)
for key, name in BUCKETS:
    c = bucket_counts.get(key, 0)
    if c:
        fam_chips += '<button class="chip fchip" data-fam="%s">%s<span class="ct">%d</span></button>' % (key, esc(name), c)
VORDER = [("worked","Worked"),("partial","Partial"),("failed","Failed"),
          ("dead-end","Dead end"),("neutral","Neutral"),("in-progress","In&nbsp;progress")]
ver_chips = ""
for key, name in VORDER:
    c = verdict_counts.get(key, 0)
    if c:
        ver_chips += '<button class="chip vchip" data-ver="%s" data-v="%s">%s<span class="ct">%d</span></button>' % (key, key, name, c)

parts.append("""
<section id="archive"><div class="wrap">
  <div class="sec-head reveal"><div class="sec-num">03</div>
    <div><div class="eyebrow">The catalogue</div>
      <h2 class="sec-title">Ninety-four things we tried</h2>
      <p class="sec-lede">Every distinct strategy variation, with the bet it made and the number it came back with. Filter by family or by how it ended.</p></div></div>
  <div class="filters reveal">
    <div class="grp"><span class="glabel">Family</span>%s</div>
  </div>
  <div class="filters reveal" style="margin-top:10px">
    <div class="grp"><span class="glabel">Verdict</span>%s</div>
    <input id="archiveSearch" class="chip" style="border-radius:100px;min-width:160px;font-family:var(--mono)" placeholder="search…" aria-label="Search approaches">
  </div>
  <div class="archive-meta reveal">Showing <b id="archiveCount">%d</b> of %d approaches</div>
  <div class="cards">%s</div>
  <div class="empty">No approaches match these filters.</div>
</div></section>
""" % (fam_chips, ver_chips, len(cards), len(cards), cards_html))

# ---- LEADERBOARD ----
lb = DATA.get("leaderboard", [])
rows = []
for r in lb:
    pct = parse_pct(r.get("score"))
    rows.append((r, pct, is_mimir(r.get("system"), r.get("note"))))
rows.sort(key=lambda x: (-(x[1] if x[1] is not None else -1)))
lb_html = ""
for r, pct, mim in rows:
    bar = ('<div class="bar"><i style="width:%.1f%%"></i></div>' % max(0, min(100, pct))) if pct is not None else "—"
    tag = '<span class="tag">Mimir</span>' if mim else ""
    sys_name = mimir_disp(r.get("system","")) if mim else r.get("system","")
    lb_html += ('<tr class="%s"><td><span class="sys">%s%s</span></td>'
                '<td class="ctx hide-sm">%s</td><td class="hide-sm">%s</td>'
                '<td class="score">%s</td>'
                '<td class="note hide-sm">%s</td></tr>') % (
        "mimir" if mim else "", esc(sys_name), tag,
        esc(r.get("context","")), bar, esc(r.get("score","")), esc(r.get("note","")))
parts.append("""
<section id="leaderboard"><div class="wrap">
  <div class="sec-head reveal"><div class="sec-num">04</div>
    <div><div class="eyebrow">Where it stood</div>
      <h2 class="sec-title">The leaderboard</h2>
      <p class="sec-lede">MIMIR's own runs (highlighted) against every named system it was measured beside, on LongMemEval / MemoryAgentBench under SubEM. Ranked by score; context disambiguates the conditions.</p></div></div>
  <div class="lb-wrap reveal"><table class="lb">
    <thead><tr><th>System</th><th class="hide-sm">Context</th><th class="hide-sm"></th><th class="num">Score</th><th class="hide-sm">Note</th></tr></thead>
    <tbody>%s</tbody>
  </table></div>
</div></section>
""" % lb_html)

# ---- GRAVEYARD ----
gv = ""
for d in DATA.get("dead_ends", []):
    gv += ('<div class="grave reveal"><div class="x">&#8224;</div>'
           '<div><b>%s</b><span>%s</span></div></div>') % (
        esc(d.get("name","")), esc(d.get("why","")))
parts.append("""
<section id="graveyard"><div class="wrap">
  <div class="sec-head reveal"><div class="sec-num">05</div>
    <div><div class="eyebrow">The graveyard</div>
      <h2 class="sec-title">Thirty-eight dead ends</h2>
      <p class="sec-lede">The failures taught more than the wins. Each headstone is an idea that was tested honestly and laid to rest — with the reason it died.</p></div></div>
  <div class="graves">%s</div>
</div></section>
""" % gv)

# ---- PROOFS ----
pf = ""
for p in DATA.get("formal_proofs", []):
    pf += ('<div class="proof reveal"><h3>%s</h3>'
           '<div class="stat">%s</div><p>%s</p></div>') % (
        esc(p.get("project","")), esc(p.get("theorems","")), esc(p.get("result","")))
parts.append("""
<section id="proofs"><div class="wrap">
  <div class="sec-head reveal"><div class="sec-num">06</div>
    <div><div class="eyebrow">The formal half</div>
      <h2 class="sec-title">Proven in Lean, tested in the wild</h2>
      <p class="sec-lede">Two sorry-free Lean&nbsp;4 libraries established what a memory system should obey. The mathematics was perfect; reality was less impressed.</p></div></div>
  <div class="proofs">%s</div>
</div></section>
""" % pf)

# ---- REFLECTION ----
parts.append("""
<section id="reflection"><div class="wrap">
  <div class="sec-head reveal"><div class="sec-num">07</div>
    <div><div class="eyebrow">Closing the journal</div>
      <h2 class="sec-title">What won, and what comes next</h2></div></div>
  <div class="pull reveal">Retrieval, done carefully, beat everything we built on top of it.</div>
  <div class="prose reveal">%s</div>
</div></section>
""" % md(DATA.get("reflection_md", "")))

# ---- FOOTER ----
parts.append("""
<footer><div class="wrap"><div class="f-grid">
  <div class="big">Mi<b>mir</b> — a search for memory that learns.</div>
  <div class="mono">%s · 347 sessions · 94 approaches · 335 theorems · #1 honest SubEM 60.33%%</div>
</div>
<div class="mono" style="margin-top:18px;color:var(--faint)">Compiled from session logs, memory notes, handoffs &amp; Lean proofs. Benchmarks were the flashlight, never the room.</div>
</div></footer>
""" % esc(st.get("date_range","")))

# =====================================================================
GOOGLE = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
          '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
          '<link href="https://fonts.googleapis.com/css2?'
          'family=Fraunces:ital,opsz,wght@0,9..144,400..700;1,9..144,400..600&'
          'family=Newsreader:ital,opsz,wght@0,6..72,400..600;1,6..72,400..500&'
          'family=Space+Grotesk:wght@400..600&'
          'family=JetBrains+Mono:wght@400..600&display=swap" rel="stylesheet">')

DOC = ("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
       "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
       "<title>Mimir — A Research Journal of AI Memory Systems</title>"
       "<meta name=\"description\" content=\"" + esc(DATA.get("subtitle",""), quote=True) + "\">"
       + GOOGLE + "<style>" + CSS + "</style>"
       "<script>document.documentElement.classList.add('js')</script>"
       "</head><body>"
       "<div class=\"grain\"></div>"
       + "\n".join(parts)
       + "<script>" + JS + "</script></body></html>")

with open(OUT, "w", encoding="utf-8") as f:
    f.write(DOC)

# report
from collections import Counter
print("WROTE", OUT, "(%d KB)" % (len(DOC)//1024))
print("cards:", len(cards), "| timeline:", len(DATA.get('timeline',[])),
      "| leaderboard:", len(lb), "| dead_ends:", len(DATA.get('dead_ends',[])),
      "| eras:", len(DATA.get('eras',[])))
print("buckets:", dict(sorted(bucket_counts.items(), key=lambda x:-x[1])))
print("mimir rows:", sum(1 for _,_,m in rows if m), "/", len(rows))
