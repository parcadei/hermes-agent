#!/usr/bin/env python3
"""Bounded high-signal digest of a Claude Code session JSONL.
Usage: python extract_session.py <session.jsonl> [max_chars]
Emits: dates, user prompts (prose), assistant prose (between tool calls),
and result-bearing lines (SubEM, %, accuracy, works/dead-end/FAILED, experiment ids).
Caps total output so an agent can read many sessions cheaply."""
import json, sys, re

fp = sys.argv[1]
MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 9000

RESULT_RE = re.compile(r"(SubEM|sub_em|accuracy|recall@|MRR|F1|\bAUC\b|\bctx[0-4]\b|"
    r"\bexp\d+\b|\bE\d{2}\b|\bh\d{1,2}\b|dead.?end|DEAD|FAILED|regress|baseline|"
    r"\d{2,3}\s*/\s*\d{2,3}|\d{1,3}(?:\.\d+)?\s*%|WORKS|VALIDATED|NEGATIVE|HURTS|"
    r"leaderboard|theorem|sorries|Lyapunov|JEPA|semiotic|NLCDM|Hopfield|Titans|TTT)", re.I)

def text_of(msg):
    if isinstance(msg, str): return msg
    if isinstance(msg, dict):
        c = msg.get("content")
        if isinstance(c, str): return c
        if isinstance(c, list):
            out=[]
            for p in c:
                if isinstance(p, dict):
                    if p.get("type")=="text": out.append(p.get("text",""))
                    elif p.get("type")=="thinking": out.append("[THINK] "+p.get("thinking",""))
                elif isinstance(p,str): out.append(p)
            return "\n".join(out)
    return ""

users=[]; assts=[]; results=set(); start=end=""
for line in open(fp, errors="ignore"):
    try: d=json.loads(line)
    except: continue
    ts=d.get("timestamp","")
    if ts:
        if not start: start=ts
        end=ts
    t=d.get("type")
    if t=="user":
        txt=text_of(d.get("message",{})).strip()
        if txt and not txt.startswith("<") and "tool_result" not in txt[:40] and "command-name" not in txt:
            users.append(txt[:600])
    elif t=="assistant":
        txt=text_of(d.get("message",{})).strip()
        if not txt: continue
        # keep prose (skip pure tool-call turns which have empty text)
        for ln in txt.splitlines():
            if RESULT_RE.search(ln):
                results.add(ln.strip()[:240])
        if len(txt)>40:
            assts.append(txt[:500])

print(f"### SESSION {fp.split('/')[-1]}")
print(f"DATES {start[:19]} -> {end[:19]} | {len(users)} user msgs, {len(assts)} asst msgs")
print("\n-- USER PROMPTS (intent) --")
# first 6 + last 3 user prompts
sel = users[:6] + (["..."]+users[-3:] if len(users)>9 else users[6:])
for u in sel: print("•", u.replace("\n"," ")[:300])
print("\n-- RESULT / EXPERIMENT MARKERS --")
for r in sorted(results)[:40]: print("◦", r)
print("\n-- ASSISTANT PROSE (sampled) --")
sela = assts[:4] + (assts[-4:] if len(assts)>8 else [])
buf=""
for a in sela:
    buf += "» "+a.replace("\n"," ")[:280]+"\n"
print(buf[:MAX//2])
