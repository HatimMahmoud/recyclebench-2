#!/usr/bin/env python3
"""Embed the RPE PROCESS LOG (section 01.e) into make_dashboard.py, filter-aware.
Injects a CSS-scoped, rl-namespaced fragment inside each per-benchmark .bview, so it
cannot collide with the dashboard's own styles. Requires the RPE section patch (01.c/d)
to have run first (it reuses RPE_OUT_DIR + old/new in build_view). Idempotent.
Run from repo root:  python3 tools/patch_dashboard_add_rpelog.py
"""
from pathlib import Path
import sys
p = Path("tools/make_dashboard.py")
if not p.exists():
    sys.exit("run from repo root; tools/make_dashboard.py not found")
s = p.read_text()
if "_rpe_log" in s:
    sys.exit("already patched (found _rpe_log).")
if "RPE_OUT_DIR" not in s:
    sys.exit("run the RPE section patch first (RPE_OUT_DIR not found in build_view).")

# 1) placeholder in the view template, right after the cards placeholder
if "{_rpe_cards}\n" not in s:
    sys.exit("{_rpe_cards} placeholder not found - is the section/cards patch applied?")
s = s.replace("{_rpe_cards}\n", "{_rpe_cards}\n\n{_rpe_log}\n", 1)

# 2) compute _rpe_log in build_view, next to the section/cards try-block.
#    Hook onto the existing 'from rpe_section import ...' line we added earlier.
hook = "        from rpe_section import render_rpe_section, render_rpe_cards\n"
if hook not in s:
    # tolerant: match any 'from rpe_section import' line
    import re
    m = re.search(r"^\s*from rpe_section import .*\n", s, re.M)
    if not m:
        sys.exit("rpe_section import hook not found in build_view.")
    hook = m.group(0)
inject = hook + (
    "        import json as _json, importlib.util as _il\n"
    "        _rlp = __import__('pathlib').Path(__file__).resolve().parents[0] / 'rpe_log.py'\n"
    "        _sp = _il.spec_from_file_location('rpe_log', _rlp)\n"
    "        _rlmod = _il.module_from_spec(_sp); _sp.loader.exec_module(_rlmod)\n"
    "        _mf = (__import__('pathlib').Path(globals().get('RPE_OUT_DIR')) / f'rpe_manifest_{view_id}.jsonl') if globals().get('RPE_OUT_DIR') else None\n"
    "        if _mf and _mf.exists():\n"
    "            _rows = _rlmod.load_rows(_mf)\n"
    "            _thr = getattr(getattr(globals().get('CFG_REF', None), 'gate', None), '__dict__', {}) or {}\n"
    "            _thr = {k: getattr(globals()['CFG_REF'].gate, k) for k in ('cosine','bertscore','nli_entail','llm_intent')} if globals().get('CFG_REF') else {'cosine':0.75,'bertscore':0.85,'nli_entail':0.60,'llm_intent':0.50}\n"
    "            _rpe_log = _rlmod.render_fragment(_rows, _thr, old, new, view_id, section_no='01\\u00b7e')\n"
    "        else:\n"
    "            _rpe_log = ''\n"
)
s = s.replace(hook, inject, 1)

# 3) also default _rpe_log='' in the except branch (so it's always defined)
s = s.replace("        _rpe_section = ''\n        _rpe_cards = ''\n",
              "        _rpe_section = ''\n        _rpe_cards = ''\n        _rpe_log = ''\n", 1)

# 4) stash cfg so build_view can read real gate thresholds: set CFG_REF in main after cfg load
s = s.replace("    global RPE_OUT_DIR\n    RPE_OUT_DIR = args.rpe_out\n",
              "    global RPE_OUT_DIR, CFG_REF\n    RPE_OUT_DIR = args.rpe_out\n", 1)
s = s.replace("    cfg = Config.load(args.config)",
              "    cfg = Config.load(args.config)\n    CFG_REF = cfg", 1)
# declare CFG_REF global near the top
s = s.replace("RPE_OUT_DIR = None  # set by main() --rpe-out; read by build_view\n",
              "RPE_OUT_DIR = None  # set by main() --rpe-out; read by build_view\nCFG_REF = None\n", 1)

p.write_text(s)
print("patched: RPE process log (01.e) embedded, CSS-scoped, filter-aware")
