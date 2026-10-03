#!/usr/bin/env python3
"""Patch tools/make_dashboard.py to add the RPE convergent-validity section (01.c)
AND the per-behaviour RPE detail cards (01.d), both filter-aware (rendered inside
each per-benchmark .bview). Reads rpe_agreement.json + rpe_manifest_<bench>.jsonl
from --rpe-out. Idempotent: refuses to double-apply. Run from repo root:
    python3 tools/patch_dashboard_add_rpe.py
"""
from pathlib import Path
import sys

p = Path("tools/make_dashboard.py")
if not p.exists():
    sys.exit("run from repo root; tools/make_dashboard.py not found")
s = p.read_text()
if "_rpe_section" in s:
    sys.exit("already patched (found _rpe_section). Revert or start clean to re-apply.")

# 1) inject BOTH placeholders after the 2x2 block
anchor = '</div><p class="q" style="margin-top:16px">{note}</p></div>\n'
if anchor not in s:
    sys.exit("2x2 anchor not found - paste build_view() so I can adjust")
s = s.replace(anchor, anchor + "\n{_rpe_section}\n{_rpe_cards}\n", 1)

# 2) define BOTH vars in build_view (after the disp line), guarded
disp_line = '    disp = "" if active else \' style="display:none"\'\n'
if disp_line not in s:
    sys.exit("disp line not found in build_view")
inject = (
    disp_line +
    "    try:\n"
    "        from rpe_section import render_rpe_section, render_rpe_cards\n"
    "        _rpe_out = globals().get('RPE_OUT_DIR')\n"
    "        _rpe_section = render_rpe_section(view_id, _rpe_out, section_no='01\\u00b7c')\n"
    "        _rpe_cards = render_rpe_cards(view_id, _rpe_out, section_no='01\\u00b7d')\n"
    "    except Exception:\n"
    "        _rpe_section = ''\n"
    "        _rpe_cards = ''\n"
)
s = s.replace(disp_line, inject, 1)

# 3) --rpe-out CLI flag
arg_anchor = '    ap.add_argument("--out", default="results/dashboard.html")\n'
if arg_anchor not in s:
    sys.exit("--out arg not found")
s = s.replace(arg_anchor, arg_anchor +
    '    ap.add_argument("--rpe-out", default=None,\n'
    '                    help="dir with rpe_agreement.json + rpe_manifest_<bench>.jsonl "\n'
    '                         "(adds the independent-probe section + detail cards)")\n', 1)

# 4) set the module global from args
s = s.replace('    args = ap.parse_args(argv)\n',
    '    args = ap.parse_args(argv)\n    global RPE_OUT_DIR\n    RPE_OUT_DIR = args.rpe_out\n', 1)

# 5) declare the global near the top so it always exists
imp = 'from recyclebench.judge import looks_like_refusal\n'
if imp in s and "RPE_OUT_DIR = None" not in s:
    s = s.replace(imp, imp + '\nRPE_OUT_DIR = None  # set by main() --rpe-out; read by build_view\n', 1)

p.write_text(s)
print("patched tools/make_dashboard.py: RPE section 01.c + detail cards 01.d + --rpe-out")
