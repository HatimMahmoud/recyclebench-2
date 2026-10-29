"""tools/compose_v2_section.py: the style-factorial section is text-free, scriptless and rebuilt from the store."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import compose_v2_section as cv2                              # noqa: E402

HAVE = cv2.latest(ROOT) is not None and (ROOT / "cache/compose_v2/candidates.jsonl").exists()


def test_absent_run_renders_a_note(tmp_path):
    assert cv2.aggregate(tmp_path) is None
    assert "Not run" in cv2.section_body(None)


@pytest.mark.skipif(not HAVE, reason="compose v2 run not on disk")
def test_real_run_section():
    agg = cv2.aggregate(ROOT)
    body = cv2.section_body(agg)
    assert "<script" not in body and "<style" not in body and 'role="tablist"' in body
    # the per-behaviour verdicts reproduce the summary's per-condition counts exactly
    for c in agg["summary"]["conditions"]:
        v = [x for x in agg["verdicts"][c["condition"]].values() if x is not None]
        assert (sum(v), len(v)) == (c["reopened"], c["resolved"])
    texts = [l for l in (ROOT / "cache/compose_v2/candidates.jsonl").read_text().splitlines()[:5]]
    import json
    for t in texts:
        assert json.loads(t)["text"][:40] not in body          # no candidate text on the page
