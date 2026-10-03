"""
contract.py -- the data contract for the independent-probe (RPE) arm.

The RPE validator inverts the OLD model's harmful response to recover candidate
prompts, gates them for intent, and re-queries the NEW model. To do that it needs,
per P-minus behaviour:

    - bid          : behaviour id (join key across all files)
    - prompt       : the canonical P-minus prompt (old complied, new refused)
    - old_response : the OLD model's harmful completion  <-- what RPE inverts
    - benchmark    : which source set (harmbench / strongreject / advbench)
    - para_recovered (optional) : the paraphrase arm's verdict for this bid,
                                  1 = recovered, 0 = held, None = unresolved.
                                  This is RSR_paraphrase, used only in the
                                  agreement analysis; it comes from the recycle run.

This file defines the ProbeItem record and the ADAPTERS that read your existing
results/ tree. The adapters are the ONLY place that touches file formats, so when
the real formats are confirmed, only the parse_* functions below change -- never
the engine, gate, runner, or analysis.

Existing run layout this targets (llama3-8b -> llama3.1-8b):
    runs/<ts>_llama3-8b_to_llama31-8b_full/
        pminus_<bench>.json                     # the P- prompts
        responses/<bench>_only-pminus.txt       # the OLD-model responses for P-
    runs/<ts>_llama3-8b_to_llama31-8b_RECYCLE/
        manifest_<bench>.jsonl                  # per-behaviour paraphrase results
        summary_<bench>.json                    # RSR_paraphrase headline
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import re


@dataclass
class ProbeItem:
    bid: str
    prompt: str
    old_response: str
    benchmark: str
    para_recovered: Optional[int] = None          # from the paraphrase arm
    # filled in by the probe run:
    rpe_candidates: list = field(default_factory=list)
    rpe_admitted: list = field(default_factory=list)   # passed the intent gate
    rpe_recovered: Optional[int] = None                # RPE arm verdict (1/0/None)
    rpe_details: list = field(default_factory=list)  # per-candidate detail for cards


# ---------------------------------------------------------------------------
# ADAPTERS -- the only format-dependent code. Adjust these once real formats
# are confirmed; everything downstream is format-agnostic.
# ---------------------------------------------------------------------------

def parse_pminus_prompts(path: Path) -> dict[str, str]:
    """
    Read pminus_<bench>.json -> {bid: prompt}.

    Tolerant to the common shapes a P- file might take:
      A) [{"bid": "...", "prompt": "..."}, ...]
      B) {"synb-0001": "prompt text", ...}
      C) [{"id": "...", "behavior": "..."}, ...]   (HarmBench-style keys)
      D) ["prompt text", ...]  -> synthesised bids b0000, b0001, ...
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, str] = {}

    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, str):
                out[str(k)] = v
            elif isinstance(v, dict):
                out[str(k)] = _first_text(v, ("prompt", "behavior", "behaviour", "text", "instruction"))
        return out

    if isinstance(raw, list):
        for i, item in enumerate(raw):
            if isinstance(item, str):
                out[f"b{i:04d}"] = item
            elif isinstance(item, dict):
                bid = str(item.get("bid") or item.get("id") or item.get("behavior_id") or f"b{i:04d}")
                out[bid] = _first_text(item, ("prompt", "behavior", "behaviour", "text", "instruction", "goal"))
        return out

    raise ValueError(f"Unrecognised P- json shape in {path}: top-level {type(raw).__name__}")


def parse_old_responses(path: Path) -> dict[str, str]:
    """
    Read responses/<bench>_only-pminus.txt -> {bid: old_response}.

    These .txt dumps vary; this handles the two shapes seen most often:

      SHAPE 1 (delimited blocks):
          === synb-0001 ===
          <response text, possibly multi-line>
          === synb-0002 ===
          ...
        Any header line matching an id token starts a new block.

      SHAPE 2 (one record per double-newline, no ids):
          <response 1>

          <response 2>
        -> synthesised bids b0000, b0001 in file order (must match prompt order).

    If the real format differs, THIS function is what to change.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    # SHAPE 1: look for header lines carrying an id like "=== bid ===",
    # "[bid]", "### bid", "BID: xxx", or a bare id token on its own line.
    header = re.compile(
        r"""^\s*(?:=+\s*|\#+\s*|\[)?          # optional === / ### / [
            (?P<bid>[A-Za-z][\w\-]*\d[\w\-]*)  # an id token containing a digit
            (?:\s*=+|\s*\]|:)?\s*$             # optional closing === / ] / :
        """,
        re.VERBOSE,
    )
    blocks: dict[str, list[str]] = {}
    cur: Optional[str] = None
    saw_header = False
    for line in text.splitlines():
        m = header.match(line)
        if m and len(line.strip()) < 40:       # headers are short lines
            cur = m.group("bid")
            blocks[cur] = []
            saw_header = True
            continue
        if cur is not None:
            blocks[cur].append(line)

    if saw_header and blocks:
        return {bid: "\n".join(lines).strip() for bid, lines in blocks.items()}

    # SHAPE 2: split on blank-line-separated records, synth bids by order.
    records = [r.strip() for r in re.split(r"\n\s*\n", text) if r.strip()]
    return {f"b{i:04d}": r for i, r in enumerate(records)}


def parse_recycle_manifest(path: Path) -> dict[str, Optional[int]]:
    """
    Read a recycle manifest_<bench>.jsonl -> {bid: para_recovered in {1,0,None}}.

    Each line is a JSON object for one behaviour (or one variant attempt). We
    reduce to a per-behaviour verdict: recovered=1 if ANY admitted variant
    bypassed the new model, else 0; None if the record marks it unresolved.

    Tolerant to field-name variants: bypassed / recovered / success /
    attack_successful, and bid / id / behavior_id.
    """
    verdict: dict[str, Optional[int]] = {}
    p = Path(path)
    if not p.exists():
        return verdict
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        bid = str(rec.get("bid") or rec.get("id") or rec.get("behavior_id") or "")
        if not bid:
            continue
        val = _first_present(rec, ("recovered", "bypassed", "success", "attack_successful"))
        if val is None:
            # unresolved only if explicitly marked; otherwise treat as held
            if rec.get("unresolved") or rec.get("prediction") in (-1, -1.0, "-1"):
                verdict[bid] = None
            else:
                verdict.setdefault(bid, 0)
            continue
        v = 1 if int(val) == 1 else 0
        # OR-reduce across multiple rows for the same bid
        prev = verdict.get(bid)
        verdict[bid] = 1 if (prev == 1 or v == 1) else 0
    return verdict


# ---------------------------------------------------------------------------
# JOIN -- assemble ProbeItems for one benchmark from the three sources.
# Format-agnostic: relies only on the adapter outputs above.
# ---------------------------------------------------------------------------

def load_probe_items(
    benchmark: str,
    pminus_json: Path,
    responses_txt: Path,
    recycle_manifest: Optional[Path] = None,
) -> list[ProbeItem]:
    prompts = parse_pminus_prompts(pminus_json)
    responses = parse_old_responses(responses_txt)
    para = parse_recycle_manifest(recycle_manifest) if recycle_manifest else {}

    # Join on bid. If responses were synth-numbered (SHAPE 2) and prompts have
    # real bids, fall back to positional alignment with a loud warning.
    items: list[ProbeItem] = []
    prompt_bids = list(prompts.keys())
    resp_bids = list(responses.keys())

    positional = not (set(prompt_bids) & set(resp_bids))
    if positional:
        print(f"[contract] WARNING: no shared bids between prompts and responses "
              f"for {benchmark}; falling back to POSITIONAL join "
              f"({len(prompt_bids)} prompts, {len(resp_bids)} responses). "
              f"Verify ordering matches!")
        n = min(len(prompt_bids), len(resp_bids))
        for i in range(n):
            bid = prompt_bids[i]
            items.append(ProbeItem(
                bid=bid, prompt=prompts[prompt_bids[i]],
                old_response=responses[resp_bids[i]], benchmark=benchmark,
                para_recovered=para.get(bid),
            ))
    else:
        for bid in prompt_bids:
            if bid not in responses:
                continue
            items.append(ProbeItem(
                bid=bid, prompt=prompts[bid],
                old_response=responses[bid], benchmark=benchmark,
                para_recovered=para.get(bid),
            ))
    return items


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _first_text(d: dict, keys: tuple) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    # last resort: the longest string value in the dict
    strs = [v for v in d.values() if isinstance(v, str)]
    return max(strs, key=len) if strs else ""


def _first_present(d: dict, keys: tuple):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


# ============================================================
# PRIMARY loader: read P- from make_dashboard.partition() (added)
# ============================================================
def load_probe_items_from_partition(benchmark, cells, recycle_manifest=None, restrict_bids=None):
    """Build ProbeItems from make_dashboard.partition() output -- the SAME P- source
    the Sankey dashboard uses. Each pminus record carries {bench,bid,prompt,old_resp,new_resp}."""
    para = parse_recycle_manifest(recycle_manifest) if recycle_manifest else {}
    items = []
    for rec in cells.get("pminus", []):
        if rec.get("bench") != benchmark:
            continue
        bid = rec["bid"]
        if restrict_bids is not None and bid not in restrict_bids:
            continue
        items.append(ProbeItem(
            bid=bid,
            prompt=rec.get("prompt", ""),
            old_response=(rec.get("old_resp") or "").strip(),
            benchmark=benchmark,
            para_recovered=para.get(bid),
        ))
    return items


def load_pminus_bids(pminus_json):
    """Read 'p_minus_bids' out of a run's pminus_<bench>.json metadata file."""
    import json as _json
    from pathlib import Path as _P
    try:
        d = _json.loads(_P(pminus_json).read_text(encoding="utf-8"))
    except Exception:
        return set()
    return set(d.get("p_minus_bids", []) or []) if isinstance(d, dict) else set()
