# Bring your own data

Every arm of this study reads **one** shape: a CSV with a `Goal` column, one request per row
(`recyclebench.datasets.load_jailbreakbench`). Adding a corpus is therefore writing **one recipe
file** — not a converter, and not a change to any arm.

```bash
python3 tools/prepare_external.py --recipe datasets/<yours>.yaml
```

That writes `data/<yours>.csv` plus `data/<yours>.csv.provenance.json`, and the corpus is then usable
by every command in this repo.

---

## 1. Write a recipe

`datasets/sciencesafety.yaml` is the worked example. The whole schema:

```yaml
name: mycorpus                 # names the download dir under data/external/
license: MIT                   # the upstream licence; record it

source:
  kind: github                 # github | local
  repo: owner/name
  commit: <full 40-char sha>   # REQUIRED and immutable: a branch name is rejected
  root: data/subdir            # optional prefix for every path below
  paths:                       # recipe order == CSV order (deterministic)
    - a.json
    - b.json

format: json                   # json (a list of objects per file) | jsonl | csv

mapping:
  goal: Prompt                 # REQUIRED: the field holding the request
  category: "@stem"            # a field name, or @stem / @parent / @literal:<x>
  extra:                       # optional extra CSV columns <- source fields
    Task: Task
    RiskType: Risk Type

out: data/mycorpus.csv

dedup:                         # optional but recommended
  against: [sorrybench, harmbench, advbench, strongreject]
  cutoff: 0.85
```

**`category`** becomes the `Category` column, which is what per-category readouts (RQ2) group by.
`@stem` uses the source file's name, which is right when one file is one domain or category;
`@parent` uses its directory; `@literal:x` sets a constant; anything else is read as a field name.

For a corpus already on disk, use `kind: local` and give `paths` (optionally under `root`); absolute
paths are respected.

## 2. What the adapter guarantees

* **Deterministic** — same recipe + same source bytes ⇒ byte-identical CSV. Nothing is shuffled or
  sampled. Records keep recipe order, then in-file order.
* **Pinned** — a github source is fetched at an immutable commit into
  `data/external/<name>_<commit7>/`, and every file's sha256 is recorded.
* **Provenance** — `<out>.provenance.json` holds the recipe hash, commit, licence, per-file hashes,
  the field mapping, the row count at each filter step, and the overlap report. Cite it in a prereg.
* **Overlap-aware** — behaviours duplicating an already-screened benchmark are **reported** (and with
  `--drop-overlap`, removed), so a "new" corpus cannot silently re-screen seen items. Same
  normalisation and difflib cutoff as `tools/prepare_heldout.py`.
* **Quiet** — prints counts, hashes and field names only, never prompt text, so logs and terminal
  scrollback carry no harmful content.

Useful flags: `--dry-run` (report, write nothing), `--no-fetch` (offline; sources must be present),
`--out` (override the recipe's `out:`).

## 3. Run it through the pipeline

Using a pair that exists in your config's `lineage:` (see `configs/`):

```bash
# 1. size the regression set, judged by the config's judge  -> |P^-|
python3 -m recyclebench pilot-pminus --config configs/<pair>.yaml \
    --old <old> --new <new> --behaviours data/mycorpus.csv --cache cache/<mine>.jsonl

# 2. take the JUDGED P^- (not the offline-rule dump; see the warning below)
python3 tools/pminus_subset.py --pilot results/pminus_pilot.json \
    --behaviours data/mycorpus.csv --out <run>/mycorpus_p_minus_judged.csv

# 3. recycle: paraphrase -> gate -> freeze -> re-screen  -> RSR, PBR
python3 -m recyclebench run-real --config configs/<pair>.yaml \
    --behaviours <run>/mycorpus_p_minus_judged.csv \
    --manifest <run>/manifest_mycorpus.jsonl --cache cache/<mine>.jsonl

# 4. the 2x2 partition, including the negative-flip control
python3 tools/pminus_report.py --config configs/<pair>.yaml --old <old> --new <new> \
    --cache cache/<mine>.jsonl data/mycorpus.csv --screen-inverse --dump <run>/cells

# 5. the judge-calibration sheet for this corpus (zero target calls)
python3 tools/judge_export_real.py --config configs/<pair>.yaml --new-model <new> \
    --behaviours <run>/mycorpus_p_minus_judged.csv --manifest <run>/manifest_mycorpus.jsonl \
    --cache cache/<mine>.jsonl --out audit/mycorpus/judge_sheet.csv --n-bundles 60
```

The convergent **inversion (RPE) arm** needs no code change — `tools/rpe_lineage.py` takes
`--behaviours`. It does need a verdict store first (`tools/judge_lineage.py`, a judge replay with zero
target calls), and every inversion candidate must be gated (`tools/rpe_gate.py`) or the closed-loop
artifact has empty `A_gate`/`A_both` *by construction*. The **style factorial** takes
`tools/run_compose_v2.py --behaviours <your judged P⁻ CSV>`; omitting the flag keeps the registered
anchor set, so that result is unaffected.

`run_sciencesafety_20261101.sh` is a worked, resumable queue doing exactly this, with an isolated
cache, a hard stop, per-stage markers, and a restore of the fixed-path scratch files;
`run_sciencesafety_rpe_20261101.sh` chains the follow-on arms behind it (a separate file, because
bash reads a script incrementally and editing a running one can make it execute garbage). Copy both.

## 4. Five things that will bite you

1. **Use the judged P⁻, not the dumped cell.** `pminus_report.py --dump` rebuilds its partition with
   the offline refusal rule, because `run-real` never stores judge verdicts. On SORRY-Bench that rule
   gives P⁻ = 14/2/61/19/33 where the `qwen2.5:14b` judge gives 18/3/49/12/13. Feeding the dump into
   `run-real` recycles the wrong behaviours. `tools/pminus_subset.py` exists for this.
2. **Ids in a subset CSV are row positions.** `load_jailbreakbench` ignores a `bid` column and
   renumbers rows, so ids inside a run built on a subset are not ids in your corpus. **Join back on
   prompt text.** (`pminus_subset.py` keeps a `source_bid` column to make that join checkable.)
3. **Cache keys depend on the config key, the prompt and the seed** —
   `sha256(f"{model}|{prompt}|{seed}")`, where the canonical seed comes from the behaviour's **bid**.
   So a subset whose rows sit at different positions produces different seeds and re-queries the
   model. A *prefix* subset keeps its bids and reuses the cache; an arbitrary subset does not. Keep
   config keys stable, and give a new corpus its own cache file until you want it merged.
4. **A new corpus has no judge calibration.** Corrected (Rogan–Gladen) figures need human labels from
   *that* corpus; applying another corpus's sheet is a **transfer**, reported as a sensitivity and
   labelled, never as a headline. Export a sheet with `judge_export_real.py` and label it.
5. **Your corpus may be off the instrument's calibrated range.** Both the judge ("did the reply
   attempt the request") and the gate (BERTScore ≥ 0.85, bidirectional NLI ≥ 0.60) were calibrated on
   short, blunt requests. A corpus of long, technical or research-framed prompts can move the
   admission rate and can score a benign reading as compliance. Report the admission rate and the
   computed component-score distribution beside an already-calibrated corpus before trusting an RSR
   difference. `PREREG_SCIENCESAFETY_20261101.md` registers exactly these threats for a worked case.

## 5. Register before you run

A new corpus is a new population, so it needs its own registration: the pinned source and CSV hash,
the pair, the estimands, the decision rule, and the threats above. `PREREG_SCIENCESAFETY_20261101.md`
is the template. Keep the run isolated (its own cache and run directory, not registered in
`tools/study_data.py` and on no dashboard) until you know it carries a non-empty P⁻ — then fold it in
by an amendment, not silently.
