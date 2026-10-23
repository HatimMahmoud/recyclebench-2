#!/usr/bin/env bash
# overnight_stack.sh -- the queue of long (> 3 h) or sign-off-gated jobs, run one night at a time.
#
#   bash overnight_stack.sh --list               # what is queued and what each job still needs (no model calls)
#   bash overnight_stack.sh S1 S2                # run these jobs tonight (preflight, then detaches; log below)
#   bash overnight_stack.sh --stamp              # AFTER sign-off of prereg Amendment 1: freeze the replication
#
# Jobs (all resumable; each skips what is already cached):
#   S1  qwen judge, HarmBench lineage, every variant (PBR backbone)          ~4,700 calls  ~2.5 h
#   S2  HarmBench judge-calibration sheet for blind labelling (judge_export_real, 60 bundles)  ~15 min
#   S3  CONFIRMATORY replication, llama3-8b -> 3.1-8b on the held-out set (run-real)  ~5-6 h  [needs --stamp]
#   S4  analysis + dashboards from cache (no model calls)                    ~5 min
#   S6  qwen judge AdvBench + StrongREJECT, Llama-2/3 lineage (canonical + RSR variants) ~3,400 calls ~1.5 h
#   S7  qwen judge HarmBench/AdvBench/StrongREJECT on the llama3.2 manifests (3.1-8B -> 3.2-1B -> 3.2-3B)  ~1 h
#   S8  qwen judge AdvBench + StrongREJECT, every variant (PBR backbone)         ~10,800 calls ~4.5 h
#   S5  white-box probe of every re-opened anchor fix (SORRY-Bench, 3-8B vs 3.1-8B; GGUF weights in PyTorch/MPS,
#       no Ollama calls; ~16 GB RAM per model, run alone)                    ~30-45 min
# Nothing here overwrites results: analysis and dashboards are dated, the replication has its own run dir.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
# stop starting new work STOP_HOURS after launch (default 11 h); S4 (no model calls) always runs at the end
STOP_HOURS="${STOP_HOURS:-11}"
STOP_EPOCH="${STOP_EPOCH:-$(( $(date +%s) + STOP_HOURS * 3600 ))}"
STOP_AT="$(date -r "$STOP_EPOCH" +%H:%M)"
L32=results/runs/20260801_165721_llama32-1b_to_3b_recycle
L3=results/runs/20260805_025658_llama3-8b_to_llama31-8b_RECYCLE
LOG=logs/overnight_stack_$(date +%Y%m%d).log
CFG=configs/real_llama_lineage_all.yaml
SMOKE=results/runs/20260805_124501_llama3-8b_to_llama31-8b_RECYCLE_qwen3b_SMOKE/manifest_harmbench.jsonl
CCFG=configs/confirm_heldout_3_31.yaml
CCSV=data/heldout_jbb_guided.csv
CDIR=results/runs/confirm_heldout_3_31
STAMP=$CDIR/STAMP.json
A70="$HOME/Downloads/all_local_results_llama3_1_70b 1.jsonl"

stamp_ok() {  # the replication may run only if the stamp exists and config + CSV are byte-identical to it
  [ -f "$STAMP" ] || { echo "S3 blocked: no $STAMP (sign off Amendment 1, then: bash overnight_stack.sh --stamp)"; return 1; }
  python3 - "$STAMP" "$CCFG" "$CCSV" << 'EOF'
import hashlib, json, sys
st = json.load(open(sys.argv[1]))
for key, p in (("config_sha256", sys.argv[2]), ("behaviours_sha256", sys.argv[3])):
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if h != st[key]:
        sys.exit(f"S3 blocked: {p} changed since stamping ({h[:12]} != {st[key][:12]}) -- that needs a new amendment")
print(f"stamp ok ({st['stamped']})")
EOF
}

if [ "${1:-}" = "--stamp" ]; then
  [ -f "$STAMP" ] && { echo "already stamped: $STAMP"; exit 1; }
  if [ -e "$CDIR/manifest.jsonl" ] || grep -q "heldout" logs/*.log 2>/dev/null; then
    echo "refusing: there are traces of a held-out run already"; exit 1
  fi
  mkdir -p "$CDIR"
  python3 -m recyclebench prereg --config "$CCFG" --behaviours "$CCSV" --out "$CDIR/PREREG_CONFIRMATORY.md" || exit 1
  python3 - "$STAMP" "$CCFG" "$CCSV" "$CDIR/PREREG_CONFIRMATORY.md" << 'EOF'
import datetime, hashlib, json, sys
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump(dict(stamped=datetime.datetime.now().isoformat(timespec="seconds"), config=sys.argv[2], config_sha256=sha(sys.argv[2]),
               behaviours=sys.argv[3], behaviours_sha256=sha(sys.argv[3]), prereg=sys.argv[4], prereg_sha256=sha(sys.argv[4]),
               amendment="prereg.md, Amendment 1 (2026-10-22)"), open(sys.argv[1], "w"), indent=1)
print(open(sys.argv[1]).read())
EOF
  exit 0
fi

if [ "${1:-}" = "--list" ]; then
  echo "S1:"; python3 tools/judge_lineage.py --config $CFG --behaviours data/harmbench.csv --manifest $SMOKE \
      --models llama2-7b llama2-13b llama3-8b llama31-8b --out cache/judge_qwen_harmbench.jsonl --scope all --dry-run | tail -1
  echo "S2:"; [ -f audit/harmbench/judge_sheet.csv ] && echo "  sheet exists (audit/harmbench/judge_sheet.csv)" || echo "  not exported yet"
  echo "S3:"; stamp_ok || true
  echo "S4: always runnable"
  for b in advbench strongreject; do echo "S6/S8 $b:"; python3 tools/judge_lineage.py --config $CFG --behaviours data/$b.csv \
      --manifest $L3/manifest_$b.jsonl --models llama2-7b llama2-13b llama3-8b llama31-8b --out cache/judge_qwen_$b.jsonl --scope all --dry-run | tail -2; done
  for b in harmbench advbench strongreject; do echo "S7 $b:"; python3 tools/judge_lineage.py --config $CFG --behaviours data/$b.csv \
      --manifest $L32/manifest_$b.jsonl --models llama31-8b llama32-1b llama32-3b --out cache/judge_qwen_$b.jsonl --scope rsr --dry-run | tail -1; done
  echo "S5: $(ls results/mech/anchor_sorrybench_all_*.json 2>/dev/null | tail -1 || echo 'not run yet')"
  exit 0
fi

if [ -z "${STACK_DETACHED:-}" ]; then
  [ $# -gt 0 ] || { echo "name the jobs to run, e.g. bash overnight_stack.sh S1 S2 S4 (see --list)"; exit 1; }
  curl -sf http://localhost:11434/api/tags > /dev/null || { echo "Ollama is not up"; exit 1; }
  if pgrep -f "recyclebench (run-real|pilot-pminus)|fill_responses.py|judge_lineage.py|judge_rpe.py|regate_intent.py|judge_export_real.py|mech_probe.py" > /dev/null; then
    echo "another model job is running:"; pgrep -fl "recyclebench|fill_responses|judge_lineage|judge_rpe|regate_intent|judge_export_real"; exit 1
  fi
  for j in "$@"; do [ "$j" = S3 ] && { stamp_ok || exit 1; }; done
  pmset -g batt | grep -q "AC Power" || echo "WARNING: not on AC power -- plug in (caffeinate cannot hold the Mac awake on battery)"
  mkdir -p logs
  STACK_DETACHED=1 STOP_EPOCH="$STOP_EPOCH" nohup caffeinate -is bash "$0" "$@" >> "$LOG" 2>&1 &
  echo "launched pid $! -- jobs: $* -- stops new work at $STOP_AT -- tail -f $LOG"
  exit 0
fi

past_stop() { [ "$(date +%s)" -ge "$STOP_EPOCH" ]; }
run_always() { local label="$1"; shift
  echo; echo "== $(date '+%H:%M:%S') START $label"; local t0=$SECONDS
  "$@" || echo "== $(date '+%H:%M:%S') FAILED $label (exit $?) -- continuing"
  echo "== $(date '+%H:%M:%S') END $label ($(( (SECONDS - t0) / 60 )) min)"; }
unload_ollama() { ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | while read -r m; do [ -n "$m" ] && ollama stop "$m"; done; }
run() { local label="$1"; shift
  if past_stop; then echo "== $(date '+%H:%M:%S') SKIP $label (past $STOP_AT)"; return; fi
  echo; echo "== $(date '+%H:%M:%S') START $label"; local t0=$SECONDS
  "$@" || echo "== $(date '+%H:%M:%S') FAILED $label (exit $?) -- continuing"
  echo "== $(date '+%H:%M:%S') END $label ($(( (SECONDS - t0) / 60 )) min)"; }

echo "######## overnight stack $(date) · jobs: $* · stop at $STOP_AT"
for job in "$@"; do
  case "$job" in
    S1) run "S1 qwen judge HarmBench lineage, all variants (PBR)" \
          python3 tools/judge_lineage.py --config $CFG --behaviours data/harmbench.csv --manifest $SMOKE \
            --models llama2-7b llama2-13b llama3-8b llama31-8b --out cache/judge_qwen_harmbench.jsonl --scope all --stop-at "$STOP_AT" ;;
    S2) if [ -f audit/harmbench/judge_sheet.csv ]; then echo "== S2 skipped: audit/harmbench/judge_sheet.csv exists (may hold labels)"
        else mkdir -p audit/harmbench
          run "S2 HarmBench judge-calibration sheet" \
            python3 tools/judge_export_real.py --config configs/recycle_3_31.yaml --new-model llama31-8b \
              --behaviours data/harmbench.csv --manifest $SMOKE --cache cache/llama_gen.jsonl --n-bundles 60 \
              --out audit/harmbench/judge_sheet.csv
        fi ;;
    S3) if stamp_ok; then
          run "S3 CONFIRMATORY replication (held-out JBB + GuidedBench)" \
            python3 -m recyclebench run-real --config $CCFG --behaviours $CCSV --manifest $CDIR/manifest.jsonl \
              --cache cache/llama_gen.jsonl
        fi ;;
    S4) run_always "S4a analysis" python3 tools/thesis_analysis.py
        run_always "S4b master dashboard" python3 tools/master_dashboard.py --attack-70b-responses "$A70"
        run_always "S4c pair dashboards" python3 tools/pair_dashboard.py --attack-70b-responses "$A70"
        run_always "S4d lineage dashboard" python3 tools/lineage_dashboard.py ;;
    S5) unload_ollama
        run "S5 white-box probe, every re-opened anchor fix" python3 tools/mech_probe.py --all ;;
    S6) for b in advbench strongreject; do
          run "S6 qwen judge $b, Llama-2/3 lineage (canonical + RSR)" \
            python3 tools/judge_lineage.py --config $CFG --behaviours data/$b.csv --manifest $L3/manifest_$b.jsonl \
              --models llama2-7b llama2-13b llama3-8b llama31-8b --out cache/judge_qwen_$b.jsonl --scope rsr --stop-at "$STOP_AT"
        done ;;
    S7) for b in harmbench advbench strongreject; do
          run "S7 qwen judge $b, llama3.2 manifests (canonical + RSR)" \
            python3 tools/judge_lineage.py --config $CFG --behaviours data/$b.csv --manifest $L32/manifest_$b.jsonl \
              --models llama31-8b llama32-1b llama32-3b --out cache/judge_qwen_$b.jsonl --scope rsr --stop-at "$STOP_AT"
        done ;;
    S8) for b in advbench strongreject; do
          run "S8 qwen judge $b, every variant (PBR)" \
            python3 tools/judge_lineage.py --config $CFG --behaviours data/$b.csv --manifest $L3/manifest_$b.jsonl \
              --models llama2-7b llama2-13b llama3-8b llama31-8b --out cache/judge_qwen_$b.jsonl --scope all --stop-at "$STOP_AT"
        done ;;
    *) echo "unknown job $job" ;;
  esac
done
echo; echo "######## finished $(date)"
