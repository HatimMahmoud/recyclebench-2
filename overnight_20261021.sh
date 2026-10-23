#!/usr/bin/env bash
# overnight_20261021.sh -- one night's queue of targeted fills + judge replays (stops at STOP_AT).
#
#   bash overnight_20261021.sh            # preflight, then detaches (nohup + caffeinate); log below
#   tail -f logs/overnight_20261021.log
#   bash overnight_20261021.sh --dry-run  # print every job's remaining work; no model calls
#
# Jobs are grouped by model so Ollama swaps models at most four times. Every job is resumable and skips
# what is cached, so re-running this script after an interruption continues where it stopped. A job that
# fails is logged and the queue moves on. Nothing here writes results/ or overwrites anything: responses
# go to cache/llama_gen.jsonl (append-only), verdicts to cache/judge_*.jsonl (append-only).
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
STOP_AT="${STOP_AT:-07:45}"
LOG=logs/overnight_20261021.log
CFG=configs/real_llama_lineage_all.yaml
SMOKE=results/runs/20260805_124501_llama3-8b_to_llama31-8b_RECYCLE_qwen3b_SMOKE/manifest_harmbench.jsonl
L3=results/runs/20260805_025658_llama3-8b_to_llama31-8b_RECYCLE
SORRY_MAN=results/runs/lineage_sorrybench_20261020/manifest.jsonl
DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="--dry-run"

if [ -z "$DRY" ] && [ -z "${OVERNIGHT_DETACHED:-}" ]; then
  # ---- preflight (foreground, so problems show before you walk away) ----
  curl -sf http://localhost:11434/api/tags > /tmp/ollama_tags.json || { echo "Ollama is not up on :11434"; exit 1; }
  for m in llama3.1:8b llama2:13b qwen2.5:14b; do
    grep -q "\"$m\"" /tmp/ollama_tags.json || { echo "model $m not pulled (ollama pull $m)"; exit 1; }
  done
  if pgrep -f "recyclebench (run-real|pilot-pminus)|fill_responses.py|judge_lineage.py|judge_rpe.py|run_independent_probe.py" > /dev/null; then
    echo "another cache/judge writer is running:"; pgrep -fl "recyclebench|fill_responses|judge_lineage|judge_rpe|run_independent_probe"; exit 1
  fi
  [ -e cache/llama_gen.jsonl.lock ] && echo "note: stale cache/llama_gen.jsonl.lock will be cleared by the first job if its pid is dead"
  pmset -g batt | grep -q "AC Power" || echo "WARNING: not on AC power -- caffeinate cannot keep the Mac awake on battery. Plug in."
  echo "Keep the lid open. Ollama server should run with OLLAMA_KEEP_ALIVE>=1h (current: $(ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | tr '\n' ' '))"
  mkdir -p logs
  OVERNIGHT_DETACHED=1 nohup caffeinate -is bash "$0" >> "$LOG" 2>&1 &
  echo "launched pid $! -- stops at $STOP_AT -- tail -f $LOG"
  exit 0
fi

past_stop() { [ -z "$DRY" ] && [ "$(date +%H%M)" -ge "${STOP_AT/:/}" ] && [ "$(date +%H)" -lt 12 ]; }
run() {  # run <label> <cmd...>
  local label="$1"; shift
  if past_stop; then echo "== $(date '+%H:%M:%S') SKIP $label (past $STOP_AT)"; return; fi
  echo; echo "== $(date '+%H:%M:%S') START $label"
  local t0=$SECONDS
  "$@" || echo "== $(date '+%H:%M:%S') FAILED $label (exit $?) -- continuing"
  echo "== $(date '+%H:%M:%S') END $label ($(( (SECONDS - t0) / 60 )) min)"
}
FILL() { python3 tools/fill_responses.py --config "$CFG" --stop-at "$STOP_AT" $DRY "$@"; }
JUDGE() { python3 tools/judge_lineage.py --config "$CFG" --stop-at "$STOP_AT" $DRY "$@"; }

echo "######## overnight queue $(date) · stop at $STOP_AT ${DRY:+· DRY RUN}"

# ---- A. llama3.1:8b ------------------------------------------------------------------------------
run "A1 llama31-8b canonical HarmBench (3.1 -> 3.2 pairs, PBR)" \
  FILL --model llama31-8b --behaviours data/harmbench.csv --canonical
run "A2 llama31-8b SMOKE rewordings on P-(3-8b -> 3.1-8b) HarmBench" \
  FILL --model llama31-8b --behaviours data/harmbench.csv --manifest "$SMOKE" --only-pminus llama3-8b:llama31-8b
run "A3 llama31-8b all SMOKE rewordings HarmBench (PBR)" \
  FILL --model llama31-8b --behaviours data/harmbench.csv --manifest "$SMOKE"

# ---- B. llama2:13b on the 2-7b -> 2-13b fixes ----------------------------------------------------
run "B1 llama2-13b SMOKE rewordings on P-(2-7b -> 2-13b) HarmBench" \
  FILL --model llama2-13b --behaviours data/harmbench.csv --manifest "$SMOKE" --only-pminus llama2-7b:llama2-13b
run "B2 llama2-13b rewordings on P-(2-7b -> 2-13b) AdvBench" \
  FILL --model llama2-13b --behaviours data/advbench.csv --manifest "$L3/manifest_advbench.jsonl" --only-pminus llama2-7b:llama2-13b
run "B3 llama2-13b rewordings on P-(2-7b -> 2-13b) StrongREJECT" \
  FILL --model llama2-13b --behaviours data/strongreject.csv --manifest "$L3/manifest_strongreject.jsonl" --only-pminus llama2-7b:llama2-13b

# ---- C. qwen2.5:14b judge replays (no target calls) ----------------------------------------------
run "C1 qwen re-judge of RPE admitted replies" \
  python3 tools/judge_rpe.py --stop-at "$STOP_AT" $DRY
run "C2 qwen judge SORRY-Bench lineage (canonical + RSR variants)" \
  JUDGE --behaviours data/sorrybench.csv --manifest "$SORRY_MAN" --out cache/judge_lineage_sorrybench.jsonl --scope rsr
run "C3 qwen judge HarmBench, Llama-2/3 lineage (canonical + RSR variants)" \
  JUDGE --behaviours data/harmbench.csv --manifest "$SMOKE" --models llama2-7b llama2-13b llama3-8b llama31-8b \
        --out cache/judge_qwen_harmbench.jsonl --scope rsr

# ---- D. llama2:13b remaining SMOKE rewordings (PBR) ----------------------------------------------
run "D1 llama2-13b all SMOKE rewordings HarmBench (PBR)" \
  FILL --model llama2-13b --behaviours data/harmbench.csv --manifest "$SMOKE"

# ---- tail: judge every SORRY-Bench variant (PBR) until the stop time -------------------------------
run "T1 qwen judge SORRY-Bench lineage, all variants (PBR)" \
  JUDGE --behaviours data/sorrybench.csv --manifest "$SORRY_MAN" --out cache/judge_lineage_sorrybench.jsonl --scope all

echo; echo "######## finished $(date)"
for f in cache/llama_gen.jsonl cache/judge_lineage_sorrybench.jsonl cache/judge_qwen_harmbench.jsonl cache/judge_rpe_qwen.jsonl; do
  [ -e "$f" ] && echo "  $(wc -l < "$f" | tr -d ' ') lines  $f"
done
