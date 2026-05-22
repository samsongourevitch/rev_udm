#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?usage: $0 {temperature|nucleus|ppl} RUN_NAME [hydra overrides...]}
RUN_NAME=${2:?usage: $0 {temperature|nucleus|ppl} RUN_NAME [hydra overrides...]}
shift 2

case "$MODE" in
  temperature)
    RUNNER="eval_scripts/temperature_sweep/run_temperature_manifest.py"
    MANIFEST_ROOT="samples/temp_sweep"
    ;;
  nucleus)
    RUNNER="eval_scripts/nucleus_sweep/run_nucleus_manifest.py"
    MANIFEST_ROOT="samples/nucleus_sweep"
    ;;
  ppl)
    RUNNER="eval_scripts/ppl_eval/run_ppl_manifest.py"
    MANIFEST_ROOT="samples/ppl_eval"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 1
    ;;
esac

PYTHON_BIN=${PYTHON_BIN:-python}
MANIFEST=${MANIFEST:-${MANIFEST_ROOT}/${RUN_NAME}/pending_manifest.jsonl}
N=$(wc -l < "$MANIFEST")

for i in $(seq 0 $((N - 1))); do
  "$PYTHON_BIN" "$RUNNER" \
    manifest.path="$(realpath "$MANIFEST")" \
    launcher.manifest_idx="$i" \
    run.name="$RUN_NAME" \
    "$@"
done
