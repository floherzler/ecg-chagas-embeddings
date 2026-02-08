#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for batch-exporting notebook-style XAI sanity figures.
# Example:
#   scripts/analysis/run_xai_sanity_exports.sh \
#     --run-ids t1-exp03-bp-focal,t3-exp03-bp-sc-supstd \
#     --candidates analysis/embeddings_probe/xai_summary/lead_7/sanity_candidates_excellent.csv

RUN_SPECS="configs/analysis/embeddings_probe_runs.toml"
OUT_DIR="analysis/embeddings_probe"
CANDIDATES="analysis/embeddings_probe/xai_summary/lead_7/sanity_candidates_excellent.csv"
RUN_IDS=""
LEAD_INDEX="7"
MAX_CASES="0"
SKIP_EXISTING="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-ids)
      RUN_IDS="${2:-}"
      shift 2
      ;;
    --candidates)
      CANDIDATES="${2:-}"
      shift 2
      ;;
    --run-specs)
      RUN_SPECS="${2:-}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --lead-index)
      LEAD_INDEX="${2:-}"
      shift 2
      ;;
    --max-cases)
      MAX_CASES="${2:-}"
      shift 2
      ;;
    --no-skip-existing)
      SKIP_EXISTING="0"
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  scripts/analysis/run_xai_sanity_exports.sh --run-ids <comma-separated-run-ids> [options]

Options:
  --candidates <csv>      Candidate exams CSV (default: sanity_candidates_excellent.csv)
  --run-specs <toml>      Run specs TOML
  --out-dir <dir>         Analysis output root
  --lead-index <int>      Lead index (default: 7)
  --max-cases <int>       0 = all, otherwise first N rows
  --no-skip-existing      Recompute even if PNGs already exist
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${RUN_IDS}" ]]; then
  echo "Missing required --run-ids" >&2
  exit 1
fi

CMD=(
  uv run python scripts/analysis/export_xai_sanity_cases.py
  --run_specs "${RUN_SPECS}"
  --out_dir "${OUT_DIR}"
  --candidates_csv "${CANDIDATES}"
  --run_ids "${RUN_IDS}"
  --lead_index "${LEAD_INDEX}"
  --max_cases "${MAX_CASES}"
)

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  CMD+=(--skip_existing)
fi

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" "${CMD[@]}"
