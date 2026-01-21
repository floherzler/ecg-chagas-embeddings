# Embeddings Probe Analysis Pipeline

This folder contains a small, reproducible analysis pipeline for comparing trained models from tracks 1–3 on a fixed held-out test fold (**fold 4**).

## Configure runs

Edit `configs/analysis/embeddings_probe_runs.toml` and fill `checkpoint_path` for each run:
- local `.ckpt` path, or
- W&B artifact URI as either `wandb:<entity>/<project>/<artifact>:<alias>` or the bare form `<entity>/<project>/<artifact>:<alias>`.

## Run the pipeline

All scripts default to writing into:

`./analysis/embeddings_probe/`

Override with `--out_dir` if needed (e.g. to `/tmp/...` for quick tests).

## Output layout

- Shared (one per probe definition): `probe_index.csv`, `probe_metadata.csv`, `test_scores.csv`, `embedding_metrics.csv`
- Per run: `runs/<run_id>/memmap/`, `runs/<run_id>/coords/`, `runs/<run_id>/plots/`

1) Build probe subset + metadata (from fold4):

```bash
.venv/bin/python scripts/analysis/build_probe_set.py
```

By default, negatives are split 50/50 between `CODE15` and `PTBXL` (SaMi-Trop has no fold4 negatives in `processedMaster/metadata.csv`).
Override with:

```bash
.venv/bin/python scripts/analysis/build_probe_set.py --neg_frac_code15 0.5 --neg_frac_ptbxl 0.5 --neg_frac_samitrop 0.0
```

2) Evaluate each run on full fold4:

```bash
.venv/bin/python scripts/analysis/evaluate_test_models.py
```

3) Extract probe embeddings (Pattern A memmaps):

```bash
.venv/bin/python scripts/analysis/extract_probe_embeddings.py
```

4) Compute TTC embedding metrics + collapse diagnostics:

```bash
.venv/bin/python scripts/analysis/compute_embedding_metrics.py
```

5) Compute PCA/UMAP coordinates from stored embeddings:

```bash
.venv/bin/python scripts/analysis/compute_projections.py --normalize
```

5b) Correlate probe metadata with PCA axes (enc space):

```bash
.venv/bin/python scripts/analysis/compute_pca_correlations.py --space enc --write_into_test_scores
```

5c) Ranking agreement + screening overlap (full test fold logits):

Writes model×model CSVs (Spearman rho, top-5% IoU, top-5% Kendall tau) and per-sample top-5% consensus
(`c_i = #models where sample i is in top-5%`):

```bash
.venv/bin/python scripts/analysis/compute_ranking_agreement.py --set test
```

Outputs are written under `analysis/embeddings_probe/ranking_agreement/test/`.

6) Generate thesis-style plots (hex tiling only):

```bash
.venv/bin/python scripts/analysis/plot_probe_hex_panels.py --run_id <RUN_ID> --out_dir ./analysis/embeddings_probe --clean
```

## One-command plot generation (probe)

Generate the “thesis-style” plots for one or more runs and archive older PNGs:

```bash
.venv/bin/python scripts/analysis/plot_probe_hex_panels.py --run_specs configs/analysis/embeddings_probe_runs.toml --clean
```

Control hex size globally via `--gridsize` (lower => bigger hexes), e.g.:

```bash
.venv/bin/python scripts/analysis/plot_probe_hex_panels.py --run_specs configs/analysis/embeddings_probe_runs.toml --clean --gridsize 30 --mincnt 3
```

## Metadata correlations (CODE15)

Pair plot / scatterplot matrix (samples rows for speed):

```bash
.venv/bin/python scripts/analysis/code15_pairplot.py --sample_n 5000 --corner
```

Focused correlation report (recommended over pairplot for mixed binary/continuous columns):

```bash
.venv/bin/python scripts/analysis/code15_correlation_report.py
```

## Notebook: model comparison

Heatmap + radar summary (reads `analysis/embeddings_probe/test_scores.csv` and `analysis/embeddings_probe/embedding_metrics.csv`):

- `scripts/analysis/notebooks/model_comparison.ipynb`

## One-shot runner

Run everything end-to-end for a single checkpoint (local path or W&B artifact URI):

```bash
.venv/bin/python scripts/analysis/run_probe_pipeline.py \
  --checkpoint "ag-lukassen/ecg-chagas-embeddings-cli/model-oxpihakx:v12" \
  --run_id "t1-exp01-bp-bce-rot10" \
  --track t1 \
  --preprocessing bp \
  --out_dir ./analysis/embeddings_probe \
  --plots
```

Or run all runs from an existing TOML registry:

```bash
.venv/bin/python scripts/analysis/run_probe_pipeline.py \
  --run_specs configs/analysis/embeddings_probe_runs.toml \
  --out_dir ./analysis/embeddings_probe
```

## Ranking agreement (full test set)

See `scripts/analysis/RANKING_AGREEMENT.md` for the detailed description of:

- full-test logits persistence,
- model×model agreement matrices (Spearman / IoU / Kendall τ),
- per-sample screening consensus (`c_i`).
