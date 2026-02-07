# Results Pipeline Checklist (4-Stage)

Legend:
- [x] present in Results
- [~] computed but not integrated (or weakly integrated)
- [ ] missing compute/decision for narrative

## Stage 1: Task Utility

- [x] Per-run clinical table included (`thesis/chapters/04_results.tex:13`, `thesis/tables/classification_per_run.tex`).
- [x] Preprocessing effect narrative included (`thesis/chapters/04_results.tex:15`-`thesis/chapters/04_results.tex:24`).
- [x] Rotation effect narrative included (`thesis/chapters/04_results.tex:21`-`thesis/chapters/04_results.tex:24`).
- [~] TPR@5 by preprocessing overview figure currently commented out (table/text carry the argument).
- [~] Verified vs CODE15 split: table is included in main text; the overview split figure is currently commented out.
- [x] CODE15 quality confounder tables included (`thesis/chapters/04_results.tex:52`-`thesis/chapters/04_results.tex:90`).
- [~] Group-level summary table exists but only appendix (`thesis/tables/classification_by_group.tex`, `thesis/chapters/90_appendix.tex:6`).
- [ ] Decide whether to move one compact group-level result into main Results for faster takeaways.

## Stage 2: Embedding Health

- [x] Per-run embedding table included (`thesis/chapters/04_results.tex:98`, `thesis/tables/embedding_per_run.tex`).
- [x] Main embedding-health utility figure included (currently CAC$_1$ vs TPR@5\%; GPU view kept in appendix).
- [x] PCA correlation by group table included (`thesis/chapters/04_results.tex:116`, `thesis/tables/pca_corr_by_group.tex`).
- [~] `embedding_by_group.tex` computed but only appendix (`thesis/tables/embedding_by_group.tex`, `thesis/chapters/90_appendix.tex:7`).
- [x] Qualitative embedding section is filled with concrete UMAP results and interpretation.
- [x] Embedding callouts are present (main UMAP triplets + appendix references for additional panels).

## Stage 3: Cross-Model Consistency and Consensus Selection

- [x] Spearman heatmap included (`thesis/chapters/04_results.tex:133`-`thesis/chapters/04_results.tex:138`).
- [x] RRA dataset plot + KW table included (`thesis/chapters/04_results.tex:214`-`thesis/chapters/04_results.tex:233`).
- [x] Probe-overlap table included (`thesis/chapters/04_results.tex:152`-`thesis/chapters/04_results.tex:165`).
- [x] IoU heatmap is now included in main text side-by-side with Spearman.
- [x] Kendall/IoU/Spearman are all discussed quantitatively in Stage 3 (with pairwise summary table).
- [x] Sample-consensus thresholds are integrated with a dedicated coverage table (no placeholder remains).
- [x] Agreement-structure narrative is filled (including regime-clustering and bp-sc-norm instability).
- [x] Consensus threshold policy is explicitly stated (e.g., \(c_i\ge 8\), \(c_i\ge 16\)).
- [x] Candidate availability tables for XAI are included under `xai_summary/lead_7` outputs.

## Stage 4: Plausibility with XAI

- [x] Lead-mass table + 4 condition heatmaps included (`thesis/chapters/04_results.tex:179`-`thesis/chapters/04_results.tex:203`).
- [x] Per-run ST-DFT-LRP 4x4 plot assets exist (including lead_7 variants) in `analysis/embeddings_probe/xai_summary/plots/`.
- [x] V2/lead_7 focus is integrated in Stage 4 text, paths, and availability tables.
- [~] `summarize_stdftlrp_beat_agg.py` outputs (`stdftlrp_summary_per_run.csv`, `stdftlrp_summary_by_model_group.csv`) are present but not explicitly tied into chapter narrative.
- [x] ST-DFT-LRP methodological interpretation section is filled (sample-level sanity + aggregated 4x4 interpretation).
- [x] Results are consistently written against `lead_7` artifacts for Stage 4.

## Pipeline Wiring Notes (V2)

- `run_probe_pipeline.py --stdftlrp_lead_index 7` does pass through to per-run compute (`run_probe_pipeline.py` -> `run_stdftlrp_pipeline.py` -> `compute_stdftlrp_beat_aggregates.py`).
- `summarize_stdftlrp_beat_agg.py` is not called by `run_probe_pipeline.py`.
- `summarize_stdftlrp_beat_agg.py` currently auto-prefers `lead_1` if present (`scripts/analysis/summarize_stdftlrp_beat_agg.py:101`-`scripts/analysis/summarize_stdftlrp_beat_agg.py:103`), and has no lead flag.
