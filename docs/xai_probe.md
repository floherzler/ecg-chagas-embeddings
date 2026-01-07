# XAI Probe (STFT/DFT “Virtual Inspection Layer”) for ECG Runs

This project includes an optional **XAI probe** that explains model behavior in **frequency / time–frequency space** without changing training: it computes **time-domain LRP relevances** via `zennit` and then propagates them to **DFT / STDFT space** using the upstream **DFT-LRP** implementation from the paper (“virtual inspection layers”).

The intent matches the “virtual inspection layer” idea (Montavon et al.): interpret relevance in an invertible transform domain (here: STFT magnitude), without inserting new layers or retraining.

## What It Does

For a fixed **probe subset** of the validation set:

1. Compute time-domain relevance per sample and lead via LRP for `pos_logit`, shape `[B, 12, 2500]`.
2. Propagate relevance through a (S)DFT “virtual inspection layer” (DFT-LRP) to obtain relevance in frequency and (optionally) time–frequency.
3. Aggregate frequency-domain relevance mass into configured bands and log **band relevance fractions**.
4. Log additional small summary metrics:
   - `xai/near_cutoff_ratio` (Clever-Hans proxy)
   - `xai/lead_entropy` (lead concentration)
   - `xai/total_mass` and `xai/grad_norm` (sanity)

The existing UMAP logging (4746 examples) stays unchanged; the probe is separate and small by design.

## How To Enable

The probe is implemented as a Lightning callback:

- `src/ecg_chagas_embeddings/callbacks/xai_probe.py`
- callback class: `ecg_chagas_embeddings.callbacks.xai_probe.XAIProbeCallback`

Requirements:
- The upstream DFT-LRP submodule at `external/dft-lrp` (run `git submodule update --init --recursive`)
- Python dependency: `zennit` (add via `uv add zennit`)

You can enable it by adding the callback to your trainer config. A ready-to-use config snippet exists at:

- `configs/xai_probe.yaml`

Example invocation (depending on how you run LightningCLI in your environment):

```bash
python main.py fit --config configs/base.yaml --config configs/track1.yaml --config configs/xai_probe.yaml
```

## Probe Subset Selection (Reproducible)

The callback builds the probe subset once using the validation dataset’s `metadata`:

- Stratified selection by `chagas` label (`n_pos`, `n_neg`)
- Deterministic sampling with `seed`

The selected IDs are persisted to:

- `artifacts/xai_probe/xai_probe_ids.json` (inside the run directory)

You can also provide your own IDs:

- `probe_ids_path`: `.txt`/`.csv`-like (one id per line, or comma/space-separated) or `.json` list

## What Gets Logged

### Band relevance fractions (pooled over leads)

For each band `[low, high)`:

- `xai/rel_frac_<low>_<high>`
- `xai_pos/rel_frac_<low>_<high>` (only positive-labeled probe samples)
- `xai_neg/rel_frac_<low>_<high>` (only negative-labeled probe samples)

Band fractions are normalized by total **DFT-LRP relevance** mass in `[min_band_low, freq_max_hz)`.

### Additional metrics

- `xai/near_cutoff_ratio`: `rel_frac_[0.67–2] / (rel_frac_[5–15] + eps)` if those exact bands exist
- `xai/lead_entropy`: normalized entropy over per-lead attribution mass (`0` = single-lead focus, `1` = uniform)
- `xai/total_mass`: mean total frequency-domain relevance mass in the covered frequency range

### Per-sample CSV (for later analysis)

By default the callback writes a small CSV each time it runs:

- `artifacts/xai_probe/xai_probe_epoch_XXXX.csv`

Columns include: `sample_id`, `label`, per-band fractions, `lead_entropy`, `near_cutoff_ratio`, `total_mass`, `grad_norm`.

If you want W&B artifacts as well, set:

- `log_wandb_artifact: true`

## Defaults / Performance Notes

- Uses DFT-LRP (LRP in time domain + relevance propagation through DFT weights)
- Schedule: runs on epoch 0 and then every `every_n_epochs`
- Uses `torch.autocast(..., enabled=False)` and runs outside Lightning’s `inference_mode` to allow gradients.

Time–frequency visualizations (`compute_timefreq=true`) can be memory-heavy for `T=2500`; keep it off for routine runs and enable only when you need plots.

## Notebook (Sanity Check)

See:

- `src/ecg_chagas_embeddings/notebooks/xai_probe_sanity.ipynb`

It includes a cell where you can paste a W&B run URL or artifact reference, download a checkpoint, and run the probe quickly.
