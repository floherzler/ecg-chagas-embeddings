import math

import torch

from ecg_chagas_embeddings.callbacks.xai_probe import (
    compute_dft_band_fractions_from_relevance,
    extract_pos_logit,
    lead_entropy_from_relevance_time,
)


def test_extract_pos_logit_shapes():
    x = torch.randn(3, 12, 100)

    class M1(torch.nn.Module):
        def forward(self, _x):
            return torch.randn(_x.shape[0])

    class M2(torch.nn.Module):
        def forward(self, _x):
            return torch.randn(_x.shape[0], 1)

    class M3(torch.nn.Module):
        def forward(self, _x):
            return torch.randn(_x.shape[0], 2)

    assert extract_pos_logit(M1()(x)).shape == (3,)
    assert extract_pos_logit(M2()(x)).shape == (3,)
    assert extract_pos_logit(M3()(x)).shape == (3,)


def test_dft_band_fractions_sum_to_one_default_bands():
    torch.manual_seed(0)
    B, L, T = 4, 12, 2500
    F = T // 2 + 1

    # Fake frequency-domain relevance (already rfft-binned).
    rel_f = torch.randn(B, L, F)
    bands = [(0.67, 2.0), (2.0, 5.0), (5.0, 15.0), (15.0, 45.0)]

    pooled, per_lead, total_mass = compute_dft_band_fractions_from_relevance(
        relevance_freq=rel_f,
        fs_hz=400.0,
        signal_length=T,
        freq_max_hz=45.0,
        bands_hz=bands,
        per_lead=True,
    )

    assert pooled.shape == (B, len(bands))
    assert per_lead is not None
    assert per_lead.shape == (B, L, len(bands))
    assert total_mass.shape == (B,)
    assert torch.isfinite(pooled).all()

    sums = pooled.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4)


def test_sine_relevance_peaks_in_5_15_band():
    fs = 400.0
    f0 = 10.0
    B, L, T = 2, 12, 2500

    # Build a frequency-domain relevance concentrated at ~10 Hz.
    freqs = torch.fft.rfftfreq(T, d=1.0 / fs)
    k0 = int(torch.argmin((freqs - f0).abs()).item())
    rel_f = torch.zeros(B, L, T // 2 + 1)
    rel_f[:, :, k0] = 1.0

    bands = [(0.67, 2.0), (2.0, 5.0), (5.0, 15.0), (15.0, 45.0)]
    pooled, _, _ = compute_dft_band_fractions_from_relevance(
        relevance_freq=rel_f,
        fs_hz=fs,
        signal_length=T,
        freq_max_hz=45.0,
        bands_hz=bands,
        per_lead=False,
    )

    mean_fracs = pooled.mean(dim=0)
    assert int(mean_fracs.argmax().item()) == 2
    assert float(mean_fracs[2].item()) > 0.9


def test_lead_entropy_bounds():
    torch.manual_seed(0)
    B, L, T = 3, 12, 100
    r = torch.randn(B, L, T)
    h = lead_entropy_from_relevance_time(r)
    assert h.shape == (B,)
    assert torch.isfinite(h).all()
    assert (h >= 0).all()
    assert (h <= 1 + 1e-6).all()

