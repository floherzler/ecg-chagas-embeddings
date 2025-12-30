import torch

from ecg_chagas_embeddings.data.augmentation import ECGAugmentation


def _sample_signal(length: int = 400) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(12, length)


def _val_augmenter() -> ECGAugmentation:
    return ECGAugmentation(
        crop_size=300,
        n_views=2,
        mask_prob=0.2,
        max_mask_duration=80,
        gaussian_noise_std=0.01,
        scaling=(0.95, 1.05),
        max_time_warp=0.01,
        mode="val",
        base_seed=1234,
    )


def test_val_views_are_deterministic_per_key():
    signal = _sample_signal()
    augmenter = _val_augmenter()

    first = augmenter(signal, key="exam-42")
    second = augmenter(signal, key="exam-42")

    assert first.shape[0] == 2
    assert torch.allclose(first, second)
    assert torch.allclose(first[0], second[0])
    assert torch.allclose(first[1], second[1])


def test_val_augmented_view_differs_across_keys():
    signal = _sample_signal()
    augmenter = _val_augmenter()

    first = augmenter(signal, key="exam-42")
    second = augmenter(signal, key="exam-43")

    assert not torch.allclose(first[1], second[1])


def test_train_views_remain_stochastic():
    signal = _sample_signal()
    augmenter = ECGAugmentation(
        crop_size=300,
        n_views=2,
        mask_prob=0.3,
        max_mask_duration=60,
        gaussian_noise_std=0.05,
        scaling=(0.9, 1.1),
        mode="train",
        base_seed=2024,
    )

    first = augmenter(signal)
    second = augmenter(signal)

    assert first.shape == second.shape
    assert not torch.allclose(first, second)
