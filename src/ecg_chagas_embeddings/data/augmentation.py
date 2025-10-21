from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class RandomAugmentation:
    """Parent class for handling randomness in augmentations."""

    def __init__(self, seed=None):
        """
        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def random_uniform(self, low, high):
        """Generates a random float between `low` and `high`."""
        return torch.empty(1).uniform_(low, high, generator=self.rng).item()

    def random_int(self, low, high):
        """Generates a random integer between `low` and `high - 1`."""
        return torch.randint(low, high, (1,), generator=self.rng).item()

    def random_mask(self, p, shape):
        """Generates a random mask with probability `p`."""
        return (torch.rand(shape, generator=self.rng) > p).int()


class RandomCropOrPad(RandomAugmentation):
    """Crops or pads the ECG signal to a target length (sync across views)."""

    def __init__(self, target_length, seed=None):
        super().__init__(seed)
        self.target_length = target_length

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal: [C, N] or [V, C, N]
        Returns:
            [C, target_length] or [V, C, target_length]
        """
        if signal.dim() not in (2, 3):
            raise ValueError(f"Expected [C,N] or [V,C,N], got {tuple(signal.shape)}")

        N = signal.shape[-1]

        # Exact length: no-op
        if N == self.target_length:
            return signal

        # Pad (same left/right for all views)
        if N < self.target_length:
            padding = self.target_length - N
            left_pad = self.random_int(0, padding + 1)
            right_pad = padding - left_pad
            # F.pad pads the last dimension when given a 2-tuple
            return F.pad(signal, (left_pad, right_pad))

        # Crop (same start for all views)
        start = self.random_int(0, N - self.target_length + 1)
        end = start + self.target_length
        if signal.dim() == 3:  # [V, C, N]
            return signal[:, :, start:end]
        else:  # [C, N]
            return signal[:, start:end]


class RandomMaskChannels(RandomAugmentation):
    """Randomly masks a subset of channels in the ECG signal (sync across views)."""

    def __init__(self, mask_prob=0.1, seed=None):
        """
        Args:
            mask_prob (float): Probability of masking a channel (set it to zero).
            seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(seed)
        self.mask_prob = float(mask_prob)

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal (torch.Tensor): [C, N] or [V, C, N]

        Returns:
            torch.Tensor: Same shape as input with some channels masked to zero.
        """
        if signal.dim() == 2:
            # [C, N]
            C = signal.shape[0]
            # 1 = keep, 0 = mask
            keep = (torch.rand(C, device=signal.device) > self.mask_prob).to(
                signal.dtype
            )
            if keep.sum() == 0:
                # ensure at least one channel remains
                keep[self.random_int(0, C)] = 1.0
            return signal * keep.unsqueeze(1)

        elif signal.dim() == 3:
            # [V, C, N]
            V, C, _ = signal.shape
            keep = (torch.rand(C, device=signal.device) > self.mask_prob).to(
                signal.dtype
            )
            if keep.sum() == 0:
                keep[self.random_int(0, C)] = 1.0
            # same channel mask across all views
            return signal * keep.view(1, C, 1)

        else:
            raise ValueError(f"Expected [C,N] or [V,C,N], got {tuple(signal.shape)}")


class TimeWarping(RandomAugmentation):
    """Randomly stretches or compresses the ECG in time.

    For ECG, prefer tiny max_warp (e.g., 0.005–0.01). By default, the same warp
    is applied across all views to preserve alignment of intervals.
    """

    def __init__(self, max_warp=0.2, seed=None, per_view: bool = False):
        """
        Args:
            max_warp (float): Maximum fractional warp (0.01 = ±1%).
            seed (int, optional): RNG seed.
            per_view (bool): If True, each view gets its own warp. Default False.
        """
        super().__init__(seed)
        self.max_warp = float(max_warp)
        self.per_view = bool(per_view)

    def _warp_once(
        self, x: torch.Tensor, warp_factor: float, orig_len: int
    ) -> torch.Tensor:
        """x: [B,C,L] (B can be 1 or V). Returns [B,C,orig_len]."""
        new_len = max(1, int(round(orig_len * warp_factor)))
        x_res = F.interpolate(x, size=new_len, mode="linear", align_corners=False)
        if new_len >= orig_len:
            return x_res[:, :, :orig_len]
        # right-pad with zeros to match original length
        out = torch.zeros(
            x.size(0), x.size(1), orig_len, dtype=x.dtype, device=x.device
        )
        out[:, :, :new_len] = x_res
        return out

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal: [C,N] or [V,C,N]
        Returns:
            same rank with time-warp applied.
        """
        if signal.dim() == 2:
            # [C,N] -> [1,C,N] for interpolate
            C, N = signal.shape
            warp = 1.0 + self.random_uniform(-self.max_warp, self.max_warp)
            x = signal.unsqueeze(0)  # [1,C,N]
            y = self._warp_once(x, warp, N).squeeze(0)  # [C,N]
            return y

        elif signal.dim() == 3:
            # [V,C,N]
            V, C, N = signal.shape
            if self.per_view:
                # different warp per view
                outs = []
                for v in range(V):
                    warp = 1.0 + self.random_uniform(-self.max_warp, self.max_warp)
                    xv = signal[v : v + 1]  # [1,C,N]
                    outs.append(self._warp_once(xv, warp, N))  # [1,C,N]
                return torch.cat(outs, dim=0)  # [V,C,N]
            else:
                # same warp across all views
                warp = 1.0 + self.random_uniform(-self.max_warp, self.max_warp)
                return self._warp_once(signal, warp, N)  # [V,C,N]

        else:
            raise ValueError(f"Expected [C,N] or [V,C,N], got {tuple(signal.shape)}")


class TimeMasking(RandomAugmentation):
    """Randomly masks a time segment of the ECG signal (sync across views)."""

    def __init__(self, max_mask_duration=50, seed=None):
        """
        Args:
            max_mask_duration (int): Maximum duration (in samples) of the masked segment.
            seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(seed)
        self.max_mask_duration = max_mask_duration

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal (torch.Tensor): [C, N] or [V, C, N]

        Returns:
            torch.Tensor: Time-masked signal with same rank/shape except masked segment.
        """
        if signal.dim() == 2:
            # [C, N]
            N = signal.shape[1]
            L = self.random_int(1, min(self.max_mask_duration, N) + 1)
            start = self.random_int(0, N - L + 1)
            signal[:, start : start + L] = 0.0
            return signal

        elif signal.dim() == 3:
            # [V, C, N]
            N = signal.shape[2]
            L = self.random_int(1, min(self.max_mask_duration, N) + 1)
            start = self.random_int(0, N - L + 1)
            signal[:, :, start : start + L] = 0.0
            return signal

        else:
            raise ValueError(f"Expected [C,N] or [V,C,N], got {tuple(signal.shape)}")


class AmplitudeScaling(RandomAugmentation):
    """Randomly scales the amplitude of the ECG signal (per-view by default)."""

    def __init__(self, min_scale=0.8, max_scale=1.2, seed=None, per_view: bool = True):
        """
        Args:
            min_scale (float): Minimum scaling factor.
            max_scale (float): Maximum scaling factor.
            seed (int, optional): Random seed for reproducibility.
            per_view (bool): If True, each view gets its own scale. If False, all
                             views share the same scale. Default: True.
        """
        super().__init__(seed)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.per_view = bool(per_view)

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal (torch.Tensor): [C, N] or [V, C, N]

        Returns:
            torch.Tensor: Amplitude-scaled signal with same shape as input.
        """
        if signal.dim() == 2:
            # [C, N] — single view
            scale = self.random_uniform(self.min_scale, self.max_scale)
            return signal * float(scale)

        elif signal.dim() == 3:
            # [V, C, N] — multiple views
            V = signal.shape[0]
            if self.per_view:
                # independent scale per view (list of V scalars from class RNG)
                scales = [
                    float(self.random_uniform(self.min_scale, self.max_scale))
                    for _ in range(V)
                ]
                scales = torch.tensor(
                    scales, dtype=signal.dtype, device=signal.device
                ).view(V, 1, 1)
            else:
                # same scale for all views
                s = float(self.random_uniform(self.min_scale, self.max_scale))
                scales = torch.tensor(
                    [s] * V, dtype=signal.dtype, device=signal.device
                ).view(V, 1, 1)
            return signal * scales

        else:
            raise ValueError(f"Expected [C,N] or [V,C,N], got {tuple(signal.shape)}")


class GaussianNoise(RandomAugmentation):
    """Adds Gaussian noise to the ECG signal (per-view by default)."""

    def __init__(
        self, mean: float = 0.0, std: float = 0.01, seed=None, per_view: bool = True
    ):
        """
        Args:
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.
            seed (int, optional): Random seed for reproducibility.
            per_view (bool): If True, each view gets independent noise. If False,
                             all views share the same noise realization.
        """
        super().__init__(seed)
        self.mean = float(mean)
        self.std = float(std)
        self.per_view = bool(per_view)

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal (torch.Tensor): [C, N] or [V, C, N]
        Returns:
            torch.Tensor: Same shape as input, with noise added.
        """
        if signal.dim() == 2:
            # [C, N]
            noise = torch.empty_like(signal).normal_(
                mean=self.mean, std=self.std, generator=self.rng
            )
            return signal + noise

        if signal.dim() == 3:
            # [V, C, N]
            V, C, N = signal.shape
            if self.per_view:
                noise = torch.empty_like(signal).normal_(
                    mean=self.mean, std=self.std, generator=self.rng
                )
            else:
                base = torch.empty(
                    C, N, dtype=signal.dtype, device=signal.device
                ).normal_(mean=self.mean, std=self.std, generator=self.rng)
                noise = base.unsqueeze(0).expand(V, -1, -1).contiguous()
            return signal + noise

        raise ValueError(f"Expected [C,N] or [V,C,N], got {tuple(signal.shape)}")


class RandomWandering(RandomAugmentation):
    """Adds low-frequency wandering noise to the ECG signal (shared across views by default)."""

    def __init__(
        self,
        max_amplitude=1.0,
        frequency_range=(0.5, 2.0),
        seed=None,
        per_view: bool = False,
    ):
        """
        Args:
            max_amplitude (float): Maximum amplitude of the wandering noise.
            frequency_range (tuple): Range of wandering frequencies (in cycles over the window).
            seed (int, optional): Random seed for reproducibility.
            per_view (bool): If True, each view gets its own wandering; else all share the same.
        """
        super().__init__(seed)
        self.max_amplitude = float(max_amplitude)
        self.frequency_range = tuple(frequency_range)
        self.per_view = bool(per_view)

    def _make_wander(self, length: int, *, device, dtype):
        # Sample amplitude (0..max), frequency within range, and random phase
        amp = self.random_uniform(0.0, self.max_amplitude)
        freq = self.random_uniform(self.frequency_range[0], self.frequency_range[1])
        phase = self.random_uniform(0.0, 2.0 * float(torch.pi))

        t = torch.arange(length, device=device, dtype=dtype)
        # freq is interpreted as "cycles over this window"
        wander = float(amp) * torch.sin(
            2.0 * torch.pi * float(freq) * t / float(length) + float(phase)
        )
        return wander  # [N]

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal (torch.Tensor): [C, N] or [V, C, N]
        Returns:
            torch.Tensor: Same shape as input with wandering added (lead-synchronous).
        """
        if signal.dim() == 2:
            # [C, N]
            C, N = signal.shape
            wander = self._make_wander(
                N, device=signal.device, dtype=signal.dtype
            )  # [N]
            return signal + wander.unsqueeze(0)  # broadcast over channels

        elif signal.dim() == 3:
            # [V, C, N]
            V, C, N = signal.shape
            if self.per_view:
                # independent wandering per view (still lead-synchronous within a view)
                wanders = []
                for _ in range(V):
                    w = self._make_wander(
                        N, device=signal.device, dtype=signal.dtype
                    )  # [N]
                    wanders.append(w)
                wander = torch.stack(wanders, dim=0).unsqueeze(1)  # [V,1,N]
                return signal + wander  # broadcast over channels
            else:
                # shared wandering across all views
                wander = self._make_wander(
                    N, device=signal.device, dtype=signal.dtype
                )  # [N]
                return signal + wander.view(1, 1, N)  # broadcast over views & channels

        else:
            raise ValueError(f"Expected [C,N] or [V,C,N], got {tuple(signal.shape)}")


class Compose:
    """
    Applies augmentations in sequence.
    If n_views > 1, duplicates the input [C,N] -> [V,C,N] at the start.
    Augmentations must accept either [C,N] or [V,C,N] and return same rank.
    """

    def __init__(self, *augmentations, n_views: int = 1):
        self.augs = list(augmentations)
        self.n_views = int(n_views)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # duplicate upfront (simple & robust)
        if self.n_views > 1 and x.dim() == 2:
            x = torch.stack([x.clone() for _ in range(self.n_views)], dim=0)  # [V,C,N]
        for aug in self.augs:
            x = aug(x)
        return x


class ECGAugmentation:
    def __init__(
        self,
        crop_size: int = 2500,
        n_views: int = 2,
        max_time_warp: Optional[float] = None,  # e.g. 0.005–0.01; None = off
        scaling: Optional[Tuple[float, float]] = None,  # e.g. (0.98, 1.02)
        gaussian_noise_std: Optional[float] = None,  # e.g. 0.003
        wandering_max_amplitude: Optional[float] = None,  # usually None if you bandpass
        wandering_frequency_range: Optional[Tuple[float, float]] = None,
        max_mask_duration: Optional[int] = None,  # e.g. 60–100 samples @ 400 Hz
        mask_prob: Optional[float] = None,  # e.g. 0.02–0.05
        # Optional toggles (keep simple defaults):
        per_view_noise: bool = True,
        per_view_scaling: bool = True,
        per_view_warp: bool = False,  # keep intervals aligned across views
        per_view_wandering: bool = False,  # keep shared if you enable wandering
    ):
        """
        Returns n_views for training (SupCon) and 1 view for validation if n_views=1.
        Order: Crop -> (optional) TimeMask -> (optional) ChannelMask -> Noise -> Scaling -> Warp -> Wander
        """
        augs = []

        # --- Shared content frame first (same for all views by aug design) ---
        augs.append(RandomCropOrPad(crop_size))

        if max_mask_duration is not None:
            augs.append(TimeMasking(max_mask_duration))

        if mask_prob is not None:
            augs.append(RandomMaskChannels(mask_prob))

        # --- Per-view appearance tweaks (your aug classes handle [V,C,N]) ---
        if gaussian_noise_std is not None:
            augs.append(GaussianNoise(std=gaussian_noise_std, per_view=per_view_noise))

        if scaling is not None:
            min_scale, max_scale = scaling
            augs.append(
                AmplitudeScaling(min_scale, max_scale, per_view=per_view_scaling)
            )

        if max_time_warp is not None:
            augs.append(TimeWarping(max_warp=max_time_warp, per_view=per_view_warp))

        if (wandering_max_amplitude is not None) and (
            wandering_frequency_range is not None
        ):
            augs.append(
                RandomWandering(
                    max_amplitude=wandering_max_amplitude,
                    frequency_range=wandering_frequency_range,
                    per_view=per_view_wandering,
                )
            )

        self.transform = Compose(*augs, n_views=n_views)

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        # signal: [C, N]  -> returns [V, C, N] if n_views>1, else [C, N]
        return self.transform(signal)
