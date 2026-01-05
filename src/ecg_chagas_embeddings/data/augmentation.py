import hashlib
from typing import Callable, List, Literal, Optional, Tuple, cast

import torch
import torch.nn.functional as F


class RandomAugmentation:
    """Parent class for handling randomness in augmentations."""

    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        self.rng = torch.Generator()
        init_seed = (
            int(seed)
            if seed is not None
            else int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
        )
        self.rng.manual_seed(init_seed)

    def _resolve_generator(
        self, generator: Optional[torch.Generator]
    ) -> torch.Generator:
        return generator if generator is not None else self.rng

    def random_uniform(self, low, high, generator: Optional[torch.Generator] = None):
        """Generates a random float between `low` and `high`."""
        rng = self._resolve_generator(generator)
        return torch.empty(1).uniform_(low, high, generator=rng).item()

    def random_int(self, low, high, generator: Optional[torch.Generator] = None):
        """Generates a random integer between `low` and `high - 1`."""
        rng = self._resolve_generator(generator)
        return torch.randint(low, high, (1,), generator=rng).item()

    def random_mask(
        self, p, shape, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Generates a random mask with probability `p`."""
        rng = self._resolve_generator(generator)
        return (torch.rand(shape, generator=rng) > p).int()


class RandomCropOrPad(RandomAugmentation):
    """Crops or pads the ECG signal to a target length (sync across views)."""

    def __init__(self, target_length, seed=None):
        super().__init__(seed)
        self.target_length = target_length

    def __call__(
        self, signal: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
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
            left_pad = self.random_int(0, padding + 1, generator=generator)
            right_pad = padding - left_pad
            # F.pad pads the last dimension when given a 2-tuple
            return F.pad(signal, (left_pad, right_pad))

        # Crop (same start for all views)
        start = self.random_int(0, N - self.target_length + 1, generator=generator)
        end = start + self.target_length
        if signal.dim() == 3:  # [V, C, N]
            return signal[:, :, start:end]
        else:  # [C, N]
            return signal[:, start:end]


class VCGFrontalAxisRotation(RandomAugmentation):
    """
    Approximate a frontal-plane axis rotation by:
      12-lead (subset [I, II, V1..V6]) -> Frank XYZ (linear) -> rotate about Z -> map back.

    Notes:
    - This is a plausible linear-model augmentation (not a strict physiological simulator).
    - Intended to be applied on *linear* signals (e.g., bandpassed) before nonlinear clipping
      and per-lead normalization.
    - We preserve the original null-space residual so angle=0° reproduces the input exactly.
    """

    # Assumes standard 12-lead channel order:
    # [I, II, III, aVR/AVR, aVL/AVL, aVF/AVF, V1, V2, V3, V4, V5, V6]
    _I = 0
    _II = 1
    _III = 2
    _AVR = 3
    _AVL = 4
    _AVF = 5
    _V1 = 6
    _V2 = 7
    _V3 = 8
    _V4 = 9
    _V5 = 10
    _V6 = 11

    def __init__(
        self,
        max_abs_deg: float = 15.0,
        p: float = 1.0,
        *,
        per_view: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__(seed)
        self.max_abs_deg = float(max_abs_deg)
        self.p = float(p)
        self.per_view = bool(per_view)

        # Wikipedia / Dower-like approximation matching the notebook:
        # L := [I, II, V1, V2, V3, V4, V5, V6]^T  ->  XYZ := A @ L
        A = torch.tensor(
            [
                [-0.156, 0.010, 0.172, 0.074, -0.122, -0.231, -0.239, -0.194],  # X
                [-0.227, 0.887, 0.057, -0.019, -0.106, -0.022, 0.041, 0.048],  # Y
                [-0.022, -0.102, 0.229, 0.310, 0.246, 0.063, -0.055, -0.108],  # Z
            ],
            dtype=torch.float32,
        )  # [3,8]
        self._A = A
        self._A_pinv = torch.linalg.pinv(A)  # [8,3]

    def __call__(
        self, signal: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Args:
            signal: [C, N] or [V, C, N]
        Returns:
            Same shape as input.
        """
        if self.p <= 0 or self.max_abs_deg <= 0:
            return signal

        if signal.dim() not in (2, 3):
            raise ValueError(f"Expected [C,N] or [V,C,N], got {tuple(signal.shape)}")

        rng = self._resolve_generator(generator)
        if torch.rand((), generator=rng).item() > self.p:
            return signal

        squeeze = False
        if signal.dim() == 2:
            x = signal.unsqueeze(0)
            squeeze = True
        else:
            x = signal

        V, C, _ = x.shape
        if C < 12:
            raise ValueError(f"Expected 12 leads, got C={C}")

        # Sample rotation angles (radians).
        max_rad = float(self.max_abs_deg) * 3.141592653589793 / 180.0
        if self.per_view and V > 1:
            theta = (torch.rand((V,), generator=rng) * 2.0 - 1.0) * max_rad
        else:
            theta0 = (torch.rand((1,), generator=rng) * 2.0 - 1.0) * max_rad
            theta = theta0.expand(V)

        theta = theta.to(device=x.device, dtype=torch.float32)
        c = torch.cos(theta).view(V, 1, 1)
        s = torch.sin(theta).view(V, 1, 1)

        A = self._A.to(device=x.device, dtype=x.dtype)
        A_pinv = self._A_pinv.to(device=x.device, dtype=x.dtype)

        I = x[:, self._I, :]  # noqa: E741
        II = x[:, self._II, :]
        V1 = x[:, self._V1, :]
        V2 = x[:, self._V2, :]
        V3 = x[:, self._V3, :]
        V4 = x[:, self._V4, :]
        V5 = x[:, self._V5, :]
        V6 = x[:, self._V6, :]

        L = torch.stack([I, II, V1, V2, V3, V4, V5, V6], dim=1)  # [V,8,N]

        XYZ = torch.einsum("ij,vjn->vin", A, L)  # [V,3,N]
        L_proj = torch.einsum("ij,vjn->vin", A_pinv, XYZ)  # [V,8,N]
        residual = L - L_proj

        X = XYZ[:, 0:1, :]
        Y = XYZ[:, 1:2, :]
        Z = XYZ[:, 2:3, :]

        Xr = c * X - s * Y
        Yr = s * X + c * Y
        XYZr = torch.cat([Xr, Yr, Z], dim=1)

        L_rot = torch.einsum("ij,vjn->vin", A_pinv, XYZr) + residual  # [V,8,N]

        out = x.clone()
        out[:, self._I, :] = L_rot[:, 0, :]
        out[:, self._II, :] = L_rot[:, 1, :]
        out[:, self._V1, :] = L_rot[:, 2, :]
        out[:, self._V2, :] = L_rot[:, 3, :]
        out[:, self._V3, :] = L_rot[:, 4, :]
        out[:, self._V4, :] = L_rot[:, 5, :]
        out[:, self._V5, :] = L_rot[:, 6, :]
        out[:, self._V6, :] = L_rot[:, 7, :]

        # Enforce limb-lead identities for the remaining limb leads.
        I_rot = out[:, self._I, :]
        II_rot = out[:, self._II, :]
        out[:, self._III, :] = II_rot - I_rot
        out[:, self._AVR, :] = -(I_rot + II_rot) / 2.0
        out[:, self._AVL, :] = I_rot - 0.5 * II_rot
        out[:, self._AVF, :] = II_rot - 0.5 * I_rot

        return out.squeeze(0) if squeeze else out


class RandomMaskChannels(RandomAugmentation):
    """Randomly masks a subset of channels in the ECG signal (sync across views)."""

    def __init__(self, mask_prob=0.1, *, apply_prob: float = 1.0, seed=None):
        """
        Args:
            mask_prob (float): Probability of masking a channel (set it to zero).
            apply_prob (float): Probability of applying the augmentation at all.
            seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(seed)
        self.mask_prob = float(mask_prob)
        self.apply_prob = float(apply_prob)

    def __call__(
        self, signal: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Args:
            signal (torch.Tensor): [C, N] or [V, C, N]

        Returns:
            torch.Tensor: Same shape as input with some channels masked to zero.
        """
        rng = self._resolve_generator(generator)
        if self.apply_prob <= 0.0:
            return signal
        if (
            self.apply_prob < 1.0
            and torch.rand((), generator=rng).item() > self.apply_prob
        ):
            return signal
        if signal.dim() == 2:
            # [C, N]
            C = signal.shape[0]
            # 1 = keep, 0 = mask
            keep = (
                torch.rand(C, device=signal.device, generator=rng) > self.mask_prob
            ).to(signal.dtype)
            if keep.sum() == 0:
                # ensure at least one channel remains
                keep[self.random_int(0, C, generator=rng)] = 1.0
            return signal * keep.unsqueeze(1)

        elif signal.dim() == 3:
            # [V, C, N]
            V, C, _ = signal.shape
            keep = (
                torch.rand(C, device=signal.device, generator=rng) > self.mask_prob
            ).to(signal.dtype)
            if keep.sum() == 0:
                keep[self.random_int(0, C, generator=rng)] = 1.0
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

    def __call__(
        self, signal: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Args:
            signal: [C,N] or [V,C,N]
        Returns:
            same rank with time-warp applied.
        """
        if signal.dim() == 2:
            # [C,N] -> [1,C,N] for interpolate
            C, N = signal.shape
            warp = 1.0 + self.random_uniform(
                -self.max_warp, self.max_warp, generator=generator
            )
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
                    warp = 1.0 + self.random_uniform(
                        -self.max_warp, self.max_warp, generator=generator
                    )
                    xv = signal[v : v + 1]  # [1,C,N]
                    outs.append(self._warp_once(xv, warp, N))  # [1,C,N]
                return torch.cat(outs, dim=0)  # [V,C,N]
            else:
                # same warp across all views
                warp = 1.0 + self.random_uniform(
                    -self.max_warp, self.max_warp, generator=generator
                )
                return self._warp_once(signal, warp, N)  # [V,C,N]

        else:
            raise ValueError(f"Expected [C,N] or [V,C,N], got {tuple(signal.shape)}")


class TimeMasking(RandomAugmentation):
    """Randomly masks a time segment of the ECG signal (sync across views)."""

    def __init__(self, max_mask_duration=50, *, apply_prob: float = 1.0, seed=None):
        """
        Args:
            max_mask_duration (int): Maximum duration (in samples) of the masked segment.
            apply_prob (float): Probability of applying the augmentation at all.
            seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(seed)
        self.max_mask_duration = max_mask_duration
        self.apply_prob = float(apply_prob)

    def __call__(
        self, signal: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Args:
            signal (torch.Tensor): [C, N] or [V, C, N]

        Returns:
            torch.Tensor: Time-masked signal with same rank/shape except masked segment.
        """
        rng = self._resolve_generator(generator)
        if self.apply_prob <= 0.0:
            return signal
        if (
            self.apply_prob < 1.0
            and torch.rand((), generator=rng).item() > self.apply_prob
        ):
            return signal
        if signal.dim() == 2:
            # [C, N]
            N = signal.shape[1]
            L = self.random_int(
                1, min(self.max_mask_duration, N) + 1, generator=generator
            )
            start = self.random_int(0, N - L + 1, generator=generator)
            signal[:, start : start + L] = 0.0
            return signal

        elif signal.dim() == 3:
            # [V, C, N]
            N = signal.shape[2]
            L = self.random_int(
                1, min(self.max_mask_duration, N) + 1, generator=generator
            )
            start = self.random_int(0, N - L + 1, generator=generator)
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

    def __call__(
        self, signal: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Args:
            signal (torch.Tensor): [C, N] or [V, C, N]

        Returns:
            torch.Tensor: Amplitude-scaled signal with same shape as input.
        """
        if signal.dim() == 2:
            # [C, N] — single view
            scale = self.random_uniform(
                self.min_scale, self.max_scale, generator=generator
            )
            return signal * float(scale)

        elif signal.dim() == 3:
            # [V, C, N] — multiple views
            V = signal.shape[0]
            if self.per_view:
                # independent scale per view (list of V scalars from class RNG)
                scales = [
                    float(
                        self.random_uniform(
                            self.min_scale, self.max_scale, generator=generator
                        )
                    )
                    for _ in range(V)
                ]
                scales = torch.tensor(
                    scales, dtype=signal.dtype, device=signal.device
                ).view(V, 1, 1)
            else:
                # same scale for all views
                s = float(
                    self.random_uniform(
                        self.min_scale, self.max_scale, generator=generator
                    )
                )
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

    def __call__(
        self, signal: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Args:
            signal (torch.Tensor): [C, N] or [V, C, N]
        Returns:
            torch.Tensor: Same shape as input, with noise added.
        """
        rng = self._resolve_generator(generator)
        if signal.dim() == 2:
            # [C, N]
            noise = torch.empty_like(signal).normal_(
                mean=self.mean, std=self.std, generator=rng
            )
            return signal + noise

        if signal.dim() == 3:
            # [V, C, N]
            V, C, N = signal.shape
            if self.per_view:
                noise = torch.empty_like(signal).normal_(
                    mean=self.mean, std=self.std, generator=rng
                )
            else:
                base = torch.empty(
                    C, N, dtype=signal.dtype, device=signal.device
                ).normal_(mean=self.mean, std=self.std, generator=rng)
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

    def _make_wander(
        self,
        length: int,
        *,
        device,
        dtype,
        generator: Optional[torch.Generator] = None,
    ):
        # Sample amplitude (0..max), frequency within range, and random phase
        amp = self.random_uniform(0.0, self.max_amplitude, generator=generator)
        freq = self.random_uniform(
            self.frequency_range[0], self.frequency_range[1], generator=generator
        )
        phase = self.random_uniform(0.0, 2.0 * float(torch.pi), generator=generator)

        t = torch.arange(length, device=device, dtype=dtype)
        # freq is interpreted as "cycles over this window"
        wander = float(amp) * torch.sin(
            2.0 * torch.pi * float(freq) * t / float(length) + float(phase)
        )
        return wander  # [N]

    def __call__(
        self, signal: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
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
                N, device=signal.device, dtype=signal.dtype, generator=generator
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
                        N,
                        device=signal.device,
                        dtype=signal.dtype,
                        generator=generator,
                    )  # [N]
                    wanders.append(w)
                wander = torch.stack(wanders, dim=0).unsqueeze(1)  # [V,1,N]
                return signal + wander  # broadcast over channels
            else:
                # shared wandering across all views
                wander = self._make_wander(
                    N, device=signal.device, dtype=signal.dtype, generator=generator
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

    def __call__(
        self, x: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        # duplicate upfront (simple & robust)
        if self.n_views > 1 and x.dim() == 2:
            x = torch.stack([x.clone() for _ in range(self.n_views)], dim=0)  # [V,C,N]
        for aug in self.augs:
            x = aug(x, generator=generator)
        return x


def _stable_hash_to_int(key: str) -> int:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


_MAX_SEED_VALUE = 2**63 - 1


class ECGAugmentation:
    def __init__(
        self,
        crop_size: int = 2500,
        n_views: int = 2,
        axis_rotation_max_deg: Optional[float] = None,
        axis_rotation_prob: float = 1.0,
        max_time_warp: Optional[float] = None,  # e.g. 0.005–0.01; None = off
        scaling: Optional[Tuple[float, float]] = None,  # e.g. (0.98, 1.02)
        gaussian_noise_std: Optional[float] = None,  # e.g. 0.003
        wandering_max_amplitude: Optional[float] = None,  # usually None if you bandpass
        wandering_frequency_range: Optional[Tuple[float, float]] = None,
        max_mask_duration: Optional[int] = None,  # e.g. 60–100 samples @ 400 Hz
        time_mask_apply_prob: float = 1.0,
        mask_prob: Optional[float] = None,  # e.g. 0.02–0.05
        channel_mask_apply_prob: float = 1.0,
        # Optional toggles (keep simple defaults):
        per_view_noise: bool = True,
        per_view_scaling: bool = True,
        per_view_warp: bool = False,  # keep intervals aligned across views
        per_view_wandering: bool = False,  # keep shared if you enable wandering
        per_view_axis_rotation: bool = True,
        *,
        mode: Literal["train", "val"] = "train",
        base_seed: int = 42,
        val_anchor_clean: bool = True,
    ):
        """
        Returns two views for both training and validation.
        Validation is deterministic per sample (view0 anchor, view1 augmented) using
        seeds derived from `base_seed` and a stable hash of the provided key.
        Order: Crop -> (optional) AxisRotation -> (optional) TimeMask -> (optional) ChannelMask -> Noise -> Scaling -> Warp -> Wander
        """
        if n_views != 2:
            raise ValueError("ECGAugmentation currently supports n_views=2.")

        mode_str = str(mode)
        if mode_str not in ("train", "val"):
            raise ValueError(f"mode must be 'train' or 'val', got {mode!r}")
        self.mode: Literal["train", "val"] = cast(Literal["train", "val"], mode_str)
        self.n_views = int(n_views)
        self.base_seed = int(base_seed)
        self.val_anchor_clean = bool(val_anchor_clean)

        # --- Shared content frame first (same for all views by aug design) ---
        self.crop = RandomCropOrPad(crop_size)

        post_crop_augs: List[Callable[[torch.Tensor], torch.Tensor]] = []

        if (
            axis_rotation_max_deg is not None
            and axis_rotation_max_deg > 0
            and axis_rotation_prob > 0
        ):
            post_crop_augs.append(
                VCGFrontalAxisRotation(
                    max_abs_deg=float(axis_rotation_max_deg),
                    p=float(axis_rotation_prob),
                    per_view=bool(per_view_axis_rotation),
                )
            )
        if max_mask_duration is not None:
            post_crop_augs.append(
                TimeMasking(
                    max_mask_duration,
                    apply_prob=float(time_mask_apply_prob),
                )
            )

        if mask_prob is not None:
            post_crop_augs.append(
                RandomMaskChannels(
                    mask_prob,
                    apply_prob=float(channel_mask_apply_prob),
                )
            )

        # --- Per-view appearance tweaks (your aug classes handle [V,C,N]) ---
        if gaussian_noise_std is not None:
            post_crop_augs.append(
                GaussianNoise(std=gaussian_noise_std, per_view=per_view_noise)
            )

        if scaling is not None:
            min_scale, max_scale = scaling
            post_crop_augs.append(
                AmplitudeScaling(min_scale, max_scale, per_view=per_view_scaling)
            )

        if max_time_warp is not None:
            post_crop_augs.append(
                TimeWarping(max_warp=max_time_warp, per_view=per_view_warp)
            )

        if (wandering_max_amplitude is not None) and (
            wandering_frequency_range is not None
        ):
            post_crop_augs.append(
                RandomWandering(
                    max_amplitude=wandering_max_amplitude,
                    frequency_range=wandering_frequency_range,
                    per_view=per_view_wandering,
                )
            )

        self.post_crop_augs = post_crop_augs

        # Train transform: stochastic, two independent views.
        self.train_transform = Compose(
            self.crop, *self.post_crop_augs, n_views=self.n_views
        )

        # Validation transforms: deterministic per-view seeds.
        anchor_augs = (
            [self.crop] if self.val_anchor_clean else [self.crop, *self.post_crop_augs]
        )
        self.anchor_transform = Compose(*anchor_augs, n_views=1)
        self.val_view_transform = Compose(self.crop, *self.post_crop_augs, n_views=1)

    def _compute_seed(self, key: str, view_index: int) -> int:
        base = int(self.base_seed) + _stable_hash_to_int(key) + 1000 * int(view_index)
        return base % _MAX_SEED_VALUE

    @staticmethod
    def _make_generator(seed: int, device: torch.device) -> torch.Generator:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        return gen

    def __call__(
        self, signal: torch.Tensor, *, key: Optional[str] = None
    ) -> torch.Tensor:
        """
        Args:
            signal: [C, N]
            key: string used for deterministic seeding in validation mode (e.g., exam_id).
        Returns:
            [2, C, N] stacked views.
        """
        if not isinstance(signal, torch.Tensor):
            signal = torch.as_tensor(signal, dtype=torch.float32)

        if self.mode == "train":
            # stochastic each call; relies on internal RNG state of augmentations
            return self.train_transform(signal)

        if key is None:
            raise ValueError(
                "Validation mode requires a stable `key` (e.g., exam_id) for deterministic augmentations."
            )

        anchor_seed = self._compute_seed(str(key), view_index=0)
        aug_seed = self._compute_seed(str(key), view_index=1)
        device = signal.device

        anchor_gen = self._make_generator(anchor_seed, device=device)
        aug_gen = self._make_generator(aug_seed, device=device)

        # Clone inputs so that in-place ops do not leak across views.
        anchor_input = signal.clone()
        view_input = signal.clone()

        anchor_view = self.anchor_transform(anchor_input, generator=anchor_gen)
        aug_view = self.val_view_transform(view_input, generator=aug_gen)

        if anchor_view.dim() == 3:
            anchor_view = anchor_view.squeeze(0)
        if aug_view.dim() == 3:
            aug_view = aug_view.squeeze(0)

        return torch.stack([anchor_view, aug_view], dim=0)
