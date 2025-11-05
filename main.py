from __future__ import annotations

import sys
from pathlib import Path

from lightning.pytorch.cli import LightningCLI

# Allow running the script without installing the package by adding the src layout to PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ecg_chagas_embeddings.models.resnet18_ecg_flex import LitResNet18  # noqa: E402
from ecg_chagas_embeddings.data.datamodule import ECGDataModule  # noqa: E402


def main():
    # use LightningCLI for hyperparameter management
    LightningCLI(
        model_class=LitResNet18,
        datamodule_class=ECGDataModule,
        save_config_callback=None,
        seed_everything_default=42,
        run=True,
    )

    print("Hello imported world!")


if __name__ == "__main__":
    main()
