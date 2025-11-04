from lightning.pytorch.cli import LightningCLI

from ecg_chagas_embeddings.models.resnet18_ecg_flex import LitResNet18
from ecg_chagas_embeddings.data.datamodule import ECGDataModule


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
