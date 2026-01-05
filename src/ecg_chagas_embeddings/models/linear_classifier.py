from typing import Optional

import torch
from torch import nn

from lightning import LightningModule


class LinearClassifier(nn.Module):
    """Simple linear classifier with optional dropout."""

    def __init__(
        self, input_size: int = 2048, num_classes: int = 2, p_dropout: float = 0.0
    ):
        """
        Args:
            input_size: Feature dimension of input
            num_classes: Number of classes to predict
            p_dropout: Dropout probability
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=p_dropout), nn.Linear(input_size, num_classes, bias=True)
        )

    def forward(self, x) -> torch.Tensor:
        return self.classifier(x)


class MLPClassifier(nn.Module):
    """MLP classifier with a single hidden layer and BatchNorm."""

    def __init__(
        self,
        input_size: int = 2048,
        hidden_dim_size: int = 2048,
        num_classes: int = 2,
        p_dropout: float = 0.0,
    ):
        """
        Args:
            input_size: Feature dimension of input
            hidden_dim_size: Hidden layer dimension
            num_classes: Number of classes to predict
            p_dropout: Dropout probability
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_dim_size, bias=False),
            nn.BatchNorm1d(hidden_dim_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),
            nn.Linear(hidden_dim_size, num_classes, bias=True),
        )

    def forward(self, x) -> torch.Tensor:
        return self.classifier(x)


class FineTuneClassifier(LightningModule):
    """
    LightningModule for fine-tuning a pre-trained backbone with a classifier head.
    Supports linear or MLP classifiers on top of various backbone architectures.
    """

    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        num_ftrs: int = 2048,
        num_classes: int = 2,
        max_epochs: int = 100,
        lr: float = 1e-3,
        nesterov: bool = False,
        p_dropout: float = 0.0,
        weight_decay: float = 0,
        hidden_dim_size: Optional[int] = None,
        warmup_epochs: int = 0,
        optimizer_name: str = "adam",
        trainable_encoder: bool = False,
        use_backbone: bool = True,
        base_model_path: Optional[str] = None,
        enable_metrics: bool = False,
    ):
        """
        Args:
            base_model: Pre-trained encoder model
            num_ftrs: Feature dimension from encoder
            num_classes: Number of classes to predict
            max_epochs: Maximum number of training epochs
            lr: Learning rate
            nesterov: Whether to use Nesterov momentum (for SGD)
            p_dropout: Dropout probability in classifier
            weight_decay: L2 regularization strength
            hidden_dim_size: If provided, use MLP instead of linear classifier
            warmup_epochs: Number of LR warmup epochs
            optimizer_name: Optimizer to use ("adam" or "sgd")
            trainable_encoder: Whether to train the encoder
            use_backbone: Whether to access the backbone directly
            base_model_path: Path to load base model from
            enable_metrics: Instantiate torchmetrics objects (disable to keep it lightweight)
        """
        super().__init__()

        self.lr = lr
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.p_dropout = p_dropout
        self.nesterov = nesterov
        self.warmup_epochs = warmup_epochs
        self.optimizer_name = optimizer_name
        self.trainable_encoder = trainable_encoder
        self.use_backbone = use_backbone

        if base_model is None and base_model_path is None:
            raise ValueError(
                "Provide a `base_model` instance or `base_model_path` for feature extraction."
            )
        self.base_model = base_model

        if base_model_path is not None:
            ckpt = torch.load(base_model_path, map_location="cpu")
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            if self.base_model is None:
                raise ValueError(
                    "`base_model_path` was given but `base_model` is None."
                )
            self.base_model.load_state_dict(state_dict, strict=False)

        # Initialize classifier
        if hidden_dim_size is None:
            self.classifier = LinearClassifier(
                input_size=num_ftrs, num_classes=num_classes, p_dropout=p_dropout
            )
        else:
            self.classifier = MLPClassifier(
                input_size=num_ftrs,
                hidden_dim_size=hidden_dim_size,
                num_classes=num_classes,
                p_dropout=p_dropout,
            )

        # Metrics are optional; default False to keep this head lightweight.
        self.enable_metrics = enable_metrics

        self.criterion = torch.nn.CrossEntropyLoss()

        # Configure base model training mode
        if not self.trainable_encoder:
            if self.use_backbone:
                for param in self.base_model.parameters():
                    param.requires_grad = False
                self.base_model.eval()
            else:
                self.base_model.eval()

        else:
            self.base_model.train()

    def forward(self, x) -> torch.Tensor:
        """Extract features and classify."""
        if not self.trainable_encoder:
            with torch.no_grad():
                y_hat = self._extract_features(x)
        else:
            y_hat = self._extract_features(x)

        # Flatten features if needed
        y_hat = y_hat.view(y_hat.size(0), -1)

        return self.classifier(y_hat)

    def _extract_features(self, x) -> torch.Tensor:
        """Extract features from the base model."""
        if self.use_backbone:
            out = self.base_model(x)
        else:
            out = self.base_model(x)

        # Handle different return types from various backbones
        if isinstance(out, dict):
            # common keys used in the codebase for representations
            for key in ("feats", "features", "embedding", "embeddings"):
                if key in out:
                    return out[key]
            # fallback: return the first tensor value
            for val in out.values():
                if torch.is_tensor(val):
                    return val
            return out

        if isinstance(out, (tuple, list)):
            # LitResNet18 returns (feats, proj, logits); keep the first element
            if len(out) > 0:
                return out[0]
            raise ValueError("Backbone returned an empty tuple/list.")

        return out

    def _unpack_batch(self, batch):
        """
        Accept both tuple-style batches (x, y) and dict batches produced by the
        existing ECG datamodule. For dicts, we prefer the raw signal under
        'ecg', falling back to the first view in 'ecg_views'.
        """
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        elif isinstance(batch, dict):
            x = batch.get("ecg")
            if x is None and "ecg_views" in batch:
                views = batch["ecg_views"]
                # expected shape [B, V, C, T]; take view 0
                x = views[:, 0] if views.dim() == 4 else views
            y = batch.get("chagas", batch.get("labels", batch.get("y")))
            if y is None or x is None:
                raise ValueError(
                    "Batch dict must contain 'ecg' or 'ecg_views' and 'chagas'/'labels'/'y'."
                )
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        # Flatten labels to [B]
        if isinstance(y, torch.Tensor):
            y = y.view(-1).long()
        return x, y

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training step."""
        x, y = self._unpack_batch(batch)

        preds = self.forward(x)
        loss = self.criterion(preds, y)

        # Log metrics
        self.log(
            "train.loss.epoch",
            loss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log("train.loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = self._unpack_batch(batch)

        preds = self.forward(x)
        loss = self.criterion(preds, y)
        probs = preds.softmax(-1)  # noqa: F841

        # Log metrics
        self.log("val.loss", loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y = self._unpack_batch(batch)

        preds = self.forward(x)
        loss = self.criterion(preds, y)
        probs = preds.softmax(-1)  # noqa: F841

        # Log metrics
        self.log("test.loss", loss)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Configure optimizer
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.SGD(
                self.classifier.parameters(),
                lr=self.lr,
                nesterov=self.nesterov,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )

        # Configure learning rate scheduler (warmup → cosine without bolts dependency)
        lr_decay_rate = 0.1
        if self.warmup_epochs > 0:
            warmup_start_lr = self.lr * 0.1
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start_lr / self.lr,
                total_iters=self.warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, self.max_epochs - self.warmup_epochs),
                eta_min=self.lr * (lr_decay_rate**3),
                last_epoch=-1,
            )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_epochs]
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs,
                eta_min=self.lr * (lr_decay_rate**3),
                last_epoch=-1,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
