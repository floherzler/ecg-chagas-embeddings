import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm

from ecg_chagas_embeddings.helper_code import compute_accuracy, compute_challenge_score


def _padding(downsample, kernel_size):
    """Compute required padding"""
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding


def _downsample(n_samples_in, n_samples_out):
    """Compute downsample rate"""
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease")
    if n_samples_in % n_samples_out != 0:
        raise ValueError(
            "Number of samples for two consecutive blocks "
            "should always decrease by an integer factor."
        )
    return downsample


class ResBlock1d(nn.Module):
    """Residual network unit for unidimensional signals."""

    def __init__(
        self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate
    ):
        if kernel_size % 2 == 0:
            raise ValueError(
                "The current implementation only support odd values for `kernel_size`."
            )
        super(ResBlock1d, self).__init__()

        # Forward path
        padding1 = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(
            n_filters_in, n_filters_out, kernel_size, padding=padding1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        padding2 = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(
            n_filters_out,
            n_filters_out,
            kernel_size,
            stride=downsample,
            padding=padding2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Skip connection
        skip_connection_layers = []

        # Deal with downsampling
        if downsample > 1:
            maxpool = nn.MaxPool1d(downsample, stride=downsample)
            skip_connection_layers += [maxpool]

        # Deal with n_filters dimension increase
        if n_filters_in != n_filters_out:
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
            skip_connection_layers += [conv1x1]

        # Build skip connection layer
        if skip_connection_layers:
            self.skip_connection = nn.Sequential(*skip_connection_layers)
        else:
            self.skip_connection = None

    def forward(self, x, y):
        """Residual unit."""
        if self.skip_connection is not None:
            y = self.skip_connection(y)
        else:
            y = y

        # 1st layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # 2nd layer
        x = self.conv2(x)
        x += y  # Sum skip connection and main connection
        y = x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x, y


class LitResNet1d(pl.LightningModule):
    """Residual network for unidimensional signals.
    Parameters
    ----------
    input_dim : tuple
        Input dimensions. Tuple containing dimensions for the neural network
        input tensor. Should be like: ``(n_filters, n_samples)``.
    blocks_dim : list of tuples
        Dimensions of residual blocks.  The i-th tuple should contain the dimensions
        of the output (i-1)-th residual block and the input to the i-th residual
        block. Each tuple shoud be like: ``(n_filters, n_samples)``. `n_samples`
        for two consecutive samples should always decrease by an integer factor.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. The current implementation
        only supports odd kernel sizes. Default is 17.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027, Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(
        self,
        input_dim,
        blocks_dim,
        n_classes,
        criterion,
        kernel_size=17,
        dropout_rate=0.8,
    ):
        super(LitResNet1d, self).__init__()
        self.save_hyperparameters()

        self.criterion = criterion
        self.train_step_losses = []
        self.val_step_losses = []
        self.validation_step_outputs = []

        # First layer
        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]  # 12, 64
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]  # 4096, 4096
        downsample = _downsample(n_samples_in, n_samples_out)
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(
            n_filters_in,
            n_filters_out,
            kernel_size,
            bias=False,
            stride=downsample,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()
        # no relu?

        # Residual block layers
        self.res_blocks = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            resblk1d = ResBlock1d(
                n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate
            )
            self.add_module(
                "resblock1d_{nr}".format(nr=i), resblk1d
            )  # make the resblocks actual modules (self.resblock_0 etc)
            self.res_blocks += [resblk1d]

        # Linear layer
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin = nn.Linear(last_layer_dim, n_classes)

        # number of residual blocks
        self.n_blk = len(blocks_dim)

    def forward(self, x):
        """Implement ResNet1d forward propagation"""
        # First layers
        x = self.conv1(x)
        x = self.bn1(x)

        # no relu here?
        x = self.relu(x)

        # Residual blocks
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)

        # Flatten array
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.lin(x)
        return x

    def training_step(self, batch, batch_idx):
        signals, labels = batch
        outputs = self(signals)
        loss = self.criterion(outputs, labels)
        self.train_step_losses.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        signals, labels = batch
        logits = self(signals)
        probs = torch.sigmoid(logits)
        if torch.isnan(probs).any():
            print("NaN in probs")
            print(f"probs: {probs}")
            print(f"logits: {logits}")
            print(f"labels: {labels}")
        preds = (probs > 0.5).long()
        loss = self.criterion(logits, labels)
        self.validation_step_outputs.append(
            (labels.view(-1), probs.view(-1), preds.view(-1))
        )
        self.val_step_losses.append(loss)
        return {"val_loss": loss, "gt": labels, "probs": probs, "preds": preds}

    def on_validation_epoch_end(self):
        gts = (
            torch.cat([batch[0] for batch in self.validation_step_outputs], dim=0)
            .to("cpu")
            .numpy()
        )
        probs = (
            torch.cat([batch[1] for batch in self.validation_step_outputs], dim=0)
            .to("cpu")
            .numpy()
        )
        preds = (
            torch.cat([batch[2] for batch in self.validation_step_outputs], dim=0)
            .to("cpu")
            .numpy()
        )

        self.validation_step_outputs.clear()

        num_gts_ones = np.sum(gts == 1.0)
        tqdm.write(f"Positive cases in epoch: {num_gts_ones} of total {len(gts)}")

        acc = compute_accuracy(gts, preds)
        try:
            score = compute_challenge_score(gts, probs)
        except IndexError:
            score = 0.0
            tqdm.write("Error in computing challenge score. Setting score to 0.0")

        quartiles = {}

        tqdm.write(f"probs: {probs[:10]}")
        tqdm.write(f"\n### Quartiles for score {score}:")
        for cls in [0, 1]:
            cls_probs = probs[gts == cls]
            if len(cls_probs) == 0:
                quartiles[f"class_{cls}"] = {"q1": 0.0, "median": 0.0, "q3": 0.0}
            else:
                cls_probs = np.asarray(cls_probs, dtype=np.float64)
                q1 = float(np.nanpercentile(cls_probs, 25))
                med = float(np.nanmedian(cls_probs))
                q3 = float(np.nanpercentile(cls_probs, 75))
                quartiles[f"class_{cls}"] = {"q1": q1, "median": med, "q3": q3}
            tqdm.write(
                f"    class {cls} -> q1: {q1:.4f}, median: {med:.4f}, q3: {q3:.4f}\n"
            )

        # Log quartiles
        for cls, stats in quartiles.items():
            for k, v in stats.items():
                self.log(
                    f"val/{cls}_{k}", v, on_epoch=True, on_step=False, prog_bar=False
                )

        self.log("val/acc", acc, prog_bar=False, on_epoch=True, on_step=False)
        self.log("val/score", score, prog_bar=True, on_epoch=True, on_step=False)

        if not len(self.train_step_losses) == 0 and not len(self.val_step_losses) == 0:
            self.log(
                "train/loss",
                torch.stack([x for x in self.train_step_losses]).mean(),
                on_epoch=True,
                on_step=False,
                prog_bar=False,
            )
            self.log(
                "val/loss",
                torch.stack([x for x in self.val_step_losses]).mean(),
                on_epoch=True,
                on_step=False,
                prog_bar=False,
            )
        self.train_step_losses.clear()
        self.val_step_losses.clear()

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "base_lr",
            },
        }
