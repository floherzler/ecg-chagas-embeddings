from lightning import LightningDataModule
from ecg_chagas_embeddings.data.dataset import get_train_val_loaders


class ECGDataModule(LightningDataModule):
    def __init__(
        self,
        meta_path,
        data_dir,
        batch_size: int = 256,
        pos_fraction: float = 0.25,
        train_folds=(0, 1, 2, 3),
        valid_folds=(4,),
        num_workers: int = 4,
        prefetch_factor: int = 16,
        oversample: bool = True,
        use_sup_con: bool = False,
        use_prototypes: bool = False,
        val_n_views: int | None = None,
        **augment_kwargs,
    ):
        super().__init__()
        self.cfg = dict(
            meta_path=meta_path,
            data_dir=data_dir,
            batch_size=batch_size,
            pos_fraction=pos_fraction,
            train_folds=train_folds,
            valid_folds=valid_folds,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            oversample=oversample,
            use_sup_con=use_sup_con,
            use_prototypes=use_prototypes,
            val_n_views=val_n_views,
            **augment_kwargs,
        )
        self._train_loader = None
        self._val_loader = None

    def setup(self, stage=None):
        if stage in (None, "fit"):
            train, val = get_train_val_loaders(**self.cfg)
            self._train_loader = train
            self._val_loader = val
        # add test/predict branches if you need them later

    def train_dataloader(self):
        if self._train_loader is None:
            self.setup("fit")
        return self._train_loader

    def val_dataloader(self):
        if self._val_loader is None:
            self.setup("fit")
        return self._val_loader
