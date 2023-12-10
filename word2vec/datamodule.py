from torch.utils.data import DataLoader

import lightning as L

from download import download_and_extract, DATA_URL
from dataset import Word2VecDataset


class Word2VecDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir,
        force_download=False,
        subsample_threshold=1e-5,
        sampler_cache_size=100000,
        window_size=10,
        negative_per_context=5,
        batch_size=512,
        num_workers=1,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.data_dir = data_dir
        self.force_download = force_download
        self.ds_hparams = {
            "subsample_threshold": subsample_threshold,
            "sampler_cache_size": sampler_cache_size,
            "window_size": window_size,
            "negative_per_context": negative_per_context,
        }
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        download_and_extract(DATA_URL, self.data_dir, force=self.force_download)
        self.train_ds = Word2VecDataset(self.data_dir, split="train", **self.ds_hparams)
        self.train_vocab = self.train_ds.vocab
        self.val_ds = Word2VecDataset(
            self.data_dir, split="valid", vocab=self.train_vocab, **self.ds_hparams
        )
        self.test_ds = Word2VecDataset(
            self.data_dir, split="test", vocab=self.train_vocab, **self.ds_hparams
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
