import lightning as L

from model import EmbeddingModel
from datamodule import Word2VecDataModule


def train():
    dm = Word2VecDataModule(data_dir=".data")
    dm.prepare_data()
    model = EmbeddingModel(
        vocab_size=len(dm.train_vocab),
        embedding_dim=128,
    )

    trainer = L.Trainer(
        max_epochs=5,
        fast_dev_run=False,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    train()
