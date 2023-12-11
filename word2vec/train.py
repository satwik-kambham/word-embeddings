import lightning as L

from model import EmbeddingModel
from datamodule import Word2VecDataModule


def train():
    L.seed_everything(42)

    SUBSAMPLE_THRESHOLD = 1e-3
    SAMPLE_CACHE_SIZE = 100000
    WINDOW_SIZE = 10
    NEGATIVE_PER_CONTEXT = 5
    BATCH_SIZE = 1024
    NUM_WORKERS = 2
    EMBEDDING_DIM = 300

    dm = Word2VecDataModule(
        data_dir=".data",
        subsample_threshold=SUBSAMPLE_THRESHOLD,
        sampler_cache_size=SAMPLE_CACHE_SIZE,
        window_size=WINDOW_SIZE,
        negative_per_context=NEGATIVE_PER_CONTEXT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    dm.prepare_data()

    dm.train_vocab.save("vocab.json")

    model = EmbeddingModel(
        vocab_size=len(dm.train_vocab),
        embedding_dim=EMBEDDING_DIM,
    )

    trainer = L.Trainer(
        max_epochs=5,
        fast_dev_run=False,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    train()
