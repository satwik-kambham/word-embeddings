import torch
import torch.nn as nn

import lightning as L
import torchmetrics as tm


class EmbeddingModel(L.LightningModule):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.save_hyperparameters()

        self.embedding_dim = embedding_dim
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)

        def init_weights(module):
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

        self.apply(init_weights)

        self.accuracy = tm.Accuracy(task="binary")

    def forward(self, word, context):
        word = self.word_embedding(word)
        context = self.context_embedding(context)
        return (word * context).sum(dim=1)

    def training_step(self, batch, batch_idx):
        centre, context, label = batch
        y_hat = self(centre, context)
        loss = nn.functional.binary_cross_entropy_with_logits(y_hat, label.float())
        self.log("train_loss", loss)

        pred = torch.round(torch.sigmoid(y_hat))
        self.accuracy(pred, label)
        self.log(
            "train_acc", self.accuracy, prog_bar=False, on_step=True, on_epoch=False
        )

        return loss

    def validation_step(self, batch, batch_idx):
        centre, context, label = batch
        y_hat = self(centre, context)
        loss = nn.functional.binary_cross_entropy_with_logits(y_hat, label.float())
        self.log("val_loss", loss)

        pred = torch.round(torch.sigmoid(y_hat))
        self.accuracy(pred, label)
        self.log("val_acc", self.accuracy, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
