import os

from torch.utils.data import Dataset

from preprocess import preprocess
from vocab import Vocab
from subsample import subsample
from sampler import get_centre_contexts
from negative_sampler import NegativeSampler, get_centre_contexts_negatives_pairs_lazy

SPLITS = ["train", "valid", "test"]


class Word2VecDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        vocab=None,
        subsample_threshold=1e-5,
        sampler_cache_size=100000,
        window_size=10,
        negative_per_context=5,
    ):
        self.data_dir = data_dir
        self.split = split
        self.subsample_threshold = subsample_threshold
        self.sampler_cache_size = sampler_cache_size
        self.window_size = window_size
        self.negative_per_context = negative_per_context

        # Reading and preprocessing data
        self.data = self.read_data(split)
        self.data = preprocess(self.data)

        # Creating vocabulary if not provided
        self.vocab = vocab
        if vocab is None:
            self.vocab = Vocab()
            self.vocab.build(self.data)

        # Sub-sampling
        self.data = subsample(self.data, self.subsample_threshold)

        # Encoding data
        self.data = self.vocab.encode(self.data)

        # Negative sampling
        self.sampler = NegativeSampler(self.data, cache_size=self.sampler_cache_size)

        # Getting centre-context pairs with negative samples
        self.centre_contexts = get_centre_contexts(self.data, self.window_size)
        self.pair_iter = get_centre_contexts_negatives_pairs_lazy(
            self.centre_contexts,
            self.sampler,
            negative_per_context=self.negative_per_context,
        )
        self.pairs_len = (
            len(self.centre_contexts)
            * self.window_size
            * 2
            * (1 + self.negative_per_context)
        )

    def read_data(self, split):
        assert split in SPLITS, f"split must be one of {SPLITS}"
        filepath = os.path.join(
            self.data_dir,
            f"wikitext-2/wiki.{split}.tokens",
        )
        with open(filepath, "r") as f:
            return f.read()

    def __len__(self):
        return self.pairs_len

    def __getitem__(self, idx):
        try:
            return next(self.pair_iter)
        except StopIteration:
            self.pair_iter = get_centre_contexts_negatives_pairs_lazy(
                self.centre_contexts,
                self.sampler,
                negative_per_context=self.negative_per_context,
            )
            return next(self.pair_iter)
