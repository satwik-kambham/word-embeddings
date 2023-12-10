import random
import itertools

from subsample import get_word_counts


class NegativeSampler:
    def __init__(self, corpus, cache_size=100000):
        self.word_counts = get_word_counts(corpus)
        self.words = list(self.word_counts.keys())
        self.sampling_weights = [self.word_counts[word] ** 0.75 for word in self.words]
        self.cumulative_weights = list(
            itertools.accumulate(
                self.sampling_weights,
            )
        )
        self.cache_size = cache_size
        self.cache = random.choices(
            self.words, cum_weights=self.cumulative_weights, k=self.cache_size
        )
        self.cache_idx = 0

    def sample_negative(self, contexts, k=5):
        negatives = []
        for _ in range(k):
            while True:
                word = self.cache[self.cache_idx]
                self.cache_idx = self.cache_idx + 1

                if self.cache_idx >= self.cache_size:
                    self.cache = random.choices(
                        self.words,
                        cum_weights=self.cumulative_weights,
                        k=self.cache_size,
                    )
                    self.cache_idx = 0

                if word not in contexts:
                    negatives.append(word)
                    break
        return negatives


def get_centre_contexts_negatives_pairs_lazy(
    centre_contexts, sampler, negative_per_context=5
):
    for centre, contexts in centre_contexts:
        for context in contexts:
            yield centre, context, 1
            negatives = sampler.sample_negative(contexts, k=negative_per_context)
            for negative in range(negative_per_context):
                yield centre, negatives[negative], 0
