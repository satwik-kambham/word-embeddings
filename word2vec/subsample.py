import math
import random


def get_word_counts(text):
    counts = {}
    for word in text:
        if word not in counts:
            counts[word] = 0
        counts[word] += 1
    return counts


def keep_word(count, total, threshold):
    prob = 1 - math.sqrt(threshold * total / count)
    return random.random() > prob


def subsample(text, threshold=1e-5):
    counts = get_word_counts(text)
    total = len(text)
    return [word for word in text if keep_word(counts[word], total, threshold)]
