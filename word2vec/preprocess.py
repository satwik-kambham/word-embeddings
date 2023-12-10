import string


def split_words(text):
    return text.split()


def remove_punctuation(words):
    return [
        word.strip(string.punctuation)
        for word in words
        if word.strip(string.punctuation)
    ]


def lower(words):
    return [word.lower() for word in words]


def preprocess(text):
    words = split_words(text)
    words = remove_punctuation(words)
    words = lower(words)
    return words
