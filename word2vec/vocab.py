class Vocab:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        self.build(["<unk>"])

    def __len__(self):
        return len(self.word2idx)

    def build(self, text):
        for word in text:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def encode(self, text):
        return [self.word2idx.get(word, self.word2idx["<unk>"]) for word in text]

    def decode(self, tokens):
        return [self.idx2word.get(idx, "<unk>") for idx in tokens]
