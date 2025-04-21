#simple character level tokenizer for FEN strings
class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_inv = {v: k for k, v in self.vocab.items()}

    def encode(self, modified_fen):
        tokens = []
        fen_split = modified_fen.split()
        for token in fen_split:
            if token in self.vocab:
                tokens.append(self.vocab[token])
            else:
                tokens.append(self.vocab["[UNK]"])
        return tokens

    def decode(self, tokens):
        return [self.vocab_inv[token] for token in tokens]

    def __len__(self):
        return len(self.vocab)

    def __call__(self, fen):
        return self.encode(fen)
