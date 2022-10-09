from typing import Optional
import torch
from tokenizer import Tokenizer

class BigramModel:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.N = torch.zeros(
            (tokenizer.vocab_size, tokenizer.vocab_size),
            dtype=torch.int32
        )
        self.P: Optional[torch.Tensor] = None

    def train(self, words: list[str], smoothing: int = 0) -> None:
        for word in words:
            tokens = self.tokenizer.tokenize(word)
            encodings = self.tokenizer.encode(tokens)
            for enc1, enc2 in zip(encodings, encodings[1:]):
                self.N[enc1, enc2] += 1
        P = (self.N + smoothing).float()
        P /= P.sum(dim=1, keepdim=True)
        self.P = P

    def sample_next(self, token: str, generator: torch.Generator) -> str:
        if self.P is None:
            raise RuntimeError('Model must be trained before sampling')
        enc = self.tokenizer.encode(token)
        p = self.P[enc]
        enc = torch.multinomial(p, num_samples=1, replacement=False, generator=generator).item()
        next_token = self.tokenizer.decode(int(enc))
        return next_token

    def generate(self, generator: torch.Generator) -> str:
        tokens = [self.tokenizer.start_token]
        while True:
            next_token = self.sample_next(tokens[-1], generator)
            tokens += next_token
            if next_token == self.tokenizer.end_token:
                break
        return self.tokenizer.untokenize(tokens)

    def eval(self, words: list[str]) -> float:
        if self.P is None:
            raise RuntimeError('Model must be trained before eval')
        n = 0
        log_likelihood = 0.0
        for word in words:
            tokens = self.tokenizer.tokenize(word)
            encodings = self.tokenizer.encode(tokens)
            for enc1, enc2 in zip(encodings, encodings[1:]):
                n += 1
                log_likelihood += torch.log(self.P[enc1, enc2])
        return - log_likelihood / n
