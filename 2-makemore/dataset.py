from typing import Optional, overload

Bigram = tuple[str, str]

class MakemoreDataset:
    def __init__(
        self,
        vocabulary: list[str],
        start_token: str,
        end_token: Optional[str] = None
    ) -> None:
        assert len(vocabulary) == len(set(vocabulary)), 'vocabulary should not contain duplicates'
        self.vocabulary = vocabulary
        self.start_token = start_token
        self.end_token = end_token if end_token else start_token
        self.vocab_size = len(vocabulary) + len(set((self.start_token, self.end_token)))
        self._itos = [self.start_token, *self.vocabulary, self.end_token]
        self._stoi = { s: i for i, s in enumerate(self._itos) }

    @overload
    def encode(self, tokens: str) -> int:
        ...
    @overload
    def encode(self, tokens: list[str]) -> list[int]:
        ...
    def encode(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return self._stoi[tokens]
        else:
            return [self._stoi[t] for t in tokens]

    @overload
    def decode(self, encodings: int) -> str:
        ...
    @overload
    def decode(self, encodings: list[int]) -> list[str]:
        ...
    def decode(self, encodings: int | list[int]) -> str | list[str]:
        if isinstance(encodings, int):
            return self._itos[encodings]
        else:
            return [self._itos[e] for e in encodings]

    def get_bigrams(self, words: list[str]) -> list[Bigram]:
        bigrams: list[Bigram] = []
        for word in words:
            chars = [self.start_token] + list(word) + [self.end_token]
            for first_char, second_char in zip(chars, chars[1:]):
                bigrams.append((first_char, second_char))
        return bigrams

    @staticmethod
    def compute_vocabulary(words: list[str]) -> list[str]:
        return sorted(set(''.join(words)))
