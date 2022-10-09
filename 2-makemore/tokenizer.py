from typing import Optional, overload

Bigram = tuple[str, str]

class Tokenizer:
    def __init__(
        self,
        vocabulary: list[str],
        start_token: str,
        end_token: Optional[str] = None
    ) -> None:
        assert len(vocabulary) == len(set(vocabulary)), 'vocabulary should not contain duplicates'
        vocabulary.insert(0, start_token)
        if end_token is not None and end_token != start_token:
            vocabulary.append(end_token)
        self.vocabulary = vocabulary
        self.start_token = start_token
        self.end_token = end_token if end_token else start_token
        self.vocab_size = len(vocabulary)
        self._itos = self.vocabulary
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

    def compute_bigrams(self, words: list[str]) -> list[Bigram]:
        bigrams: list[Bigram] = []
        for word in words:
            chars = [self.start_token] + list(word) + [self.end_token]
            for first_char, second_char in zip(chars, chars[1:]):
                bigrams.append((first_char, second_char))
        return bigrams

    def __len__(self) -> int:
        return self.vocab_size

    @staticmethod
    def compute_vocabulary(words: list[str]) -> list[str]:
        return sorted(set(''.join(words)))
