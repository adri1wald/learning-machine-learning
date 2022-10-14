from typing import Optional, overload

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

    @overload
    def tokenize(
        self,
        words: str,
        with_start: bool | int = True,
        with_end: bool | int = True
    ) -> list[str]:
        ...
    @overload
    def tokenize(
        self,
        words: list[str],
        with_start: bool | int = True,
        with_end: bool | int = True
    ) -> list[list[str]]:
        ...
    def tokenize(
        self,
        words: str | list[str],
        with_start: bool | int = True,
        with_end: bool | int = True
    ) -> list[str] | list[list[str]]:
        if isinstance(words, str):
            tokens: list[str] = []
            if with_start:
                tokens.extend([self.start_token] * int(with_start))
            tokens.extend(list(words))
            if with_end:
                tokens.extend([self.end_token] * int(with_end))
            return tokens
        else:
            return [
                self.tokenize(w, with_start=with_start, with_end=with_end)
                for w in words
            ]

    def untokenize(self, tokens: list[str], exclude_special_tokens: bool = True) -> str:
        if exclude_special_tokens:
            tokens = [
                token for token in tokens
                if token != self.start_token and token != self.end_token
            ]
        return ''.join(tokens)

    def __len__(self) -> int:
        return self.vocab_size

    @staticmethod
    def compute_vocabulary(words: list[str]) -> list[str]:
        return sorted(set(''.join(words)))
