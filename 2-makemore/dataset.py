from typing import Optional

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

    def encode(self, token: str) -> int:
        return self._stoi[token]

    def decode(self, enc: int) -> str:
        return self._itos[enc]

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
