import torch
from tokenizer import Tokenizer
# from viz import draw_freqs

def main():
    # setup tokenizer
    WORDS = open('./names.txt').read().splitlines()
    VOCABULARY = Tokenizer.compute_vocabulary(WORDS)
    START_TOKEN = '.'
    tokenizer = Tokenizer(
        vocabulary=VOCABULARY,
        start_token=START_TOKEN
    )

    words = WORDS #[:10]

    N = torch.zeros(
        (tokenizer.vocab_size, tokenizer.vocab_size),
        dtype=torch.int32
    )

    bigrams = compute_bigrams(words, tokenizer)
    for tok1, tok2 in bigrams:
        enc1 = tokenizer.encode(tok1)
        enc2 = tokenizer.encode(tok2)
        N[enc1, enc2] += 1

    # draw_freqs(N, tokenizer)

    P = N[0].float()
    P = P / P.sum()
    print(P)

Bigram = tuple[str, str]
def compute_bigrams(words: list[str], tokenizer: Tokenizer) -> list[Bigram]:
    bigrams: list[Bigram] = []
    for word in words:
        tokens = tokenizer.tokenize(word)
        for tok1, tok2 in zip(tokens, tokens[1:]):
            bigrams.append((tok1, tok2))
    return bigrams

if __name__ == '__main__':
    main()
