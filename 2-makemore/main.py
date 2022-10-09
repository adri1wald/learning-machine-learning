import torch
from tokenizer import Tokenizer
from viz import draw_freqs

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

    bigrams = tokenizer.compute_bigrams(words)
    for tok1, tok2 in bigrams:
        enc1 = tokenizer.encode(tok1)
        enc2 = tokenizer.encode(tok2)
        N[enc1, enc2] += 1

    draw_freqs(N, tokenizer)

if __name__ == '__main__':
    main()
