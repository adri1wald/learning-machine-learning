import torch
import matplotlib.pyplot as plt
from dataset import MakemoreDataset

def main():
    # setup dataset interface
    WORDS = open('./names.txt').read().splitlines()
    VOCABULARY = MakemoreDataset.compute_vocabulary(WORDS)
    START_TOKEN = '<S>'
    END_TOKEN = '<E>'
    dataset = MakemoreDataset(
        vocabulary=VOCABULARY,
        start_token=START_TOKEN,
        end_token=END_TOKEN
    )

    words = WORDS #[:10]

    N = torch.zeros(
        (dataset.vocab_size, dataset.vocab_size),
        dtype=torch.int32
    )

    bigrams = dataset.get_bigrams(words)
    for tok1, tok2 in bigrams:
        enc1 = dataset.encode(tok1)
        enc2 = dataset.encode(tok2)
        N[enc1, enc2] += 1

    plt.imsave('./figures/freqs.png', N)

if __name__ == '__main__':
    main()
