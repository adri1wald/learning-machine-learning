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

    fig = plt.figure(figsize=(16, 16))
    plot = fig.add_subplot(111)
    plot.imshow(N, cmap='Blues')
    plot.axis('off')
    for i in range(dataset.vocab_size):
        for j in range(dataset.vocab_size):
            bigram = ''.join(dataset.decode([i, j]))
            freq = str(N[i, j].item())
            plot.text(j, i, bigram, ha='center', va='bottom', color='gray')
            plot.text(j, i, freq, ha='center', va='top', color='gray')

    fig.savefig('./figures/freqs.png')

if __name__ == '__main__':
    main()
