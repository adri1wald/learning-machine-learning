import torch
import matplotlib.pyplot as plt
from tokenizer import Tokenizer

def draw_freqs(N: torch.Tensor, tokenizer: Tokenizer) -> None:
    fig = plt.figure(figsize=(25, 25))
    fontdict = { 'fontsize': 14 }
    plot = fig.add_subplot(111)
    plot.axis('off')

    plot.imshow(N, cmap='Blues')
    for i in range(tokenizer.vocab_size):
        for j in range(tokenizer.vocab_size):
            bigram = ''.join(tokenizer.decode([i, j]))
            freq = str(N[i, j].item())
            plot.text(j, i, bigram, ha='center', va='bottom', color='gray', fontdict=fontdict)
            plot.text(j, i, freq, ha='center', va='top', color='gray', fontdict=fontdict)

    fig.savefig('./figures/freqs.png', bbox_inches='tight')
