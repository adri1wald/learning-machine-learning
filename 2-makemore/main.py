import torch
from tokenizer import Tokenizer
from bigram_model import BigramModel

def main():
    # globals
    # TODO: figure out why this gives different results to andrej's code
    g = torch.Generator().manual_seed(2147483647)

    # setup tokenizer
    WORDS = open('./names.txt').read().splitlines()
    VOCABULARY = Tokenizer.compute_vocabulary(WORDS)
    START_TOKEN = '.'
    tokenizer = Tokenizer(
        vocabulary=VOCABULARY,
        start_token=START_TOKEN
    )

    words = WORDS #[:10]

    # "train" bigram model
    model = BigramModel(tokenizer)
    model.train(words, smoothing=1)

    # sample
    for _ in range(10):
        word = model.generate(g)
        print(word)
    
    # eval
    nll = model.eval(words)
    print(f'Negative log likelihood: {nll}')

if __name__ == '__main__':
    main()
