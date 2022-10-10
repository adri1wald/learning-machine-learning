import torch
from neural_bigram_model import NeuralBigramModel
from tokenizer import Tokenizer

def init_gen():
    return torch.Generator().manual_seed(2147483647)

def main():
    # globals
    # TODO: figure out why this gives different results to andrej's code
    g = init_gen()

    # setup tokenizer
    WORDS = open('./names.txt').read().splitlines()
    VOCABULARY = Tokenizer.compute_vocabulary(WORDS)
    START_TOKEN = '.'
    tokenizer = Tokenizer(
        vocabulary=VOCABULARY,
        start_token=START_TOKEN
    )

    words = WORDS

    model = NeuralBigramModel(tokenizer, generator=g)

    xs, ys = model.create_dataset(words)
    LR = 50
    REG_STRENGTH = 0.01
    # training loop
    for epoch in range(200):
        loss = model.compute_loss(xs, ys, reg_strength=REG_STRENGTH)
        if epoch % 10 == 0:
            print(loss.item())
        model.backward(loss, lr=LR)

    # sample
    # reinitialise generator so its the same as BigramModel generations
    g = init_gen()
    for _ in range(10):
        word = model.generate(g)
        print(word)

    # eval
    nll = model.eval(words)
    print(f'Negative log likelihood: {nll}')

if __name__ == '__main__':
    main()
