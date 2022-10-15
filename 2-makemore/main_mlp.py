import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mlp_model import MlpModel
from tokenizer import Tokenizer

def init_gen():
    return torch.Generator().manual_seed(2147483647)

def main():
    g = init_gen()

    WORDS = open('./names.txt').read().splitlines()
    VOCABULARY = Tokenizer.compute_vocabulary(WORDS)
    START_TOKEN = '.'
    tokenizer = Tokenizer(
        vocabulary=VOCABULARY,
        start_token=START_TOKEN
    )

    # Slice here to work on subset of data
    words = WORDS
    
    CONTEXT_SIZE = 3
    EMBEDDING_DIM = 2
    HIDDEN_DIM = 100
    model = MlpModel(
        tokenizer,
        context_size=CONTEXT_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        generator=g
    )
    X, Y = model.create_dataset(words)
    
    MINIBATCH_SIZE = 32
    EPOCHS = 1000
    lre = torch.linspace(-3, 0, 1000)
    lrs = 10**lre

    lrei: list[float] = []
    lossi: list[float] = []

    for epoch in range(EPOCHS):
        # minibatch construct
        mbis = torch.randint(0, X.shape[0], (MINIBATCH_SIZE, ))
        Xmb = X[mbis]
        Ymb = Y[mbis]

        # forward pass
        logits = model.forward(Xmb)
        # loss calc
        loss = F.cross_entropy(logits, Ymb)
        # backward pass
        lr = lrs[epoch].item()
        model.backward(loss, lr)

        # stats
        lrei.append(lre[epoch].item())
        lossi.append(loss.item())

        if epoch % 100 == 0:
            print(f"Training loss = {loss.item()}")
        if epoch % 500 == 0:
            eval_loss = model.eval(words)
            print(f"Eval loss = {eval_loss}")

    # plot loss against lr
    fig = plt.figure(figsize=(16, 10))
    plot = fig.add_subplot(111)

    plot.plot(lrei, lossi)

    fig.savefig('./figures/loss-vs-lr.png', bbox_inches='tight')

    # sample
    g = init_gen()
    for _ in range(10):
        word = model.generate(g)
        print(word)

if __name__ == '__main__':
    main()
