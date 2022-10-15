import torch 
import torch.nn.functional as F
from mlp_model import MlpModel
from lr_scheduler import LearningRateScheduler
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
    scheduler = LearningRateScheduler([
        (40000, 0.1),
        (10000, 0.01)
    ])

    for epoch, lr in scheduler:
        # minibatch construct
        mbis = torch.randint(0, X.shape[0], (MINIBATCH_SIZE, ))
        Xmb = X[mbis]
        Ymb = Y[mbis]

        # forward pass
        logits = model.forward(Xmb)
        # loss calc
        loss = F.cross_entropy(logits, Ymb)
        # backward pass
        model.backward(loss, lr)

        # stats
        if epoch % 100 == 0:
            print(f"Training loss = {loss.item()}")
        if epoch % 1000 == 0:
            eval_loss = model.eval(words)
            print(f"Eval loss = {eval_loss}")

    # sample
    g = init_gen()
    for _ in range(10):
        word = model.generate(g)
        print(word)

if __name__ == '__main__':
    main()
