import torch 
import torch.nn.functional as F
from tokenizer import Tokenizer

def init_gen():
    return torch.Generator().manual_seed(2147483647)

def create_dataset(
    words: list[str],
    tokenizer: Tokenizer,
    context_size: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[list[int]] = []
    ys: list[int] = []
    for word in words:
        tokens = tokenizer.tokenize(word, with_start=context_size)
        encodings = tokenizer.encode(tokens)
        for idx in range(len(encodings) - context_size):
            # print(
            #     ''.join(tokens[idx:idx+context_size]),
            #     '---->',
            #     tokens[idx+context_size]
            # )
            encs_in = encodings[idx:idx+context_size]
            enc_out = encodings[idx+context_size]
            xs.append(encs_in)
            ys.append(enc_out)
    return torch.tensor(xs), torch.tensor(ys)

def main():
    g = init_gen()

    WORDS = open('./names.txt').read().splitlines()
    VOCABULARY = Tokenizer.compute_vocabulary(WORDS)
    START_TOKEN = '.'
    CONTEXT_SIZE = 3
    EMBEDDING_DIM = 2
    HIDDEN_DIM = 100
    LR = 0.1
    MINIBATCH_SIZE = 32
    tokenizer = Tokenizer(
        vocabulary=VOCABULARY,
        start_token=START_TOKEN
    )

    words = WORDS
    X, Y = create_dataset(words, tokenizer, CONTEXT_SIZE)
    
    ## MLP
    # The embedding matrix
    C = torch.randn((tokenizer.vocab_size, EMBEDDING_DIM), generator=g)
    # Layer 1
    W1 = torch.randn((CONTEXT_SIZE * EMBEDDING_DIM, HIDDEN_DIM), generator=g)
    b1 = torch.randn(HIDDEN_DIM, generator=g)
    # Layer 2
    W2 = torch.randn((HIDDEN_DIM, tokenizer.vocab_size), generator=g)
    b2 = torch.randn(tokenizer.vocab_size, generator=g)
    # Collect parameters
    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True

    for epoch in range(100):
        # minibatch construct
        idxs = torch.randint(0, X.shape[0], (MINIBATCH_SIZE, ))
        Xmb = X[idxs]
        Ymb = Y[idxs]

        ## Forward pass
        embs = C[Xmb]
        h = (embs.view(-1, CONTEXT_SIZE * EMBEDDING_DIM) @ W1 + b1).tanh()
        logits = h @ W2 + b2

        ## Loss calculation
        # - Much more efficient than manual softmax + picking out probs
        # - Will cluster up operations and even use fused kernels
        # - More numerically stable than doing logits.exp() which can result in inf
        loss = F.cross_entropy(logits, Ymb)
        if epoch % 3 == 0:
            print(f"loss={loss.item()}")

        ## Backward pass
        # zero grad
        for p in parameters:
            p.grad = None
        # backprop
        loss.backward()
        # update
        for p in parameters:
            p.data += -LR * p.grad # type: ignore

if __name__ == '__main__':
    main()
