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
    L1_SIZE = 100
    tokenizer = Tokenizer(
        vocabulary=VOCABULARY,
        start_token=START_TOKEN
    )

    words = WORDS[:5]
    X, Y = create_dataset(words, tokenizer, CONTEXT_SIZE)
    
    # The embedding matrix
    C = torch.randn((tokenizer.vocab_size, EMBEDDING_DIM), generator=g)
    # Layer 1
    W1 = torch.randn((CONTEXT_SIZE * EMBEDDING_DIM, L1_SIZE), generator=g)
    b1 = torch.randn(L1_SIZE, generator=g)
    # Layer 2
    W2 = torch.randn((L1_SIZE, tokenizer.vocab_size), generator=g)
    b2 = torch.randn(tokenizer.vocab_size, generator=g)

    # forward pass
    embs = C[X]
    h = (embs.view(-1, CONTEXT_SIZE * EMBEDDING_DIM) @ W1 + b1).tanh()
    logits = h @ W2 + b2

    # loss calc
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(Y.nelement()), Y].log().mean()
    print(loss)
    

if __name__ == '__main__':
    main()
