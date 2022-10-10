import torch
import torch.nn.functional as F
from tokenizer import Tokenizer

def create_dataset(words: list[str], tokenizer: Tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[int] = []
    ys: list[int] = []
    for word in words:
        tokens = tokenizer.tokenize(word)
        encodings = tokenizer.encode(tokens)
        for enc1, enc2, in zip(encodings, encodings[1:]):
            xs.append(enc1)
            ys.append(enc2)
    # prefer use of torch.tensor over torch.Tensor, torch.tensor infers the dtype
    # whereas torch.Tensor uses f32 unless otherwise specified
    return torch.tensor(xs), torch.tensor(ys)


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

    words = WORDS
    xs, ys = create_dataset(words, tokenizer)
    
    # the "neural network"
    W = torch.randn((tokenizer.vocab_size, tokenizer.vocab_size), requires_grad=True, generator=g)
    LR = 50

    # training loop
    for epoch in range(200):
        ## forward pass
        xenc: torch.Tensor = F.one_hot(xs, num_classes=tokenizer.vocab_size).float()
        # "log counts"
        logits = xenc @ W
        #loss calc
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        loss = -probs[torch.arange(xs.nelement()), ys].log().mean()
        print(f"loss={loss.item()}")

        ## backward pass
        W.grad = None
        loss.backward()

        ## update
        W.data += -LR * W.grad # type: ignore

if __name__ == '__main__':
    main()
