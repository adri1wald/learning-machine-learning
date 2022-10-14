from venv import create
import torch 
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
    tokenizer = Tokenizer(
        vocabulary=VOCABULARY,
        start_token=START_TOKEN
    )

    words = WORDS[:5]
    xs, ys = create_dataset(words, tokenizer, CONTEXT_SIZE)
    print(xs.shape)
    print(ys.shape)
    



if __name__ == '__main__':
    main()
