import torch
import torch.nn.functional as F
from tokenizer import Tokenizer

class MlpModel:
    def __init__(
        self,
        tokenizer: Tokenizer,
        generator: torch.Generator,
        context_size: int = 3,
        embedding_dim: int = 2,
        hidden_dim: int = 100
    ) -> None:
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        ## Model definition
        # The embedding matrix
        self.C = torch.randn((tokenizer.vocab_size, embedding_dim), generator=generator)
        # Layer 1
        self.W1 = torch.randn((context_size * embedding_dim, hidden_dim), generator=generator)
        self.b1 = torch.randn(hidden_dim, generator=generator)
        # Layer 2
        self.W2 = torch.randn((hidden_dim, tokenizer.vocab_size), generator=generator)
        self.b2 = torch.randn(tokenizer.vocab_size, generator=generator)
        # Collect parameters
        self.parameters = [
            self.C,
            self.W1,
            self.b1,
            self.W2,
            self.b2
        ]
        for p in self.parameters:
            p.requires_grad = True

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        embs = self.C[X]
        embs_view = embs.view(-1, self.context_size * self.embedding_dim)
        h = (embs_view @ self.W1 + self.b1).tanh()
        logits = h @ self.W2 + self.b2
        return logits

    def backward(self, loss: torch.Tensor, lr: float) -> None:
        ## Backward pass
        # zero grad
        for p in self.parameters:
            p.grad = None
        # backprop
        loss.backward()
        # update
        for p in self.parameters:
            p.data += -lr * p.grad # type: ignore

    def create_dataset(self, words: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        xs: list[list[int]] = []
        ys: list[int] = []
        for word in words:
            tokens = self.tokenizer.tokenize(word, with_start=self.context_size)
            encodings = self.tokenizer.encode(tokens)
            for idx in range(len(encodings) - self.context_size):
                # print(
                #     ''.join(tokens[idx:idx+self.context_size]),
                #     '---->',
                #     tokens[idx+self.context_size]
                # )
                encs_in = encodings[idx:idx+self.context_size]
                enc_out = encodings[idx+self.context_size]
                xs.append(encs_in)
                ys.append(enc_out)
        return torch.tensor(xs), torch.tensor(ys)
