import torch
import torch.nn.functional as F
from tokenizer import Tokenizer

class NeuralBigramModel:
    def __init__(
        self,
        tokenizer: Tokenizer,
        generator: torch.Generator
    ) -> None:
        self.tokenizer = tokenizer
        self.W = torch.randn(
            (tokenizer.vocab_size, tokenizer.vocab_size),
            requires_grad=True,
            generator=generator
        )

    def compute_loss(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        reg_strength: float,
    ) -> torch.Tensor:
        probs = self.compute_probs(xs)
        ## loss calc
        # regularization: penalize large weights
        reg_term = reg_strength * (self.W ** 2).mean()
        loss = -probs[torch.arange(xs.nelement()), ys].log().mean() + reg_term
        return loss

    def compute_probs(self, xs: torch.Tensor) -> torch.Tensor:
        xenc: torch.Tensor = F.one_hot(xs, num_classes=self.tokenizer.vocab_size).float()
        logits = xenc @ self.W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        return probs

    def backward(self, loss: torch.Tensor, lr: float) -> None:
        self.W.grad = None
        loss.backward()
        self.W.data += -lr * self.W.grad # type: ignore

    def sample_next(self, token: str, generator: torch.Generator) -> str:
        enc = torch.tensor([self.tokenizer.encode(token)])
        probs = self.compute_probs(enc)
        enc = torch.multinomial(probs, num_samples=1, replacement=False, generator=generator).item()
        next_token = self.tokenizer.decode(int(enc))
        return next_token

    def generate(self, generator: torch.Generator) -> str:
        tokens = [self.tokenizer.start_token]
        while True:
            next_token = self.sample_next(tokens[-1], generator)
            tokens += next_token
            if next_token == self.tokenizer.end_token:
                break
        return self.tokenizer.untokenize(tokens)

    def eval(self, words: list[str]) -> float:
        xs, ys = self.create_dataset(words)
        probs = self.compute_probs(xs)
        nll = -probs[torch.arange(xs.nelement()), ys].log().mean()
        return nll.item()

    def create_dataset(self, words: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        xs: list[int] = []
        ys: list[int] = []
        for word in words:
            tokens = self.tokenizer.tokenize(word)
            encodings = self.tokenizer.encode(tokens)
            for enc1, enc2, in zip(encodings, encodings[1:]):
                xs.append(enc1)
                ys.append(enc2)
        # prefer use of torch.tensor over torch.Tensor, torch.tensor infers the dtype
        # whereas torch.Tensor uses f32 unless otherwise specified
        return torch.tensor(xs), torch.tensor(ys)
