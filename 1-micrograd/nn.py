from typing import Union
import random
from value import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self) -> list[Value]:
        return []

class Neuron(Module):
    def __init__(self, nin: int) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x: Union[list[Value], list[float]]) -> Value:
        assert len(x) == len(self.w)
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        result = act.tanh()
        return result

    def parameters(self) -> list[Value]:
        return self.w + [self.b]

class Layer(Module):
    def __init__(self, nin: int, nout: int) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: Union[list[Value], list[float]]) -> list[Value]:
        results = [n(x) for n in self.neurons]
        return results

    def parameters(self) -> list[Value]:
        params: list[Value] = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

class MLP(Module):
    def __init__(self, nin: int, nouts: list[int]) -> None:
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x: Union[list[Value], list[float]]) -> list[Value]:
        for layer in self.layers:
            x = layer(x)
        return x # type: ignore

    def parameters(self) -> list[Value]:
        params: list[Value] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
