from __future__ import annotations
import math

class Value:
    def __init__(
        self,
        data: float,
        _children: tuple[Value, ...] = (),
        _op: str = ''
    ) -> None:
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def tanh(self):
        x = self.data
        e = math.exp(2 * x)
        t = (e - 1) / (e + 1)
        result = Value(t, (self, ), 'tanh')
        def _backward():
            self.grad = (1 - t**2) * result.grad
        result._backward = _backward
        return result

    def __add__(self, other: Value) -> Value:
        result = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad = 1.0 * result.grad
            other.grad = 1.0 * result.grad
        result._backward = _backward
        return result

    def __mul__(self, other: Value) -> Value:
        result = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad = other.data * result.grad
            other.grad = self.data * result.grad
        result._backward = _backward
        return result

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
