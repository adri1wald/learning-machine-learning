from __future__ import annotations
import math
from typing import Union

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

    def backward(self) -> None:
        topo: list[Value] = []
        visited: set[Value] = set()
        def build_topo(value: Value):
            if value in visited:
                return
            visited.add(value)
            for child in value._prev:
                build_topo(child)
            topo.append(value)
        build_topo(self)
        self.grad = 1
        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        def step(value: Value) -> None:
            value.grad = 0
            for child in value._prev:
                step(child)
        step(self)

    def tanh(self) -> Value:
        e = math.exp(2 * self.data)
        t = (e - 1) / (e + 1)
        result = Value(t, (self, ), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * result.grad
        result._backward = _backward
        return result

    def exp(self) -> Value:
        e = math.exp(self.data)
        result = Value(e, (self, ), 'exp')
        def _backward():
            self.grad += e * result.grad
        result._backward = _backward
        return result

    def __add__(self, other: Union[Value, float]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += result.grad
            other.grad += result.grad
        result._backward = _backward
        return result

    def __radd__(self, other: Union[Value, float]) -> Value:
        return self + other

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Union[Value, float]) -> Value:
        return self + -other

    def __rsub__(self, other: Union[Value, float]) -> Value:
        return -self + other

    def __mul__(self, other: Union[Value, float]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad
        result._backward = _backward
        return result

    def __rmul__(self, other: Union[Value, float]) -> Value:
        return self * other

    def __truediv__(self, other: Union[Value, float]) -> Value:
        return self * other**-1

    def __pow__(self, other: Union[int, float]) -> Value:
        assert isinstance(other, (int, float)), 'Only support int / float powers'
        p = self.data**other
        result = Value(p, (self, ), f'**{other}')
        def _backward():
            self.grad += other * self.data**(other - 1) * result.grad
        result._backward = _backward
        return result

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
