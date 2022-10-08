from __future__ import annotations

class Value:
    def __init__(
        self,
        data: float,
        _children: tuple[Value, ...] = (),
        _op: str = ''
    ) -> None:
        self.data = data
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other: Value) -> Value:
        result = Value(self.data + other.data, (self, other), '+')
        return result

    def __mul__(self, other: Value) -> Value:
        result = Value(self.data * other.data, (self, other), '*')
        return result
