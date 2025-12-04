import math


class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # TODO: Implement backward for addition
            # Your derivation: x̄ = z̄, ȳ = z̄
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # TODO: Implement backward for multiplication
            # Your derivation: x̄ = z̄ · y, ȳ = z̄ · x
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, n):
        assert isinstance(n, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** n, (self,), f'**{n}')

        def _backward():
            # TODO: Implement backward for power
            # Your derivation: x̄ = z̄ · n · x^(n-1)
            self.grad += out.grad * n * (self.data**(n-1))
        out._backward = _backward

        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            # TODO: Implement backward for exp
            # Your derivation: x̄ = z̄ · z
            self.grad += out.grad * out.data
        out._backward = _backward

        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')

        def _backward():
            # TODO: Implement backward for tanh
            # Your derivation: x̄ = z̄ · (1 - z²)
            self.grad += out.grad * (1-out.data**2)
        out._backward = _backward

        return out

    def backward(self):
        # TODO: Implement topological sort + reverse-mode autodiff
        # 1. Build topological order via DFS
        # 2. Seed self.grad = 1.0
        # 3. Call _backward() for each node in reverse topo order
        topo = []
        visited = set()
        def build_topo(v):
            if v in visited:
                return
            visited.add(v)
            for parent in v._prev:
                build_topo(parent)
            topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

if __name__ == "__main__":
    # Test case: L = (a + b) * c
    a = Value(2.0)
    b = Value(3.0)
    c = Value(4.0)

    d = a + b      # d = 5
    L = d * c      # L = 20

    L.backward()

    print(f"L = {L.data}")       # Expected: 20.0
    print(f"∂L/∂a = {a.grad}")   # Expected: 4.0
    print(f"∂L/∂b = {b.grad}")   # Expected: 4.0
    print(f"∂L/∂c = {c.grad}")   # Expected: 5.0

