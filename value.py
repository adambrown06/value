import math
import random


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
    
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rpow__(self, other):
        # other ** self where other is a constant (e.g., 2 ** x)
        # = exp(self * ln(other))
        return (self * math.log(other)).exp()

    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += out.grad * (1.0 if self.data > 0 else 0.0)
        out._backward = _backward

        return out

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            # σ'(x) = σ(x) * (1 - σ(x)) = out.data * (1 - out.data)
            self.grad += out.grad * out.data * (1 - out.data)
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

class Neuron:
    def __init__(self, nin: int):
        self.w = [Value(random.gauss(0, 0.1)) for _ in range(nin)]
        self.b = Value(0.0)
    
    def __call__(self, x: list[Value]) -> Value:
        # activation = tanh(sum(w_i * x_i) + b)
        act = sum((wi * xi) for wi, xi in zip(self.w, x)) + self.b
        return act.tanh()

    def parameters(self) -> list[Value]:
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
       
    def __call__(self, x: list[Value]):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        self.layers = []
        sizes = [nin] + nouts
        for i in range(len(sizes) - 1):
            self.layers.append(Layer(sizes[i], sizes[i + 1]))
    
    def __call__(self, x: list[Value]):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    

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

    n = Neuron(2)
    x = [Value(0.5), Value(-1.2)]
    y = n(x); y.backward()
    eps = 1e-4
    w0 = n.w[0].data
    n.w[0].data = w0 + eps; y_pos = n(x).data
    n.w[0].data = w0 - eps; y_neg = n(x).data
    n.w[0].data = w0
    print("w0 grad analytic:", n.w[0].grad, "numeric:", (y_pos - y_neg) / (2*eps))


    neural_net = MLP(1,[16, 16, 1])
    x_mlp = [Value(0.2)]
    y_pred = neural_net(x_mlp)  # single Value because final layer has 1 neuron
    loss = y_pred  # placeholder: in real training, compute loss vs target
    loss.backward()
    print(f"MLP output (used as loss placeholder): {loss.data}")


