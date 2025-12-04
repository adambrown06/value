# Micrograd: Autodiff Engine from Scratch

A minimal automatic differentiation (autodiff) engine built in pure Python. Implements reverse-mode autodiff (backpropagation) for scalar values.

## What Is This?

This is a from-scratch implementation of the core algorithm that powers PyTorch, TensorFlow, and JAX. It automatically computes gradients using the chain rule — no manual derivative calculations needed in user code.

**90 lines of code. Zero dependencies (except `math`). Full backpropagation.**

## Features

- ✅ Dynamic computational graph construction
- ✅ Reverse-mode automatic differentiation
- ✅ Gradient accumulation for multi-path dependencies
- ✅ Topological sort via DFS
- ✅ Operations: `+`, `*`, `**`, `exp`, `tanh`

## Installation

No installation needed. Just:

```bash
python micrograd.py
```

## Usage

```python
from micrograd import Value

# Define inputs
a = Value(2.0)
b = Value(3.0)
c = Value(4.0)

# Build computation graph
d = a + b           # 5.0
L = d * c           # 20.0

# Compute all gradients
L.backward()

# Access gradients
print(f"∂L/∂a = {a.grad}")  # 4.0
print(f"∂L/∂b = {b.grad}")  # 4.0
print(f"∂L/∂c = {c.grad}")  # 5.0
```

## How It Works

### 1. Computational Graph

Each operation creates a node that tracks:
- The computed value (`data`)
- The gradient w.r.t. final output (`grad`)
- Parent nodes (`_prev`)
- How to backpropagate (`_backward`)

```
     a ──┐
         ├──[+]── d ──┐
     b ──┘            ├──[×]── L
                  c ──┘
```

### 2. Forward Pass

Operations build the graph automatically:

```python
d = a + b  # Creates node with d._prev = {a, b}
L = d * c  # Creates node with L._prev = {d, c}
```

### 3. Backward Pass

`L.backward()` computes gradients in three steps:

1. **Topological sort** (DFS): Order nodes so each comes after its parents
2. **Seed gradient**: Set `L.grad = 1.0` (∂L/∂L = 1)
3. **Backpropagate**: Call each node's `_backward()` in reverse order

Each `_backward()` implements the local chain rule:

```python
# For z = x * y:
x.grad += z.grad * y  # ∂L/∂x = ∂L/∂z · ∂z/∂x
y.grad += z.grad * x  # ∂L/∂y = ∂L/∂z · ∂z/∂y
```

## Supported Operations

| Operation | Math | Backward Rule |
|-----------|------|---------------|
| Addition | `z = x + y` | `x̄ = z̄`, `ȳ = z̄` |
| Multiplication | `z = x · y` | `x̄ = z̄ · y`, `ȳ = z̄ · x` |
| Power | `z = x^n` | `x̄ = z̄ · n · x^(n-1)` |
| Exponential | `z = e^x` | `x̄ = z̄ · z` |
| Tanh | `z = tanh(x)` | `x̄ = z̄ · (1 - z²)` |

## Example: Gradient Verification

```python
# Analytical gradient
x = Value(2.0)
y = x ** 3          # y = 8.0
y.backward()
print(x.grad)       # 12.0 (= 3 * 2^2)

# Numerical gradient (finite difference)
eps = 1e-5
numerical = ((2.0 + eps)**3 - (2.0 - eps)**3) / (2 * eps)
print(numerical)    # ~12.0 ✓
```

## The Math Behind It

All backward rules were derived by hand using the chain rule:

```
∂L/∂x = ∂L/∂z · ∂z/∂x
```

Where:
- `L` = final loss/output
- `z` = intermediate node that uses `x`
- `∂L/∂z` = already computed (upstream gradient)
- `∂z/∂x` = local derivative (operation-specific)

## Why This Matters

Neural networks have millions of parameters but one scalar loss. Computing gradients manually is impossible.

**Solution**: Reverse-mode autodiff computes all gradients in one backward pass — the same cost as one forward pass.

This 90-line engine implements the same core algorithm as:
- PyTorch's `autograd`
- TensorFlow's `GradientTape`
- JAX's `grad`

## Learning Resources

This implementation was built by:
1. Deriving every backward rule from scratch (calculus)
2. Understanding topological sort and DFS
3. Implementing the chain rule as code

No copying. Pure understanding.

## What's Next?

Extend this engine with:
- `Neuron`, `Layer`, `MLP` classes
- More operations: ReLU, subtraction, division
- Graph visualization (Graphviz)
- Tensor support (move from scalars to arrays)

## License

MIT

---

**Built from scratch. Derived by hand. Understood completely.**

