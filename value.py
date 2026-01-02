"""
Automatic Differentiation Engine and Neural Network Implementation

This module implements:
- Value: Scalar autograd engine with reverse-mode automatic differentiation
- Neuron, Layer, MLP: Neural network building blocks
- Bigram language models: Both counting-based and neural network versions
"""

import math
import random
import matplotlib.pyplot as plt
import urllib.request
import os

# ============================================================================
# Value Class: Core Autograd Engine
# ============================================================================

class Value:
    def __init__(self, data, _children=(), _op=''):
        """Initialize a Value node in the computational graph.
        
        Args:
            data: The scalar value
            _children: Tuple of parent nodes (for graph construction)
            _op: Operation that created this node (for visualization)
        """
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

    def log(self):
        """Natural logarithm: z = ln(x)
        
        Backward: x̄ = z̄ / x
        """
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward

        return out

    def backward(self):
        """Reverse-mode automatic differentiation (backpropagation).
        
        Computes gradients for all nodes in the computational graph by:
        1. Building topological order via DFS
        2. Seeding output gradient = 1.0
        3. Calling _backward() for each node in reverse topological order
        """
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

# ============================================================================
# Neural Network Building Blocks
# ============================================================================

class Neuron:
    """Single neuron: computes tanh(weighted_sum + bias)"""
    
    def __init__(self, nin: int):
        """Initialize neuron with random weights and zero bias.
        
        Args:
            nin: Number of input dimensions
        """
        self.w = [Value(random.gauss(0, 0.1)) for _ in range(nin)]
        self.b = Value(0.0)
    
    def __call__(self, x: list[Value], activation=True) -> Value:
        """Forward pass: activation(sum(w_i * x_i) + b)
        
        Args:
            x: List of input Value objects
            activation: If True, apply tanh activation; else return raw sum
        """
        out = sum((wi * xi) for wi, xi in zip(self.w, x)) + self.b
        if activation:
            return out.tanh()
        else:
            return out

    def parameters(self) -> list[Value]:
        """Return list of all trainable parameters (weights + bias)."""
        return self.w + [self.b]


class Layer:
    """A layer of neurons: applies same transformation to all inputs."""
    
    def __init__(self, nin, nout, activation=True):
        """Initialize layer with nout neurons, each taking nin inputs.
        
        Args:
            nin: Number of input dimensions
            nout: Number of neurons (output dimensions)
            activation: Whether to apply activation function
        """
        self.neurons = [Neuron(nin) for _ in range(nout)]
        self.activation = activation
       
    def __call__(self, x: list[Value]):
        """Forward pass through all neurons in the layer."""
        return [n(x, self.activation) for n in self.neurons]

    def parameters(self):
        """Return all parameters from all neurons in this layer."""
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Multi-Layer Perceptron: stack of fully-connected layers."""
    
    def __init__(self, nin, nouts):
        """Initialize MLP with specified architecture.
        
        Args:
            nin: Input dimension
            nouts: List of output dimensions for each layer
                   Final layer has no activation (for logits)
        """
        self.layers = []
        sizes = [nin] + nouts
        for i in range(len(sizes) - 1):
            # No activation on final layer (for logits in classification)
            use_activation = (i < len(sizes) - 2)
            self.layers.append(Layer(sizes[i], sizes[i + 1], activation=use_activation))
    
    def __call__(self, x: list[Value]):
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        """Return all trainable parameters from all layers."""
        return [p for layer in self.layers for p in layer.parameters()]

# ============================================================================
# Sine Fitting Example
# ============================================================================

def generate_data():
    """Generate training data for sin(x) fitting."""
    xs = [x / 100.0 for x in range(-300, 301, 5)]  # -3.0 to 3.0 step ~0.05
    ys = [math.sin(x) for x in xs]
    return xs, ys


def train(mlp, xs, ys, lr_start, lr_end, epochs):
    """Train MLP to fit sin(x) using MSE loss and linear learning rate decay."""
    for i in range(epochs):
        # Forward pass
        y_pred = [mlp([Value(x)]) for x in xs]
        loss = sum((yp - ygt)**2 for yp, ygt in zip(y_pred, ys)) / len(ys)
        
        # Zero gradients
        for p in mlp.parameters():
            p.grad = 0.0
        
        # Backward pass
        loss.backward()
        
        # Learning rate decay (linear)
        lr = lr_start if epochs == 1 else lr_start - (lr_start - lr_end) * (i / (epochs - 1))
        
        # Update parameters
        for p in mlp.parameters():
            p.data -= lr * p.grad
        
        # Logging
        if i % 50 == 0 or i == epochs - 1:
            print(f"epoch {i:03d} loss {loss.data:.6f} lr {lr:.5f}")

def plot_results(mlp, xs_train, ys_train):
    """Plot model predictions vs ground truth sin(x) curve."""
    # Generate fine grid for smooth curve
    xs_fine = [x / 100.0 for x in range(-300, 301)]  # step 0.01
    ys_true = [math.sin(x) for x in xs_fine]
    ys_pred = [mlp([Value(x)]).data for x in xs_fine]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(xs_train, ys_train, color='blue', s=10, label='ground truth', alpha=0.6)
    plt.plot(xs_fine, ys_pred, color='red', linewidth=2, label='model prediction')
    plt.title('Perfect sin(x) fit with from-scratch autograd')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sine_fit.png', dpi=150)
    plt.close()
    print("Plot saved as sine_fit.png")

# ============================================================================
# Bigram Language Model: Counting-Based Version
# ============================================================================

def download_text():
    """Download tiny_shakespeare.txt if not present, return text as string."""
    filename = 'tiny_shakespeare.txt'
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Downloaded {len(text)} characters")
    
    return text

def build_vocab(text):
    """Build character vocabulary mappings.
    
    Returns:
        stoi: Dictionary mapping character -> index
        itos: List mapping index -> character
        vocab_size: Number of unique characters
    """
    unique_chars = sorted(set(text))

    itos = list(unique_chars)
    stoi = {}
    for i, ch in enumerate(itos):
        stoi[ch] = i
    
    vocab_size = len(itos)

    return stoi, itos, vocab_size

def create_dataset(text, stoi):
    """Create bigram training pairs from text.
    
    Returns:
        xs: List of previous character indices
        ys: List of next character indices
    """
    xs = []
    ys = []

    for i in range(len(text) - 1):
        ch1 = text[i]
        ch2 = text[i + 1]

        x = stoi[ch1]
        y = stoi[ch2]

        xs.append(x)
        ys.append(y)

    return xs, ys

def build_counts(xs, ys, vocab_size):
    """Build bigram counts table: counts[prev_char][next_char] = frequency."""
    counts = [[0] * vocab_size for _ in range(vocab_size)]

    for i in range(len(xs)):
        prev = xs[i]
        next_char = ys[i]

        counts[prev][next_char] += 1
    
    return counts

def counts_to_probs(counts, vocab_size):
    """Convert counts table to probability distributions (row-normalize).
    
    Each row sums to 1.0, representing P(next_char | prev_char).
    """
    probs = [[0] * vocab_size for _ in range(vocab_size)]

    for i in range(vocab_size):
        row_sum = sum(counts[i][j] for j in range(vocab_size))

        if row_sum == 0:
            # Uniform distribution if character never appears
            probs[i] = [1 / vocab_size for j in range(vocab_size)]
        else:
            for j in range(vocab_size):
                probs[i][j] = counts[i][j] / row_sum
    
    return probs


def sample_next_char(prev_char_id, probs, itos):
    """Sample next character using roulette wheel sampling based on probabilities."""
    dist = probs[prev_char_id]
    r = random.random()
    
    cumsum = 0.0
    sample_index = len(dist) - 1  # Fallback to last character
    for i in range(len(dist)):
        cumsum += dist[i]
        if cumsum >= r:
            sample_index = i
            break
    
    sampled_char = itos[sample_index]
    return sampled_char

def evaluate_loss(xs, ys, probs):
    """Compute average negative log-likelihood (cross-entropy) loss."""
    total_nll = 0.0
    n = len(xs)
    
    for i in range(n):
        prev_id = xs[i]
        true_next = ys[i]

        prob = probs[prev_id][true_next]
        if prob == 0:
            prob = 1e-10  # Avoid log(0)
        
        nll = -math.log(prob)
        total_nll += nll
    
    average_nll = total_nll / n

    return average_nll

def generate_text(probs, itos, stoi, start_char, num_chars):
    """Generate text character-by-character using counting-based bigram model."""
    result_text = start_char
    current_char = start_char
    current_id = stoi[current_char]

    for i in range(num_chars):
        next_char = sample_next_char(current_id, probs, itos)
        result_text += next_char

        current_char = next_char
        current_id = stoi[current_char]

    return result_text

# ============================================================================
# Bigram Language Model: Neural Network Version
# ============================================================================

def one_hot_encode(char_id, vocab_size):
    """Convert character index to one-hot vector of Value objects."""
    one_hot = [Value(0.0) for _ in range(vocab_size)]
    one_hot[char_id] = Value(1.0)
    return one_hot


def softmax(logits):
    """Convert logits to probability distribution using softmax.
    
    Uses max subtraction for numerical stability.
    """
    max_val = max(v.data for v in logits)    
    exp_logits = [(logit - max_val).exp() for logit in logits]
    
    # Manually accumulate sum to avoid recursion issues with Python's sum()
    sum_exp = Value(0.0)
    for exp_logit in exp_logits:
        sum_exp = sum_exp + exp_logit
    
    probs = [exp_logit / sum_exp for exp_logit in exp_logits]
    return probs


def cross_entropy_loss(probs, target_id):
    """Compute cross-entropy loss: -log(prob[target])."""
    target_prob = probs[target_id]
    return -target_prob.log()


def forward_bigram_nn(prev_char_id, mlp, vocab_size):
    """Forward pass: one-hot -> MLP -> logits -> softmax -> probabilities."""
    one_hot = one_hot_encode(prev_char_id, vocab_size)
    logits = mlp(one_hot)
    probs = softmax(logits)
    return probs

def train_bigram_nn(mlp, xs, ys, vocab_size, epochs, lr_start, lr_end):
    """Train neural network bigram model using gradient descent."""
    for epoch in range(epochs):
        # Accumulate losses in a list instead of chaining additions
        losses = []

        # Forward pass: accumulate loss over all examples
        for i in range(len(xs)):
            prev_id = xs[i]
            true_next_id = ys[i]

            probs = forward_bigram_nn(prev_id, mlp, vocab_size)
            loss_i = cross_entropy_loss(probs, true_next_id)
            losses.append(loss_i)
        
        # Sum all losses at once (more efficient graph structure)
        total_loss = Value(0.0)
        for loss in losses:
            total_loss = total_loss + loss
        
        avg_loss = total_loss / len(xs)
        
        # Zero gradients
        for p in mlp.parameters():
            p.grad = 0.0

        # Backward pass
        avg_loss.backward()

        # Learning rate decay (linear)
        lr = lr_start - (lr_start - lr_end) * (epoch / (epochs - 1)) if epochs > 1 else lr_start

        # Update parameters
        for p in mlp.parameters():
            p.data -= lr * p.grad
        
        # Logging
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'epoch: {epoch}, average loss: {avg_loss.data:.4f}, learning rate: {lr:.5f}')

def generate_bigram_nn(mlp, itos, stoi, vocab_size, start_char, num_chars):
    """Generate text character-by-character using trained neural network."""
    result_text = start_char
    current_id = stoi[start_char]
    
    for i in range(num_chars):
        # Forward pass to get probabilities
        probs = forward_bigram_nn(current_id, mlp, vocab_size)
        prob_values = [p.data for p in probs]
        
        # Sample next character (roulette wheel)
        r = random.random()
        cumsum = 0.0
        for j in range(len(prob_values)):
            cumsum += prob_values[j]
            if cumsum >= r:
                sample = j
                break
        
        next_char = itos[sample]
        result_text += next_char
        current_id = sample

    return result_text


# ============================================================================
# Main: Run Examples
# ============================================================================

if __name__ == "__main__":
    '''# Sine fit training
    print("\n" + "="*50)
    print("Training MLP to fit sin(x)")
    print("="*50)
    
    xs, ys = generate_data()
    mlp = MLP(1, [16, 16, 1])
    train(mlp, xs, ys, lr_start=0.1,lr_end=.05, epochs=1000)
    plot_results(mlp, xs, ys)'''

    # Bigram language model
    text = download_text()
    stoi, itos, vocab_size = build_vocab(text)
    xs, ys = create_dataset(text, stoi)

    '''counts = build_counts(xs, ys, vocab_size)
    probs = counts_to_probs(counts, vocab_size)

    loss = evaluate_loss(xs, ys, probs)
    print(f"Average NLL loss: {loss:.4f}")

    generated = generate_text(probs, itos, stoi, start_char='\n', num_chars=1000)
    print("\nGenerated text:")
    print(generated)'''

    # Neural Net Bigram Model
    print("\n" + "="*50)
    print("Training Neural Net Bigram Model")
    print("="*50)

    # Create MLP: vocab_size input (one-hot) -> hidden -> vocab_size output (logits)
    mlp_bigram = MLP(vocab_size, [64, vocab_size])

    # Train on small subset for speed (pure Python is slow)
    train_bigram_nn(mlp_bigram, xs[:50], ys[:50], vocab_size,
                    epochs=100, lr_start=0.5, lr_end=0.01)

    # Generate text from neural net
    print("\nGenerated text (neural net):")
    generated_nn = generate_bigram_nn(mlp_bigram, itos, stoi, vocab_size,
                                      start_char='\n', num_chars=200)
    print(generated_nn)