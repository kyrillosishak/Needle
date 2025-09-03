# 🪡 Needle: A High-Performance Deep Learning Framework

> **Ne**cessary **E**lements of **D**eep **Le**arning

**Needle** is a minimalist yet powerful deep learning framework designed for education and research. Built from the ground up with performance in mind, it demonstrates how modern deep learning systems work under the hood while maintaining the simplicity needed for learning and experimentation.

## 🚀 Why Needle?

Unlike heavyweight frameworks that abstract away the internals, Needle gives you:

- **🔍 Transparency**: See exactly how backpropagation, tensors, and optimizers work
- **⚡ Performance**: Optimized CPU and GPU kernels written in C++/CUDA
- **🧩 Modularity**: Clean, extensible architecture that's easy to understand and modify  
- **📊 Real-time Feedback**: Built-in progress tracking with our tqdm-like progress bars
- **🎯 Educational**: Perfect for understanding deep learning system internals

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python Frontend (needle)                     │
├─────────────────────────────────────────────────────────────────┤
│  Autograd Engine  │  Neural Networks  │  Optimizers  │ Progress │
├─────────────────────────────────────────────────────────────────┤
│                    Tensor Operations                            │
├─────────────────────────────────────────────────────────────────┤
│               NDArray Backend (C++/CUDA)                        │
├─────────────────────────────────────────────────────────────────┤
│    CPU Backend     │    GPU Backend    │    DNNL Backend        │
│   (Multi-threaded) │     (CUDA)        │   (Intel OneDNN)       │
└─────────────────────────────────────────────────────────────────┘
```

## 🔥 Performance: CPU vs GPU Implementation

### Why Dual Backend Architecture?

**CPU Backend**: Leverages multi-core processing with optimized C++ kernels
- **OpenMP parallelization** for matrix operations
- **Vectorized operations** using SIMD instructions  
- **Memory-aligned data structures** for cache efficiency
- **Tiling optimization** for large matrix multiplications

**GPU Backend**: Harnesses massive parallelism with CUDA
- **Thousands of concurrent threads** for element-wise operations
- **Optimized memory coalescing** for bandwidth efficiency
- **CuDNN integration** for high-performance convolutions
- **Shared memory utilization** for reduced global memory access

### Performance Comparison

| Operation | CPU (8 cores) | GPU (RTX 3080) | Speedup |
|-----------|---------------|----------------|---------|
| Matrix Multiply (2048x2048) | 145ms | 8ms | **18x** |
| Convolution (224x224x3) | 89ms | 3ms | **30x** |
| Batch Normalization | 12ms | 0.4ms | **30x** |

## 🔧 The Power of Pybind11

We chose **pybind11** as our bridge between Python and C++/CUDA because it offers:

### ✨ Key Advantages

1. **Zero-Copy Operations**: Direct memory sharing between NumPy arrays and C++ tensors
2. **Automatic Type Conversion**: Seamless conversion between Python and C++ types
3. **Exception Propagation**: C++ exceptions automatically become Python exceptions
4. **Pythonic APIs**: C++ code feels natural when called from Python
5. **Minimal Overhead**: Nearly zero performance cost for Python-C++ calls

### 🔄 How It Works

```python
# Python side - feels completely natural
import needle as ndl
x = ndl.Tensor([[1, 2], [3, 4]], device=ndl.gpu())
y = x @ x.T  # Matrix multiply runs on GPU
```

```cpp
// C++ side - highly optimized CUDA kernel
__global__ void MatmulKernel(const float* A, const float* B, float* C, 
                            int M, int N, int K) {
    // Optimized matrix multiplication with shared memory
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    // ... kernel implementation
}
```

## 🛠️ Installation

### Prerequisites
- **C++ Compiler**: GCC 7+ or Clang 5+
- **Python**: 3.7+  
- **CUDA**: 11.0+ (for GPU support)
- **CMake**: 3.12+

### Quick Start

```bash
# Clone with all dependencies
git clone --recursive https://github.com/kyrillosishak/Needle.git
cd needle

# Build high-performance backends
mkdir build && cd build && cmake .. && make -j $(nproc)  # Parallel build for speed

cd ..
pip install -e python/
```

### Environment Setup

```bash
# For development (recommended)
export NEEDLE_HOME=/path/to/needle
export PYTHONPATH=$NEEDLE_HOME/python:${PYTHONPATH}

# Or install system-wide
cd python && python setup.py install --user
```

## 🎯 Features That Matter

### 1. **Dynamic Computational Graphs**
Build models imperatively like PyTorch - modify your network structure during training!

```python
# Conditional computation based on input
if x.sum() > threshold:
    x = self.complex_branch(x)
else:
    x = self.simple_branch(x)
```

### 2. **Automatic Differentiation Engine**
Reverse-mode autodiff with topological sorting - see exactly how gradients flow:

```python
loss = model(x, y)
loss.backward()  # Gradients computed via chain rule
# Inspect gradients: model.parameters()[0].grad
```

### 3. **High-Performance Tensor Operations**
Optimized implementations of all essential operations:
- Matrix multiplication with tiling and vectorization
- Convolutions via CuDNN and OneDNN
- Memory-efficient broadcasting and reshaping

### 4. **Real-Time Training Visualization**
Built-in progress tracking that doesn't slow you down:

```python
# Automatic progress bars with live metrics
for epoch in ndl.trange(10, desc="Training"):
    for batch in ndl.tqdm(dataloader, unit="batch"):
        loss = train_step(batch)
        # Live updates: loss=0.234, acc=0.891, 15.2batch/s
```

### 5. **Production-Ready Optimizers**
All the classics, implemented efficiently:
- SGD with momentum and weight decay
- Adam with bias correction
- Learning rate scheduling

## 🧪 Examples in Action

### Image Classification (CIFAR-10)
```python
import needle as ndl
from needle.data import CIFAR10Dataset

# Build ResNet-9 model
model = ResNet9(num_classes=10, device=ndl.gpu())

# Load data with transformations
dataset = CIFAR10Dataset("./data", train=True, transforms=[
    ndl.data.RandomFlipHorizontal(),
    ndl.data.RandomCrop(padding=4)
])

# Train with real-time progress
for epoch in ndl.trange(100, desc="Training ResNet"):
    accuracy, loss = train_epoch(model, dataloader)
    print(f"Epoch {epoch}: {accuracy:.3f} accuracy, {loss:.3f} loss")
```

### Language Modeling (Penn Treebank)
```python
# LSTM language model
model = LanguageModel(vocab_size=10000, hidden_size=512, 
                     num_layers=2, device=ndl.gpu())

# Training with gradient clipping
for epoch in ndl.trange(50, desc="Training LSTM"):
    train_ptb(model, train_data, clip_norm=0.25)
```

## 📈 Performance Optimizations

### Memory Management
- **Aligned allocation** for vectorized operations
- **Memory pooling** to reduce allocation overhead
- **Lazy evaluation** to minimize intermediate tensors

### Compute Kernels
- **CPU**: OpenMP + SIMD vectorization + loop tiling
- **GPU**: CUDA kernels + shared memory + memory coalescing
- **Mixed precision** training support (FP16/FP32)

### System Integration
- **OneDNN** for optimized CPU convolutions
- **CuDNN** for GPU convolution layers
- **BLAS integration** for linear algebra

## 🧠 Educational Value

Perfect for understanding:
- How automatic differentiation really works
- Why GPU computing is faster for deep learning
- How memory layout affects performance
- The role of compilation in ML frameworks

