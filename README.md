# Metal Flex Attention 

This project demonstrates the implementation of Flex Attention using the MLX framework with custom Metal source kernels. Flex Attention is an efficient attention mechanism designed to reduce memory usage and computational overhead, making it suitable for large-scale models and long sequences.

## Features

- Custom fused kernels for optimized performance
- Metal-native implementation
- Support for sparse attention patterns
- Memory-efficient computation

## Installation

```bash
git clone https://github.com/jckwind/metal-flex-attention
cd metal-flex-attention
pip install -r requirements.txt
```

## Requirements

- MLX
- Python 3.8+
- Additional dependencies listed in `requirements.txt`

## Implementation Details

### Custom Kernels

This implementation uses fused custom kernels for:
- Matrix Multiplication

## Benchmarks


