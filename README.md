# AMG_Encoder

Python bindings for adaptive multi-grid encoder CUDA kernels. This library provides efficient CUDA implementations for encoding 3D positions into feature vectors using adaptive multi-grid transformations.

## Features

- Fast CUDA kernels for multi-grid encoding
- Support for batched operations
- Automatic gradient computation (PyTorch autograd compatible)
- Half, single, and double precision support
- Efficient memory usage through shared memory optimizations

## Requirements

- Python ≥ 3.11
- PyTorch with CUDA support
- CUDA capable GPU
- C++ compiler with C++14 support

## Installation
From your python environment, run:
```bash
pip install git+https://github.com/skywolf829/AMG_Encoder.git --extra-index-url https://download.pytorch.org/whl/cu124
```
## Usage
Please see [AMGSRN++](https://github.com/skywolf829/AMGSRN) for usage examples within a PyTorch model.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
