# Fast Neural Kernel Embeddings for General Activations

[![Tests](https://github.com/neurips2022sub/ntk_activations/actions/workflows/test.yaml/badge.svg)](https://github.com/neurips2022sub/ntk_activations/actions/workflows/test.yaml)
[![Pytype](https://github.com/neurips2022sub/ntk_activations/actions/workflows/pytype.yaml/badge.svg)](https://github.com/neurips2022sub/ntk_activations/actions/workflows/pytype.yaml)
[![Coverage](https://codecov.io/gh/neurips2022sub/ntk_activations/branch/main/graph/badge.svg)](https://codecov.io/gh/neurips2022sub/ntk_activations)

**Anonymous code supplement for the [NeurIPS 2022 submission](https://openreview.net/forum?id=yLilJ1vZgMe).**

This codebase provides [`neural_tangents`](https://github.com/google/neural-tangents) implementations of NTK and NNGP kernels for new nonlinearities:

* `Elementwise` (derives the NTK kernel in closed form via automatic differentiation from any provided NNGP kernel expression)
* `ElementwiseNumerical` (uses Gaussian quadrature and autodiff to approximate both NNGP and NTK given only the elementwise nonlinearity function)
* `Cos`
* `ExpNormalized`
* `Exp`
* `Gabor`
* `Gaussian`
* `Gelu`
* `Hermite`
* `Monomial`
* `Polynomial`
* `Rbf`
* `RectifiedMonomial`
* `Sigmoid_like`
* `Sign`
* `Sin`

It also provides dual kernel approximation via Hermite expansion. See `ntk_activations/dual_kernels.py`.

## Usage

Clone the package: 
```commandline 
git clone https://github.com/neurips2022sub/ntk_activations.git
cd ntk_activations 
pip install -e .
```

To run examples:
```commandline
python examples/elementwise.py
python examples/elementwise_numerical.py
```
or open the [example Colab](https://colab.research.google.com/github/neurips2022sub/ntk_activations/blob/main/example.ipynb).

To run tests for extensions of [`neural_tangents`](https://github.com/google/neural-tangents):
```commandline
python tests/stax_extensions_test.py
```

To run tests for kernel approximations with Hermite expansion for GeLU activation:
```commandline
python examples/dual_kernel_approx.py --act gelu
```
For other activations such as `relu`, `sin`, `gaussian`, `erf`, `abs`, please replace the argument `gelu` with the other one (e.g., `--act erf`).
