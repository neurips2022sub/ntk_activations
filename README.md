# Fast Neural Kernel Embeddings for General Activations

Anonymous code supplement for the [NeurIPS 2022 submission](https://openreview.net/forum?id=yLilJ1vZgMe).

This codebase provides [`neural_tangents`](https://github.com/google/neural-tangents) implementations of NTK and NNGP kernels for new nonlinearities:

* `Elementwise` (derives the NTK kernel in closed form via automatic differentiation from any provided NNGP kernel expression)
* `ElementwiseNumerical` (uses Gaussian quadrature and autodiff to approximate both NNGP and NTK given only the elementwise nonlinearity function)
* `Cos`
* `ExpNormalized`
* `Exp`
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

To run tests:
```commandline
python tests/stax_extensions_test.py
```