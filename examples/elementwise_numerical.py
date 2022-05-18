"""Example of approximating the NNGP and NTK using quadrature and autodiff."""

import jax.nn
from jax import random, numpy as np
from neural_tangents import stax
from ntk_activations import stax_extensions

key1, key2 = random.split(random.PRNGKey(1))

x1 = random.normal(key1, (10, 3))
x2 = random.normal(key2, (20, 3))

# Consider a nonlinearity for which we know the closed-form expression (GeLU).
_, _, kernel_fn_closed_form = stax.serial(
  stax.Dense(1),
  stax_extensions.Gelu(),  # Contains the closed-form GeLU NNGP/NTK expression.
  stax.Dense(1)
)
kernel_closed_form = kernel_fn_closed_form(x1, x2)

# Construct the layer from only the elementwise forward-pass GeLU.
_, _, kernel_fn_numerical = stax.serial(
  stax.Dense(1),
  # Approximation using Gaussian quadrature and autodiff.
  stax_extensions.ElementwiseNumerical(jax.nn.gelu, deg=25),
  stax.Dense(1)
)
kernel_numerical = kernel_fn_numerical(x1, x2)

# The two kernels are close!
assert np.max(np.abs(kernel_closed_form.nngp - kernel_numerical.nngp)) < 1e-3
assert np.max(np.abs(kernel_closed_form.ntk - kernel_numerical.ntk)) < 1e-3
print('Gaussian quadrature approximation of the kernel is accurate!')
