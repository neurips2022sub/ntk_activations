"""Example of automatically deriving the closed-form NTK from NNGP."""

from jax import numpy as np, random
from neural_tangents import stax
from ntk_activations import stax_extensions


# Consider the normalized exponential kernel from
# https://arxiv.org/abs/2003.02237 (page 6).
def nngp_fn(cov12, var1, var2):
  prod = np.sqrt(var1 * var2)
  return prod * np.exp(cov12 / prod - 1)


# This kernel has no known corresponding elementwise nonlinearity.
# `stax_extensions.Elementwise` derives the NTK kernel automatically under the
# hood using automatic differentiation, without the need to know the respective
# nonlinearity or computing the integrals by hand.
_, _, kernel_fn = stax.serial(stax.Dense(1),
                              stax_extensions.Elementwise(nngp_fn=nngp_fn))

# Below we construct the kernel using the manually-derived NTK expression.
_, _, kernel_fn_manual = stax.serial(stax.Dense(1),
                                     stax_extensions.ExpNormalized())

key = random.PRNGKey(1)
x1 = random.normal(key, (10, 2))
x2 = random.normal(key, (20, 2))

k_auto = kernel_fn(x1, x2, 'ntk')
k_manual = kernel_fn_manual(x1, x2, 'ntk')

# The two kernels match!
assert np.max(np.abs(k_manual - k_auto)) < 1e-6
print('NTK derived via automatic differentiation matches the hand-derived NTK!')
