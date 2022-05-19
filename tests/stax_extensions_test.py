"""Tests for `stax_extensions.py`.

Adapted from
https://github.com/google/neural-tangents/blob/main/tests/stax/elementwise_test.py
"""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as onp
import jax.numpy as np
import jax.random as random
import neural_tangents as nt
from neural_tangents import stax
from ntk_activations import stax_extensions


class ActivationTest(parameterized.TestCase):

  def _test_activation(self, phi, get):
    key1, key2, key_mc = random.split(random.PRNGKey(1), 3)
    x1 = np.cos(random.normal(key1, (3, 2)))
    x2 = np.cos(random.normal(key2, (2, 2)))

    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(1024),
        phi,
        stax.Dense(1 if get == 'ntk' else 1024)
    )

    analytic_kernel = kernel_fn(x1, x2, get, diagonal_spatial=True)
    mc_kernel_fn = nt.monte_carlo_kernel_fn(
        init_fn=init_fn,
        apply_fn=apply_fn,
        key=key_mc,
        n_samples=800,
        implementation=2,
        vmap_axes=0
    )

    if get == 'cov1':
      empirical_kernel = np.diag(mc_kernel_fn(x1, None, 'nngp'))
    else:
      empirical_kernel = mc_kernel_fn(x1, x2, get)

    onp.testing.assert_allclose(analytic_kernel, empirical_kernel,
                                atol=0.01, rtol=0.03)

  @parameterized.product(
    phi=[stax_extensions.Sign, stax_extensions.Sigmoid_like],
    get=['cov1', 'nngp', 'ntk'],
  )
  def test_nonparametric(
      self,
      phi,
      get,
  ):
    self._test_activation(phi(), get)

  @parameterized.product(
      phi=[stax_extensions.Sin, stax_extensions.Cos],
      get=['cov1', 'nngp', 'ntk'],
      a=[2., 0.3],
      b=[1.5, 0.3],
      c=[0., -np.pi/4., np.pi/2.]
  )
  def test_abc(
      self,
      phi,
      get,
      a,
      b,
      c
  ):
    self._test_activation(phi(a=a, b=b, c=c), get)

  @parameterized.product(
      phi=[stax_extensions.Exp, stax_extensions.Gaussian],
      get=['cov1', 'nngp', 'ntk'],
      a=[-0.5, 0.25],
      b=[-0.5, -0.1, 0.1],
  )
  def test_ab(
      self,
      phi,
      get,
      a,
      b,
  ):
    self._test_activation(phi(a=a, b=b), get)

  @parameterized.product(
    phi=[stax_extensions.Rbf],
    get=['cov1', 'nngp', 'ntk'],
    gamma=[1e-6, 1e-4, 1e-2, 1.0, 2.],
  )
  def test_rbf(
      self,
      phi,
      get,
      gamma
  ):
    self._test_activation(phi(gamma=gamma), get)

  @parameterized.product(
    phi=[stax_extensions.Monomial, stax_extensions.RectifiedMonomial],
    get=['cov1', 'nngp', 'ntk'],
    degree=[0, 1, 2, 3, 4, 5],
  )
  def test_monomial(
      self,
      phi,
      get,
      degree
  ):
    self._test_activation(phi(degree=degree), get)

  @parameterized.product(
    phi=[stax_extensions.Gelu],
    get=['cov1', 'nngp', 'ntk'],
    approximate=[True, False],
  )
  def test_gelu(
      self,
      phi,
      get,
      approximate
  ):
    self._test_activation(phi(approximate=approximate), get)

  @parameterized.product(
      phi=[stax_extensions.ExpNormalized],
      get=['nngp', 'ntk'],
      do_clip=[True, False],
      gamma=[1., 2., 0.5],
      shift=[-1., 0., 1.]
  )
  def test_exp_normalized(
      self,
      phi,
      get,
      do_clip,
      gamma,
      shift
  ):
    key = random.PRNGKey(0)
    x1 = random.normal(key, (2, 3))
    x2 = random.normal(key, (3, 3))

    phi = phi(do_clip=do_clip, gamma=gamma, shift=shift)

    _, _, kernel_fn = stax.serial(
        stax.Dense(1),
        phi,
        stax.Dense(1),
        phi,
        stax.Dense(1)
    )
    k_12 = kernel_fn(x1, x2, get=get)
    self.assertEqual(k_12.shape, (x1.shape[0], x2.shape[0]))

    k_11 = kernel_fn(x1, None, get=get)
    self.assertEqual(k_11.shape, (x1.shape[0],) * 2)
    self.assertGreater(np.min(np.linalg.eigvalsh(k_11)), 0)

    k_22 = kernel_fn(x2, None, get=get)
    self.assertEqual(k_22.shape, (x2.shape[0],) * 2)
    self.assertGreater(np.min(np.linalg.eigvalsh(k_22)), 0)

  def test_exp_normalized_ntk(self):
    def nngp_fn(cov12, var1, var2):
      prod = np.sqrt(var1 * var2)
      return prod * np.exp(cov12 / prod - 1)

    _, _, kernel_fn = stax.serial(stax.Dense(1),
                                  stax_extensions.Elementwise(nngp_fn=nngp_fn))

    _, _, kernel_fn_manual = stax.serial(stax.Dense(1),
                                         stax_extensions.ExpNormalized())

    key = random.PRNGKey(1)
    x1 = random.normal(key, (5, 2))
    x2 = random.normal(key, (6, 2))

    k = kernel_fn(x1, x2, 'ntk')
    k_manual = kernel_fn_manual(x1, x2, 'ntk')
    onp.testing.assert_allclose(k_manual, k, rtol=0.05, atol=0.02)

  @parameterized.product(
      phi=[stax_extensions.Hermite],
      get=['nngp', 'ntk'],
      degree=[0, 1, 2, 3, 4, 5, 6],
  )
  def test_hermite(
      self,
      phi,
      get,
      degree,
  ):
    phi = phi(degree=degree)
    key1, key2, key_mc = random.split(random.PRNGKey(1), 3)

    x1 = np.cos(random.normal(key1, (2, 3)))
    x2 = np.cos(random.normal(key2, (3, 3)))

    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(2 * 2 * 8192),
        stax.LayerNorm(),
        phi,
        stax.Dense(1) if get == 'ntk' else stax.Identity()
    )

    analytic_kernel = kernel_fn(x1, x2, get)
    mc_kernel_fn = nt.monte_carlo_kernel_fn(
        init_fn=init_fn,
        apply_fn=apply_fn,
        key=key_mc,
        n_samples=800,
        implementation=2
    )
    mc_kernel = mc_kernel_fn(x1, x2, get)
    rtol = degree / 2. * 1e-2
    onp.testing.assert_allclose(mc_kernel, analytic_kernel, rtol=rtol, atol=0.1)


class ElementwiseTest(parameterized.TestCase):

  @parameterized.product(
      phi=[stax.Identity, stax.Erf, stax.Relu, stax_extensions.Sin],
  )
  def test_elementwise(
      self,
      phi
  ):

    def nngp_fn(cov12, var1, var2):
      if phi == stax.Identity:
        res = cov12

      elif phi == stax.Erf:
        prod = (1 + 2 * var1) * (1 + 2 * var2)
        res = np.arcsin(2 * cov12 / np.sqrt(prod)) * 2 / np.pi

      elif phi == stax.Relu:
        prod = var1 * var2
        sqrt = np.sqrt(np.maximum(prod - cov12 ** 2, 1e-30))
        angles = np.arctan2(sqrt, cov12)
        dot_sigma = (1 - angles / np.pi) / 2
        res = sqrt / (2 * np.pi) + dot_sigma * cov12

      elif phi == stax_extensions.Sin:
        sum_ = (var1 + var2)
        s1 = np.exp((-0.5 * sum_ + cov12))
        s2 = np.exp((-0.5 * sum_ - cov12))
        res = (s1 - s2) / 2

      else:
        raise NotImplementedError(phi)

      return res

    _, _, kernel_fn = stax.serial(
      stax.Dense(1),
      stax_extensions.Elementwise(nngp_fn=nngp_fn),
      stax.Dense(1),
      stax_extensions.Elementwise(nngp_fn=nngp_fn)
    )
    _, _, kernel_fn_manual = stax.serial(
      stax.Dense(1),
      phi(),
      stax.Dense(1),
      phi()
    )

    key1, key2 = random.split(random.PRNGKey(1))
    x1 = random.normal(key1, (5, 3))
    x2 = random.normal(key2, (6, 3))

    k = kernel_fn(x1, x2)
    k_manual = kernel_fn_manual(x1, x2)

    onp.testing.assert_allclose(k_manual.nngp, k.nngp, rtol=0.001, atol=0.001)
    onp.testing.assert_allclose(k_manual.ntk, k.ntk, rtol=0.001, atol=0.001)


class ElementwiseNumericalTest(parameterized.TestCase):

  @parameterized.product(
      phi=[stax.Erf, stax_extensions.Gelu, stax_extensions.Sin],
      get=['nngp', 'ntk'],
  )
  def test_elementwise_numerical(
      self,
      phi,
      get
  ):
    key1, key2 = random.split(random.PRNGKey(1))

    x1 = random.normal(key1, (3, 7))
    x2 = random.normal(key2, (5, 7))

    _, _, kernel_fn = stax.serial(
        stax.Dense(1),
        phi(),
        stax.Dense(1)
    )
    analytic_kernel = kernel_fn(x1, x2, get)

    _, _, kernel_fn = stax.serial(
        stax.Dense(1),
        stax_extensions.ElementwiseNumerical(lambda x: phi()[1]((), x), deg=25),
        stax.Dense(1)
    )
    numerical_kernel = kernel_fn(x1, x2, get)

    onp.testing.assert_allclose(analytic_kernel, numerical_kernel, rtol=2e-3)


if __name__ == '__main__':
  absltest.main()
