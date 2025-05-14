import unittest
import jax.numpy as jnp
import logging

from jax_zero_contour import (
    zero_contour_finder,
    value_and_grad_wrapper,
    split_curves
)


def f(x, y):
    # The zeros of this function are circles
    # with radii equal to the ints
    r = jnp.sqrt(x**2 + y**2 + 1e-15)
    return jnp.sinc(r)


v_and_g_rev = value_and_grad_wrapper(f)
v_and_g_fwd = value_and_grad_wrapper(f, forward_mode_differentiation=True)


def f_no_contour(x, y):
    # This function as no zeros
    r = jnp.sqrt(x**2 + y**2 + 1e-15)
    return jnp.sinc(r) + 0.5


v_and_g_no_contour = value_and_grad_wrapper(f_no_contour)


class TestZeroContourFinder(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_contour_1(self):
        path, values, stopping_condition = zero_contour_finder(
            v_and_g_rev,
            jnp.array([0.0, 0.6]),
            tol=1e-7
        )
        output_r = jnp.sqrt((path**2).sum(axis=1))
        self.assertTrue(jnp.allclose(output_r, 1))
        self.assertTrue(jnp.abs(values).max() < 1e-6)
        self.assertEqual(stopping_condition.tolist(), [2])

    def test_contour_2(self):
        path, values, stopping_condition = zero_contour_finder(
            v_and_g_fwd,
            jnp.array([0.0, 1.6]),
            tol=1e-7
        )
        output_r = jnp.sqrt((path**2).sum(axis=1))
        self.assertTrue(jnp.allclose(output_r, 2))
        self.assertTrue(jnp.abs(values).max() < 1e-6)
        self.assertEqual(stopping_condition.tolist(), [2])

    def test_contour_2_small_batch(self):
        path, values, stopping_condition = zero_contour_finder(
            v_and_g_rev,
            jnp.array([0.0, 1.6]),
            delta=0.01,
            N=50,
            max_iter=20,
            tol=1e-7
        )
        output_r = jnp.sqrt((path**2).sum(axis=1))
        self.assertTrue(jnp.allclose(output_r, 2))
        self.assertTrue(jnp.abs(values).max() < 1e-6)
        self.assertEqual(stopping_condition.tolist(), [0, 2])

    def test_no_contour(self):
        path, values, stopping_condition = zero_contour_finder(
            v_and_g_no_contour,
            jnp.array([0.0, 1.6]),
            tol=1e-7
        )
        self.assertIsNone(path)
        self.assertIsNone(values)
        self.assertIsNone(stopping_condition)

    def test_split_curves(self):
        x1 = jnp.arange(0.0, 1.0, 0.1)
        y1 = jnp.zeros_like(x1)
        xy1 = jnp.vstack([x1, y1]).T
        x2 = jnp.arange(2.0, 3.0, 0.1)
        y2 = jnp.zeros_like(x2)
        xy2 = jnp.vstack([x2, y2]).T
        xy = jnp.vstack([xy1, xy2])
        expected = [xy1, xy2]
        result = split_curves(xy, threshold=0.11)
        self.assertEqual(len(result), len(expected))
        self.assertTrue(jnp.allclose(result[0], expected[0]))
        self.assertTrue(jnp.allclose(result[1], expected[1]))
