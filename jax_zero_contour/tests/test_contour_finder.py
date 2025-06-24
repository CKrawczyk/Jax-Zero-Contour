import unittest
import jax.numpy as jnp
import logging
import jax

from jax_zero_contour import (
    ZeroSolver,
    split_curves
)


@jax.tree_util.Partial
def f(pos):
    # The zeros of this function are circles
    # with radii equal to the ints
    r = jnp.sqrt(jnp.sum(pos**2, axis=0) + 1e-15)
    return jnp.sinc(r)


@jax.tree_util.Partial
def f_no_contour(pos):
    # This function as no zeros
    r = jnp.sqrt(jnp.sum(pos**2, axis=0) + 1e-15)
    return jnp.sinc(r) + 0.5


class TestZeroContourFinder(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_contour_1(self):
        zs = ZeroSolver(tol=1e-7)
        path, stopping_condition = zs.zero_contour_finder(
            f,
            jnp.array([0.0, 0.6])
        )
        output_r = jnp.sqrt((path['path'][0]**2).sum(axis=1))
        fdx = jnp.isfinite(output_r)
        self.assertTrue(jnp.allclose(output_r[fdx], 1))
        self.assertTrue(jnp.abs(path['value'][0][fdx]).max() < 1e-6)
        self.assertEqual(stopping_condition.tolist(), [[2, 2]])

    def test_contour_2(self):
        zs = ZeroSolver(tol=1e-7, forward_mode_differentiation=True)
        path, stopping_condition = zs.zero_contour_finder(
            f,
            jnp.array([0.0, 1.6])
        )
        output_r = jnp.sqrt((path['path'][0]**2).sum(axis=1))
        fdx = jnp.isfinite(output_r)
        self.assertTrue(jnp.allclose(output_r[fdx], 2))
        self.assertTrue(jnp.abs(path['value'][0][fdx]).max() < 1e-6)
        self.assertEqual(stopping_condition.tolist(), [[2, 2]])

    def test_contour_both(self):
        zs = ZeroSolver(tol=1e-7, forward_mode_differentiation=True)
        path, stopping_condition = zs.zero_contour_finder(
            f,
            jnp.array([[0.0, 0.6], [0.0, 1.6]])
        )
        output_r = jnp.sqrt((path['path']**2).sum(axis=2))
        fdx = jnp.isfinite(output_r)
        self.assertTrue(jnp.allclose(output_r[0][fdx[0]], 1))
        self.assertTrue(jnp.allclose(output_r[1][fdx[1]], 2))
        self.assertTrue(jnp.abs(path['value'][fdx]).max() < 1e-6)
        self.assertEqual(stopping_condition.tolist(), [[2, 2], [2, 2]])

    def test_contour_2_small_batch(self):
        zs = ZeroSolver(tol=1e-7)
        path, stopping_condition = zs.zero_contour_finder(
            f,
            jnp.array([0.0, 1.6]),
            delta=0.01,
            N=1000
        )
        output_r = jnp.sqrt((path['path'][0]**2).sum(axis=1))
        fdx = jnp.isfinite(output_r)
        self.assertTrue(jnp.allclose(output_r[fdx], 2))
        self.assertTrue(jnp.abs(path['value'][0][fdx]).max() < 1e-6)
        self.assertEqual(stopping_condition.tolist(), [[0, 2]])

    def test_no_contour(self):
        zs = ZeroSolver(tol=1e-7)
        path, _ = zs.zero_contour_finder(
            f_no_contour,
            jnp.array([0.0, 1.6])
        )
        finite_path = jnp.isfinite(path['path'])
        finite_value = jnp.isfinite(path['value'])
        self.assertFalse(finite_path.any())
        self.assertFalse(finite_value.any())

    # def test_no_contour(self):
    #     zs = ZeroSolver()
    #     with self.assertRaises(jaxlib.xla_extension.XlaRuntimeError):
    #         print(zs.zero_contour_finder(
    #             f_no_contour,
    #             jnp.array([0.0, 1.6]),
    #             tol=1e-7,
    #             silent_fail=False
    #         ))

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

    def test_path_reduce(self):
        zs = ZeroSolver()
        data = {
            'path': jnp.array([
                [[0.0, 0.0], [0.0, 0.0], [jnp.nan, jnp.nan]],
                [[1.0, 1.0], [jnp.nan, jnp.nan], [jnp.nan, jnp.nan]]
            ]),
            'value': jnp.array([
                [0.0, 0.0, jnp.nan],
                [0.0, jnp.nan, jnp.nan]
            ])
        }
        expected = {
            'path': [
                jnp.array([[0.0, 0.0], [0.0, 0.0]]),
                jnp.array([[1.0, 1.0]])
            ],
            'value': [
                jnp.array([0.0, 0.0]),
                jnp.array([0.0])
            ]
        }
        result = zs.path_reduce(data)
        self.assertEqual(result.keys(), expected.keys())
        self.assertEqual(len(result['path']), len(expected['path']))
        self.assertEqual(len(result['value']), len(expected['value']))
        for i in range(len(expected['path'])):
            self.assertTrue(jnp.allclose(result['path'][i], expected['path'][i]))
            self.assertTrue(jnp.allclose(result['value'][i], expected['value'][i]))
