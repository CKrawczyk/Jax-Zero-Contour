'''Find and follow a zero value contour for any 2D function written in Jax.'''

import jax
import jax.numpy as jnp
from jax.tree_util import Partial


class ZeroSolver():
    stopping_conditions = {
        0: 'none',
        1: 'end_point',
        2: 'closed_loop'
    }

    @staticmethod
    def value_and_grad_forward(f):
        # forward mode value and grad
        def value_and_fwd_grad(*args):
            def _wrapper(*args):
                out = f(*args)
                return out, out

            grads, out = jax.jacfwd(_wrapper, argnums=(0,), has_aux=True)(*args)
            return out, grads
        return Partial(value_and_fwd_grad)

    @staticmethod
    def value_and_grad_rev(f):
        # reverse mode value and grad
        return Partial(jax.value_and_grad(f, argnums=(0,), has_aux=False))

    @staticmethod
    def tangent_solve_fwd(g, y):
        # forward mode tangent solve
        j = jax.jacfwd(g)(y)
        j_2 = jnp.sum(j**2)
        return y * j / j_2

    @staticmethod
    def tangent_solve_rev(g, y):
        # reverse mode tangent solve
        j = jax.jacrev(g)(y)
        j_2 = jnp.sum(j**2)
        return y * j / j_2

    @staticmethod
    @Partial(jax.vmap, in_axes=(0, 0, None))
    def step_tangent(pos, grad, delta):
        # take a step perpendicular to the gradient (e.g. Euler-Lagrange)
        alpha = jnp.linalg.norm(grad)
        step = jnp.array([1.0, -1.0]) * grad[::-1]
        return pos + delta * step / alpha

    @staticmethod
    @jax.vmap
    def stack(*args):
        # vectorize stack across batches
        return jnp.vstack([*args])

    @staticmethod
    def swap(x):
        # swap first and second axis
        return jnp.swapaxes(x, 0, 1)

    @staticmethod
    @Partial(jax.vmap, in_axes=(0, 0, None))
    def vec_roll(*args):
        # vectorize roll across first two arguments
        return jnp.roll(*args)

    @staticmethod
    @Partial(jax.vmap, in_axes=(0, 0, None))
    def vec_slice(*args):
        # vectorized dynamic_index_in_dim across first to arguments
        return jax.lax.dynamic_index_in_dim(*args)

    @staticmethod
    def trim_paths(paths, cut_index):
        # write all value on a path to nan after a contour terminates
        n_points = paths['path'].shape[0]
        N = paths['path'].shape[1]
        index = jnp.arange(N)
        index_col = jnp.stack([index, index]).T
        index_full = jnp.stack([index_col] * n_points)
        index_value = jnp.stack([index] * n_points)
        mask_path = index_full > cut_index.reshape(n_points, 1, 1)
        mask_value = index_value > cut_index.reshape(n_points, 1)
        return {
            'path': jnp.where(mask_path, jnp.nan, paths['path']),
            'value': jnp.where(mask_value, jnp.nan, paths['value'])
        }

    @staticmethod
    def stack_and_roll(init_pos, init_h, paths, paths_rev, roll_index):
        # reverse the reverse path and stack it with the initial position
        # and the forward path.  Once finished roll the resulting arrays
        # so the non-nan values are at the start of the array.
        n_points = init_h.shape[0]
        N = paths['path'].shape[1]
        roll_amount = roll_index - N + 1
        return {
            'path': ZeroSolver.vec_roll(
                ZeroSolver.stack(
                    paths_rev['path'][:, ::-1, :],
                    init_pos.reshape(n_points, 1, 2),
                    paths['path']
                ),
                roll_amount,
                0
            ),
            'value': ZeroSolver.vec_roll(
                jnp.hstack([
                    paths_rev['value'][:, ::-1],
                    init_h.reshape(n_points, 1),
                    paths['value']
                ]),
                roll_amount,
                0
            )
        }

    @staticmethod
    def path_reduce(paths):
        '''A helper function to remove the NaN values from a contour path dictionary.
        Because the size of the output is dependent on the inputs this function can
        not be jit'ed.

        Parameters
        ----------
        paths : dict
            output path dictionary from the zero_contour_finder function

        Returns
        -------
        paths: dict
            the paths object with the jax.numpy.nan values removed
        '''
        return {
            'path': [p[jnp.isfinite(p).all(axis=1)] for p in paths['path']],
            'value': [v[jnp.isfinite(v)] for v in paths['value']]
        }

    def __init__(self, tol=1e-6, max_newton=5, forward_mode_differentiation=False):
        '''A class for solving zero contour values for a function.

        Parameters
        ----------
        tol : float, optional
            Newton's steps are used to bring each proposed point on the contour to
            be within this tolerance of zero, by default 1e-6.
        max_newton : int, optional
            The maximum number of Newton's steps to run inside the path integrator,
            by default 5.  To get from the initial guess to a point on the contour
            5 * max_newton steps are used.
        forward_mode_differentiation : bool, optional
            If True use forward mode auto-differentiation, otherwise use reverse mode,
            by default False
        '''
        self.tol = tol
        self.max_newton = max_newton
        self.forward_mode_differentiation = forward_mode_differentiation
        if self.forward_mode_differentiation:
            self.value_and_grad = ZeroSolver.value_and_grad_forward
            self.tangent_solve = ZeroSolver.tangent_solve_fwd
        else:
            self.value_and_grad = ZeroSolver.value_and_grad_rev
            self.tangent_solve = ZeroSolver.tangent_solve_rev

    def threshold_cut(self, paths):
        # nan any positions where the function evaluates to more than 10x the tol
        mask = paths['value'] > 20 * self.tol
        stack_mask = jnp.stack([mask, mask], axis=2)
        return {
            'path': jnp.where(stack_mask, jnp.nan, paths['path']),
            'value': jnp.where(mask, jnp.nan, paths['value'])
        }

    def step_parallel(self, state):
        # Take a Newton's step and calculate the value and grad
        # at the new position
        count, pos, h, grad, f = state
        alpha_2 = jnp.sum(grad**2)
        new_pos = pos - h * grad / alpha_2
        h, (grad,) = self.value_and_grad(f)(new_pos)
        return count + 1, new_pos, h, grad, f

    def parallel_break(self, state, factor=1):
        # Stop Newton's method if the function is within
        # `tol` of zero or the max number of steps is reached
        count, _, h, _, _ = state
        return (jnp.abs(h) > self.tol) & (count <= factor * self.max_newton)

    def step_parallel_tol(self, f, init_guess, factor=1):
        # use while loop to run Newton's method
        h, (grad,) = self.value_and_grad(f)(init_guess)
        state = (1, init_guess, h, grad, f)
        _, pos, h, grad, _ = jax.lax.while_loop(
            Partial(self.parallel_break, factor=factor),
            self.step_parallel,
            state
        )
        return pos, {'value': h, 'grad': grad}

    @Partial(jax.vmap, in_axes=(None, None, 0, None))
    def newton(self, f, init_guess, factor):
        # wrap Newton's method in `custom_root` to make it
        # auto diff correctly
        return jax.lax.custom_root(
            f,
            init_guess,
            Partial(self.step_parallel_tol, factor=factor),
            self.tangent_solve,
            has_aux=True
        )

    def step_one_tp_inner(self, carry):
        # take one Euler-Lagrange step followed by Newton's method
        pos_in, pos_start, _, _, h, grad, delta, f = carry
        pos = ZeroSolver.step_tangent(pos_in, grad, delta)
        pos, aux = self.newton(f, pos, 1)
        h = aux['value']
        grad = aux['grad']
        delta_travel = jnp.linalg.norm(pos_in - pos, axis=1)
        delta_start = jnp.linalg.norm(pos_start - pos, axis=1)
        return pos, h, grad, delta_travel, delta_start

    def null_step_one_tp_inner(self, carry):
        # once all paths have terminated just return zero for everything
        pos_in, _, _, _, h, grad, _, _ = carry
        pos_like = jnp.zeros_like(pos_in)
        h_like = jnp.zeros_like(h)
        return pos_like, h_like, grad, h_like, h_like

    def step_one_tp(self, carry, index):
        pos_in, pos_start, cut, stop_condition, _, _, delta, f = carry
        # check if all paths have terminated
        cond1 = cut == 0
        pos, h, grad, delta_travel, delta_start = jax.lax.cond(
            jnp.any(cond1),
            self.step_one_tp_inner,
            self.null_step_one_tp_inner,
            carry
        )
        # if Newton's mehtod move a point very far from the previous one
        # or very close to the previous one flag it as an endpoint for the path
        cond2 = (delta_travel > 2 * jnp.abs(delta)) | (delta_travel < 0.25 * jnp.abs(delta))
        stop_condition = jnp.where((stop_condition == 0) & cond2, 1, stop_condition)
        # if the new point is close the starting poin flag the path as closed
        cond3 = (delta_start < 1.1 * jnp.abs(delta)) & jnp.all(pos_in != pos_start, axis=1)
        stop_condition = jnp.where((stop_condition == 0) & cond3, 2, stop_condition)
        # record the index if either condition is set this step (the index before is the last
        # valid point of the path)
        cut = jnp.where((cond1) & (cond2 | cond3), index, cut)
        return (pos, pos_start, cut, stop_condition, h, grad, delta, f), {'path': pos, 'value': h}

    def take_steps(self, f, N, pos_start, pos, delta, h, grad):
        # scan over single step function
        carry = (
            pos,
            pos_start,
            jnp.zeros_like(h, dtype=int),
            jnp.zeros_like(h, dtype=int),
            h,
            grad,
            delta,
            f
        )
        final_state, paths = jax.lax.scan(self.step_one_tp, carry, xs=jnp.arange(N))
        return final_state, paths

    def excepting_message(self, failed_init_index):
        # error message to use if initalization fails
        jax.debug.print('Index of failed input(s): {i}', i=jnp.nonzero(failed_init_index)[0])
        raise ValueError(f'No zero contour found after 5 * max_newton ({5 * self.max_newton}) iterations')

    @Partial(jax.jit, static_argnums=(0, 4, 5))
    def zero_contour_finder(self, f, init_guess, delta=0.1, N=1000, silent_fail=True):
        '''Find the zero contour of a 2D function.

        After a path hits an endpoint or closes any further points on the contour are written
        to jax.numpy.nan.  The final output will be shifted so that the finite parts of the contour are
        brought to the front of the array.  The points in the resulting paths are ordered.

        Any points along the contour that have a function evaluation greater than 20 times the tolerance
        are also written to jax.numpy.nan.

        Parts of this code use jax.lax.cond to stop the calculation early when certain termination
        conditions are satisfied. As a result it should not be combined with jax.vmap.

        Parameters
        ----------
        f : function
            The function you want to find the zero contours for, it should have as input
            one positional argument that is an array shape (1, 2)
            (e.g. jnp.array([x_value, y_value])) and returns a single value
        init_guess : jax.numpy.array
            Initial guesses for points near the zero contour, one guess per row.
        delta : float, optional
            The step size to take along the contour when searching for a new point,
            by default 0.1.
        N : int, optional
            The total number of steps to take in *each* direction from the starting point(s).
            The final path will be 2N+1 in size (N points in the forward direction, N points
            in the reverse direction, with the initial point in the middle). By default 1000.
        silent_fail : bool, optional
            If False the code will raise an exception if *any* of the initial points do lead
            to a zero value with the tolerance value within the number of allowed max_newton steps.
            If True the code will continue anyways. By default True

        Returns
        -------
        paths : dict
            The return dictionary will have two keys.  "path": jax.numpy.array with shape (number of
            guesses, 2N+1, 2) with the contours paths for each guess.  "value": jax.numpy.array with
            shape (number of guesses, 2N+1) with the function value at each point on the path
        stop_output : jax.numpy.array
            List containing the stopping conditions for each guess
        '''
        init_guess = jnp.atleast_2d(init_guess)
        init_pos, aux = self.newton(f, init_guess, 5)
        init_h = aux['value']
        init_grad = aux['grad']
        if not silent_fail:
            failed_init_index = ~jnp.isfinite(init_pos).all(axis=1) | (jnp.abs(init_h) > self.tol)
            jax.lax.cond(
                failed_init_index.any(),
                lambda idx: jax.debug.callback(self.excepting_message, idx),
                lambda _: None,
                failed_init_index
            )
        final_state_fwd, paths_fwd = self.take_steps(
            f, N, init_pos, init_pos, delta, init_h, init_grad
        )
        cut_index_fwd = jnp.where(final_state_fwd[3] == 0, N - 1, final_state_fwd[2] - 1)
        paths_fwd = ZeroSolver.trim_paths(jax.tree_util.tree_map(ZeroSolver.swap, paths_fwd), cut_index_fwd)
        end_points = ZeroSolver.vec_slice(paths_fwd['path'], cut_index_fwd, 0).squeeze()
        final_state_rev, paths_rev = self.take_steps(
            f, N, end_points, init_pos, -delta, init_h, init_grad
        )
        cut_index_rev = jnp.where(final_state_rev[3] == 0, N - 1, final_state_rev[2] - 1)
        # If forward pass closed, don't use the reverse pass
        cut_index_rev = jnp.where(final_state_fwd[3] == 2, -1, cut_index_rev)
        paths_rev = ZeroSolver.trim_paths(jax.tree_util.tree_map(ZeroSolver.swap, paths_rev), cut_index_rev)

        paths_combined = ZeroSolver.stack_and_roll(init_pos, init_h, paths_fwd, paths_rev, cut_index_rev)
        stopping_conditions = jnp.stack([final_state_fwd[3], final_state_rev[3]]).T
        paths_combined = self.threshold_cut(paths_combined)
        return paths_combined, stopping_conditions


def split_curves(a, threshold):
    '''Given a set of sorted points, split it into multiple arrays
    if the distance between adjacent points is larger than the given
    threshold.  Used to split an array into unique contours for plotting.

    Parameters
    ----------
    a : jnp.array
        Sorted list of positions (see the sort_by_distance function)
    threshold : float
        If adjacent points are greater than this distance apart, split
        the list at that position.

    Returns
    -------
    list of jnp.arrays
        List of split arrays.  If the first and last points of a sub-array
        are within the threshold of each other the first point is repeated
        at the end of the array (e.g. the contour is closed).
    '''
    # distance to next point
    d = jnp.sum(jnp.diff(a, axis=0)**2, axis=1)
    jump = d > threshold
    cut_points = jump.nonzero()[0] + 1
    cut_points = jnp.concat([jnp.array([0], dtype=int), cut_points, jnp.array([a.shape[0]], dtype=int)])
    output = []
    for idx in range(cut_points.shape[0] - 1):
        cut = jax.lax.dynamic_slice(a, (cut_points[idx], 0), slice_sizes=(cut_points[idx + 1] - cut_points[idx], a.shape[1]))
        if jnp.sum((cut[0] - cut[-1])**2) < threshold:
            cut = jnp.vstack([cut, cut[0]])
        output.append(cut)
    return output
