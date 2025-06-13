'''Find and follow a zero value contour for any 2D function written in Jax.'''

import logging
import jax
import jax.numpy as jnp
from jax.tree_util import Partial

logger = logging.getLogger(__name__)


def step_tangent(pos, grad, delta):
    # take a step perpendicular to the gradient (e.g. Euler-Lagrange)
    # jax.debug.print("{pos}", pos=pos)
    (q, p) = pos
    (dq, dp) = grad
    alpha = jnp.sqrt(dq**2 + dp**2)
    return jnp.array([
        q + delta * dp / alpha,
        p - delta * dq / alpha
    ])


vec_step_tangent = jax.vmap(step_tangent, in_axes=(0, 0, None))


def step_parallel(state, value_and_grad_function):
    # take a step along the gradient (e.g. Newton's method)
    count, pos, h, grad = state
    (q, p) = pos
    dq, dp = grad
    alpha_2 = dq**2 + dp**2
    new_pos = jnp.array([
        q - h * dq / alpha_2,
        p - h * dp / alpha_2
    ])
    h, grad = value_and_grad_function(new_pos[0], new_pos[1])
    return count + 1, new_pos, h, grad


def parallel_break(state, tol, max_newton):
    # Stop Newton's method if the function is within
    # `tol` of zero or the max number of steps is reached
    count, _, h, _ = state
    return (jnp.abs(h) > tol) & (count <= max_newton)


@jax.jit
def step_parallel_tol(init_pos, value_and_grad_function, tol, max_newton):
    # use while loop to run Newton's method
    partial_break = Partial(parallel_break, tol=tol, max_newton=max_newton)
    partial_step = Partial(step_parallel, value_and_grad_function=value_and_grad_function)
    h, grad = value_and_grad_function(init_pos[0], init_pos[1])
    return jax.lax.while_loop(
        partial_break,
        partial_step,
        (1, init_pos, h, grad)
    )


vec_step_parallel_tol = jax.vmap(step_parallel_tol, in_axes=(0, None, None, None))


def take_step(carry, delta, value_and_grad_function, tol, max_newton):
    pos_in, pos_start, _, _, h, grad = carry
    pos = vec_step_tangent(pos_in, grad, delta)
    _, pos, h, grad = vec_step_parallel_tol(pos, value_and_grad_function, tol, max_newton)
    delta_travel = jnp.linalg.norm(pos_in - pos, axis=1)
    delta_start = jnp.linalg.norm(pos_start - pos, axis=1)
    return pos, h, grad, delta_travel, delta_start


def null_step(carry, *_):
    pos_in, _, _, _, h, grad = carry
    pos_like = jnp.zeros_like(pos_in)
    h_like = jnp.zeros_like(h)
    return pos_like, h_like, grad, h_like, h_like


def vec_step_one_tp(delta, value_and_grad_function, tol, max_newton, carry, index):
    pos_in, pos_start, cut, stop_condition, _, _ = carry
    cond1 = cut == 0
    pos, h, grad, delta_travel, delta_start = jax.lax.cond(
        jnp.any(cond1),
        take_step,
        null_step,
        carry,
        delta,
        value_and_grad_function,
        tol,
        max_newton
    )
    cond2 = delta_travel > 2 * jnp.abs(delta)
    stop_condition = jnp.where((stop_condition == 0) & cond2, 1, stop_condition)
    cond3 = (delta_start < 1.1 * jnp.abs(delta)) & jnp.all(pos_in != pos_start, axis=1)
    stop_condition = jnp.where((stop_condition == 0) & cond3, 2, stop_condition)
    cut = jnp.where((cond1) & (cond2 | cond3), index, cut)
    return (pos, pos_start, cut, stop_condition, h, grad), {'path': pos, 'value': h}


def trim_paths(paths, cut_index):
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


def threshold_cut(paths, tol):
    mask = paths['value'] > 20 * tol
    stack_mask = jnp.stack([mask, mask], axis=2)
    return {
        'path': jnp.where(stack_mask, jnp.nan, paths['path']),
        'value': jnp.where(mask, jnp.nan, paths['value'])
    }


def swap(x):
    return jnp.swapaxes(x, 0, 1)


vec_slice = jax.vmap(jax.lax.dynamic_index_in_dim, in_axes=(0, 0, None))


def vec_step(value_and_grad_function, N, tol, max_newton, pos_start, pos, delta, h, grad):
    part_step = jax.tree_util.Partial(vec_step_one_tp, delta, value_and_grad_function, tol, max_newton)
    carry = (pos, pos_start, jnp.zeros_like(h, dtype=int), jnp.zeros_like(h, dtype=int), h, grad)
    final_state, paths = jax.lax.scan(part_step, carry, xs=jnp.arange(N))
    return final_state, paths


vec_roll = jax.vmap(jnp.roll, in_axes=(0, 0, None))


@jax.vmap
def stack(*args):
    return jnp.vstack([*args])


def stack_and_roll(init_pos, init_h, paths, paths_rev, roll_index):
    n_points = init_h.shape[0]
    N = paths['path'].shape[1]
    roll_amount = roll_index - N + 1
    return {
        'path': vec_roll(
            stack(paths_rev['path'][:, ::-1, :], init_pos.reshape(n_points, 1, 2), paths['path']),
            roll_amount,
            0
        ),
        'value': vec_roll(
            jnp.hstack([paths_rev['value'][:, ::-1], init_h.reshape(n_points, 1), paths['value']]),
            roll_amount,
            0
        )
    }


@Partial(jax.jit, static_argnames=('N', 'silent_fail'))
def zero_contour_finder(
    value_and_grad_function,
    init_guess,
    delta=0.1,
    N=1000,
    tol=1e-6,
    max_newton=5,
    silent_fail=True
):
    '''Find the zero contour of a 2D function.

    Parameters
    ----------
    value_and_grad_function : function
        A function of x and y that that returns the target function and its Jacobian,
        it is recommended that this function be jited.  This function must be wrapped
        in jax.tree_util.Partial.
    init_guess : jax.numpy.array
        Initial guesses for points near the zero contour, one guess per row.
    delta : float, optional
        The step size to take along the contour when searching for a new point,
        by default 0.1.
    N : int, optional
        The total number of steps to take in *each* direction from the starting point(s).
        The final path will be 2N+1 in size (N points in the forward direction, N points
        in the reverse direction, with the initial point in the middle).
    tol : float, optional
        Newton's steps are used to bring each proposed point on the contour to
        be within this tolerance of zero, by default 1e-6.
    max_newton : int, optional
        The maximum number of Newton's steps to run, by default 5.


    Returns
    -------
    paths : dict
        The return dictionary will have two keys
        - "path": jax.numpy.array with shape (number of guesses, 2N+1, 2) with the contours
            paths for each guess.
        - "value": jax.numpy.array with shape (number of guesses, 2N+1) with the function value at
            each point on the path
    stop_output : jax.numpy.array
        List containing the stopping conditions for each guess

    Note: after a path hits an endpoint or closes any further points on the contour are written
    to jax.numpy.nan.  The final output will be shifted so that the finite parts of the contour are
    brought to the front of the array.  The points in the resulting paths are ordered.

    Parts of this code use jax.lax.cond to stop taking new steps after all contours have reached
    an endpoint or closed.  It should not be combined with jax.vmap as a result.
    '''
    init_guess = jnp.atleast_2d(init_guess)
    _, init_pos, init_h, init_grad = vec_step_parallel_tol(
        init_guess,
        value_and_grad_function,
        tol,
        5 * max_newton
    )
    if not silent_fail:
        def excepting_message(failed_init_index):
            jax.debug.print('Index of failed input(s): {i}', i=jnp.nonzero(failed_init_index)[0])
            raise ValueError(f'No zero contour found after 5*max_newton ({5 * max_newton}) iterations')
        failed_init_index = ~jnp.isfinite(init_pos).all(axis=1) | (jnp.abs(init_h) > 1e-6)
        jax.lax.cond(
            failed_init_index.any(),
            lambda idx: jax.debug.callback(excepting_message, idx),
            lambda _: None,
            failed_init_index
        )
    final_state_fwd, paths_fwd = vec_step(
        value_and_grad_function,
        N,
        tol,
        max_newton,
        init_pos,
        init_pos,
        delta,
        init_h,
        init_grad
    )
    # If not closed set to end of list
    cut_index_fwd = jnp.where(final_state_fwd[3] == 0, N - 1, final_state_fwd[2] - 1)
    paths_fwd = trim_paths(jax.tree_util.tree_map(swap, paths_fwd), cut_index_fwd)
    end_points = vec_slice(paths_fwd['path'], cut_index_fwd, 0).squeeze()
    final_state_rev, paths_rev = vec_step(
        value_and_grad_function,
        N,
        tol,
        max_newton,
        end_points,
        init_pos,
        -delta,
        init_h,
        init_grad
    )
    # If not closed set to end of list
    cut_index_rev = jnp.where(final_state_rev[3] == 0, N - 1, final_state_rev[2] - 1)
    # If forward pass closed, don't use the reverse pass
    cut_index_rev = jnp.where(final_state_fwd[3] == 2, -1, cut_index_rev)
    paths_rev = trim_paths(jax.tree_util.tree_map(swap, paths_rev), cut_index_rev)

    paths_combined = stack_and_roll(init_pos, init_h, paths_fwd, paths_rev, cut_index_rev)
    stopping_conditions = jnp.stack([final_state_fwd[3], final_state_rev[3]]).T
    paths_combined = threshold_cut(paths_combined, tol)
    return paths_combined, stopping_conditions


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


def value_and_grad_wrapper(f, forward_mode_differentiation=False):
    '''Helper wrapper that uses either forward or reverse mode autodiff
    to find an inputs function value and gradient

    Parameters
    ----------
    f : function
        The function you want to evaluate and take the gradient of
    forward_mode_differentiation : bool, optional
        If True use forward mode auto-differentiation, otherwise use reverse mode,
        by default False

    Returns
    -------
    function
        A jited function that takes in x and y and returns the inputs
        functions value and gradient at that position
    '''
    # inspired by numpyro https://github.com/pyro-ppl/numpyro/blob/b49b8f8d389d6357ab04003a003ef9fa16ee2e43/numpyro/infer/hmc_util.py#L242C1-L252C51
    if forward_mode_differentiation:
        def value_and_fwd_grad(x, y):
            def _wrapper(x, y):
                out = f(x, y)
                return out, out

            grads, out = jax.jacfwd(_wrapper, argnums=(0, 1), has_aux=True)(x, y)
            return out, grads
        return Partial(jax.jit(value_and_fwd_grad))
    else:
        return Partial(jax.jit(jax.value_and_grad(f, argnums=(0, 1), has_aux=False)))


stopping_conditions = {
    0: 'none',
    1: 'end_point',
    2: 'closed_loop'
}


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
