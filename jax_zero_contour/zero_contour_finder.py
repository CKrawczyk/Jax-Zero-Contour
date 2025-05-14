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


def step_one_tp(delta, value_and_grad_function, tol, max_newton, carry, index):
    # Take one setp perpendicular followed by Newton's method to
    # bring it back onto the zero contour
    pos_in, pos_start, cut, stop_condition, h, grad = carry
    pos = step_tangent(pos_in, grad, delta)
    _, pos, h, grad = step_parallel_tol(pos, value_and_grad_function, tol, max_newton)

    # how far did this step travel
    delta_travel = jnp.linalg.norm(pos_in - pos)
    # how close is it to closing
    delta_start = jnp.linalg.norm(pos_start - pos)
    # cut is the first index at which a stopping condition has been met
    cond1 = cut == 0
    # Newton's method moved very far from the initial point
    # this is an indication the function has a discontinuity and an
    # end point has been reached
    cond2 = delta_travel > 2 * jnp.abs(delta)
    stop_condition = jax.lax.select(
        (stop_condition == 0) & cond2,
        1,
        stop_condition
    )
    # check if the contour has closed
    cond3 = (
        delta_start < 1.1 * jnp.abs(delta)
    ) & (
        jnp.all((pos_in != pos_start))
    )
    stop_condition = jax.lax.select(
        (stop_condition == 0) & cond3,
        2,
        stop_condition
    )
    # set the cut value if either of the above are met for the first time
    cut = jax.lax.select(
        (cond1) & (cond2 | cond3),
        index,
        cut
    )
    return (pos, pos_start, cut, stop_condition, h, grad), jnp.hstack([pos, h])


@Partial(jax.jit, static_argnames=('N',))
def step_H_tp(value_and_grad_function, N, tol, max_newton, pos_start, pos, delta, h, grad):
    # use `scan` to run a single "batch" of N steps
    step_one_part = Partial(step_one_tp, delta, value_and_grad_function, tol, max_newton)
    return jax.lax.scan(
        step_one_part,
        (pos, pos_start, 0, 0, h, grad),
        xs=jnp.arange(N)
    )


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


def zero_contour_finder(
    value_and_grad_function,
    init_guess,
    delta=0.1,
    N=100,
    max_iter=10,
    tol=1e-6,
    max_newton=5
):
    '''Find the zero contour of a 2D function.

    Parameters
    ----------
    value_and_grad_function : function
        A function of x and y that that returns the target function and its Jacobian,
        it is recommended that this function be jited.  This function must be wrapped
        in jax.tree_util.Partial.
    init_guess : jax.numpy.array
        Initial guess for a point near the zero contour.
    delta : float, optional
        The step size to take along the contour when searching for a new point,
        by default 0.1.
    N : int, optional
        Batch size for calculating new points along the contour, by default 100.
        After each batch the stopping conditions are checked: does the contour
        hit an end point or does the contour close.
    max_iter : int, optional
        The maximum number of batches to run in each direction from the initial
        point, by default 10.
    tol : float, optional
        Newton's steps are used to bring each proposed point on the contour to
        be within this tolerance of zero, by default 1e-6.
    max_newton : int, optional
        The maximum number of Newton's steps to run, by default 5.


    Returns
    -------
    Path : jax.numpy.array
        The ordered points along the zero contour
    E : jax.numpy.array
        The value of the function evaluated on the contour
    stop_output : list
        List containing the stopping conditions for the forward and backward batches


    Because the number of points in the contour is not know beforehand, this function can
    not be jited.
    '''
    # Use the initial guess to find a point on the contour
    _, init_pos, h, grad = step_parallel_tol(init_guess, value_and_grad_function, tol, 5 * max_newton)
    init_output = jnp.hstack([init_pos, h])
    if ~(jnp.isfinite(init_pos).all()) or (jnp.abs(h) > tol):
        # If this fails return
        logger.warning(f'No zero contour found after 5*max_newton ({5 * max_newton}) iterations')
        return None, None, None
    step_part = Partial(
        step_H_tp,
        value_and_grad_function,
        N,
        tol,
        max_newton
    )

    # From the initial position on the contour, step forward (clockwise) by `N` steps
    i = 1
    (_, _, cut_fwd, stop_fwd, h, grad), path_fwd = step_part(init_pos, init_pos, delta, h, grad)
    # Trim the results if a stopping point is found (either an endpoint or a closed contour)
    cut_point_fwd = jax.lax.select(cut_fwd == 0, N, cut_fwd)
    path_trim_fwd = jax.lax.dynamic_slice(path_fwd, (0, 0), (cut_point_fwd, 3))
    # If no stopping point is found go another `N` steps for at most `max_iter` rounds
    while (stop_fwd == 0) and (i <= max_iter):
        (_, _, cut_fwd, stop_fwd, h, grad), path_fwd_cont = step_part(init_pos, path_trim_fwd[-1, :2], delta, h, grad)
        cut_point_fwd = jax.lax.select(cut_fwd == 0, N, cut_fwd)
        path_trim_fwd_cont = jax.lax.dynamic_slice(path_fwd_cont, (0, 0), (cut_point_fwd, 3))
        # Append new path to the existing path
        path_trim_fwd = jnp.vstack([
            path_trim_fwd,
            path_trim_fwd_cont
        ])
        i += 1
    # record what the stopping condition was
    stop_output = jnp.array([stop_fwd])

    # If not a closed loop, step backwards (counter-clockwise) by `N` steps
    if stop_fwd != 2:
        j = 1
        # Note that the last point from above is entered as the new point to check a closed loop
        # against
        (_, _, cut_bak, stop_bak, h, grad), path_bak = step_part(path_trim_fwd[-1, :2], init_pos, -delta, h, grad)
        # Trim the results if a stopping point is found (either an endpoint or a closed contour)
        cut_point_bak = jax.lax.select(cut_bak == 0, N, cut_bak)
        path_trim_bak = jax.lax.dynamic_slice(path_bak, (0, 0), (cut_point_bak, 3))
        # If no stopping point is found go another `N` steps for at most `max_iter` rounds
        while (stop_bak == 0) and (j <= max_iter):
            (_, _, cut_bak, stop_bak, h, grad), path_bak_cont = step_part(path_trim_fwd[-1, :2], path_trim_bak[-1, :2], -delta, h, grad)
            cut_point_bak = jax.lax.select(cut_bak == 0, N, cut_bak)
            path_trim_bak_cont = jax.lax.dynamic_slice(path_bak_cont, (0, 0), (cut_point_bak, 3))
            # Append new path to the existing path
            path_trim_bak = jnp.vstack([
                path_trim_bak,
                path_trim_bak_cont
            ])
            j += 1

        # Reverse the backwards path and append it to the start point and forward path
        path_trim = jnp.vstack([
            path_trim_bak[::-1],
            init_output,
            path_trim_fwd
        ])
        # record what the stopping conditions were
        stop_output = jnp.array([stop_fwd, stop_bak])
    else:
        # Append the start point at the start of the path
        path_trim = jnp.vstack([
            init_output,
            path_trim_fwd
        ])
    return path_trim[:, :2], path_trim[:, 2], stop_output


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
