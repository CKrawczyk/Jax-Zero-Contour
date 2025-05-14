'''Find and follow a zero value contour for any 2D function written in Jax.'''


from .zero_contour_finder import (
    zero_contour_finder,
    split_curves,
    value_and_grad_wrapper,
    stopping_conditions
)

__version__ = '1.0.0'
