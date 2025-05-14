# Jax-Zero-Contour

Find and follow a zero value contour for any 2D function written in Jax.  This package provides a function that takes in any 2D Jax function, its gradient, and an initial guess, finds the closest zero of the function, and follows that zero contour with a fixed step size until it either:

1. closes back on itself
2. hits an end point (e.g. a discontinuity in the function)
3. reaches some maximum number of steps

See the [documentation](https://ckrawczyk.github.io/Jax-Zero-Contour/) for more details.

## Installation

```bash
pip install jax_zero_contour
```

