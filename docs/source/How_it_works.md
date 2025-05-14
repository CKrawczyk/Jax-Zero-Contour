# How it works

The goal of this function is to take a 2D function and find a set of points where this function is equal to zero.

## Traditional methods

The most traditional way to go about this is to evaluate the function on a grid, find adjacent gird cells with different signs, and use interpolation to find the zero on the edge between the cells.  While this works well for most plotting situations, this has several drawbacks:
1. If the function is discontinuous this can lead to a "false zero" being found as two adjacent cells can have different sign but not have the function pass through zero between them
2. A fixed grid might not be able to capture the shape of the contour in areas where it is changing quickly
3. If the contour covers a large extent in either dimension a large grid is needed leading to a large number of function evaluations
4. The zeros are only as accurate as the interpolation function being used (and this is typically quite low)

## This method

To try and overcome these limitations, this code instead makes use of Jax's auto-differentiation framework to apply gradient based methods for finding the zeros of the function.  This is done in a two step process:
1. Zero finding
2. Zero following

### Zero finding

For zero finding we using Newton's method.  This is an iterative method that takes in a starting point and each step moves it closer to a zero of the function by following it either up/down hill with a step size proportional to the gradient of the function.

In 1D function {math}`f(x)` Newton's method is:

```{math}
x_{i+1} = x_i - \frac{f(x_i)}{f'(x_i)}
```

For a ND function {math}`F(\vec{x})`, this needs to be adjusted a bit:

```{math}
\vec{x}_{i+1} = \vec{x}_i - J^{+}_{F}(\vec{x}_i) F(\vec{x}_i)
```

Where {math}`J^{+}_{F}(\vec{x}_i)` is the pseudoinverse of the jacobian of {math}`F` given by {math}`J^{+} = (J^T J)^{-1}J^T`.

We have a 2D function, so the jacobian {math}`\vec{J}` is a 2 element vector, and the pseudoinverse is {math}`J^{+} = \frac{\vec{J}(\vec{x}_i)}{|\vec{J}(\vec{x}_i)|^2}`.

So our final Newton's setp is:
```{math}
\vec{x}_{i+1} = \vec{x}_i - \frac{\vec{J}(\vec{x}_i) F(\vec{x}_i)}{|\vec{J}(\vec{x}_i)|^2} 
```

### Zero following

Once a zero has been found the contour can be followed by setting up a dynamical system with the following conditions:
- Let {math}`t` = "length along the contour"
- Let {math}`F(x, y)` be the function we want the contours of
- Let {math}`F(x_0, y_0) = 0` be an initial point on the contour

and we want to find a set of values at time {math}`i` such that {math}`F(x_i, y_i) = 0` for all {math}`i`.  This is the same as saying:
```{math}
\frac{d}{dt} F(x, y) = 0
```

From the law of partial derivatives this implies:

```{math}
\frac{d}{dt} F(x, y) = \frac{\partial F}{\partial x}\frac{dx}{dt} + \frac{\partial F}{\partial y}\frac{dy}{dt} + \frac{\partial F}{\partial t} = 0
```

As {math}`F` is time independent the final {math}`\frac{\partial f}{\partial t}` term is zero.  This expression can be rearranged as:

```{math}
\frac{\partial F}{\partial x}\frac{dx}{dt} = -\frac{\partial F}{\partial y}\frac{dy}{dt}
```

leading to:

```{math}
\frac{\frac{dx}{dt}}{\frac{\partial F}{\partial y}} = -\frac{\frac{dy}{dt}}{\frac{\partial F}{\partial x}} = c
```

for any constant {math}`c`.  Setting {math}`c=1` we can see this reduces to:

```{math}
\frac{dx}{dt} &=& \frac{\partial F}{\partial y} \\
\frac{dy}{dt} &=& -\frac{\partial F}{\partial x}
```

At this point we can see that this is just a relabeling of Hamilton's equations from physics where the {math}`q \rightarrow x`, {math}`p \rightarrow y`, and {math}`H \rightarrow F`.  

```{note}
Fundamentally this connection with Hamilton's equation is not to surprising.  The physical system solves for all value in the position-momentum phase space where the energy is constant.  If position and momentum are both 1D values and we pick an initial state where the energy is zero, we will exactly set up a system that would solve for the zero contour of the energy function.

Swapping position for {math}`x`, momentum for {math}`y`, and the energy function for {math}`F(x, y)` we have exactly recovered the problem we are trying to solve.
```

Now we can construct a PDE solver for our initial value problem.  If we use a simple Euler step the path will quickly diverge from the true contour unless the step size is very small.  To avoid these errors from getting too large we will do a two step process for solving the path.

1. Take an Euler step normalized by the magnitude of the gradient
2. Use Newton's method to bring the proposed point back onto the zero contour (to some desired tolerance)

The update for the first step looks like:
```{math}
\vec{x}_{i+1/2} = \vec{x}_i + \delta \frac{R_{90}\vec{J}(\vec{x}_i)}{|\vec{J}(\vec{x}_i)|}
```
Where {math}`R_{90}` is a 90 degree rotation matrix.

The update for the second step is the same as above (multiple of these steps might be needed):
```{math}
\vec{x}_{i+1} = \vec{x}_{i+1/2} - \frac{\vec{J}(\vec{x}_{i+1/2}) F(\vec{x}_{i+1/2})}{|\vec{J}(\vec{x}_{i+1/2})|^2}
```

When written in this way we can see that this algorithm can be rephrased as "walk perpendicular to the gradient by a fixed distance" followed by "walk along the gradient until you are at the value zero."
