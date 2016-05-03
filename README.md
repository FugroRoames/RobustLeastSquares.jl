# RobustLeastSquares

[![Build Status](https://travis-ci.org/FugroRoames/RobustLeastSquares.jl.svg?branch=master)](https://travis-ci.org/FugroRoames/RobustLeastSquares.jl)

**RobustLeastSquares** attempts to make robust estimation using least-squares
problems painless and simple. It provide these useful features:

1. A set of linear equation solvers, to perform linear least-squares optimization steps.
2. A set of *M*-estimators for robust estimation. An estimator is used to remove
   outliers by reducing the cost (loss) function for abnormally large residuals, and
   gets its name from the statistical method of estimating parameters from
   certain statistical distributions (e.g. Gaussian statistics correspond to the
   standard *L2*-estimator, etc).
3. A high-level optimization routine that performs iteratively reweighted least-squares,
   solving the non-linear optimization of the cost/loss function by
   a series of quadratic minimization steps.


### Quick-start

Suppose we want to find the "best estimate" of `x` to the equations `Ax = b`,
where we know our data (`A` and `b`) suffer from incomplete information,
statistical errors and wild outliers. *RobustLeastSquares* lets us solve this
robustly and easily:

```julia
x_estimated = reweighted_lsqr(A, b, HuberEstimator(1.0); refit = true, n_iter = 20)
```

### Solvers

The package has three in-built solvers - *QR* factorization of the ordinary
least-squares problem, the normal-form of the least-squares problem, and conjugate
gradient on the former.

We are trying to find *x* such that *||r||* is minimized where the residual *r = Ax - b* (i.e. find the *x* so
that *Ax* is as close as possible to *b* - if *A* is invertible, the solution is
*inv(A).b* and *r = 0*).

The solution in the general case may be found directly
using the "normal equations" *A'A x = A'b* where *A'A* is now a square,
Hermitian matrix, and even if *A'A* is singular, there always exists an exact
solution to this equation (since *A'b* is guaranteed to have zero component in
the kernel of *A'A*). This is more poorly numerically conditioned than
minimizing *r*, but can be faster depending on the sparsity and shape of *A*.

A *weighted* least-squares problem introduces a weight vector *w* which is
multiplies *r* element-wise to scale some residual elements to be larger or smaller,
increasing or decreasing their importance in the optimization procedure (which
is what lets us reduce the impact of outliers on the solution). Writing the
diagonal matrix *W = diag(w)*, we are trying to minimize *||Wr|| = ||W(Ax-b)||*, or in the
normal form, *A'WA = A'Wb*.

The solver is called using the `solve()` function

```julia
x = solve(A, b, weights, method, x0)
#    A: M x N matrix (dense, sparse, etc)
#    b: M-vector
#    weights: scaling factors to be applied to each element of *r*
#    method: a symbol, one of :qr, :normal or :cg
#    x0: initial starting vector, for use with :cg method.
```

### M-Estimators

But how does one obtain the optimal weight vector for reducing the impact of
outliers? An M-estimator (typically) dampens the impact of the largest elements
of *r*, making them impact the solution less than in the typical *L2*-estimator
where the cost function is `sum(r.^2)`. See [this](http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html) excellent resource for an
overview.

Included *M*-estimators are represented by Julia instances of subtypes of `MEstimator`

1. `L2Estimator()`, *cost* = `sum(r.^2)`
2. `L1Estimator()`, *cost* = `sum(abs(r))`
3. `L1L2Estimator()`, behaves *L2* for *r << 1*, and *L1* for *r >> 1*
4. `FairEstimator(c)`, behaves *L2* for *r << c*, and *L1* for *r >> c*
5. `HuberEstimator(c)`, behaves *L2* for *r << c*, and *L1* for *r >> c*
6. `CauchyEstimator(c)`, behaves *L2* for *r << c*, and logarithmic for *r >> c* (non-convex)
7. `GemanEstimator()`,  behaves *L2* for *r << 1*, and Lorentzian for *r >> 1* (non-convex)
8. `WelschEstimator(c)`, behaves *L2* for *r << c*, and Gaussian for *r >> c* (non-convex)
9. `TukeyEstimator(c)`, behaves *L2* for *r < c*, and complete cutoff for *r > c* (non-convex)

As you descend the list, the outliers are more strongly cut-out of the optimization.
The non-convex estimators may not have a unique minimum and care must be
taken to reach a good-quality solution.

An *M*-estimator `est::MEstimator` provides an interface for obtaining the cost function and weight
vectors through the methods `estimator_rho(r,est)`, `estimator_psi(r,est)`, `estimator_weight(r,est)` and
`estimator_sqrtweight(r,est)`. These are called automatically from the optimizer
below or may be integrated into your own code.

There is one more special estimator called a `MultiEstimator` which allows you
to apply different estimators to different parts of your residual vector. This
may be useful to, e.g., apply the Huber-estimator to residuals based on your
measurements and the *L2*-estimator to the residuals corresponding to regularization.

### Reweighted least-squares optimizer

Finally, one can optimize an entire problem with the `reweighted_lsqr()` function.
Briefly, this takes the following syntax:

```julia
(sol, res) = reweighted_lsqr(A::AbstractMatrix, b::AbstractVector, estimator::MEstimator = L2Estimator(), x0 = nothing; method::Symbol=:qr, n_iter::Integer=10, refit::Bool = false, quiet::Bool = false, kwargs...)
#   A: M x N matrix
#   b: M-Vector
#   estimator: An instance of an MEstimator
#   x0: initial guess at solution
#   method: symbols :qr, :normal or :cg
#   n_iter: number of times to repeat the reweighting
#   refit: should the constant c for the MEstimator be refitted using the median-absolute-deviation method?
#
#   sol: the final solution, a N-Vector
#   res: the final residuals, an M-Vector
```
