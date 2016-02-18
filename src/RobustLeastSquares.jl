module RobustLeastSquares

using StatsBase
using IterativeSolvers
using Logging

export reweighted_least_squares, mestimator_mad
export MEstimator, MEstimatorConvex, MEstimatorNonConvex
export L2Estimator, L1Estimator, L1L2Estimator, HuberEstimator, FairEstimator, CauchyEstimator, SubsetEstimator, SubsetEstimatorConvex, DualEstimator, DualEstimatorConvex



using PyCall
@pyimport pyamg
@pyimport scipy.sparse as scipy_sparse

# Helper function for python matrices
function py_csc(A::SparseMatrixCSC)
    # create an empty sparse matrix in Python
    Apy = scipy_sparse.csc_matrix(size(A))
    # write the values
    Apy[:data] = copy(A.nzval)
    # write the indices
    Apy[:indices] = A.rowval - 1
    Apy[:indptr] = A.colptr - 1
    return Apy
end
py_csr(A::SparseMatrixCSC) = py_csc(A)[:tocsr]()


# Weight functions
# ----------------
# A good overview of these can be found at:
# http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html

# Standard L2 estimator
"""
An m-estimator is a cost/loss function used in modified (weighted) least squares
problems of the form:
    min ∑ᵢ ρ(rᵢ)
"""
abstract MEstimator
abstract MEstimatorConvex <: MEstimator
abstract MEstimatorNonConvex <: MEstimator

# Default implementations from definintion of cost-function rho
"The cost (a.k.a. loss) function ρ for the M-estimator"
function m_estimator_rho end
"The derivative of the cost (a.k.a. loss) function ψ for the M-estimator"
function m_estimator_psi end
# TODO m_estimator_psi(r,est::MEstimator) = first derivative of m_estimator_rho(r,est) w.r.t. r
"The weight function, w, for the M-estimator, to be used for modifying least-square problems"
m_estimator_weight(r,est::MEstimator) = m_estimator_psi(r,est) ./ r
"The square root of the weight function, sqrt(w), for the M-estimator, to be used for modifying the normal equations of a least-squares problem"
m_estimator_sqrtweight(r,est::MEstimator) = sqrt(m_estimator_weight(r,est))

"The (convex) L2 M-estimator is that of the standard least squares problem."
immutable L2Estimator <: MEstimatorConvex; end
L2Estimator(width) = L2Estimator()
m_estimator_rho(r,::L2Estimator) = 0.5*r.^2
m_estimator_psi(r,::L2Estimator) = r
m_estimator_weight(r,::L2Estimator) = ones(size(r))
m_estimator_sqrtweight(r,::L2Estimator) = ones(size(r))

# Standard L1 estimator
"The standard L1 M-estimator takes the absolute value of the residual, and is convex but non-smooth."
immutable L1Estimator <: MEstimatorConvex; end
L1Estimator(width) = L1Estimator()
m_estimator_rho(r,::L1Estimator) = abs(r)
m_estimator_psi(r,::L1Estimator) = sign(r)
m_estimator_weight(r,::L1Estimator) = 1.0 ./ abs(r)
m_estimator_sqrtweight(r,::L1Estimator) = abs(r) .^ (-0.5)

# Estimator interpolating between L1 and L2 norms (analytically smooth) TODO Add width function
"The convex L1-L2 estimator interpolates smoothly between L2 behaviour for small residuals and L1 for outliers."
immutable L1L2Estimator <: MEstimatorConvex; width::Float64; end
m_estimator_rho(r,::L1L2Estimator) = 2.0*(sqrt(1.0 + 0.5*r.*r)-1.0)
m_estimator_psi(r,::L1L2Estimator) = r ./ sqrt(1 + 0.5*r.*r)
m_estimator_weight(r,::L1L2Estimator) = 1.0 / sqrt(1+0.5*r.*r)
m_estimator_sqrtweight(r,::L1L2Estimator) = (1+0.5*r.*r) .^ (-1/4)

# Huber Estimator for M-estimation
# C2-continuous, behaving as L1 at long distances and L2 at short
"The convex Huber estimator switches from between quadratic and linear cost/loss function at a certain cutoff."
immutable HuberEstimator <: MEstimatorConvex; width::Float64; end
function m_estimator_rho(r,est::HuberEstimator)
    rho = 0.5*r.^2
    absr = abs(r)
    rho[absr .> est.width] = est.width * (absr[absr .> est.width] - 0.5*est.width)
    return rho
end
function m_estimator_psi(r,est::HuberEstimator)
    psi = r
    absr = abs(r)
    psi[absr .> est.width] = est.width * sign(absr[absr .> est.width])
    return psi
end
function m_estimator_weight(r,est::HuberEstimator)
    w = ones(size(r))
    absr = abs(r)
    w[absr .> est.width] = est.width ./ absr[abs .> est.width]
    return w
end
function m_estimator_sqrtweight(r,est::HuberEstimator)
    w = ones(size(r))
    absr = abs(r)
    w[absr .> est.width] = sqrt(est.width ./ absr[absr .> est.width])
    return w
end

# The "fair" weighting factor for interpolating between L1 and L2 regularization
# (C3 continuous but not analytically smooth)
"""The (convex) "fair" estimator switches from between quadratic and linear cost/loss function at a certain cutoff, and is C3 but non-analytic."""
immutable FairEstimator <: MEstimatorConvex; width::Float64; end
m_estimator_rho(r,est::FairEstimator) = est.width^2 * (abs(r)/est.width - log(1 + abs(r)/est.width))
m_estimator_psi(r,est::FairEstimator) = r ./ (1.0 + abs(r)/est.width)
m_estimator_weight(r,est::FairEstimator) = 1.0 ./ (1.0 + abs(r)/est.width)
m_estimator_sqrtweight(r,est::FairEstimator) = 1.0 ./ sqrt(1.0 + abs(r)/est.width)

# The Cauchy (or Lorentzian) estimator. Non-convex with logarithmically-diverging
# cost function means that wild outliers are mostly ignored, while still having a
# somewhat-robust field-of-influence
"""The non-convex Cauchy estimator switches from between quadratic behaviour to logarithmic tails. This rejects outliers but may result in mutliple minima."""
immutable CauchyEstimator <: MEstimatorNonConvex; width::Float64; end
m_estimator_rho(r,est::CauchyEstimator) = 0.5*est.width^2 * log(1 + r.*r/(est.width*est.width))
m_estimator_psi(r,est::CauchyEstimator) = r ./ (1.0 + r.*r/(est.width*est.width))
m_estimator_weight(r,est::CauchyEstimator) = 1.0 ./ (1.0 + r.*r/(est.width*est.width))
m_estimator_sqrtweight(r,est::CauchyEstimator) = 1.0 ./ sqrt(1.0 + r.*r/(est.width*est.width))


"""
A custom type for applying an estimators to only a subset of the range
(uses L2 for any unspecified range)
"""
immutable SubsetEstimatorConvex{T<:MEstimatorConvex} <: MEstimatorConvex
    rng::UnitRange{Int64}
    est::T
end
immutable SubsetEstimator{T<:MEstimator} <: MEstimatorNonConvex
    rng::UnitRange{Int64}
    est::T
end

function m_estimator_rho{T}(r,est::Union{SubsetEstimatorConvex{T},SubsetEstimator{T}})
    rho = 0.5*r.^2
    rho[est.rng] = m_estimator_rho(r[est.rng],est.est)
    return rho
end
function m_estimator_psi{T}(r,est::Union{SubsetEstimatorConvex{T},SubsetEstimator{T}})
    psi = r
    psi[est.rng] = m_estimator_psi(r[est.rng],est.est)
    return psi
end
function m_estimator_weight{T}(r,est::Union{SubsetEstimatorConvex{T},SubsetEstimator{T}})
    w = ones(size(r))
    w[est.rng] = m_estimator_weight(r[est.rng],est.est)
    return w
end
function m_estimator_sqrtweight{T}(r,est::Union{SubsetEstimatorConvex{T},SubsetEstimator{T}})
    w = ones(size(r))
    w[est.rng] = m_estimator_sqrtweight(r[est.rng],est.est)
    return w
end


"""
A custom type for applying two estimators to different ranges
(uses L2 for any unspecified range)
"""
immutable DualEstimatorConvex{T1<:MEstimatorConvex,T2<:MEstimatorConvex} <: MEstimatorConvex
    rng1::UnitRange{Int64}
    est1::T1
    rng2::UnitRange{Int64}
    est2::T2
end
immutable DualEstimator{T1<:MEstimator,T2<:MEstimator} <: MEstimatorNonConvex
    rng1::UnitRange{Int64}
    est1::T1
    rng2::UnitRange{Int64}
    est2::T2
end
#DualEstimator = union(DualEstimatorConvex,DualEstimatorNonConvex)
function m_estimator_rho{T1,T2}(r,est::Union{DualEstimatorConvex{T1,T2},DualEstimator{T1,T2}})
    rho = 0.5*r.^2
    rho[est.rng1] = m_estimator_rho(r[est.rng1],est.est1)
    rho[est.rng2] = m_estimator_rho(r[est.rng2],est.est2)
    return rho
end
function m_estimator_psi{T1,T2}(r,est::Union{DualEstimatorConvex{T1,T2},DualEstimator{T1,T2}})
    psi = r
    psi[est.rng1] = m_estimator_psi(r[est.rng1],est.est1)
    psi[est.rng2] = m_estimator_psi(r[est.rng2],est.est2)
    return phi
end
function m_estimator_weight{T1,T2}(r,est::Union{DualEstimatorConvex{T1,T2},DualEstimator{T1,T2}})
    w = ones(size(r))
    w[est.rng1] = m_estimator_weight(r[est.rng1],est.est1)
    w[est.rng2] = m_estimator_weight(r[est.rng2],est.est2)
    return w
end
function m_estimator_sqrtweight{T1,T2}(r,est::Union{DualEstimatorConvex{T1,T2},DualEstimator{T1,T2}})
    w = ones(size(r))
    w[est.rng1] = m_estimator_sqrtweight(r[est.rng1],est.est1)
    w[est.rng2] = m_estimator_sqrtweight(r[est.rng2],est.est2)
    return w
end

"Recreate an M-Estimator of the same type using the MAD (median absolute deviation) of the residual"
mestimator_mad{T<:MEstimator}(Est::Type{T}, res, factor=1.0) = Est(factor*1.43*StatsBase.mad(res))
function mestimator_mad{T}(Est::Union{Type{SubsetEstimator{T}},Type{SubsetEstimatorConvex{T}}}, rng, res, factor=1.0)
    return Est(rng, T(factor*1.43*StatsBase.mad(res[rng])))
end
function mestimator_mad{T1,T2}(Est::Union{Type{DualEstimator{T1,T2}},Type{DualEstimatorConvex{T1,T2}}}, rng1, rng2, res, factor=1.0)
    return Est(rng1, T1(factor*1.43*StatsBase.mad(res[rng1])), rng2, T2(factor*1.43*StatsBase.mad(res[rng2])))
end

mestimator_mad(est::MEstimator, res, factor=1.0) = typeof(est)(factor*1.43*StatsBase.mad(res))
function mestimator_mad{T}(est::Union{SubsetEstimator{T},SubsetEstimatorConvex{T}}, res, factor=1.0)
    return typeof(est)(est.rng, T(factor*1.43*StatsBase.mad(res[est.rng])))
end
function mestimator_mad{T1,T2}(est::Union{DualEstimator{T1,T2},DualEstimatorConvex{T1,T2}}, res, factor=1.0)
    return typeof(est)(est.rng1, T1(factor*1.43*StatsBase.mad(res[est.rng1])), est.rng2, T2(factor*1.43*StatsBase.mad(res[est.rng2])))
end





function solve(A,b,weights=ones(length(b)),method=:qr,x0=nothing)
    if method == :qr
        return (spdiagm(weights)*A) \ (spdiagm(weights)*b)
    elseif method == :qrnormal
        return (A'*spdiagm(weights)*A) \ (A'*spdiagm(weights)*b)
    elseif method == :lsqr
        # Use a conjugate gradient method to find the solutoin (less memory)
        (sol,ch) = lsqr!(sol,spdiagm(weights)*A,spdiagm(weights)*b)
        return sol
    elseif method == :amg
        if isa(x0,Void)
            error("Failed to specify starting vector for AMG method")
        end

        # load into Python
        A_py_csr = py_csr(A'*spdiagm(weights)*A)

        #sol = ml[:solve](A'*spdiagm(weights)*b, tol=1e-10)
        sol = pyamg.solve(A_py_csr,A'*spdiagm(weights)*b,tol=1e-10,x0=x0, maxiter=50)
    else
        error("Method :$method should have been one of :qr, :qrnormal :lsqr or :amg")
    end
end

"""
(sol,res) = reweighted_least_squares(sol::AbstractVector, A::AbstractMatrix,b::AbstractVector,estimator::MEstimator = L2Estimator,useqr::Type = Val{true}; n_iter=10)

Solves a reweighted least squares problem: min ∑ᵢ ρ((A*sol - b)ᵢ) using the specified MEstimator for ρ and starting with sol (for initial weights).
"""
function reweighted_least_squares(A::AbstractMatrix,b::AbstractVector,estimator::MEstimator = L2Estimator, x0 = nothing;method::Symbol=:qr, n_iter::Integer=10, reweight_MAD::Bool = false, quiet::Bool = false, kwargs...)
    local sol, res, weights
    #res = Vector{Float64}()
    if isa(x0,Void)
        weights = ones(b)
    else
        res = A*x0 - b
        if reweight_MAD
            weights = m_estimator_sqrtweight(res, mestimator_mad(estimator,res,3.0))
        else
            weights = m_estimator_sqrtweight(res, estimator)
        end
    end

    # Reweighted least squares...

    if issparse(A)
        quiet || info("Solving a $(size(A)) reweighted least-squares problem. $(typeof(A)) matrix has $(nnz(A)) non-zero elements. Using $method method.")
    else
        quiet || info("Solving a $(size(A)) reweighted least-squares problem with a $(typeof(A)). Using $method method.")
    end

    for i=1:n_iter
        # Use the QR decomposition to find the solution
        # TODO: change \ to enforce qrfact! to avoid corner cases (e.g. non-invertable square matrices)
        if i == 1
            sol = solve(A,b,weights,method,x0)
        else
            sol = solve(A,b,weights,method,sol)
        end
        res = A*sol - b

        # reweight tile matches
        if reweight_MAD
            weights = m_estimator_sqrtweight(res, mestimator_mad(estimator,res,3.0))
        else
            weights = m_estimator_sqrtweight(res, estimator)
        end

        quiet || debug("Iteration $i, RMS residual $(sqrt(sum(res.*res)/length(res)))))")
    end

    quiet || info("Root-mean-square weighted residual error = $(sqrt(sum(res.^2)/length(res)))")

    return (sol,res,weights)
end

# package code goes here

end # module
