module RobustLeastSquares

using StatsBase
using IterativeSolvers
using Logging

export reweighted_lsqr, refit_estimator

export MEstimator, L2Estimator, L1Estimator, L1L2Estimator, HuberEstimator, FairEstimator, CauchyEstimator, GemanEstimator, WelschEstimator,  TukeyEstimator, MultiEstimator
export estimator_rho, estimator_psi, estimator_weight, estimator_sqrtweight

include("MEstimators.jl")

function solve(A,b,weights=ones(length(b)),method=:qr,x0=nothing)
    if method == :qr
        return (scale(weights,A)) \ (weights.*b)
        # this works for sparse matrices:
        #return Base.LinAlg.SparseMatrix.SPQR.solve(0,qrfact(sparse(spdiagm(weights)*A)),Base.LinAlg.SparseMatrix.CHOLMOD.Dense(spdiagm(weights)*b))
    elseif method == :normal
        return (A' * (scale(weights.^2,A))) \ (A' * (weights.^2.*b))
    elseif method == :cg
        # Use a conjugate gradient method to find the solution (less memory)
        if x0 == nothing
            x0 = zeros(size(A,2))
        end
        (sol,ch) = lsqr!(x0,scale(weights,A),weights.*b)
        return sol
    else
        error("Method :$method should have been one of :qr, :normal or :cg")
    end
end

"""
(sol,res) = reweighted_lsqr(sol::AbstractVector, A::AbstractMatrix,b::AbstractVector,estimator::MEstimator = L2Estimator,useqr::Type = Val{true}; n_iter=10)

Solves a reweighted least squares problem: min ∑ᵢ ρ((A*sol - b)ᵢ) using the specified MEstimator for ρ and starting with sol (for initial weights).
"""
function reweighted_lsqr(A::AbstractMatrix,b::AbstractVector,estimator::MEstimator = L2Estimator(), x0 = nothing;method::Symbol=:qr, n_iter::Integer=10, refit::Bool = false, quiet::Bool = false, kwargs...)
    local sol, res, weights

    s1,s2 = size(A)
    if s1 == s2
        warn("Encountered square matrix of size $s1. Julia will revert to linear solvers instead of least-square solvers, and throw an error if the matrix is singular.")
    end

    # Set the initial weights
    if x0 === nothing
        weights = ones(b)
    else
        res = A*x0 - b
        if refit
            weights = estimator_sqrtweight(res, refit_estimator(estimator, res, 3.0))
        else
            weights = estimator_sqrtweight(res, estimator)
        end
    end

    # Perform the reweighted least squares
    if issparse(A)
        quiet || info("Solving a $(size(A)) reweighted least-squares problem. $(typeof(A)) matrix has $(nnz(A)) non-zero elements. Using $method method.")
    else
        quiet || info("Solving a $(size(A)) reweighted least-squares problem with a $(typeof(A)). Using $method method.")
    end

    for i=1:n_iter
        if i == 1
            sol = solve(A,b,weights,method,x0)
        else
            sol = solve(A,b,weights,method,sol)
        end
        res = A*sol - b

        if refit
            weights = estimator_sqrtweight(res, refit_estimator(estimator,res,3.0))
        else
            weights = estimator_sqrtweight(res, estimator)
        end

        quiet || debug("Iteration $i, RMS residual $(sqrt(sum(res.*res)/length(res)))))")
    end

    quiet || info("Root-mean-square weighted residual error = $(sqrt(sum(res.^2)/length(res)))")

    return (sol,res,weights)
end

end # module
