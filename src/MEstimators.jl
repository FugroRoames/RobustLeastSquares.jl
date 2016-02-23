# Weight functions
# ----------------
# A good overview of these can be found at:
# http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html

"""
An m-estimator is a cost/loss function used in modified (weighted) least squares
problems of the form:
    min ∑ᵢ ρ(rᵢ)
"""
abstract MEstimator

# Default implementations from definintion of cost-function rho
"The cost (a.k.a. loss) function ρ for the M-estimator"
function m_estimator_rho end
"The derivative of the cost (a.k.a. loss) function ψ for the M-estimator"
function m_estimator_psi end
# TODO m_estimator_psi(r,est::MEstimator) = first derivative of m_estimator_rho(r,est) w.r.t. r
"""
The weight function, w, for the M-estimator, to be used for modifying the normal
equations of a least-square problem
"""
m_estimator_weight(r,est::MEstimator) = m_estimator_psi(r,est) ./ r
"""
The square root of the weight function, sqrt(w), for the M-estimator, to be used
for modifying a least-squares problem
"""
m_estimator_sqrtweight(r,est::MEstimator) = sqrt(m_estimator_weight(r,est))
isconvex(::MEstimator) = false

"The (convex) L2 M-estimator is that of the standard least squares problem."
immutable L2Estimator <: MEstimator; end
L2Estimator(width) = L2Estimator()
estimator_rho(r,::L2Estimator) = 0.5*r.^2
estimator_psi(r,::L2Estimator) = r
estimator_weight(r,::L2Estimator) = ones(size(r))
estimator_sqrtweight(r,::L2Estimator) = ones(size(r))
isconvex(::L2Estimator) = true

"""
The standard L1 M-estimator takes the absolute value of the residual, and is
convex but non-smooth.
"""
immutable L1Estimator <: MEstimator; end
L1Estimator(width) = L1Estimator()
estimator_rho(r,::L1Estimator) = abs(r)
estimator_psi(r,::L1Estimator) = sign(r)
estimator_weight(r,::L1Estimator) = 1.0 ./ abs(r)
estimator_sqrtweight(r,::L1Estimator) = abs(r) .^ (-0.5)
isconvex(::L1Estimator) = true

"""
The convex L1-L2 estimator interpolates smoothly between L2 behaviour for small
residuals and L1 for outliers.
"""
immutable L1L2Estimator <: MEstimator; width::Float64; end
estimator_rho(r,::L1L2Estimator) = 2.0*(sqrt(1.0 + 0.5*r.*r)-1.0)
estimator_psi(r,::L1L2Estimator) = r ./ sqrt(1 + 0.5*r.*r)
estimator_weight(r,::L1L2Estimator) = 1.0 / sqrt(1+0.5*r.*r)
estimator_sqrtweight(r,::L1L2Estimator) = (1+0.5*r.*r) .^ (-1/4)
isconvex(::L1L2Estimator) = true

"""
The convex Huber estimator switches from between quadratic and linear cost/loss
function at a certain cutoff.
"""
immutable HuberEstimator <: MEstimator; width::Float64; end
function estimator_rho(r,est::HuberEstimator)
    rho = 0.5*r.^2
    absr = abs(r)
    rho[absr .> est.width] = est.width * (absr[absr .> est.width] - 0.5*est.width)
    return rho
end
function estimator_psi(r,est::HuberEstimator)
    psi = r
    absr = abs(r)
    psi[absr .> est.width] = est.width * sign(absr[absr .> est.width])
    return psi
end
function estimator_weight(r,est::HuberEstimator)
    w = ones(size(r))
    absr = abs(r)
    w[absr .> est.width] = est.width ./ absr[abs .> est.width]
    return w
end
function estimator_sqrtweight(r,est::HuberEstimator)
    w = ones(size(r))
    absr = abs(r)
    w[absr .> est.width] = sqrt(est.width ./ absr[absr .> est.width])
    return w
end
isconvex(::HuberEstimator) = true

"""
The (convex) "fair" estimator switches from between quadratic and linear
cost/loss function at a certain cutoff, and is C3 but non-analytic.
"""
immutable FairEstimator <: MEstimator; width::Float64; end
estimator_rho(r,est::FairEstimator) = est.width^2 * (abs(r)/est.width - log(1 + abs(r)/est.width))
estimator_psi(r,est::FairEstimator) = r ./ (1.0 + abs(r)/est.width)
estimator_weight(r,est::FairEstimator) = 1.0 ./ (1.0 + abs(r)/est.width)
estimator_sqrtweight(r,est::FairEstimator) = 1.0 ./ sqrt(1.0 + abs(r)/est.width)
isconvex(::FairEstimator) = true

"""
The non-convex Cauchy estimator switches from between quadratic behaviour to
logarithmic tails. This rejects outliers but may result in mutliple minima.
"""
immutable CauchyEstimator <: MEstimator; width::Float64; end
estimator_rho(r,est::CauchyEstimator) = 0.5*est.width^2 * log(1 + r.*r/(est.width*est.width))
estimator_psi(r,est::CauchyEstimator) = r ./ (1.0 + r.*r/(est.width*est.width))
estimator_weight(r,est::CauchyEstimator) = 1.0 ./ (1.0 + r.*r/(est.width*est.width))
estimator_sqrtweight(r,est::CauchyEstimator) = 1.0 ./ sqrt(1.0 + r.*r/(est.width*est.width))
isconvex(::CauchyEstimator) = false

"""
A custom M-Estimator which composes multiple estimators on different ranges.

Construct with syntax:
    MultiEstimator(MEstimator => Range, MEstimator => Range, ...)

Unfilled ranges will automatically use the standard L2-estimator.
"""
immutable MultiEstimator{Estimators<:Tuple,Ranges<:Tuple}
    est::Estimators
    rng::Ranges

    function MultiEstimator(est::Estimators,rng::Ranges)
        if length(est) != length(rng)
            error("Must have same number of estimators and ranges")
        end
        for i = 1:length(est)
            if !isa(est[i],MEstimator)
                error("Expecting an M-estimator, got $(est[i]) of type $(typeof(est[i]))")
            end
            if !method_exists(Base.getindex,(Vector{Float64},typeof(rng[i])))
                error("Expecting an indexible range, got $(rng[i]) of type $(typeof(rng[i]))")
            end
        end
        new(est,rng)
    end
end
# Convenience constructors for up to 4 different ranges
MultiEstimator{Est1<:MEstimator,T1}(x::Pair{Est1,T1}) = MultiEstimator{Tuple{Est1},Tuple{T1}}((x.first,), (x.second,))
MultiEstimator{Est1<:MEstimator,T1,Est2<:MEstimator,T2}(x1::Pair{Est1,T1},x2::Pair{Est2,T2}) = MultiEstimator{Tuple{Est1,Est2},Tuple{T1,T2}}((x1.first,x2.first), (x1.second,x2.second))
MultiEstimator{Est1<:MEstimator,T1,Est2<:MEstimator,T2,Est3<:MEstimator,T3}(x1::Pair{Est1,T1},x2::Pair{Est2,T2},x3::Pair{Est3,T3}) = MultiEstimator{Tuple{Est1,Est2,Est3},Tuple{T1,T2,T3}}((x1.first,x2.first,x3.first), (x1.second,x2.second,x3.second))
MultiEstimator{Est1<:MEstimator,T1,Est2<:MEstimator,T2,Est3<:MEstimator,T3,Est4<:MEstimator,T4}(x1::Pair{Est1,T1},x2::Pair{Est2,T2},x3::Pair{Est3,T3},x4::Pair{Est4,T4}) = MultiEstimator{Tuple{Est1,Est2,Est3,Est4},Tuple{T1,T2,T3,T4}}((x1.first,x2.first,x3.first,x4.first), (x1.second,x2.second,x3.second,x4.second))
function estimator_rho(r,est::MultiEstimator)
    rho = 0.5*r.^2
    for i = 1:length(est.est)
        rho[est.rng[i]] = estimator_rho(r[est.rng[i]],est.est[i])
    end
    return rho
end
function estimator_psi(r,est::MultiEstimator)
    psi = r
    for i = 1:length(est.est)
        psi[est.rng[i]] = estimator_psi(r[est.rng[i]],est.est[i])
    end
    return psi
end
function estimator_weight(r,est::MultiEstimator)
    w = ones(size(r))
    for i = 1:length(est.est)
        w[est.rng[i]] = estimator_weight(r[est.rng[i]],est.est[i])
    end
    return w
end
function estimator_sqrtweight(r,est::MultiEstimator)
    w = ones(size(r))
    for i = 1:length(est.est)
        w[est.rng[i]] = estimator_sqrtweight(r[est.rng[i]],est.est[i])
    end
    return w
end
function isconvex(r,est::MultiEstimator)
    out = true
    for i = 1:length(est.est)
        out = out && isconvex(est.est[i])
    end
    return out
end

#=
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

=#

"""
Recreate an M-Estimator of the same type using the MAD (median absolute
deviation) of the residual
"""
reweight_mad{T<:MEstimator}(Est::Union{T,Type{T}}, res, factor=1.0) = T(factor*1.43*StatsBase.mad(res))
function estimator_mad{T<:MultiEstimator}(est::T, res, factor=1.0)
    tmp = ntuple(i->(typeof(est.est[i])(factor*1.43*StatsBase.mad(res[est.rng[i]])) => rng.rng[i]),length(est.est))
    return MultiEstimator(tmp...)
end


#function mestimator_mad{T}(Est::Union{Type{SubsetEstimator{T}},Type{SubsetEstimatorConvex{T}}}, rng, res, factor=1.0)
#    return Est(rng, T(factor*1.43*StatsBase.mad(res[rng])))
#end
#function mestimator_mad{T1,T2}(Est::Union{Type{DualEstimator{T1,T2}},Type{DualEstimatorConvex{T1,T2}}}, rng1, rng2, res, factor=1.0)
#    return Est(rng1, T1(factor*1.43*StatsBase.mad(res[rng1])), rng2, T2(factor*1.43*StatsBase.mad(res[rng2])))
#end

#mestimator_mad(est::MEstimator, res, factor=1.0) = typeof(est)(factor*1.43*StatsBase.mad(res))
#function mestimator_mad{T}(est::Union{SubsetEstimator{T},SubsetEstimatorConvex{T}}, res, factor=1.0)
#    return typeof(est)(est.rng, T(factor*1.43*StatsBase.mad(res[est.rng])))
#end
#function mestimator_mad{T1,T2}(est::Union{DualEstimator{T1,T2},DualEstimatorConvex{T1,T2}}, res, factor=1.0)
#    return typeof(est)(est.rng1, T1(factor*1.43*StatsBase.mad(res[est.rng1])), est.rng2, T2(factor*1.43*StatsBase.mad(res[est.rng2])))
#end
