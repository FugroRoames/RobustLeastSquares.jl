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
function estimator_rho end
"The derivative of the cost (a.k.a. loss) function ψ for the M-estimator"
function estimator_psi end
# TODO m_estimator_psi(r,est::MEstimator) = first derivative of m_estimator_rho(r,est) w.r.t. r
"""
The weight function, w, for the M-estimator, to be used for modifying the normal
equations of a least-square problem
"""
estimator_weight(r,est::MEstimator) = m_estimator_psi(r,est) ./ r
"""
The square root of the weight function, sqrt(w), for the M-estimator, to be used
for modifying a least-squares problem
"""
estimator_sqrtweight(r,est::MEstimator) = sqrt(m_estimator_weight(r,est))
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
estimator_psi(r,::L1L2Estimator) = r ./ sqrt(1.0 + 0.5*r.*r)
estimator_weight(r,::L1L2Estimator) = 1.0 ./ sqrt(1+0.5*r.*r)
estimator_sqrtweight(r,::L1L2Estimator) = (1.0 + 0.5*r.*r) .^ (-1/4)
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
    psi = copy(r)
    absr = abs(r)
    psi[absr .> est.width] = est.width * sign(r[absr .> est.width])
    return psi
end
function estimator_weight(r,est::HuberEstimator)
    w = ones(size(r))
    absr = abs(r)
    w[absr .> est.width] = est.width ./ absr[absr .> est.width]
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
estimator_rho(r,est::FairEstimator) = est.width^2 * (abs(r)/est.width - log(1.0 + abs(r)/est.width))
estimator_psi(r,est::FairEstimator) = r ./ (1.0 + abs(r)/est.width)
estimator_weight(r,est::FairEstimator) = 1.0 ./ (1.0 + abs(r)/est.width)
estimator_sqrtweight(r,est::FairEstimator) = 1.0 ./ sqrt(1.0 + abs(r)/est.width)
isconvex(::FairEstimator) = true

"""
The non-convex Cauchy estimator switches from between quadratic behaviour to
logarithmic tails. This rejects outliers but may result in mutliple minima.
"""
immutable CauchyEstimator <: MEstimator; width::Float64; end
estimator_rho(r,est::CauchyEstimator) = 0.5*est.width^2 * log(1.0 + r.*r/(est.width*est.width))
estimator_psi(r,est::CauchyEstimator) = r ./ (1.0 + r.*r/(est.width*est.width))
estimator_weight(r,est::CauchyEstimator) = 1.0 ./ (1.0 + r.*r/(est.width*est.width))
estimator_sqrtweight(r,est::CauchyEstimator) = 1.0 ./ sqrt(1.0 + r.*r/(est.width*est.width))
isconvex(::CauchyEstimator) = false


"""
The non-convex Geman-McClure for strong supression of ourliers and does not guarantee a unique solution
"""
immutable GemanEstimator <: MEstimator; end
estimator_rho(r,est::GemanEstimator) = 0.5*r.*r ./(1.0 + r.*r)
estimator_psi(r,est::GemanEstimator) = r ./ (1.0 + r.*r).^2
estimator_weight(r,est::GemanEstimator) = 1.0 ./ (1.0 + r.*r).^2
estimator_sqrtweight(r,est::GemanEstimator) = 1.0 ./ (1.0 + r.*r)
isconvex(::GemanEstimator) = false


"""
The non-convex Welsch for strong supression of ourliers and does not guarantee a unique solution
"""
immutable WelschEstimator <: MEstimator; width::Float64; end
estimator_rho(r,est::WelschEstimator) = 0.5*est.width^2 *( 1.0 - exp(-(r/est.width).^2) )
estimator_psi(r,est::WelschEstimator) = r.* exp(-(r/est.width).^2)
estimator_weight(r,est::WelschEstimator) = exp(-(r/est.width).^2)
estimator_sqrtweight(r,est::WelschEstimator) = exp(-0.5*(r/est.width).^2)
isconvex(::WelschEstimator) = false


"""
The non-convex Tukey biweight estimator which completly suppress the outliers,
and does not guarantee a unique solution
"""
immutable TukeyEstimator <: MEstimator; width::Float64; end
function estimator_rho(r,est::TukeyEstimator)
    rho = est.width^2/6.0 * (1.0 - (1.0 - (r/est.width).^2 ).^3)
    rho[est.width .< abs(r)] = est.width^2/6.0
    return rho
end
function estimator_psi(r,est::TukeyEstimator)
    psi = r.* (1.0 - (r/est.width).^2).^2
    psi[est.width .< abs(r)] = 0.0
    return psi
end
function estimator_weight(r,est::TukeyEstimator)
    w = (1.0 - (r/est.width).^2).^2
    w[est.width .< abs(r)] = 0.0
    return w
end
function estimator_sqrtweight(r,est::TukeyEstimator)
    w = (1.0 - (r/est.width).^2)
    w[est.width .< abs(r)] = 0.0
    return w
end
isconvex(::TukeyEstimator) = true


"""
A custom M-Estimator which composes multiple estimators on different ranges.

Construct with syntax:
    MultiEstimator(MEstimator => Range, MEstimator => Range, ...)

Unfilled ranges will automatically use the standard L2-estimator.
"""
immutable MultiEstimator{Estimators<:Tuple,Ranges<:Tuple} <: MEstimator
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


"""
Recreate an M-Estimator of the same type using the median absolute
deviation of the residual
"""
refit_estimator{T<:MEstimator}(Est::Union{T,Type{T}}, res, factor=1.0) = T(factor*1.43*StatsBase.mad(res))
function refit_estimator{T<:MultiEstimator}(est::T, res, factor=1.0)
    tmp = ntuple(i->(typeof(est.est[i])(factor*1.43*StatsBase.mad(res[est.rng[i]])) => est.rng[i]),length(est.est))
    return MultiEstimator(tmp...)
end
