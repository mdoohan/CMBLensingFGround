# interface with MuseInference.jl

using .MuseInference: AbstractMuseProblem, MuseResult, Transformedθ, UnTransformedθ
using .MuseInference.AbstractDifferentiation

export CMBLensingMuseProblem

struct CMBLensingMuseProblem{DS<:DataSet,DS_SIM<:DataSet} <: AbstractMuseProblem
    ds :: DS
    ds_for_sims :: DS_SIM
    parameterization 
    MAP_joint_kwargs
    θ_fixed 
    x 
    latent_vars
    #autodiff = AD.HigherOrderBackend((AD.ForwardDiffBackend(tag=false), AD.ZygoteBackend()))
    #transform_θ = identity
    #inv_transform_θ = identity
end
#CMBLensingMuseProblem(ds, ds_for_sims=ds; kwargs...) = CMBLensingMuseProblem(;ds, ds_for_sims, kwargs...)

function CMBLensingMuseProblem(
    ds, 
    ds_for_sims = ds; 
    parameterization = 0, 
    MAP_joint_kwargs = (;), 
    θ_fixed = (;), 
    latent_vars = nothing,
    autodiff = AD.HigherOrderBackend((AD.ForwardDiffBackend(tag=false), AD.ZygoteBackend())),
)
    parameterization == 0 || error("only parameterization=0 (unlensed parameterization) currently implemented")
    CMBLensingMuseProblem(ds, ds_for_sims, parameterization, MAP_joint_kwargs, θ_fixed, ds.d, latent_vars, autodiff)
end

mergeθ(prob::CMBLensingMuseProblem, θ) = isempty(prob.θ_fixed) ? θ : (;prob.θ_fixed..., θ...)

function MuseInference.standardizeθ(prob::CMBLensingMuseProblem, θ)
    θ isa Union{NamedTuple,ComponentVector} || error("θ should be a NamedTuple or ComponentVector")
    1f0 * ComponentVector(θ) # ensure component vector and float
end

MuseInference.transform_θ(prob::CMBLensingMuseProblem, θ) = prob.transform_θ(θ)
MuseInference.inv_transform_θ(prob::CMBLensingMuseProblem, θ) = prob.inv_transform_θ(θ)

function MuseInference.logLike(prob::CMBLensingMuseProblem, d, z, θ, ::UnTransformedθ) 
    logpdf(prob.ds; z..., θ = mergeθ(prob, θ), d)
end
function MuseInference.logLike(prob::CMBLensingMuseProblem, d, z, θ, ::Transformedθ) 
    MuseInference.logLike(prob, d, z, MuseInference.inv_transform_θ(prob, θ), UnTransformedθ())
end

function MuseInference.∇θ_logLike(prob::CMBLensingMuseProblem, d, z, θ, θ_space)
    AD.gradient(prob.autodiff, θ -> MuseInference.logLike(prob, d, z, θ, θ_space), θ)[1]
end

function MuseInference.sample_x_z(prob::CMBLensingMuseProblem, rng::AbstractRNG, θ) 
    sim = simulate(rng, prob.ds_for_sims, θ = mergeθ(prob, θ))
    if prob.latent_vars == nothing
        # this is a guess which might not work for everything necessarily
        z = LenseBasis(FieldTuple(delete(sim, (:f̃, :d, :μ))) )
    else
        z = LenseBasis(FieldTuple(select(sim, prob.latent_vars)))
    end
    x = sim.d
    (;x, z)
end

function MuseInference.sample_x_z(prob::CMBLensingMuseProblem{FGroundDataSet}, rng::AbstractRNG, θ) 
    s = simulate(rng, prob.ds_for_sims, θ = CMBLensing.mergeθ(prob, θ))
    (;x=s.d, z=(;s.f, s.ϕ, s.g))
end

function MuseInference.ẑ_at_θ(prob::CMBLensingMuseProblem, d, zguess, θ; ∇z_logLike_atol=nothing)
    @unpack ds = prob
    Ωstart = delete(NamedTuple(zguess), :f)
    MAP = MAP_joint(mergeθ(prob, θ), @set(ds.d=d), Ωstart; fstart=zguess.f, prob.MAP_joint_kwargs...)
    LenseBasis(FieldTuple(;delete(MAP, :history)...)), MAP.history
end

function MuseInference.ẑ_at_θ(prob::CMBLensingMuseProblem{<:NoLensingDataSet}, d, (f₀,), θ; ∇z_logLike_atol=nothing)
    @unpack ds = prob
    LenseBasis(FieldTuple(f=argmaxf_logpdf(I, mergeθ(prob, θ), @set(ds.d=d); fstart=f₀, prob.MAP_joint_kwargs...))), nothing
end

function MuseInference.muse!(result::MuseResult, ds::DataSet, θ₀=nothing; parameterization=0, MAP_joint_kwargs=(;), kwargs...)
    muse!(result, CMBLensingMuseProblem(ds; parameterization, MAP_joint_kwargs), θ₀; kwargs...)
end