
### abstract DataSet

abstract type DataSet end

copy(ds::DS) where {DS<:DataSet} = DS(fields(ds)...)
hash(ds::DataSet, h::UInt64) = foldr(hash, (typeof(ds), fieldvalues(ds)...), init=h)
function show(io::IO, ds::DataSet)
    print(io, typeof(ds), "(", join(String.(fieldnames(typeof(ds))), ", "), ")")
end

function (ds::DataSet)(θ) 
    DS = typeof(ds)
    DS(map(fieldvalues(ds)) do v
        (v isa Union{ParamDependentOp,DataSet}) ? v(θ) : v
    end...)
end
(ds::DataSet)(;θ...) = ds((;θ...))
adapt_structure(to, ds::DS) where {DS <: DataSet} = DS(adapt(to, fieldvalues(ds))...)

# called when simulating a DataSet, this gets the batching right
function simulate(rng::AbstractRNG, ds::DataSet, dist::MvNormal{<:Any,<:PDFieldOpWrapper})
    Nbatch = (isnothing(ds.d) || batch_length(ds.d) == 1) ? () : batch_length(ds.d)
    rand(rng, dist; Nbatch)
end

# mixed DataSet wrapper, 
struct Mixed{DS<:DataSet} <: DataSet
    ds :: DS
end

# prior by default is attached to DataSet object for easy tweaking
logprior(ds::DataSet; Ω...) = ds.logprior(;Ω...)

### builtin DataSet objects

@kwdef mutable struct NoLensingDataSet <: DataSet
    d = nothing             # data
    Cf                      # unlensed field covariance
    Cn                      # noise covariance
    Cn̂ = Cn                 # approximate noise covariance, diagonal in same basis as Cf
    M  = I                  # user mask
    M̂  = M                  # approximate user mask, diagonal in same basis as Cf
    Mborder = nothing       # border mask for use with super sample prior
    B  = I                  # beam and instrumental transfer functions
    B̂  = B                  # approximate beam and instrumental transfer functions, diagonal in same basis as Cf
    logprior = (;_...) -> 0 # default no prior
end

@composite @kwdef mutable struct BaseDataSet <: DataSet
    NoLensingDataSet...
    Cϕ               # ϕ covariance
    Cf̃ = nothing     # lensed field covariance (not always needed)
    D  = I           # mixing matrix for mixed parametrization
    G  = I           # reparametrization for ϕ
    L  = LenseFlow   # lensing operator, possibly cached for memory reuse
    Nϕ = nothing     # some estimate of the ϕ noise, used in several places for preconditioning
    σ²κ = nothing    # Width of <κ> logprior
end

@composite @kwdef mutable struct FGroundDataSet <: DataSet
    BaseDataSet...
    Cg               # Foreground covariance, just poisson for now 
    Ng               # Initial noise estimate for hessian preconditioner
end

@fwdmodel function (ds::FGroundDataSet)(; f, ϕ, g, θ=(;), d=ds.d)
    @unpack Cg, Cf, Cϕ, Cn, L, M, B = ds
    f ~ MvNormal(0, Cf(θ))
    ϕ ~ MvNormal(0, Cϕ(θ))
    g ~ MvNormal(0, Cg(θ))
    f̃ ← L(ϕ) * f
    μ = M(θ) * (B(θ) * (f̃ + g))
    d ~ MvNormal(μ, Cn(θ))
    #@isdefined(_logpdf) && (_logpdf[] += logprior(ds; f, ϕ, g, θ, d))
end


@fwdmodel function (ds::BaseDataSet)(; f, ϕ, θ=(;), d=ds.d)
    @unpack Cf, Cϕ, Cn, L, M, B = ds
    f ~ MvNormal(0, Cf(θ))
    ϕ ~ MvNormal(0, Cϕ(θ))
    f̃ ← L(ϕ) * f
    μ = M(θ) * (B(θ) * f̃)
    d ~ MvNormal(μ, Cn(θ))
    #@isdefined(_logpdf) && (_logpdf[] += logprior(ds; f, ϕ, θ, d))
end

@fwdmodel function (ds::NoLensingDataSet)(; f, θ=(;), d=ds.d)
    @unpack Cf, Cn, M, B = ds
    f ~ MvNormal(0, Cf(θ))
    μ = M(θ) * (B(θ) * f)
    d ~ MvNormal(μ, Cn(θ))
    #@isdefined(_logpdf) && (_logpdf[] += logprior(ds; f, θ, d))
end

######### Super-Sample prior
function logprior(ds::FGroundDataSet; ϕ, _...)
    (;Mborder, σ²κ) = ds
    -(sum(Mborder * (∇² * ϕ)) / sum(diag(Mborder)))^2 / (2*σ²κ)
end

function logprior(ds::BaseDataSet; ϕ, _...)
    (;Mborder, σ²κ) = ds
    -(sum(Mborder * (∇² * ϕ)) / sum(diag(Mborder)))^2 / (2*σ²κ)
end


# performance optimization (shouldn't need this once we have Diffractor)
function gradientf_logpdf(ds::BaseDataSet; f, ϕ, θ=(;), d=ds.d)
    @unpack Cf, Cϕ, Cn, L, M, B = ds
    (Lϕ, Mθ, Bθ) = (L(ϕ), M(θ), B(θ))
    Lϕ' * (Bθ' * (Mθ' * (pinv(Cn(θ)) * (d - Mθ * (Bθ * (Lϕ * f)))))) - pinv(Cf(θ)) * f
end


## mixing
function Distributions.logpdf(mds::Mixed{<:DataSet}; θ=(;), Ω...)
    ds = mds.ds
    logpdf(ds; unmix(ds; θ, Ω...)...) - logdet(ds.D, θ) - logdet(ds.G, θ)
end

"""
    mix(ds::DataSet; f, ϕ, [θ])
    
Compute the mixed `(f°, ϕ°)` from the unlensed field `f` and lensing potential
`ϕ`, given the definition of the mixing matrices in `ds` evaluated at parameters
`θ` (or at fiducial values if no `θ` provided).
"""
function mix(ds::DataSet; f, ϕ, θ=(;), Ω...)
    @unpack D, G, L = ds
    f° = L(ϕ) * D(θ) * f
    ϕ° = G(θ) * ϕ
    (; f°, ϕ°, θ, Ω...)
end


"""
    unmix(f°, ϕ°,    ds::DataSet)
    unmix(f°, ϕ°, θ, ds::DataSet)

Compute the unmixed/unlensed `(f, ϕ)` from the mixed field `f°` and mixed
lensing potential `ϕ°`, given the definition of the mixing matrices in `ds`
evaluated at parameters `θ` (or at fiducial values if no `θ` provided). 
"""
function unmix(ds::DataSet; f°, ϕ°, θ=(;), Ω...)
    @unpack D, G, L = ds
    ϕ = G(θ) \ ϕ°
    f = D(θ) \ (L(ϕ) \ f°)
    (; f, ϕ, θ, Ω...)
end


## preconditioning

# Should return an operator which is fast to apply and which
# approximates the Hessian of logpdf w.r.t. the symbols in Ω.

Hessian_logpdf_preconditioner(Ω::Union{Symbol,Tuple}, ds::DataSet) = Hessian_logpdf_preconditioner(Val(Ω), ds)

function Hessian_logpdf_preconditioner(Ω::Val{:f}, ds::DataSet)
    @unpack Cf, B̂, M̂, Cn̂ = ds
    pinv(Cf) + B̂'*M̂'*pinv(Cn̂)*M̂*B̂
end

function Hessian_logpdf_preconditioner(Ω::Val{(:ϕ°,)}, ds::DataSet)
    @unpack Cϕ, Nϕ = ds
    Diagonal(FieldTuple(ϕ°=diag(pinv(Cϕ)+pinv(Nϕ))))
end

function Hessian_logpdf_preconditioner(Ω::Val{(:ϕ°,:g)}, ds::FGroundDataSet)
    @unpack Cg, Cϕ, Nϕ = ds
    Diagonal(FieldTuple(ϕ°=diag(pinv(Cϕ)+pinv(Nϕ)), g=diag(pinv(Cg))))
end

@doc doc"""

    load_sim(;kwargs...)

The starting point for many typical sessions. Creates a `BaseDataSet`
object with some simulated data, returing the DataSet and simulated
truths, which can then be passed to other maximization / sampling
functions. E.g.:

```julia
@unpack f,ϕ,ds = load_sim(;
    θpix  = 2,
    Nside = 128,
    pol   = :P,
    T     = Float32
)
```

Keyword arguments: 

* `θpix` — Angular resolution, in arcmin. 
* `Nside` — Number of pixels in the map as an `(Ny,Nx)` tuple, or a
  single number for square maps. 
* `pol` — One of `:I`, `:P`, or `:IP` to select intensity,
  polarization, or both. 
* `T = Float32` — Precision, either `Float32` or `Float64`.
* `storage = Array` — Set to `CuArray` to use GPU.
* `Nbatch = nothing` — Number of batches of data in this dataset.
* `μKarcminT = 3` — Noise level in temperature in μK-arcmin.
* `ℓknee = 100` — 1/f noise knee.
* `αknee = 3` — 1/f noise slope.
* `beamFWHM = 0` — Beam full-width-half-max in arcmin.
* `pixel_mask_kwargs = (;)` — NamedTuple of keyword arguments to
  pass to `make_mask` to create the pixel mask.
* `bandpass_mask = LowPass(3000)` — Operator which performs
  Fourier-space masking.
* `fiducial_θ = (;)` — NamedTuple of keyword arguments passed to
  `camb()` for the fiducial model.
* `seed = nothing` — Specific seed for the simulation.
* `L = LenseFlow` — Lensing operator.

Returns a named tuple of `(;f, f̃, ϕ, n, ds, Cℓ, proj)`.


"""
function load_sim(;
    
    # basic configuration
    θpix,
    Nside,
    pol,
    T = Float32,
    storage = Array,
    rotator = (0,90,0),
    Nbatch = nothing,
    
    # noise parameters, or set Cℓn or even Cn directly
    μKarcminT = 3,
    ℓknee = 100,
    αknee = 3,
    Cℓn = nothing,
    Cn = nothing,
    
    # beam parameters, or set B directly
    beamFWHM = 0,
    B = nothing, B̂ = nothing,
    
    # mask parameters, or set M directly
    pixel_mask_kwargs = nothing,
    bandpass_mask = LowPass(3000),
    M = nothing, M̂ = nothing,

    # theory
    Cℓ = nothing,
    fiducial_θ = (;),
    rfid = nothing,
    
    seed = nothing,
    rng = MersenneTwister(seed),
    D = nothing,
    G = nothing,
    Nϕ_fac = 2,
    L = LenseFlow,

    σ²κ = nothing

)
    
    # projection
    Ny, Nx = Nside .* (1,1)
    proj = ProjLambert(; Ny, Nx, θpix, T, storage, rotator)

    # the biggest ℓ on the 2D fourier grid
    ℓmax = round(Int,ceil(√2*proj.nyquist)+1)
    
    # CMB Cℓs
    if (rfid != nothing)
        @warn "`rfid` will be removed in a future version. Use `fiducial_θ=(r=...,)` instead."
        fiducial_θ = merge(fiducial_θ,(r=rfid,))
    end
    Aϕ₀ = T(get(fiducial_θ, :Aϕ, 1))
    fiducial_θ = Base.structdiff(fiducial_θ, NamedTuple{(:Aϕ,)}) # remove Aϕ key if present
    if (Cℓ == nothing)
        Cℓ = camb(;fiducial_θ..., ℓmax=ℓmax)
    else
        if !isempty(fiducial_θ)
            error("Can't pass both `Cℓ` and `fiducial_θ` parameters which affect `Cℓ`, choose one or the other.")
        elseif maximum(Cℓ.total.TT.ℓ) < ℓmax
            error("ℓmax of `Cℓ` argument should be higher than $ℓmax for this configuration.")
        end
    end
    r₀ = T(Cℓ.params.r)
    
    # noise Cℓs (these are non-debeamed, hence beamFWHM=0 below; the beam comes in via the B operator)
    if (Cℓn == nothing)
        Cℓn = noiseCℓs(μKarcminT=μKarcminT, beamFWHM=0, ℓknee=ℓknee, αknee=αknee, ℓmax=ℓmax)
    end
    
    # some things which depend on whether we chose :I, :P, or :IP
    pol = Symbol(pol)
    ks,F,F̂,nF = @match pol begin
        :I  => ((:TT,),            FlatMap,    FlatFourier,    1)
        :P  => ((:EE,:BB),         FlatQUMap,  FlatEBFourier,  2)
        :IP => ((:TT,:EE,:BB,:TE), FlatIQUMap, FlatIEBFourier, 3)
        _   => throw(ArgumentError("`pol` should be one of :I, :P, or :IP"))
    end
    
    # covariances
    Cϕ₀ = Cℓ_to_Cov(:I,  proj, (Cℓ.total.ϕϕ))
    Cfs = Cℓ_to_Cov(pol, proj, (Cℓ.unlensed_scalar[k] for k in ks)...)
    Cft = Cℓ_to_Cov(pol, proj, (Cℓ.tensor[k]          for k in ks)...)
    Cf̃  = Cℓ_to_Cov(pol, proj, (Cℓ.total[k]           for k in ks)...)
    Cn̂  = Cℓ_to_Cov(pol, proj, (Cℓn[k]                for k in ks)...)
    if (Cn == nothing); Cn = Cn̂; end
    Cf = ParamDependentOp((;r=r₀,   _...)->(Cfs + (T(r)/r₀)*Cft))
    Cϕ = ParamDependentOp((;Aϕ=Aϕ₀, _...)->(T(Aϕ) * Cϕ₀))
    
    # data mask
    if (M == nothing)
        Mfourier = Cℓ_to_Cov(pol, proj, ((k==:TE ? 0 : 1) * bandpass_mask.diag.Wℓ for k in ks)...; units=1)
        if (pixel_mask_kwargs != nothing)
            Mpix = adapt(storage, Diagonal(F(repeated(T.(make_mask(copy(rng),Nside,θpix; pixel_mask_kwargs...).Ix),nF)..., proj)))
        else
            Mpix = I
        end
        M = Mfourier * Mpix
        if (M̂ == nothing)
            M̂ = Mfourier
        end
    else
        if (M̂ == nothing)
            M̂ = M
        end
    end
    if (M̂ isa DiagOp{<:BandPass})
        M̂ = Diagonal(M̂ * one(diag(Cf)))
    end
    
    # beam
    if (B == nothing)
        B = Cℓ_to_Cov(pol, proj, ((k==:TE ? 0 : 1) * sqrt(beamCℓs(beamFWHM=beamFWHM)) for k=ks)..., units=1)
    end
    if (B̂ == nothing)
        B̂ = B
    end
    
    # preallocate lensing operator memory
    Lϕ = precompute!!(L(zero(diag(Cϕ))), zero(diag(Cf)))

    # put everything in DataSet
    ds = BaseDataSet(;Cn, Cn̂, Cf, Cf̃, Cϕ, M, M̂, B, B̂, D, L=Lϕ, σ²κ)
    
    # simulate data
    @unpack f,f̃,ϕ,d = simulate(rng, ds)
    ds.d = d

    # with the DataSet created, we now more conveniently create the mixing matrices D and G
    ds.Nϕ = Nϕ = quadratic_estimate(ds).Nϕ / Nϕ_fac
    if (G == nothing)
        G₀ = sqrt(I + 2 * Nϕ * pinv(Cϕ()))
        ds.G = ParamDependentOp((;Aϕ=Aϕ₀, _...)->(pinv(G₀) * sqrt(I + 2 * Nϕ * pinv(Cϕ(Aϕ=Aϕ)))))
    end
    if (D == nothing)
        σ²len = T(deg2rad(5/60)^2)
        ds.D = ParamDependentOp(
            function (;r=r₀, _...)
                Cfr = Cf(;r=r)
                sqrt((Cfr + I*σ²len + 2*Cn̂) * pinv(Cfr))
            end,
        )
    end

    if Nbatch != nothing
        d = ds.d *= batch(ones(Int,Nbatch))
        ds.L = precompute!!(L(ϕ*batch(ones(Int,Nbatch))), ds.d)
    end
    
    return (;f, f̃, ϕ, d, ds, ds₀=ds(), Cℓ, proj)
    
end


# make foreground dataset with lensing covariance parameterised with bandpowers

function load_fground_ds(;
    
    # basic configuration
    θpix,
    Nside,
    pol,
    T = Float64,
    storage = Array,
    rotator = (0,90,0),
    Nbatch = nothing,
    
    # noise parameters, or set Cℓn or even Cn directly
    μKarcminT = 3,
    ℓknee = 100,
    αknee = 3,
    Cℓn = nothing,
    Cn = nothing,
    
    # beam parameters, or set B directly
    beamFWHM = 0,
    B = nothing, B̂ = nothing,
    
    # mask parameters, or set M directly
    pixel_mask_kwargs = nothing,
    bandpass_mask = LowPass(3000),
    M = nothing, M̂ = nothing,

    # theory
    Cℓ = nothing,
    fiducial_θ = (;),
    rfid = nothing,
    
    seed = 42,
    rng = nothing,
    D = nothing,
    G = nothing,
    Nϕ_fac = 2,
    L = LenseFlow(11),

    σ²κ = nothing,

    # Bin edges for fields
    ℓedges_ϕ = nothing,
    ℓedges_T = nothing,
    ℓedges_g = nothing,

    # Foreground field parameterization
    # Either supply a spectrum, or specify amplitude, power law index and pivot scale
    # if ℓedges_g = nothing, Cg will be a ParamDependentOp, where Cg = Cg(Ag, αg)
    Ag = 5e-6,
    αg = 0,
    ℓpivot_fg = 1500,
    Cℓ_fg = nothing, # Template for foreground spectrum.

    Ng = nothing, ######## Initial noise est for hessian pre-conditioner on g. Not implemented for now
)
    
    ℓedges_ϕ == nothing ? ( log_edges = range(log(150),log(3000), 13) ; ℓedges_ϕ = T.(exp.(log_edges))  ) : ()
    ℓedges_ϕ = T.(ℓedges_ϕ) 

    ###################### check bin limits against map dimensions
    length(Nside) == 1 ? N=Nside : N = findmax(Nside)[1]
    ℓmin = 2*180/(N*(θpix/60))# Simulated power spectra have lower ℓ = 2ℓmin
    #ℓmin > ℓedges_ϕ[1] ? @warn("WARNING : ℓedges_ϕ[1] too small for map dimensions. ℓmin = $ℓmin ℓedges_ϕ[1] = $(ℓedges_ϕ[1])")  : () <- wrong. 23/03/23
    ℓmax = (√2)*180/(θpix/60)
    #println("ℓmin = $ℓmin : ℓedges_ϕ[1] = $(ℓedges_ϕ[1]) \n ℓmax = $ℓmax : ℓedges_ϕ[end] = $(ℓedges_ϕ[end])")
    ℓend = floor(Int32,ℓedges_ϕ[end])
 
    RNG = @something(rng, MersenneTwister(seed))
    ################### Baseline Sim from CMBLensing
    @unpack ds,proj = load_sim(;
        # basic configuration
    θpix, Nside, pol, T, storage, rotator, Nbatch,
    
    # noise parameters, or set Cℓn or even Cn directly
    μKarcminT, ℓknee, αknee, Cℓn, Cn,
    
    # beam parameters, or set B directly
    beamFWHM, B, B̂,
    
    # mask parameters, or set M directly
    pixel_mask_kwargs, bandpass_mask,
    M, M̂,

    # theory
    Cℓ, fiducial_θ, rfid,

    rng = RNG, D, G, Nϕ_fac, L
    );
    
    ###########################  Foreground Covariance
    Ag₀ = T(Ag) ;  αg₀ = T(αg)
    ####### Make Cg dependent on one Amplitude/tilt or bandpowers
    ℓs=Cℓ.unlensed_scalar.TT.ℓ
    if ℓedges_g == nothing
        # Not neccesarily correct
        Cg = ParamDependentOp( (;Ag=Ag₀, αg=αg₀, _...)-> Cℓ_to_Cov(  :I, proj, (Cℓs(ℓs , Ag*(ℓs./ℓpivot_fg).^αg)) ) )
        #=
        Cg0 = LambertFourier( ( (1/ℓpivot_fg)*proj.ℓmag) ,proj )
        # Get Inf at ℓ = 0 if αg -ve 
        if αg < 0
            ind_zero = Cg0 .== 0
            Cg0[ind_zero] .= 1e-3
        end
        Cg = let Cg0 = Cg0
            ParamDependentOp( (;Ag=Ag₀, αg=αg₀, _...)->Diagonal(Ag*Cg0.^αg ./ proj.Ωpix) )
        end
        =#
    else
        if Cℓ_fg == nothing 
            Cℓ_fg = Cℓs(ℓs, Ag*(ℓs./ℓpivot_fg).^αg)
        end
        Cg = Cℓ_to_Cov(:I, proj,( Cℓ_fg , ℓedges_g, :Ag))
    end
    ###########################  Bandpower dependent Cϕ and Cf
    ########################### Mixing matrix G, also depends on Aϕ

    Nϕ=ds.Nϕ # QE noise est : Nϕ=quadratic_estimate(ds).Nϕ / Nϕ_fac
    nbins_ϕ = length(ℓedges_ϕ)-1
    
    Cϕ=Cℓ_to_Cov(:I, proj,(Cℓ.unlensed_total.ϕϕ, ℓedges_ϕ, :Aϕ))
    G₀ = sqrt(I + Nϕ * pinv(Cϕ()))
    Aϕ₀= ones(nbins_ϕ)
    G = ParamDependentOp((;Aϕ=Aϕ₀, _...)->(pinv(G₀) * sqrt(I + 2 * Nϕ * pinv(Cϕ(Aϕ=Aϕ)))))
    

    ℓedges_T==nothing ? Cf=ds.Cf : Cf=Cℓ_to_Cov(:I, proj,(Cℓ.unlensed_total.TT, ℓedges_T, :AT))
    ########################## Simulate data
    
    ######## Initial noise est
    ######## for hessian pre-conditioner on g
    @unpack Nϕ = ds
    Ng == nothing ? Ng=Nϕ*10^15 : ()
    ######## Include Cf̃ in dataset
    Cf̃  = Cℓ_to_Cov(:I, proj,Cℓ.total.TT)

    ######
    ds = FGroundDataSet(;Cf, Cf̃, Cϕ, Cg, Ng, Cn=ds.Cn, Cn̂=ds.Cn̂, M=ds.M, M̂=ds.M̂, B=ds.B, B̂=ds.B̂, Nϕ=ds.Nϕ, L=ds.L, D=ds.D, G, σ²κ)
    @unpack f,g,ϕ,d = simulate(RNG,ds)
    ds.d = d;
    
    
    return (;ds,f,g,ϕ, proj)
end


function load_nolensing_sim(; 
    lensed_covariance = false, 
    lensed_data = false,
    L = lensed_data ? LenseFlow : I,
    kwargs...
)
    @unpack f, f̃, ϕ, ds, ds₀, Cℓ, proj = load_sim(; L, kwargs...)
    @unpack d, Cf, Cf̃, Cn, Cn̂, M, M̂, B, B̂ = ds
    Cf_nl = lensed_covariance ? Cf̃ : Cf
    ds_nl = NoLensingDataSet(; d, Cf=Cf_nl, Cn, Cn̂, M, M̂, B, B̂)
    (;f, f̃, ϕ, ds=ds_nl, ds₀=ds_nl(), Cℓ, proj)
end


### distributed DataSets

# bijection between (name::Symbol) => hash(Main.$name)
const distributed_datasets = (name_hash=Bijection{Symbol,UInt}(), objid_hash=Bijection{UInt,UInt}())

@doc doc"""
    CMBLensing.@distributed ds1 ds2 ...

Assuming `ds1`, `ds2`, etc... are DataSet objects which are defined in
the Main module on all workers, this makes it so that whenever these
objects are shipped to a worker as part of a remote call, the data is
not actually sent, but rather the worker just refers to their existing
local copy. Typical usage:

    @everywhere ds = load_sim(seed=1, ...)
    CMBLensing.@distributed ds
    pmap(1:n) do i
        # do something with ds
    end

Note that `hash(ds)` must yield the same value on all processors, ie
the macro checks that it really is the same object on all processors.
Sometimes setting the same random seed is not enough to ensure this as
there may be tiny numerical differences in the simulated data. In this
case you can try:

    @everywhere ds.d = $(ds.d)

after loading the dataset to explicitly set the data based on the
simulation on the master process.

Additionally, if the dataset object has fields which are custom types,
these must have an appropriate `Base.hash` defined. 
"""
macro distributed(datasets...)
    distributed1(name, ds) = quote
        hash_ds = hash($ds::DataSet)
        for id in workers()
            if hash_ds != remotecall_fetch(()->hash(Base.eval(Main,$name)), id)
                error("Main.$($name) on master and worker $(id) do not match.")
            end
        end
        ($name in domain(distributed_datasets.name_hash)) && delete!(distributed_datasets.name_hash, $name)
        (objectid($ds) in domain(distributed_datasets.objid_hash)) && delete!(distributed_datasets.objid_hash, objectid($ds))
        distributed_datasets.name_hash[$name] = distributed_datasets.objid_hash[objectid($ds)] = hash_ds
    end
    Expr(:block, [distributed1(name, ds) for (name, ds) in zip(QuoteNode.(datasets), esc.(datasets))]..., nothing)
end
function Serialization.serialize(s::AbstractSerializer, ds::DataSet)
    hash_ds = hash(ds)
    original_hash = get(distributed_datasets.objid_hash, objectid(ds), nothing)
    name = get(inv(distributed_datasets.name_hash), hash_ds, nothing)
    if name == original_hash == nothing
        Base.@invoke(Serialization.serialize(s::AbstractSerializer, ds::Any))
    elseif original_hash != hash_ds
        error("DataSet object has been modified since it was marked as distributed.")
    else
        hash(Base.eval(Main,name)) == hash_ds || error("Main.$name has been modified since it was marked as distributed.")
        Serialization.writetag(s.io, Serialization.OBJECT_TAG)
        Serialization.serialize(s, Val{:DataSet})
        Serialization.serialize(s, name)
    end
end
function Serialization.deserialize(s::AbstractSerializer, ::Type{Val{:DataSet}})
    name = Serialization.deserialize(s)
    Base.eval(Main, name)
end
