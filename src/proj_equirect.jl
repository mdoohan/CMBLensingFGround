# TODO: still need to check the spin(+2) or spin(-2) sta
# TODO: summary methods for BlockDiagEquiRect{B} and Adjoint{T,BlockDiagEquiRect{B}}


# Type defs
# ================================================

struct ProjEquiRect{T} <: CartesianProj

    Ny          :: Int
    Nx          :: Int
    θspan       :: Tuple{Float64,Float64}
    φspan       :: Tuple{Float64,Float64}
    θ           :: Vector{Float64} 
    φ           :: Vector{Float64} 
    θ∂          :: Vector{Float64} 
    φ∂          :: Vector{Float64} 
    Ω           :: Vector{Float64} 
    
    storage

end

struct BlockDiagEquiRect{B<:Basis, P<:ProjEquiRect, T, A<:AbstractArray{T}}  <: ImplicitOp{T}

    blocks :: A
    proj :: P

end

struct AzFourier <: S0Basis end
const  QUAzFourier = Basis2Prod{    𝐐𝐔, AzFourier }
const IQUAzFourier = Basis3Prod{ 𝐈, 𝐐𝐔, AzFourier }

# Type Alias
# ================================================

make_field_aliases(
    "EquiRect",  ProjEquiRect, 
    extra_aliases=OrderedDict(
        "AzFourier"    => AzFourier,
        "QUAzFourier"  => QUAzFourier,
        "IQUAzFourier" => IQUAzFourier,
    ),
)

typealias_def(::Type{<:ProjEquiRect{T}}) where {T} = "ProjEquiRect{$T}"

typealias_def(::Type{F}) where {B,M<:ProjEquiRect,T,A,F<:EquiRectField{B,M,T,A}} = "EquiRect$(typealias(B)){$(typealias(A))}"

# Proj 
# ================================================

@memoize function ProjEquiRect(θ, φ, θ∂, φ∂, θspan, φspan, ::Type{T}, storage) where {T}
    Ny, Nx = length(θ), length(φ)
    Ω  = rem2pi(φ∂[2] .- φ∂[1], RoundDown) .* diff(.- cos.(θ∂))
    ProjEquiRect{T}(Ny, Nx, θspan, φspan, θ, φ, θ∂, φ∂, Ω, storage)
end

function ProjEquiRect(; T=Float32, storage=Array, kwargs...)

    arg_error() = error("Constructor takes either (θ, φ, θ∂, φ∂) or (Ny, Nx, θspan, φspan) keyword arguments.")
    
    if all(haskey.(Ref(kwargs), (:θ, :φ, :θ∂, :φ∂)))
        !any(haskey.(Ref(kwargs), (:Ny, :Nx, :θspan, :φspan))) || arg_error()
        @unpack (θ, φ, θ∂, φ∂) = kwargs
    elseif all(haskey.(Ref(kwargs), (:Ny, :Nx, :θspan, :φspan)))
        !all(haskey.(Ref(kwargs), (:θ, :φ, :θ∂, :φ∂))) || arg_error()
        @unpack (Ny, Nx, θspan, φspan) = kwargs
        θ, θ∂ = @ondemand(CirculantCov.θ_grid)(; θspan, N=Ny, type=:equiθ)
        φ, φ∂ = @ondemand(CirculantCov.φ_grid)(; φspan, N=Nx)
    else
        arg_error()
    end

    ProjEquiRect(θ, φ, θ∂, φ∂, θspan, φspan, real_type(T), storage)

end



# Field Basis
# ================================================
"""
Jperm(ℓ::Int, n::Int) return the column number in the J matrix U^2
where U is unitary FFT. The J matrix looks like this:

|1   0|
|  / 1|
| / / |
|0 1  |

"""
function Jperm end

function Jperm(ℓ::Int, n::Int)
    @assert 1 <= ℓ <= n
    ℓ==1 ? 1 : n - ℓ + 2
end

# AzFourier <-> Map
function AzFourier(f::EquiRectMap)
    nφ = f.Nx
    EquiRectAzFourier(m_rfft(f.arr, 2) ./ √nφ, f.proj)
end

function Map(f::EquiRectAzFourier)
    nφ = f.Nx
    EquiRectMap(m_irfft(f.arr, nφ, 2) .* √nφ, f.proj)
end

# QUAzFourier <-> QUMap
function QUAzFourier(f::EquiRectQUMap)
    nθ, nφ = f.Ny, f.Nx
    qiumap = complex.(f.Qx, f.Ux) 
    Uf = m_fft(qiumap, 2) ./ √nφ
    f▫ = similar(Uf, 2nθ, nφ÷2+1)
    for ℓ = 1:nφ÷2+1
        if (ℓ==1) | ((ℓ==nφ÷2+1) & iseven(nφ))
            f▫[1:nθ, ℓ]     .= Uf[:,ℓ]
            f▫[nθ+1:2nθ, ℓ] .= conj.(Uf[:,ℓ])
        else
            f▫[1:nθ, ℓ]     .= Uf[:,ℓ]
            f▫[nθ+1:2nθ, ℓ] .= conj.(Uf[:,Jperm(ℓ,nφ)])
        end
    end
    EquiRectQUAzFourier(f▫, f.proj)
end

function QUMap(f::EquiRectQUAzFourier)
    nθₓ2, nφ½₊1 = size(f.arr)
    nθ, nφ = f.Ny, f.Nx
    @assert nφ½₊1 == nφ÷2+1
    @assert 2nθ   == nθₓ2

    pθk = similar(f.arr, nθ, nφ)
    for ℓ = 1:nφ½₊1
        if (ℓ==1) | ((ℓ==nφ½₊1) & iseven(nφ))
            pθk[:,ℓ] .= f.arr[1:nθ,ℓ]
        else
            pθk[:,ℓ]  .= f.arr[1:nθ,ℓ]
            pθk[:,Jperm(ℓ,nφ)] .= conj.(f.arr[nθ+1:2nθ,ℓ])
        end
    end
    qiumap = m_ifft(pθk, 2) .* √nφ
    EquiRectQUMap(cat(real(qiumap), imag(qiumap), dims=3), f.proj)
end


function Base.getindex(f::EquiRectS0, k::Symbol)
    @match k begin
        :Ix => Map(f).arr
        :Il => AzFourier(f).arr
        _ => error("Invalid EquiRectS0 index $k")
    end
end
function Base.getindex(f::EquiRectS2, k::Symbol)
    @match k begin
        :Px => (qu=QUMap(f); complex.(qu.Qx,qu.Ux))
        :Pl => QUAzFourier(f).arr
        _ => error("Invalid EquiRectS2 index $k")
    end
end

function Base.summary(io::IO, f::EquiRectField)
    @unpack Ny,Nx,Nbatch = f
    print(io, "$(length(f))-element $Ny×$Nx$(Nbatch==1 ? "" : "(×$Nbatch)")-pixel ")
    Base.showarg(io, f, true)
end

# block-diagonal operator
# ================================================

# NOTE: these block diag operators only work in AzBasis (at the moment) 
# due to the requirement that f.arr needs to be a matrix in the code below.

AzBasis = Union{QUAzFourier, AzFourier}

# ## Constructors

function BlockDiagEquiRect{B}(block_matrix::A, proj::P) where {B<:AzBasis, P<:ProjEquiRect, T, A<:AbstractArray{T}}
    BlockDiagEquiRect{B,P,T,A}(block_matrix, proj)
end

# The following allows construction by a vector of blocks

function BlockDiagEquiRect{B}(vector_of_blocks::Vector{A}, proj::P) where {B<:AzBasis, P<:ProjEquiRect, T, A<:AbstractMatrix{T}}
    block_matrix = Array{T}(undef, size(vector_of_blocks[1])..., length(vector_of_blocks))
    for b in eachindex(vector_of_blocks)
        block_matrix[:,:,b] .= vector_of_blocks[b]
    end
    BlockDiagEquiRect{B}(block_matrix, proj)
end

# ## Linear Algebra: tullio accelerated (operator, field)

# M * f

LinearAlgebra.:*(M::BlockDiagEquiRect{B}, f::EquiRectField) where {B<:Basis} = M * B(f)

function LinearAlgebra.:*(M::BlockDiagEquiRect{B}, f::F) where {B<:AzBasis, F<:EquiRectField{B}}
    promote_metadata_strict(M.proj, f.proj) # ensure same projection
    F(@tullio(Bf[p,iₘ] := M.blocks[p,q,iₘ] * f.arr[q,iₘ]), f.proj)
end

# M' * f

LinearAlgebra.:*(M::Adjoint{T,<:BlockDiagEquiRect{B}}, f::EquiRectField) where {T, B<:Basis} = M * B(f)

function LinearAlgebra.:*(M::Adjoint{T,<:BlockDiagEquiRect{B}}, f::F) where {T, B<:AzBasis, F<:EquiRectField{B}}
    promote_metadata_strict(M.parent.proj, f.proj) # ensure same projection
    F(@tullio(Bf[p,iₘ] := conj(M.parent.blocks[q,p,iₘ]) * f.arr[q,iₘ]), f.proj)
end

# f' * M

function LinearAlgebra.:*(f::Adjoint{T,<:EquiRectField}, M::BlockDiagEquiRect{B}) where {T, B<:AzBasis}
    (M' * f.parent)'
end

# ## Linear Algebra: tullio accelerated (operator, operator)

# M₁ * M₂

function LinearAlgebra.:*(M₁::BlockDiagEquiRect{B}, M₂::BlockDiagEquiRect{B}) where {B<:AzBasis}
    promote_metadata_strict(M₁.proj, M₂.proj) # ensure same projection
    BlockDiagEquiRect{B}(@tullio(M₃[p,q,iₘ] := M₁.blocks[p,j,iₘ] * M₂.blocks[j,q,iₘ]), M₁.proj)
end

# M₁' * M₂

function LinearAlgebra.:*(M₁::Adjoint{T,<:BlockDiagEquiRect{B}}, M₂::BlockDiagEquiRect{B}) where {T, B<:AzBasis}
    promote_metadata_strict(M₁.parent.proj, M₂.proj) # ensure same projection
    BlockDiagEquiRect{B}(@tullio(M₃[p,q,iₘ] := conj(M₁.parent.blocks[j,p,iₘ]) * M₂.blocks[j,q,iₘ]), M₁.parent.proj)
end

# M₁ * M₂'

function LinearAlgebra.:*(M₁::BlockDiagEquiRect{B}, M₂::Adjoint{T,<:BlockDiagEquiRect{B}}) where {T, B<:AzBasis}
    promote_metadata_strict(M₁.proj, M₂.parent.proj) # ensure same projection
    BlockDiagEquiRect{B}(@tullio(M₃[p,q,iₘ] := M₁.blocks[p,j,iₘ] * conj(M₂.parent.blocks[q,j,iₘ])), M₁.proj)
end

# M₁ + M₂, M₁ - M₂, M₁ \ M₂, M₁ / M₂ ... also with mixed adjoints
# QUESTION: some of these may be sped up with @tullio

for op in (:+, :-, :/, :\)

    quote 

        function LinearAlgebra.$op(M₁::BlockDiagEquiRect{B}, M₂::BlockDiagEquiRect{B}) where {B<:AzBasis}
            promote_metadata_strict(M₁.proj, M₂.proj) # ensure same projection
            BlockDiagEquiRect{B}(
                map( $op, eachslice(M₁.blocks;dims=3), eachslice(M₂.blocks;dims=3) ),
                M₁.proj,
            )
        end

        function LinearAlgebra.$op(M₁::Adjoint{T,<:BlockDiagEquiRect{B}}, M₂::BlockDiagEquiRect{B}) where {T, B<:AzBasis}
            promote_metadata_strict(M₁.parent.proj, M₂.proj) # ensure same projection
            BlockDiagEquiRect{B}(
                map( (m1,m2)->$op(m1',m2), eachslice(M₁.parent.blocks;dims=3), eachslice(M₂.blocks;dims=3) ),
                M₁.proj
            )
        end

        function LinearAlgebra.$op(M₁::BlockDiagEquiRect{B}, M₂::Adjoint{T,<:BlockDiagEquiRect{B}}) where {T, B<:AzBasis}
            promote_metadata_strict(M₁.proj, M₂.parent.proj) # ensure same projection
            BlockDiagEquiRect{B}(
                map( (m1,m2)->$op(m1,m2'), eachslice(M₁.blocks;dims=3), eachslice(M₂.parent.blocks;dims=3) ),
                M₁.proj,
            )
        end

    end |> eval 

end

# ## Linear Algebra: with arguments (operator, )

# - M₁,  inv(M₁) and sqrt(M₁)
# REMARK: use mapblocks if you want more specific dispatch

for op in (:-, :sqrt, :inv, :pinv)

    quote
        function LinearAlgebra.$op(M₁::BlockDiagEquiRect{B}) where {B<:AzBasis}
            BlockDiagEquiRect{B}(
                mapslices($op, M₁.blocks, dims = [1,2]), 
                M₁.proj
            )
        end
    end |> eval

end

# logdet and logabsdet

function LinearAlgebra.logdet(M₁::BlockDiagEquiRect{B}) where {B<:AzBasis} 
    sum(logdet, eachslice(M₁.blocks; dims=3))
end

function LinearAlgebra.logabsdet(M₁::BlockDiagEquiRect{B}) where {B<:AzBasis} 
    sum(x->logabsdet(x)[1], eachslice(M₁.blocks; dims=3))
end

# dot products (including f' * g)

LinearAlgebra.dot(a::EquiRectField, b::EquiRectField) = dot(Ł(a).arr, Ł(b).arr)

function Base.:*(f::Adjoint{T,EquiRectField{A}}, g::EquiRectField{B}) where {T, A<:AzBasis, B<:AzBasis}
    dot(f,g)
end



# mapblocks 
# =====================================

function mapblocks(fun::Function, M::BlockDiagEquiRect{B}, f::EquiRectField) where {B<:AzBasis} 
    mapblocks(fun, M, B(f))
end

function mapblocks(fun::Function, M::BlockDiagEquiRect{B}, f::F) where {B<:AzBasis, F<:EquiRectField{B}}
    promote_metadata_strict(M.proj, f.proj) # ensure same projection
    Mfarr = similar(f.arr)
    y_    = eachcol(Mfarr)
    x_    = eachcol(f.arr)
    Mb_   = eachslice(M.blocks; dims = 3) 
    for (y, x, Mb) in zip(y_, x_, Mb_)
        y .= fun(Mb, x)
    end
    F(Mfarr, f.proj)
end 

function mapblocks(fun::Function, Ms::BlockDiagEquiRect{B}...) where {B<:AzBasis}
    map(M->promote_metadata_strict(M.proj, Ms[1].proj), Ms) 
    BlockDiagEquiRect{B}(
        map(
            i->fun(getindex.(getproperty.(Ms,:blocks),:,:,i)...), # This looks miserable:(
            axes(Ms[1].blocks,3),
        ),
        Ms[1].proj,
    )
end 

# Other methods
# ========================================= 

# ## simulation

global_rng_for(::Type{<:BlockDiagEquiRect}) = Random.default_rng()

function white_noise(::Type{T}, pj::ProjEquiRect, rng::AbstractRNG) where {T<:Real}
    EquiRectMap(randn(T, pj.Ny, pj.Nx), pj)
end

function white_noise(::Type{T}, pj::ProjEquiRect, rng::AbstractRNG) where {T<:Complex}
    qiu = randn(T, pj.Ny, pj.Nx)
    EquiRectQUMap(real(qiu), imag(qiu), pj)
end

white_noise(ξ::EquiRectField{Map, P},         rng::AbstractRNG) where {T, P<:ProjEquiRect{T}} = white_noise(T, ξ.proj, rng)
white_noise(ξ::EquiRectField{QUMap, P},       rng::AbstractRNG) where {T, P<:ProjEquiRect{T}} = white_noise(Complex{real(T)}, ξ.proj, rng)
white_noise(ξ::EquiRectField{AzFourier, P},   rng::AbstractRNG) where {T, P<:ProjEquiRect{T}} = AzFourier(white_noise(T, ξ.proj, rng))
white_noise(ξ::EquiRectField{QUAzFourier, P}, rng::AbstractRNG) where {T, P<:ProjEquiRect{T}} = QUAzFourier(white_noise(Complex{real(T)}, ξ.proj, rng))

function simulate(rng::AbstractRNG, M::BlockDiagEquiRect{AzFourier,ProjEquiRect{T}}) where {T}
    spin0_whitepix_fld = white_noise(real(T), M.proj, rng) 
    mapblocks(M, spin0_whitepix_fld) do Mb, vb 
        sqrt(Hermitian(Mb)) * vb
    end
end

function simulate(rng::AbstractRNG, M::BlockDiagEquiRect{QUAzFourier,ProjEquiRect{T}}) where {T}
    spin2_whitepix_fld = white_noise(Complex{real(T)}, M.proj, rng) 
    mapblocks(M, spin2_whitepix_fld) do Mb, vb 
        sqrt(Hermitian(Mb)) * vb
    end
end

# adapt_structure

function adapt_structure(storage, L::BlockDiagEquiRect{B}) where {B}
    BlockDiagEquiRect{B}(adapt(storage, L.blocks), adapt(storage, L.proj))
end

function Base.size(L::BlockDiagEquiRect{<:AzBasis})  
    n,m,p = size(L.blocks)
    @assert n==m
    sz = n*p
    return (sz, sz)
end

# covariance operators
# ================================================

function Cℓ_to_Cov(::Val{:I}, proj::ProjEquiRect{T}, CI::InterpolatedCℓs; units=1, ℓmax=10_000, progress=true) where {T}
    
    @unpack θ, φ, Ω = proj
    nθ, nφ  = length(θ), length(φ)
    ℓ       = 0:ℓmax

    CIℓ = CI(ℓ)
    CIℓ[.!isfinite.(CIℓ)] .= 0

    @assert real(T) == T
    I▫  = zeros(T,nθ,nθ,nφ÷2+1)
    # TODO: do we want ngrid as an optional argmuent to Cℓ_to_Cov?
    Γ_I  = @ondemand(CirculantCov.Γθ₁θ₂φ₁φ⃗_Iso)(ℓ, CIℓ; ngrid=50_000)
    # using full resolution ComplexF64 for internal construction
    ptmW    = FFTW.plan_fft(Vector{ComplexF64}(undef, nφ)) 

    pbar = Progress(nθ, progress ? 1 : Inf, "Cℓ_to_Cov: ")
    for k = 1:nθ
        for j = 1:nθ
            Iγⱼₖℓ⃗ = @ondemand(CirculantCov.γθ₁θ₂ℓ⃗)(θ[j], θ[k], φ, Γ_I,  ptmW)
            for ℓ = 1:nφ÷2+1
                I▫[j,k, ℓ] = real(Iγⱼₖℓ⃗[ℓ])
            end
        end
        next!(pbar)
    end

    return BlockDiagEquiRect{AzFourier}(I▫, proj)
    
end


function Cℓ_to_Cov(::Val{:P}, proj::ProjEquiRect{T}, CEE::InterpolatedCℓs, CBB::InterpolatedCℓs; units=1, ℓmax=10_000, progress=true) where {T}
    
    @unpack θ, φ, Ω = proj
    nθ, nφ  = length(θ), length(φ)
    ℓ       = 0:ℓmax

    CBBℓ = CBB(ℓ)
    CEEℓ = CEE(ℓ)
    CBBℓ[.!isfinite.(CBBℓ)] .= 0
    CEEℓ[.!isfinite.(CEEℓ)] .= 0

    @assert real(T) == T
    EB▫   = zeros(Complex{T},2nθ,2nθ,nφ÷2+1)
    # TODO: do we want ngrid as an optional argmuent to Cℓ_to_Cov?
    ΓC_EB  = @ondemand(CirculantCov.ΓCθ₁θ₂φ₁φ⃗_CMBpol)(ℓ, CEEℓ, CBBℓ; ngrid=50_000)    
    # using full resolution ComplexF64 for internal construction
    ptmW    = FFTW.plan_fft(Vector{ComplexF64}(undef, nφ)) 
    
    pbar = Progress(nθ, progress ? 1 : Inf, "Cℓ_to_Cov: ")
    for k = 1:nθ
        for j = 1:nθ
            EBγⱼₖℓ⃗, EBξⱼₖℓ⃗ = @ondemand(CirculantCov.γθ₁θ₂ℓ⃗_ξθ₁θ₂ℓ⃗)(θ[j], θ[k], φ, ΓC_EB..., ptmW)
            for ℓ = 1:nφ÷2+1
                Jℓ = Jperm(ℓ, nφ)
                EB▫[j,   k   , ℓ] = EBγⱼₖℓ⃗[ℓ]
                EB▫[j,   k+nθ, ℓ] = EBξⱼₖℓ⃗[ℓ]
                EB▫[j+nθ,k   , ℓ] = conj(EBξⱼₖℓ⃗[Jℓ])
                EB▫[j+nθ,k+nθ, ℓ] = conj(EBγⱼₖℓ⃗[Jℓ])
            end
        end
        next!(pbar)
    end

    return BlockDiagEquiRect{QUAzFourier}(EB▫, proj)

end


# promotion
# ================================================

promote_basis_generic_rule(::Map, ::AzFourier) = Map()

promote_basis_generic_rule(::QUMap, ::QUAzFourier) = QUMap()

# used in broadcasting to decide the resulting metadata when
# broadcasting over two fields
function promote_metadata_strict(metadata₁::ProjEquiRect{T₁}, metadata₂::ProjEquiRect{T₂}) where {T₁,T₂}

    if (
        metadata₁.Ny    === metadata₂.Ny    &&
        metadata₁.Nx    === metadata₂.Nx    &&
        metadata₁.θspan === metadata₂.θspan &&   
        metadata₁.φspan === metadata₂.φspan   
    )
        
        # always returning the "wider" metadata even if T₁==T₂ helps
        # inference and is optimized away anyway
        promote_type(T₁,T₂) == T₁ ? metadata₁ : metadata₂
        
    else
        error("""Can't broadcast two fields with the following differing metadata:
        1: $(select(fields(metadata₁),(:Ny,:Nx,:θspan,:φspan)))
        2: $(select(fields(metadata₂),(:Ny,:Nx,:θspan,:φspan)))
        """)
    end

end


# used in non-broadcasted algebra to decide the resulting metadata
# when performing some operation across two fields. this is free to do
# more generic promotion than promote_metadata_strict (although this
# is currently not used, but in the future could include promoting
# resolution, etc...). the result should be a common metadata which we
# can convert both fields to then do a succesful broadcast
promote_metadata_generic(metadata₁::ProjEquiRect, metadata₂::ProjEquiRect) = 
    promote_metadata_strict(metadata₁, metadata₂)


### preprocessing
# defines how ImplicitFields and BatchedReals behave when broadcasted
# with ProjEquiRect fields. these can return arrays, but can also
# return `Broadcasted` objects which are spliced into the final
# broadcast, thus avoiding allocating any temporary arrays.

function preprocess((_,proj)::Tuple{<:Any,<:ProjEquiRect}, r::Real)
    r isa BatchedReal ? adapt(proj.storage, reshape(r.vals, 1, 1, 1, :)) : r
end
# need custom adjoint here bc Δ can come back batched from the
# backward pass even though r was not batched on the forward pass
@adjoint function preprocess(m::Tuple{<:Any,<:ProjEquiRect}, r::Real)
    preprocess(m, r), Δ -> (nothing, Δ isa AbstractArray ? batch(real.(Δ[:])) : Δ)
end



### adapting

# dont adapt the fields in proj, instead re-call into the memoized
# ProjLambert so we always get back the singleton ProjEquiRect object
# for the given set of parameters (helps reduce memory usage and
# speed-up subsequent broadcasts which would otherwise not hit the
# "===" branch of the "promote_*" methods)
function adapt_structure(storage, proj::ProjEquiRect{T}) where {T}
    # TODO: make sure these are consistent with any arguments that
    # were added to the memoized constructor
    @unpack Ny, Nx, θspan, φspan = proj
    T′ = eltype(storage)
    ProjEquiRect(;Ny, Nx, T=(T′==Any ? T : real(T′)), θspan, φspan, storage)
end
adapt_structure(::Nothing, proj::ProjEquiRect{T}) where {T} = proj


### etc...
# TODO: see proj_lambert.jl and adapt the things there for EquiRect
# maps, or even better, figure out what can be factored out into
# generic code that works for both Lambert and EquiRect
