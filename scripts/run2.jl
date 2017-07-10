using JLD
using CMBLensing
using CMBLensing: @dictpack, jrk4, δlnΠᶠ_δfϕ, cache, LP, HP
using Optim
using PyCall
using Base.Iterators: repeated

@pyimport scipy.interpolate as itp

"""
Calculates geometric weights wℓ for the cooling covariance Ĉfℓ s.t.

    Ĉfℓ = Cfℓ^wℓ * Cf̃ℓ^(1-wℓ)

Arguments:
* f_obs : true f̃ delensed by current ϕ estimate
* f/f̃ : true un/lensed field
"""
function fit_wℓ(f_obs,f,f̃; ℓedges=[1:200:2000; 3000; 5000], s=1)
    (ℓ,Cℓb_obs),(ℓ,Cℓb),(ℓ,Cℓb̃) = (get_Cℓ(x,which=[:BB],ℓedges=ℓedges) for x in (f_obs,f,f̃))
    iℓ = @. !isnan(Cℓb_obs[:]) & (Cℓb_obs[:] > 0)
    ℓ = round.(Int,ℓ[iℓ])
    lnCℓb_obs, lnCℓb, lnCℓb̃ = @. (log(Cℓb_obs[:][iℓ]), log(Cℓb[:][iℓ]), log(Cℓb̃[:][iℓ]))
    z = @. (lnCℓb_obs-lnCℓb̃)/(lnCℓb-lnCℓb̃)
    iz = @. !isnan(z) & (z>0)
    itpz = itp.UnivariateSpline(log.(ℓ[iz]), log.(z[iz]), s=s)
    min.(1,exp.(itpz(log.(1:10000))))
end


"""
Iterative conditional algorithm
"""
function run2(;
    Θpix = 3,
    nside = 64,
    T = Float32,
    L = LenseFlow{jrk4{7}},
    outfile = nothing,
    seed = nothing,
    mask = nothing,
    r = 0.05,
    r_data = 0.05,
    Cℓf = class(lmax=8000,r=r),
    Cℓf_data = (r==r_data ? Cℓf : class(lmax=8000,r=r_data)),
    use = :TEB,
    ℓmax_data = 3000,
    ℓmax_masking_hack = 10000,
    μKarcminT = 1,
    beamFWHM = 1.5,
    ℓknee = 100,
    ws = 1:20,
    Ncg = 100,
    Ncg0 = 5000,
    cgtol = 1e-4)
    
    # Cℓs
    Cℓf̃ = Dict(k=>Cℓf[Symbol("ln_$k")] for (k,v) in Cℓf if Symbol("ln_$k") in keys(Cℓf))
    Cℓn = noisecls(μKarcminT, beamFWHM=beamFWHM, ℓknee=ℓknee)
    
    # types which depend on whether T/E/B
    SS,ks = Dict(:TEB=>((S0,S2),(:tt,:ee,:bb,:te)), :EB=>((S2,),(:ee,:bb)), :T=>((S0,),(:tt,)))[use]
    F,F̂,nF = Dict(:TEB=>(FlatIQUMap,FlatTEBFourier,3), :EB=>(FlatS2QUMap,FlatS2EBFourier,2), :T=>(FlatS0Map,FlatS0Fourier,1))[use]
    
    ## covariances
    P = Flat{Θpix,nside}
    Cϕ = Cℓ_to_cov(T,P,S0, Cℓf[:ℓ], Cℓf[:ϕϕ])
    Cf,Cf̃,Cn,Cf_data = (Cℓ_to_cov(T,P,SS..., Cℓf[:ℓ], (Cℓx[k] for k=ks)...) for Cℓx in (Cℓf,Cℓf̃,Cℓn,Cℓf_data))
    
    # field prior mask
    if iseven(nside)
        Ml = ones(Complex{T},nside÷2+1,nside)
        i = indexin([-FFTgrid(T,P).nyq],FFTgrid(T,P).k)[1]
        Ml[:,i] = Ml[i,:] = 0
        Mff = FullDiagOp(F̂{T,P}(repeated(Ml,nF)...))
    else
        Mff = 1
    end
    Mf = Squash * Mff * Squash
    
    # data mask
    Mdf = Mff*FullDiagOp(F̂{T,P}(repeated(Cℓ_2D(P,1:ℓmax_data,ones(ℓmax_data)),nF)...))
    if mask!=nothing
        Mdr = FullDiagOp(F{T,P}(repeated(T.(sptlike_mask(nside,Θpix; (mask==true?():mask)...)),nF)...))
    else
        Mdr = 1
    end
    Md = Squash * Mdr * Mdf * Squash
    # ϕ prior mask
    Mϕ = Squash
    
    ## simulate data
    seed!=nothing && srand(seed)
    f = simulate(Cf_data)
    ϕ = simulate(Cϕ)
    f̃ = L(ϕ)*f
    d = f̃ + simulate(Cn)

    target_lnP = mean(let n=simulate(Cn); -n⋅(Md'*(Cn\(Md*n)))/2 end for i=1:100)
    @show target_lnP
    rundat = @dictpack Θpix nside T r r_data μKarcminT d target_lnP Cℓf Cℓf_data Cℓn f f̃ ϕ

    trace = []
    ϕcur = 0ϕ
    local hist, fcur, f̃cur
    
    for (i,w) in enumerate(ws)
        
        if w==:auto
            wℓ = fit_wℓ(L(ϕcur)\f̃,f,f̃)[round.(Int,Cℓf[:ℓ])]
            w500 = wℓ[500]
            Cfw = Cℓ_to_cov(T,P,SS..., Cℓf[:ℓ], ((@. sign(Cℓf[k])*abs(Cℓf̃[k])^(1-wℓ)*abs(Cℓf[k])^wℓ) for k=ks)...);
        else
            w500 = w
            Cfw = Cf*w + Cf̃*(1-w)
        end
        
        ds = DataSet(d, Cn, Cfw, Cϕ, Md, Mf, Mϕ)
        
        let L = (w==0?IdentityOp:cache(L(ϕcur))),
            P = nan2zero.(sqrtm((nan2zero.(Mdf * Cn^-1) .+ nan2zero.(Mff * Cfw^-1)))^-1)
            A = L'*(Md'*(Cn^-1)*Md*L) + Mf'*Cfw^-1*Mf
            b = L'*(Md'*(Cn^-1)*Md*d)
            fcur,hist = pcg(P, A, b, i==1?0*b:(Squash*(P\fcur)), nsteps=(w==0?Ncg0:Ncg), tol=cgtol)
            f̃cur = L*fcur
        end

        local α,ϕnew,lnPw
        if i!=endof(ws)
            ϕnew = Mϕ*Cϕ*(δlnΠᶠ_δfϕ(LP(ℓmax_masking_hack)*fcur,ϕcur,ds) * δfϕ_δf̃ϕ(L(ϕcur),fcur,f̃cur))[2]
            α = (res = optimize(α->(-lnP(1,f̃cur,(1-α)*ϕcur+α*ϕnew,ds,L)), T(0), T(1), abs_tol=1e-6)).minimizer
            ϕcur = (1-α)*ϕcur+α*ϕnew
            lnPw = -res.minimum
        end
        
        lnP1 = lnP(1,f̃cur,ϕcur,DataSet(d, Cn, Cf, Cϕ, Md, Mf, Mϕ),L)
        
        if i!=endof(ws)
            push!(trace, @dictpack Cfw f̃cur fcur ϕcur ϕnew lnPw lnP1 α hist w)
            @printf("%i %.4f %.2f %.2f %i %.4f\n",i,w500,lnPw,lnP1,length(hist),α)
        else
            push!(trace, @dictpack Cfw f̃cur fcur ϕcur lnPw=>lnP1 lnP1 hist w)
            @printf("%i %.4f %.2f %i\n",i,w500,lnP1,length(hist))
        end
        if w==:auto; trace[end][:wℓ]=wℓ; end
        
        outfile!=nothing && save(outfile,"rundat",rundat,"trace",trace)
            
    end
    
    f̃cur, fcur, ϕcur, trace, rundat
    
end
