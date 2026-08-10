# CMBLensing.jl (foreground-modelling fork)

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://cosmicmar.com/CMBLensing.jl/stable) [![Gitter](https://badges.gitter.im/CMBLensing-jl/community.svg)](https://gitter.im/CMBLensing-jl/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

[![](https://img.shields.io/badge/source-github-blue)](https://github.com/marius311/CMBLensing.jl)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/marius311/CMBLensing.jl/gh-pages?urlpath=lab) 
[![](https://github.com/marius311/CMBLensing.jl/workflows/runtests/badge.svg)](https://github.com/marius311/CMBLensing.jl/actions?query=workflow%3Aruntests+branch%3Amaster) [![](https://github.com/marius311/CMBLensing.jl/workflows/docbuild/badge.svg)](https://github.com/marius311/CMBLensing.jl/actions?query=workflow%3Adocbuild+branch%3Amaster) 

> **Note:** This is a *modified fork* of [CMBLensing.jl](https://github.com/marius311/CMBLensing.jl)
> by Marius Millea. It adds foreground modelling (`FGroundDataSet`) used in the work of
> Doohan et al., [arXiv:2502.20801](https://arxiv.org/abs/2502.20801) — *"On Biases from
> Correlated, Non-Gaussian Foregrounds in Optimal Lensing Inference"* (M. Doohan PhD thesis,
> University of Melbourne, 2026). All upstream functionality is unchanged; see
> [**Modifications in this fork**](#modifications-in-this-fork) below. The badges above point
> at the upstream repository and its CI.


CMBLensing.jl is a next-generation tool for analysis of the lensed Cosmic Microwave Background. It is written in [Julia](https://julialang.org/) and transparently callable from Python.


At its heart, CMBLensing.jl maximizes, samples, or performs MUSE inference on the Bayesian posterior for the CMB lensing problem. It also contains tools to quickly manipulate and process CMB maps, set up modified posteriors with a probabilistic programming language, and take gradients using automatic differentiation.

### Highlights
* Fully Nvidia GPU compatible (1-2 orders of magnitude speedups over CPU, depending on the problem size and hardware).
* Automatic differentiation (via [Zygote.jl](https://fluxml.ai/Zygote.jl/)) provides for-free gradients of your custom posteriors.
* Includes the following algorithms to lense a map:
    * `LenseFlow` ([Millea, Anderes, & Wandelt 2017](https://arxiv.org/abs/1708.06753))
    * `Taylens` ([Næss & Louis 2013](https://arxiv.org/abs/1307.0719))
    * Taylor series expansion to any order
    * Bilinear interpolation
* Maximize and sample $\mathcal{P}(f,\phi,\theta\,|\,d)$, the joint maximum a posteriori estimate of the lensing potential, $\phi$, the  temperature and/or polarization fields, $f$, and cosmological parameters, $\theta$ ([Millea, Anderes, & Wandelt 2017](https://arxiv.org/abs/1708.06753), [Millea, Anderes, & Wandelt 2020](https://arxiv.org/abs/2002.00965))
* Maximize $\mathcal{P}(\phi\,|\,d,\theta)$, i.e. the marginal maximum a posteriori estimate of the lensing potential, $\phi$, at fixed cosmological parameters, $\theta$ ([Carron & Lewis 2017](https://arxiv.org/abs/1704.08230))
* Compute MUSE inferences of bandpowers of $\phi$ and unlensed $f$ via [MuseInference.jl](https://github.com/marius311/MuseInference.jl) ([Millea & Seljak, 2021](https://arxiv.org/abs/2112.09354)).
* Do basic quadratic estimation of $\phi$ ([Hu & Okamoto 2003](https://arxiv.org/abs/astro-ph/0111606))

## Modifications in this fork

At small angular scales the CMB *temperature* field carries the most lensing signal-to-noise, but
it is contaminated by extragalactic foregrounds (notably the tSZ effect and the CIB) which are
both non-Gaussian and *correlated* with the lensing signal. Ignoring them biases the recovered
lensing potential and temperature power spectra. This fork extends CMBLensing.jl so that the
foreground is modelled as an additional field in the posterior, and its amplitude is inferred
*jointly* with the CMB field $f$, the lensing potential $\phi$, and cosmological parameters
$\theta$ — rather than being neglected or subtracted ahead of time. See Doohan et al.,
[arXiv:2502.20801](https://arxiv.org/abs/2502.20801).

### `FGroundDataSet`

The new `FGroundDataSet` (defined in [`src/dataset.jl`](src/dataset.jl)) extends the standard
`BaseDataSet` with an additive foreground field `g`, drawn from a Gaussian with covariance `Cg`.
The foreground is added to the *lensed* CMB $\tilde f = L(\phi)\,f$ before the beam `B` and mask
`M` are applied, so the generative (forward) model becomes:

```julia
f ~ MvNormal(0, Cf(θ))            # unlensed CMB
ϕ ~ MvNormal(0, Cϕ(θ))            # lensing potential
g ~ MvNormal(0, Cg(θ))            # foreground field  (new)
f̃  = L(ϕ) * f                    # lensed CMB
μ  = M(θ) * (B(θ) * (f̃ + g))     # foreground added before beam/mask  (was M·B·f̃)
d ~ MvNormal(μ, Cn(θ))           # data
```

Compared to `BaseDataSet`, `FGroundDataSet` carries two extra fields: `Cg` (the foreground
covariance operator) and `Ng` (an initial foreground-noise estimate used by the Hessian
preconditioner). Specializations of `logprior`, `Hessian_logpdf_preconditioner`, and the MUSE
`sample_x_z` machinery are provided so that the existing `MAP_joint` and
[MuseInference.jl](https://github.com/marius311/MuseInference.jl) pipelines work with the
foreground field and infer its amplitude(s).

### Foreground covariance parameterizations

`load_fground_ds` builds an `FGroundDataSet` together with a matching simulation. The foreground
spectrum `Cg` can be parameterized in three ways, selected by the `Cℓ_fg` and `ℓedges_g` keywords:

1. **Power law** *(default: `Cℓ_fg = nothing`, `ℓedges_g = nothing`)* — a power-law spectrum
   $C_\ell \propto A_g\,(\ell/\ell_\mathrm{pivot})^{\alpha_g}$ with inferable amplitude `Ag` and
   tilt `αg` about the pivot `ℓpivot_fg`.
2. **Scaled template** *(supply `Cℓ_fg`, leave `ℓedges_g = nothing`)* — a fixed template spectrum
   `Cℓ_fg` (a `Cℓs`) with a single inferable amplitude: `Cg = Ag * Cℓ_fg`.
3. **Bandpowers** *(supply both `Cℓ_fg` and `ℓedges_g`)* — `Cg` becomes a bandpower-dependent
   operator with one amplitude `Ag` per $\ell$-bin, analogous to the `Aϕ`/`AT` bandpower
   treatment of `Cϕ`/`Cf`.

### Usage

```julia
using CMBLensing

# Build a foreground-contaminated dataset + simulation
(; ds, f, g, ϕ, proj) = load_fground_ds(
    θpix  = 3,
    Nside = 256,
    pol   = :I,
    # foreground: power-law amplitude Ag and tilt αg about ℓpivot_fg (option 1),
    # or pass Cℓ_fg (± ℓedges_g) for options 2/3
    Ag = 1, αg = 0, ℓpivot_fg = 1500,
)

# `ds` is an FGroundDataSet and plugs into the usual joint-MAP / MUSE machinery, e.g.
# MAP_joint(ds)  or  a CMBLensingMuseProblem built from `ds`
```

`load_nofground_ds` builds the matched *foreground-free* dataset (same configuration, no `g`
field), which is convenient for quantifying the bias introduced by the foregrounds.

Everything else — installation, the upstream API, and the example notebooks — is documented via
the upstream links below and is unchanged in this fork.

## Documentation

The best place to get started is to read the [documentation](https://cosmicmar.com/CMBLensing.jl/) (which is a work-in-progress, but contains many useful examples). 

Most of the pages in the documentation are Jupyter notebooks, and you can click the "launch binder" link at the top of each page to launch a Jupyterlab server running the notebook in your browser (courtesy of [binder](https://mybinder.org/)). 

You can also clone the repository and open the notebooks in [docs/src](https://github.com/marius311/CMBLensing.jl/tree/master/docs/src) if you want to run them locally (which will usually lead to higher performance). The notebooks are stored as `.md` files rather than `.ipynb` format. Its recommended to install [Jupytext](jupytext) (`pip install jupytext`) and then you can run these `.md` directly from Jupyterlab by right-clicking on them and selecting `Open With -> Notebook`. Otherwise, run the script `docs/make_notebooks.sh` to convert the `.md` files to `.ipynb` which you can then open as desired. 


## Installation

### Requirements

* Julia 1.7+
* _(recommended)_ An Nvidia GPU and [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) for GPU support
* _(recommended)_ Python 3 + matplotlib (used for plotting)
* _(recommended)_ [pycamb](https://github.com/cmbant/CAMB) to generate $C_\ell$'s (run `pip install --user camb`)
* _(recommended)_ [JuliaMono](https://github.com/cormullion/juliamono/releases) font to ensure characters like `f̃, ϕ, ∇, ℓ`, etc... are rendered correctly

### Native installation

To install the Julia package locally, run:

```juliapkg
pkg> add CMBLensing
```

(type `]` at the Julia REPL to reach the `pkg>` prompt)

> **This fork:** the `add CMBLensing` command above installs the *upstream* package, which does
> **not** include `FGroundDataSet`. To use this fork, clone this repository and develop it
> locally instead, e.g. `pkg> dev /path/to/CMBLensingFGround` (or
> `pkg> add https://github.com/mdoohan/CMBLensingFGround.git`).

*(recommended)* After installing, switch your Julia to use Intel MKL FFT libraries, which provide significantly faster FFTs when running on CPU. You can do so by running `using CMBLensing.FFTW; FFTW.set_provider!("mkl")` from the same environment in which you added CMBLensing. This only needs to be done once per-environment (see also [here](https://github.com/JuliaMath/FFTW.jl#mkl)).


### Docker installation

Also provided is a Docker container which includes a Jupyterlab server and all the recommended and optional dependencies to run and use `CMBLensing.jl`. Launch this container with:

```sh
git clone https://github.com/marius311/CMBLensing.jl.git
cd CMBLensing.jl
docker-compose pull main
docker-compose up main
```

The first time you run this, it will automatically download the (~1Gb) container from the Docker hub. The command will prompt you with the URL which you should open in a browser to access the notebook.

To run the notebook on a different port than the default `8888`, do `PORT=1234 docker-compose up main` where `1234` is whatever port number you want.

You can also build the container locally by replacing `docker-compose pull main` with `docker-compose build main`.


## Citation & attribution

This fork is built on top of [CMBLensing.jl](https://github.com/marius311/CMBLensing.jl) by
Marius Millea and contributors — please cite the upstream package and the relevant lensing
method papers linked above.

The foreground-modelling additions in this fork (`FGroundDataSet` and related machinery) were
developed for and are described in:

> M. Doohan et al., *"On Biases from Correlated, Non-Gaussian Foregrounds in Optimal Lensing
> Inference"*, [arXiv:2502.20801](https://arxiv.org/abs/2502.20801) (M. Doohan PhD thesis,
> University of Melbourne, 2026).

If you use these additions, please cite that work in addition to upstream CMBLensing.jl.
