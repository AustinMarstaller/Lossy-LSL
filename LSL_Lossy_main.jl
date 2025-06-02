#=
MAIN FILE.

ROM algorithm performed here.
=#

using LinearAlgebra
using Infiltrator

include("./LSL_Lossy_synthetic_data.jl")
include("./lambda_construction.jl")
## PRELIMINARY 
const number_of_mu        = 10
const condNumber_fineness = 20
const max_lambda_density  = 6 # Legacy. Used for varying number_of_lambda in a for-loop.
const number_of_lambda    = 1
const α                   = 0.5;

λ,μ = lambda_construction(number_of_lambda, number_of_mu)
const wavelength    = 2π / sqrt(last(μ))
const h             = wavelength/20  # Grid point spacing is 20x smaller than wavelength  

x   = collect(0:h:1);       # Lattice in column vector. Legacy: it was a row vector   

const σ = 0.05
const γ = 25 #default 0.75
p           = γ*exp.(-(x.-0.2).^2 / σ^2); # p = 0 for reference problem
p_reference = zeros(length(x),1);

solutions_per_frequency = Matrix{Complex}(undef, length(x), length(λ))
for j = 1:length(λ)
    # For every λᵢ ∈ λ::Vector, generate the synthetic data via the Finite Difference scheme
    solutions_per_frequency[:,j] = forward_solver(x,h,α,p,λ[j])
end
## ROM 
