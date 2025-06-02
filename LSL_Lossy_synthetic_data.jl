#=
SYNTHETIC DATA.

Finite difference scheme applied to the system to obtain synthetic data.

INPUT: x - discrete spatial grid,
       h - grid spacing,
       p - potential function
       λ := -ω² < 0 from the Fourier transform of ∂ₜₜ
OUTPUT: v(x,λ)::Vector{Float64} - numerical solution of the 1-dimensional frequency domain wave equation w/ potential function. 
=#

using LinearAlgebra
using SparseArrays
using Infiltrator
using CairoMakie
CairoMakie.activate!()


function forward_solver(x,h,α,p,λ)
   ω  = sqrt(-λ)
   c2 = -(exp(im*ω))/(im*ω*( (1-α)*exp(im*ω) - (1+α)*exp(-im*ω) ))
   c1 = (1 + im*ω*(1 - α)*c2) / (im*ω*(1+α))
   
   exact_solution(x,λ) = c1*exp.(im*ω*x) + c2*exp.(-im*ω*x);
       
   C                        = spdiagm(-1 => 1/h^2 * ones(length(x)-1), 0 => 2/h^2 * ones(length(x)), 1 => 1/h^2 * ones(length(x)-1)) + spdiagm(p); 
   C[1,2]                   = 2/h^2;
   C[length(x),length(x)-1] = 2/h^2;

   D      = spdiagm(zeros(length(x)));
   D[1,1] = α*(2/h);

   A = -C + im*ω*D - λ*I;

   f     = zeros(length(x),1);
   f[1]  = 2/h;
        
   # Solve the linear system Av = f
   v = A\f;
   v = vec(v)
   
   function benchmark_plotting(x,λ,v)
    fig = Figure(;
        figure_padding=(5,5,10,10),
        backgroundcolor=:snow2,
        size=(600,400),
        )
    ax = Axis(fig[1,1];
        xlabel="x",
        ylabel="y",
        title="Title",
        )
      
    lines!(ax, x, (abs.(exact_solution(x,λ))).^2;
        color=:black,
        label="Exact sol."
        )
    lines!(ax, x, (abs.(v)).^2; 
         color=:red,
         linestyle=:dash,
         label="Approx.")

     #axislegend("legend"; position=:rt)    
    save("benchmark_plot_test.png",fig)
end

   benchmark_plotting(x,λ,v)
   
   return v
end