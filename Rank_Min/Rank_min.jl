include("../ProximalAlgorithms.jl/src/ProximalAlgorithms.jl")
using LinearAlgebra
using ProximalOperators
include("Hkl.jl")

#RUN ONE OF THE FOLLOWING
#Dryer System
include("SISO.jl")
b = b[2:end-1]
L = Hkls.Hkl()
β_f = 0.70
ϵ = 0*b.+0.40;
τ=1.0; σ=τ;

A = L*b;
θ = 2.; μ = 1.0; iter = Nothing;
β_h=1.0;

include("quad_perturb.jl")
g = IndBox(-ϵ,ϵ);
g_ext = Translate(quad_perturb(g,β_f),-b);
h = NuclearNorm(β_h);

x₀ = b;
y₀ = prox(ProximalAlgorithms.convex_conjugate(h),(1+σ)*L*b,σ)[1];

solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-8), maxit=10000);
(x_afba, y_afba), it, iter = solver(x0 = x₀, y0 = y₀, g=g_ext, h=h, L=L, gamma=(τ,σ), xout=true);

using Plots
p = plot((iter.g_list+iter.h_list)[1:200], label="Total Cost",legendfont=12, tickfont=12, guidefont=12, lw=4, legend=:topright, alpha=0.7)
plot!(iter.g_list[1:200],label="Cost g(x)", lw=4, alpha=0.7)
plot!(iter.h_list[1:200],label="Cost h(Lx)", lw=4, alpha=0.7)
xlabel!("Iteration"); ylabel!("Cost(x,Lx)", lw=4, alpha=0.7)
savefig(p,"RM_SISO1_Cost_Evolution.svg")

#4th order system with 0.1 Noise #IN MATLAB
include("SISO_2.jl")
L = Hkls.Hkl()
β_f = 1.2
ϵ = 0*b.+0.3;
τ = 0.4; σ = 0.4;

A = L*b;
θ = 2.; μ = 1.0; iter = Nothing;
β_h=1.0;

f = Translate(SqrNormL2(β_f),-b);
g = IndBox(-ϵ,ϵ);
g_mod = Translate(g,-b);
h = NuclearNorm(β_h);

x₀ = b;
y₀ = prox(ProximalAlgorithms.convex_conjugate(h),(1+σ)*L*b,σ)[1];

solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-8), maxit=10000);
(x_afba, y_afba), it, iter = solver(x0 = x₀, y0 = y₀, f=f, beta_f=β_f, g=g_mod, h=h, L=L, gamma=(τ,σ), xout=true);

rank(A,1e-3)
rank(L*x_afba,1e-3)

maximum(abs.(x_afba-b))

using Plots
p = plot((iter.f_list+iter.h_list)[1:200],label="Total Cost",legendfont=12, tickfont=12, guidefont=12, lw=4, alpha=0.7)
plot!(iter.f_list[1:200],label="Cost f(x)", lw=4, alpha=0.7)
plot!(iter.g_list[1:200],label="Cost g(x)", lw=4, alpha=0.7)
plot!(iter.h_list[1:200],label="Cost h(Lx)", lw=4, alpha=0.7)
xlabel!("Iteration"); ylabel!("Cost(x,Lx)", lw=4)
savefig(p,"RM_SISO2_Cost_Evolution.svg")

#6th order system with 2 inputs and 2 outputs
include("MIMO_2.jl");
b = reshape(hcat(b1,b2),(size(b1)[1], size(b1)[2], 2));
b = b[2:54,:,:];
L = Hkls.Hkl(2,2)
β_f = 0.79;
ϵ = 0*b.+0.3;

# If the matrix is PSD, the minimization of the trace can be used
A = L*b;
θ = 2.; μ = 1.0; iter = Nothing;
β_h=1.0; τ = 0.4; σ = 0.4;

f = Translate(SqrNormL2(β_f),-b);
g = IndBox(-ϵ,ϵ);
g_mod = Translate(g,-b);
h = NuclearNorm(β_h);

x₀ = b;
y₀ = prox(ProximalAlgorithms.convex_conjugate(h),(1+σ)*L*b,σ)[1];

solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-8), maxit=10000);
(x_afba, y_afba), it, iter = solver(x0 = x₀, y0 = y₀, f=f, beta_f=β_f, g=g_mod, h=h, L=L, gamma=(τ,σ), xout=true);

#Plot Results
using Plots
p = plot((iter.f_list+iter.h_list)[1:200].-350,label="Total Cost",legendfont=12, tickfont=12, guidefont=12, lw=4, alpha=0.7)
plot!(iter.f_list[1:200],label="Cost f(x)", lw=4, alpha=0.7)
plot!(iter.g_list[1:200],label="Cost g(x)", lw=4, alpha=0.7)
plot!(iter.h_list[1:200].-350,label="Cost h(Lx)", lw=4, alpha=0.7)
xlabel!("Iteration"); ylabel!("Cost(x,Lx)", lw=4)
savefig(p,"RM_MIMO_Cost_Evolution.svg")