include("../ProximalAlgorithms.jl/src/ProximalAlgorithms.jl")
using LinearAlgebra
using ProximalOperators
using ProximalCore
include("Hkl.jl")

#2nd order system with 3 inputs and 2 outputs
include("MIMO.jl");
b = reshape(hcat(b1,b2,b3),(size(b1)[1], size(b1)[2], 3));
b = b[1:51,:,:];
L = Hkls.Hkl(3,2)

# If the matrix is PSD, the minimization of the trace can be used
A = L*b;
θ = 2.; μ = 1.0; iter = Nothing;
τ = 0.3; σ = 0.3; β_f = 1.1; β_h=1.0;

f = Translate(SqrNormL2(β_f),-b);
ϵ = 0*b.+0.30;
g = IndBox(-ϵ,ϵ);
g_mod = Translate(g,-b);
include("quad_perturb.jl")
g_ext = Translate(quad_perturb(g,β_f),-b)
h = NuclearNorm(β_h);

x₀ = b;
y₀ = prox(ProximalAlgorithms.convex_conjugate(h),(1+σ)*L*b,σ)[1];

#Original AFBA
solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-10), maxit=1000);
(x_afba, y_afba), it, iter = solver(x0 = x₀, y0 = y₀, f=f, beta_f=β_f, g=g_mod, h=h, L=L, gamma=(τ,σ),xout=true);
r1_cost_AFBA_2_0 = (iter.f_list+iter.g_list+iter.h_list);
rp1_AFBA_2_0 = iter.p_list;
rd1_AFBA_2_0 = iter.d_list;
rpd1_AFBA_2_0 = max.(rp1_AFBA_2_0,rd1_AFBA_2_0);
print("MinCost (",argmin(iter.f_list+iter.h_list),"),: ",minimum(iter.f_list+iter.h_list),"\n")

#Linesearchs
τ = 0.3; σ = 0.3;
y₀ = prox(ProximalAlgorithms.convex_conjugate(h),(1+σ)*L*b,σ)[1];
solver = ProximalAlgorithms.AAFBA(theta = θ, mu = μ, tol = Float64(1e-10), maxit=1000);
(x_afba, y_afba), it, iter = solver(x0 = x₀, y0 = y₀, f=f, beta_f=β_f, g=g_mod, h=h, L=L, gamma=τ, t=sqrt(σ/τ), xout=true);
r1_cost_LAAFBA = (iter.f_list+iter.g_list+iter.h_list);
rp1_LAAFBA = iter.p_list;
rd1_LAAFBA = iter.d_list;
rpd1_LAAFBA = max.(rp1_LAAFBA,rd1_LAAFBA);
LAAFBA_counter_r1 = iter.g_counter;
print("MinCost (",argmin(iter.f_list+iter.h_list),"),: ",minimum(iter.f_list+iter.h_list),"\n")

#Linesearch Dual
include("MIMO.jl");
b = reshape(hcat(b1,b2,b3),(size(b1)[1], size(b1)[2], 3));
b = b[1:51,:,:];
L = -Hkls.UnHkl(3,2)
τ = 0.3; σ = 0.3;

l = ProximalCore.convex_conjugate(Translate(SqrNormL2(β_f),-b));
ϵ = 0*b.+0.3;
h_conv = ProximalOperators.Conjugate(IndBox(b.-ϵ,b.+ϵ));
g_conv = ProximalOperators.Conjugate(NuclearNorm(β_h));

solver = ProximalAlgorithms.GPDAL(theta = θ, mu = μ, tol = Float64(1e-10), maxit=1000);
(x_afba, y_afba), it, iter = solver(x0 = y₀, y0 = x₀, l=l, beta_l=β_f, g=g_conv, h=h_conv, L=L, gamma = (σ,τ), xout=true, dual=true);
r1_cost_GPDAL = (iter.f_list+iter.g_list+iter.h_list);
rp1_GPDAL = iter.p_list;
rd1_GPDAL = iter.d_list;
rpd1_GPDAL = max.(rp1_GPDAL, rd1_GPDAL);
GPDAL_counter_r1 = iter.g_counter;
print("MinCost (",argmin(iter.f_list+iter.g_list+iter.h_list),"),: ",minimum(iter.f_list+iter.g_list+iter.h_list),"\n")

solver = ProximalAlgorithms.EGRPDA(theta = θ, mu = μ, tol = Float64(1e-10), maxit=1000);
(x_afba, y_afba), it, iter = solver(x0 = y₀, y0 = x₀, l=l, beta_l=β_f, g=g_conv, h=h_conv, L=L, gamma = (σ,τ), xout=true, dual=true, ψ=1.55);
r1_cost_EEGRPDA = (iter.f_list+iter.g_list+iter.h_list);
rp1_EEGRPDA = iter.p_list;
rd1_EEGRPDA = iter.d_list;
rpd1_EEGRPDA = max.(rp1_EEGRPDA, rd1_EEGRPDA);
EEGRPDA_counter_r1 = iter.g_counter;
print("MinCost (",argmin(iter.f_list+iter.g_list+iter.h_list),"),: ",minimum(iter.f_list+iter.h_list),"\n")

optcost_r1 = minimum(r1_cost_EEGRPDA) - 1e-3

using Plots
using LaTeXStrings
p = plot((r1_cost_AFBA_2_0[1:1000].-optcost_r1)./optcost_r1,label="AFBA", lw=4, yaxis=:log, legend=:bottomright, legendfont = 12, tickfont = 12, guidefont = 12, alpha=0.7)
plot!((r1_cost_GPDAL[GPDAL_counter_r1[1:1000]].-optcost_r1)./optcost_r1,label="GPDAL", lw=4, yaxis=:log, alpha=0.7)
plot!((r1_cost_EEGRPDA[EEGRPDA_counter_r1[1:1000]].-optcost_r1)./optcost_r1,label="EGRPDA-L", lw=4, yaxis=:log, alpha=0.7)
plot!((r1_cost_LAAFBA[LAAFBA_counter_r1[1:1000]].-optcost_r1)./optcost_r1,label="AAFBA-L", lw=4, yaxis=:log, alpha=0.7)
xlabel!("Gradient Evaluations"); ylabel!("Cost (x,Lx)", lw=4)
savefig(p,"RM_Cost_Comparison_1.svg")

using Plots
p = plot((rpd1_AFBA_2_0[1:1000]),label="AFBA", lw=4, yaxis=:log, legend=:topright, legendfont = 12, tickfont = 12, guidefont = 12, alpha=0.7)
plot!((rpd1_GPDAL[GPDAL_counter_r1[1:1000]]),label="GPDAL", lw=4, yaxis=:log, alpha=0.7)
plot!((rpd1_EEGRPDA[EEGRPDA_counter_r1[1:1000]]),label="EGRPDA-L", lw=4, yaxis=:log, alpha=0.7)
plot!((rpd1_LAAFBA[LAAFBA_counter_r1[1:1000]]),label="AAFBA-L", lw=4, yaxis=:log, alpha=0.7)
xlabel!("Gradient Evaluations"); ylabel!("Primal-Dual Gap", lw=4)
savefig(p,"RM_PD_Comparison_1.svg")
