include("../ProximalAlgorithms.jl/src/ProximalAlgorithms.jl")
using LinearAlgebra
using ProximalOperators
using ProximalCore
include("Hkl.jl")

include("MIMO_2.jl");
b = reshape(hcat(b1,b2),(size(b1)[1], size(b1)[2], 2));
b = b[2:54,:,:];
L = Hkls.Hkl(2,2)
β_f = 0.9;
ϵ = 0*b.+0.3;

A = L*b;
θ = 2.; μ = 1.0; iter = Nothing;
τ = 0.3; σ = τ; β_h=1.0; β_g = β_h;

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
θ = 1.; μ = 0.0; iter = Nothing;
solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-32), maxit=2000);
(x_afba, y_afba), it, iter = solver(x0 = x₀, y0 = y₀, f=f, beta_f=β_f, g=g_mod, h=h, L=L, gamma=(τ,σ),xout=true);
r2_cost_AFBA_1_0 = (iter.f_list+iter.h_list);
rp2_AFBA_1_0 = iter.p_list;
rd2_AFBA_1_0 = iter.d_list;
rpd2_AFBA_1_0 = max.(rp2_AFBA_1_0,rd2_AFBA_1_0);
print("MinCost (",argmin(iter.f_list+iter.h_list),"),: ",minimum(iter.f_list+iter.h_list),"\n")

#Linesearches
τ = 0.3; σ = τ;
y₀ = prox(ProximalAlgorithms.convex_conjugate(h),(1+σ)*L*b,σ)[1];
solver = ProximalAlgorithms.AAFBA(theta = θ, mu = μ, tol = Float64(1e-14), maxit=2000);
(x_afba, y_afba), it, iter = solver(x0 = x₀, y0 = y₀, f=f, beta_f = β_f, g=g_mod, h=h, L=L, gamma=τ, t=sqrt(σ/τ), xout=true);
r2_cost_LAAFBA = (iter.f_list+iter.h_list);
rp2_LAAFBA = iter.p_list;
rd2_LAAFBA = iter.d_list;
rpd2_LAAFBA = max.(rp2_LAAFBA,rd2_LAAFBA);
print("MinCost (",argmin(iter.f_list+iter.h_list),"),: ",minimum(iter.f_list+iter.h_list),"\n")
LAAFBA_counter_r2 = iter.g_counter;
LAAFBA_step_r2 = iter.step_list;

#Linesearch Dual
include("MIMO_2.jl");
b = reshape(hcat(b1,b2),(size(b1)[1], size(b1)[2], 2));
b = b[2:54,:,:];
L = -Hkls.UnHkl(2,2)
τ = 0.3; σ = 1*τ;

l = ProximalCore.convex_conjugate(Translate(SqrNormL2(β_f),-b));
ϵ = 0*b.+0.3;
h_conv = ProximalOperators.Conjugate(IndBox(b.-ϵ,b.+ϵ));
g_conv = ProximalOperators.Conjugate(NuclearNorm(β_g));

solver = ProximalAlgorithms.GPDAL(theta = θ, mu = μ, tol = Float64(1e-14), maxit=2000);
(x_afba, y_afba), it, iter = solver(x0 = y₀, y0 = x₀, l=l, beta_l=β_f, g=g_conv, h=h_conv, L=L, gamma = (σ,τ), xout=true, dual=true);
r2_cost_GPDAL = (iter.f_list+iter.h_list);
rp2_GPDAL = iter.p_list;
rd2_GPDAL = iter.d_list;
rpd2_GPDAL = max.(rp2_GPDAL,rd2_GPDAL);
GPDAL_counter_r2 = iter.g_counter;
print("MinCost (",argmin(iter.f_list+iter.h_list),"),: ",minimum(iter.f_list+iter.h_list),"\n")

solver = ProximalAlgorithms.EGRPDA(theta = θ, mu = μ, tol = Float64(1e-14), maxit=2000);
(x_afba, y_afba), it, iter = solver(x0 = y₀, y0 = x₀, l=l, beta_l=β_f, g=g_conv, h=h_conv, L=L, gamma = (σ,τ), xout=true, dual=true, ψ=1.55);
r2_cost_EEGRPDA = (iter.f_list+iter.h_list);
rp2_EEGRPDA = iter.p_list;
rd2_EEGRPDA = iter.d_list;
rpd2_EEGRPDA = max.(rp2_EEGRPDA, rd2_EEGRPDA);
EEGRPDA_counter_r2 = iter.g_counter;
print("MinCost (",argmin(iter.f_list+iter.h_list),"),: ",minimum(iter.f_list+iter.h_list),"\n")

optcost_r2 = minimum(r2_cost_EEGRPDA)-1e-3

using Plots
using LaTeXStrings
p = plot((r2_cost_AFBA_1_0[1:1000].-optcost_r2)./optcost_r2,label="AFBA", lw=4, yaxis=:log, legend=:bottomright, legendfont = 12, tickfont = 12, guidefont = 12, alpha=0.7)
plot!((r2_cost_GPDAL[GPDAL_counter_r2[1:1000]].-optcost_r2)./optcost_r2,label="GPDAL", lw=4, yaxis=:log, alpha=0.7)
plot!((r2_cost_EEGRPDA[EEGRPDA_counter_r2[1:1000]].-optcost_r2)./optcost_r2,label="EGRPDA-L", lw=4, yaxis=:log, alpha=0.7)
plot!((r2_cost_LAAFBA[LAAFBA_counter_r2[1:1000]].-optcost_r2)./optcost_r2,label="AAFBA-L", lw=4, yaxis=:log, alpha=0.7)
xlabel!("Gradient Evaluations"); ylabel!("Cost (x,Lx)", lw=4)
savefig(p,"RM_Cost_Comparison_2.svg")

p = plot((rpd2_AFBA_1_0[1:1000]),label="AFBA", lw=4, yaxis=:log, legend=:topright, legendfont = 12, tickfont = 12, guidefont = 12, alpha=0.7)
plot!((rpd2_GPDAL[GPDAL_counter_r2[1:1000]]),label="GPDAL", lw=4, yaxis=:log, alpha=0.7)
plot!((rpd2_EEGRPDA[EEGRPDA_counter_r2[1:1000]]),label="EGRPDA-L", lw=4, yaxis=:log, alpha=0.7)
plot!((rpd2_LAAFBA[LAAFBA_counter_r2[1:1000]]),label="AAFBA-L", lw=4, yaxis=:log, alpha=0.7)
xlabel!("Gradient Evaluations"); ylabel!("Primal-Dual Gap", lw=4, alpha=0.7)
savefig(p,"RM_PD_Comparison_2.svg")