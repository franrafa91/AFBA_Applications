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
τ = 0.3; σ = τ; β_f = 1.1; β_h=1.0;

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
solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-30), maxit=2500);
(x_afba, y_afba), it, iter = solver(x0 = x₀, y0 = y₀, f=f, beta_f=β_f, g=g_mod, h=h, L=L, gamma=(τ,σ),xout=true);
r1_cost_AFBA_2_0 = (iter.f_list+iter.h_list);
rp1_AFBA_2_0 = iter.p_list;
rd1_AFBA_2_0 = iter.d_list;
rpd1_AFBA_2_0 = max.(rp1_AFBA_2_0,rd1_AFBA_2_0);
print("MinCost (",argmin(iter.f_list+iter.h_list),"),: ",minimum(iter.f_list+iter.h_list),"\n")

θ = 1.; μ = 1.0; iter = Nothing;
solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-32), maxit=2000);
(x_afba, y_afba), it, iter = solver(x0 = x₀, y0 = y₀, f=f, beta_f=β_f, g=g_mod, h=h, L=L, gamma=(τ,σ),xout=true);
r1_cost_AFBA_1_1 = (iter.f_list+iter.h_list);
rp1_AFBA_1_1 = iter.p_list;
rd1_AFBA_1_1 = iter.d_list;
rpd1_AFBA_1_1 = max.(rp1_AFBA_1_1,rd1_AFBA_1_1);
print("MinCost (",argmin(iter.f_list+iter.h_list),"),: ",minimum(iter.f_list+iter.h_list),"\n")

#Preconditioning
τ_j = zeros(size(b)[1])
for j in 1:size(b)[1]
    τ_j[j] = 1/sqrt((size(b)[1]+1)/2-abs((size(b)[1]+1)/2-j))
end
τ = τ_j; σ = 2*sum(τ_j/(size(τ_j)[1]));
y₀ = prox(ProximalAlgorithms.convex_conjugate(h),(1+σ)*L*b,σ)[1];

θ = 2.; μ = 1.0; iter = Nothing;
solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-30), maxit=2000);
(x_afba, y_afba), it, iter = solver(x0 = x₀, y0 = y₀, f=f, beta_f=β_f, g=g_mod, h=h, L=L, gamma=(τ,σ),xout=true);
r1_cost_Prec_2_0 = (iter.f_list+iter.h_list);
rp1_Prec_2_0 = iter.p_list;
rd1_Prec_2_0 = iter.d_list;
rpd1_Prec_2_0 = max.(rp1_Prec_2_0,rd1_Prec_2_0);
print("MinCost (",argmin(iter.f_list+iter.h_list),"),: ",minimum(iter.f_list+iter.h_list),"\n")

θ = 1.; μ = 1.0; iter = Nothing;
solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-30), maxit=2000);
(x_afba, y_afba), it, iter = solver(x0 = x₀, y0 = y₀, f=f, beta_f=β_f, g=g_mod, h=h, L=L, gamma=(τ,σ),xout=true);
r1_cost_Prec_1_1 = (iter.f_list+iter.h_list);
rp1_Prec_1_1 = iter.p_list;
rd1_Prec_1_1 = iter.d_list;
rpd1_Prec_1_1 = max.(rp1_Prec_1_1,rd1_Prec_1_1);
print("MinCost (",argmin(iter.f_list+iter.h_list),"),: ",minimum(iter.f_list+iter.h_list),"\n")

#Linesearchs
τ = 0.3; σ = 1*τ;
y₀ = prox(ProximalAlgorithms.convex_conjugate(h),(1+σ)*L*b,σ)[1];
solver = ProximalAlgorithms.AAFBA(theta = θ, mu = μ, tol = Float64(1e-30), maxit=2000);
(x_afba, y_afba), it, iter = solver(x0 = x₀, y0 = y₀, f=f, beta_f=β_f, g=g_mod, h=h, L=L, gamma=τ, t=sqrt(σ/τ), xout=true);
r1_cost_AAFBA = (iter.f_list+iter.g_list+iter.h_list);
rp1_AAFBA = iter.p_list;
rd1_AAFBA = iter.d_list;
rpd1_AAFBA = max.(rp1_AAFBA,rd1_AAFBA);
AAFBA_counter_r1 = iter.g_counter;
print("MinCost (",argmin(iter.f_list+iter.h_list),"),: ",minimum(iter.f_list+iter.h_list),"\n")

#Linesearch Dual
include("MIMO.jl");
b = reshape(hcat(b1,b2,b3),(size(b1)[1], size(b1)[2], 3));
b = b[1:51,:,:];
L = -Hkls.UnHkl(3,2)
τ = 0.3; σ = 1*τ;

l = ProximalCore.convex_conjugate(Translate(SqrNormL2(β_f),-b));
ϵ = 0*b.+0.3;
h_conv = ProximalOperators.Conjugate(IndBox(b.-ϵ,b.+ϵ));
g_conv = ProximalOperators.Conjugate(NuclearNorm(β_h));

optcost_r1 = minimum(r1_cost_AAFBA)

using Plots
using LaTeXStrings
p = plot((r1_cost_AFBA_2_0[1:2000].-optcost_r1)./optcost_r1,label="AFBA (2,0)", lw=4, yaxis=:log, legend=:topright, legendfont = 12, tickfont = 12, guidefont = 12, alpha=0.7)
plot!((r1_cost_AFBA_1_1[1:2000].-optcost_r1)./optcost_r1,label="AFBA (1,1)", lw=4,yaxis=:log, legend=:topright, legendfont = 12, tickfont = 12, guidefont = 12, alpha=0.7)
plot!((r1_cost_Prec_2_0[1:2000].-optcost_r1)./optcost_r1,label="Prec (2,0)", lw=4, yaxis=:log, alpha=0.7)
plot!((r1_cost_Prec_1_1[1:2000].-optcost_r1)./optcost_r1,label="Prec (1,1)", lw=4,  yaxis=:log, alpha=0.7)
xlabel!("Gradient Evaluations"); ylabel!("Cost (x,Lx)", lw=4)
savefig(p,"RM_precon_Cost_Comparison.svg")

p = plot((rpd1_AFBA_2_0[1:2000]),label="AFBA (2,0)", lw=4, yaxis=:log, legend=:topright, legendfont = 12, tickfont = 12, guidefont = 12, alpha=0.7)
plot!((rpd1_AFBA_1_1[1:2000]),label="AFBA (1,1)", lw=4,yaxis=:log, legend=:topright, legendfont = 12, tickfont = 12, guidefont = 12, alpha=0.7)
plot!((rpd1_Prec_2_0[1:2000]),label="Prec (2,0)", lw=4, yaxis=:log, alpha=0.7)
plot!((rpd1_Prec_1_1[1:2000]),label="Prec (1,1)", lw=4,  yaxis=:log, alpha=0.7)
xlabel!("Gradient Evaluations"); ylabel!("Primal-Dual Gap", lw=4)
savefig(p,"RM_precon_PD_Comparison.svg")