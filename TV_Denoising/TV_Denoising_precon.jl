## Using the 'ROF' model for total variation based image denoising.
include("../ProximalAlgorithms.jl/src/ProximalAlgorithms.jl")
using JLD2
using ProximalOperators
using Noise, Images, Colors, FileIO, ImageFiltering, TestImages

# Image Treatment
@load "img_noisy.tif" img

convert(Array{RGB{Float64}}, img);
img = Float64.(channelview(img)[:,:,:]);
size(img)[1] == 3 ? (c, n, m) = size(img) : ((n, m) = size(img); c=1);
new_img = similar(img);

# Construct the Linear Transformation for TV
include("concatLMs.jl")
include("TV_2D.jl")
include("indBallL2_row.jl")
K = ConcatLMs.ConcatLM(TV_2D(n,m)); #

#Comparison
it = 0; iter = nothing; x_afba=nothing; y_afba=nothing;
λ = 0.8; τ = 0.1; σ = 1.0/sqrt(8*τ);
ρ=τ/σ;
b = vec(img)
g = Translate(NormL1(Float64(1)),-b);
h = NormL2_row(λ);    hᶜ = IndBallL2_row(λ);
y₀= prox(hᶜ,K*b+σ*K*b,σ)[1];

#Original AFBA
θ = 2.; μ = 1.;
solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-8), maxit=1100);
(x_afba, y_afba), it, iter = solver(x0 = b, y0 = y₀, g = g, h=h, L=K, gamma = (τ,σ), xout=true);
t_cost_AFBA_2_0 = (iter.g_list+iter.h_list);
tp_AFBA_2_0 = iter.p_list;
td_AFBA_2_0 = iter.d_list;
tpd_AFBA_2_0 = tp_AFBA_2_0 + td_AFBA_2_0;

optcost_tv_pre = minimum(t_cost_AFBA_2_0);
t_optstep = t_cost_AFBA_2_0;

#Step Size 1
it = 0; iter = nothing; x_afba=nothing; y_afba=nothing;
λ = 0.8; τ = 0.3; σ = 1/(8*τ);
ρ=τ/σ;
b = vec(img)
g = Translate(NormL1(Float64(1)),-b);
h = NormL2_row(λ);    hᶜ = IndBallL2_row(λ);
y₀= prox(hᶜ,K*b+σ*K*b,σ)[1];

#Original AFBA
θ = 2.; μ = 1.;
solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-8), maxit=1100);
(x_afba, y_afba), it, iter = solver(x0 = b, y0 = y₀, g = g, h=h, L=K, gamma = (τ,σ), xout=true);
t_cost_AFBA_2_0 = (iter.g_list+iter.h_list);
tp_AFBA_2_0 = iter.p_list;
td_AFBA_2_0 = iter.d_list;
tpd_AFBA_2_0 = max.(tp_AFBA_2_0,td_AFBA_2_0);

θ = 1.; μ = 1.;
solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-8), maxit=1100);
(x_afba, y_afba), it, iter = solver(x0 = b, y0 = y₀, g = g, h=h, L=K, gamma = (τ,σ), xout=true);
t_cost_AFBA_1_1 = (iter.g_list+iter.h_list);
tp_AFBA_1_1 = iter.p_list;
td_AFBA_1_1 = iter.d_list;
tpd_AFBA_1_1 = max.(tp_AFBA_1_1,td_AFBA_1_1);

#Preconditioning
K1, K2 = TV_2D(n,m);
K_compl = K1*im+K2;
K_compl_adj = copy(K_compl');

τ_j = zeros(size(K_compl)[2])
for j in 1:size(K_compl)[2]
    τ_j[j] = min(1,1/sum(abs.(K_compl[:,j])));
end
σ_i = zeros(size(K_compl)[1])
for i in 1:size(K_compl)[1]
    σ_i[i] = min(1,1/sum(abs.(K_compl_adj[:,i])));
end

θ = 2.; μ = 1.;
λ = 0.8; τ = τ_j.*ρ; σ = σ_i./ρ;#1/(8τ);
y₀= prox(hᶜ,K*b+σ*K*b,σ)[1];
solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-8), maxit=1100);
(x_afba, y_afba), it, iter = solver(x0 = b, y0 = y₀, g = g, h=h, L=K, gamma = (τ,σ), xout=true);
t_cost_Prec_2_0 = (iter.g_list+iter.h_list);
tp_Prec_2_0 = iter.p_list;
td_Prec_2_0 = iter.d_list;
tpd_Prec_2_0 = max.(tp_Prec_2_0,td_Prec_2_0);


θ = 1.; μ = 1.;
solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-8), maxit=1100);
(x_afba, y_afba), it, iter = solver(x0 = b, y0 = y₀, g = g, h=h, L=K, gamma = (τ,σ), xout=true);
t_cost_Prec_1_1 = (iter.g_list+iter.h_list);
tp_Prec_1_1 = iter.p_list;
td_Prec_1_1 = iter.d_list;
tpd_Prec_1_1 = max.(tp_Prec_1_1,td_Prec_1_1);

using Plots
using LaTeXStrings
p = plot((t_cost_AFBA_2_0[1:1000].-optcost_tv_pre)./optcost_tv_pre,label="AFBA (2,0)", lw=4, yaxis=:log, legend=:topright, legendfont = 12, tickfont = 12, guidefont=12, alpha=0.7)
plot!((t_cost_AFBA_1_1[1:1000].-optcost_tv_pre)./optcost_tv_pre,label="AFBA (1,1)", lw=4,yaxis=:log, alpha=0.7)
plot!((t_cost_Prec_2_0[1:1000].-optcost_tv_pre)./optcost_tv_pre,label="Prec (2,0)", lw=4, yaxis=:log, alpha=0.7)
plot!((t_cost_Prec_1_1[1:1000].-optcost_tv_pre)./optcost_tv_pre,label="Prec (1,1)", lw=4,  yaxis=:log, alpha=0.7)
xlabel!("Matrix Multiplication Evaluations"); ylabel!("Cost (x,∇x)", lw=4, alpha=0.7)
savefig(p,"TV_precon_Cost_Comparison_1.svg")

using Plots
p = plot((tpd_AFBA_2_0[1:1000]),label="AFBA (2,0)", lw=4, yaxis=:log, legend=:topright, legendfont = 12, tickfont = 12, guidefont=12, alpha=0.7)
plot!((tpd_AFBA_1_1[1:1000]),label="AFBA (1,1)", lw=4,yaxis=:log, legend=:topright, legendfont = 12, tickfont = 12, guidefont=12, alpha=0.7)
plot!((tpd_Prec_2_0[1:1000]),label="Prec (2,0)", lw=4, yaxis=:log, alpha=0.7)
plot!((tpd_Prec_1_1[1:1000]),label="Prec (1,1)", lw=4,  yaxis=:log, alpha=0.7)
xlabel!("Matrix Multiplication Evaluations"); ylabel!("Primal Dual Gap", lw=4, alpha=0.7)
savefig(p,"TV_precon_PD_comparison_1.svg")

#Step Size 2
it = 0; iter = nothing; x_afba=nothing; y_afba=nothing;
λ = 0.8; τ = 0.25; σ = 1/(8*τ);
ρ=τ/σ;
b = vec(img)
g = Translate(NormL1(Float64(1)),-b);
h = NormL2_row(λ);    hᶜ = IndBallL2_row(λ);
y₀= prox(hᶜ,K*b+σ*K*b,σ)[1];

#Original AFBA
θ = 2.; μ = 1.;
solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-8), maxit=1100);
(x_afba, y_afba), it, iter = solver(x0 = b, y0 = y₀, g = g, h=h, L=K, gamma = (τ,σ), xout=true);
t_cost_AFBA_2_0 = (iter.g_list+iter.h_list);
tp_AFBA_2_0 = iter.p_list;
td_AFBA_2_0 = iter.d_list;
tpd_AFBA_2_0 = max.(tp_AFBA_2_0,td_AFBA_2_0);

θ = 1.; μ = 1.;
solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-8), maxit=1100);
(x_afba, y_afba), it, iter = solver(x0 = b, y0 = y₀, g = g, h=h, L=K, gamma = (τ,σ), xout=true);
t_cost_AFBA_1_1 = (iter.g_list+iter.h_list);
tp_AFBA_1_1 = iter.p_list;
td_AFBA_1_1 = iter.d_list;
tpd_AFBA_1_1 = max.(tp_AFBA_1_1,td_AFBA_1_1);

#Preconditioning
K1, K2 = TV_2D(n,m);
K_compl = K1*im+K2;
K_compl_adj = copy(K_compl');

τ_j = zeros(size(K_compl)[2])
for j in 1:size(K_compl)[2]
    τ_j[j] = min(1,1/sum(abs.(K_compl[:,j])));
end
σ_i = zeros(size(K_compl)[1])
for i in 1:size(K_compl)[1]
    σ_i[i] = min(1,1/sum(abs.(K_compl_adj[:,i])));
end

θ = 2.; μ = 1.;
λ = 0.8; τ = τ_j.*ρ; σ = σ_i./ρ;#1/(8τ);
y₀= prox(hᶜ,K*b+σ*K*b,σ)[1];
solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-8), maxit=1100);
(x_afba, y_afba), it, iter = solver(x0 = b, y0 = y₀, g = g, h=h, L=K, gamma = (τ,σ), xout=true);
t_cost_Prec_2_0 = (iter.g_list+iter.h_list);
tp_Prec_2_0 = iter.p_list;
td_Prec_2_0 = iter.d_list;
tpd_Prec_2_0 = max.(tp_Prec_2_0,td_Prec_2_0);

θ = 1.; μ = 1.;
solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-8), maxit=1100);
(x_afba, y_afba), it, iter = solver(x0 = b, y0 = y₀, g = g, h=h, L=K, gamma = (τ,σ), xout=true);
t_cost_Prec_1_1 = (iter.g_list+iter.h_list);
tp_Prec_1_1 = iter.p_list;
td_Prec_1_1 = iter.d_list;
tpd_Prec_1_1 = max.(tp_Prec_1_1,td_Prec_1_1);

p = plot((t_cost_AFBA_2_0[1:1000].-optcost_tv_pre)./optcost_tv_pre,label="AFBA (2,0)", lw=4, yaxis=:log, legend=:topright, legendfont = 12, tickfont = 12, guidefont=12, alpha=0.7)
plot!((t_cost_AFBA_1_1[1:1000].-optcost_tv_pre)./optcost_tv_pre,label="AFBA (1,1)", lw=4,yaxis=:log, alpha=0.7)
plot!((t_cost_Prec_2_0[1:1000].-optcost_tv_pre)./optcost_tv_pre,label="Prec (2,0)", lw=4, yaxis=:log, alpha=0.7)
plot!((t_cost_Prec_1_1[1:1000].-optcost_tv_pre)./optcost_tv_pre,label="Prec (1,1)", lw=4,  yaxis=:log, alpha=0.7)
xlabel!("Matrix Multiplication Evaluations"); ylabel!("Cost (x,∇x)", lw=4, alpha=0.7)
savefig(p,"TV_precon_Cost_Comparison_2.svg")

using Plots
p = plot((tpd_AFBA_2_0[1:1000]),label="AFBA (2,0)", lw=4, yaxis=:log, legend=:topright, legendfont = 12, tickfont = 12, guidefont=12, alpha=0.7)
plot!((tpd_AFBA_1_1[1:1000]),label="AFBA (1,1)", lw=4,yaxis=:log, legend=:topright, legendfont = 12, tickfont = 12, guidefont=12, alpha=0.7)
plot!((tpd_Prec_2_0[1:1000]),label="Prec (2,0)", lw=4, yaxis=:log, alpha=0.7)
plot!((tpd_Prec_1_1[1:1000]),label="Prec (1,1)", lw=4,  yaxis=:log, alpha=0.7)
xlabel!("Matrix Multiplication Evaluations"); ylabel!("Primal Dual Gap", lw=4, alpha=0.7)
savefig(p,"TV_precon_PD_comparison_2.svg")