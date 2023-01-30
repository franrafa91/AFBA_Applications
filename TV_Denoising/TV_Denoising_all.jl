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

# First Stepsize
it = 0; iter = nothing; x_afba=nothing; y_afba=nothing;
λ = 0.8; τ = 0.1; σ = 1/(8τ);
b = vec(img)
g = Translate(NormL1(Float64(1)),-b);
h = NormL2_row(λ);    hᶜ = IndBallL2_row(λ);
y₀= prox(hᶜ,K*b+σ*K*b,σ)[1];

θ = 1.; μ = 1.;
solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-8), maxit=2800);
(x_afba, y_afba), it, iter = solver(x0 = b, y0 = y₀, g = g, h=h, L=K, gamma = (τ,σ), xout=true);
t_cost_AFBA_1_1 = (iter.g_list+iter.h_list);
tp_AFBA_1_1 = iter.p_list;
td_AFBA_1_1 = iter.d_list;
tpd_AFBA_1_1 = max.(tp_AFBA_1_1,td_AFBA_1_1);

θ = 1.; μ = 1.;
λ = 0.8; τ = 0.1; σ = 1.0/(8*τ);
y₀= prox(hᶜ,K*b+σ*K*b,σ)[1];

δ = 0.99; r = 0.7; 
solver = ProximalAlgorithms.GPDAL(theta = θ, mu = μ, tol = Float64(1e-8), maxit=2000);
(x_afba, y_afba), it, iter = solver(x0 = b, y0 = y₀, g = g, h=h, L=K, gamma = (τ,σ),δ=δ, r=r, xout=true)
t_cost_GPDAL = (iter.g_list+iter.h_list);
GPDAL_counter = iter.g_counter;
tp_GPDAL = iter.p_list;
td_GPDAL = iter.d_list;
tpd_GPDAL = max.(tp_GPDAL,td_GPDAL);
t_step_GPDAL = iter.step_list;

solver = ProximalAlgorithms.EGRPDA(theta = θ, mu = μ, tol = Float64(1e-8), maxit=2000);
(x_afba, y_afba), it, iter = solver(x0 = b, y0 = y₀, g = g, h=h, L=K, gamma = (τ,σ),δ=δ, r=r, xout=true)
t_cost_EGRPDA = (iter.g_list+iter.h_list);
EGRPDA_counter = iter.g_counter;
tp_EGRPDA = iter.p_list;
td_EGRPDA = iter.d_list;
tpd_EGRPDA = max.(tp_EGRPDA, td_EGRPDA);
t_step_EGRPDA = iter.step_list;

solver = ProximalAlgorithms.AAFBA(theta = θ, mu = μ, tol = Float64(1e-8), maxit=2800);
(x_afba, y_afba), it, iter = solver(x0 = b, y0 = y₀, g = g, h=h, L=K, gamma = τ, t=sqrt(σ/τ), xout=true, opnorm=true, c=1.05)
t_cost_AAFBA = (iter.g_list+iter.h_list);
AAFBA_counter = iter.g_counter;
tp_AAFBA = iter.p_list;
td_AAFBA = iter.d_list;
tpd_AAFBA = max.(tp_AAFBA, td_AAFBA);
t_step_AAFBA = iter.step_list;

solver = ProximalAlgorithms.AAFBA(theta = θ, mu = μ, tol = Float64(1e-8), maxit=2000);
(x_afba, y_afba), it, iter = solver(x0 = b, y0 = y₀, g = g, h=h, L=K, gamma = τ, t=sqrt(σ/τ), xout=true, c=1.2)
t_cost_LAAFBA = (iter.g_list+iter.h_list);
LAAFBA_counter = iter.g_counter;
tp_LAAFBA = iter.p_list;
td_LAAFBA = iter.d_list;
tpd_LAAFBA = max.(tp_LAAFBA,td_LAAFBA);
t_step_LAAFBA = iter.step_list;
t_eta_LAAFBA = iter.eta_list;

len = length(t_cost_AFBA_1_1)
t_min_AFBA_1_1 = similar(t_cost_AFBA_1_1)
t_min_AAFBA = similar(t_cost_AAFBA)
t_min_GPDAL = similar(t_cost_GPDAL)
t_min_EGRPDA = similar(t_cost_EGRPDA)
t_min_LAAFBA = similar(t_cost_LAAFBA)
for i in 1:2801
    t_min_AFBA_1_1[i] = minimum(t_cost_AFBA_1_1[1:i])
    t_min_AAFBA[i] = minimum(t_cost_AAFBA[1:i])
end
for i in 1:2001
    t_min_LAAFBA[i] = minimum(t_cost_LAAFBA[1:i])
    t_min_GPDAL[i] = minimum(t_cost_GPDAL[1:i])
    t_min_EGRPDA[i] = minimum(t_cost_EGRPDA[1:i])
end

optcost = minimum(t_cost_EGRPDA)
minimum(t_cost_AAFBA)

using Plots
using LaTeXStrings
p = plot((t_cost_AFBA_1_1[1:2000].-optcost)./optcost,label="AFBA", lw=4, yaxis=:log, legend=:topright, legendfont = 12, tickfont = 12, guidefont = 12, alpha=0.7)
plot!((t_cost_AAFBA[AAFBA_counter[1:2000]].-optcost)./optcost,label="AAFBA", lw=4, yaxis=:log, alpha=0.7)
plot!((t_cost_GPDAL[GPDAL_counter[1:2000]].-optcost)./optcost,label="GPDAL", lw=4, yaxis=:log, alpha=0.7)
plot!((t_cost_EGRPDA[EGRPDA_counter[1:2000]].-optcost)./optcost,label="GRPDA-L", lw=4, yaxis=:log, alpha=0.7)
plot!((t_cost_GPDAL[LAAFBA_counter[1:2000]].-optcost)./optcost,label="AAFBA-L", lw=4, yaxis=:log, alpha=0.7)
xlabel!("Matrix Multiplication Evaluations"); ylabel!("Cost (x,∇x)", lw=4, alpha=0.7)
savefig(p,"TV_Cost_Comparison_1.svg")

tpd_min_AFBA_1_1 = similar(t_cost_AFBA_1_1)
tpd_min_AAFBA = similar(t_cost_AAFBA)
tpd_min_GPDAL = similar(t_cost_GPDAL)
tpd_min_EGRPDA = similar(t_cost_EGRPDA)
tpd_min_LAAFBA = similar(t_cost_LAAFBA)
for i in 1:2500
    tpd_min_AFBA_1_1[i] = minimum(tpd_AFBA_1_1[1:i])
    tpd_min_AAFBA[i] = minimum(tpd_AAFBA[1:i])
end
for i in 1:2000
    tpd_min_LAAFBA[i] = minimum(tpd_LAAFBA[1:i])
    tpd_min_GPDAL[i] = minimum(tpd_GPDAL[1:i])
    tpd_min_EGRPDA[i] = minimum(tpd_EGRPDA[1:i])
end

p = plot((tpd_AFBA_1_1[1:2000]),label="AFBA", lw=4, yaxis=:log, legend=:topright, legendfont = 12, tickfont = 12, guidefont = 12, alpha=0.7)
plot!((tpd_AAFBA[AAFBA_counter[1:2000]]),label="AAFBA", lw=4, yaxis=:log, alpha=0.7)
plot!((tpd_GPDAL[GPDAL_counter[1:2000]]),label="GPDAL", lw=4, yaxis=:log, alpha=0.7)
plot!((tpd_EGRPDA[EGRPDA_counter[1:2000]]),label="GRPDA-L", lw=4, yaxis=:log, alpha=0.7)
plot!((tpd_LAAFBA[LAAFBA_counter[1:2000]]),label="AAFBA-L", lw=4, yaxis=:log, alpha=0.7)
xlabel!("Matrix Multiplication Evaluations"); ylabel!("Primal Dual Gap", lw=4, alpha=0.7)
savefig(p,"TV_PD_Comparison_1.svg")


p = plot((tpd_min_AFBA_1_1[1:2000]),label="AFBA", lw=4, yaxis=:log, legend=:topright, legendfont = 12, tickfont = 12, guidefont = 12, alpha=0.7)
plot!((tpd_min_AAFBA[AAFBA_counter[1:2000]]),label="AAFBA", lw=4, yaxis=:log, alpha=0.7)
plot!((tpd_min_GPDAL[GPDAL_counter[1:2000]]),label="GPDAL", lw=4, yaxis=:log, alpha=0.7)
plot!((tpd_min_EGRPDA[EGRPDA_counter[1:2000]]),label="GRPDA-L", lw=4, yaxis=:log, alpha=0.7)
plot!((tpd_min_LAAFBA[LAAFBA_counter[1:2000]]),label="AAFBA-L", lw=4, yaxis=:log, alpha=0.7)
xlabel!("Matrix Multiplication Evaluations"); ylabel!("Primal Dual Gap", lw=4, alpha=0.7)
savefig(p,"TV_PD_Min_Comparison_1.svg")

p = plot([1:500], t_step_LAAFBA[1:500], label="AAFBA-L", lw=4, legendfont=12, tickfont=12, guidefont=12, alpha=0.7)
plot!([1:500], t_step_AAFBA[1:500], label="AAFBA", lw=4, legendfont=12, tickfont=12, guidefont=12, alpha=0.7)
xlabel!("Iteration"); ylabel!("Stepsize (τ)"); ylims!((0,0.35))
savefig(p,"TV_Stepsize_Comparison_1.svg")

p = plot(t_eta_LAAFBA[1:500], label="AAFBA-L", lw=4, legendfont=12, tickfont=12, guidefont=12, alpha=0.7)
xlabel!("Iteration"); ylabel!("Spectral Norm Estimate"); ylims!((0,3.3))
savefig(p,"TV_LAAFBA_Opnorm_1.svg")


# Second Stepsize
it = 0; iter = nothing; x_afba=nothing; y_afba=nothing;
λ = 0.7; τ = 0.3; σ = 0.3;
b = vec(img)
g = Translate(NormL1(Float64(1)),-b);
h = NormL2_row(λ);    hᶜ = IndBallL2_row(λ);
y₀= prox(hᶜ,K*b+σ*K*b,σ)[1];

θ = 1.; μ = 1.;
solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-8), maxit=2800);
(x_afba, y_afba), it, iter = solver(x0 = b, y0 = y₀, g = g, h=h, L=K, gamma = (τ,σ), xout=true);
t_cost_AFBA_1_1 = (iter.g_list+iter.h_list);
tp_AFBA_1_1 = iter.p_list;
td_AFBA_1_1 = iter.d_list;
tpd_AFBA_1_1 = max.(tp_AFBA_1_1,td_AFBA_1_1);

θ = 1.; μ = 1.;
λ = 0.7; τ = 0.3; σ = 0.3;
y₀= prox(hᶜ,K*b+σ*K*b,σ)[1];

δ = 0.99; r = 0.7; 
solver = ProximalAlgorithms.GPDAL(theta = θ, mu = μ, tol = Float64(1e-8), maxit=2000);
(x_afba, y_afba), it, iter = solver(x0 = b, y0 = y₀, g = g, h=h, L=K, gamma = (τ,σ),δ=δ, r=r, xout=true)
t_cost_GPDAL = (iter.g_list+iter.h_list);
GPDAL_counter = iter.g_counter;
tp_GPDAL = iter.p_list;
td_GPDAL = iter.d_list;
tpd_GPDAL = max.(tp_GPDAL,td_GPDAL);
t_step_GPDAL = iter.step_list;

solver = ProximalAlgorithms.EGRPDA(theta = θ, mu = μ, tol = Float64(1e-8), maxit=2000);
(x_afba, y_afba), it, iter = solver(x0 = b, y0 = y₀, g = g, h=h, L=K, gamma = (τ,σ),δ=δ, r=r, xout=true)
t_cost_EGRPDA = (iter.g_list+iter.h_list);
EGRPDA_counter = iter.g_counter;
tp_EGRPDA = iter.p_list;
td_EGRPDA = iter.d_list;
tpd_EGRPDA = max.(tp_EGRPDA, td_EGRPDA);
t_step_EGRPDA = iter.step_list;

solver = ProximalAlgorithms.AAFBA(theta = θ, mu = μ, tol = Float64(1e-8), maxit=2800);
(x_afba, y_afba), it, iter = solver(x0 = b, y0 = y₀, g = g, h=h, L=K, gamma = τ, t=sqrt(σ/τ), xout=true, opnorm=true, c=1.05)
t_cost_AAFBA = (iter.g_list+iter.h_list);
AAFBA_counter = iter.g_counter;
tp_AAFBA = iter.p_list;
td_AAFBA = iter.d_list;
tpd_AAFBA = max.(tp_AAFBA, td_AAFBA);
t_step_AAFBA = iter.step_list;

solver = ProximalAlgorithms.AAFBA(theta = θ, mu = μ, tol = Float64(1e-8), maxit=2000);
(x_afba, y_afba), it, iter = solver(x0 = b, y0 = y₀, g = g, h=h, L=K, gamma = τ, t=sqrt(σ/τ), xout=true, c=1.2)
t_cost_LAAFBA = (iter.g_list+iter.h_list);
LAAFBA_counter = iter.g_counter;
tp_LAAFBA = iter.p_list;
td_LAAFBA = iter.d_list;
tpd_LAAFBA = max.(tp_LAAFBA,td_LAAFBA);
t_step_LAAFBA = iter.step_list;
t_eta_LAAFBA = iter.eta_list;

len = length(t_cost_AFBA_1_1)
t_min_AFBA_1_1 = similar(t_cost_AFBA_1_1)
t_min_AAFBA = similar(t_cost_AAFBA)
t_min_GPDAL = similar(t_cost_GPDAL)
t_min_EGRPDA = similar(t_cost_EGRPDA)
t_min_LAAFBA = similar(t_cost_LAAFBA)
for i in 1:2801
    t_min_AFBA_1_1[i] = minimum(t_cost_AFBA_1_1[1:i])
    t_min_AAFBA[i] = minimum(t_cost_AAFBA[1:i])
end
for i in 1:2001
    t_min_LAAFBA[i] = minimum(t_cost_LAAFBA[1:i])
    t_min_GPDAL[i] = minimum(t_cost_GPDAL[1:i])
    t_min_EGRPDA[i] = minimum(t_cost_EGRPDA[1:i])
end

optcost = minimum(t_cost_EGRPDA)
minimum(t_cost_AAFBA)

p = plot((t_cost_AFBA_1_1[1:2000].-optcost)./optcost,label="AFBA", lw=4, yaxis=:log, legend=:topright, legendfont = 12, tickfont = 12, guidefont = 12, alpha=0.7)
plot!((t_cost_AAFBA[AAFBA_counter[1:2000]].-optcost)./optcost,label="AAFBA", lw=4, yaxis=:log, alpha=0.7)
plot!((t_cost_GPDAL[GPDAL_counter[1:2000]].-optcost)./optcost,label="GPDAL", lw=4, yaxis=:log, alpha=0.7)
plot!((t_cost_EGRPDA[EGRPDA_counter[1:2000]].-optcost)./optcost,label="GRPDA-L", lw=4, yaxis=:log, alpha=0.7)
plot!((t_cost_GPDAL[LAAFBA_counter[1:2000]].-optcost)./optcost,label="AAFBA-L", lw=4, yaxis=:log, alpha=0.7)
xlabel!("Matrix Multiplication Evaluations"); ylabel!("Cost (x,∇x)", lw=4, alpha=0.7)
savefig(p,"TV_Cost_Comparison_2.svg")

tpd_min_AFBA_1_1 = similar(t_cost_AFBA_1_1)
tpd_min_AAFBA = similar(t_cost_AAFBA)
tpd_min_GPDAL = similar(t_cost_GPDAL)
tpd_min_EGRPDA = similar(t_cost_EGRPDA)
tpd_min_LAAFBA = similar(t_cost_LAAFBA)
for i in 1:2500
    tpd_min_AFBA_1_1[i] = minimum(tpd_AFBA_1_1[1:i])
    tpd_min_AAFBA[i] = minimum(tpd_AAFBA[1:i])
end
for i in 1:2000
    tpd_min_LAAFBA[i] = minimum(tpd_LAAFBA[1:i])
    tpd_min_GPDAL[i] = minimum(tpd_GPDAL[1:i])
    tpd_min_EGRPDA[i] = minimum(tpd_EGRPDA[1:i])
end

p = plot((tpd_AFBA_1_1[1:2000]),label="AFBA", lw=4, yaxis=:log, legend=:topright, legendfont = 12, tickfont = 12, guidefont = 12, alpha=0.7)
plot!((tpd_AAFBA[AAFBA_counter[1:2000]]),label="AAFBA", lw=4, yaxis=:log, alpha=0.7)
plot!((tpd_GPDAL[GPDAL_counter[1:2000]]),label="GPDAL", lw=4, yaxis=:log, alpha=0.7)
plot!((tpd_EGRPDA[EGRPDA_counter[1:2000]]),label="GRPDA-L", lw=4, yaxis=:log, alpha=0.7)
plot!((tpd_LAAFBA[LAAFBA_counter[1:2000]]),label="AAFBA-L", lw=4, yaxis=:log, alpha=0.7)
xlabel!("Matrix Multiplication Evaluations"); ylabel!("Primal Dual Gap", lw=4, alpha=0.7)
savefig(p,"TV_PD_Comparison_2.svg")

p = plot((tpd_min_AFBA_1_1[1:2000]),label="AFBA", lw=4, yaxis=:log, legend=:topright, legendfont = 12, tickfont = 12, guidefont = 12, alpha=0.7)
plot!((tpd_min_AAFBA[AAFBA_counter[1:2000]]),label="AAFBA", lw=4, yaxis=:log, alpha=0.7)
plot!((tpd_min_GPDAL[GPDAL_counter[1:2000]]),label="GPDAL", lw=4, yaxis=:log, alpha=0.7)
plot!((tpd_min_EGRPDA[EGRPDA_counter[1:2000]]),label="GRPDA-L", lw=4, yaxis=:log, alpha=0.7)
plot!((tpd_min_LAAFBA[LAAFBA_counter[1:2000]]),label="AAFBA-L", lw=4, yaxis=:log, alpha=0.7)
xlabel!("Matrix Multiplication Evaluations"); ylabel!("Primal Dual Gap", lw=4, alpha=0.7)
savefig(p,"TV_PD_Min_Comparison_2.svg")

p = plot([1:500], t_step_LAAFBA[1:500], label="AAFBA-L", lw=4, legendfont=12, tickfont=12, guidefont=12, alpha=0.7, legend=:bottomright)
plot!([1:500], t_step_AAFBA[1:500], label="AAFBA", lw=4, legendfont=12, tickfont=12, guidefont=12, alpha=0.7)
xlabel!("Iteration"); ylabel!("Stepsize (τ)"); ylims!((0,0.4))
savefig(p,"TV_Stepsize_Comparison_2.svg")

p = plot(t_eta_LAAFBA[1:500], label="AAFBA-L", lw=4, legendfont=12, tickfont=12, guidefont=12, alpha=0.7)
xlabel!("Iteration"); ylabel!("Spectral Norm Estimate"); ylims!((0,3.3))
savefig(p,"TV_LAAFBA_Opnorm_2.svg")
