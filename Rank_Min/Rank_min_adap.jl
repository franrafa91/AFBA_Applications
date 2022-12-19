using LinearAlgebra
include("../ProximalAlgorithms.jl/src/ProximalAlgorithms.jl")
using ProximalOperators
include("Hkl.jl")

#RUN ONE OF THE FOLLOWING
#With 0.1 Noise #IN MATLAB
b₂ = [0.929493932436651, 0.0663366640516890, -0.659800663040562, -0.618826116586924, -0.174919468706439, 0.563109812870338, 0.448972944539084, 0.0830422151567688, -0.365805461800296, -0.218133133915710, 0.0891427777939436, 0.406453085322704, 0.384798455360949, 0.0710762719147541, -0.193998250514231, -0.105497764593557, -0.284218519521583, 0.0998759174800447, 0.170319356836481, 0.0869887250577018, -0.316710826960521, -0.340008176668079, 0.0889829273099628, 0.0859798442695906, 0.0664890194532654, 0.000518342317220242, -0.0303925572007856];
b = b₂;
L = Hkls.Hkl()

#4th order system with 0.1 Noise #IN MATLAB
b₄ = [0.00401557284587662, 0.913026332837551, -0.115421539802584, -0.683322060485645, -0.0414096206667291, -0.0508412753483633, 0.0382194374822677, 0.400593647566248, -0.235005811613336, -0.139739339238354, -0.116310116698392, 0.0155445833083502, 0.316684805757508, 0.0425541582875782, -0.293326406641291, 0.0500193637764752, -0.136463856377924, -0.0795010767747800, 0.00140345744847949, 0.0811885485587010, -0.112995679452976, -0.0210383033982032, -0.197442292264161, 0.0287629245141421, 0.0259751593680660, -0.173447182345009, -0.110824288643139];
b = b₄;
L = Hkls.Hkl()

#2nd order system with 3 inputs and 2 outputs
include("MIMO.jl");
b = reshape(hcat(b1,b2,b3),(size(b1)[1], size(b1)[2], 3));
b = b[1:51,:,:];
L = Hkls.Hkl(3,2)

# If the matrix is PSD, the minimization of the trace can be used
A = L*b;
θ = 2; μ = 1.0; iter = Nothing;
τ = 1e-5; σ = 10*τ; β_f = 1.0; β_h=1.0;

f = Postcompose(Translate(NormL2(1.0),-b),1*β_f/2.,0.);
ϵ = 0*b.+0.30;
g_mod = IndBox(b.-ϵ,b.+ϵ);
h = NuclearNorm(β_h);

x₀ = b;
y₀ = prox(ProximalAlgorithms.convex_conjugate(h),(1+σ)*L*b)[1];

# With Adaptive AFBA
solver = ProximalAlgorithms.AAFBA(theta = θ, mu = μ, tol = Float64(1e-6), maxit=20000);
(x_afba, y_afba), it, iter = solver(x0 = x₀, y0 = y₀, f=f, beta_f=β_f, g=g_mod, h=h, L=L, gamma=τ, t=sqrt(σ/τ), out=true);

# For Comparison with Original AFBA
# solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-6), maxit=1000);
# (x_afba, y_afba), it, iter = solver(x0 = x₀, y0 = y₀, f=f, beta_f=β_f, g=g_mod, h=h, L=L, gamma=(τ,σ), out=true);

#Plot Results
using Plots
p = plot((iter.f_list+iter.g_list+iter.h_list),label="Total Cost",lw=2)
plot!(iter.f_list,label="Cost f(x)",lw=2,legendfont=10)
plot!(iter.g_list,label="Cost g(x)",lw=2)
plot!(iter.h_list,label="Cost h(Lx)",lw=2)
xlabel!("Iteration"); ylabel!("Cost(x,Lx)")
(iter.f_list+iter.g_list+iter.h_list)[end]


plot(iter.step_list,label="γ",lw=2, legendfont=10,legend=40)
ylabel!("Stepsize (γ)"); xlabel!("Iteration")

scatter((1:it)[iter.step_min.==3],iter.step_min[iter.step_min.==3],label="3. Curvature Limit")
scatter!((1:it)[iter.step_min.==2],iter.step_min[iter.step_min.==2],label="2. Estimated Opnorm Limit")
scatter!((1:it)[iter.step_min.==1],iter.step_min[iter.step_min.==1],label="1. Stepsize Increase",legendfont=10)
ylabel!("Criteria for Stepsize Update"); xlabel!("Iteration")

plot(iter.eta_list,label="ηₖ",lw=2)
ylabel!("Opnorm Estimate (η)"); xlabel!("Iteration")


rank(A,1e-3)
rank(L*x_afba,1e-3)

print(svd(A).S)
print(svd(L*x_afba).S)

print(x_afba)