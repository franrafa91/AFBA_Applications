## Using the 'ROF' model for total variation based image denoising.
using ProximalAlgorithms
using ProximalOperators
using Noise, TestImages, Images, Colors, FileIO, ImageFiltering
using SparseArrays

# Image Treatment
img_clean = testimage("lighthouse");
img_blurred = imfilter(img_clean, Kernel.gaussian(0));
img = salt_pepper(img_blurred)

convert(Array{RGB{Float64}}, img);
img = Float64.(channelview(img)[:,:,:]);
size(img)[1] == 3 ? (c, n, m) = size(img) : ((n, m) = size(img); c=1);

new_img = similar(img);

# Construct the Linear Transformation for TV
include("concatLMs.jl")
include("TV_2D.jl")
K = ConcatLMs.ConcatLM(TV_2D(n,m)); #

# Apply Primal-Dual Algorithm
θ = 2; μ = 0; iter = 0;
solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-2))
λ = 0.6; τ = 0.1; σ = 1.0/(8*τ);
include("indBallL2_row.jl")
for ch ∈ 1:c
    c==1 ?  b = vec(img) : b = vec(img[ch,:,:]);

    g = Translate(NormL1(Float64(1)),-b);
    h = NormL2_row(λ);    hᶜ = IndBallL2_row(λ);
    y₀= prox(hᶜ,(1+σ)*K*b)[1];

    (x_afba, y_afba), iter = solver(x0 = b, y0 = y₀, g = g, h=h, L=K, gamma = (τ,σ))

    c==1 ? new_img = reshape(x_afba,n,m) : new_img[ch,:,:] = reshape(x_afba,n,m)
end
c==1 ? colorview(Gray,new_img) : colorview(RGB,new_img)

#I still need to define the cost function of h in indBallL2_row.jl