## Using the 'ROF' model for total variation based image denoising.

using ProximalAlgorithms
using ProximalOperators
using Noise, TestImages, Images, Colors, FileIO, ImageFiltering
using SparseArrays


# img_clean = load("C:/Users/franr/OneDrive/Desktop/KU Leuven/Thesis/TVL1denoise/TVL1denoise/sweden.png")
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

# Apply Proximal Algorithm (Chambolle Pock in this case)
include("indBallL2_row.jl")
for ch ∈ 1:c
    c==1 ?  b = vec(img) : b = vec(img[ch,:,:]);

    λ = 0.6; τ = 0.1; σ = 1.0/(8*τ); θ = 1.0;
    n_iter = 50;

    f = Translate(NormL1(Float64(1)),-b)
    hᶜ = IndBallL2_row(λ);

    p = K*b;
    pᵢ = p; bᵢ = b; p_prox = p;
    v = similar(b); v_prox = similar(b);

    for i ∈ 1:n_iter
        pᵢ = p_prox + σ*(K*bᵢ);
        prox!(p_prox,hᶜ,pᵢ,σ);

        d = K'*p_prox;

        v = bᵢ - τ*d;
        prox!(v_prox,f,v,τ);

        bᵢ = v_prox + θ*(v_prox-bᵢ);
    end

    c==1 ? new_img = reshape(bᵢ,n,m) : new_img[ch,:,:] = reshape(bᵢ,n,m)
end

c==1 ? colorview(Gray,new_img) : colorview(RGB,new_img)