## Using the 'ROF' model for total variation based image denoising.
include("../ProximalAlgorithms.jl/src/ProximalAlgorithms.jl")
using JLD2
using ProximalOperators
using Noise, TestImages, Images, Colors, FileIO, ImageFiltering

# Image Input
# If testimage is desired, uncomment this section
img_clean = testimage("lighthouse")
img_blurred = imfilter(img_clean, Kernel.gaussian(0));
img = salt_pepper(img_blurred, 0.3)

# and comment this section
# @load "img_noisy.tif" img

# Image Treatment
save(File{format"PNG"}("Noisy_Image.png"),img)
convert(Array{RGB{Float64}}, img);
img = Float64.(channelview(img)[:,:,:]);
size(img)[1] == 3 ? (c, n, m) = size(img) : ((n, m) = size(img); c=1);
new_img = similar(img);

# Construct the Linear Transformation for TV
include("concatLMs.jl")
include("TV_2D.jl")
K = ConcatLMs.ConcatLM(TV_2D(n,m)); #

# Apply Primal-Dual Algorithm
θ = 2.; μ = 1.;
it = 0; iter = nothing; x_afba=nothing; y_afba=nothing;
solver = ProximalAlgorithms.AFBA(theta = θ, mu = μ, tol = Float64(1e-2), maxit=1000);
λ = 0.8; τ = 0.2; σ = 1/(8τ);
include("indBallL2_row.jl")

function timeTV(c, img, new_img)
    it = 0; iter = nothing; x_afba=nothing; y_afba=nothing;
    for ch ∈ 1:c
        c==1 ?  b = vec(img) : b = vec(img[ch,:,:]);

        g = Translate(NormL1(Float64(1)),-b);
        h = NormL2_row(λ);    hᶜ = IndBallL2_row(λ);
        y₀= prox(hᶜ,K*b+σ*K*b,σ)[1];

        (x_afba, y_afba), it, iter = solver(x0 = b, y0 = y₀, g = g, h=h, L=K, gamma = (τ,σ), xout=true);
        c==1 ? new_img = reshape(x_afba,n,m) : new_img[ch,:,:] = reshape(x_afba,n,m)
    end
    return it, iter, x_afba, y_afba, new_img
end    
it, iter, x_afba, y_afba, new_img = timeTV(c, img, new_img)
new_img = map(clamp01nan,new_img)
print(size(new_img))
if c>1 new_img = colorview(RGB,new_img) end
save("Denoised_Image.png",new_img)

using Plots
p = plot((iter.g_list+iter.h_list),label="Total Cost",legendfont=12, tickfont=12, guidefont=12, lw=4, alpha=0.7)
plot!(iter.g_list,label="Linear Fit Cost g(x)", lw=4, alpha=0.7)
plot!(iter.h_list,label="Directional Difference Regul. h(∇x)", lw=4, alpha=0.7)
xlabel!("Iteration"); ylabel!("Cost(x,∇x)", lw=6)
savefig(p,"TV_Cost_Evolution.svg")