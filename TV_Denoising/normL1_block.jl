# L1 norm (times a constant, or weighted)

export NormL1_block
import ProximalCore: prox!


"""
    NormL1(λ=1)

With a nonnegative scalar parameter λ, return the ``L_1`` norm
```math
f(x) = λ\\cdot∑_i|x_i|.
```
With a nonnegative array parameter λ, return the weighted ``L_1`` norm
```math
f(x) = ∑_i λ_i|x_i|.
```
"""
struct NormL1_block{T}
    lambda::T
    function NormL1_block{T}(lambda::T) where T
        if !(eltype(lambda) <: Real)
            error("λ must be real")
        end
        if any(lambda .< 0)
            error("λ must be nonnegative")
        else
            new(lambda)
        end
    end
end

is_separable(f::Type{<:NormL1_block}) = true
is_convex(f::Type{<:NormL1_block}) = true
is_positively_homogeneous(f::Type{<:NormL1_block}) = true

NormL1_block(lambda::R=1) where R = NormL1_block{R}(lambda)

function (f::NormL1_block)(x)
    R = eltype(x)
    s1,s2 = size(x)
    @assert round(Int32,s1/s2) == s1/s2
    s = round(Int32,s1/s2)
    z = zeros(R,s1)
    for j∈0:s2-1
        z[1:s] = z[1:s] + x[1+j*s:((j+1)*s),j+1]
    end
    z = z./s2
    return f.lambda*norm(z,1)
end



function prox!(y::AbstractArray, f::NormL1_block, x::AbstractArray, gamma)
    @assert size(y) == size(x)
    R = eltype(x)
    s1,s2 = size(x)
    @assert round(Int32,s1/s2) == s1/s2
    s = round(Int32,s1/s2)
    z = zeros(R,s)
    for j∈0:s2-1
        z[1:s] = z[1:s] + x[1+j*s:((j+1)*s),j+1]
    end
    z = z./s2
    fill!(y,0)
    gl = gamma * f.lambda
    for i∈1:s
        y[i,1] = z[i] + (z[i] <= -gl ? gl : (z[i] >= gl ? -gl : -z[i]))
    end
    for j∈2:s2
        y[s+1:j*s,j] = y[1:s,1]
    end
    return x
    # return sum(f.lambda .* abs.(y))
end

function prox!(y, f::NormL1_block, x::AbstractArray{<:Complex}, gamma)
    error("Not implemented for Complex matrices")
end

function gradient!(y, f::NormL1_block, x)
    R = eltype(x)
    s1,s2 = size(x)
    @assert round(Int32,s1/s2) == s1/s2
    s = round(Int32,s1/s2)
    z = zeros(R,s1)
    for j∈0:floor(s2-1)
        z[1+j*s:((j+1)*s)] = x[1+j*s:((j+1)*s),j+1]
    end
    fill!(y,0)
    for j∈0:s2-1
        for i∈1:s
            y[i+j*s,j+1] = f.lambda .* sign(z[i+j*s])
        end
    end
    return f(x)
end

function prox_naive(f::NormL1_block, x, gamma)
    y = similar(x)
    prox!(y,f,x,gamma)
    return y, sum(f.lambda .* abs.(y))
end
