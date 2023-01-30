###DEFINE INDBALL THAT WILL RETURN PROX PER ROW
export IndBallL2_row
import ProximalCore: prox!, convex_conjugate


struct NormL2_row{R}
    lambda::R
    function NormL2_row{R}(lambda::R) where R
        if lambda < 0
            error("parameter λ must be nonnegative")
        else
            new(lambda)
        end
    end
end

is_convex(f::Type{<:NormL2_row}) = true
is_positively_homogeneous(f::Type{<:NormL2_row}) = true

NormL2_row(lambda::R=1) where R = NormL2_row{R}(lambda)

function (f::NormL2_row)(x)
    s1, s2 = size(x)
    y = zeros(eltype(x),s1)
    for i in 1:s2
        y .+= x[:,i].^2
    end
    y = sqrt.(y)
    return λ*sum(y)
end

convex_conjugate(f::NormL2_row) = IndBallL2_row(f.lambda)

struct IndBallL2_row{R}
    r::R
    function IndBallL2_row{R}(r::R) where {R}
        if r <= 0
            error("parameter r must be positive")
        else
            new(r)
        end
    end
end

is_convex(f::Type{<:IndBallL2_row}) = true
is_set(f::Type{<:IndBallL2_row}) = true

IndBallL2_row(r::R=1) where R = IndBallL2_row{R}(r)

function isapprox_le(x::Number, y::Number; atol::Real=0, rtol::Real=Base.rtoldefault(x,y,atol))
    x <= y || (isfinite(x) && isfinite(y) && abs(x-y) <= max(atol, rtol*max(abs(x), abs(y))))
end

function (f::IndBallL2_row)(x)
    s1, s2 = size(x)
    R = real(eltype(x))
    y = zeros(R,s1);
    for i ∈ 1:s1
        if isapprox_le(norm(x[i,:]), f.r, atol=eps(R), rtol=sqrt(eps(R)))
            y[i] = 0
        else
            y[i] = Inf
        end
    end
    return sum(y)
end

function prox!(y, f::IndBallL2_row, x, gamma)
    s1, s2 = size(x)
    R = real(eltype(x))
    scal = zeros(R,s1)
    for i ∈ 1:s1
        t = norm(x[i,:])/f.r
        if t > 1
            scal[i] = t
        else
            scal[i] = 1
        end
    end
    y[:,:] = x./reshape(repeat(scal,s2),s1,s2)
    return R(0)
end

function prox_naive(f::IndBallL2_row, x, gamma)
    y=similar(x)
    prox!(y, f, x, gamma)
    return y, real(eltype(x))(0)
end