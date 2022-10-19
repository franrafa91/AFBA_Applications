module Hkls

export Hkl
export UnHkl

using LinearAlgebra
import LinearAlgebra: mul!, opnorm
using Base: require_one_based_indexing

struct Hkl
end

Base.size(::Hkl,b::AbstractVector) = [(length(b)+1)/2,(length(b)+1)/2]

function mul!(A::AbstractMatrix, L::Hkl, b::AbstractVector)
    n::Int,m::Int = size(L,b);
    if (size(A)[1] != size(A)[2] || ndims(A) != 2 || size(A)[1] != n)
        error("DimensionMismatch for Output")
    end
    for i∈1:m
        A[:,i] = b[i:i+n-1]
    end
    return nothing
end

function Base.:(*)(L::Hkl, b::AbstractVector)
    n::Int,m::Int = size(L,b);
    A = zeros(eltype(b),n,m);
    mul!(A,L,b)
    return A
end

function Base.:(*)(α::Real, L::Hkl, b::AbstractVector)
    n::Int,m::Int = size(L,b);
    αb = α*b;
    A = zeros(eltype(b),n,m);
    mul!(A,L,αb)
    return A
end

#I'm not actually sure of this value
opnorm(L::Hkl) = 1.0;



struct UnHkl
end

LinearAlgebra.adjoint(::UnHkl) = Hkl()
LinearAlgebra.adjoint(::Hkl) = UnHkl()

function Base.size(::UnHkl,A::AbstractMatrix)
    n::Int, m::Int = size(A)
    if (n != m || ndims(A) != 2)
        error("Square Hankel Matrix expected")
    end
    return n+m-1
end

function mul!(b::AbstractVector, La::UnHkl, A::AbstractMatrix)
    L = Hkl()
    n::Int,m::Int = size(L,b);
    if (size(A)[1] != size(A)[2] || ndims(A) != 2 || size(A)[1] != n)
        error("DimensionMismatch for Input/Output")
    end
    b[1:n] = A[:,1];
    b[n:n+m-1] = A[n,:];
    return nothing
end

function Base.:(*)(La::UnHkl, A::AbstractMatrix)
    y = zeros(eltype(A),size(La,A))
    mul!(y,La,A)
    return y
end

function Base.:(*)(α::Real, La::UnHkl, A::AbstractMatrix)
    y = zeros(eltype(A),size(La,A))
    mul!(y,La,A)
    y *= α
    return y
end

end