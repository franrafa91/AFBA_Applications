module Hkls

export Hkl
export UnHkl

using LinearAlgebra
import LinearAlgebra: mul!, opnorm
using Base: require_one_based_indexing

struct Hkl
    in::Int
    out::Int
    sign::Bool
    function Hkl(in::Int=1, out::Int=1, sign::Bool = true)
        return new(in, out, sign)
    end
end

Base.size(::Hkl,b::AbstractArray) = [trunc(Int,(size(b)[1]+1)/2),trunc(Int,(size(b)[1]+1)/2)]

function mul!(A::AbstractMatrix, L::Hkl, b::AbstractArray)
    n::Int,m::Int = size(L,b);
    l::Int = size(b)[1]
    if (size(A)[1]/L.out != size(A)[2]/L.in || ndims(A) != 2 || size(A)[1] != n*L.out)
        error("DimensionMismatch for Output")
    end
    for i∈1:n
        for j∈1:m
            A[(i-1)*L.out+1:i*L.out,(j-1)*L.in+1:(j*L.in)] = b[i+j-1,:,:]
        end
    end
    if (!L.sign) A .*= -1 end
    return nothing
end

function Base.:(*)(L::Hkl, b::AbstractArray)
    n::Int,m::Int = size(L,b);
    A = zeros(eltype(b),n*L.out,m*L.in);
    mul!(A,L,b)
    return A
end

function Base.:(*)(V::AbstractArray, L::Hkl, b::AbstractArray)
    n::Int,m::Int = size(L,b);
    αb = similar(b)
    for i in L.in
        for j in L.out
            αb[:,i,j] = V.*b[:,i,j];
        end
    end
    A = zeros(eltype(b),n*L.out,m*L.in);
    mul!(A,L,αb)
    return A
end

function Base.:(*)(α::Real, L::Hkl, b::AbstractArray)
    n::Int,m::Int = size(L,b);
    αb = α*b;
    A = zeros(eltype(b),n*L.out,m*L.in);
    mul!(A,L,αb)
    return A
end

function Base.:(-)(L::Hkl)
    return Hkl(L.in,L.out,!L.sign)
end

#Dummy opnorm value to allow AFBA to run
opnorm(L::Hkl) = 1.0;



struct UnHkl
    in::Int
    out::Int
    sign::Bool
    function UnHkl(in::Int=1, out::Int=1, sign::Bool=true)
        return new(in, out, sign)
    end    
end

function LinearAlgebra.adjoint(L::Hkl)
    return UnHkl(L.in, L.out, L.sign)
end
function LinearAlgebra.adjoint(U::UnHkl)
    return Hkl(U.in, U.out, U.sign)
end

function Base.size(La::UnHkl,A::AbstractMatrix)
    n::Int, m::Int = size(A)
    n = n/La.out
    m = m/La.in
    if (n != m || ndims(A) != 2)
        error("Square Hankel Matrix expected")
    end
    return n+m-1
end

function mul!(b::AbstractArray, La::UnHkl, A::AbstractMatrix)
    n::Int,m::Int = size(La',b);
    if (size(A)[1]/La.out != size(A)[2]/La.in || ndims(A) != 2 || size(A)[1] != n*La.out)
        error("DimensionMismatch for Input/Output")
    end
    for i∈1:n
        for j∈i:m
            b[i+j-1,:,:] = A[(i-1)*La.out+1:i*La.out,(j-1)*La.in+1:(j*La.in)]
        end
    end
    if (!La.sign) b .*= -1 end
    return nothing
end

function Base.:(*)(La::UnHkl, A::AbstractMatrix)
    y = zeros(eltype(A),(size(La,A),La.out,La.in))
    mul!(y,La,A)
    return y
end

function Base.:(*)(V::AbstractArray, La::UnHkl, A::AbstractMatrix)
    y = zeros(eltype(A),(La.out*size(La,A)[1],La.in*size(La,A)[2]))
    mul!(y,La,A)
    for i in L.in
        for j in L.out
            y[:,i,j] .= V.*y[:,i,j]
        end
    end
    return y
end

function Base.:(*)(α::Real, La::UnHkl, A::AbstractMatrix)
    y = zeros(eltype(A),(La.out*size(La,A)[1],La.in*size(La,A)[2]))
    mul!(y,La,A)
    y *= α
    return y
end

function Base.:(-)(U::UnHkl)
    return UnHkl(U.in, U.out, !U.sign)
end

end