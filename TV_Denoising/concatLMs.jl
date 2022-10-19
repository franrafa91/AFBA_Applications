module ConcatLMs

export ConcatLM

using LinearAlgebra
import LinearAlgebra: mul!, opnorm
using Base: require_one_based_indexing

struct ConcatLM
    A::AbstractMatrix
    B::AbstractMatrix
    function ConcatLM(M1::AbstractMatrix,M2::AbstractMatrix)
        if size(M1)!=size(M2)
            error("Matrices to Concatenate do not have the same size")
        end
        new(M1,M2)
    end
end

ConcatLM(M::Tuple) = ConcatLM(M[1],M[2])

Base.size(L::ConcatLM) = (size(L.A)[1],size(L.B)[2],2)

function mul!(y::AbstractMatrix, L::ConcatLM, l::AbstractVector)
    if (size(y)[1] != size(L.A)[1] || ndims(y) != 2)
        error("DimensionMismatch for Output")
    end
    if size(L.A)[2] != length(l)
        error("DimensionMismatch for Input")
    end
    y[:,1] = L.A*l
    y[:,2] = L.B*l
    return nothing
end

function Base.:(*)(L::ConcatLM, l::AbstractVector)
    y = zeros(size(L.A)[1],2)
    mul!(y,L,l)
    return y
end

function Base.:(*)(α::Real, L::ConcatLM, l::AbstractVector)
    y = zeros(size(L.A)[1],2)
    mul!(y,L,l)
    y *= α
    return y
end

opnorm(::ConcatLM) = sqrt(8);

struct AdjointConcatLM
    L::ConcatLM
    function AdjointConcatLM(L::ConcatLM)
        return new(L)
    end
end

LinearAlgebra.adjoint(A::ConcatLM) = AdjointConcatLM(A)
LinearAlgebra.adjoint(A::AdjointConcatLM) = ConcatLM(A)

Base.size(T::AdjointConcatLM) = (size(T.L.A)[1],size(T.L.B)[2],2)

function mul!(y::AbstractVector, T::AdjointConcatLM, d::AbstractMatrix)
    if length(y) != size(T.L.A)[2]
        error("DimensionMismatch for Output")
    end
    if size(T.L.A)[1] != size(d)[1]
        error("DimensionMismatch for Input")
    end
    mul!(y,transpose(T.L.A),d[:,1])
    mul!(y,transpose(T.L.B),d[:,2],1,1)
    return nothing
end

function Base.:(*)(T::AdjointConcatLM, d::AbstractMatrix)
    y = zeros(size(T.L.A)[2])
    mul!(y,T,d)
    return y
end

function Base.:(*)(α::Real, T::AdjointConcatLM, d::AbstractMatrix)
    y = zeros(size(T.L.A)[2])
    mul!(y,T,d)
    y *= α
    return y
end

end