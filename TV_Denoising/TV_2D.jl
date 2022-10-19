export TV_2D

using SparseArrays
using LinearAlgebra

function TV_2D(n::Int, m::Int)
    A1 = spdiagm(0 => repeat([-1.0],n));
    B1 = Matrix(I,m,m);B1[m,m]=0;
    K1 = kron(B1,A1);
    K1[:,n+1:end] = K1[:,n+1:end]-K1[:,1:n*(m-1)];

    A2 = spdiagm(0 => repeat([-1.0],n-1),1=> repeat([1],n-1));
    B2 = Matrix(I,m,m);
    K2 = kron(B2,A2);
    return K1, K2
end