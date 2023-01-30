# Reyes, F. Rafael, "Speeding up Asymmetric Forward-Backward-Adjoint
# Splitting Algorithms: Methods and Applications"
# Mathematical Engineering Master's Thesis - KU LEUVEN
#
# X.K.Chang, J.Yang, and H.Zhang. "Golden Ratio Primal-Dual Algorithm
# with Linesearch". In: SIAM Journal on Optimization 32 (2022), pp.
# 1584-1613. DOI: https://doi.org/10.1137/21M1420319.

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalCore: Zero, IndZero
import ProximalCore: convex_conjugate
using ProximalOperators: is_convex
import ProximalOperators: Conjugate
using LinearAlgebra
using Printf

function costs(iter::IteratorType, state::StateType) where {IteratorType, StateType}
    if !iter.dual
        if iter.xout
            append!(iter.f_list,iter.f(state.x))
            if (typeof(iter.g)<:IndBox)
                g_ext = IndBox(iter.g.lb.-1e-10,iter.g.ub.+1e-10)
                append!(iter.g_list,g_ext(state.x))
            else
                append!(iter.g_list,iter.g(state.x))
            end
            mul!(state.temp_y, iter.L, state.x)
            append!(iter.h_list,iter.h(state.temp_y))
        elseif iter.yout
            mul!(state.temp_x, iter.L', state.y)
            append!(iter.f_list,iter.f(state.temp_x))
            append!(iter.g_list,iter.g(state.temp_x))
            append!(iter.h_list,iter.h(state.y))
        end
    else
        if iter.xout
            append!(iter.f_list,convex_conjugate(iter.l)(state.y))
            if (typeof(Conjugate(iter.h))<:IndBox)
                hc_ext = IndBox(Conjugate(iter.h).lb.-1e-10,Conjugate(iter.h).ub.+1e-10)
                append!(iter.g_list,hc_ext(state.y))
            else
                append!(iter.g_list,Conjugate(iter.h)(state.y))
            end
            mul!(state.temp_x, iter.L', state.y)
            append!(iter.h_list,Conjugate(iter.g)(state.temp_x))
        end
        if iter.yout
            mul!(state.temp_y, iter.L, state.x)
            append!(iter.f_list,convex_conjugate(iter.l)(state.x))
            append!(iter.g_list,Conjugate(iter.h)(state.x))
            append!(iter.h_list,Conjugate(iter.g)(state.temp_y))
        end
    end
end

function convex_conjugate(A::ProximalCore.ConvexConjugate)
    if ProximalOperators.is_convex(A.f)
        return A.f #For convex function f
    else
        return ProximalCore.convex_conjugate(A)
    end
end

function Conjugate(A::ProximalOperators.Conjugate)
    if ProximalOperators.is_convex(A.f)
        return A.f #For convex function f
    else
        return ProximalOperators.Conjugate(A)
    end
end

"""
    GRPDAIteration(; <keyword-arguments>)

Iterator implementing the asymmetric forward-backward-adjoint algorithm (AFBA, see [1]).

This iterator solves convex optimization problems of the form

    minimize g(x) + h(L x),

where `g` and `h` are possibly nonsmooth. And `L` is a linear mapping.

Points `x0` and `y0` are the initial primal and dual iterates, respectively.
If unspecified, functions `g` and `h` default to the identically zero function
and `L` defaults to the identity.

# Arguments
- `x0`: initial primal point.
- `y0`: initial dual point.
- `g=Zero()`: proximable objective term.
- `h=Zero()`: proximable objective term.
- `L=I`: linear operator (e.g. a matrix).
- `gamma1`: primal stepsize (see [1] for the default choice).
- `gamma2`: dual stepsize (see [1] for the default choice).
"""
Base.@kwdef struct GRPDAIteration{R,Tx,Ty,Tf,Tg,Th,Tl,TL,Tbetaf,Tbetal,Ttheta,Tmu,Tlambda}
    f::Tf = Zero()
    g::Tg = Zero()
    h::Th = Zero()
    l::Tl = IndZero()
    L::TL = if isa(h, Zero) L = 0 * I else I end
    x0::Tx
    y0::Ty
    beta_f::Tbetaf = if isa(f, Zero)
        real(eltype(x0))(0)
    else
        error("argument beta_f must be specified together with f")
    end
    beta_l::Tbetal = if isa(l, IndZero)
        real(eltype(x0))(0)
    else
        error("argument beta_l must be specified together with l")
    end
    theta::Ttheta = real(eltype(x0))(1)
    mu::Tmu = real(eltype(x0))(1)
    lambda::Tlambda = real(eltype(x0))(1)
    gamma::Tuple{R, R}
    δ::R=0.99
    r::R=0.7
    ψ::R = 1.5
    xout::Bool = false
    yout::Bool = false
    dual::Bool = false
    f_list::Vector{R} = []
    g_list::Vector{R} = []
    h_list::Vector{R} = []
    p_list::Vector{R} = []
    d_list::Vector{R} = []
    g_counter::Vector{R} = []
    step_list::Vector{R} = []
end

Base.IteratorSize(::Type{<:GRPDAIteration}) = Base.IsInfinite()

Base.@kwdef struct GRPDAState{Tx,Ty,R}
    x::Tx
    y::Ty
    lsearch::R # τₖ, βₖ, ϕₖ
    z::Tx = similar(x)
    xbar::Tx = similar(x)
    ybar::Ty = similar(y)
    FPR_x::Tx = similar(x)
    FPR_y::Ty = similar(y)
    temp_x::Tx = similar(x)
    temp_y::Ty = similar(y)
end

function Base.iterate(iter::GRPDAIteration, state::GRPDAState = GRPDAState(x=copy(iter.x0), y=copy(iter.y0), z=copy(iter.x0), lsearch=[iter.gamma[1],iter.gamma[2]/iter.gamma[1],(1+iter.ψ)/(iter.ψ^2)]))
    if state.x == iter.x0
        costs(iter, state)
        append!(iter.step_list, state.lsearch[1])
    end

    τₖ₋₁, β, ϕ = state.lsearch

    # perform xbar-update step
    state.z .= (iter.ψ-1)/iter.ψ.*state.x + 1/iter.ψ.*state.z; #z
    mul!(state.temp_x, iter.L', state.y)
    state.temp_x .*= -τₖ₋₁
    state.temp_x .+= state.z
    prox!(state.xbar, iter.g, state.temp_x, τₖ₋₁)

    τₖ = ϕ*τₖ₋₁; σₖ = β*τₖ;
    # perform linesearch iteration
    for i in 0:100
        if (i != 0) τₖ *= iter.r end
        σₖ = β*τₖ

        # perform ybar-update step
        mul!(state.temp_y, iter.L, state.xbar)
        state.temp_y .*= σₖ
        state.temp_y .+= state.y
        prox!(state.ybar, convex_conjugate(iter.h), state.temp_y, σₖ)
        
        state.temp_y .= state.ybar-state.y
        mul!(state.temp_x,iter.L',state.temp_y)
        append!(iter.g_counter,length(iter.step_list))
        if ((τₖ₋₁*σₖ)*norm(state.temp_x)^2
            <=iter.δ^2*iter.ψ*norm(state.temp_y)^2)
            break
        end
    end

    # the residues
    state.FPR_x .= state.xbar .- state.x
    state.FPR_y .= state.ybar .- state.y

    # Primal Inclusion
    mul!(state.temp_x, iter.L', state.FPR_y)
    state.temp_x .+= (state.z-state.xbar)./τₖ₋₁
    append!(iter.p_list,norm(state.temp_x))

    # Dual Inclusion
    state.temp_y .= -state.FPR_y./σₖ
    append!(iter.d_list,norm(state.temp_y))

    # perform x-update step
    state.x .= state.xbar

    # perform y-update step
    state.y .= state.ybar

    costs(iter, state)
    append!(iter.step_list, τₖ)

    state.lsearch[1] = τₖ
    return state, state
end

default_stopping_criterion(tol, ::GRPDAIteration, state::GRPDAState) = norm(state.FPR_x, Inf) + norm(state.FPR_y, Inf) <= tol
default_solution(::GRPDAIteration, state::GRPDAState) = (state.xbar, state.ybar)
default_display(it, ::GRPDAIteration, state::GRPDAState) = @printf("%6d | %7.4e\n", it, norm(state.FPR_x, Inf) + norm(state.FPR_y, Inf))

"""
    GRPDA(; <keyword-arguments>)

Constructs the asymmetric forward-backward-adjoint algorithm (AFBA, see [1]).

This algorithm solves convex optimization problems of the form

    minimize g(x) + h(L x),

where `g` and `h` are possibly nonsmooth and `L` is a linear mapping.

The returned object has type `IterativeAlgorithm{GRPDAIteration}`,
and can be called with the problem's arguments to trigger its solution.

# Arguments
- `maxit::Int=10_000`: maximum number of iteration
- `tol::1e-5`: tolerance for the default stopping criterion
- `stop::Function`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=100`: every how many iterations to display the algorithm state
- `display::Function`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `AFBAIteration` constructor upon call
"""
GRPDA(;
    maxit=10_000,
    tol=1e-5,
    stop=(iter, state) -> default_stopping_criterion(tol, iter, state),
    solution=default_solution,
    verbose=false,
    freq=100,
    display=default_display,
    kwargs...
) = IterativeAlgorithm(GRPDAIteration; maxit, stop, solution, verbose, freq, display, kwargs...)