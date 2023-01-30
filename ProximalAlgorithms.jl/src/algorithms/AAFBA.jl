# Reyes, F. Rafael, "Speeding up Asymmetric Forward-Backward-Adjoint
# Splitting Algorithms: Methods and Applications"
# Mathematical Engineering Master's Thesis - KU LEUVEN
#
# Latafat, P. et al. "Adaptive proximal algorithms for convex optimization
# under local Lipschitz continuity of the gradient." In: arXiv (2023). DOI:
# https://doi.org/10.48550/arXiv.2301.04431.
#
using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalCore: Zero, IndZero, convex_conjugate
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

"""
    AAFBAIteration(; <keyword-arguments>)

Iterator implementing the asymmetric forward-backward-adjoint algorithm (AAFBA, see [1]).

This iterator solves convex optimization problems of the form

    minimize f(x) + g(x) + h(L x),

where `f` is smooth, `g` and `h` are possibly nonsmooth and `L` is a linear mapping.

Points `x0` and `y0` are the initial primal and dual iterates, respectively.
If unspecified, functions `f`, `g`, and `h` default to the identically zero
function, and `L` defaults to the identity.

# Arguments
- `x0`: initial primal point.
- `y0`: initial dual point.
- `f=Zero()`: smooth objective term.
- `g=Zero()`: proximable objective term.
- `h=Zero()`: proximable objective term.
- `l=IndZero()`: strongly convex function.
- `L=I`: linear operator (e.g. a matrix).
- `gamma1`: primal stepsize (see [1] for the default choice).
- `gamma2`: dual stepsize (see [1] for the default choice).
"""
Base.@kwdef struct AAFBAIteration{R,V,Tx,Ty,Tf,Tg,Th,TL,Tbetaf,Ttheta,Tmu,Tlambda,Tgamma}
    f::Tf = Zero()
    g::Tg = Zero()
    h::Th = Zero()
    L::TL = if isa(h, Zero) L = 0 * I else I end
    x0::Tx
    y0::Ty
    beta_f::Tbetaf = if isa(f, Zero)
        real(eltype(x0))(0)
    else
        error("argument beta_f must be specified together with f")
    end
    theta::Ttheta = real(eltype(x0))(1)
    mu::Tmu = real(eltype(x0))(1)
    lambda::Tlambda = real(eltype(x0))(1)
    gamma::Tgamma = 1e-5
    t::R = 1.0
    L_norm::R = 1.0
    δ::R = 1e-5
    c::R = 1.1
    r::R = 1.5
    R_inc::R = 1.05
    xout::Bool = false
    yout::Bool = false
    dual::Bool = false
    opnorm::Bool = false
    f_list::Vector{V} = []
    g_list::Vector{V} = []
    h_list::Vector{V} = []
    p_list::Vector{V} = []
    d_list::Vector{V} = []
    eta_list::Vector{V} = []
    step_list::Vector{V} = []
    step_min::Vector{V} = []
    g_counter::Vector{V} = []
end

Base.IteratorSize(::Type{<:AAFBAIteration}) = Base.IsInfinite()


Base.@kwdef struct AAFBAState{Tx,Ty,V}
    x::Tx
    y::Ty
    xbar::Tx = similar(x)
    ybar::Ty = similar(y)
    gradf::Tx = similar(x)
    gradf_new::Tx = similar(x)
    # gradl::Ty = similar(y)
    FPR_x::Tx = similar(x)
    FPR_y::Ty = similar(y)
    temp_x::Tx = similar(x)
    temp_y::Ty = similar(y)
    adap::V #γₖ, γₖ₋₁, ηₑ, ηₖ, Lₖ, Cₖ
end

function Base.iterate(iter::AAFBAIteration, state::AAFBAState = AAFBAState(x=copy(iter.x0), y=copy(iter.y0), adap = [iter.gamma, iter.gamma, (iter.opnorm ? opnorm(iter.L) : iter.L_norm), (iter.opnorm ? opnorm(iter.L) : iter.L_norm), 0.0, 0.0]))
    if state.x == iter.x0
        costs(iter,state)
        gradient!(state.gradf_new, iter.f, state.x)
    end

    τₖ₋₁ = copy(state.adap[2])
    τₖ = copy(state.adap[1])
    τₖ₊₁ = 0.; σₖ₊₁ = 0.;
    append!(iter.step_list,τₖ)
    append!(iter.eta_list,state.adap[3])

    # perform xbar-update step
    state.gradf .= state.gradf_new
    mul!(state.temp_x, iter.L', state.y) 
    state.temp_x .+= state.gradf
    state.temp_x .*= -τₖ
    state.temp_x .+= state.x
    prox!(state.xbar, iter.g, state.temp_x, τₖ)

    # Calculate Lₖ and Cₖ
    gradient!(state.gradf_new, iter.f, state.xbar)
    state.gradf .-= state.gradf_new
    state.temp_x .= state.x - state.xbar
    state.adap[5] = -dot(state.gradf,state.temp_x)
    state.adap[5] /= norm(state.temp_x)^2
    if isnan(state.adap[5]) state.adap[5] = 0 end

    state.adap[6] = norm(state.gradf)^2
    state.adap[6] /= dot(state.gradf,state.temp_x)
    if isnan(state.adap[6]) state.adap[6] = 0 end

    Δ = τₖ*state.adap[5]*(τₖ*state.adap[6]-1)
    χ = iter.t^2*τₖ^2*state.adap[4]^2*(1+iter.δ)^2
    state.adap[3] = iter.R_inc*max(1,copy(state.adap[4]))

    while true
        a = τₖ*sqrt(1+τₖ/τₖ₋₁)
        b = 1/(2*iter.c*iter.t*state.adap[3])
        if iter.opnorm
           c = τₖ*sqrt((1-4*χ*(1+iter.δ)^2)/(2*(1+iter.δ)*(Δ^2+χ*(1-4*χ*(1+iter.δ)^2))+Δ)) 
        else
            c = τₖ*sqrt((1-4*χ)/(2*(1+iter.δ)*(sqrt(Δ^2+(iter.t*state.adap[3]*τₖ)^2*(1-4*χ))+Δ)))
        end
        
        τₖ₊₁ = min(a,b,c)
        σₖ₊₁ = iter.t^2*τₖ₊₁

        # Perform ybar-update step
        state.temp_x .= state.xbar - state.x
        state.temp_x .*= τₖ₊₁/τₖ
        state.temp_x .+= state.xbar
        mul!(state.temp_y, iter.L, state.temp_x)
        state.temp_y .*= σₖ₊₁
        state.temp_y .+= state.y
        prox!(state.ybar,convex_conjugate(iter.h),state.temp_y,σₖ₊₁)

        # Estimate Linear Operator Norm
        state.temp_y .= state.ybar - state.y
        mul!(state.temp_x,iter.L',state.temp_y)
        if !iter.opnorm
            state.adap[4] = norm(state.temp_x)
            state.adap[4] /= norm(state.temp_y)
        end

        append!(iter.g_counter,length(iter.step_list))
        if (iter.opnorm)
            append!(iter.step_min,argmin([a,b,c]))
            break
        else
            r1 = 1/(2*iter.c*iter.t*state.adap[4])
            r2 = τₖ*sqrt((1-4*χ)/(2*(sqrt(Δ^2+(iter.t*state.adap[4]*τₖ)^2*(1-4*χ))+Δ)))
            if τₖ₊₁ <= min(r1,r2)
                append!(iter.step_min,argmin([a,b,c]))
                break
            else
                state.adap[3] *= iter.r
            end
        end
    end

    # # the residues
    state.FPR_x .= state.xbar - state.x
    state.FPR_y .= state.ybar - state.y

    # Primal Inclusion
    mul!(state.temp_x, iter.L', state.FPR_y)
    state.temp_x .-= state.FPR_x./τₖ 
    state.temp_x .-= state.gradf
    append!(iter.p_list,norm(state.temp_x))

    # Dual Inclusion
    mul!(state.temp_y, iter.L, state.FPR_x)
    state.temp_y .*= τₖ₊₁/τₖ
    state.temp_y .-= state.FPR_y./σₖ₊₁
    append!(iter.d_list,norm(state.temp_y))
    
    # # perform x-update step
    state.x .= state.xbar

    # # perform y-update step
    state.y .= state.ybar

    state.adap[1:2] = [τₖ₊₁,τₖ]
    costs(iter,state)
return state, state
end

default_stopping_criterion(tol, ::AAFBAIteration, state::AAFBAState) = norm(state.FPR_x, Inf) + norm(state.FPR_y, Inf) <= tol
default_solution(::AAFBAIteration, state::AAFBAState) = (state.x, state.y)
default_display(it, ::AAFBAIteration, state::AAFBAState) = @printf("%6d | %7.4e\n", it, norm(state.FPR_x, Inf) + norm(state.FPR_y, Inf))

"""
    AAFBA(; <keyword-arguments>)

Constructs the asymmetric forward-backward-adjoint algorithm (AAFBA, see [1]).

This algorithm solves convex optimization problems of the form

    minimize f(x) + g(x) + h(L x),

where `f` is smooth, `g` and `h` are possibly nonsmooth and `L` is a linear mapping.

The returned object has type `IterativeAlgorithm{AAFBAIteration}`,
and can be called with the problem's arguments to trigger its solution.

# Arguments
- `maxit::Int=10_000`: maximum number of iteration
- `tol::1e-5`: tolerance for the default stopping criterion
- `stop::Function`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=100`: every how many iterations to display the algorithm state
- `display::Function`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `AAFBAIteration` constructor upon call
"""
AAFBA(;
    maxit=10_000,
    tol=1e-5,
    stop=(iter, state) -> default_stopping_criterion(tol, iter, state),
    solution=default_solution,
    verbose=false,
    freq=100,
    display=default_display,
    kwargs...
) = IterativeAlgorithm(AAFBAIteration; maxit, stop, solution, verbose, freq, display, kwargs...)