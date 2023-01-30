export quad_perturb, l2_overbox
import ProximalCore: prox!, convex_conjugate, is_convex
import ProximalOperators: IndBox

#EVERYTHING IS WRONG!!! DO IT AS IN EXAMPLE 6.23 FOR PROX OF L1 OVER BOX

struct l2_overbox{R}
    box::IndBox
    beta::R
    function l2_overbox{R}(box::IndBox,beta::R) where {R}
        if beta < 0
            error("parameter β must be nonnegative")
        else
            new(box,beta)
        end
    end
end

l2_overbox(box::IndBox,beta::R=1.0) where {R} = l2_overbox{R}(box,beta)

is_convex(f::l2_overbox) = true

function (f::l2_overbox)(x)
    y = f.box(x)
    y += f.beta*norm(x)
end

function prox!(y, f::l2_overbox, x, gamma)
    z = similar(y)
    prox!(y,f.box, x, gamma)
    l2 = NormL2(f.beta)
    prox!(z,l2,x,gamma)
    y .= min.(y,z)
    return eltype(x)(0)
end

function prox_naive(f::l2_overbox, x, gamma)
    y=similar(x)
    prox!(y, f, x, gamma)
    return y, eltype(x)(0)
end


struct quad_perturb{R,T}
    g::T
    beta::R
    function quad_perturb{R, T}(g::T,beta::R) where {R, T}
        if beta < 0
            error("parameter β must be nonnegative")
        else
            new(g, beta)
        end
    end
end

quad_perturb(g::T, beta::R) where {R,T} = quad_perturb{R,T}(g,beta)

is_convex(f::quad_perturb) = is_convex(f.g)

function (f::quad_perturb)(x)
    if typeof(f.g)<:IndBox
        g_ext = IndBox(f.g.lb.-1e-8,f.g.ub.+1e-8)
        y = g_ext(x)
    else
        y = f.g(x)
    end
    y += f.beta/2*norm(x)^2
    return y
end

function prox!(y, f::quad_perturb, x, gamma)
    # g_frac = Postcompose(f.g,1.0/(f.beta+1))
    prox!(y, f.g, x./(gamma*f.beta+1),gamma/(gamma*f.beta+1))
    return Real(0)
end

function prox_naive(f::quad_perturb, x, gamma)
    y=R(x)
    prox!(y, f, x, gamma)
    return y, real(eltype(x))(0)
end