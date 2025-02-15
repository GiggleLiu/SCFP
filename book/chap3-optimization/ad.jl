using DifferentiationInterface
import Mooncake, LinearAlgebra
import ChainRulesCore
using ChainRulesCore: unthunk, NoTangent, @thunk, AbstractZero

# the forward function
function symeigen(A::AbstractMatrix)
    E, U = LinearAlgebra.eigen(A)
    E, Matrix(U)
end

# the backward function for the eigen decomposition
# References: Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
function symeigen_back(E::AbstractVector{T}, U, E̅, U̅; η=1e-40) where T
    all(x->x isa AbstractZero, (U̅, E̅)) && return NoTangent()
    η = T(η)
    if U̅ isa AbstractZero
        D = LinearAlgebra.Diagonal(E̅)
    else
        F = E .- E'
        F .= F./(F.^2 .+ η)
        dUU = U̅' * U .* F
        D = (dUU + dUU')/2
        if !(E̅ isa AbstractZero)
            D = D + LinearAlgebra.Diagonal(E̅)
        end
    end
    U * D * U'
end

# port the backward function to ChainRules
function ChainRulesCore.rrule(::typeof(symeigen), A)
    E, U = symeigen(A)
    function pullback(y̅)
        A̅ = @thunk symeigen_back(E, U, unthunk.(y̅)...)
        return (NoTangent(), A̅)
    end
    return (E, U), pullback
end

# port the backward function to Mooncake
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(symeigen), Matrix{Float64}}

# prepare a test function
function tfunc(A, target)
    E, U = symeigen(A)
    return sum(abs2, U[:, 1] - target)
end

# use the Mooncake AD engine
backend = DifferentiationInterface.AutoMooncake(; config=nothing)

# the function only takes one argument, so we wrap it in a tuple
wrapped(x) = tfunc(x...)

# pre-allocate the memory for the gradient, speed up the gradient computation
A = randn(Float64, 100, 100); A += A'
house = LinearAlgebra.normalize(vcat(zeros(20), 50 .+ (0:29), 80 .- (29:-1:0) , zeros(20)))
prep = DifferentiationInterface.prepare_gradient(wrapped, backend, (A, house))

# compute the gradient
g2 = DifferentiationInterface.gradient(wrapped, prep, backend, (A, house))

Mooncake.TestUtils.test_rule(Mooncake.Xoshiro(123), wrapped, (A, house); is_primitive=false)

using Optim

function train(house::Vector{Float64})
    A = randn(Float64, 100, 100); A += A'
    function objective(A)
        E, U = symeigen(A)
        return sum(abs2, U[:, 1] - house)
    end
    prep = DifferentiationInterface.prepare_gradient(objective, backend, A)
    function gradient!(g, A)
        g .= DifferentiationInterface.gradient(objective, prep, backend, A)
    end
    res = Optim.optimize(objective, gradient!, A, Optim.LBFGS())
    return res.minimizer
end

train(house)

using CairoMakie