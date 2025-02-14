using DifferentiationInterface
import Mooncake, ChainRulesCore, LinearAlgebra
using ChainRulesCore: unthunk, NoTangent, @thunk

# define the forward function
function lstsq(A, b)
    return A \ b
end

# define the backward function
function lstsq_back(A, b, x, x̅)
    Q, R_ = LinearAlgebra.qr(A)
    R = LinearAlgebra.UpperTriangular(R_)
    y = R' \ x̅
    z = R \ y
    residual = b .- A*x
    b̅ = Q * y
    return residual * z' - b̅ * x', b̅
end

# port the backward function to ChainRules
function ChainRulesCore.rrule(::typeof(lstsq), A, b)
	x = lstsq(A, b) 
    function pullback(x̅)
        A̅, b̅ = @thunk lstsq_back(A, b, x, unthunk(x̅))
        return (NoTangent(), A̅, b̅)
    end
    return x, pullback
end

# port the backward function to Mooncake
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(lstsq), Matrix{Float64}, Vector{Float64}}

# test
T = Float64
M, N = 10, 5
A = randn(T, M, N)
b = randn(T, M)
op = randn(N, N)
op += op'

function tfunc(A, b)
    x = lstsq(A, b)
    return x'*op*x
end

backend = DifferentiationInterface.AutoMooncake(; config=nothing)
wrapped(x) = tfunc(x...)
prep = DifferentiationInterface.prepare_gradient(wrapped, backend, (A, b))
g2 = DifferentiationInterface.gradient(wrapped, prep, backend, (A, b))
Mooncake.TestUtils.test_rule(Mooncake.Xoshiro(123), wrapped, (A, b); is_primitive=false)