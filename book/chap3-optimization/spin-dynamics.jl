using Graphs, KrylovKit

struct SpinVector{D, T}
    vec::NTuple{D, T}
end
function flatten(spins::Vector{SpinVector{D, T}}) where {D, T}
    res = Vector{T}(undef, D * length(spins))
    for j in 1:length(spins), i in 1:D
        res[i + (j-1) * D] = spins[j].vec[i]
    end
    return res
end
function unflatten!(spins::Vector{SpinVector{D, T}}, vec::Vector{T}) where {D, T}
    @assert length(vec) == D * length(spins) "Input must be a vector of length $(D * length(spins)), got $(length(vec))"
    for j in 1:length(spins)
        spins[j] = SpinVector(ntuple(i -> vec[i + (j-1) * D], D))
    end
    return spins
end

struct ClassicalSpinSystem{T}
    topology::SimpleGraph{Int}
    coupling::Vector{T}
end

struct ClassicalSpinHamiltonian{T} <: AbstractMatrix{T}
    field::Vector{T}
end
srange(k::Int) = 3 * (k-1) + 1:3 * k
ClassicalSpinHamiltonian(::Type{T}; nspins::Int) where T = ClassicalSpinHamiltonian(Vector{T}(undef, nspins * 3))
Base.size(h::ClassicalSpinHamiltonian) = (length(h.field), length(h.field))
Base.size(h::ClassicalSpinHamiltonian, i::Int) = size(h)[i]
function LinearAlgebra.mul!(res::Vector{T}, h::ClassicalSpinHamiltonian{T}, v::Vector{T}) where T
    @assert length(v) == size(h, 2) "Input must be a vector of length $(size(h, 2)), got $(length(v))"
    @inbounds for i in 1:length(v) ÷ 3
        a, b, c = srange(i)
        res[a] = -h.field[c] * v[b] + h.field[b] * v[c]
        res[b] = h.field[c] * v[a] - h.field[a] * v[c]
        res[c] = -h.field[b] * v[a] + h.field[a] * v[b]
    end
    return res
end
function hamiltonian(sys::ClassicalSpinSystem{T}, state::Vector{T}) where T
    h = ClassicalSpinHamiltonian(T; nspins=nv(sys.topology))
    return hamiltonian!(h, sys, state)
end
function hamiltonian!(h::ClassicalSpinHamiltonian{T}, sys::ClassicalSpinSystem{T}, state::Vector{T}) where T
    @assert length(state) == size(h, 2) "Input must be a vector of length $(size(h, 2)), got $(length(state))"
    fill!(h.field, zero(T))
    for (e, J) in zip(edges(sys.topology), sys.coupling)
        i, j = src(e), dst(e)
        h.field[srange(i)] .+= J .* state[srange(j)]
        h.field[srange(j)] .+= J .* state[srange(i)]
    end
    return h
end

function simulate!(spins::Vector{SpinVector{D, T}}, sys::ClassicalSpinSystem{T}; nsteps::Int, dt::T) where {D, T}
    # evolve with the Lie group integrator
    state = flatten(spins)
    h = hamiltonian(sys, state)
    for i in 1:nsteps
        state = exponentiate(h, dt, state)
        # update effective Hamiltonian
        hamiltonian!(h, sys, state)
    end
    return unflatten!(spins, state)
end

@testset "Spin dynamics" begin
    sys = ClassicalSpinSystem(SimpleGraph(3), [1.0, 1.0, 1.0])
    spins = [SpinVector(ntuple(i -> rand(T), 3)) for _ in 1:nv(sys.topology)]
    simulate!(spins, sys; nsteps=10, dt=0.1)
end
