using JuMP, COSMO, Graphs, LinearAlgebra

function maxcut_sdp(G::SimpleGraph)
    n = nv(G)
    model = Model(COSMO.Optimizer)
    @variable(model, X[1:n, 1:n], PSD)
    for i in 1:n
        @constraint(model, X[i, i] == 1)
    end
    @objective(model, Min, sum(X[e.src, e.dst] for e in edges(G)))
    optimize!(model)
    return project_hyperplane(value(X), randn(n))
end

function project_hyperplane(X::AbstractMatrix{Float64}, H::Vector{Float64})
    n = length(H)
    @assert size(X, 1) == size(X, 2) == n
    #res = cholesky(X .+ Diagonal(1e-4 * ones(n)))
    res = eigen(X)
    return [dot(res.values[:, i], H) > 0 ? 1 : -1 for i in 1:n]
end

G = random_regular_graph(100, 3)
approx_maxcut = maxcut_sdp(G)

using GenericTensorNetworks
sg = SpinGlass(G, ones(Int, ne(G)), zeros(Int, nv(G)))
solve(sg, SizeMin())[]
energy(sg, approx_maxcut)


