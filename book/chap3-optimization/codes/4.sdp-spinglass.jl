using JuMP, Clarabel, Graphs, LinearAlgebra

function maxcut_sdp(G::SimpleGraph)
    n = nv(G)
    model = Model(Clarabel.Optimizer)
    @variable(model, X[1:n, 1:n], PSD)
    for i in 1:n
        @constraint(model, X[i, i] == 1)
    end
    @objective(model, Min, sum(X[e.src, e.dst] for e in edges(G)))
    optimize!(model)
    return project_hyperplane(value(X), randn(n))
end

# `X` is the optimal solution of the SDP
# `H` is a vector orthogonal to the hyperplane
function project_hyperplane(X::AbstractMatrix{Float64}, H::Vector{Float64})
    n = length(H)
    @assert size(X, 1) == size(X, 2) == n
    # solve the Cholesky decomposition through eigen decomposition (stabler)
    res = eigen(X)
    U = Diagonal(sqrt.(max.(res.values, 0))) * res.vectors'
    return [dot(U[:, i], H) > 0 ? 1 : -1 for i in 1:n]
end

G = random_regular_graph(100, 3)
approx_maxcut = maxcut_sdp(G)

cut_size = sum(approx_maxcut[e.src] != approx_maxcut[e.dst] for e in edges(G))
