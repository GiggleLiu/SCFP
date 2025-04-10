# This file implements the SDP relaxation for the MaxCut problem
# The MaxCut problem is to find a partition of the vertices of a graph into two sets
# such that the number of edges crossing the partition is maximized
using JuMP, Clarabel, Graphs, LinearAlgebra

# Solve the MaxCut problem using semidefinite programming (SDP) relaxation
# The SDP relaxation is:
#   min ∑_{(i,j)∈E} (1/4)(1 - x_i x_j)
#   s.t. x_i^2 = 1 for all i
# Which is equivalent to:
#   min ∑_{(i,j)∈E} X_{ij}
#   s.t. X_{ii} = 1 for all i
#        X ⪰ 0 (positive semidefinite)
function maxcut_sdp(G::SimpleGraph)
    n = nv(G)  # Number of vertices in the graph
    model = Model(Clarabel.Optimizer)
    
    # Create a positive semidefinite matrix variable X
    @variable(model, X[1:n, 1:n], PSD)
    
    # Add constraints: diagonal elements must be 1 (representing unit vectors)
    for i in 1:n
        @constraint(model, X[i, i] == 1)
    end
    
    # Objective: minimize the sum of X[i,j] for all edges (i,j)
    # This is equivalent to maximizing the cut size after transformation
    @objective(model, Min, sum(X[e.src, e.dst] for e in edges(G)))
    
    optimize!(model)
    
    # Use random hyperplane rounding to convert SDP solution to a cut
    return project_hyperplane(value(X), randn(n))
end

# Project the SDP solution onto a random hyperplane to get a valid cut
# `X` is the optimal solution of the SDP
# `H` is a random vector defining the hyperplane
function project_hyperplane(X::AbstractMatrix{Float64}, H::Vector{Float64})
    n = length(H)
    @assert size(X, 1) == size(X, 2) == n
    
    # Solve the Cholesky decomposition through eigen decomposition (more stable)
    # X = U'U where U is the matrix we need for projection
    res = eigen(X)
    U = Diagonal(sqrt.(max.(res.values, 0))) * res.vectors'
    
    # Assign vertices to sides of the cut based on their projection onto H
    # If dot product is positive, assign +1, otherwise -1
    return [dot(U[:, i], H) > 0 ? 1 : -1 for i in 1:n]
end

# Create a random 3-regular graph with 100 vertices for testing
G = random_regular_graph(100, 3)

# Solve the MaxCut problem using our SDP relaxation
approx_maxcut = maxcut_sdp(G)

# Calculate the size of the resulting cut
cut_size = sum(approx_maxcut[e.src] != approx_maxcut[e.dst] for e in edges(G))
