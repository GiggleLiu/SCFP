# Examples of Dominant Eigenvalue Solvers in Graph Theory
# This file demonstrates several important applications of eigenvalue methods to graph problems

using Graphs, LinearAlgebra, SparseArrays, KrylovKit
using Random; Random.seed!(42)

println("=== Examples of Dominant Eigenvalue Solvers in Graph Theory ===\n")

# ============================================================================
# Example 1: PageRank Algorithm
# ============================================================================
println("1. PageRank Algorithm")
println("="^50)

"""
    pagerank(graph, α=0.85, tol=1e-8)

Compute PageRank centrality using the power method.
The PageRank vector is the dominant eigenvector of the Google matrix.
"""
function pagerank(graph, α=0.85, tol=1e-8)
    n = nv(graph)
    # Create the transition matrix P
    A = adjacency_matrix(graph)
    # Convert to row-stochastic matrix (outgoing probabilities)
    P = zeros(n, n)
    for i in 1:n
        deg = degree(graph, i)
        if deg > 0
            for j in neighbors(graph, i)
                P[j, i] = 1.0 / deg  # P[j,i] = probability of going from i to j
            end
        else
            # Handle dangling nodes
            P[:, i] .= 1.0 / n
        end
    end
    
    # Google matrix: G = α*P + (1-α)/n * ones(n,n)
    # Instead of forming G explicitly, we use it implicitly in power iteration
    
    # Power iteration to find dominant eigenvector
    x = ones(n) / n  # Initial uniform distribution
    for iter in 1:1000
        x_new = α * P * x + (1-α) / n * ones(n)
        
        if norm(x_new - x, 1) < tol
            println("PageRank converged in $iter iterations")
            return x_new
        end
        x = x_new
    end
    
    return x
end

# Create a simple directed graph representing web pages
g = SimpleDiGraph(6)
add_edge!(g, 1, 2); add_edge!(g, 1, 3)  # Page 1 links to 2, 3
add_edge!(g, 2, 3); add_edge!(g, 2, 4)  # Page 2 links to 3, 4
add_edge!(g, 3, 1); add_edge!(g, 3, 4); add_edge!(g, 3, 5)  # Page 3 links to 1, 4, 5
add_edge!(g, 4, 5); add_edge!(g, 4, 6)  # Page 4 links to 5, 6
add_edge!(g, 5, 6)  # Page 5 links to 6
add_edge!(g, 6, 1)  # Page 6 links to 1

pr_scores = pagerank(g)
println("PageRank scores:")
for i in 1:nv(g)
    println("  Page $i: $(round(pr_scores[i], digits=4))")
end
println()

# ============================================================================
# Example 2: Eigenvector Centrality
# ============================================================================
println("2. Eigenvector Centrality")
println("="^50)

"""
    eigenvector_centrality(graph)

Compute eigenvector centrality - the dominant eigenvector of the adjacency matrix.
High centrality means the node is connected to other high-centrality nodes.
"""
function eigenvector_centrality(graph)
    A = adjacency_matrix(graph)
    
    # Find the largest eigenvalue and corresponding eigenvector using KrylovKit
    λ, vecs, info = eigsolve(A, randn(nv(graph)), 1, :LR)
    
    if info.converged >= 1
        # Take the absolute value and normalize
        centrality = abs.(real.(vecs[1]))
        centrality = centrality / sum(centrality)  # Normalize to sum to 1
        println("Dominant eigenvalue: $(round(real(λ[1]), digits=4))")
        return centrality
    else
        error("Eigenvalue computation did not converge")
    end
end

# Create Zachary's Karate Club graph (famous social network)
karate = smallgraph(:karate)
eig_cent = eigenvector_centrality(karate)

println("Top 5 nodes by eigenvector centrality:")
sorted_indices = sortperm(eig_cent, rev=true)
for i in 1:5
    node = sorted_indices[i]
    println("  Node $node: $(round(eig_cent[node], digits=4))")
end
println()

# ============================================================================
# Example 3: Spectral Gap and Graph Connectivity
# ============================================================================
println("3. Spectral Gap and Graph Connectivity")
println("="^50)

"""
    spectral_gap(graph)

Compute the spectral gap (1 - λ₂) of the transition matrix.
Larger gaps indicate better connectivity and faster mixing of random walks.
"""
function spectral_gap(graph)
    A = adjacency_matrix(graph)
    D = Diagonal([degree(graph, i) for i in 1:nv(graph)])
    
    # Create the transition matrix P = D^(-1) * A
    # For undirected graphs, this is symmetric
    P = zeros(nv(graph), nv(graph))
    for i in 1:nv(graph)
        deg = degree(graph, i)
        if deg > 0
            for j in neighbors(graph, i)
                P[i, j] = 1.0 / deg
            end
        end
    end
    
    # Find the largest two eigenvalues
    λs, vecs, info = eigsolve(P, randn(nv(graph)), 2, :LR)
    
    if info.converged >= 2
        λ1, λ2 = real.(λs[1:2])
        gap = λ1 - λ2  # λ1 should be 1 for a connected graph
        println("Largest eigenvalue λ₁: $(round(λ1, digits=6))")
        println("Second largest eigenvalue λ₂: $(round(λ2, digits=6))")
        println("Spectral gap: $(round(gap, digits=6))")
        return gap
    else
        error("Could not compute enough eigenvalues")
    end
end

# Compare spectral gaps of different graph structures
println("Comparing spectral gaps of different graphs:")

# Path graph (poorly connected)
path_graph = path_graph(10)
println("\nPath graph (10 nodes):")
gap_path = spectral_gap(path_graph)

# Complete graph (perfectly connected)
complete_graph = complete_graph(10)
println("\nComplete graph (10 nodes):")
gap_complete = spectral_gap(complete_graph)

# Cycle graph (intermediate connectivity)
cycle_graph = cycle_graph(10)
println("\nCycle graph (10 nodes):")
gap_cycle = spectral_gap(cycle_graph)

println("\nSummary: Complete graph has the largest spectral gap (best mixing),")
println("path graph has the smallest gap (worst mixing).\n")

# ============================================================================
# Example 4: Epidemic Threshold
# ============================================================================
println("4. Epidemic Threshold")
println("="^50)

"""
    epidemic_threshold(graph)

The epidemic threshold is related to the largest eigenvalue of the adjacency matrix.
For SIS model: threshold = 1/λ₁, where λ₁ is the largest eigenvalue of A.
"""
function epidemic_threshold(graph)
    A = adjacency_matrix(graph)
    
    # Find the largest eigenvalue (spectral radius)
    λ, vecs, info = eigsolve(A, randn(nv(graph)), 1, :LR)
    
    if info.converged >= 1
        λ_max = real(λ[1])
        threshold = 1.0 / λ_max
        println("Largest eigenvalue of adjacency matrix: $(round(λ_max, digits=4))")
        println("Epidemic threshold (τ = 1/λ₁): $(round(threshold, digits=4))")
        println("Infection spreads if transmission rate > $(round(threshold, digits=4))")
        return threshold, λ_max
    else
        error("Eigenvalue computation did not converge")
    end
end

# Compare epidemic thresholds for different network structures
println("Epidemic thresholds for different networks:")

networks = [
    ("Random graph (n=20, p=0.1)", erdos_renyi(20, 0.1)),
    ("Scale-free graph (n=20)", barabasi_albert(20, 2)),
    ("Small-world graph (n=20)", watts_strogatz(20, 4, 0.3))
]

for (name, net) in networks
    println("\n$name:")
    if is_connected(net)
        epidemic_threshold(net)
    else
        println("  Graph is disconnected - analyzing largest component")
        components = connected_components(net)
        largest_comp = components[argmax(length.(components))]
        subgraph_net = induced_subgraph(net, largest_comp)[1]
        epidemic_threshold(subgraph_net)
    end
end

println()

# ============================================================================
# Example 5: Community Detection using Spectral Clustering
# ============================================================================
println("5. Community Detection using Spectral Clustering")
println("="^50)

"""
    spectral_clustering(graph, k=2)

Simple spectral clustering using the normalized Laplacian.
Returns cluster assignments based on the k smallest eigenvectors.
"""
function spectral_clustering(graph, k=2)
    n = nv(graph)
    A = adjacency_matrix(graph)
    D = Diagonal([degree(graph, i) for i in 1:nv(graph)])
    
    # Normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
    D_sqrt_inv = Diagonal([degree(graph, i) > 0 ? 1.0/sqrt(degree(graph, i)) : 0.0 for i in 1:n])
    L_norm = I - D_sqrt_inv * A * D_sqrt_inv
    
    # Find the k smallest eigenvalues and eigenvectors
    λs, vecs, info = eigsolve(L_norm, randn(n), k, :SR)  # :SR for smallest real
    
    if info.converged >= k
        # Create feature matrix from eigenvectors
        X = hcat([real.(vecs[i]) for i in 1:k]...)
        
        # Simple k-means-like clustering (just use sign of second eigenvector for k=2)
        if k == 2
            clusters = [v >= 0 ? 1 : 2 for v in X[:, 2]]
        else
            # For k > 2, would need proper k-means clustering
            clusters = ones(Int, n)  # Placeholder
        end
        
        println("Spectral clustering completed")
        println("Smallest $(k) eigenvalues: $(round.(real.(λs[1:k]), digits=4))")
        return clusters
    else
        error("Could not compute enough eigenvectors")
    end
end

# Create a graph with two obvious communities
g_communities = SimpleGraph(12)
# Community 1: nodes 1-6 (densely connected)
for i in 1:6, j in (i+1):6
    if rand() < 0.7  # 70% connection probability within community
        add_edge!(g_communities, i, j)
    end
end
# Community 2: nodes 7-12 (densely connected)
for i in 7:12, j in (i+1):12
    if rand() < 0.7
        add_edge!(g_communities, i, j)
    end
end
# Sparse connections between communities
for i in 1:6, j in 7:12
    if rand() < 0.1  # 10% connection probability between communities
        add_edge!(g_communities, i, j)
    end
end

clusters = spectral_clustering(g_communities, 2)
println("Cluster assignments:")
for i in 1:nv(g_communities)
    println("  Node $i: Cluster $(clusters[i])")
end

# Evaluate clustering quality
community1_correct = sum(clusters[1:6] .== clusters[1])
community2_correct = sum(clusters[7:12] .== clusters[7])
accuracy = (community1_correct + community2_correct) / 12
println("Clustering accuracy: $(round(accuracy * 100, digits=1))%")

println("\n" * "="^70)
println("Summary: Dominant eigenvalue solvers are fundamental tools in graph theory for:")
println("1. PageRank: Ranking importance of nodes")
println("2. Centrality measures: Identifying influential nodes")
println("3. Connectivity analysis: Understanding graph structure")
println("4. Epidemic modeling: Predicting disease spread thresholds")
println("5. Community detection: Finding clusters in networks")
println("="^70)
