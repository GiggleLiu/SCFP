"""
    hungarian(cost_matrix::AbstractMatrix{T}) where T<:Real

Solve the assignment problem using the Hungarian algorithm (Kuhn-Munkres algorithm).

Given an n×m cost matrix, finds the optimal assignment of rows to columns
that minimizes the total cost. Returns a tuple of (assignment, total_cost).

# Arguments
- `cost_matrix`: An n×m matrix where entry (i,j) is the cost of assigning row i to column j

# Returns
- `assignment`: Vector of length n where assignment[i] is the column assigned to row i
                (0 if unassigned when m < n)
- `total_cost`: The minimum total cost of the optimal assignment

# Example
```julia
cost = [
    4 2 8
    4 3 7
    3 1 6
]
assignment, cost = hungarian(cost)
# assignment = [2, 3, 1], cost = 12
```
"""
function hungarian(cost_matrix::AbstractMatrix{T}) where T<:Real
    n, m = size(cost_matrix)
    dim = max(n, m)
    
    # Create square cost matrix (pad with large value for proper matching)
    INF = typemax(T) == Inf ? T(Inf) : T(10) * sum(abs, cost_matrix) + one(T)
    C = fill(zero(T), dim, dim)
    C[1:n, 1:m] .= cost_matrix
    
    # Potentials for rows and columns
    u = zeros(T, dim + 1)
    v = zeros(T, dim + 1)
    
    # Matching: p[j] = row matched to column j (1-indexed, 0 means unmatched)
    p = zeros(Int, dim + 1)
    
    # For each row, find augmenting path and update matching
    for i in 1:dim
        # p[0] is a virtual column matched to row i
        p[dim + 1] = i
        
        # Minimum slack to each column
        minv = fill(INF, dim + 1)
        
        # Which row gives the minimum slack to each column
        way = zeros(Int, dim + 1)
        
        # Columns visited
        used = falses(dim + 1)
        
        # Start from virtual column 0 (stored at index dim+1)
        j0 = dim + 1
        
        while p[j0] != 0
            used[j0] = true
            i0 = p[j0]
            delta = INF
            j1 = 0
            
            # Update minimum slack for all unvisited columns
            for j in 1:dim
                if !used[j]
                    # Reduced cost
                    cur = C[i0, j] - u[i0] - v[j]
                    if cur < minv[j]
                        minv[j] = cur
                        way[j] = j0
                    end
                    if minv[j] < delta
                        delta = minv[j]
                        j1 = j
                    end
                end
            end
            
            # Update potentials
            for j in 1:(dim + 1)
                if used[j]
                    u[p[j]] += delta
                    v[j] -= delta
                else
                    minv[j] -= delta
                end
            end
            
            j0 = j1
        end
        
        # Reconstruct augmenting path
        while j0 != dim + 1
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
        end
    end
    
    # Build assignment from matching
    assignment = zeros(Int, n)
    for j in 1:dim
        if p[j] != 0 && p[j] <= n && j <= m
            assignment[p[j]] = j
        end
    end
    
    # Calculate total cost
    total_cost = zero(T)
    for i in 1:n
        if assignment[i] > 0
            total_cost += cost_matrix[i, assignment[i]]
        end
    end
    
    return assignment, total_cost
end

"""
    hungarian_maximum(profit_matrix::AbstractMatrix{T}) where T<:Real

Solve the maximum weight assignment problem.
Converts to minimization by negating the matrix.

# Returns
- `assignment`: Optimal assignment vector
- `total_profit`: Maximum total profit
"""
function hungarian_maximum(profit_matrix::AbstractMatrix{T}) where T<:Real
    assignment, neg_cost = hungarian(-profit_matrix)
    return assignment, -neg_cost
end

# Demo and testing
# Example 4: Larger random matrix
using Random
using BenchmarkTools
Random.seed!(42)
cost4 = rand(1:20, 200, 200)
println("Cost matrix:")
display(cost4)

assignment4, total4 = hungarian(cost4)

@btime assignment4, total4 = hungarian(cost4)
println("\nOptimal assignment: $assignment4")
println("Minimum total cost: $total4")

# Verify by showing individual assignments
println("\nVerification:")
for i in 1:5
    println("  Row $i → Column $(assignment4[i]), cost = $(cost4[i, assignment4[i]])")
end