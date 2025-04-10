using JuMP, SCIP, LinearAlgebra

# This function solves the spin glass ground state problem using integer programming
# The spin glass Hamiltonian is: H = -∑_{i<j} J_{ij} s_i s_j - ∑_i h_i s_i
# where J is the coupling matrix, h is the external field, and s_i = ±1 are the spins
function solve_spin_glass(J::Matrix, h::Vector; verbose = false)
    n = length(h)
    @assert size(J, 1) == size(J, 2) == n "The size of J and h must be the same!"
    @assert ishermitian(J) "J must be a Hermitian matrix!"

    model = Model(SCIP.Optimizer)
    !verbose && set_silent(model)

    # Define binary variables for spins
    # We use a mapping: s[i] = 0 represents spin up (+1)
    #                   s[i] = 1 represents spin down (-1)
    @variable(model, 0 <= s[i = 1:n] <= 1, Int)            
    
    # Auxiliary variables to linearize the quadratic terms
    # d[i,j] represents the product s_i * s_j after mapping
    @variable(model, 0 <= d[i = 1:n, j = i+1:n] <= 1, Int) 

    # Add constraints to enforce d[i,j] = s_i * s_j after mapping
    for i = 1:n
        for j = i+1:n
            # These constraints ensure:
            # d[i,j] = 0 when (s[i],s[j]) is (0,0) or (1,1)
            # d[i,j] = 1 when (s[i],s[j]) is (0,1) or (1,0)
            @constraint(model, d[i,j] <= s[i] + s[j])
            @constraint(model, d[i,j] <= 2 - s[i] - s[j])
            @constraint(model, d[i,j] >= s[i] - s[j])
            @constraint(model, d[i,j] >= s[j] - s[i])  # Fixed the duplicate constraint
        end
    end
    
    # Objective function: minimize the energy of the spin glass
    # We need to convert our binary variables back to ±1 spins using the mapping:
    # s_i = 1 - 2*s[i] and s_i*s_j = 1 - 2*d[i,j]
    @objective(model, Min, sum(J[i,j] * (1 - 2 * d[i,j]) for i = 1:n, j = i+1:n) + sum((1 - 2 * s[i]) * h[i] for i = 1:n))
    
    optimize!(model)
    energy = objective_value(model)
    @assert is_solved_and_feasible(model) "The problem is infeasible!"
    
    # Convert binary variables back to ±1 spins
    return energy, 1 .- 2 .* value.(s)
end

# Example: Create an anti-ferromagnetic complete graph with 10 nodes
J = triu(fill(1.0, 10, 10), 1)  # Upper triangular matrix filled with 1.0
J += J'  # Make it symmetric by adding its transpose
h = zeros(size(J, 1))  # No external field
Emin, configuration = solve_spin_glass(J, h)
