using JuMP, SCIP, LinearAlgebra

function solve_spin_glass(J::Matrix, h::Vector; verbose = false)
    n = length(h)
    @assert size(J, 1) == size(J, 2) == n "The size of J and h must be the same!"
    @assert ishermitian(J) "J must be a Hermitian matrix!"

    model = Model(SCIP.Optimizer)
    !verbose && set_silent(model)

    @variable(model, 0 <= s[i = 1:n] <= 1, Int)            # spin: 0 -> 1, 1 -> -1
    @variable(model, 0 <= d[i = 1:n, j = i+1:n] <= 1, Int) # d[i,j] = s_i s_j

    for i = 1:n
        for j = i+1:n
            # map: (0, 0), (1, 1) -> 0 and (0, 1), (1, 0) -> 1
            @constraint(model, d[i,j] <= s[i] + s[j])
            @constraint(model, d[i,j] <= 2 - s[i] - s[j])
            @constraint(model, d[i,j] >= s[i] - s[j])
            @constraint(model, d[i,j] >= s[i] - s[j])
        end
    end
    
    @objective(model, Min, sum(J[i,j] * (1 - 2 * d[i,j]) for i = 1:n, j = i+1:n) + sum((1 - 2 * s[i]) * h[i] for i = 1:n))
    optimize!(model)
    energy = objective_value(model)
    @assert is_solved_and_feasible(model) "The problem is infeasible!"
    return energy, 1 .- 2 .* value.(s)
end

J = triu(fill(1.0, 10, 10), 1)
J += J'  # anti-ferromagnetic complete graph
h = zeros(size(J, 1))
Emin, configuration = solve_spin_glass(J, h)
