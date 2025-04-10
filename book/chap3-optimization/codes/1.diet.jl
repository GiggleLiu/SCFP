# Import the JuMP optimization modeling package and SCIP solver
using JuMP, SCIP

# Create a new optimization model using the SCIP solver
model = Model(SCIP.Optimizer)

# Define decision variables x1 and x2, both constrained to be non-negative
@variable(model, x1 >= 0)
@variable(model, x2 >= 0)

# Add nutritional requirement constraints
@constraint(model, 0x1 + 7x2 >= 8)    # Requirement for nutrient 1
@constraint(model, 4x1 + 2x2 >= 15)   # Requirement for nutrient 2
@constraint(model, 2x1 + x2 >= 3)     # Requirement for nutrient 3

# Set the objective function to minimize the total cost
# where x1 costs 6 units and x2 costs 3.5 units
@objective(model, Min, 6x1 + 3.5x2)

# Solve the optimization problem
optimize!(model)

# Return the optimal values of x1, x2, and the minimum cost
value(x1), value(x2), objective_value(model)
