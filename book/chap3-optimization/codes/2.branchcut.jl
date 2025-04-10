# Sub-problem 1
using JuMP, SCIP

model = Model(SCIP.Optimizer)

@variable(model, x1 >= 0)
@variable(model, x2 >= 0)
@constraint(model, x1 + x2 <= 5)
@constraint(model, 4 * x1 + 7 * x2 <= 28)

# Sub-problem 2: uncomment the following line
# @constraint(model, x2 <= 2)

# Sub-problem 3: uncomment the following line
# @constraint(model, x2 >= 3)

# Sub-problem 4: uncomment the following 2 lines
# @constraint(model, x2 >= 3)
# @constraint(model, x1 >= 2)

# Sub-problem 5: uncomment the following 2 lines
# @constraint(model, x2 <= 2)
# @constraint(model, x1 <= 1)

# Sub-problem 6: uncomment the following 2 lines
# @constraint(model, x1 <= 1)
# @constraint(model, x2 >= 4)

# Sub-problem 7: uncomment the following 2 lines
# @constraint(model, x1 <= 1)
# @constraint(model, x2 <= 3)

@objective(model, Max, 5 * x1 + 6 * x2)

optimize!(model)
value(x1), value(x2), objective_value(model)

##### The direct approach #####
# Import the JuMP optimization modeling package and SCIP solver
using JuMP, SCIP

# Create a new optimization model using the SCIP solver
model = Model(SCIP.Optimizer)

# Define integer decision variables x1 and x2, both constrained to be non-negative
@variable(model, x1 >= 0, Int)
@variable(model, x2 >= 0, Int)

# Add constraints to the model
@constraint(model, x1 + x2 <= 5)    # Total resource constraint
@constraint(model, 4 * x1 + 7 * x2 <= 28)    # Budget constraint

# Set the objective function to maximize the total value
@objective(model, Max, 5 * x1 + 6 * x2)

# Solve the optimization problem
optimize!(model)

# Return the optimal values of x1, x2, and the maximum objective value
value(x1), value(x2), objective_value(model)