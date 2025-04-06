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
using JuMP, SCIP

model = Model(SCIP.Optimizer)

@variable(model, x1 >= 0, Int)
@variable(model, x2 >= 0, Int)
@constraint(model, x1 + x2 <= 5)
@constraint(model, 4 * x1 + 7 * x2 <= 28)

@objective(model, Max, 5 * x1 + 6 * x2)

optimize!(model)
value(x1), value(x2), objective_value(model)