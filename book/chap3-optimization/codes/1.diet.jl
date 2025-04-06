using JuMP, SCIP

model = Model(SCIP.Optimizer)

@variable(model, x1 >= 0)
@variable(model, x2 >= 0)

@constraint(model, 0x1 + 7x2 >= 8)
@constraint(model, 4x1 + 2x2 >= 15)
@constraint(model, 2x1 + x2 >= 3)

@objective(model, Min, 6x1 + 3.5x2)

optimize!(model)

value(x1), value(x2), objective_value(model)
