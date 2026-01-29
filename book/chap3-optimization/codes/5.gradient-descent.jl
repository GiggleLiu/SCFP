using ForwardDiff  # for obtaining the gradient of an analytic funciton, with automatic differentiation, i.e. can differentiate a function automatically.
using CairoMakie

function gradient_descent(f, x; niters::Int, learning_rate::Real)
    history = [x]
    for i = 1:niters
        g = ForwardDiff.gradient(f, x)
        x -= learning_rate * g
        push!(history, x)
    end
    return history
end

# Step 1: Define the Rosenbrock function
function rosenbrock(x)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

# Visualize the Rosenbrock function landscape
fig = Figure(resolution = (600, 450))
ax = Axis(fig[1, 1], xlabel="x₁", ylabel="x₂", title="Rosenbrock Function Landscape (Heatmap + Contours)")

# Create grid
x1 = LinRange(-2, 2, 300)
x2 = LinRange(-1, 3, 300)
Z = [rosenbrock([x, y]) for x in x1, y in x2]  # rows = y, cols = x

# Heatmap of the function values
heatmap!(ax, x1, x2, Z; colormap = :viridis, colorrange=(0, 300))

# Add contour lines
contour!(ax, x1, x2, Z; levels=1500, linewidth=1, color=:black)
scatter!(ax, [1.0], [1.0]; color=:red, markersize=14, marker=:star5, strokewidth=2, strokecolor=:black, label="Minimum")
text!(ax, "Minimum", position = (1.08, 1.05), align = (:left, :bottom), fontsize=16, color=:red)

fig
save(joinpath(@__DIR__, "rosenbrock_landscape.png"), fig)


# Step 2: Compute gradient (we'll use ForwardDiff)
using ForwardDiff
∇f = x -> ForwardDiff.gradient(rosenbrock, x)
# Step 3: Test the gradient at point (0, 0)
println("Gradient at (0,0): ", ∇f([0.0, 0.0]))
# Step 4: Run gradient descent
x0 = [-1.0, -1.0] # Starting point
history = gradient_descent(rosenbrock, x0; niters=10000, learning_rate=0.002)
println("Final point: ", history[end])
println("Final value: ", rosenbrock(history[end]))

function visualize_trajectory(history)
    # Prepare grid for Rosenbrock function (same as before)
    x1 = LinRange(-2, 2, 300)
    x2 = LinRange(-1, 3, 300)
    Z = [rosenbrock([x, y]) for x in x1, y in x2]  # Note the order: rows = y, cols = x

    # Extract trajectory from history
    traj = hcat(history...)  # each column is x at a step

    fig2 = Figure(resolution = (600, 480))
    ax2 = Axis(fig2[1,1], xlabel="x₁", ylabel="x₂", title="Gradient Descent Trajectory on Rosenbrock Heatmap")

    # Plot heatmap
    heatmap!(ax2, x1, x2, Z; colormap = :viridis, colorrange=(0, 300))

    # Overlay trajectory
    lines!(ax2, traj[1, :], traj[2, :], color=:red, linewidth=2, label="Trajectory")
    scatter!(ax2, [traj[1,1]], [traj[2,1]]; color=:orange, markersize=10, label="Start")
    scatter!(ax2, [traj[1,end]], [traj[2,end]]; color=:green, markersize=10, label="End")
    scatter!(ax2, [1.0], [1.0]; color=:red, markersize=14, marker=:star5, strokewidth=2, strokecolor=:black, label="Minimum")
    text!(ax2, "Minimum", position = (1.08, 1.05), align = (:left, :bottom), fontsize=16, color=:red)
    axislegend(ax2)

    fig2
end

visualize_trajectory(history)

history = gradient_descent(rosenbrock, x0; niters=10000, learning_rate=0.002)
visualize_trajectory(history)

function gradient_descent_momentum(f, x; niters::Int, β::Real, learning_rate::Real)
    history = [x]
    v = zero(x)  # Initialize velocity to zero
    
    for i = 1:niters
        g = ForwardDiff.gradient(f, x)
        v = β .* v .- learning_rate .* g  # Update velocity
        x += v                           # Move by velocity
        push!(history, x)
    end
    return history
end

history = gradient_descent_momentum(rosenbrock, x0; niters=10000, learning_rate=0.0005, β=0.9)
visualize_trajectory(history)

function adagrad_optimize(f, x; niters, learning_rate, ϵ=1e-8)
    rt = zero(x)
    η = zero(x)
    history = [x]
    for step in 1:niters
        Δ = ForwardDiff.gradient(f, x)
        @. rt = rt + Δ^2
        @. η = learning_rate / sqrt(rt + ϵ)
        x = x .- Δ .* η
        push!(history, x)
    end
    return history
end

history = adagrad_optimize(rosenbrock, x0; niters=10000, learning_rate=1.0)
visualize_trajectory(history)

function adam_optimize(f, x; niters, learning_rate, β1=0.9, β2=0.999, ϵ=1e-8)
    mt = zero(x)
    vt = zero(x)
    βp1 = β1
    βp2 = β2
    history = [x]
    for step in 1:niters
        Δ = ForwardDiff.gradient(f, x)
        @. mt = β1 * mt + (1 - β1) * Δ
        @. vt = β2 * vt + (1 - β2) * Δ^2
        @. Δ = mt / (1 - βp1) / (√(vt / (1 - βp2)) + ϵ) * learning_rate
        βp1, βp2 = βp1 * β1, βp2 * β2
        x = x .- Δ
        push!(history, x)
    end
    return history
end

history = adam_optimize(x -> 10000 * rosenbrock(x), x0; niters=10000, learning_rate=0.01)
visualize_trajectory(history)

using Optimisers, ForwardDiff

# Available optimizers include:
# Descent, Momentum, Nesterov, RMSProp, Adam, AdaGrad, etc.

function optimize_with_optimisers(f, x0, optimizer_type; niters=1000)
    method = optimizer_type(0.01)  # learning rate
    state = Optimisers.setup(method, x0)
    history = [x0]
    x = copy(x0)
    
    for i = 1:niters
        grad = ForwardDiff.gradient(f, x)
        state, x = Optimisers.update(state, x, grad)
        push!(history, copy(x))
    end
    
    return history
end

history = optimize_with_optimisers(rosenbrock, x0, Optimisers.Adam)
visualize_trajectory(history)

function newton_optimizer(f, x; tol=1e-5)
    k = 0
    history = [x]
    while k < 3
        k += 1
        gk = ForwardDiff.gradient(f, x)
        hk = ForwardDiff.hessian(f, x)
        dx = -hk \ gk  # Solve Hk * dx = -gk
        x += dx
        push!(history, x)
        sum(abs2, dx) < tol && break
    end
    return history
end

history = newton_optimizer(rosenbrock, x0)
visualize_trajectory(history)
rosenbrock(history[end])

using Optim, ForwardDiff

function optimize_bfgs(f, x0; iterations=1000)
    options = Optim.Options(iterations=iterations, store_trace=true, extended_trace=true)
    result = Optim.optimize(f, (g, x) -> g .= ForwardDiff.gradient(f, x), x0, LBFGS(), options)
    # Extract trajectory from trace
    history = [state.metadata["x"] for state in result.trace]
    return history
end

history = optimize_bfgs(rosenbrock, x0; iterations=20)
visualize_trajectory(history)
rosenbrock(history[end])

