using ForwardDiff
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
(1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
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
contour!(ax, x1, x2, Z; levels=15, linewidth=1, color=:black)
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

history = gradient_descent(rosenbrock, x0; niters=10000, learning_rate=0.00002)
visualize_trajectory(history)


###### 