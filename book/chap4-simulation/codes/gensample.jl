using TensorInference, ProblemReductions, Graphs
using CairoMakie
"""
    create_spin_glass_model(size=(10, 10), β=1.0)

Create a ferromagnetic spin glass model on a grid of given size with inverse temperature β.
"""
function create_spin_glass_model(n::Int, β::Real)
    graph = grid((n, n))
    problem = SpinGlass(graph, -ones(Int, ne(graph)), zeros(Int, nv(graph)))
    pmodel = TensorNetworkModel(problem, β)
    return problem, pmodel
end

"""
    plot_energy_histogram(energies, β)

Create a histogram of energies with mean energy line.
"""
function plot_energy_histogram(energies_list, βs)
    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1])
    
    # Create histograms for each beta value with different colors
    colors = [:skyblue, :salmon, :lightgreen, :orchid]
    for (i, (energies, β)) in enumerate(zip(energies_list, βs))
        # Calculate bin edges to ensure consistent binning across distributions
        bin_edges = range(minimum(minimum.(energies_list)), maximum(maximum.(energies_list)), length=30)
        
        # Create histogram with transparency to see overlaps
        hist!(ax, energies, bins=bin_edges, color=(colors[mod1(i, length(colors))], 0.6), 
              strokewidth=1, strokecolor=:black, label="β=$β")
        
        # Add a vertical line for the mean energy of each distribution
        mean_energy = sum(energies) / length(energies)
        vlines!(ax, mean_energy, color=colors[mod1(i, length(colors))], linestyle=:dash, linewidth=2, 
                label="Mean Energy (β=$β): $(round(mean_energy, digits=2))")
    end
    
    # Add labels and title
    ax.xlabel = "Energy"
    ax.ylabel = "Frequency"
    ax.title = "Energy Distributions of Ferromagnetic Ising Model Samples for Different β Values"
    
    # Add legend
    axislegend(ax)
    
    return fig
end

# Main execution
βs = [0.15, 0.3, 0.6, 1.2]
energies_list = []

for β in βs
    problem, pmodel = create_spin_glass_model(10, β)
    samples = sample(pmodel, 1000)
    energies = energy.(Ref(problem), samples)
    push!(energies_list, energies)
end

fig = plot_energy_histogram(energies_list, βs)
save(joinpath(dirname(@__DIR__), "images", "ising-energy-distribution.svg"), fig)
