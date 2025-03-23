using ProblemReductions, Graphs, LinearAlgebra, Printf
using CairoMakie

function transition_matrix(model::SpinGlass, beta::T) where T
    N = num_variables(model)
    P = zeros(T, 2^N, 2^N)  # P[i, j] = probability of transitioning from j to i
    readbit(cfg, i::Int) = (cfg >> (i - 1)) & 1  # read the i-th bit of cfg
    int2cfg(cfg::Int) = [readbit(cfg, i) for i in 1:N]
    for j in 1:2^N
        for i in 1:2^N
            if count_ones((i-1) ⊻ (j-1)) == 1  # Hamming distance is 1
                P[i, j] = 1/N * min(one(T), exp(-beta * (energy(model, int2cfg(i-1)) - energy(model, int2cfg(j-1)))))
            end
        end
        P[j, j] = 1 - sum(P[:, j])  # rejected transitions
    end
    return P
end

# Analyze spectral properties of the transition matrix
function spectral_gap(P)
    eigenvalues = LinearAlgebra.eigvals(P, sortby=x -> real(x))
    # The spectral gap is 1 - λ₂
    @assert eigenvalues[end] ≈ 1
    return 1.0 - real(eigenvalues[end-1])
end

function main(N::Int, J::Float64)
    graph = Graphs.cycle_graph(N)
    model = ProblemReductions.SpinGlass(graph, J * ones(ne(graph)), zeros(nv(graph)))
    betas = 1 ./ [0.5, 1.0, 2.0, 4.0, 10.0]
    gaps = Float64[]
    for beta in betas
        P = transition_matrix(model, beta)
        gap = spectral_gap(P)
        
        println("1/T: $beta, Spectral gap: $gap")
        push!(gaps, gap)
        
        # Estimate mixing time
        mixing_time = ceil(1.0 / gap)
        println("Estimated mixing time: $mixing_time steps")
    end

    # Plot spectral gap vs temperature
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1], 
        xlabel="1/T (β)", 
        ylabel="Spectral Gap (1 - λ₂)",
        title="Spectral Gap vs 1/T",
        yscale=log10,
        limits=((0, 2.2), (1e-4, 0.5)),
        )
    lines!(ax, betas, gaps)
    # Add text annotations for each data point showing the gap value
    for (x, y) in zip(betas, gaps)
        text!(ax, @sprintf("%.3e", y), position=(x, y), align=(:center, :bottom), offset=(10, 5))
    end
    # Add markers to make the data points more visible
    scatter!(ax, betas, gaps, markersize=10)
    fig
end

main(6, -1.0)
