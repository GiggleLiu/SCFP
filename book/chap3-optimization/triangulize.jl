using GenericTensorNetworks, Graphs

g = grid((10, 10))
net = SpinGlass(g, UnitWeight(ne(g)), zeros(Int, nv(g)))
configs = collect(solve(net, ConfigsMin(tree_storage=true))[].c)

function landscape_topology(is_neighbor, configs)
    nv = length(configs)
    graph = SimpleGraph(nv)
    for i in 1:nv
        for j in i+1:nv
            if is_neighbor(configs[i], configs[j])
                add_edge!(graph, i, j)
            end
        end
    end
    return graph
end

landscape = landscape_topology((x, y) -> hamming_distance(x, y) <= 2, configs)
connected_components(landscape)

function grid_with_triangles(n::Int, p::Float64)
    ntry = ceil(Int, n^2 * p)
    g = grid((n, n))
    for _ in 1:ntry
        i = rand(1:nv(g))
        # randomly pick a 2nd nearest neighbor and connect them
        k = rand(neighbors(g, rand(neighbors(g, i))))
        i !== k && add_edge!(g, i, k)
    end
    return g
end

for p = 0.0:0.3:3.0
    g = grid_with_triangles(11, p)
    net = GenericTensorNetwork(SpinGlass(g, UnitWeight(ne(g)), zeros(Int, nv(g))); optimizer=TreeSA(ntrials=1, niters=5))
    configs = collect(solve(net, ConfigsMin(tree_storage=true))[].c)
    landscapee = landscape_topology((x, y) -> hamming_distance(x, y) <= 2, configs)
    num_components = length(connected_components(landscapee))
    println("p = $p, number of edges: $(ne(g)), ground state degeneracy: $(length(configs)), number of components: $num_components")
end
