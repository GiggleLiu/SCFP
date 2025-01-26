using JuMP, HiGHS, Graphs, LinearAlgebra, Test

# Find one valid assignment with integer programming
function lattice_sites(num_sites::Vector{Int}, multiplicity::Vector{Int}, num_atoms::Vector{Int})
    model = Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    @variable(model, x[1:length(num_sites), 1:length(num_atoms)] >= 0, Int)
    for (ia, na) in enumerate(num_atoms)
        @constraint(model, sum(x[i, ia] * num_sites[i] for i in 1:length(num_sites)) == na)
    end
    for (i, ms) in enumerate(multiplicity)
        @constraint(model, sum(x[i, ia] for ia in 1:length(num_atoms)) <= ms)
    end
    optimize!(model)
    mask = value.(x)
    return mask2assign(mask)
end
mask2assign(mask::AbstractMatrix) = [findall(!iszero, mask[:, ia]) for ia in 1:size(mask, 2)]

# Find all valid assignments with a naive branching
function valid_combinations(num_sites::Vector{Int}, multiplicity::Vector{Int}, num_atoms::Vector{Int}, assigned::Vector{Vector{Int}} = [Int[] for ia in 1:length(num_atoms)])
    @assert !all(iszero, multiplicity) && all(>(0), num_sites)
    jatom = findfirst(!iszero, num_atoms)
    jatom === nothing && return [assigned]
    maxsite = isempty(assigned[jatom]) ? 0 : maximum(assigned[jatom])
    candidate_site = findall(isite -> (maxsite <= isite) && (multiplicity[isite] > 0) && (num_sites[isite] <= num_atoms[jatom]), 1:length(num_sites))
    isempty(candidate_site) && return Vector{Vector{Int}}[]   # invalid branch
    return mapreduce(vcat, candidate_site) do site
        sub_num_atoms = copy(num_atoms)
        sub_multiplicity = copy(multiplicity)
        sub_assigned = copy.(assigned)
        sub_multiplicity[site] -= 1
        sub_num_atoms[jatom] -= num_sites[site]
        # jatom is assigned to the site
        push!(sub_assigned[jatom], site)
        valid_combinations(num_sites, sub_multiplicity, sub_num_atoms, sub_assigned)
    end
end

# Count all valid assignments
function count_combinations(num_sites, multiplicity, num_atoms)
    # Find all valid assignments for each atom
    solutions_single = [first.(valid_combinations(num_sites, multiplicity, num_atoms[i:i])) for i in 1:length(num_atoms)]
    # Count the number of valid assignments for each combination of multiplicity
    INTMAX = 100
    base = lcm(filter(<(INTMAX), multiplicity))
    tables = count_table.(solutions_single, Ref(multiplicity), base, INTMAX)
    # Join the tables, conflicts are detected
    return reduce((table1, table2) -> join_count_table(table1, table2, base), tables)
end

# Join the tables, conflicts are detected, `base` is the least common multiple of the multiplicities
function join_count_table(table1, table2, base)
    new_table = Dict{Vector{Int}, BigInt}()
    for (key1, value1) in table1
        for (key2, value2) in table2
            new_key = key1 + key2  # combine the occupied sites to make a new key
            if all(<=(base), new_key)  # `base` decides the maximum value of each site
                new_table[new_key] = get(new_table, new_key, 0) + value1 * value2
            end
        end
    end
    return new_table
end

# Count the number of valid assignments for each combination of occupied sites and multiplicity
# `base` is the least common multiple of the multiplicities
# `INTMAX` is the maximum value of multiplicity. If a multiplicity is greater than `INTMAX`, it is considered as unlimited
# The result is a dictionary, the key is the number of sites occupied, the value is the number of valid assignments
function count_table(solutions::Vector{Vector{Int}}, multiplicity::Vector{Int}, base::Int, INTMAX::Int)
    # map: number of sites occupied/multiplicity => number of valid assignments
    # the key is multiplied by `base` to become integer
    table = Dict{Vector{Int}, BigInt}()  # use BigInt to avoid overflow
    for solution in solutions
        occ = zeros(Int, length(multiplicity))
        for site in solution
            occ[site] += 1
        end
        key = map(i -> multiplicity[i] >= INTMAX ? 0 : occ[i] * (base ÷ multiplicity[i]), 1:length(multiplicity))
        table[key] = get(table, key, 0) + 1
    end
    return table
end

# Check if the result is valid
function is_valid(num_sites, multiplicity, num_atoms, result)
    occ = Dict{Int, Int}()   # map: site => number of atoms
    for sites in result
        for site in sites
            occ[site] = get(occ, site, 0) + 1
        end
    end
    all(sum(num_sites[i] for i in result[ia]) == num_atoms[ia] for ia in 1:length(num_atoms)) && # Check if the number of sites occupied is correct
    all(multiplicity[i] >= get(occ, i, 0) for i in 1:length(num_sites)) # Check if the multiplicity is correct
end

@testset "count_table" begin
    solution = [[1, 2, 2], [1, 2, 2, 3], [1, 2, 3, 3]]
    multiplicity = [3, 2, 10000000]
    @test count_table(solution, multiplicity, 6, 100) == Dict([2, 6, 0]=>2, [2, 3, 0]=>1)
end

@testset "lattice_sites" begin
    num_sites = [1, 1, 2, 2, 2, 3]
    multiplicity = [1, 1, 1, 1, 1, 1]
    num_atoms = [2, 3]
    result = lattice_sites(num_sites, multiplicity, num_atoms)
    @test is_valid(num_sites, multiplicity, num_atoms, result)
    res = valid_combinations(num_sites, multiplicity, num_atoms)
    @test length(res) == 16
    @test all(is_valid(num_sites, multiplicity, num_atoms, r) for r in res)
    @test sum(values(count_combinations(num_sites, multiplicity, num_atoms))) == 16
end

@testset "valid_combinations" begin
    num_sites = [8, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2]
    # 100 means no restriction on multiplicity
    multiplicity = [100, 100, 100, 100, 100, 100, 100, 1, 1, 1, 1]
    num_atoms = [8, 4, 16]
    res = valid_combinations(num_sites, multiplicity, num_atoms)
    @test length(res) == 150840
    @test all(is_valid(num_sites, multiplicity, num_atoms, r) for r in res)
    count = sum(values(count_combinations(num_sites, multiplicity, num_atoms)))
    @test count == 150840
    @test typeof(count) == BigInt
end

