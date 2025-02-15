# The Computer Language Benchmarks Game
# https://salsa.debian.org/benchmarksgame-team/benchmarksgame/

struct Node
    left::Node
    right::Node
    function Node(left::Node, right::Node)
        new(left, right)
    end
    function Node()
        new()
    end
end

function make(d)
    if d == 0
        Node()
    else
        Node(make(d-1), make(d-1))
    end
end

check(t::Node) = 1 + (isdefined(t, :left) ? check(t.left) : 0) + (isdefined(t, :right) ? check(t.right) : 0)

function loop_depths(d, min_depth, max_depth)
    for i = 0:div(max_depth - d, 2)
        niter = 1 << (max_depth - d + min_depth)
        c = 0
        for j = 1:niter
            c += check(make(d)) 
        end
        println("$niter trees of depth $d check: $c")
        d += 2
    end
end

function perf_binary_trees(N::Int=10)
    min_depth = 4
    max_depth = N
    stretch_depth = max_depth + 1

    # create and check stretch tree
    let c = check(make(stretch_depth))
        println("stretch tree of depth $stretch_depth check: $c")
    end

    long_lived_tree = make(max_depth)

    loop_depths(min_depth, min_depth, max_depth)
    println("long lived tree of depth $max_depth check: $(check(long_lived_tree))")
end

perf_binary_trees(21)