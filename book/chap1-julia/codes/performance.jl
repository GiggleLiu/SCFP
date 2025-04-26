using BenchmarkTools

# 1. No Global variables
global_var = 1
function add1()
    global global_var
    global_var += 1
end

@btime for i=1:100
    add1()
end

function add100()
    local_var = 1
    for i = 1:100
        local_var += 1
    end
    return local_var
end

@btime add100()

# 2. No Abstract fields

mutable struct AbstractFieldType
    a::Real
end

function add100_v1()
    t = AbstractFieldType(1)
    for i = 1:100
        t.a += 1
    end
    return t.a
end

@btime add100_v1()

mutable struct ConcreteFieldType{T <: Real}
    a::T
end

function add100_v2()
    t = ConcreteFieldType(1)
    for i = 1:100
        t.a += 1
    end
    return t.a
end

@btime add100_v2()

# 3. Use static types for small applications
using StaticArrays
function rotation_matrix_v1(θ)
    return [cos(θ) -sin(θ); sin(θ) cos(θ)]
end
function rotation_matrix_v2(θ)
    return SMatrix{2, 2}(cos(θ), -sin(θ), sin(θ), cos(θ))
end

function rotate_vectors(R, vectors::Matrix)
    @assert size(vectors, 1) == 2
    for i = 1:size(vectors, 2)
        vectors[:, i] = R * view(vectors, :, i)
    end
    return vectors
end

@btime rotate_vectors(rotation_matrix_v1(π/4), $(randn(2, 10000)))

@btime rotate_vectors(rotation_matrix_v2(π/4), $(randn(2, 10000)))

