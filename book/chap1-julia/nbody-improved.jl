using Base.Cartesian

const SOLAR_MASS = 4 * pi * pi
const DAYS_PER_YEAR = 365.24
# Precalculate the pairs of bodies that must be compared so that it
# doesn't have to be done each loop.
const PAIRS = Tuple((i, j) for i=1:4 for j=i+1:5)

# Use a struct instead of mutable struct since a struct can be stored
# inline in an array avoiding the overhead of following a pointer
struct Body
    x::NTuple{3,Float64}
    v::NTuple{3,Float64}
    m::Float64
end

# Advance all bodies in the system by one timestep of dt.
@inline function advance!(bodies, dt)
    Δx = map(p -> (@inbounds bodies[p[1]].x .- bodies[p[2]].x), PAIRS)
    mag = map(x -> dt * inv(sqrt(sum(abs2, x)))^3, Δx)

    # Update the velocities of each body using the precalculated mag.
    k = 1
    @inbounds for i in 1:lastindex(bodies)-1
        bi = bodies[i]
        iv = bi.v
        for j in i+1:lastindex(bodies)
            iv = iv .- Δx[k] .* (mag[k] * bodies[j].m)
            bodies[j] = Body(bodies[j].x,
                             bodies[j].v .+ Δx[k] .* (mag[k] * bi.m),
                             bodies[j].m)
            k += 1
        end
        bodies[i] = Body(bi.x, iv, bi.m)
    end

    # Advance body positions using the updated velocities.
    @inbounds for i=1:5
        bi = bodies[i]
        bodies[i] = Body(bi.x .+ bi.v .* dt, bi.v, bi.m)
    end
end

# Total energy of the system
function energy(bodies)
    e = 0.0
    # Kinetic energy of bodies
    @inbounds for b in bodies
        e += 0.5 * b.m * sum(b.v .* b.v)
    end
    
    # Potential energy between body i and body j
    @inbounds for (i, j) in PAIRS
        Δx = bodies[i].x .- bodies[j].x
        e -= bodies[i].m * bodies[j].m / sqrt(sum(Δx .* Δx))
    end
    e
end

# Mutate bodies array according to symplectic integrator in advance!
# for n iterations.
function simulate!(bodies, n, dt)
    for _ = 1:n
        advance!(bodies, dt)
    end
    return bodies
end

# Doing the allocation of the Vector{Body} as a global constant
# instead of within the simulate! function speeds up inference
# considerably. Inference takes less than 60% of the time it would
# otherwise for an overall speedup of 2%-3%.
const bodies = [
    # Jupiter
    Body(( 4.841e+0, -1.160e+0, -1.036e-1),   # x, y, z
         ( 1.660e-3, 7.699e-3, -6.905e-5) .* DAYS_PER_YEAR,  # vx, vy, vz
         9.547e-4 * SOLAR_MASS),  # mass
    # Saturn
    Body(( 8.343e+0, 4.125e+0, -4.035e-1),
         (-2.767e-3, 4.998e-3, 2.304e-5) .* DAYS_PER_YEAR,
         2.858e-4 * SOLAR_MASS),
    # Uranus
    Body(( 1.289e+1, -1.511e+1, -2.23e-1),
         ( 2.96e-3, 2.378e-3, -2.96e-5) .* DAYS_PER_YEAR,
         4.36e-5 * SOLAR_MASS),
    # Neptune
    Body(( 1.537e+1, -2.591e+1, 1.792e-1),
         ( 2.680e-3, 1.628e-3, -9.515e-5) .* DAYS_PER_YEAR,
         5.151e-5 * SOLAR_MASS),
    # Sun
    Body((0.0, 0.0, 0.0), (-1.061e-6, -8.966e-6, 6.553e-8) .* DAYS_PER_YEAR, 3.947e+1)
]

println(energy(bodies))
@time simulate!(bodies, 50000000, 0.01);
println(energy(bodies))
using Test
@test energy(bodies) ≈ -0.16902434059489277
