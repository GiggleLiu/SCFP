using Base.Cartesian

struct Body{T<:Real}  # T is the compute type
    x::NTuple{3, T}   # position
    v::NTuple{3, T}   # velocity
    m::T              # mass
end

# Advance all bodies in the system by one timestep of dt.
@inline function advance!(bodies, dt)
    # Calculate the force on each body due to the other bodies in the system.
    @inbounds for i in 1:lastindex(bodies)-1, j in i+1:lastindex(bodies)
        Δx = bodies[i].x .- bodies[j].x
        distance = sum(abs2, Δx)
        mag = dt * inv(sqrt(distance))^3
        bodies[i] = Body(bodies[i].x, bodies[i].v .- Δx .* (mag * bodies[j].m), bodies[i].m)
        bodies[j] = Body(bodies[j].x, bodies[j].v .+ Δx .* (mag * bodies[i].m), bodies[j].m)
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
    @inbounds for i in 1:lastindex(bodies)-1, j in i+1:lastindex(bodies)
        Δx = bodies[i].x .- bodies[j].x
        e -= bodies[i].m * bodies[j].m / sqrt(sum(Δx .* Δx))
    end
    e
end

simulate!(bodies, n, dt) = for _ = 1:n advance!(bodies, dt) end

function solar_system()
    SOLAR_MASS = 4 * π^2
    DAYS_PER_YEAR = 365.24
    jupiter = Body(( 4.841e+0, -1.160e+0, -1.036e-1), ( 1.660e-3, 7.699e-3, -6.905e-5) .* DAYS_PER_YEAR, 9.547e-4 * SOLAR_MASS)
    saturn = Body(( 8.343e+0, 4.125e+0, -4.035e-1), (-2.767e-3, 4.998e-3, 2.304e-5) .* DAYS_PER_YEAR, 2.858e-4 * SOLAR_MASS)
    uranus = Body(( 1.289e+1, -1.511e+1, -2.23e-1), ( 2.96e-3, 2.378e-3, -2.96e-5) .* DAYS_PER_YEAR, 4.36e-5 * SOLAR_MASS)
    neptune = Body(( 1.537e+1, -2.591e+1, 1.792e-1), ( 2.680e-3, 1.628e-3, -9.515e-5) .* DAYS_PER_YEAR, 5.151e-5 * SOLAR_MASS)
    sun = Body((0.0, 0.0, 0.0), (-1.061e-6, -8.966e-6, 6.553e-8) .* DAYS_PER_YEAR, 3.947e+1)
    return [jupiter, saturn, uranus, neptune, sun]
end

bodies = solar_system()
println(energy(bodies))
@time simulate!(bodies, 50000000, 0.01);
println(energy(bodies))