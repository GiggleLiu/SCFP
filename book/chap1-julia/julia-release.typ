#import "../book.typ": book-page, cross-link, heading-reference
#show: book-page.with(title: "My First Package")
#let boxed(it, width: 100%) = box(stroke: 1pt, inset: 10pt, radius: 4pt, width: width)[#it]

#show ref: it => {
  let el = it.element
  if el != none and el.func() == heading {
    // Override heading references.
    link(el.location(), el.body)
  } else {
    // Other references as usual.
    it
  }
}

= My First Package
Julia package development workflow is useful for both project code and package development. It enables developers to manage project dependencies, automate unit tests and documentation, and publish packages.
The Julia development workflow usually includes the following steps:

- @sec:create-package: Create a new package using #link("https://github.com/JuliaCI/PkgTemplates.jl", "PkgTemplates").
- @sec:configure-dependencies: Configure project dependencies.
- @sec:develop-package: Develop the package (code, tests, docs).
- @sec:publish-package: Upload the package to GitHub or other git hosting services.
- @sec:register-package: (Optional) Register the package in the #link("https://github.com/JuliaRegistries/General", "General") registry.

== Create a package <sec:create-package>

We use #link("https://github.com/JuliaCI/PkgTemplates.jl", "PkgTemplates") to generate a new package. In the Julia REPL:

```julia
julia> using PkgTemplates

julia> tpl = Template(;
    user="YourUsername",  # Replace with your GitHub username
    authors="Your Name",
    julia=v"1",
    plugins=[
        License(; name="MIT"),       # MIT license
        Git(; ssh=true),             # Use Git protocol
        GitHubActions(; x86=false),  # Set up GitHub Actions
        Codecov(),                   # Enable code coverage tracking
        Documenter{GitHubActions}(), # Configure documentation building
    ],
)

julia> tpl("MyFirstPackage")     # Create a new package named MyFirstPackage
```
The above template includes several useful plugins:
- `License`: Specifies the package license (MIT in this case)
- `Git`: Initializes Git repository with SSH protocol
- `GitHubActions`: Sets up continuous integration
- `Codecov`: Enables code coverage tracking
- `Documenter`: Configures documentation building

#boxed([
  *Tips: package naming convention*\
  - Longer than 5 characters
  - Camel case, e.g. `MyFirstPackage`
])

After running the above code, you will get a new package named `MyFirstPackage` in the `~/.julia/dev` directory. The template creates the following directory structure:

```bash
.
├── .git/              # Git repository files
├── .github/           
│   ├── dependabot.yml
│   └── workflows/     # CI/CD configuration
│       ├── CI.yml
│       ├── CompatHelper.yml
│       └── TagBot.yml
├── .gitignore         # Files ignored by Git
├── LICENSE            # Package license
├── Project.toml       # Package metadata and dependencies
├── README.md          # Package documentation
├── docs/              # Documentation source
│   ├── Project.toml
│   ├── make.jl
│   └── src/
│       └── index.md
├── src/               # Package source code
│   └── MyFirstPackage.jl
└── test/              # Test files
    └── runtests.jl
```

Key files and directories:

- `.github/`: Contains GitHub Actions workflows for:
  - `CI.yml`: Continuous integration testing
  - `CompatHelper.yml`: Manages dependency compatibility
  - `TagBot.yml`: Automates release tagging
- `Project.toml`: Defines package metadata, dependencies, and version constraints
- `docs/`: Documentation files with their own dependency environment
- `src/`: Source code directory
- `test/`: Test files directory

== Manage dependencies <sec:configure-dependencies>

To add dependencies to your package:

1. Navigate to the package directory:
```bash
cd ~/.julia/dev/MyFirstPackage
julia --project
```

2. Check the current environment:
```julia
(@v1.10) pkg> st
Project MyFirstPackage v1.0.0-DEV
Status `~/.julia/dev/MyFirstPackage/Project.toml` (empty project)
```

3. Add dependencies:
```julia
(@v1.10) pkg> add LinearAlgebra

(@v1.10) pkg> st
Project MyFirstPackage v1.0.0-DEV
Status `~/.julia/dev/MyFirstPackage/Project.toml`
  [37e2e46d] LinearAlgebra
```

The `Project.toml` file will now contain:

```toml
name = "MyFirstPackage"
uuid = "594718ca-da39-4ff3-a299-6d8961b2aa49"
authors = ["Your Name"]
version = "1.0.0-DEV"

[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[compat]
julia = "1.10"

[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test"]
```

=== Specifying Version Compatibility

Add version constraints in the `[compat]` section:

```toml
[compat]
julia = "1.10"
LinearAlgebra = "1"  # Matches all 1.x.y versions
```

Version specifier rules:
- `1`: Matches `1.0.0`, `1.1.0`, `1.2.0`, etc.
- `0.8`: Matches `0.8.0`, `0.8.1`, `0.8.2`, etc.
- `1.2`: Matches `1.2.0`, `1.2.1`, etc.

The version numbers starting with `0` indicate development versions. The first non-zero component should be incremented when making breaking changes to exported functions.

== Develop the package <sec:develop-package>

=== Write source code

Let's create a package that implements the Lorenz system. First, edit the main module file:

_File_: `src/MyFirstPackage.jl`
```julia
module MyFirstPackage

using LinearAlgebra

# Export public interfaces
export Lorenz, integrate_step
export Point, Point2D, Point3D
export RungeKutta, Euclidean

include("lorenz.jl")

end
```

Then create the implementation file:

_File_: `src/lorenz.jl`
```julia
"""
    Point{D, T}

A point in D-dimensional space, with coordinates of type T.
"""
struct Point{D, T <: Real}
    data::NTuple{D, T}
end

Point(x::Real...) = Point((x...,))
const Point2D{T} = Point{2, T}
const Point3D{T} = Point{3, T}

# Vector operations
LinearAlgebra.dot(x::Point, y::Point) = mapreduce(*, +, x.data, y.data)
Base.:*(x::Real, y::Point) = Point(x .* y.data)
Base.:/(y::Point, x::Real) = Point(y.data ./ x)
Base.:+(x::Point, y::Point) = Point(x.data .+ y.data)
Base.isapprox(x::Point, y::Point; kwargs...) = all(isapprox.(x.data, y.data; kwargs...))

# Collection interface
Base.getindex(p::Point, i::Int) = p.data[i]
Base.broadcastable(p::Point) = p.data
Base.iterate(p::Point, args...) = iterate(p.data, args...)

# Lorenz system implementation
struct Lorenz
    σ::Float64
    ρ::Float64
    β::Float64
end

function field(p::Lorenz, u)
    x, y, z = u
    Point(p.σ*(y-x), x*(p.ρ-z)-y, x*y-p.β*z)
end

# Integration methods
abstract type AbstractIntegrator end
struct RungeKutta{K} <: AbstractIntegrator end
struct Euclidean <: AbstractIntegrator end

function integrate_step(f, ::RungeKutta{4}, t, y, Δt)
    k1 = Δt * f(t, y)
    k2 = Δt * f(t+Δt/2, y + k1 / 2)
    k3 = Δt * f(t+Δt/2, y + k2 / 2)
    k4 = Δt * f(t+Δt, y + k3)
    return y + k1/6 + k2/3 + k3/3 + k4/6
end

function integrate_step(f, ::Euclidean, t, y, Δt)
    return y + Δt * f(t, y)
end

function integrate_step(lz::Lorenz, int::AbstractIntegrator, u, Δt)
    return integrate_step((t, u) -> field(lz, u), int, zero(Δt), u, Δt)
end
```

=== Writing Tests

Create a test suite in two files:

_File_: `test/runtests.jl`
```julia
using Test
using MyFirstPackage

@testset "lorenz" begin
    include("lorenz.jl")
end
```

_File_: `test/lorenz.jl`
```julia
using Test, MyFirstPackage

@testset "Point" begin
    p1 = Point(1.0, 2.0)
    p2 = Point(3.0, 4.0)
    @test p1 + p2 ≈ Point(4.0, 6.0)
end

@testset "step" begin
    lz = Lorenz(10.0, 28.0, 8/3)
    int = RungeKutta{4}()
    r1 = integrate_step(lz, int, Point(1.0, 1.0, 1.0), 0.0001)
    eu = Euclidean()
    r2 = integrate_step(lz, eu, Point(1.0, 1.0, 1.0), 0.0001)
    @test isapprox(r1, r2; rtol=1e-5)
end
```

Run tests in the package environment:
```julia
(@v1.10) pkg> test
```

=== Creating Examples

Create an example that visualizes the Lorenz attractor:

_File_: `examples/lorenz.jl`
```julia
using CairoMakie, MyFirstPackage
set_theme!(theme_black())

# Initialize system
lz = Lorenz(10, 28, 8/3)
y = Point(1.0, 1.0, 1.0)
 
points = Observable(Point3f[])
colors = Observable(Int[])

# Create visualization
fig, ax, l = lines(points, color = colors,
    colormap = :inferno, transparency = true, 
    axis = (; type = Axis3, protrusions = (0, 0, 0, 0), 
              viewmode = :fit, limits = (-30, 30, -30, 30, 0, 50)))

# Generate animation
record(fig, "lorenz.mp4", 1:120) do frame
    for i in 1:50
        y = integrate_step(lz, RungeKutta{4}(), y, 0.01)
        push!(points[], Point3f(y...))
        push!(colors[], frame)
    end
    ax.azimuth[] = 1.7pi + 0.3 * sin(2pi * frame / 120)
    notify(points); notify(colors)
    l.colorrange = (0, frame)
end
```

#figure(
  image("images/lorenz.gif", width: 80%),
  caption: [Visualization of the Lorenz attractor]
)

== Document the package <sec:document-package>

Documentation is built using Documenter.jl. To build the documentation:

1. Navigate to the docs directory and start Julia:
  ```bash
cd docs
julia --project
```

2. Set up the documentation environment:
  ```julia
(@v1.10) pkg> dev ..        # Add local package
(@v1.10) pkg> instantiate   # Install dependencies
```

3. Build the documentation:
  ```julia
julia> include("make.jl")
```

  The generated HTML files will be in `docs/build/`. To preview the documentation during development, one can use #link("https://github.com/tlienart/LiveServer.jl", "LiveServer.jl"):
  ```julia
julia> using MyFirstPackage, LiveServer

julia> servedocs()
```

== Publish the package <sec:publish-package>

=== Create a GitHub repository

1. Create a new repository named `MyFirstPackage.jl` on GitHub

2. Verify your remote repository configuration:
  ```bash
git remote -v
origin git@github.com:YourUsername/MyFirstPackage.jl.git (fetch)
origin git@github.com:YourUsername/MyFirstPackage.jl.git (push)
```

3. Push your code:
  ```bash
git add -A
git commit -m "Initial commit"
git push
```

=== Set up GitHub Actions

The `.github/workflows` directory contains three important configuration files:

1. `CI.yml`: Handles continuous integration
   - Runs tests on pull requests and main branch updates
   - Builds and deploys documentation
   
2. `TagBot.yml`: Automates release tagging
   - Creates GitHub releases when versions are registered
   - Updates package version tags

3. `CompatHelper.yml`: Manages dependency compatibility
   - Monitors dependency updates
   - Creates PRs to update version bounds

#boxed([
  *Tips: Learn from established packages*\
  It is always a good practise to learn from established packages. For GitHub Actions configuration examples, #link("https://github.com/under-Peter/OMEinsum.jl", "OMEinsum.jl") is a good example.
])

== Register the package <sec:register-package>

To make your package accessible to the Julia community with `] add MyFirstPackage`, you need to register it in the #link("https://github.com/JuliaRegistries/General", "General") registry. This can be done automataically with #link("https://github.com/JuliaRegistries/Registrator.jl", "Registrator.jl").

== Case study: OMEinsum.jl

OMEinsum.jl provides a good example of package organization:

```
.
├── .github/          # CI/CD configuration
├── Project.toml      # Package metadata
├── benchmark/        # Performance tests
├── docs/             # Documentation
├── examples/         # Usage examples
├── ext/              # Package extensions
├── src/              # Source code
└── test/             # Test suite
```

The `Project.toml` shows the dependencies and version compatibility:

```toml
name = "OMEinsum"
uuid = "ebe7aa44-baf0-506c-a96f-8464559b3922"
authors = ["Andreas Peter <andreas.peter.ch@gmail.com>"]
version = "0.8.4"

[deps]
AbstractTrees = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
BatchedRoutines = "a9ab73d0-e05c-5df1-8fde-d6a4645b8d8e"
ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
Combinatorics = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MacroTools = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
OMEinsumContractionOrders = "6f22d1fd-8eed-4bb7-9776-e7d684900715"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
TupleTools = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"

[weakdeps]
AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[extensions]
AMDGPUExt = "AMDGPU"
CUDAExt = "CUDA"
EnzymeExt = "Enzyme"

[compat]
AMDGPU = "0.8"
AbstractTrees = "0.3, 0.4"
BatchedRoutines = "0.2"
CUDA = "4, 5"
ChainRulesCore = "1"
Combinatorics = "1.0"
Enzyme = "0.13.16"
MacroTools = "0.5"
OMEinsumContractionOrders = "0.9"
TupleTools = "1.2, 1.3"
julia = "1"

[extras]
Documenter = "e30172f5-a6a5-5a46-863b-614d45cd2de4"
DoubleFloats = "497a8b3b-efae-58df-a0af-a86822472b78"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LuxorGraphPlot = "1f49bdf2-22a7-4bc4-978b-948dc219fbbc"
Polynomials = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
ProgressMeter = "92933f4c-e287-5a05-a399-4b506db050ca"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
TropicalNumbers = "b3a74e9c-7526-4576-a4eb-79c0d4c32334"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[targets]
test = ["Test", "CUDA", "Documenter", "Enzyme", "LinearAlgebra", "LuxorGraphPlot", "ProgressMeter", "SymEngine", "Random", "Zygote", "DoubleFloats", "TropicalNumbers", "ForwardDiff", "Polynomials"]

```

* Quiz: *
1. Is ChainRulesCore 1.2 compatible with OMEinsum?
2. If ChainRulesCore 2.0 is released, what should be done to keep OMEinsum compatible?
3. If I fixed the bug in OMEinsum, how to release a new version? How is it different compared with when I first release the package?