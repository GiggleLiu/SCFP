#import "../book.typ": book-page, cross-link, heading-reference
#set heading(numbering: "1.")
#show: book-page.with(title: "My First Package")
#let boxed(it, width: 100%) = box(stroke: 1pt, inset: 10pt, radius: 4pt, width: width)[#it]

= My First Package
You need to know the Julia package development workflow, regardless of whether you are writing project code for your own use or developing a package. Julia package system enables developers to manage project dependencies, automate unit tests and documentation, and publish packages.
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

To make your package available to the Julia community:

1. Ensure your package meets these requirements:
   - All tests pass
   - Documentation is complete
   - Version compatibility is specified
   - License is included

2. Register with the General registry using Registrator.jl:
   - Create a release on GitHub
   - The Registrator bot will create a PR in JuliaRegistries/General
   - Wait for review and merge

3. After registration:
   - TagBot will automatically create a GitHub release
   - Your package will be available via Julia's package manager
   - Users can install it with `] add MyFirstPackage`

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

Key features:
- 89% test coverage
- Comprehensive documentation
- Automated CI/CD pipeline
- Extension system for optional features

The `Project.toml` shows advanced package management:

```toml
[weakdeps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

[extensions]
CUDAExt = "CUDA"

[compat]
AbstractTrees = "0.3, 0.4"
CUDA = "4, 5"
julia = "1"
```

* Excercises *
1. Is ChainRulesCore 1.2 compatible with OMEinsum?
2. What should be done when ChainRulesCore 2.0 is released?
3. How should bug fixes be released?
4. What version changes are needed for API changes?