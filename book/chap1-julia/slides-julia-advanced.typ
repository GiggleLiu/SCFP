#import "@preview/touying:0.4.2": *
#import "@preview/touying-simpl-hkustgz:0.1.0" as hkustgz-theme
#import "@preview/cetz:0.2.2": *
#import "../shared/characters.typ": ina, christina, murphy

#let globalvars = state("t", 0)
#let timecounter(minutes) = [
  #globalvars.update(t => t + minutes)
  #place(dx: 100%, dy: 0%, align(right, text(16pt, red)[#context globalvars.get()min]))
]
#let clip(image, top: 0pt, bottom: 0pt, left: 0pt, right: 0pt) = {
  box(clip: true, image, inset: (top: -top, right: -right, left: -left, bottom: -bottom))
}
#set cite(style: "apa")

#let m = hkustgz-theme.register()

#show raw.where(block: true): it=>{
  block(radius:4pt, fill:gray.transparentize(90%), inset:1em, width:99%, text(it))
}

// Global information configuration
#let m = (m.methods.info)(
  self: m,
  title: [Julia: Advanced Topics],
  subtitle: [],
  author: [Jin-Guo Liu],
  date: datetime.today(),
  institution: [HKUST(GZ) - FUNH - Advanced Materials Thrust],
)

// Extract methods
#let (init, slides) = utils.methods(m)
#show: init

// Extract slide functions
#let (slide, empty-slide, title-slide, outline-slide, new-section-slide, ending-slide) = utils.slides(m)
#show: slides.with()

#outline-slide()

= Create a Julia package
== What is a Julia package?

Julia package management is important, regardless of whether you are a package developer or not.

- Source code
- Unit tests
- Dependencies specification
- CI/CD
- Documentation
- License
- README
- ...

== Case study: OMEinsum.jl

To install OMEinsum

Type `]` in a Julia REPL to enter the package mode (powered by *package manager*).

```julia-repl
(@v1.10) pkg> add OMEinsum
```
The string before `pkg>` is the environment name, and the default value is the Julia version name, which is also known as the *global environment*.

== GitHub based package management

*public repository*: https://github.com/under-Peter/OMEinsum.jl

== The package registry

Registered to the #link("https://github.com/JuliaRegistries/General", "General") registry.

== Package manager resolves the environment with the known registries.

Package manager resolves the environment with the known registries.
```julia-repl
(@v1.10) pkg> registry status
Registry Status 
 [23338594] General (https://github.com/JuliaRegistries/General.git)
```
- Multiple registries can be used
- `General` is the only one that accessible by all Julia users.

== The global package environment
```bash
(base) ➜  .julia cat ~/.julia/environments/v1.10/Project.toml
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
MethodAnalysis = "85b6ec6f-f7df-4429-9514-a64bcd9ee824"
OMEinsum = "ebe7aa44-baf0-506c-a96f-8464559b3922"
...
```

- Packages are identified by UUIDs.

== The file storing *resolved dependencies*:
```bash
cat ~/.julia/environments/v1.10/Manifest.toml
```

== Quiz

Suppose I have `OMEinsum` installed in the global environment, and now I want to experiment with `Yao`. What if they have conflicting dependencies? e.g.
- The latest OMEinsum depends on `A` at version `2` and `B` at version `3`
- `A` at version `2` depends on `C` at version `5`.
- The latest `Yao` depends on `C` at version `4`.

What can I do?

== Install OMEinsum in a local environment

A *local environment* (recommended) can be specified in any folder, which provides a more isolated environment.

```bash
$ mkdir localenv
$ cd localenv
$ julia --project=.
```

Alternatively, environments can be switched in the pkg mode with
```julia-repl
pkg> activate dir-name

pkg> activate    # the global environment

pkg> activate --temp # a temperary environment
```

== The resolved versions
```julia-repl
(localenv) pkg> add OMEinsum

shell> ls
Manifest.toml   Project.toml

shell> cat Project.toml
[deps]
OMEinsum = "ebe7aa44-baf0-506c-a96f-8464559b3922"

shell> cat Manifest.toml
...
```

== Steps to create a package

1. Create a package
2. Specify the dependency
3. Develop the package
4. Open-source the package
5. Register the package

== 1. Create a package

We use #link("https://github.com/JuliaCI/PkgTemplates.jl", "PkgTemplates.jl"). Open a Julia REPL and type the following commands to initialize a new package named `MyFirstPackage`:

#box(text(14pt)[```julia
julia> using PkgTemplates

julia> tpl = Template(;
    user="GiggleLiu",
    authors="GiggleLiu",
    julia=v"1.10",
    plugins=[
        License(; name="MIT"),
        Git(; ssh=true),
        GitHubActions(; x86=true),
        Codecov(),
        Documenter{GitHubActions}(),
    ],
)
```
])

== Unit Tests
- Unit Tests is a software testing method, which consists of:
  - a collection of inputs and expected outputs for a function.
  - *assertion* that the function returns the expected output for a given input.

- Test passing: the function is working as expected (assertion is true), i.e. no feature breaks.
- Test passing: the function is working as expected (assertion is true), i.e. no feature breaks.
- Test coverage: the percentage of the code that is covered by tests, i.e. the higher the coverage, the more *robust* the code is.

== What if I do not have three machines?

- CI/CD (for Continuous Integration/Continuous Deployment) is an *automation process* that runs the unit tests, documentation building, and deployment.

- Workflow: code updated > CI/CD tasks created > virtual machines initialized on CI machines > virtual machines run tests > tests pass and status updated

== How to write tests?

```julia
    user="GiggleLiu",
```
where the username `"GiggleLiu"` should be replaced with your GitHub username.
Many plugins are used in the above example:

```julia
        License(; name="MIT"),
```
- `License`: to choose a license for the package. Here we use the MIT license, which is a permissive free software license.

== Popular open-source licenses

- #link("https://en.wikipedia.org/wiki/MIT_License", "MIT"): a permissive free software license, featured with a short and simple permissive license with conditions only requiring preservation of copyright and license notices.
- #link("https://en.wikipedia.org/wiki/Apache_License", "Apache2"): a permissive free software license, featured with a contributor license agreement and a patent grant.
- #link("https://en.wikipedia.org/wiki/GNU_General_Public_License", "GPL"): a copyleft free software license, featured with a strong copyleft license that requires derived works to be available under the same license.

== Git

```julia
        Git(; ssh=true),
```

- `Git`: to initialize a Git repository for the package. Here we use the SSH protocol for Git for convenience. Using #link("https://docs.github.com/en/authentication/securing-your-account-with-two-factor-authentication-2fa/configuring-two-factor-authentication", "two-factor authentication (2FA)") can make your GitHub account more secure.

== GitHub Actions

```julia
        GitHubActions(; x86=true),
```
- `GitHubActions`: to enable continuous integration (CI) with #link("https://docs.github.com/en/actions", "GitHub Actions").

== Codecov

```julia
        Codecov(),
```

- `Codecov`: to enable code coverage tracking with #link("https://about.codecov.io/", "Codecov"). It is a tool that helps you to measure the test coverage of your code. A package with high test coverage is more reliable.

== Documenter

```julia
        Documenter{GitHubActions}(),
```

- `Documenter`: to enable documentation building and deployment with #link("https://documenter.juliadocs.org/stable/", "Documenter.jl") and #link("https://pages.github.com/", "GitHub pages").


== Create the package

```julia
julia> tpl("MyFirstPackage")
```

After running the above commands, a new directory named `MyFirstPackage` will be created in the folder `~/.julia/dev/` - the default location for Julia packages.

== The file structure of the package

```bash
tree .   
.
├── .git
│   ...
├── .github
│   ├── dependabot.yml
│   └── workflows
│       ├── CI.yml
│       ├── CompatHelper.yml
│       └── TagBot.yml
├── .gitignore
├── LICENSE
├── Manifest.toml
├── Project.toml
├── README.md
├── docs
│   ├── Manifest.toml
│   ├── Project.toml
│   ├── make.jl
│   └── src
│       └── index.md
├── src
│   └── MyFirstPackage.jl
└── test
    └── runtests.jl
```

==
- `.git` and `.gitignore`: the files that are used by Git. The `.gitingore` file contains the files that should be ignored by Git. By default, the `.gitignore` file contains the following lines:
  ```gitignore
  *.jl.*.cov
  *.jl.cov
  *.jl.mem
  /Manifest.toml
  /docs/Manifest.toml
  /docs/build/
  ```

- `.github`: the folder that contains the GitHub Actions configuration files.
- `LICENSE`: the file that contains the license of the package. The MIT license is used in this package.
- `README.md`: the manual that shows up in the GitHub repository of the package, which contains the description of the package.
- `Project.toml`: the file that contains the metadata of the package, including the name, UUID, version, dependencies and compatibility of the package.
- `Manifest.toml`: the file that contains the exact versions of all the packages that are compatible with each other. It is usually automatically resolved from the `Project.toml` file, and it is not recommended pushing it to the remote repository.
- `docs`: the folder that contains the documentation of the package. It has its own `Project.toml` and `Manifest.toml` files, which are used to manage the documentation environment. The `make.jl` file is used to build the documentation and the `src` folder contains the source code of the documentation.

- `src`: the folder that contains the source code of the package.
- `test`: the folder that contains the test code of the package, which contains the main test file `runtests.jl`.

= Manage dependencies
== Specify the dependency
To *add a new dependency*, you can use the following command in the package path:
```bash
$ cd ~/.julia/dev/MyFirstPackage

$ julia --project
```

This will open a Julia REPL in the package environment. To check the package environment, you can type the following commands in the package mode (press `]`) of the REPL:

```julia-repl
(MyFirstPackage) pkg> st
Project MyFirstPackage v1.0.0-DEV
Status `~/.julia/dev/MyFirstPackage/Project.toml` (empty project)
```

==

After that, you can add a new dependency by typing:
```julia-repl
(MyFirstPackage) pkg> add OMEinsum

(MyFirstPackage) pkg> st
Project MyFirstPackage v1.0.0-DEV
Status `~/.julia/dev/MyFirstPackage/Project.toml`
  [ebe7aa44] OMEinsum v0.8.1
```
Press `backspace` to exit the package mode and then type
```julia-repl
julia> using OMEinsum
```
The dependency is added correctly if no error is thrown.

==

Type `;` to enter the shell mode and then type
```julia-repl
shell> cat Project.toml
name = "MyFirstPackage"
uuid = "594718ca-da39-4ff3-a299-6d8961b2aa49"
authors = ["GiggleLiu"]
version = "1.0.0-DEV"

[deps]
OMEinsum = "ebe7aa44-baf0-506c-a96f-8464559b3922"

[compat]
julia = "1.10"

[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test"]
```
You will see that the dependency `OMEinsum` is added to the `[deps]` section of the `Project.toml` file.

==

We also need to specify which version of `OMEinsum` is *compatible* with the current package. To do so, you need to edit the `[compat]` section of the `Project.toml` file with your favorite editor.
```toml
[compat]
julia = "1.10"
OMEinsum = "0.8"
```

Here, we have used the most widely used dependency version specifier `=`, which means matching the first nonzero component of the version number.

==

For example:

- `1` matches `1.0.0`, `1.1.0`, `1.1.1`, but not `2.0.0`.
- `0.8` matches `0.8.0`, `0.8.1`, `0.8.2`, but not `0.9.0` or `0.7.0`.
- `1.2` matches `1.2.0`, `1.3.1`, but not `1.2.0` or `2.0.0`.

==

- whenever an exported function is changed in a package, the first nonzero component of the version number should be increased.
- version number starts with `0` is considered as a development version, and it is not stable.

Please check the Julia documentation about #link("https://pkgdocs.julialang.org/v1/compatibility/", "package compatibility") for advanced usage.

== Develop the package
Developers develop packages in the package environment. The package development process includes:

1. Edit the source code of the package
The source code of the package is located in the `src` folder of the package path.

==

Let us add a simple function to the package. The source code of the package is as follows:

*File*: `src/MyFirstPackage.jl`

```julia
module MyFirstPackage

# using packages

# exported interfaces
export Lorenz, rk4_step
export Point, Point2D, Point3D

# include other source code
include("lorenz.jl")

end
```

==

*File*: `src/lorenz.jl`

```julia
struct Point{D, T <: Real}
    data::NTuple{D, T}
end
const Point2D{T} = Point{2, T}
const Point3D{T} = Point{3, T}
Point(x::Real...) = Point((x...,))
LinearAlgebra.dot(x::Point, y::Point) = mapreduce(*, +, x.data .* y.data)
Base.:*(x::Real, y::Point) = Point(x .* y.data)
Base.:/(y::Point, x::Real) = Point(y.data ./ x)
Base.:+(x::Point, y::Point) = Point(x.data .+ y.data)
Base.isapprox(x::Point, y::Point; kwargs...) = all(isapprox.(x.data, y.data; kwargs...))
Base.getindex(p::Point, i::Int) = p.data[i]
Base.broadcastable(p::Point) = p.data
Base.iterate(p::Point, args...) = iterate(p.data, args...)
```

==

```julia
struct Lorenz
    σ::Float64
    ρ::Float64
    β::Float64
end

function field(p::Lorenz, u)
    x, y, z = u
    Point(p.σ*(y-x), x*(p.ρ-z)-y, x*y-p.β*z)
end
```

== Runge-Kutta 4th order method

```julia
# Runge-Kutta 4th order method
function rk4_step(f, t, y, Δt)
    k1 = Δt * f(t, y)
    k2 = Δt * f(t+Δt/2, y + k1 / 2)
    k3 = Δt * f(t+Δt/2, y + k2 / 2)
    k4 = Δt * f(t+Δt, y + k3)
    return y + k1/6 + k2/3 + k3/3 + k4/6
end

function rk4_step(l::Lorenz, u, Δt)
    return rk4_step((t, u) -> field(l, u), zero(Δt), u, Δt)
end
```

==

To use this function, you can type the following commands in the package environment:
```julia-repl
julia> using MyFirstPackage

julia> MyFirstPackage.greet("Julia")
"Hello, Julia!"
```

==

2. Write tests for the package

We always need to write tests for the package. The test code of the package is located in the `test` folder of the package path.

*File*: `test/runtests.jl`
```julia
using Test
using MyFirstPackage

@testset "lorenz" begin
    include("lorenz.jl")
end

```

== Lorenz attractor

- #link("https://www.youtube.com/watch?v=hIYqkydaMdw", "YouTube: Chaos Theory - the language of (in)stability")

==

*File*: `test/lorenz.jl`
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

==

To run the tests, you can use the following command in the package environment:
```julia-repl
(MyFirstPackage) pkg> test
  ... 
  [8e850b90] libblastrampoline_jll v5.8.0+1
Precompiling project...
  1 dependency successfully precompiled in 1 seconds. 21 already precompiled.
     Testing Running tests...
Test Summary: | Pass  Total  Time
lorenz        |    2      2  0.1s
     Testing MyFirstPackage tests passed
```

Cheers! All tests passed.

==

1. Write documentation for the package

The documentation is built with #link("https://documenter.juliadocs.org/stable/", "Documenter.jl"). The build script is `docs/make.jl`. To *build the documentation*, you can use the following command in the package path:
```bash
$ cd docs
$ julia --project make.jl
```
Instantiate the documentation environment if necessary. For seamless *debugging* of documentation, it is highly recommended using the #link("https://github.com/tlienart/LiveServer.jl", "LiveServer.jl") package.

== 4. Open-source the package
To open-source the package, you need to push the package to a public repository on GitHub.

1. First create a GitHub repository with the same as the name of the package. In this example, the repository name should be `GiggleLiu/MyFirstPackage.jl`. To check the remote repository of the package, you can use the following command in the package path:
   ```bash
   $ git remote -v
   origin	git@github.com:GiggleLiu/MyFirstPackage.jl.git (fetch)
   origin	git@github.com:GiggleLiu/MyFirstPackage.jl.git (push)
   ```

==

2. Then push the package to the remote repository:
   ```bash
   $ git add -A
   $ git commit -m "Initial commit"
   $ git push
   ```

==

3. After that, you need to check if all your GitHub Actions are passing. You can check the status of the GitHub Actions from the badge in the `README.md` file of the package repository. The configuration of GitHub Actions is located in the `.github/workflows` folder of the package path. Its file structure is as follows:
   ```bash
   .github
   ├── dependabot.yml
   └── workflows
       ├── CI.yml
       ├── CompatHelper.yml
       └── TagBot.yml
   ```

==

   - The `CI.yml` file contains the configuration for the CI of the package, which is used to automate the process of
      - *Testing* the package after a pull request is opened, or the main branch is updated. This process can be automated with the #link("https://github.com/julia-actions/julia-runtest", "julia-runtest") action.
      - Building the *documentation* after the main branch is updated. Please check the #link("https://documenter.juliadocs.org/stable/man/hosting/", "Documenter documentation") for more information.

==

   - The `TagBot.yml` file contains the configuration for the #link("https://github.com/JuliaRegistries/TagBot", "TagBot"), which is used to automate the process of tagging a release after a pull request is merged.
   - The `CompatHelper.yml` file contains the configuration for the #link("https://github.com/JuliaRegistries/CompatHelper.jl", "CompatHelper"), which is used to automate the process of updating the `[compat]` section of the `Project.toml` file after a pull request is merged.

==

   Configuring GitHub Actions is a bit complicated. For beginners, it is a good practise to mimic the configuration of another package, e.g. #link("https://github.com/under-Peter/OMEinsum.jl", "OMEinsum.jl").

== 5. Register the package
Package registration is the process of adding the package to the `General` registry. To do so, you need to create a pull request to the `General` registry and wait for the pull request to be reviewed and merged.
This process can be automated by the #link("https://github.com/JuliaRegistries/Registrator.jl", "Julia registrator"). If the pull request meets all guidelines, your pull request will be merged after a few days. Then, your package is available to the public. 

==

A good practice is to *tag a release* after the pull request is merged so that your package version update can be reflected in your GitHub repository. This process can be automated by the #link("https://github.com/JuliaRegistries/TagBot", "TagBot").

==


The file structure of #link("https://github.com/under-Peter/OMEinsum.jl", "OMEinsum.jl")

#image("images/omeinsum.png", width: 60%)

- `build/passing`: the tests executed by GitHub Actions are passing.
- `codecov/89%`: the code coverage is 89%, meaning that 89% of the code is covered by tests.
- `docs/dev`: the documentation is built and deployed with GitHub pages.

==

Now, let's take a look at the file structure of the package by running the following command in the package path (`~/.julia/dev/OMEinsum`):
```bash
$ tree . -L 1 -a
.
├── .git
├── .github
├── .gitignore
├── LICENSE
├── Project.toml
├── README.md
├── benchmark
├── docs
├── examples
├── ext
├── ome-logo.png
├── src
└── test
```

== Hands-on

1. Create a new package named `MyFirstPackage.jl` in your local machine.
2. Add a simple function to the package.
3. Write tests for the package.
4. Build the documentation for the package.
5. Open-source the package to GitHub.


