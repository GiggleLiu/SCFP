#import "@preview/touying:0.6.1": *
#import "@preview/touying-simpl-hkustgz:0.1.2": *
#import "@preview/cetz:0.4.1": canvas, draw
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm
#import "../shared/characters.typ": ina, christina, murphy

#let globalvars = state("t", 0)
#let timecounter(minutes) = [
  #globalvars.update(t => t + minutes)
  #place(dx: 100%, dy: -5%, align(right, text(16pt, red)[#context globalvars.get()min]))
]
#let clip(image, top: 0pt, bottom: 0pt, left: 0pt, right: 0pt) = {
  box(clip: true, image, inset: (top: -top, right: -right, left: -left, bottom: -bottom))
}
#set cite(style: "apa")

#let show-graph(vertices, edges, radius:0.2) = {
  import draw: *
  for (k, (i, j)) in vertices.enumerate() {
    circle((i, j), radius:radius, name: str(k), fill:white)
  }
  for (k, l) in edges {
    line(str(k), str(l))
  }
}

#show raw.where(block: true): it=>{
  block(radius:4pt, fill:gray.transparentize(90%), inset:1em, width:99%, text(it))
}

#show: hkustgz-theme.with(
  config-info(
    title: [Julia: Advanced topics],
    subtitle: [],
    author: [Jin-Guo Liu],
    date: datetime.today(),
    institution: [HKUST(GZ) - FUNH - Advanced Materials Thrust],
  ),
)

#title-slide()
#outline-slide()

== Tower of software quality

#figure(canvas({
  import draw: *
  let s(it) = text(18pt, it)
  line((0, 0), (-8, -8), (8, -8), close: true)
  line((-2, -2), (2, -2))
  line((-4, -4), (4, -4))
  line((-6, -6), (6, -6))
  content((0, -7), highlight(s[Correctness & Robustness]))
  content((0, -5), highlight(s[Reproducibility & Reliability]))
  content((0, -3), s[Efficiency (JIT)])
  content((6, -1), s[Extensibility, easy to use et al.])
}))

== No script

#align(center, text(36pt, strike[Script]))

== This lecture

We focus on correctness:
- Reproducibable environment: package *versioning* and local environment
- Metric of correctness: *unit test* and test coverage
- Consistency: setup *CI/CD* to automate the testing and deployment

= Unit test and CI/CD

// == Julia software ecosystem
// #timecounter(2)
// === General-purpose ecosystem
// - #link("https://github.com/jump-dev/JuMP.jl", "JuMP.jl") (2.3k) - #highlight[Mathematical optimization, e.g. LP, SDP, MIP etc.]
// - #link("https://github.com/SciML/DifferentialEquations.jl", "DifferentialEquations.jl") (2.9k) - #highlight[Solving differential equations]
// - #link("https://github.com/MakieOrg/Makie.jl", "Makie.jl") (2.5k) - Data visualization
// - #link("https://github.com/Jutho/KrylovKit.jl", "KrylovKit.jl") (333) - Large-scale sparse linear algebra
// - #link("https://github.com/JuliaNLSolvers/Optim.jl", "Optim.jl") (1.1k) - Optimization, i.e. finding the minimum of a function

// === Domain-specific ecosystem
// - #link("https://github.com/ITensor/ITensors.jl", "ITensors.jl") (570) - Tensor networks
// - #link("https://github.com/QuantumBFS/Yao.jl", "Yao.jl") (1k) - Quantum computing
// - #link("https://github.com/JuliaMolSim/DFTK.jl", "DFTK.jl") (460) - Density functional theory


== Correctness, correctness and correctness!
#timecounter(2)

Many open-source developers maintain the codebase with $>10k$ lines of code. e.g. the quantum simulator `Yao.jl`:

#box(text(16pt)[```bash
➜  Yao git:(master) ✗ cloc .
Julia                          245           4777           6108          18198
```
])

- Spent enourmous time debugging?
- A piece of code was working yesterday, but not today?
- A piece of code was working on your machine, but not on your colleague's?

#align(center, box(stroke: black, inset: 10pt, text(16pt)[Discussion: what is your practise to ensure the correctness of your code?]))

== Modern software engineering

#figure(canvas({
  import draw: *
  let s(it) = text(16pt, it)
  content((-1, 0), box(stroke: black, inset: 10pt, width: 10em,  align(center, s[Unit Test
```julia
@test sin(π/2) ≈ 1
```
  ])), name: "code")
  content((7, 0), box(stroke: black, inset: 10pt, s[GitHub]), name: "github")
  content((14, 0), box(stroke: black, inset: 10pt, s[CI/CD]), name: "cicd")
  line("code", "github", mark: (end: "straight"), name: "upload")
  content((rel: (0, -0.5), to: "upload.mid"), s[Upload])

  line("github", "cicd", mark: (end: "straight"), name: "trigger")
  content((rel: (0, -0.5), to: "trigger.mid"), s[Trigger])
  bezier("cicd.south", "github.south", (rel: (-1.0, -1.0)), (rel: (-2.0, -2.0)), mark: (end: "straight"))
  content((rel: (0, 1.5), to: "cicd"), s[Create an empty virtual machine\ and run the tests])
  content((rel: (-3, -2.5), to: "cicd"), s[Pass or fail?])
}))

- _Unit test_: a collection of inputs and expected outputs for a function. It is used to verify the correctness of functions.
- _CI/CD_: Continuous Integration/Continuous Deployment, which is an automation process that runs the unit tests, documentation building, and deployment.

// == Script or package?
// #timecounter(1)
// - When to write a script? Never.
// - When to write a "package"? Always.
//   - Simple to use, little overhead
//   - Easy to reproduce results (dependency management)
//   - Easy to test (automated testing to ensure the correctness)
//   - Easy to share and install (can be installed by others)
//   - Easy to document and distribute
//   - ...

== Live code: Run tests of TropicalNumbers.jl
#timecounter(4)

- Install `TropicalNumbers.jl` in developer mode
  #box(text(16pt)[```julia
(@v1.11) pkg> activate --temp  # create a temporary environment

(jl_shdToL) pkg> dev TropicalNumbers  # develop the package
```
  ])

- Then you will see a new folder named `TropicalNumbers` in the `~/.julia/dev` folder
  #box(text(16pt)[```bash
  cd ~/.julia/dev/TropicalNumbers
  julia --project
  ```
  ])
  To run tests in `test/runtests.jl`:
  #box(text(16pt)[```julia
  pkg> test
  ```
  ])

== Unit Tests
#timecounter(1)
- Unit Tests is a software testing method, which consists of:
  - a collection of inputs and expected outputs for a function.
  - "unit" means the test is for the smallest unit of the code, which is different from the _system tests_ that test the whole system.

- Test passing: the function is working as expected (assertion is true), i.e. no feature breaks. Otherwise, the test will fail.
- Test coverage: the percentage of the code that is covered by tests, i.e. the higher the coverage, the more *robust* the code is.

==
#timecounter(1)

#box(text(13pt)[```julia
julia> @testset "tropical max plus" begin
           @test Tropical(3) * Tropical(4) == Tropical(7)  # pass
           @test Tropical(3) * Tropical(4) == Tropical(4)  # fail
       end
tropical max plus: Test Failed at REPL[2]:3
  Expression: Tropical(3) * Tropical(4) == Tropical(4)
   Evaluated: 7ₜ == 4ₜ

Stacktrace:
 [1] macro expansion
   @ ~/.julia/juliaup/julia-1.11.3+0.aarch64.apple.darwin14/share/julia/stdlib/v1.11/Test/src/Test.jl:679 [inlined]
 [2] macro expansion
   @ REPL[2]:3 [inlined]
 [3] macro expansion
   @ ~/.julia/juliaup/julia-1.11.3+0.aarch64.apple.darwin14/share/julia/stdlib/v1.11/Test/src/Test.jl:1704 [inlined]
 [4] top-level scope
   @ REPL[2]:2
Test Summary:     | Pass  Fail  Total  Time
tropical max plus |    1     1      2  0.0s
ERROR: Some tests did not pass: 1 passed, 1 failed, 0 errored, 0 broken.
```
])

==
#timecounter(1)

#image("images/tropicalpackage.png", width: 80%, alt: "TropicalNumbers.jl")

- The CI of #link("https://github.com/TensorBFS/TropicalNumbers.jl", "TropicalNumbers.jl") pass, meaning all tests pass. CI resolves the issue that a developer may not have a fresh machine to run the tests.
- The code coverage of TropicalNumbers is 86%, meaning 86% of the code is covered by tests.

== Live demo: GitHub Actions
#timecounter(2)

#image("images/github-actions.png", alt: "GitHub Actions")

Let's check the details of tests runs: https://github.com/TensorBFS/TropicalNumbers.jl/actions

== Live demo: Configuration of CI/CD
#timecounter(2)

CI/CD are configured in the `.github/workflows` folder of the package path.
Each file is a CI/CD configuration in the `.yml` format.

```bash
.github
├── dependabot.yml
└── workflows
    ├── ci.yml        # automated testing and test coverage
    └── TagBot.yml    # update the version of the package after a release
```
- Note: normally, you do not need to change the configuration files since the `PkgTemplates.jl` has already configured the CI/CD for you.
- Note: the results of the CI/CD are reflected in the badge in the `README.md` file.
== Configure the dependency: Project.toml
#timecounter(1)
`Project.toml`: the file that contains the metadata of the package, including the name, UUID, version, dependencies and compatibility of the package. Package manager resolves the environment with the known registries and generates the `Manifest.toml` file.

Usually, we put `Manifest.toml` in the `.gitignore` file.

== Live demo: Project.toml
#timecounter(1)

#box(text(16pt)[```bash
$ cat Project.toml
name = "TropicalNumbers"
uuid = "b3a74e9c-7526-4576-a4eb-79c0d4c32334"  # unique identifier of the package
authors = ["GiggleLiu"]
version = "0.6.2"  # version of the package

[compat]
julia = "1"

[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
Documenter = "e30172f5-a6a5-5a46-863b-614d45cd2de4"

[targets]
test = ["Test", "Documenter"]
```
])

== Compatibility
#timecounter(2)

- Format: "`major.minor.patch`", major version "0" means the package is not very stable.

#box(text(16pt)[```toml
[compat]
AMDGPU = "0.8"  # matches `0.8.0`, `0.8.1`, `0.8.2`, but not `0.9.0` or `0.7.0`.
CUDA = "4, 5"   # matches `4.0.0`, `4.1.0`, `4.2.0`, `5.0.0`, but not `3.0.0` or `6.0.0`.
ChainRulesCore = "1.2.1"  # matches `1.2.1`, `1.3.1`, but not `1.2.0` or `2.0.0`.
julia = "1"
```
])

- The *first nonzero number* must exactly match.
- The *rest nonzero numbers* specifies the lower bound.
- Multiple versions are separated by "`,`".

== Resolving dependencies
#timecounter(2)
```bash
$ julia --project   # open the package environment
```
Press `]` to enter the package mode and then type
```julia-repl
(TropicalNumbers) pkg> instantiate
```
to see the `Manifest.toml` file. The `Manifest.toml` file is generated by the package manager when the package is installed. It contains the exact versions of all the packages that are compatible with each other.
```bash
$ cat  Manifest.toml
```

== The global package environment
#timecounter(1)
```bash
$ cat ~/.julia/environments/v1.11/Project.toml
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
MethodAnalysis = "85b6ec6f-f7df-4429-9514-a64bcd9ee824"
OMEinsum = "ebe7aa44-baf0-506c-a96f-8464559b3922"
...
```

Note: it is not recommended to install all packages in the global environment. It may cause _version conflicts_.


==
#timecounter(1)
#figure(canvas({
  import draw: *
  let s(it) = text(18pt, it)
  content((0, 0), box(stroke: black, inset: 10pt, radius: 4pt, align(center, s[Registry])), name: "General")
  content((0, 5), box(stroke: black, inset: 10pt, radius: 4pt, align(center, s[Developers\  (Package)])), name: "Developer")
  content((12, 0), box(stroke: black, inset: 10pt, radius: 4pt, s[User]), name: "User")
  line("General", "Developer", mark: (start: "straight"), name: "reg")
  content((rel:(2.5, 0), to: "reg.mid"), s[Register package\ Version update])
  line("Developer.east", "User", mark: (end: "straight"), name: "download")
  line("General", "User", mark: (start: "straight", end: "straight"), name: "query")
  content((rel:(0, -0.5), to: "query.mid"), s[1. Sync])
  content((rel:(2.4, 0), to: "download.mid"), s[3. Download])
  content((rel: (0.0, -1), to: "User"), [`] instantiate` or `]update`])
  rect((-3, -1), (3, 6.5), stroke: (dash: "dashed"))
  content((0, 7.5), image("images/github.png", width: 2em, alt: "GitHub"))
  bezier("User.east", "User.north", (rel: (1.0, 1.0)), (rel: (-1.0, 1.0)), mark: (end: "straight"))
  content((rel: (4.0, 1.5), to: "User"), s[2. Resolve versions])
}))

== Package registries
#timecounter(1)

Package manager resolves the environment with the known registries.
```julia
(@v1.10) pkg> registry status
Registry Status 
 [23338594] General (https://github.com/JuliaRegistries/General.git)
```
- Multiple registries can be used
- `General` is the only one that accessible by all Julia users.


= Develop a package
== File structure of a package
#timecounter(2)

#box(text(16pt)[```bash
$ tree . -a    # installing `tree` required
.
├── .git              # Git repository
├── .github           # GitHub Actions configuration
├── .gitignore        # Git ignore rules

├── LICENSE           # License of the package
├── Project.toml      # Dependency specification
├── README.md         # Description of the package

├── docs              # Documentation (not covered in this course)
├── src               # Source code
└── test              # Test code
```
])
Note: To install `tree` command, use `brew install tree` on macOS or `sudo apt install tree` on Ubuntu/WSL.

== Package content
#timecounter(2)
- `src`: the folder that contains the source code of the package.
  - `src/TropicalNumbers.jl`: the main source code of the package. It contains a module named `TropicalNumbers`. It includes multiple other files with the `include` statement.
    #box(text(16pt)[```julia
    module TropicalNumbers
      using Package-1, Package-2, ...  # import other packages
      export API-1, API-2...           # export the API
      include("file-1.jl")             # include other files
      include("file-2.jl")            
      ...
    end
    ```
    ])


== Package meta-information
#timecounter(1)
- `LICENSE`: the file that contains the license of the package.
  - #link("https://en.wikipedia.org/wiki/MIT_License", "MIT"): a permissive free software license, featured with a short and simple permissive license with conditions only requiring preservation of copyright and license notices.
  - #link("https://en.wikipedia.org/wiki/Apache_License", "Apache2"): a permissive free software license, featured with a contributor license agreement and a patent grant.
  - #link("https://en.wikipedia.org/wiki/GNU_General_Public_License", "GPL"): a copyleft free software license, featured with a strong copyleft license that requires derived works to be available under the same license.

= Live demo: Create a package
== 1. Create a package
#timecounter(2)

We use #link("https://github.com/JuliaCI/PkgTemplates.jl", "PkgTemplates.jl"). Open a Julia REPL and type the following commands to initialize a new package named `MyFirstPackage`:

#box(text(14pt)[```julia
julia> using PkgTemplates

julia> tpl = Template(;
    user="GiggleLiu",  # My GitHub username
    authors="GiggleLiu",
    julia=v"1",
    plugins=[
        License(; name="MIT"),  # Use MIT license
        Git(; ssh=true),        # Use SSH protocol for Git rather than HTTPS
        GitHubActions(; x86=false),   # Use GitHub Actions for CI, do not setup x86 machine for testing
        Codecov(),                    # Use Codecov for code coverage
        Documenter{GitHubActions}(),  # Use Documenter for documentation
    ],
)
```
])

== Comments on the plugins
#timecounter(2)
- `Git(; ssh=true)`: Use SSH protocol for Git rather than HTTPS. Using HTTPS with #link("https://docs.github.com/en/authentication/securing-your-account-with-two-factor-authentication-2fa/configuring-two-factor-authentication", "two-factor authentication (2FA)") for `Git` is more secure.
- `GitHubActions(; x86=false)`: Enable continuous integration (CI) with #link("https://docs.github.com/en/actions", "GitHub Actions").
- `Codecov`: to enable code coverage tracking with #link("https://about.codecov.io/", "Codecov"). It is a tool that helps you to measure the test coverage of your code. A package with high test coverage is more reliable.
- `Documenter`: to enable documentation building and deployment with #link("https://documenter.juliadocs.org/stable/", "Documenter.jl") and #link("https://pages.github.com/", "GitHub pages") (a static site deployment service).


== Create the package
#timecounter(1)

```julia
julia> tpl("MyFirstPackage")
```

After running the above commands, a new directory named `MyFirstPackage` will be created in the folder `~/.julia/dev/` - the default location for Julia packages.

== 2. Specify the dependency
#timecounter(1)
To *add a new dependency*, you can use the following command in the package path:
```bash
$ cd ~/.julia/dev/MyFirstPackage

$ julia --project
```

This will open a Julia REPL in the package environment. To check the package environment, you can type the following commands in the package mode (press `]`) of the REPL:

```julia
(MyFirstPackage) pkg> st
Project MyFirstPackage v1.0.0-DEV
Status `~/.julia/dev/MyFirstPackage/Project.toml` (empty project)
```

==
#timecounter(1)

After that, you can add a new dependency by typing:
#box(text(16pt)[```julia
(MyFirstPackage) pkg> add TropicalNumbers

(MyFirstPackage) pkg> st   # check the dependency
Project MyFirstPackage v1.0.0-DEV
Status `~/.julia/dev/MyFirstPackage/Project.toml`
  [b3a74e9c] TropicalNumbers v0.6.2
```
])
Press `backspace` to exit the package mode and then type
#box(text(16pt)[```julia
julia> using TropicalNumbers
```
])
The dependency is added correctly if no error is thrown.

==
#timecounter(1)

Type `;` to enter the shell mode and then type
#box(text(12pt)[```julia
shell> cat Project.toml
name = "MyFirstPackage"
uuid = "8402d085-69c7-4601-8e57-9b55d4db68af"
authors = ["GiggleLiu"]
version = "1.0.0-DEV"

[deps]
TropicalNumbers = "b3a74e9c-7526-4576-a4eb-79c0d4c32334"   # new

[compat]
TropicalNumbers = "0.6.2"  # new
julia = "1"

[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test"]
```
])

== Hands on: Develop the package
#timecounter(15)

https://scfp.jinguo-group.science/chap1-julia/julia-release.html

1. Create a new package named `MyFirstPackage.jl` in your local machine.
2. Implement the Lorenz simulation.
3. Write tests for the package.
4. Answer the following questions:
  - Estimate the number of FLOPs.
  - Benchmark and profile the performance, and share the results with the class.

== Homework

https://github.com/CodingThrust/AMAT5315-2025Spring-Homeworks/tree/main/hw3

== Get help from communities
#timecounter(3)

#grid(columns: 2,  gutter: 50pt, [https://julialang.org/community/

- Discourse (for questions): https://discourse.julialang.org/
- Zulip (for discussions): https://julialang.zulipchat.com/
- Slack (for discussions): https://julialang.org/slack/
], [
  #image("images/juliazulip.png", width: 150pt, alt: "Julia Zulip")
])


// == Theory: Graph
// #timecounter(2)

// Graph $G = (V, E)$, where $V$ is the set of vertices and $E$ is the set of edges.
// #align(center, canvas(length:0.9cm, {
//   import draw: *
//   let vrotate(v, theta) = {
//     let (x, y) = v
//     return (x * calc.cos(theta) - y * calc.sin(theta), x * calc.sin(theta) + y * calc.cos(theta))
//   }

//   // petersen graph
//   let vertices1 = range(5).map(i=>vrotate((0, 2), i*72/180*calc.pi))
//   let vertices2 = range(5).map(i=>vrotate((0, 1), i*72/180*calc.pi))
//   let edges = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 7), (6, 8), (7, 9), (8, 5), (9, 6))
//   show-graph((vertices1 + vertices2).map(v=>(v.at(0) + 4, v.at(1)+4)), edges, radius:0.2)
// }))

// #box(text(16pt)[```julia
// julia> using Graphs   # Graphs.jl is a package for graph theory

// julia> g = smallgraph(:petersen)  # a famous graph for testing
// {10, 15} undirected simple Int64 graph
// ```
// ])

// ==
// #timecounter(1)

// #box(text(16pt)[```julia
// julia> vertices(g)
// Base.OneTo(10)

// julia> [edges(g)...]   # `...` is the splat operator, it expands the edges into a vector
// 15-element Vector{Graphs.SimpleGraphs.SimpleEdge{Int64}}:
//  Edge 1 => 2
//  Edge 1 => 5
//  Edge 1 => 6
//  Edge 2 => 3
//  ⋮
//  Edge 6 => 9
//  Edge 7 => 9
//  Edge 7 => 10
//  Edge 8 => 10
// ```
// ])

// Note: `edges(g)` returns an iterator, its elements are `SimpleEdge` objects.

// == Adjacency matrix
// #timecounter(3)

// The adjacency matrix $A in bb(Z)_2^(|V| times |V|)$ is defined as
// $
// A_(i j) = cases(1\, quad (i,j) in E, 0\, quad "otherwise")  
// $

// #box(text(14pt)[```julia
// julia> adjacency_matrix(g)
// 10×10 SparseArrays.SparseMatrixCSC{Int64, Int64} with 30 stored entries:
//  ⋅  1  ⋅  ⋅  1  1  ⋅  ⋅  ⋅  ⋅
//  1  ⋅  1  ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅
//  ⋅  1  ⋅  1  ⋅  ⋅  ⋅  1  ⋅  ⋅
//  ⋅  ⋅  1  ⋅  1  ⋅  ⋅  ⋅  1  ⋅
//  1  ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  1
//  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  1  ⋅
//  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  1
//  ⋅  ⋅  1  ⋅  ⋅  1  ⋅  ⋅  ⋅  1
//  ⋅  ⋅  ⋅  1  ⋅  1  1  ⋅  ⋅  ⋅
//  ⋅  ⋅  ⋅  ⋅  1  ⋅  1  1  ⋅  ⋅
// ```
// ])

// == Shortest path
// #timecounter(2)

// The shortest path problem is to find the shortest path between two vertices in a graph. The tropical matrix multiplication approach is one of the most efficient ways to solve the shortest path problem.

// Hint: Use Tropical matrix multiplication!

// $ (A B)_(i k) = min_j (A_(i j) + B_(j k)). $

// By powering the adjacency matrix $A$ for $|V|$ times with Min-Plus Tropical algebra, we can get the shortest paths length between any two vertices.
// $
//   (A^(|V|))_(i j) = min_(k_1, k_2, dots, k_(|V|-1)) (A_(i k_1) + A_(k_1 k_2) + dots + A_(k_(|V|-1) j))
// $

// == Big $O$ notation
// #timecounter(2)

// - Let $n$ be the number of vertices.
// - The computational complexity of the shortest path problem is $O(n^3 log n)$.
// - The complexity of (tropical) matrix multiplication is $O(n^3)$.
//   #box(text(16pt)[```julia
// for i in 1:n
//   for j in 1:n
//     for k in 1:n
//       A[i, j] = max(A[i, j], A[i, k] + A[k, j])  # number of operations: n^3
//     end
//   end
// end
// ```
// ])

// Note: Big $O$ notation characterizes the *scaling* of the computational time with respect to the size of the input.

// Quiz: Why the complexity of our shortest path algorithm is not $O(n^4)$?

// == Run the tests
// #timecounter(1)

// To run the tests, you can use the following command in the package environment:
// ```julia-repl
// (MyFirstPackage) pkg> test
//   ... 
//   [8e850b90] libblastrampoline_jll v5.8.0+1
// Precompiling project...
//   1 dependency successfully precompiled in 1 seconds. 21 already precompiled.
//      Testing Running tests...
// Test Summary: | Pass  Total  Time
// shortest-path |    2      2  0.1s
//      Testing MyFirstPackage tests passed
// ```

// Cheers! All tests passed.

// == Volunteer
// #timecounter(10)

// Show-case your test cases.

// == Onliner's implementation
// #timecounter(2)

// #box(text(16pt)[```julia
// using TropicalNumbers, LinearAlgebra, Graphs
// tmat = (map(x->iszero(x) ? zero(TropicalMinPlus{Float64}) : TropicalMinPlus(1.0), adjacency_matrix(smallgraph(:petersen))) + Diagonal(fill(TropicalMinPlus(0.0), 10)))^10
// ```
// ])
// - "`map`": apply a function to each element of the input array.
// - "`Diagonal`": create a diagonal matrix from a vector.
// - "`x -> ...`": a anonymous function that takes an argument `x` and returns a value.
// - "`expression ? branch 1 : branch 2`": a ternary operator that returns `branch 1` if `expression` is true, otherwise returns `branch 2`.
// - "`fill(x, n)`": create a vector of length `n` with all elements being `x`.

// Note: `zero(TropicalMinPlus{Float64})` is `Inf`.

// == Next steps (left as homework)
// #timecounter(2)

// (TODO: ask Zhongyi if he has a note)

// The documentation is built with #link("https://documenter.juliadocs.org/stable/", "Documenter.jl"). The build script is `docs/make.jl`. To *build the documentation*, you can use the following command in the package path:
// ```bash
// $ cd docs
// $ julia --project make.jl
// ```
// Instantiate the documentation environment if necessary. For seamless *debugging* of documentation, it is highly recommended using the #link("https://github.com/tlienart/LiveServer.jl", "LiveServer.jl") package.

// Configure CI/CD permissions and secrets in the repository.

// == Memory access

// #canvas({
//   import draw: *
//   let s(it) = text(12pt, white, it)
//   let t(it) = text(12pt, black, it)
//   let cpu(loc, size, color: gray.darken(50%)) = {
//     let  r = 0.8
//     let space = 0.2
//     for i in range(4) {
//       line((loc.at(0) - r * size, loc.at(1) - (i - 1.5) * space * size), (loc.at(0) + r * size, loc.at(1) - (i - 1.5) * space * size), stroke: (paint: color, thickness: 2pt))
//       line((loc.at(0) - (i - 1.5) * space * size, loc.at(1) - r * size), (loc.at(0) - (i - 1.5) * space * size, loc.at(1) + r * size), stroke: (paint: color, thickness: 2pt))
//     }
//     rect((loc.at(0) - size / 2, loc.at(1) - size / 2), (loc.at(0) + size / 2, loc.at(1) + size / 2), stroke: color, fill: color, radius: 0.1 * size)
//     content(loc, s[CPU])
//   }
//   cpu((-1, 0), 1)
//   on-yz(x: 1, {
//     rect((-0.6, -0.7), (0.6, 0.7), stroke: none, fill: green.darken(50%))
//     content((0, 0), s[$mat(1; 1)$])
//   })
//   on-yz(x: 2, {
//     rect((-1, -1), (1, 1), stroke: none, fill: green.darken(50%))
//     content((0, 0), s[$mat(1;1; 2;2)$])
//   })
//   rect((1.8, -0.0), (2.2, 0.8), stroke: (dash: "dotted", paint: white))
//   on-yz(x: 3.5, {
//     rect((-1.5, -1.5), (1.5, 1.5), stroke: none, fill: green.darken(50%))
//     content((0, 0), s[$mat(1,3; 1,2; 2,-4; 2, 3)$])
//   })
//   rect((3, -0.8), (3.4, 0.8), stroke: (dash: "dotted", paint: white))
//   content((1, 1), t[$L_1$])
//   content((2, 1.5), t[$L_2$])
//   content((3.5, 2), t[$L_3$])
//   content((6, 0), t[$mat(1,3,1,-2; 1,2,0, 5; 2,-4,-2,1; 2,3,1,-2)$])
//   rect((4.9, -0.8), (5.9, 0.8), stroke: (dash: "dotted"))
// })

// - A typical CPU clock cycle is: 0.3 ns.
// - A typical memory access latency is: 50 ns, i.e. $~100$ times slower!

// == Example: Triangular Lattice
// #timecounter(2)

// Generate a triangular lattice using broadcasting:

// ```julia
// # Basis vectors
// b1 = [1, 0]
// b2 = [0.5, sqrt(3)/2]
// n = 5

// # Two equivalent approaches
// mesh1 = [i * b1 + j * b2 for i in 1:n, j in 1:n]  # List comprehension
// mesh2 = (1:n) .* Ref(b1) .+ (1:n)' .* Ref(b2)     # Broadcasting

// # Visualization with CairoMakie
// using CairoMakie
// scatter(vec(getindex.(mesh2, 1)), vec(getindex.(mesh2, 2)))
// ```

// This demonstrates Julia's concise syntax for mathematical operations.

// == Hands-on: Rigid body simulation
// #timecounter(20)

// 1. Check the case study: Hamiltonian dynamics at the bottom of this page: https://scfp.jinguo-group.science/chap1-julia/julia-basic.html . Create a local project folder, copy-paste the program into a local file: `nbody.jl`. Open the project with VSCode.
// 2. Use `@benchmark` to benchmark the performance of the program, and `Profile` to profile the program. Save the benchmark and profile results to a markdown file.
// 3. Remove the type annotation of the field `m` of the `Body` type, and compare the performance of the original and the modified versions.
//   ```julia
// struct Body{T <: Real}
//     x::NTuple{3, T}
//     v::NTuple{3, T}
//     m   # remove the type annotation
// end
// ```

// == Walk through the code
// #timecounter(5)

// Defining a type with `struct`:

// ```julia
// struct Body{T <: Real}
//     x::NTuple{3,T}
//     v::NTuple{3,T}
//     m::T
// end
// ```

// - `::`: type declaration
// - `<:`: subtype
// - `NTuple{3,T}`: a tuple of 3 elements of type `T`, tuples are immutable and faster.
// - `T`: the type parameter name

// == Function and loops
// #timecounter(2)

// #box(text(16pt)[```julia
// function simulate!(bodies::Vector{Body{T}}, n::Int, dt::T) where T
//     # Advance velocities by half a timestep
//     step_velocity!(bodies, dt/2)
//     # Advance positions and velocities by one timestep
//     for _ = 1:n
//         step_position!(bodies, dt)
//         step_velocity!(bodies, dt)
//     end
//     # Advance velocities backwards by half a timestep
//     step_velocity!(bodies, -dt/2)
// end
// ```
// ])

// - `!` is part of function name, it is a convention for _in-place operations_.
// - `Vector{Body{T}}`: a vector of `Body{T}` type, vectors are _mutable_.
// - `where T`: infer the type parameter `T` from the argument.

// == The Hamiltonian Dynamics
// #timecounter(2)
// In the Hamiltonian dynamics simulation, we have the following equation of motion:
// $ m (partial^2 bold(x))/(partial t^2) = bold(f)(bold(x)). $

// Equivalently, by denoting $bold(v) = (partial bold(x))/(partial t)$, we have the first-order differential equations:
// $
// cases(m (partial bold(v))/(partial t) &= bold(f)(bold(x)),
// (partial bold(x))/(partial t) &= bold(v))
// $

// == The Verlet Algorithm
// #timecounter(3)
// It is a typical Hamiltonian dynamics, which can be solved numerically by the Verlet algorithm @Verlet1967. The algorithm is as follows:

// #algorithm({
//   import algorithmic: *
//   Function("Verlet", args: ([$bold(x)$], [$bold(v)$], [$bold(f)$], [$m$], [$d t$], [$n$]), {
//     Assign([$bold(v)$], [$bold(v) + (bold(f)(bold(x)))/m d t \/ 2$ #h(2em)#Ic([update velocity at time $d t \/ 2$])])
//     For(cond: [$i = 1 dots n$], {
//       Cmt[time step $t = i d t$]
//       Assign([$bold(x)$], [$bold(x) + bold(v) d t$ #h(2em)#Ic([update position at time $t$])])
//       Assign([$bold(v)$], [$bold(v) + (bold(f)(bold(x)))/m d t$ #h(2em)#Ic([update velocity at time $t + d t\/2$])])
//     })
//     Assign([$bold(v)$], [$bold(v) - (bold(f)(bold(x)))/m d t \/ 2$ #h(2em)#Ic([velocity at time $n d t$])])
//     Return[$bold(x)$, $bold(v)$]
//   })
// })

// The Verlet algorithm is a simple yet robust algorithm for solving the differential equation of motion. It is the most widely used algorithm in molecular dynamics simulation.

// == Broadcasting
// #timecounter(2)

// #box(text(16pt)[```julia
// function step_velocity!(bodies::Vector{Body{T}}, dt::T) where T
//     # Calculate the force on each body due to the other bodies in the system.
//     @inbounds for i in 1:lastindex(bodies)-1, j in i+1:lastindex(bodies)
//         Δx = bodies[i].x .- bodies[j].x 
//         distance = sum(abs2, Δx)
//         mag = dt * inv(sqrt(distance))^3   # `^` is power operator
//         bodies[i] = Body(bodies[i].x, bodies[i].v .- Δx .* (mag * bodies[j].m), bodies[i].m)
//         bodies[j] = Body(bodies[j].x, bodies[j].v .+ Δx .* (mag * bodies[i].m), bodies[j].m)
//     end
// end
// ```
// ])

// - `bodies[i].x`: access the `x` field of the `i`-th element of `bodies`.
// - "`.`": broadcast operator, apply the operation element-wise.
// - `sum(abs2, Δx)`: apply `abs2` to each element of `Δx`, and then sum the results.
// - `@inbounds`: a macro, skip the bounds check for the loop.

// == Step position
// #timecounter(1)

// #box(text(16pt)[```julia
// function step_position!(bodies::Vector{Body{T}}, dt::T) where T
//     @inbounds for i in eachindex(bodies)
//         bi = bodies[i]
//         bodies[i] = Body(bi.x .+ bi.v .* dt, bi.v, bi.m)
//     end
// end
// ```
// ])

// == Total energy of the system
// #timecounter(2)

// #box(text(16pt)[```julia
// function energy(bodies::Vector{Body{T}}) where T
//     e = zero(T)
//     # Kinetic energy of bodies
//     @inbounds for b in bodies
//         e += T(0.5) * b.m * sum(abs2, b.v)
//     end
    
//     # Potential energy between body i and body j
//     @inbounds for i in 1:lastindex(bodies)-1, j in i+1:lastindex(bodies)
//         Δx = bodies[i].x .- bodies[j].x
//         e -= bodies[i].m * bodies[j].m / sqrt(sum(abs2, Δx))
//     end
//     return e
// end
// ```
// ])
// - `zero(T)`: return a zero value of type `T`.
// - `T(0.5)`: convert the value `0.5` to type `T`.

// == Main simulation - avoid using global variables!
// #timecounter(2)

// #box(text(12pt)[```julia
// function solar_system()
//     SOLAR_MASS = 4 * π^2
//     DAYS_PER_YEAR = 365.24
//     jupiter = Body((4.841e+0, -1.160e+0, -1.036e-1),
//         ( 1.660e-3, 7.699e-3, -6.905e-5) .* DAYS_PER_YEAR,
//         9.547e-4 * SOLAR_MASS)
//     saturn = Body((8.343e+0, 4.125e+0, -4.035e-1),
//         (-2.767e-3, 4.998e-3, 2.304e-5) .* DAYS_PER_YEAR,
//         2.858e-4 * SOLAR_MASS)
//     uranus = Body((1.289e+1, -1.511e+1, -2.23e-1),
//         ( 2.96e-3, 2.378e-3, -2.96e-5) .* DAYS_PER_YEAR,
//         4.36e-5 * SOLAR_MASS)
//     neptune = Body((1.537e+1, -2.591e+1, 1.792e-1),
//         ( 2.680e-3, 1.628e-3, -9.515e-5) .* DAYS_PER_YEAR,
//         5.151e-5 * SOLAR_MASS)
//     sun = Body((0.0, 0.0, 0.0),
//         (-1.061e-6, -8.966e-6, 6.553e-8) .* DAYS_PER_YEAR,
//         SOLAR_MASS)
//     return [jupiter, saturn, uranus, neptune, sun]
// end
// ```
// ])
// Because global variables are not type stable, since they can be changed at any time.

// == Main simulation
// #timecounter(1)
// ```julia
// bodies = solar_system()
// @info "Initial energy: $(energy(bodies))"
// @time simulate!(bodies, 50000000, 0.01);
// @info "Final energy: $(energy(bodies))"
// ```
// - `@info`: print the message and the value of the variable. similar functions/macros are `print`, `println`, `display`, `show`, `@warn`, `@error`, `@debug`, `@show`, etc.
// - `$`: interpolate the value of the variable.
// - `@time`: time the execution of the code.

// == Type stability
// #timecounter(1)

// ```julia
// julia> @code_warntype step_velocity!(bodies, 0.01)
// ```

== Video Watching

- High Performance in Dynamic Languages (Steven Johnson):
 https://www.youtube.com/watch?v=6JcMuFgnA6U&ab_channel=MITOpenCourseWare

// #bibliography("refs.bib")