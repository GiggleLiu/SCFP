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

#let show-graph(vertices, edges, radius:0.2) = {
  import draw: *
  for (k, (i, j)) in vertices.enumerate() {
    circle((i, j), radius:radius, name: str(k), fill:white)
  }
  for (k, l) in edges {
    line(str(k), str(l))
  }
}

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

== Review

- Terminal: where you can run the commands and control the system.
  - Git: version control system.
  - Vim: text editor.
  - SSH: connect to the remote server.
- VSCode: a code editor.
- Julia: a high-level, high-performance programming language for technical computing.
  - Benchmarking and profiling
  - Loops and functions
  - Type system and just-in-time compilation (JIT)
  - Multiple dispatch
  - #text(red)[Package development]
  - #text(red)[High-performance computing]
  - #text(red)[Arrays and some useful functionals]

= Julia ecosystem
== Featured ecosystem
=== Domain-specific ecosystem
- #link("https://github.com/ITensor/ITensors.jl", "ITensors.jl") (570) - Tensor networks
- #link("https://github.com/QuantumBFS/Yao.jl", "Yao.jl") (1k) - Quantum computing
- #link("https://github.com/JuliaMolSim/DFTK.jl", "DFTK.jl") (460) - Density functional theory

=== General-purpose ecosystem
- #link("https://github.com/jump-dev/JuMP.jl", "JuMP.jl") (2.3k) - #highlight[Mathematical optimization]
- #link("https://github.com/SciML/DifferentialEquations.jl", "DifferentialEquations.jl") (2.9k) - #highlight[Solving differential equations]
- #link("https://github.com/MakieOrg/Makie.jl", "Makie.jl") (2.5k) - Data visualization
- #link("https://github.com/Jutho/KrylovKit.jl", "KrylovKit.jl") (333) - Large-scale linear algebra
- #link("https://github.com/JuliaNLSolvers/Optim.jl", "Optim.jl") (1.1k) - Optimization

== Communities

#grid(columns: 2,  gutter: 50pt, [https://julialang.org/community/

- Discourse (for questions): https://discourse.julialang.org/
- Zulip (for discussions): https://julialang.zulipchat.com/
- Slack (for discussions): https://julialang.org/slack/
], [
  #image("images/juliazulip.png", width: 150pt)
])

= Package structure

== Correctness - from programming to software engineering
#timecounter(2)

- Spent enourmous time debugging?
- Can not reproduce previous results?

=== Modern software engineering

#figure(canvas({
  import draw: *
  let s(it) = text(16pt, it)
  content((0, 0), box(stroke: black, inset: 10pt, s[Code + Unit Test]), name: "code")
  content((7, 0), box(stroke: black, inset: 10pt, s[GitHub]), name: "github")
  content((14, 0), box(stroke: black, inset: 10pt, s[CI/CD]), name: "cicd")
  line("code", "github", mark: (end: "straight"), name: "upload")
  content((rel: (0, -0.5), to: "upload.mid"), s[Upload])

  line("github", "cicd", mark: (end: "straight"), name: "trigger")
  content((rel: (0, -0.5), to: "trigger.mid"), s[Trigger])
  bezier("cicd.east", "cicd.north", (rel: (1.0, 1.0)), (rel: (-1.0, 1.0)), mark: (end: "straight"))
  content((rel: (2, 2), to: "cicd"), s[Create an empty virtual machine\ and run the tests])
}))

- _Unit test_: a collection of inputs and expected outputs for a function. It is used to verify the correctness of functions.
- _CI/CD_: Continuous Integration/Continuous Deployment, which is an automation process that runs the unit tests, documentation building, and deployment.
- _Virtual machine_: implemented with the containerization technology.

== Script or package?
#timecounter(1)
- When to write a script? Never.
- When to write a "package"? Always.
  - Simple to use, little overhead
  - Easy to reproduce results (dependency management)
  - Easy to test (automated testing to ensure the correctness)
  - Easy to share and install (can be installed by others)
  - Easy to document and distribute
  - ...

== Case study: TropicalNumbers.jl
#timecounter(2)

- Install `TropicalNumbers.jl` in developer mode
  ```julia
(@v1.11) pkg> activate --temp  # create a temporary environment

(@v1.11) pkg> dev TropicalNumbers  # develop the package
```

- Then you will see a new folder named `TropicalNumbers` in the `~/.julia/dev` folder
  ```bash
  cd ~/.julia/dev/TropicalNumbers
  ```

== The file structure of the package
#timecounter(2)

#box(text(16pt)[```bash
$ tree . -a    # installing `tree` required
.
├── .git              # Git repository
├── .github              # GitHub Actions configuration
├── .gitignore           # Git ignore rules

├── LICENSE           # License of the package
├── Project.toml      # Dependency specification
├── README.md         # Description of the package

├── docs              # Documentation
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
- `test`: the folder that contains the test code of the package
  - contains the main test file `runtests.jl`, which includes multiple test files.
- `docs`: the folder that contains the documentation of the package. (not covered in this course)

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

#image("images/tropicalpackage.png")

- The CI of TropicalNumbers pass, meaning all tests pass. CI resolves the issue that a developer may not have a fresh machine to run the tests.
- The code coverage of TropicalNumbers is 86%, meaning 86% of the code is covered by tests.

== Git related files
#timecounter(1)
- `.git` and `.gitignore`: the files that are used by Git. The `.gitingore` file contains the files that should be ignored by Git. By default, the `.gitignore` file contains the following lines:
  ```gitignore
  *.jl.*.cov
  *.jl.cov         # files ending with `.jl.cov` are coverage files
  *.jl.mem
  .DS_Store
  /Manifest.toml   # Manifest.toml is automatically generated
  /dev/            # the `dev` folder
  ```

- `.github`: the folder that contains the GitHub Actions (a CI/CD system) configuration files.

==
#timecounter(2)

#image("images/github-actions.png")

Let's check the details of tests runs: https://github.com/TensorBFS/TropicalNumbers.jl/actions

== Configuration of CI/CD
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

== Package meta-information
#timecounter(1)
- `LICENSE`: the file that contains the license of the package.
  - #link("https://en.wikipedia.org/wiki/MIT_License", "MIT"): a permissive free software license, featured with a short and simple permissive license with conditions only requiring preservation of copyright and license notices.
  - #link("https://en.wikipedia.org/wiki/Apache_License", "Apache2"): a permissive free software license, featured with a contributor license agreement and a patent grant.
  - #link("https://en.wikipedia.org/wiki/GNU_General_Public_License", "GPL"): a copyleft free software license, featured with a strong copyleft license that requires derived works to be available under the same license.

- `README.md`: the manual that shows up in the GitHub repository of the package, which contains the description of the package.
- `Project.toml`: the file that contains the metadata of the package, including the name, UUID, version, dependencies and compatibility of the package. Package manager resolves the environment with the known registries and generates the `Manifest.toml` file.

== Project.toml
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
  content((0, 7.5), image("images/github.png", width: 2em))
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

= Create a package
== Steps to create a package
#timecounter(1)

1. Create a package template
2. Specify the project dependency
3. Develop the package
4. Upload the package to GitHub
5. (Optional) Register the package to the General registry

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

== 3. Develop the package: Hands on
#timecounter(15)


1. Create a new package named `MyFirstPackage.jl` in your local machine.
2. Implement the shortest path algorithm with TropicalNumbers.
3. Write tests for the package.

== Theory: Graph
#timecounter(2)

Graph $G = (V, E)$, where $V$ is the set of vertices and $E$ is the set of edges.
#align(center, canvas(length:0.9cm, {
  import draw: *
  let vrotate(v, theta) = {
    let (x, y) = v
    return (x * calc.cos(theta) - y * calc.sin(theta), x * calc.sin(theta) + y * calc.cos(theta))
  }

  // petersen graph
  let vertices1 = range(5).map(i=>vrotate((0, 2), i*72/180*calc.pi))
  let vertices2 = range(5).map(i=>vrotate((0, 1), i*72/180*calc.pi))
  let edges = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 7), (6, 8), (7, 9), (8, 5), (9, 6))
  show-graph((vertices1 + vertices2).map(v=>(v.at(0) + 4, v.at(1)+4)), edges, radius:0.2)
}))

#box(text(16pt)[```julia
julia> using Graphs   # Graphs.jl is a package for graph theory

julia> g = smallgraph(:petersen)  # a famous graph for testing
{10, 15} undirected simple Int64 graph
```
])

==
#timecounter(1)

#box(text(16pt)[```julia
julia> vertices(g)
Base.OneTo(10)

julia> [edges(g)...]   # `...` is the splat operator, it expands the edges into a vector
15-element Vector{Graphs.SimpleGraphs.SimpleEdge{Int64}}:
 Edge 1 => 2
 Edge 1 => 5
 Edge 1 => 6
 Edge 2 => 3
 ⋮
 Edge 6 => 9
 Edge 7 => 9
 Edge 7 => 10
 Edge 8 => 10
```
])

Note: `edges(g)` returns an iterator, its elements are `SimpleEdge` objects.

== Adjacency matrix
#timecounter(3)

The adjacency matrix $A in bb(Z)_2^(|V| times |V|)$ is defined as
$
A_(i j) = cases(1\, quad (i,j) in E, 0\, quad "otherwise")  
$

#box(text(14pt)[```julia
julia> adjacency_matrix(g)
10×10 SparseArrays.SparseMatrixCSC{Int64, Int64} with 30 stored entries:
 ⋅  1  ⋅  ⋅  1  1  ⋅  ⋅  ⋅  ⋅
 1  ⋅  1  ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅
 ⋅  1  ⋅  1  ⋅  ⋅  ⋅  1  ⋅  ⋅
 ⋅  ⋅  1  ⋅  1  ⋅  ⋅  ⋅  1  ⋅
 1  ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  1
 1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  1  ⋅
 ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  1
 ⋅  ⋅  1  ⋅  ⋅  1  ⋅  ⋅  ⋅  1
 ⋅  ⋅  ⋅  1  ⋅  1  1  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  1  ⋅  1  1  ⋅  ⋅
```
])

== Shortest path
#timecounter(2)

The shortest path problem is to find the shortest path between two vertices in a graph. The tropical matrix multiplication approach is one of the most efficient ways to solve the shortest path problem.

Hint: Use Tropical matrix multiplication!

$ (A B)_(i k) = min_j (A_(i j) + B_(j k)). $

By powering the adjacency matrix $A$ for $|V|$ times with Min-Plus Tropical algebra, we can get the shortest paths length between any two vertices.
$
  (A^(|V|))_(i j) = min_(k_1, k_2, dots, k_(|V|-1)) (A_(i k_1) + A_(k_1 k_2) + dots + A_(k_(|V|-1) j))
$

== Big $O$ notation
#timecounter(2)

- Let $n$ be the number of vertices.
- The computational complexity of the shortest path problem is $O(n^3 log n)$.
- The complexity of (tropical) matrix multiplication is $O(n^3)$.
  #box(text(16pt)[```julia
for i in 1:n
  for j in 1:n
    for k in 1:n
      A[i, j] = max(A[i, j], A[i, k] + A[k, j])  # number of operations: n^3
    end
  end
end
```
])

Note: Big $O$ notation characterizes the *scaling* of the computational time with respect to the size of the input.

Quiz: Why the complexity of our shortest path algorithm is not $O(n^4)$?

== Run the tests
#timecounter(1)

To run the tests, you can use the following command in the package environment:
```julia-repl
(MyFirstPackage) pkg> test
  ... 
  [8e850b90] libblastrampoline_jll v5.8.0+1
Precompiling project...
  1 dependency successfully precompiled in 1 seconds. 21 already precompiled.
     Testing Running tests...
Test Summary: | Pass  Total  Time
shortest-path |    2      2  0.1s
     Testing MyFirstPackage tests passed
```

Cheers! All tests passed.

== Volunteer
#timecounter(10)

Show-case your test cases.

== Next steps (left as homework)
#timecounter(2)

(TODO: ask Zhongyi if he has a note)

The documentation is built with #link("https://documenter.juliadocs.org/stable/", "Documenter.jl"). The build script is `docs/make.jl`. To *build the documentation*, you can use the following command in the package path:
```bash
$ cd docs
$ julia --project make.jl
```
Instantiate the documentation environment if necessary. For seamless *debugging* of documentation, it is highly recommended using the #link("https://github.com/tlienart/LiveServer.jl", "LiveServer.jl") package.

Configure CI/CD permissions and secrets in the repository.

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
