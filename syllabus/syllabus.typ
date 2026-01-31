#set document(title: "SCFP: Scientific Computing for Physicists - Syllabus", author: "Jin-Guo Liu")
#set page(margin: (x: 2.5cm, y: 2.5cm))
#set text(size: 11pt)
#set heading(numbering: "1.1")
#set par(justify: true)

#align(center)[
  #text(size: 24pt, weight: "bold")[Scientific Computing for Physicists]

  #text(size: 14pt)[Course Syllabus - 13 Week Program]

  #v(0.5em)

  #text(size: 11pt, style: "italic")[Jin-Guo Liu et al.]
]

#v(1em)

#outline(indent: auto, depth: 2)

#pagebreak()

= Course Overview

== Description

This course provides a comprehensive introduction to scientific computing with applications in physics. Students will learn modern programming practices using the Julia programming language, numerical methods for linear algebra and optimization, and simulation techniques essential for computational physics research.

The course emphasizes hands-on coding experience, with each topic accompanied by practical exercises and real-world physics applications.

== Prerequisites

- Basic programming experience (any language)
- Linear algebra fundamentals (vectors, matrices, eigenvalues)
- Calculus (derivatives, integrals, gradients)
- Basic physics knowledge (mechanics, thermodynamics)

== Learning Outcomes

Upon successful completion of this course, students will be able to:

+ Write efficient, well-documented Julia code following best practices
+ Apply numerical linear algebra techniques to solve physics problems
+ Implement and analyze optimization algorithms for complex systems
+ Use automatic differentiation for gradient-based computations
+ Perform Monte Carlo simulations and analyze statistical results
+ Leverage GPU computing for high-performance applications
+ Develop reproducible scientific software with proper version control

== Assessment Structure

#table(
  columns: (auto, 1fr, auto),
  align: (center, left, center),
  [*Component*], [*Description*], [*Weight*],
  [Weekly Labs], [Hands-on coding exercises], [30%],
  [Module Projects], [4 mini-projects (end of each module)], [40%],
  [Final Project], [Capstone project with presentation], [20%],
  [Participation], [In-class exercises, peer review], [10%],
)

== Course Materials

- *Primary:* Online textbook at #link("https://scfp.jinguo-group.science")[scfp.jinguo-group.science]
- *Software:* Julia 1.10+, VS Code with Julia extension, Git
- *Supplementary:* Package documentation, research papers (provided per topic)

#pagebreak()

= Module 1: Programming Foundations (Weeks 1-3)

== Week 1: Development Environment

#block(inset: (left: 1em))[
  *Theme:* _Setting up the scientific computing workstation_
]

=== Learning Objectives

By the end of this week, students will be able to:

+ Navigate the terminal using essential shell commands (cd, ls, mkdir, cp, mv, rm)
+ Configure SSH keys for secure remote access
+ Initialize and manage Git repositories with proper commit practices
+ Use branching and merging workflows for collaborative development
+ Write clear commit messages and maintain a clean git history
+ Understand the difference between local and remote repositories

=== Topics Covered

- Terminal basics: file system navigation, permissions, environment variables
- Shell scripting fundamentals
- Git workflow: init, add, commit, push, pull, branch, merge
- GitHub/GitLab: pull requests, issues, code review
- SSH configuration and key management

=== Readings

- `terminal.typ` - Terminal Environment
- `git.typ` - Version Control

=== Lab Exercises

+ Set up development environment with Julia, VS Code, and Git
+ Create a personal GitHub repository with README and LICENSE
+ Practice Git workflow: create branches, make commits, merge with conflicts
+ Configure SSH access to a remote server

#v(1em)
#line(length: 100%, stroke: 0.5pt + gray)

== Week 2: Julia Fundamentals I

#block(inset: (left: 1em))[
  *Theme:* _Thinking in Julia - types, functions, and the REPL_
]

=== Learning Objectives

By the end of this week, students will be able to:

+ Use the Julia REPL effectively (help mode, shell mode, package mode)
+ Define and use primitive types, arrays, and custom structs
+ Write functions with multiple dispatch
+ Apply control flow structures (if/else, for, while, comprehensions)
+ Understand Julia's type system and its role in performance
+ Debug code using print statements and the debugger

=== Topics Covered

- Julia REPL modes and workflow
- Primitive types: Int, Float, Bool, Char, String
- Collections: Array, Tuple, NamedTuple, Dict, Set
- Functions: positional args, keyword args, varargs, anonymous functions
- Multiple dispatch and method specialization
- Control flow and iteration patterns

=== Readings

- `julia-setup.typ` - Setup Julia
- `julia-basic.typ` (Sections 1-4) - Julia Basic

=== Lab Exercises

+ Implement basic mathematical functions (factorial, fibonacci, prime check)
+ Create a struct representing a physical system (e.g., Particle with position, velocity)
+ Write functions operating on custom types with multiple dispatch
+ Solve Project Euler problems #1-5 in Julia

#v(1em)
#line(length: 100%, stroke: 0.5pt + gray)

== Week 3: Julia Fundamentals II

#block(inset: (left: 1em))[
  *Theme:* _Building reliable scientific software_
]

=== Learning Objectives

By the end of this week, students will be able to:

+ Create and structure Julia packages with proper layout
+ Write unit tests using the Test standard library
+ Generate documentation with Documenter.jl
+ Set up continuous integration (CI) for automated testing
+ Use the package manager to manage dependencies
+ Profile code and identify performance bottlenecks

=== Topics Covered

- Package structure: src/, test/, docs/, Project.toml
- Module system: export, import, using
- Unit testing: `@test`, `@testset`, test coverage
- Documentation: docstrings, Documenter.jl, doctests
- CI/CD: GitHub Actions for Julia packages
- Performance: `@time`, `@benchmark`, profiling, type stability

=== Readings

- `julia-basic.typ` (Sections 5-7) - Julia Basic
- `julia-release.typ` - My First Package

=== Lab Exercises

+ Create a Julia package with proper structure
+ Write comprehensive tests achieving >80% coverage
+ Set up GitHub Actions for automated testing
+ Profile and optimize a slow function

=== Module 1 Project

*Create a Physics Utilities Package*

Build a Julia package that includes:
- Custom types for physical quantities with units
- Functions for common physics calculations
- Complete test suite and documentation
- CI pipeline with automated testing

#pagebreak()

= Module 2: Numerical Linear Algebra (Weeks 4-6)

== Week 4: Matrix Computation Basics

#block(inset: (left: 1em))[
  *Theme:* _The language of linear algebra in code_
]

=== Learning Objectives

By the end of this week, students will be able to:

+ Perform efficient matrix operations using BLAS and LAPACK
+ Implement and apply LU decomposition for solving linear systems
+ Use QR decomposition for least squares problems
+ Apply Singular Value Decomposition (SVD) for data analysis
+ Understand memory layout and its impact on performance
+ Choose appropriate matrix factorizations for different problems

=== Topics Covered

- Matrix storage: column-major order, stride, views
- BLAS levels: Level 1 (vector), Level 2 (matrix-vector), Level 3 (matrix-matrix)
- LU decomposition with pivoting
- QR decomposition: Gram-Schmidt, Householder
- SVD: theory, computation, applications
- Condition number and numerical stability

=== Readings

- `linalg.typ` - Matrix Computation

=== Lab Exercises

+ Benchmark matrix multiplication: naive vs BLAS
+ Implement LU decomposition from scratch, compare with built-in
+ Solve overdetermined systems using QR and normal equations
+ Image compression using truncated SVD

#v(1em)
#line(length: 100%, stroke: 0.5pt + gray)

== Week 5: Advanced Matrix Methods

#block(inset: (left: 1em))[
  *Theme:* _Eigenvalues and iterative methods for large systems_
]

=== Learning Objectives

By the end of this week, students will be able to:

+ Compute eigenvalues and eigenvectors using power iteration
+ Apply the QR algorithm for eigenvalue computation
+ Use Krylov subspace methods (Arnoldi, Lanczos) for large matrices
+ Implement conjugate gradient for symmetric positive definite systems
+ Understand convergence properties of iterative methods
+ Choose between direct and iterative solvers based on problem characteristics

=== Topics Covered

- Eigenvalue problems in physics (quantum mechanics, vibrations)
- Power iteration and inverse iteration
- QR algorithm for eigenvalues
- Krylov subspaces and Arnoldi iteration
- Lanczos algorithm for symmetric matrices
- Conjugate gradient method
- Preconditioning strategies

=== Readings

- `linalg-advanced.typ` - Matrix Computation (Advanced Topics)

=== Lab Exercises

+ Implement power iteration for dominant eigenvalue
+ Find vibrational modes of a molecular system
+ Compare direct vs iterative solvers for varying matrix sizes
+ Implement preconditioned conjugate gradient

#v(1em)
#line(length: 100%, stroke: 0.5pt + gray)

== Week 6: Sparse Matrices & Graphs

#block(inset: (left: 1em))[
  *Theme:* _Exploiting structure for efficiency_
]

=== Learning Objectives

By the end of this week, students will be able to:

+ Store and manipulate sparse matrices in CSR/CSC formats
+ Perform efficient sparse matrix operations
+ Represent graphs as sparse adjacency matrices
+ Implement graph algorithms (BFS, DFS, shortest path)
+ Apply PageRank and other network analysis algorithms
+ Identify when sparse representations provide advantages

=== Topics Covered

- Sparse matrix formats: COO, CSR, CSC, advantages of each
- SparseArrays.jl: construction, operations, conversions
- Graph representations: adjacency matrix, adjacency list
- Graph traversal: breadth-first, depth-first search
- Shortest paths: Dijkstra, Bellman-Ford
- PageRank algorithm and power iteration on graphs
- Applications: network analysis, finite element methods

=== Readings

- `sparse.typ` - Sparse Matrices and Graphs

=== Lab Exercises

+ Convert dense matrix to sparse, compare memory and operations
+ Implement BFS/DFS on a graph represented as sparse matrix
+ Compute PageRank for a web graph dataset
+ Solve a 2D Laplacian system using sparse methods

=== Module 2 Project

*Sparse Linear Solver for Physics*

Implement a sparse linear solver package:
- Support for multiple sparse formats
- Iterative solver (CG or GMRES)
- Preconditioner implementation
- Benchmark on physics problems (e.g., heat equation)

#pagebreak()

= Module 3: Optimization (Weeks 7-9)

== Week 7: Combinatorial Optimization

#block(inset: (left: 1em))[
  *Theme:* _Finding needles in exponentially large haystacks_
]

=== Learning Objectives

By the end of this week, students will be able to:

+ Formulate combinatorial optimization problems
+ Implement the Metropolis-Hastings algorithm
+ Apply simulated annealing with appropriate cooling schedules
+ Analyze convergence and mixing properties
+ Design problem-specific move proposals
+ Balance exploration vs exploitation in search algorithms

=== Topics Covered

- Combinatorial optimization: TSP, graph coloring, spin glasses
- Local search and hill climbing
- Metropolis algorithm and detailed balance
- Simulated annealing: temperature schedules, acceptance probability
- Parallel tempering for improved sampling
- Applications: protein folding, circuit design, scheduling

=== Readings

- `simulated-annealing.typ` - Simulated Annealing

=== Lab Exercises

+ Implement Metropolis algorithm for 2D Ising model
+ Solve traveling salesman problem with simulated annealing
+ Compare different cooling schedules
+ Apply to a graph coloring problem

#v(1em)
#line(length: 100%, stroke: 0.5pt + gray)

== Week 8: Mathematical Programming

#block(inset: (left: 1em))[
  *Theme:* _Optimization with structure and constraints_
]

=== Learning Objectives

By the end of this week, students will be able to:

+ Formulate problems as linear programs (LP)
+ Solve LPs using the simplex method and interior point methods
+ Model integer constraints and solve ILPs
+ Use JuMP.jl for optimization modeling
+ Apply convex optimization techniques
+ Recognize and exploit problem structure (sparsity, decomposition)

=== Topics Covered

- Linear programming: standard form, duality
- Simplex method: basic feasible solutions, pivoting
- Interior point methods
- Integer linear programming: branch and bound
- Mixed-integer programming
- JuMP.jl: modeling language, solvers (HiGHS, GLPK, Gurobi)
- Convex optimization basics

=== Readings

- `linear_integer.typ` - Mathematical Optimization

=== Lab Exercises

+ Model and solve a resource allocation problem
+ Implement branch and bound for a small ILP
+ Formulate a physics problem as optimization (minimum energy configuration)
+ Compare solver performance on benchmark problems

#v(1em)
#line(length: 100%, stroke: 0.5pt + gray)

== Week 9: Gradient-Based Optimization

#block(inset: (left: 1em))[
  *Theme:* _Following the gradient to optimal solutions_
]

=== Learning Objectives

By the end of this week, students will be able to:

+ Implement gradient descent with various step size strategies
+ Apply momentum and adaptive learning rate methods (Adam, RMSprop)
+ Use Newton's method and quasi-Newton methods (BFGS, L-BFGS)
+ Handle constraints with projected gradient and barrier methods
+ Diagnose and fix convergence issues
+ Choose appropriate optimization algorithms for different problems

=== Topics Covered

- Gradient descent: batch, stochastic, mini-batch
- Line search: Armijo, Wolfe conditions
- Momentum methods: classical momentum, Nesterov
- Adaptive methods: AdaGrad, RMSprop, Adam
- Second-order methods: Newton, Gauss-Newton
- Quasi-Newton: BFGS, L-BFGS
- Constrained optimization: projected gradient, augmented Lagrangian

=== Readings

- `gradient-optimization.typ` - Gradient-based Optimization

=== Lab Exercises

+ Implement gradient descent with line search
+ Compare convergence of SGD, momentum, and Adam
+ Fit a nonlinear model using L-BFGS
+ Optimize a constrained physics problem

=== Module 3 Project

*Optimization Challenge*

Solve a complex optimization problem:
- Formulate a physics optimization problem (e.g., structure optimization, parameter fitting)
- Implement multiple solution approaches
- Compare performance and solution quality
- Write a report analyzing the results

#pagebreak()

= Module 4: Advanced Topics (Weeks 10-12)

== Week 10: Automatic Differentiation

#block(inset: (left: 1em))[
  *Theme:* _Exact gradients without the pain_
]

=== Learning Objectives

By the end of this week, students will be able to:

+ Explain the difference between numerical, symbolic, and automatic differentiation
+ Implement forward-mode AD using dual numbers
+ Understand reverse-mode AD (backpropagation) for efficient gradient computation
+ Use Zygote.jl and ForwardDiff.jl for automatic gradients
+ Apply AD to optimize differentiable physics simulations
+ Debug and handle non-differentiable operations

=== Topics Covered

- Differentiation methods comparison
- Forward-mode AD: dual numbers, computational cost O(n)
- Reverse-mode AD: computational graph, adjoint method, O(1) for scalar output
- Zygote.jl: pullbacks, custom adjoints
- ForwardDiff.jl: efficient forward-mode
- Enzyme.jl: LLVM-level AD
- Differentiable programming paradigm
- Applications: neural networks, physics-informed ML

=== Readings

- `ad.typ` - Automatic Differentiation

=== Lab Exercises

+ Implement forward-mode AD with dual numbers
+ Compute gradients of a loss function with Zygote
+ Implement a custom adjoint rule
+ Train a simple neural network from scratch using AD

#v(1em)
#line(length: 100%, stroke: 0.5pt + gray)

== Week 11: Tensor Networks & Data Compression

#block(inset: (left: 1em))[
  *Theme:* _Efficient representations for high-dimensional data_
]

=== Learning Objectives

By the end of this week, students will be able to:

+ Represent tensors and perform tensor operations (contraction, decomposition)
+ Apply tensor decompositions: CP, Tucker, Tensor Train
+ Understand Matrix Product States (MPS) for quantum systems
+ Implement basic tensor network algorithms
+ Apply information-theoretic concepts to data compression
+ Use tensor networks for dimensionality reduction

=== Topics Covered

- Tensor basics: indices, contraction, einsum notation
- Tensor decompositions: CP (CANDECOMP/PARAFAC), Tucker
- Tensor Train / Matrix Product State (MPS)
- Tensor network diagrams
- DMRG algorithm for ground state search
- Information theory: entropy, mutual information
- Compressed sensing and sparse recovery
- Applications: quantum simulation, machine learning

=== Readings

- `tensor-network.typ` - Tensor Networks
- `compressed.typ` - Compressed Sensing

=== Lab Exercises

+ Implement tensor contraction with einsum
+ Decompose a tensor using SVD-based methods
+ Represent a quantum state as MPS
+ Compress an image using tensor decomposition

#v(1em)
#line(length: 100%, stroke: 0.5pt + gray)

== Week 12: Monte Carlo & GPU Computing

#block(inset: (left: 1em))[
  *Theme:* _Harnessing randomness and parallelism_
]

=== Learning Objectives

By the end of this week, students will be able to:

+ Implement Monte Carlo integration for high-dimensional integrals
+ Apply Markov Chain Monte Carlo (MCMC) for sampling from complex distributions
+ Analyze MCMC convergence and autocorrelation
+ Write GPU kernels using CUDA.jl
+ Parallelize scientific computations on GPUs
+ Profile and optimize GPU code

=== Topics Covered

- Monte Carlo integration: importance sampling, variance reduction
- MCMC: Metropolis-Hastings, Gibbs sampling
- Convergence diagnostics: trace plots, autocorrelation, R-hat
- GPU architecture: threads, blocks, grids, memory hierarchy
- CUDA.jl: CuArrays, kernel programming
- GPU optimization: memory coalescing, occupancy
- Applications: statistical physics, Bayesian inference

=== Readings

- `MCMC.typ` - Monte Carlo Methods
- `gpu.typ` - GPU Programming

=== Lab Exercises

+ Estimate π using Monte Carlo integration
+ Sample from Ising model using MCMC
+ Implement matrix multiplication on GPU
+ Port a CPU simulation to GPU, benchmark speedup

=== Module 4 Project

*Differentiable Physics Simulation*

Build a differentiable simulation:
- Implement a physics simulation (e.g., molecular dynamics, wave equation)
- Make it differentiable using AD
- Optimize system parameters using gradient descent
- Optional: GPU acceleration

#pagebreak()

= Week 13: Capstone Projects

#block(inset: (left: 1em))[
  *Theme:* _Bringing it all together_
]

== Schedule

#table(
  columns: (auto, 1fr),
  align: (center, left),
  [*Day*], [*Activity*],
  [Day 1], [Project work session, office hours],
  [Day 2], [Presentations (Group A), peer feedback],
  [Day 3], [Presentations (Group B), peer feedback],
  [Day 4], [Code review sessions, final revisions],
  [Day 5], [Course wrap-up, future directions],
)

== Project Requirements

Final projects should demonstrate:

+ *Technical depth:* Apply concepts from at least 2 modules
+ *Code quality:* Well-structured, documented, tested
+ *Scientific rigor:* Proper validation and error analysis
+ *Communication:* Clear presentation and written report

== Suggested Project Topics

=== Computational Physics
- Quantum Monte Carlo simulation
- Molecular dynamics with differentiable potentials
- Lattice gauge theory simulation
- Gravitational N-body simulation with GPU

=== Numerical Methods
- Implement a PDE solver (finite difference/element)
- Adaptive mesh refinement for physics simulations
- Multigrid solver for Poisson equation
- Spectral methods for fluid dynamics

=== Machine Learning for Physics
- Physics-informed neural network
- Neural network potential for molecular simulation
- Generative model for physics data
- Reinforcement learning for quantum control

=== Algorithm Development
- Novel tensor network algorithm
- Parallel MCMC implementation
- Automatic differentiation for complex numbers
- Sparse direct solver with GPU acceleration

== Presentation Format

- *Duration:* 15 minutes + 5 minutes Q&A
- *Content:* Problem motivation, methods, results, code demo
- *Materials:* Slides, live demo, GitHub repository

== Evaluation Criteria

#table(
  columns: (auto, 1fr, auto),
  align: (center, left, center),
  [*Criterion*], [*Description*], [*Weight*],
  [Technical], [Correctness, efficiency, appropriate methods], [30%],
  [Code Quality], [Structure, documentation, tests, reproducibility], [25%],
  [Results], [Validation, analysis, visualization], [25%],
  [Presentation], [Clarity, demo quality, Q&A handling], [20%],
)

#pagebreak()

= Appendix: Weekly Schedule Template

== Typical Week Structure

#table(
  columns: (auto, auto, 1fr),
  align: (center, center, left),
  [*Day*], [*Time*], [*Activity*],
  [Mon], [2h], [Lecture: Concepts and theory],
  [Wed], [2h], [Lab: Hands-on coding exercises],
  [Fri], [--], [Assignment due, self-study],
)

== Office Hours

- Instructor: TBD
- Teaching Assistants: TBD

== Late Policy

- Labs: 10% penalty per day, up to 3 days
- Projects: 20% penalty per day, up to 2 days
- Extensions: Request at least 24 hours in advance

== Academic Integrity

All code must be your own work. You may:
- Discuss concepts with classmates
- Use documentation and online resources
- Reference example code with citation

You may not:
- Copy code from classmates
- Use AI to generate solutions without understanding
- Submit work from previous years

== Resources

- Course website: #link("https://scfp.jinguo-group.science")
- Julia documentation: #link("https://docs.julialang.org")
- Discourse forum: #link("https://discourse.julialang.org")
- Slack/Discord: (course-specific channel)

#v(2em)

#align(center)[
  #text(style: "italic")[
    Last updated: January 2026
  ]
]
