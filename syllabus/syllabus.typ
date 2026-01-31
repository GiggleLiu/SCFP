#set document(title: "SCFP: Scientific Computing for Physicists - Course Guide", author: "Jin-Guo Liu")
#set page(margin: (x: 2.5cm, y: 2.5cm))
#set text(size: 11pt)
#set heading(numbering: "1.1")
#set par(justify: true)

#align(center)[
  #text(size: 24pt, weight: "bold")[Scientific Computing for Physicists]

  #text(size: 14pt)[Course Guide & Learning Objectives]

  #v(0.5em)

  #text(size: 11pt, style: "italic")[Jin-Guo Liu et al.]
]

#v(1em)

#outline(indent: auto, depth: 2)

#pagebreak()

= Course Overview

== Description

This course provides a comprehensive introduction to scientific computing with applications in physics. Students learn modern programming practices using the Julia programming language, numerical methods for linear algebra and optimization, and simulation techniques essential for computational physics research.

== Prerequisites

- Basic programming experience (any language)
- Linear algebra fundamentals (vectors, matrices, eigenvalues)
- Calculus (derivatives, integrals, gradients)
- Basic physics knowledge (mechanics, thermodynamics)

== Course Outcomes

Upon successful completion, students will be able to:

+ Write efficient, well-documented Julia code following best practices
+ Apply numerical linear algebra techniques to solve physics problems
+ Implement and analyze optimization algorithms for complex systems
+ Use automatic differentiation for gradient-based computations
+ Perform Monte Carlo simulations and analyze statistical results
+ Leverage GPU computing for high-performance applications

== Materials

- *Primary:* Online textbook at #link("https://scfp.jinguo-group.science")[scfp.jinguo-group.science]
- *Software:* Julia 1.10+, VS Code with Julia extension, Git

#pagebreak()

= Part I: Julia Programming

This part introduces the Julia programming language and modern development practices.

== Terminal Environment
#text(style: "italic", size: 10pt)[Reading: `chap1-julia/terminal.typ`]

=== Learning Objectives

By the end of this topic, students will be able to:

+ Navigate the terminal using essential shell commands (`cd`, `ls`, `mkdir`, `cp`, `mv`, `rm`)
+ Understand file system structure, permissions, and environment variables
+ Configure SSH keys for secure remote server access
+ Write basic shell scripts for automation
+ Use terminal multiplexers (tmux/screen) for persistent sessions

=== Key Concepts

- File system navigation and manipulation
- Environment variables and PATH configuration
- SSH configuration and key management
- Shell scripting basics

=== Exercises

+ Set up a development environment with proper directory structure
+ Configure SSH access to a remote server
+ Write a shell script to automate a repetitive task

#v(1em)
#line(length: 100%, stroke: 0.5pt + gray)

== Version Control with Git
#text(style: "italic", size: 10pt)[Reading: `chap1-julia/git.typ`]

=== Learning Objectives

+ Initialize and manage Git repositories with proper commit practices
+ Use branching and merging workflows for collaborative development
+ Write clear commit messages and maintain clean git history
+ Handle merge conflicts effectively
+ Use GitHub/GitLab for collaboration (pull requests, issues, code review)

=== Key Concepts

- Git workflow: init, add, commit, push, pull, branch, merge
- Remote repositories and collaboration
- Branching strategies (feature branches, GitFlow)
- Conflict resolution

=== Exercises

+ Create a repository with proper README, LICENSE, and .gitignore
+ Practice branching workflow: create feature branch, make changes, merge
+ Simulate and resolve a merge conflict
+ Set up a pull request workflow

#v(1em)
#line(length: 100%, stroke: 0.5pt + gray)

== Julia Setup
#text(style: "italic", size: 10pt)[Reading: `chap1-julia/julia-setup.typ`]

=== Learning Objectives

+ Install Julia and configure the development environment
+ Use VS Code with the Julia extension effectively
+ Navigate Julia's package manager (Pkg)
+ Understand Julia's project and manifest files

=== Key Concepts

- Julia installation and version management
- IDE setup and configuration
- Package management with Pkg
- Project environments and reproducibility

#v(1em)
#line(length: 100%, stroke: 0.5pt + gray)

== Julia Basics
#text(style: "italic", size: 10pt)[Reading: `chap1-julia/julia-basic.typ`]

=== Learning Objectives

+ Use the Julia REPL effectively (help mode, shell mode, package mode)
+ Define and use primitive types, arrays, and custom structs
+ Write functions with multiple dispatch
+ Apply control flow structures (if/else, for, while, comprehensions)
+ Understand Julia's type system and its role in performance
+ Debug code using print statements and the debugger
+ Profile code and identify performance bottlenecks

=== Key Concepts

- REPL modes and interactive workflow
- Type system: primitive types, abstract types, parametric types
- Collections: Array, Tuple, NamedTuple, Dict, Set
- Functions: positional args, keyword args, varargs, anonymous functions
- Multiple dispatch and method specialization
- Control flow and iteration patterns
- Performance: `@time`, `@benchmark`, type stability

=== Exercises

+ Implement mathematical functions (factorial, fibonacci, prime check)
+ Create custom structs for physical systems (Particle, Vector3D)
+ Write functions with multiple dispatch for different types
+ Profile and optimize a slow function

#v(1em)
#line(length: 100%, stroke: 0.5pt + gray)

== Package Development
#text(style: "italic", size: 10pt)[Reading: `chap1-julia/julia-release.typ`]

=== Learning Objectives

+ Create and structure Julia packages with proper layout
+ Write unit tests using the Test standard library
+ Generate documentation with Documenter.jl
+ Set up continuous integration (CI) for automated testing
+ Publish packages to the Julia registry

=== Key Concepts

- Package structure: `src/`, `test/`, `docs/`, `Project.toml`
- Module system: export, import, using
- Unit testing: `@test`, `@testset`, test coverage
- Documentation: docstrings, Documenter.jl, doctests
- CI/CD: GitHub Actions for Julia packages

=== Project

*Create a Physics Utilities Package:* Build a Julia package with custom types for physical quantities, comprehensive tests, documentation, and CI pipeline.

#v(1em)
#line(length: 100%, stroke: 0.5pt + gray)

== GPU Programming
#text(style: "italic", size: 10pt)[Reading: `chap1-julia/gpu.typ`]

=== Learning Objectives

+ Understand GPU architecture (threads, blocks, grids, memory hierarchy)
+ Write GPU kernels using CUDA.jl
+ Transfer data between CPU and GPU efficiently
+ Parallelize scientific computations on GPUs
+ Profile and optimize GPU code

=== Key Concepts

- GPU vs CPU: architecture differences
- CUDA programming model
- Memory hierarchy: global, shared, local memory
- CuArrays and array programming on GPU
- Kernel programming with CUDA.jl
- Memory coalescing and occupancy optimization

=== Exercises

+ Implement vector addition on GPU
+ Port matrix multiplication to GPU, benchmark speedup
+ Optimize memory access patterns for better performance

#pagebreak()

= Part II: Numerical Linear Algebra

This part covers numerical methods for linear algebra with applications in scientific computing.

== Matrix Computation
#text(style: "italic", size: 10pt)[Reading: `chap2-linalg/linalg.typ`]

=== Learning Objectives

+ Perform efficient matrix operations using BLAS and LAPACK
+ Understand memory layout (column-major) and its performance impact
+ Implement and apply LU decomposition for solving linear systems
+ Use QR decomposition for least squares problems
+ Apply Singular Value Decomposition (SVD) for data analysis
+ Analyze condition number and numerical stability

=== Key Concepts

- Matrix storage: column-major order, stride, views
- BLAS levels: Level 1 (vector), Level 2 (matrix-vector), Level 3 (matrix-matrix)
- LU decomposition with pivoting
- QR decomposition: Gram-Schmidt, Householder reflections
- SVD: theory, computation, applications
- Condition number and error analysis

=== Exercises

+ Benchmark naive vs BLAS matrix multiplication
+ Implement LU decomposition, compare with built-in
+ Solve overdetermined systems using QR and normal equations
+ Image compression using truncated SVD

#v(1em)
#line(length: 100%, stroke: 0.5pt + gray)

== Advanced Matrix Methods
#text(style: "italic", size: 10pt)[Reading: `chap2-linalg/linalg-advanced.typ`]

=== Learning Objectives

+ Compute eigenvalues and eigenvectors using power iteration
+ Apply the QR algorithm for eigenvalue computation
+ Use Krylov subspace methods (Arnoldi, Lanczos) for large matrices
+ Implement conjugate gradient for symmetric positive definite systems
+ Choose between direct and iterative solvers based on problem characteristics

=== Key Concepts

- Eigenvalue problems in physics (quantum mechanics, vibrations)
- Power iteration and inverse iteration
- QR algorithm for eigenvalues
- Krylov subspaces and Arnoldi iteration
- Lanczos algorithm for symmetric matrices
- Conjugate gradient method
- Preconditioning strategies

=== Exercises

+ Implement power iteration for dominant eigenvalue
+ Find vibrational modes of a molecular system
+ Compare direct vs iterative solvers for varying matrix sizes

#v(1em)
#line(length: 100%, stroke: 0.5pt + gray)

== Sparse Matrices and Graphs
#text(style: "italic", size: 10pt)[Reading: `chap2-linalg/sparse.typ`]

=== Learning Objectives

+ Store and manipulate sparse matrices in CSR/CSC formats
+ Perform efficient sparse matrix operations
+ Represent graphs as sparse adjacency matrices
+ Implement graph algorithms (BFS, DFS, shortest path)
+ Apply PageRank and network analysis algorithms

=== Key Concepts

- Sparse matrix formats: COO, CSR, CSC
- SparseArrays.jl operations and conversions
- Graph representations: adjacency matrix, adjacency list
- Graph traversal: breadth-first, depth-first search
- Shortest paths: Dijkstra, Bellman-Ford
- PageRank algorithm

=== Exercises

+ Convert dense matrix to sparse, compare memory usage
+ Implement BFS/DFS using sparse matrix representation
+ Compute PageRank for a web graph dataset

#v(1em)
#line(length: 100%, stroke: 0.5pt + gray)

== Tensor Networks
#text(style: "italic", size: 10pt)[Reading: `chap2-linalg/tensor-network.typ`]

=== Learning Objectives

+ Represent tensors and perform tensor operations (contraction, decomposition)
+ Apply tensor decompositions: CP, Tucker, Tensor Train
+ Understand Matrix Product States (MPS) for quantum systems
+ Use tensor network diagrams for algorithm design
+ Implement basic tensor network algorithms

=== Key Concepts

- Tensor basics: indices, contraction, einsum notation
- Tensor decompositions: CP (CANDECOMP/PARAFAC), Tucker
- Tensor Train / Matrix Product State (MPS)
- Tensor network diagrams
- DMRG algorithm basics
- Applications in quantum physics and machine learning

=== Exercises

+ Implement tensor contraction with einsum notation
+ Decompose a tensor using SVD-based methods
+ Represent a quantum state as MPS

=== Project

*Sparse Linear Solver:* Implement a sparse linear solver with iterative methods, preconditioning, and benchmarking on physics problems.

#pagebreak()

= Part III: Optimization

This part covers optimization methods from combinatorial to continuous optimization.

== Simulated Annealing
#text(style: "italic", size: 10pt)[Reading: `chap3-optimization/simulated-annealing.typ`]

=== Learning Objectives

+ Formulate combinatorial optimization problems
+ Implement the Metropolis-Hastings algorithm
+ Apply simulated annealing with appropriate cooling schedules
+ Analyze convergence and mixing properties
+ Design problem-specific move proposals

=== Key Concepts

- Combinatorial optimization: TSP, graph coloring, spin glasses
- Local search and hill climbing limitations
- Metropolis algorithm and detailed balance
- Simulated annealing: temperature schedules, acceptance probability
- Parallel tempering for improved sampling

=== Exercises

+ Implement Metropolis algorithm for 2D Ising model
+ Solve traveling salesman problem with simulated annealing
+ Compare different cooling schedules

#v(1em)
#line(length: 100%, stroke: 0.5pt + gray)

== Mathematical Optimization
#text(style: "italic", size: 10pt)[Reading: `chap3-optimization/linear_integer.typ`]

=== Learning Objectives

+ Formulate problems as linear programs (LP)
+ Understand the simplex method and interior point methods
+ Model integer constraints and solve ILPs
+ Use JuMP.jl for optimization modeling
+ Recognize and exploit problem structure

=== Key Concepts

- Linear programming: standard form, duality
- Simplex method: basic feasible solutions, pivoting
- Interior point methods
- Integer linear programming: branch and bound
- Mixed-integer programming
- JuMP.jl modeling language
- Convex optimization basics

=== Exercises

+ Model and solve a resource allocation problem
+ Formulate a physics problem as optimization
+ Compare solver performance on benchmark problems

#v(1em)
#line(length: 100%, stroke: 0.5pt + gray)

== Gradient-Based Optimization
#text(style: "italic", size: 10pt)[Reading: `chap3-optimization/gradient-optimization.typ`]

=== Learning Objectives

+ Implement gradient descent with various step size strategies
+ Apply momentum and adaptive learning rate methods (Adam, RMSprop)
+ Use Newton's method and quasi-Newton methods (BFGS, L-BFGS)
+ Handle constraints with projected gradient methods
+ Diagnose and fix convergence issues

=== Key Concepts

- Gradient descent: batch, stochastic, mini-batch
- Line search: Armijo, Wolfe conditions
- Momentum methods: classical momentum, Nesterov
- Adaptive methods: AdaGrad, RMSprop, Adam
- Second-order methods: Newton, Gauss-Newton
- Quasi-Newton: BFGS, L-BFGS
- Constrained optimization basics

=== Exercises

+ Implement gradient descent with line search
+ Compare convergence of SGD, momentum, and Adam
+ Fit a nonlinear model using L-BFGS

#v(1em)
#line(length: 100%, stroke: 0.5pt + gray)

== Automatic Differentiation
#text(style: "italic", size: 10pt)[Reading: `chap3-optimization/ad.typ`]

=== Learning Objectives

+ Explain the difference between numerical, symbolic, and automatic differentiation
+ Implement forward-mode AD using dual numbers
+ Understand reverse-mode AD (backpropagation)
+ Use Zygote.jl and ForwardDiff.jl for automatic gradients
+ Define custom adjoint rules
+ Apply AD to differentiable physics simulations

=== Key Concepts

- Differentiation methods comparison
- Forward-mode AD: dual numbers, computational cost
- Reverse-mode AD: computational graph, adjoint method
- Zygote.jl: pullbacks, custom adjoints
- ForwardDiff.jl: efficient forward-mode
- Differentiable programming paradigm

=== Exercises

+ Implement forward-mode AD with dual numbers
+ Compute gradients with Zygote
+ Implement a custom adjoint rule
+ Train a simple neural network using AD

=== Project

*Optimization Challenge:* Solve a complex physics optimization problem using multiple approaches, compare performance.

#pagebreak()

= Part IV: Simulation

This part covers simulation methods for physics applications.

== Monte Carlo Methods
#text(style: "italic", size: 10pt)[Reading: `chap4-simulation/MCMC.typ`]

=== Learning Objectives

+ Implement Monte Carlo integration for high-dimensional integrals
+ Apply Markov Chain Monte Carlo (MCMC) for sampling
+ Analyze MCMC convergence and autocorrelation
+ Use importance sampling for variance reduction
+ Apply MCMC to statistical physics problems

=== Key Concepts

- Monte Carlo integration basics
- Importance sampling and variance reduction
- MCMC: Metropolis-Hastings, Gibbs sampling
- Convergence diagnostics: trace plots, autocorrelation, R-hat
- Applications: statistical physics, Bayesian inference

=== Exercises

+ Estimate π using Monte Carlo integration
+ Sample from Ising model using MCMC
+ Analyze convergence of MCMC chains
+ Implement importance sampling

=== Project

*Differentiable Physics Simulation:* Build a physics simulation that is differentiable using AD, optimize system parameters.

#pagebreak()

= Appendix

== Plotting with CairoMakie
#text(style: "italic", size: 10pt)[Reading: `appendix/plotting.typ`]

=== Learning Objectives

+ Create publication-quality figures with CairoMakie
+ Customize plot appearance (colors, fonts, layouts)
+ Create animations for time-dependent data
+ Export figures in various formats (PDF, SVG, PNG)

== Compressed Sensing
#text(style: "italic", size: 10pt)[Reading: `chap2-linalg/compressed.typ`]

=== Learning Objectives

+ Understand information theory basics (entropy, coding)
+ Apply compressed sensing for sparse signal recovery
+ Use L1 minimization for sparse solutions
+ Connect to machine learning applications

#v(2em)

#align(center)[
  #text(style: "italic")[
    Course materials available at #link("https://scfp.jinguo-group.science")[scfp.jinguo-group.science]
  ]
]
