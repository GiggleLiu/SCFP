#import "@preview/shiroa:0.1.2": *

#show: book

#book-meta(
  title: "Scientific Computing For Physicsists",
  description: "Scientific computing for physicsists, with codes in Julia programming language.",
  repository: "https://github.com/GiggleLiu/SCFP",
  authors: ("Jin-Guo Liu et al",),
  summary: [
    #prefix-chapter("home.typ")[Home]
    = Julia programming language
    #chapter("chap1-julia/terminal.typ")[Terminal Environment]
    #chapter("chap1-julia/git.typ")[Version control]
    #chapter("chap1-julia/julia-setup.typ")[Setup Julia]
    #chapter("chap1-julia/julia-basic.typ")[Julia Basic]
    #chapter("chap1-julia/julia-release.typ")[My First Package]
    = Matrices and tensors
    #chapter("chap2-linalg/array.typ")[Arrays operations]
    #chapter("chap2-linalg/linalg.typ")[Matrix Computation]
    #chapter("chap2-linalg/linalg-advanced.typ")[Matrix Computation (Advanced Topics)]
    #chapter("chap2-linalg/sparse.typ")[Sparse Matrices and Graphs]
    #chapter("chap2-linalg/tensor-network.typ")[Tensor networks]
    = Mathematical optimization
    #chapter("chap3-optimization/linear_integer.typ")[Linear/Integer programming]
    #chapter("chap3-optimization/ad.typ")[Automatic differentiation]
    = Quantum systems
    #chapter("chap4-quantum/quantum-simulation.typ")[Quantum simulation]
    = Appendix
    #chapter("appendix/plotting.typ")[Plotting (CairoMakie)]
  ]
)



// re-export page template
#import "templates/page.typ": project, heading-reference
#let book-page = project
