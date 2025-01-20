
#import "@preview/shiroa:0.1.2": *

#show: book

#book-meta(
  title: "Scientific Computing For Physicsists",
  description: "Scientific computing for physicsists, with codes in Julia programming language.",
  repository: "https://github.com/Myriad-Dreamin/shiroa",
  authors: ("Myriad-Dreamin", "7mile"),
  summary: [
    #prefix-chapter("home.typ")[Home]
    = Julia programming language
    #chapter("chap1-julia/terminal.typ")[Get a terminal]
    #chapter("chap1-julia/git.typ")[Version control]
    #chapter("chap1-julia/julia-setup.typ")[Setup Julia]
    #chapter("chap1-julia/julia-why.typ")[Why Julia]
    #chapter("chap1-julia/julia-release.typ")[Release a package]
    = Matrices and tensors
    #chapter("chap2-linalg/array.typ")[Arrays]
    = Mathematical optimization
    #chapter("chap3-optimization/linearprog.typ")[Linear programming]
    = Automatic differentiation
    = Quantum systems
    #chapter("chap6-quantum/quantum-simulation.typ")[Quantum circuit]
  ]
)



// re-export page template
#import "templates/page.typ": project
#let book-page = project
