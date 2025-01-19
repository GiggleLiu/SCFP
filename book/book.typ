
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
    #chapter("chap1/terminal.typ")[Get a terminal]
    #chapter("chap1/git.typ")[Version control]
    #chapter("chap1/julia-setup.typ")[Setup Julia]
    #chapter("chap1/julia-why.typ")[Why Julia]
    #chapter("chap1/julia-release.typ")[Release a package]
    = Matrices and tensors
    #chapter("chap2/array.typ")[Arrays]
    = Mathematical optimization
    = Tensor networks
    = Automatic differentiation
    = Quantum systems
  ]
)



// re-export page template
#import "templates/page.typ": project
#let book-page = project
