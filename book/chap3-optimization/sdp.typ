#import "../book.typ": book-page, cross-link
#import "@preview/cetz:0.2.2": canvas, draw, tree, vector, plot, decorations
#import "@preview/ctheorems:1.1.3": *

#show: book-page.with(title: "Semidefinite programming")

#let definition = thmbox("definition", "Definition", inset: (x: 1.2em, top: 1em), base: none)

#align(center, [= Semidefinite programming\
_Jin-Guo Liu_])

_Semidefinite programming_ is a generalization of linear programming. It is also a convex optimization problem, hence it is easy to solve.
