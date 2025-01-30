#import "@preview/cetz:0.2.2": canvas, draw, tree
#import "@preview/ctheorems:1.1.3": *
#import "../book.typ": book-page

#set math.equation(numbering: "(1)")

#show: book-page.with(title: "Automatic differentiation")
#show: thmrules

#import "@preview/ouset:0.2.0": ouset

#let definition = thmbox("definition", "Definition", inset: (x: 1.2em, top: 1em, bottom: 1em), base: none, stroke: black)
#let theorem = thmbox("theorem", "Theorem", base: none, stroke: black)
#let proof = thmproof("proof", "Proof")

= Automatic differentiation

