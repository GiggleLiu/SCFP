
#import "@preview/shiroa:0.1.2": *

#show: book

#book-meta(
  title: "Scientific Computing For Physicsists",
  description: "Scientific computing for physicsists, with codes in Julia programming language.",
  repository: "https://github.com/Myriad-Dreamin/shiroa",
  authors: ("Myriad-Dreamin", "7mile"),
  summary: [
    #prefix-chapter("home.typ")[Home]
    #chapter("chap1/terminal.typ")[Get a terminal]
  ]
)



// re-export page template
#import "templates/page.typ": project
#let book-page = project
