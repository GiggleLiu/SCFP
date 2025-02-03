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

#let m = hkustgz-theme.register()

#show raw.where(block: true): it=>{
  block(radius:4pt, fill:gray.transparentize(90%), inset:1em, width:99%, text(it))
}

// Global information configuration
#let m = (m.methods.info)(
  self: m,
  title: [Terminal, Vim, SSH and Git],
  subtitle: [Basic programming toolchain],
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


=== Growing Adoption
Many prominent scientists and engineers have switched to Julia:

- *Steven G. Johnson*: Creator of #link("http://www.fftw.org/")[FFTW], transitioned from C++
- *Anders Sandvik*: Developer of the Stochastic Series Expansion quantum Monte Carlo method, moved from Fortran (#link("https://physics.bu.edu/~py502/")[Computational Physics course])
- *Miles Stoudenmire*: Creator of #link("https://itensor.org/")[ITensor], switched from C++
- *Jutho Haegeman* and *Chris Rackauckas*: Leading researchers in quantum physics and differential equations

#block(
  fill: rgb("#e7f3fe"),
  inset: 8pt,
  radius: 4pt,
  [
    === Should I Switch to Julia?
    Consider switching to Julia if:
    - Your computations typically run for more than 10 minutes
    - Existing tools don't meet your specific needs
    - You value both performance and code readability
  ]
)

#figure(
  image("images/benchmark.png", width: 400pt),
  caption: [Benchmark comparison of various programming languages normalized to C/C++]
)

