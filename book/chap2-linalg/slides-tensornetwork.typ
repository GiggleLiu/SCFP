#import "@preview/touying:0.4.2": *
#import "@preview/touying-simpl-hkustgz:0.1.0" as hkustgz-theme
#import "@preview/cetz:0.2.2": *
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm
#set math.mat(row-gap: 0.1em, column-gap: 0.7em)

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
  title: [Hidden Markov Models and Tensor Networks],
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

= Hidden Markov Models
== Definitions

- $bold(pi)$ is the initial probability of the hidden states.
- $bold(s)_i$ is the hidden state at time $i$.
- $bold(x)_i$ is the observation at time $i$.
- $P(bold(s)_(i+1)|bold(s)_i)$ is the transition probability of the hidden states.
- $P(bold(x)_i|bold(s)_i)$ is the emission probability of the observations.

#figure(canvas({
  import draw: *
  let s(it) = text(12pt, it)
  let boxed(it) = box(it, stroke: black, inset: 0.5em)
  let dx = 3
  content((- dx, 0), boxed(s[$bold(pi)$]), name: "pi")
  for i in range(5) {
    content((i * dx, 0), boxed(s[$bold(s)_#i$]), name: "s" + str(i))
    content((i * dx, -2.5), boxed(s[$bold(x)_#i$]), name: "x" + str(i))
    line("s" + str(i), "x" + str(i), stroke: (paint: black, thickness: 1pt), mark: (end: "straight"), name: "e" + str(i))
    content((rel: (1, 0), to: "e" + str(i) + ".mid"), s[$P(bold(x)_#i|bold(s)_#i)$])
  }
  line("pi", "s0", stroke: (paint: black, thickness: 1pt), mark: (end: "straight"), name: "e0")
  content((rel: (0, 0.5), to: "e0.mid"), s[$P(bold(s)_0|bold(pi))$])
  for i in range(4) {
    line("s" + str(i), "s" + str(i + 1), stroke: (paint: black, thickness: 1pt), mark: (end: "straight"), name: "t" + str(i))
    content((rel: (0, 0.5), to: "t" + str(i) + ".mid"), s[$P(bold(s)_#(i+1)|bold(s)_#i)$])
  }
})) <fig:markov-chain>


== The Viterbi algorithm

== Baum-Welch algorithm

= Tensor networks

== Tensor networks and HMMs

