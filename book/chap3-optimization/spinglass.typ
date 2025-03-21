#import "../book.typ": book-page, cross-link
#import "@preview/cetz:0.2.2": canvas, draw, tree, vector, plot, decorations
#import "@preview/ctheorems:1.1.3": *
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm

#show: book-page.with(title: "Spin Glass")
#show: thmrules
#set math.equation(numbering: "(1)")

#let definition = thmbox("definition", "Definition", inset: (x: 1.2em, top: 1em), base: none)
#let proposition = thmbox("proposition", "Proposition", inset: (x: 1.2em, top: 1em), base: none)
#let theorem = thmbox("theorem", "Theorem", base: none)
#let proof = thmproof("proof", "Proof")
#let nonum(eq) = math.equation(block: true, numbering: none, eq)
#let jinguo(c) = text(red)[[JG: #c]]

#let pseudo_random(seed, n) = {
  // Simple linear congruential generator
  let a = 1664525
  let c = 1013904223
  let m = calc.pow(2, 32)
  let sequence = ()
  let x = seed
  for i in range(n) {
    x = calc.rem(a * x + c, m)
    sequence.push(x / m)  // normalize to [0,1]
  }
  sequence
}

// Usage example:
#let random_numbers = pseudo_random(12345, 100)
#let random_bools(p) = random_numbers.map(x => x < p)

#align(center, [= Spin Glass\
_Jin-Guo Liu_])



== Main references
- Lecture note: #link("https://physics.bu.edu/~py502/lectures5/mc.pdf")[Monte Carlo simulations in classical statistical physics, Anders Sandvik]
- Swendsen, Robert H., and Jian-Sheng Wang. "Nonuniversal critical dynamics in Monte Carlo simulations." Physical review letters 58.2 (1987): 86.

== Ising model

An Ising model is defined by a graph $G = (V, E)$ and a set of spins $s_i in {-1, 1}$ for each vertex $i in V$. The energy of the system is given by

$
H = - sum_((i,j) in E) J_(i j) s_i s_j - sum_(i in V) h_i s_i
$

We are interested in Ising models for multiple reasons, one is from the physics perspective. We want to understand the phenomemon of phase transition, i.e. how magnetization emerges from the disorder. Another reason is from the optimization perspective, finding the ground state (the configuration with the lowest energy) of an Ising model is in general hard, which is known as the spin glass problem. It is in complexity class NP-complete. Solving the spin glass problem in time polynomial to the graph size would break the complexity hierarchy, which is unlikely to happen.

== Phase transition

In this section, we consider the ferromagnetic Ising model, where $J_(i j) = 1$. For simplicity, we limit our discussion to the $L times L$ grid.
We denote the solution space as $S$, and $bold(s) in S$ is the configuration of spins, e.g. $bold(s) = {-1, -1, 1, -1, 1, dots}$. The number of configurations is $|S| = 2^(L^2)$, which is exponentially large.

#let spinconfig(m, n, cfg, dx: 1, dy: 1, x:0, y:0) = {
  import draw: *
  let len = 0.5
  grid((x, y), (x + (m - 1) * dx, y + (n - 1) * dy), step: (dx, dy), stroke: (paint: gray, thickness: 0.5pt))
  for i in range(m) {
    for j in range(n) {
      if cfg.at(i * n + j) {
        line((x + i * dx, y + j * dy - len/2), (x + i * dx, y + j * dy + len/2), stroke: (paint: red, thickness: 2pt), mark: (end: "straight"))
      } else {
        line((x + i * dx, y + j * dy - len/2), (x + i * dx, y + j * dy + len/2), stroke: (paint: blue, thickness: 2pt), mark: (start: "straight"))
      }
    }
  }
}

#figure(canvas({
  import draw: *
  let s(it) = text(14pt, it)
  spinconfig(4, 4, random_bools(0.5).slice(4, 20))
  content((5, 1.5), s[$|S| = 2^(L^2)$])
}))

When we talk about the phase transition, we are interested in how does the macroscopic properties of the system change as the temperature $T$ changes, like how ice becomes water, magnets lose their magnetization, etc.
Computation is a valid strategy to study the phase transition. Before we move on, we need to consider the following question:
- How to describe the system?
- How to characterize the phase transition?

At finite temperature, the system can be described by a probability distribution, called the Boltzmann distribution:
$
p(bold(s)) = (e^(-beta H(bold(s))))/Z,
$
where $beta = 1 \/ k_B T$ is the inverse temperature, $Z$ is the partition function $Z = sum_bold(s) e^(-beta H(bold(s)))$ that normalizes the probability distribution.
Configurations with lower energy have higher probability to be observed. The sensitivity of the probability distribution to the energy is determined by the inverse temperature $beta$. At zero temperature, $beta$ is infinite, and only the ground state can be observed.

We characterize the macroscopic properties of the system by the statistical average of some functions over the configuration space. Among these functions, the *order parameter* can be used to characterize the phase transition.
As shown in @fig:phase-transition, the order parameter for the magnetization can be defined as
$ |m| = lr(|sum_(i=1)^(L^2) s_i\/L^2|). $ <eq:magnetization>
The statistical average of $|m|$ over the configuration space is
$
angle.l |m| angle.r = sum_(bold(s) in S) |m(bold(s))| p(bold(s))
$ <eq:magnetization-average>
At the inifinite size limit ($L arrow.r infinity$), if the statistical average $angle.l |m| angle.r$ is non-zero, the system is in the magnetized phase. If $angle.l |m| angle.r$ is zero, the system is in the disordered phase.
#figure(canvas({
  import draw: *
  let s(it) = text(14pt, it)
  spinconfig(4, 4, random_bools(0.5).slice(4, 20), x: 2)
  spinconfig(4, 4, random_bools(0.0).slice(4, 20), x: -6)
  line((-5, -1), (0, -1), stroke: (paint: blue, thickness: 2pt), mark: (start: "straight"))
  line((0, -1), (5, -1), stroke: (paint: red, thickness: 2pt), mark: (end: "straight"))
  content((0, -1.5), s[Temperature $T = 1\/beta$])
  content((0, 0), s[$T_c$], name: "T_c")
  line((0, -0.4), (0, -1), mark: (end: "straight"))
  content((6, -1), s[Hot])
  content((-6, -1), s[Cold])
  content((-4.5, 1.5), s[$angle.l |m| angle.r != 0$])
  content((3.5, 1.5), s[$angle.l |m| angle.r = 0$])
})) <fig:phase-transition>

For the ferromagnetic Ising model, the ground state is two fold degenerate, they are the all-up and all-down configurations. At zero temperature, the system is frozen in one of the ground states, i.e. $angle.l |m| angle.r = 1$.

== Integrate a function with importance sampling

How to calculate @eq:magnetization-average?
Summing over all the configurations is infeasible for large system. Instead, we can sample a portion of the configuration space and estimate the statistical average of the quantity we are interested in.
To show why is it possible, we first note that the @eq:magnetization-average can be viewed as an integral, with the integrand $|m(bold(s))| p(bold(s))$ and the integration domain $S$. An integral can be estimated by the statistical average of the integrand over a set of sampled points.
For illustrative purpose, we consider integrating a positive function $f(x)$ defined on a unit square.
$
integral f(x) d x
$
Intead of evaluating the integral directly, the importance sampling can sample $x$ with probability $p(x)$ and estimate the integral as the statistical average of $f(x)\/p(x)$.
#figure(canvas({
  import draw: *
  let s(it) = text(12pt, it)
  rect((-2, -2), (2, 2), stroke: (paint: black, thickness: 1pt), fill: blue.lighten(80%))
  circle((0, 0), radius: 0.1, fill: red, stroke: none)
  content((0, 0.5), s[$f(x) = cases(1/(pi r^2)\, &|x| < r, 0\, &"otherwise")$])
  content((0, -2.5), s[Uniform sampling])

  set-origin((6, 0))
  rect((-2, -2), (2, 2), stroke: (paint: black, thickness: 1pt), fill: blue.lighten(90%))
  rect((-0.5, -0.5), (0.5, 0.5), stroke: none, fill: blue.lighten(50%))
  circle((0, 0), radius: 0.1, fill: red, stroke: none)
  content((0, -0.8), s[$10 times$])
  content((0, -2.5), s[Importance sampling])
})) <fig:importance-sampling>

A good sampling probability can improve the efficiency of the estimation. We consider the example in @fig:importance-sampling.
The function is a peak-like function defined on a unit square, which is zero everywhere except a very small region (the red circle of radius $r << 1$) at the origin. The integral of $f(x)$ is 1. If the sampling is uniform, the sampler will spend most of the time sampling the region far away from the origin.
However, if the sample $x$ with $10 times$ more probability near the origin (in the dark blue region), as shown in the right panel, you can have $10 times$ more chance to find a sample in the peak region. So, whenever you find a sample in the dark blue region, you only count it as $0.1$ sample as a compensation. With this small change, the statistical average will be $sqrt(10)$ times more accurate. In a even more extreme case, if the sampled probability $p(x)$ is proportional to $f(x)$, the number of sample to reach exact result is $1$.

On the other hand, bad sampling probability $p(x)$ may cause the function $f(x)$ to have a poor estimate even if you have infinite samples. This happens when the *ergodicity* is broken, i.e. the system can not reach the whole configuration space. For example, in @fig:ergodicity, the function $f(x)$ has two peaks, but only one peak is accessible to the sampler. Then, no matter how many samples you have, you can not get a good estimate of the integral.
#figure(canvas({
  import draw: *
  let s(it) = text(12pt, it)
  rect((-2, -2), (2, 2), stroke: (paint: black, thickness: 1pt), fill: blue.lighten(100%))
  rect((-1.5, -1.5), (0.5, 0.5), stroke: none, fill: blue.lighten(50%))
  circle((0, 0), radius: 0.1, fill: red, stroke: none)
  circle((0.8, 0.8), radius: 0.1, fill: red, stroke: none)
  content((0, -2.5), s[Ergodicity is broken])
  content((-0.5, -0.7), s[Sample\ region])
})) <fig:ergodicity>


== Metropolis-Hastings algorithm
To evaluate the @eq:magnetization-average, we can sample the configuration space with the Boltzmann distribution $p(bold(s)) = (e^(-beta H(bold(s))))\/Z$, on each sample, we evaluate $|m(bold(s))|$ and calculate the statistical average.
The problem is that we do not know the partition function $Z$. Even if we know how to compute $p(bold(s))$, the sampling is still challenging.
Metropolis-Hastings algorithm is a sampling method to sample the configuration space with un-normalized probability.

The Metropolis-Hastings algorithm is a Markov chain. A Markov chain is a sequence of random variables $bold(s)_1, bold(s)_2, dots$ with the property that the probability of moving to the next state depends only on the current state.
It is characterized by the transition probability $P(bold(s)'|bold(s))$, the probability of moving from $bold(s)$ to $bold(s)'$.

#figure(canvas({
  import draw: *
  let s(it) = text(12pt, it)
  let boxed(it) = box(it, stroke: black, inset: 0.5em)
  let dx = 3
  for i in range(5) {
    content((i * dx, 0), boxed(s[$bold(s)_#i$]), name: "s" + str(i))
  }
  for i in range(4) {
    line("s" + str(i), "s" + str(i + 1), stroke: (paint: black, thickness: 1pt), mark: (end: "straight"), name: "t" + str(i))
    content((rel: (0, 0.5), to: "t" + str(i) + ".mid"), s[$P(bold(s)_#(i+1)|bold(s)_#i)$])
  }
    
})) <fig:markov-chain>

The algortihm is summarized as follows:
#algorithm({
  import algorithmic: *
  Function([Metropolis-Hastings], args: ([$beta$], [$n$]), {
    Assign([$bold(s)$], [Initial a random configuration])
    For(cond: [$i = 1$ to $n$], {
      Assign([$bold(s)'$], [propose a new configuration with prior probability $T(bold(s)arrow.r bold(s)')$])
      Assign([$bold(s)$], [$bold(s)' "with probability:" min(1, (T(bold(s)' arrow.r bold(s)) p(bold(s)'))/(T(bold(s) arrow.r bold(s)') p(bold(s))))$])
    })
  })
 })

In line 4, a new configuration $bold(s)'$ is proposed. The probability of proposing $bold(s)'$ from $bold(s)$ is $T(bold(s)arrow.r bold(s)')$, which is known when we design the algorithm. For example, when we propose a state transfer on two 3-state spins, we may have the following transition rule:
#figure(canvas({
  import draw: *
  let s(it) = text(12pt, it)
  let d = 1.5
  for i in range(2){
    for j in range(2){
      rect((i * d, j * d), (i * d + d, j * d + d))
    }
  }
  line((0, 0), (0, 0.5 * d), mark: (end: "straight"), stroke: (paint: blue, thickness: 2pt))
  line((0, 0), (0.5 * d, 0), mark: (end: "straight"), stroke: (paint: blue, thickness: 2pt))
  content((0, -0.5), s[$(0, 0)$])
  content((d, -0.5), s[$(1, 0)$])

  for target in ((1.5 * d, 0), (1 * d, 0.5 * d), (0.5 * d, 0)) {
    line((1 * d, 0), target, mark: (end: "straight"), stroke: (paint: red, thickness: 2pt))
  }
})) <fig:propose-new-config>
The probability of proposing $(1, 0)$ from $(0, 0)$ is $1\/3$, and the probability of proposing $(0, 1)$ from $(0, 0)$ is $1\/2$. This prior is biased towards having less configurations in state $(0, 0)$, we must compensate for this bias by adjusting the acceptance probability. In a two state spin system, the random flip of a spin is unbiased, so the acceptance probability only depends on the ratio of the probability of the new and current configurations:
$ p(bold(s)')/p(bold(s)) = e^(-beta (H(bold(s)') - H(bold(s))). $

== Demonstration

== Physical quantities that we are interested in

- Energy/spin: $angle.l H^k/n angle.r = integral H(s)^k/n p(s) d s.$
- Magnetization: $m^k = angle.l (sum_i |s_i|)^k \/ n angle.r = integral (sum_i |s_i|)^k \/n p(s) d s.$

== When it fails to thermalize: spin glass

Spin glass is computational universal, i.e. if you can cool down a spin glass system to the ground state, you can solve any problem.

Spin glass is in NP-complete, i.e. if you can cool down a spin glass system to the ground state in polynomial time, you can solve any problem in NP in polynomial time.

== The spectral gap

== Parallel tempering

== Cheeger's inequality