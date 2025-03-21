#import "../book.typ": book-page, cross-link
#import "@preview/cetz:0.2.2": canvas, draw, tree, vector, plot, decorations
#import "@preview/ctheorems:1.1.3": *
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm

#show: book-page.with(title: "Spin glass and MCMC")
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
- Lecture note of Anders Sandvik: #link("https://physics.bu.edu/~py502/lectures5/mc.pdf")[Monte Carlo simulations in classical statistical physics]
- Book: The nature of computation, @Moore2011, Chapter 12-13
- Code: https://github.com/GiggleLiu/ScientificComputingDemos/tree/main/IsingModel and https://github.com/GiggleLiu/ScientificComputingDemos/tree/main/Spinglass

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

== Physical quantities that we are interested in

- Energy/spin: $angle.l H^k/n angle.r = integral H(s)^k/n p(s) d s.$
- Magnetization: $m^k = angle.l (sum_i |s_i|)^k \/ n angle.r = integral (sum_i |s_i|)^k \/n p(s) d s.$

== Metric of a good MCMC method

=== Acceptance rate
In the ferrromagnetic phase, the MCMC method can easily get stuck in one of the ground states. A clever design can help the sampler to escape the local minimum, the cluster update proposed in @Swendsen1987 is a good example. When the prior is the same as the target distribution, the sampling the the most efficient, it has acceptance rate 1.

=== Autocorrelation time
Because a new sample in the MCMC method is generated from the previous sample, we often have time correlated samples in MCMC methods.
Since the correlated samples are not independent, we effectively have less samples than we expect.
The autocorrelation time $tau$ is the number of steps it takes for the correlation between two consecutive samples to decay to one half of the maximum correlation.
The effective number of independent samples is $n\/tau$. A good MCMC method should have a small autocorrelation time.

#jinguo([TODO: add a numeric example])

== When it fails to thermalize: spin glass

Spin glass is computational universal, i.e. if you can cool down a spin glass system to the ground state, you can solve any problem.

Spin glass is in NP-complete, i.e. if you can cool down a spin glass system to the ground state in polynomial time, you can solve any problem in NP in polynomial time.



== The spectral gap

The transition matrix $P$ of a Markov chain has eigenvalues $1 = lambda_1 > lambda_2 >= lambda_3 >= ... >= lambda_n >= -1$. The spectral gap is defined as:

$
Delta = 1 - lambda_2
$

This gap determines how quickly the Markov chain converges to its stationary distribution (the Boltzmann distribution in our case). A larger spectral gap means faster convergence:

- If $Delta$ is large, the system thermalizes quickly
- If $Delta$ is small, the system thermalizes slowly
- If $Delta approx 0$, the system may never thermalize in practical time

For a Metropolis-Hastings algorithm sampling from the Boltzmann distribution, the mixing time (time to reach equilibrium) scales as $t_"mix" ~ 1/Delta$.

== Parallel tempering

Parallel tempering (also known as replica exchange) is a Monte Carlo method designed to improve sampling efficiency for systems with rough energy landscapes, such as spin glasses. The key idea is to simulate multiple replicas of the system at different temperatures simultaneously, allowing configurations to be exchanged between temperatures.

=== Algorithm overview

In parallel tempering:

1. We simulate $M$ replicas of the system at different temperatures $T_1 < T_2 < ... < T_M$
2. Each replica evolves according to standard Metropolis dynamics at its temperature
3. Periodically, we attempt to swap configurations between adjacent temperature levels

The swap between configurations at temperatures $T_i$ and $T_(i+1)$ is accepted with probability:

$
P_"swap"(bold(s)_i, bold(s)_(i+1)) = min(1, exp(-(beta_i - beta_(i+1))(H(bold(s)_(i+1)) - H(bold(s)_i))))
$

where $beta_i = 1/T_i$ and $bold(s)_i$ is the configuration at temperature $T_i$.

=== Benefits of parallel tempering

Parallel tempering offers several advantages:

1. *Improved exploration*: Higher temperature replicas can easily cross energy barriers, while lower temperature replicas sample the relevant low-energy states
2. *Faster thermalization*: Configurations can travel up and down the temperature ladder, helping the system escape local minima
3. *Better sampling of low-energy states*: The method provides more efficient sampling of the low-temperature distribution

=== Implementation considerations

- *Temperature spacing*: The temperatures should be chosen so that the acceptance rate for swaps between adjacent temperatures is reasonable (typically 20-30%)
- *Swap frequency*: Swaps are typically attempted after each replica has undergone several Metropolis updates
- *Number of replicas*: More replicas provide better temperature coverage but increase computational cost

=== Pseudocode

#algorithm({
  import algorithmic: *
  Function("ParallelTempering", args: ([$H$], [$T_1, ..., T_M$], [$N_"steps"$]), {
    Assign([$bold(s)_1, ..., bold(s)_M$], [random initial configurations])
    For(range: [$t = 1$ to $N_"steps"$], {
      // Update each replica with Metropolis
      For(range: [$i = 1$ to $M$], {
        Assign([$bold(s)_i$], [MetropolisUpdate($bold(s)_i$, $H$, $T_i$)])
      })
      
      // Attempt swaps between adjacent temperatures
      If(cond: [$t$ mod $N_"swap"$ = 0], {
        For(range: [$i = 1$ to $M-1$], {
          Assign([$Delta E$], [$H(bold(s)_(i+1)) - H(bold(s)_i)$])
          Assign([$Delta beta$], [$1/T_i - 1/T_(i+1)$])
          If(cond: [$cal(U)(0,1) < exp(-Delta beta \cdot Delta E)$], {
            Assign([$(bold(s)_i, bold(s)_(i+1))$], [$(bold(s)_(i+1), bold(s)_i)$])
          })
        })
      })
    })
    Return([$bold(s)_1$])  // Return lowest temperature configuration
  })
})

Parallel tempering is particularly effective for spin glass systems where the energy landscape contains many local minima separated by high barriers, making standard Metropolis sampling inefficient at low temperatures.

== Cheeger's inequality

Cheeger's inequality is a fundamental result in spectral graph theory that relates the conductance (or isoperimetric constant) of a graph to its spectral gap. This relationship is particularly important in the context of spin glass systems and Markov Chain Monte Carlo methods, as it provides bounds on mixing times.

=== Conductance and the Cheeger constant

For a graph $G = (V, E)$ with vertex set $V$ and edge set $E$, the Cheeger constant (or conductance) $h(G)$ is defined as:

$
h(G) = min_(S subset V, 0 < |S| <= |V|/2) frac(|E(S, V backslash S)|, min("vol"(S), "vol"(V backslash S)))
$

where:
- $E(S, V backslash S)$ is the set of edges between $S$ and its complement
- $"vol"(S) = sum_(v in S) d_v$ is the volume of set $S$, with $d_v$ being the degree of vertex $v$

The Cheeger constant measures how well-connected the graph is, or equivalently, how difficult it is to partition the graph into disconnected components.

=== Cheeger's inequality

Cheeger's inequality relates the Cheeger constant $h(G)$ to the second smallest eigenvalue $lambda_2$ of the normalized Laplacian matrix $L = I - D^(-1/2) A D^(-1/2)$, where $D$ is the degree matrix and $A$ is the adjacency matrix:

$
frac(lambda_2, 2) <= h(G) <= sqrt(2 lambda_2)
$

This inequality provides both lower and upper bounds on the Cheeger constant in terms of the spectral gap.

#jinguo([How does this equation relate to Eq.12.27 in the nature of computation?])

=== Relation to mixing time

The mixing time of a Markov chain is the time required for the chain to approach its stationary distribution. For a reversible Markov chain, the mixing time $t_"mix"$ is related to the spectral gap $(1 - lambda_2)$ of the transition matrix:

$
t_"mix" approx frac(1, 1 - lambda_2)
$

By Cheeger's inequality, we know that:

$
1 - lambda_2 >= frac(h(G)^2, 2)
$

Therefore:

$
t_"mix" <= frac(2, h(G)^2)
$

This means that a graph with a large Cheeger constant (good expansion properties) will have a small mixing time, allowing MCMC methods to converge quickly to the stationary distribution.

=== Estimating the Cheeger constant

Exactly computing the Cheeger constant is NP-hard, but there are several approaches to estimate it:

1. *Spectral methods*: Using Cheeger's inequality, we can compute $lambda_2$ and use it as an approximation.

2. *Sampling-based methods*: For large graphs, we can use random walks to estimate the conductance.

3. *Approximation algorithms*: There exist polynomial-time algorithms that can approximate the Cheeger constant within certain factors.

For spin glass systems, estimating the Cheeger constant can provide valuable insights into the difficulty of sampling from the Boltzmann distribution at low temperatures. A small Cheeger constant indicates the presence of bottlenecks in the state space, which can significantly slow down the mixing of MCMC methods.

#algorithm({
  import algorithmic: *
  Function("EstimateCheegerConstant", args: ([$G = (V, E)$], [$k$]), {
    // Compute the normalized Laplacian matrix
    Assign([$D$], [diagonal degree matrix of $G$])
    Assign([$A$], [adjacency matrix of $G$])
    Assign([$L$], [$I - D^(-1/2) A D^(-1/2)$])
    
    // Compute the second smallest eigenvalue
    Assign([$lambda_2$], [second smallest eigenvalue of $L$])
    
    // Use spectral partitioning to find a good cut
    Assign([$v_2$], [eigenvector corresponding to $lambda_2$])
    Assign([$S_t$], [vertices with the smallest $t$ values in $D^(-1/2) v_2$])
    
    // Compute conductance for different values of t
    Assign([$h_"min"$], [$infinity$])
    For(range: [$t = 1$ to $|V|-1$], {
      Assign([$h_t$], [$|E(S_t, V backslash S_t)| \/ min("vol"(S_t), "vol"(V backslash S_t))$])
      If(cond: [$h_t < h_"min"$], {
        Assign([$h_"min"$], [$h_t$])
      })
    })
    
    Return([$h_"min"$, $lambda_2$])
  })
})

In practice, for spin glass systems, the Cheeger constant provides a quantitative measure of how "glassy" the energy landscape is. Systems with small Cheeger constants have energy landscapes with high barriers between different metastable states, making equilibration difficult and necessitating techniques like parallel tempering to efficiently sample the state space.

#bibliography("refs.bib")