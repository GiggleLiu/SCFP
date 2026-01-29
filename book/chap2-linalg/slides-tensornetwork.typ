#import "@preview/touying:0.6.1": *
#import "@preview/touying-simpl-hkustgz:0.1.2": *
#import "@preview/cetz:0.4.1": *
#import "@preview/cetz-plot:0.1.2": plot
#import "@preview/algorithmic:0.1.0"
#import "@preview/ouset:0.2.0": ouset
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

#show raw.where(block: true): it=>{
  block(radius:4pt, fill:gray.transparentize(90%), inset:1em, width:99%, text(it))
}

#show: hkustgz-theme.with(
  config-info(
    title: [Tensor Networks],
    subtitle: [Applications in Machine Learning],
    author: [Jin-Guo Liu],
    date: datetime.today(),
    institution: [HKUST(GZ) - FUNH - Advanced Materials Thrust],
  ),
)

#let tensor(location, name, label) = {
  import draw: *
  circle(location, radius: 10pt, name: name)
  content((), text(black, label))
}

#let labelnode(loc, label, name: none) = {
  import draw: *
  content(loc, text(black, label), align: center, fill:silver, frame:"rect", padding:0.07, stroke: none, name: name)
}
#let labeledge(from, to, label, name: none) = {
  import draw: *
  line(from, to, name: "line")
  labelnode("line.mid", label, name: name)
}

#title-slide()
#outline-slide()

== Hands-on: Preparation

Goto the `ScientificComputingDemos` repository and pull the latest changes:
```bash
git pull
```
Then initialize the environment:
```bash
dir=TensorRenormalizationGroup make init
```

= Tensor network representation

== Definition
_Tensor network_ is a diagrammatic representation of multilinear maps.
- A tensor is represented as a node
- An index is represented as a hyperedge (a hyperedge can connect to any number of nodes)

For example, vectors, matrices and higher order tensors can be represented as

#align(center, text(10pt, canvas({
  import draw: *
  tensor((-7, 1), "V", [$V$])
  labeledge("V", (rel: (0, 1.5)), [$i$])
  content((rel: (0, -1), to: "V"), [Vector $V_i$])
  tensor((-3, 1), "M", [$M$])
  labeledge("M", (rel: (-1.5, 0)), [$i$])
  labeledge("M", (rel: (1.5, 0)), [$j$])
  content((rel: (0, -1), to: "M"), [Matrix $M_(i j)$])
  tensor((1, 1), "A", [$A$])
  labeledge("A", (rel: (1.5, 0)), [$i$])
  labeledge("A", (rel: (0, 1.5)), [$j$])
  labeledge("A", (rel: (-1.5, 0)), [$k$])
  content((rel: (0, -1), to: "A"), [Rank-3 tensor $A_(i j k)$])
})))

== Example 1: Matrix multiplication

$
C_(i k) = sum_j A_(i j) B_(j k),
$

Diagrammatic representation:
- The tensors associated with the same variable are connected by the same hyperedge.
- If a variable appears in the output tensor, the hyperedge is left _open_. Otherwise, it is _closed_ by connecting the two tensors.

#align(center, text(10pt, canvas({
  import draw: *
  tensor((-2, 1), "A", [$A$])
  tensor((0, 1), "B", [$B$])
  labeledge("A", (rel: (-1.5, 0)), [$i$])
  labeledge("A", (rel: (1.5, 0)), [$j$])
  labeledge("B", (rel: (1.5, 0)), [$k$])
})))

`einsum` (Einstein summation convention) notation: `ij,jk->ik`
- The intputs and outputs are separated by `->`.
- The indices of different input tensors are separated by commas: `ij,jk`.
- The indices not appearing in the output are summed over.

== Example 2: Proving trace permutation rule
Prove the trace permutation rule: $tr(A B C) = tr(C A B) = tr(B C A)$.

#figure(canvas({
  import draw: *
  tensor((1, 1), "A", "A")
  tensor((3, 1), "B", "B")
  tensor((5, 1), "C", "C")
  labeledge("A", "B", "j")
  labeledge("B", "C", "k")
  bezier("A.north", "C.north", (1, 3), (5, 3), name:"line")
  content("line.mid", "i", align: center, fill:white, frame:"rect", padding:0.1, stroke: none)
}))

The corresponding einsum notation is `ij,jk,ki->`.


== Example 3: Kronecker product

The kronecker product of two matrices $A_(i j)$ and $B_(k l)$, i.e. $A_(i j) times.circle B_(k l)$, can be diagrammatically represented as

#figure(canvas({
  import draw: *
  tensor((1, 1), "A", "A")
  tensor((3, 1), "B", "B")
  labeledge("A", (rel: (0, -1.5)), "j")
  labeledge("A", (rel: (0, 1.5)), "i")
  labeledge("B", (rel: (0, -1.5)), "l")
  labeledge("B", (rel: (0, 1.5)), "k")
  set-origin((5.5, 0))
  content((0, 1), $arrow$)
  set-origin((3, 0))
  content((0, 1), `ij,kl->ijkl`)
}))


== Live Coding: OMEinsum

In this example, we use the `OMEinsum` package to compute some simple tensor network contractions.

#align(center, text(10pt, canvas({
  import draw: *
  let s(it) = text(10pt, it)
  content((-5, 0.5), s[`ai,aj,ak->ijk` = \ (Star)])
  tensor((-1.0, 0), "A", s[$A$])
  tensor((1.0, 0), "B", s[$A$])
  tensor((0, 1.0), "C", s[$B$])
  labeledge("A", (rel: (-1.2, 0)), s[$i$])
  labeledge("B", (rel: (1.2, 0)), s[$j$])
  labeledge("C", (rel: (0, 1.2)), s[$k$])
  labelnode((0, 0), s[$a$], name: "a")
  line("a", "A")
  line("a", "B")
  line("a", "C")

  set-origin((-2, -2))
  content((-3.5, 0.5), [`ia,ajb,bkc,cld,dm->ijklm` = \ (Tensor train)])

  tensor((0, 0), "A", [$A$])
  tensor((1.5, 0), "B", [$T_1$])
  tensor((3, 0), "C", [$T_2$])
  tensor((4.5, 0), "D", [$T_1$])
  tensor((6, 0), "E", [$A$])
  labeledge("A", (rel: (0, 1.2)), [$i$])
  labeledge("B", (rel: (0, 1.2)), [$j$])
  labeledge("C", (rel: (0, 1.2)), [$k$])
  labeledge("D", (rel: (0, 1.2)), [$l$])
  labeledge("E", (rel: (0, 1.2)), [$m$])

  labeledge("A", "B", [$a$])
  labeledge("B", "C", [$b$])
  labeledge("C", "D", [$c$])
  labeledge("D", "E", [$d$])
})))


== Tensor network contraction orders

- The contraction complexity is determined by the chosen contraction order represented by a binary tree.
- Finding the optimal contraction order, i.e., the contraction order with minimal complexity, is NP-complete@Markov2008.

e.g. Multiplication of three matrices $A, B, C$:

$
&X = (A times B) times C = A times (B times C)\
&cal(V)(X) = (A times.circle C^T) cal(V)(B)
$

Q: What is the difference in the diagrammatic representation?

== Contraction tree and cost function

A _contraction tree_ is a binary tree that represents a contraction order of a tensor network.
#align(center, canvas(length:1.0cm, {
  import draw: *
  set-origin((4, 0.35))
  let DY = 1.2
  let DX1 = 1.5
  let DX2 = 0.9
  let root = (0, DY)
  let left = (-DX1, 0)
  let right = (DX1, 0)
  let left_left = (-DX1 - DX2, -DY)
  let left_right = (-DX1 + DX2, -DY)
  let right_left = (DX1 - DX2, -DY)
  let right_right = (DX1 + DX2, -DY)

  for (l, t, lb) in ((root, [$$], "C"), (left, [$I_12$], "A"), (right, [$I_34$], "B"), (left_left, [$T_1$], "T_1"), (left_right, [$T_2$], "T_2"), (right_left, [$T_3$], "T_3"), (right_right, [$T_4$], "T_4")){
    tensor(l, lb, text(11pt, t))
  }

  for (a, b) in (("C", "A"), ("C", "B"), ("A", "T_1"), ("A", "T_2"), ("B", "T_3"), ("B", "T_4")){
    line(a, b)
  }


}))

The cost function of a contraction tree is defined as
$
  cal(L) = "tc" + w_s "sc" + w_("rw") "rwc",
$
- $w_s$ and $w_("rw")$ are the weights of the space complexity and read-write complexity compared to the time complexity, respectively.


== Trade-off between contraction order optimization and contraction

#figure(canvas(length:0.9cm, {
  import draw: *
  let s(it) = text(11pt, it)
  plot.plot(size: (10,7),
    x-tick-step: none,
    y-tick-step: none,
    x-label: text(13pt)[Time to optimize contraction order],
    y-label: text(13pt)[Time to contract],
    y-max: 10,
    y-min: -2,
    x-max: 10,
    x-min: 0,
    name: "plot",
    {
      let greedy = (1, 9)
      let localsearch = (4, 3)
      let bipartition = (3, 4)
      let tw = (9, 1)
      let tamaki = (5, 2)
      plot.add(
        (greedy, bipartition, localsearch, tamaki, tw), style: (stroke: black), mark:"o",
      )
      plot.add-anchor("greedy", greedy)
      plot.add-anchor("localsearch", localsearch)
      plot.add-anchor("bipartition", bipartition)
      plot.add-anchor("tw", tw)
      plot.add-anchor("tamaki", tamaki)
    }
  )
  content((rel: (2.5, 0), to: "plot.greedy"), s[Greedy (`GreedyMethod`)])
  content((rel: (2.5, 0), to: "plot.localsearch"), s[Local Search (`TreeSA`)])
  content((rel: (3.0, 0), to: "plot.bipartition"), s[Min cut (`KaHyParBipartite`)])
  content((rel: (0, -0.8), to: "plot.tw"), box(fill: white, inset: 1pt, s[Exact tree-width (`ExactTreewidth`)\ State compression]))
  content((rel: (-1.0, -0.4), to: "plot.tamaki"), box(fill: white, s[Positive instance driven], inset: 1pt))
}))


== Slicing tensor networks

Slicing is a technique to reduce the space complexity of the tensor network by looping over a subset of indices.


#figure(canvas({
  import draw: *
  let points = ((0, 0), (0, 1), (1, 0), (1, 1), (0, -1), (-2, 1), (-1, 0), (-1, 1))
  let edges = (("0", "1"), ("0", "2"), ("0", "4"), ("1", "2"), ("1", "3"), ("2", "3"), ("1", "7"), ("1", "6"), ("7", "5"), ("2", "4"), ("4", "6"), ("5", "6"), ("6", "7"))
  for (k, loc) in points.enumerate() {
    circle(loc, radius: 0.2, name: str(k), fill: black)
  }
  for (k, (a, b)) in edges.enumerate() {
    line(a, b, name: "e"+str(k), stroke: (if k == 4 {(paint: red, thickness: 2pt)} else {black}))
  }
  content((rel: (0, 0.5), to: "e4.mid"), text(14pt)[$i$])
  
  set-origin((7.5, 0))
  line((-5.5, 0), (-4.5, 0), mark: (end: "straight"))
  content((-5, 0.4), text(14pt)[slicing])
  content((-3, 0), text(14pt)[$sum_i$])
  for (k, loc) in points.enumerate() {
    circle(loc, radius: 0.2, name: str(k), fill: black)
  }
  for (k, (a, b)) in edges.enumerate() {
    line(a, b, name: "e"+str(k), stroke: (if k == 4 {(dash: "dashed")} else {black}))
  }
  content((rel: (0, 0.5), to: "e4.mid"), text(14pt)[$i$])
}))

- _Remark_: The slicing technique may increase the time complexity.



== Graph theoretical point of view: Tree decomposition
The optimal contraction order is closely related to the _tree decomposition_@Markov2008 of the line graph of the tensor network.

#figure(canvas({
  import draw: *
  let d = 1.1
  let s(it) = text(11pt, it)
  let locs_labels = ((0, 0), (d, 0), (0, -d), (0, -2 * d), (d, -2 * d), (2 * d, 0), (2 * d, -d), (2 * d, -2 * d))
  for (loc, t, name) in (((0.5 * d, -0.5 * d), s[$T_1$], "T_1"), ((1.5 * d, -0.5 * d), s[$T_2$], "T_2"), ((1.5 * d, -1.5 * d), s[$T_3$], "T_3"), ((0.5 * d, -1.5 * d), s[$T_4$], "T_4")) {
    circle(loc, radius: 0.3, name: name)
    content(loc, s[#t])
  }
  for ((loc, t), name) in locs_labels.zip((s[$A$], s[$B$], s[$C$], s[$D$], s[$E$], s[$F$], s[$G$], s[$H$])).zip(("A", "B", "C", "D", "E", "F", "G", "H")) {
    labelnode(loc, t, name: name)
  }
  for (src, dst) in (("A", "T_1"), ("B", "T_1"), ("C", "T_1"), ("F", "T_2"), ("G", "T_2"), ("B", "T_2"), ("H", "T_3"), ("E", "T_3"), ("G", "T_3"), ("D", "T_4"), ("C", "T_4"), ("E", "T_4")) {
    line(src, dst)
  }
  content((d, -3), text(12pt)[(a)])
  content((3.5, -1), text(12pt)[$arrow.double.r$])
  content((3.5, -1.5), text(10pt)[Line graph])
  set-origin((5, 0))
  let colors = (color.hsv(30deg, 90%, 70%), color.hsv(120deg, 90%, 70%), color.hsv(210deg, 90%, 70%), color.hsv(240deg, 90%, 70%), color.hsv(330deg, 90%, 70%), color.hsv(120deg, 90%, 70%), color.hsv(210deg, 90%, 70%), color.hsv(240deg, 90%, 70%))
  let texts = ("A", "B", "C", "D", "E", "F", "G", "H")
  for (loc, color, t) in locs_labels.zip(colors, texts) {
    circle(loc, radius: 0.3, name: t)
    content(loc, text(12pt, color)[#t])
  }
  for (a, b) in (("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("C", "E"), ("D", "E"), ("E", "G"), ("G", "H"), ("E", "H"), ("F", "G"), ("F", "B"), ("B", "G")) {
    line(a, b)
  }
  content((d, -3), text(12pt)[(b)])
  content((3.5, -1), text(12pt)[$arrow.double.r$])
  content((3.5, -1.5), text(10pt)[T. D.])
  set-origin((5, 0))
  for (loc, bag) in (((0, 0), "B1"), ((0, -2), "B2"), ((1, -1), "B3"), ((3, -1), "B4"), ((4, 0), "B5"), ((4, -2), "B6")) {
    circle(loc, radius: 0.55, name: bag)
  }
  let topleft = (-0.2, 0.2)
  let topright = (0.2, 0.2)
  let bottom = (0, -0.3)
  let top = (0, 0.3)
  let bottomleft = (-0.2, -0.2)
  let bottomright = (0.2, -0.2)
  let right = (0.3, 0)
  let left = (-0.3, 0)
  content((rel:topright, to: "B1"), text(10pt, colors.at(1))[B], name: "b1")
  content((rel:topleft, to: "B1"), text(10pt, colors.at(0))[A], name: "a1")
  content((rel:bottom, to: "B1"), text(10pt, colors.at(2))[C], name: "c1")

  content((rel:top, to: "B2"), text(10pt, colors.at(2))[C], name: "c2")
  content((rel:bottomleft, to: "B2"), text(10pt, colors.at(3))[D], name: "d1")
  content((rel:right, to: "B2"), text(10pt, colors.at(4))[E], name: "e1")

  content((rel:topright, to: "B3"), text(10pt, colors.at(1))[B], name: "b2")
  content((rel:left, to: "B3"), text(10pt, colors.at(2))[C], name: "c3")
  content((rel:bottomright, to: "B3"), text(10pt, colors.at(4))[E], name: "e2")

  content((rel:topleft, to: "B4"), text(10pt, colors.at(1))[B], name: "b3")
  content((rel:bottomleft, to: "B4"), text(10pt, colors.at(4))[E], name: "e3")
  content((rel:right, to: "B4"), text(10pt, colors.at(6))[G], name: "g1")

  content((rel:left, to: "B5"), text(10pt, colors.at(1))[B], name: "b4")
  content((rel:topright, to: "B5"), text(10pt, colors.at(5))[F], name: "f1")
  content((rel:bottom, to: "B5"), text(10pt, colors.at(6))[G], name: "g2")

  content((rel:left, to: "B6"), text(10pt, colors.at(4))[E], name: "e4")
  content((rel:top, to: "B6"), text(10pt, colors.at(6))[G], name: "g3")
  content((rel:bottomright, to: "B6"), text(10pt, colors.at(7))[H], name: "h1")

  line("b1", "b2", stroke: colors.at(1))
  line("b2", "b3", stroke: colors.at(1))
  line("b3", "b4", stroke: colors.at(1))
  line("c1", "c3", stroke: colors.at(2))
  line("c2", "c3", stroke: colors.at(2))
  line("e1", "e2", stroke: colors.at(4))
  line("e2", "e3", stroke: colors.at(4))
  line("e3", "e4", stroke: colors.at(4))
  line("g1", "g2", stroke: colors.at(6))
  line("g1", "g3", stroke: colors.at(6))
  content((2, -3), text(12pt)[(c)])
}),
)
#enum(numbering: "(a)", [A tensor network.], [A line graph for the tensor network. Labels are connected if and only if they appear in the same tensor.], [A tree decomposition (T. D.) of the line graph.])


= Compute the partition function of spin glass model

== Partition function of spin glass model
The partition function of a spin glass model on a graph $G = (V, E)$ is defined as
$
Z = sum_(bold(s)) exp(-beta H(bold(s))),
$
where $beta$ is the inverse temperature, $bold(s) = {s_i}_(i in V)$ are the spin variables, and the Hamiltonian is given by
$
H(bold(s)) = - sum_( (i,j) in E ) J_(i j) s_i s_j - sum_(i in V) h_i s_i,
$
where $J_(i j)$ are the coupling constants and $h_i$ are the external fields.

== Partition function to tensor network

$
  Z = sum_(bold(s)) product_( (i,j) in E ) e^(-beta J_(i j) s_i s_j) product_(i in V) e^(-beta h_i s_i)
$

It corresponds to the following tensor network:
$
  cases(Lambda = {s_i | i in V},
  cal(T) = {e^(-beta J_(i j) s_i s_j) | (i, j) in E} union {e^(-beta h_i s_i) | i in V},
  V_0 = emptyset
  )
$

== Set notation of tensor networks

A tensor network can be represented by a triple of $(Lambda, cal(T), V_0)$@Liu2022@Roa2024, where:
- $Lambda$ is the set of variables present in the network.
- $cal(T) = { T_(V_k) }_(k=1)^K$ is the set of input tensors, where each tensor $T_(V_k)$ is associated with the labels $V_k$.
- $V_0$ specifies the labels of the output tensor.

Specifically, each tensor $T_(V_k) in cal(T)$ is labeled by a set of variables $V_k subset.eq Lambda$, where the cardinality $|V_k|$ equals the rank of $T_(V_k)$.

== Tensor contraction
The multilinear map, or the *contraction*, applied to this triple is defined as
$
T_(V_0) = "contract"(Lambda, cal(T), V_0) ouset(=, "def") sum_(m in cal(D)_(Lambda without V_0)) product_(T_V in cal(T)) T_(V|M=m),
$
where $M = Lambda without V_0$. $T_(V|M=m)$ denotes a slicing of the tensor $T_V$ with the variables $M$ fixed to the values $m$. The summation runs over all possible configurations of the variables in $M$.

== Example: Matrix multiplication
Matrix multiplication can be described as the contraction of a tensor network given by
$
C_(i k) = "contract"({i,j,k}, {A_(i j), B_(j k)}, {i, k}),
$
where the input matrices $A$ and $B$ are indexed by the variable sets ${i, j}, {j, k}$, respectively, which are subsets of $Lambda = {i, j, k}$. As a remark of notation, when an set is used as subscripts, we omit the comma and the braces. The output tensor is indexed by variables ${i, k}$ and the summation runs over variables $Lambda without {i, k} = {j}$. 

== Live Coding: AFM spin glass partition function

Graph topology is as follows, each line represents a AFM coupling $J = 1$:
#canvas({
  import draw: *

  let d = 1.1
  let s(it) = text(11pt, it)
  let locs_labels = ((0, 0), (d, 0), (0, -d), (0, -2 * d), (d, -2 * d), (2 * d, 0), (2 * d, -d), (2 * d, -2 * d))
  let texts = ("A", "B", "C", "D", "E", "F", "G", "H")
  for (loc, t) in locs_labels.zip(texts) {
    circle(loc, radius: 0.2, name: t, fill: black)
  }
  for (a, b) in (("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("C", "E"), ("D", "E"), ("E", "G"), ("G", "H"), ("E", "H"), ("F", "G"), ("F", "B"), ("B", "G")) {
    line(a, b)
  }
})

1. The bruteforce enumeration approach.
2. The tensor network approach.


== Hands-on: Tensor renormalization group (TRG)

Goal: Consider a inifitely large spin glass model on a square lattice. Compute the partition function: $ln(Z)$ per site.
1. Write down its tensor network representation.
2. Compute the partition function recursively through coarse graining.

== TRG - Part 1: Preparation of tensor network

#figure(canvas({
  import draw: *
  let s(it) = text(14pt, it)
  
  // Original lattice
  content((-5.5, 2), s[Step 1: Original lattice\ (Note: extends to infinity)])
  let d = 1.0
  for i in range(4) {
    for j in range(4) {
      circle((-7 + i*d, 0.5 - j*d), radius: 0.15, name: "t"+str(i)+str(j), fill: black)
    }
  }
  // Draw horizontal connections
  for j in range(4) {
    for i in range(3) {
      line("t"+str(i)+str(j), "t"+str(i+1)+str(j))
    }
  }
  // Draw vertical connections
  for i in range(4) {
    for j in range(3) {
      line("t"+str(i)+str(j), "t"+str(i)+str(j+1))
    }
  }
  
  set-origin((8, 0))
  content((-6, 2), s[Step 2: Convert to tensor network\ Q: what are the tensor elements?])
  let d = 1.0
  for i in range(4) {
    for j in range(4) {
      circle((-7 + i*d, 0.5 - j*d), radius: 0, name: "t"+str(i)+str(j), fill: none)
    }
  }
  // Draw horizontal connections
  for j in range(4) {
    for i in range(3) {
      line("t"+str(i)+str(j), "t"+str(i+1)+str(j), name: "line"+str(i)+str(j))
      circle("line"+str(i)+str(j)+".mid", fill: white, radius: 0.2)
    }
  }
  // Draw vertical connections
  for i in range(4) {
    for j in range(3) {
      line("t"+str(i)+str(j), "t"+str(i)+str(j+1), name: "line"+str(i)+str(j))
      circle("line"+str(i)+str(j)+".mid", fill: white, radius: 0.2)
    }
  }
  circle((-6, -1.5), radius: 0.8, stroke: (dash: "dashed"))
 
  // Original lattice
  set-origin((8, 0))
  content((-5.5, 2), s[Step 3: Star contraction\ Q: what is the einsum notation\ for this star contraction?])
  let d = 1.0
  for i in range(4) {
    for j in range(4) {
      circle((-7 + i*d, 0.5 - j*d), radius: 0.2, name: "t"+str(i)+str(j), fill: white)
    }
  }
  // Draw horizontal connections
  for j in range(4) {
    for i in range(3) {
      line("t"+str(i)+str(j), "t"+str(i+1)+str(j))
    }
  }
  // Draw vertical connections
  for i in range(4) {
    for j in range(3) {
      line("t"+str(i)+str(j), "t"+str(i)+str(j+1))
    }
  }
}))
 
== TRG - Part 2: Coarse graining via SVD
Let us focus on a $2 times 2$ block:
#figure(canvas({
  import draw: *
  let s(it) = text(14pt, it)
  // Step 1: SVD decomposition
  content((3, 3), s[Step 1: Eigen decomposition on each tensor: $A = U S U^dagger$, then insert blue tensors as $U sqrt(S)$])
  let locs = ((1, -1), (1, 1), (-1, 1), (-1, -1))
  for (k, loc) in locs.enumerate() {
    circle(loc, radius: 0.3, name: "T" + str(k), fill: white)
    line("T" + str(k), (rel: (loc.at(0) * 0.8, 0)))
    line("T" + str(k), (rel: (0, loc.at(1) * 0.8)))
  }
  line("T0", "T1")
  line("T1", "T2")
  line("T2", "T3")
  line("T3", "T0")
  
  set-origin((6, 0))
  let locs = ((1, -1), (1, 1), (-1, 1), (-1, -1))
  for (k, loc) in locs.enumerate() {
    circle(loc.map(x => x * 0.7), radius: 0.3, name: "A" + str(k), fill: blue)
    circle(loc.map(x => x / 0.7), radius: 0.3, name: "B" + str(k), fill: blue)
    line("A" + str(k), "B" + str(k))
    line("B" + str(k), (rel: (loc.at(0) * 0.8, 0)))
    line("B" + str(k), (rel: (0, loc.at(1) * 0.8)))
  }
  line("A0", "A1")
  line("A1", "A2")
  line("A2", "A3")
  line("A3", "A0")
  circle((1.08, 1.08), radius: 1, stroke: (dash: "dashed"))
 
  set-origin((-10, -6.0))

  content((2, 3), s[Step 2: Contract and form new tensors])
  
  for (k, loc) in locs.enumerate() {
    circle(loc.map(x => x * 0.7), radius: 0.3, name: "A" + str(k), fill: blue)
    line("A" + str(k), (rel: (loc.at(0) * 0.5, loc.at(1) * 0.5)))
  }
  line("A0", "A1")
  line("A1", "A2")
  line("A2", "A3")
  line("A3", "A0")
  circle((0, 0), radius: 1.4, stroke: (dash: "dashed"))
  content((3, 0), [$arrow.r$])
  circle((5, 0), radius: 0.3, name: "A", fill: green)
  line("A", (rel: (0.8, 0.8)))
  line("A", (rel: (-0.8, 0.8)))
  line("A", (rel: (0.8, -0.8)))
  line("A", (rel: (-0.8, -0.8)))
 
  set-origin((12, 0))
  content((0, 2.5), s[Step 3: We got a new lattice\ Q: what is its size?])
  for (k, loc) in ((1, 0), (0, 1), (-1, 0), (0, -1)).enumerate() {
    circle(loc, radius: 0.3, fill: green, name: "T" + str(k))
    line("T" + str(k), (rel: (0.7, 0.7)))
    line("T" + str(k), (rel: (-0.7, 0.7)))
    line("T" + str(k), (rel: (0.7, -0.7)))
    line("T" + str(k), (rel: (-0.7, -0.7)))
  }
  line("T0", "T1")
  line("T1", "T2")
  line("T2", "T3")
  line("T3", "T0")
}))

== Question for you

- Can we use a more direct approach?
#figure(canvas({
  import draw: *
  let s(it) = text(14pt, it)
  // Step 1: SVD decomposition
  let locs = ((1, -1), (1, 1), (-1, 1), (-1, -1))
  for (k, loc) in locs.enumerate() {
    circle(loc, radius: 0.3, name: "T" + str(k), fill: white)
    line("T" + str(k), (rel: (loc.at(0) * 0.8, 0)))
    line("T" + str(k), (rel: (0, loc.at(1) * 0.8)))
  }
  line("T0", "T1")
  line("T1", "T2")
  line("T2", "T3")
  line("T3", "T0")
  content((4, 0), [$arrow.r$])
  
  set-origin((7, 0))
  circle((0, 0), radius: 0.4, fill: red, name: "A")
  line("A", (rel: (-1, 0)))
  line("A", (rel: (0, -1)))
  line("A", (rel: (1, 0)))
  line("A", (rel: (0, 1)))
}))

- We need to truncate the bond dimension after each SVD/Eigen decomposition. Is that optimal way to truncated the bond dimension?
 
= Data compression
== References:
- Era of big data processing: a new approach via tensor networks and tensor decompositions @Cichocki2014

== Singular value decomposition - revisited
Let us define a complex matrix $A in CC^(m times n)$, and let its singular value decomposition be
$
A = U S V^dagger
$
where $U$ and $V$ are unitary matrices and $S$ is a diagonal matrix with non-negative real numbers on the diagonal.

Q: What is the diagrammatic representation of the SVD?

== CP-decomposition

For example, the CP-decomposition of a rank-4 tensor $T$ can be represented as
$
T_(i j k l) = sum_(c) U_1^(i c) U_2^(j c) U_3^(k c) U_4^(l c) Lambda_(c)
$

#align(center, text(10pt, canvas({
  import draw: *
  tensor((-5.5, 0), "T", [$T$])
  labeledge("T", (rel: (0, 1.2)), [$i$])
  labeledge("T", (rel: (-1.2, 0)), [$j$])
  labeledge("T", (rel: (0, -1.2)), [$k$])
  labeledge("T", (rel: (1.2, 0)), [$l$])

  content((-3.5, 0), [$=$])

  tensor((-1.0, 0), "A", [$U_1$])
  tensor((1.0, 0), "B", [$U_2$])
  tensor((0, -1.0), "C", [$U_3$])
  tensor((0, 1.0), "D", [$U_4$])
  tensor((1, 1), "L", [$Lambda$])
  labeledge("D", (rel: (0, 1.2)), [$i$])
  labeledge("A", (rel: (-1.2, 0)), [$j$])
  labeledge("C", (rel: (0, -1.2)), [$k$])
  labeledge("B", (rel: (1.2, 0)), [$l$])
  labelnode((0, 0), [$c$], name: "c")
  line("c", "D")
  line("c", "C")
  line("c", "B")
  line("c", "A")
  line("c", "L")
})))

== Tucker decomposition

The Tucker decomposition of a rank-4 tensor $T$ can be represented as
$
T_(i j k l) = sum_(a,b,c,d) U_1^(i a) U_2^(j b) U_3^(k c) U_4^(l d) X_(a b c d)
$
where $U_1, U_2, U_3, U_4$ are unitary matrices and $X$ is a rank-4 tensor.

#align(center, text(10pt, canvas({
  import draw: *
  tensor((-5.5, 0), "T", [$T$])
  labeledge("T", (rel: (0, 1.2)), [$i$])
  labeledge("T", (rel: (-1.2, 0)), [$j$])
  labeledge("T", (rel: (0, -1.2)), [$k$])
  labeledge("T", (rel: (1.2, 0)), [$l$])

  content((-3.5, 0), [$=$])


  tensor((-1.5, 0), "A", [$U_1$])
  tensor((1.5, 0), "B", [$U_2$])
  tensor((0, -1.5), "C", [$U_3$])
  tensor((0, 1.5), "D", [$U_4$])
  tensor((0, 0), "X", [$X$])
  labeledge("D", (rel: (0, 1.2)), [$i$])
  labeledge("A", (rel: (-1.2, 0)), [$j$])
  labeledge("C", (rel: (0, -1.2)), [$k$])
  labeledge("B", (rel: (1.2, 0)), [$l$])
  labeledge("X", "A", [$b$])
  labeledge("X", "B", [$d$])
  labeledge("X", "C", [$c$])
  labeledge("X", "D", [$a$])
})))


== Tensor train decomposition

#align(center, text(10pt, canvas({
  import draw: *
  tensor((-3.5, 0), "T", [$T$])
  labeledge("T", (rel: (0, 1.2)), [$i$])
  labeledge("T", (rel: (-1.2, 0)), [$j$])
  labeledge("T", (rel: (0, -1.2)), [$k$])
  labeledge("T", (rel: (1.2, 0)), [$l$])

  content((-1.5, 0), [$=$])

  tensor((0, 0), "A", [$U_1$])
  tensor((1.5, 0), "B", [$U_2$])
  tensor((3, 0), "C", [$U_3$])
  tensor((4.5, 0), "D", [$A_4$])
  labeledge("A", (rel: (0, 1.2)), [$i$])
  labeledge("B", (rel: (0, 1.2)), [$j$])
  labeledge("C", (rel: (0, 1.2)), [$k$])
  labeledge("D", (rel: (0, 1.2)), [$l$])

  labeledge("A", "B", [$a$])
  labeledge("B", "C", [$b$])
  labeledge("C", "D", [$c$])
})))

Live coding: Tensor train decomposition for compressing a uniform tensor of size $2^20$.

= Probabilistic modeling
== Hidden Markov model

A Hidden Markov Model (HMM)@Bishop2006 is a simple probabilistic graphical model that describes a Markov process with unobserved (hidden) states:

- A sequence of hidden states $z_t$ following a Markov chain with transition probability $P(z_(t+1)|z_t)$
- A sequence of observations $x_t$ that depend only on the current hidden state through emission probability $P(x_t|z_t)$

The joint probability of a sequence of $T+1$ hidden states and $T$ observations can be written as:

$
P(bold(z), bold(x)) = P(z_0) product_(t=1)^T P(z_(t)|z_(t-1))P(x_t|z_t).
$
Note that the conditional probability $P(z_(t)|z_(t-1))$ can be represented as a tensor with two indices. The joint probability $P(bold(z), bold(x))$ can be represented as a tensor network diagram:


== Tensor network representation
The tensor network representation of a Hidden Markov Model (HMM) with observed variables $x_1, x_2, dots, x_T$ and hidden states $z_0, z_1, dots, z_T$. The circles are conditional probabilities $P(z_t|z_(t-1))$ and $P(x_t|z_t)$.
#let hmm(n) = {
  import draw: *
  let s(it) = text(11pt, it)
  
  // Draw transition matrices
  let dx = 2.0
  tensor((0, 0), "A0", []) 
  for i in range(1, n){
    tensor((dx*i, 0), "A" + str(i), []) 
  }
  for i in range(n - 1){
   labeledge("A" + str(i), "A" + str(i+1), s[$z_#(i+1)$], name: "z" + str(i))
  }
  labeledge("A" + str(n - 1), (rel: (1.6, 0), to:"A" + str(n - 1)), s[$z_#(n)$], name: "z" + str(n - 1))

  for i in range(n){
    tensor((rel: (0, -1), to: "z" + str(i)), "B" + str(i), [])
    line("z" + str(i), "B" + str(i))
    labeledge("B" + str(i), (rel: (0, -1.2)), s[$x_#(i+1)$])
  }
}

#figure(canvas({
  import draw: *
  hmm(5)
}),
)

== Likelihood
The likelihood of the observed sequence:
$
P(bold(x)|theta) = sum_(bold(z)) P(bold(x), bold(z)|theta)
$

#figure(canvas({
  import draw: *
  hmm(5)
  let s(it) = text(11pt, it)
  for i in range(5){
    tensor((rel: (0, -1.6), to: "B" + str(i)), "p" + str(i), s[$x_#(i+1)$])
  }
  tensor((rel: (2, 0), to: "A4"), "e", [id])
}))
where nodes with $x_t$ are observed variables, which are represented as projection tensors.

== Tensor network for likelihood

Let $overshell(x)_t$ denotes an observed variable $x_t$ with a fixed value. It is equivalent to contracting the following tensor network:
$
  cases(Lambda = {z_0, z_1, dots, z_T},
  cal(T) = {P(z_0), P(z_1|z_0), dots, P(z_T|z_(T-1)), P(overshell(x)_1|z_1), P(overshell(x)_2|z_2), dots, P(overshell(x)_T|z_T)},
  V_0 = emptyset
  )
$ <eq:decoding-tensor>
Since $overshell(x)_1, overshell(x)_2, dots, overshell(x)_T$ are fixed and not involved in the contraction, $P(overshell(x)_t|z_t)$ is a vector indexed by $z_t$ rather than a matrix.

#figure(box(inset: 10pt, stroke: black)[a function over finite domain $arrow.l.r.double$ a tensor])


== Decoding
This is the _decoding problem_ of HMM: Given a sequence of observations $bold(x) = (x_1, x_2, ..., x_T)$, how to find the most likely sequence of hidden states $bold(z)$? The equivalent mathematical formulation is:
$
  arg max_(bold(z)) P(z_0) product_(t=1)^T P(z_(t)|z_(t-1))P(overshell(x)_t|z_t),
$ <eq:decoding>
To solve it, we first convert the above tensor network into the following form:
$
  arg max_(bold(z)) sum_(bold(z)) log P(z_0) + sum_(t=1)^T log P(z_t|z_(t-1)) + sum_(t=1)^T log P(overshell(x)_t|z_t),
$

It is equivalent to a tropical tensor network@Liu2021 $(Lambda, {log(t) | t in cal(T)}, V_0)$, where $log(t)$ is element-wise logarithm of $t$.

== Baum-Welch algorithm
The Baum-Welch algorithm is an expectation-maximization (EM) algorithm used to find the unknown parameters of a Hidden Markov Model (HMM). It addresses the _learning problem_ of HMM:

#figure(box(inset: 10pt, stroke: black)[
Given a sequence of observations $bold(x) = (x_1, x_2, ..., x_T)$, how to estimate the model parameters $theta = (A, B, pi)$, where $A$ is the transition probability matrix, $B$ is the emission probability matrix, and $pi$ is the initial state distribution?
])

==

#figure(canvas({
  import draw: *
  hmm(5)
  let s(it) = text(11pt, it)
  for i in range(5){
    tensor((rel: (0, -1.6), to: "B" + str(i)), "p" + str(i), s[$x_#(i+1)$])
    content("B"+str(i), s[$B$])
    if i == 0{
      content("A"+str(i), s[$pi$])
    }
    else{
      content("A"+str(i), s[$A$])
    }
  }
  line("z1", (rel: (0, 1), to: "z1"), stroke: blue)
  content((rel: (0, 1.3), to: "z1"), text(11pt, blue)[$eta_2 (x_2, z_2)$])
  line("z2", (rel: (0, 1), to: "z2"), stroke: red)
  line("z3", (rel: (0, 1), to: "z3"), stroke: red)
  content((rel: (0, 1.3), to: "A3"), text(11pt, red)[$xi_3 (z_3, z_4)$])
  tensor((rel: (2, 0), to: "A4"), "e", s[id])
}))

The transition probability from state $i$ to state $j$ is given by
$
xi_t (i,j) = P(z_t=i, z_(t+1)=j | bold(x), theta)
$ <eq:transition-probability>

The emission probability from state $i$ to symbol $k$ is given by
$
eta_t (i,k) = P(x_t=k | z_t=i, theta)
$ <eq:emission-probability>

== Backward-mode automatic differentiation
In practice, to evaluate tensor networks with multiple open indices, we can utilize the backward-mode automatic differentiation.

=== Differentiation - cut correspondence

#figure(canvas({
  import draw: *
  let s(it) = text(11pt, it)
  content((-1, 0), s[$X = $])
  tensor((0, 0), "A", s[$U_1$])
  tensor((1.5, 0), "B", s[$U_2$])
  tensor((3, 0), "C", s[$U_3$])
  tensor((4.5, 0), "D", s[$A_4$])
  labeledge("A", (rel: (0, 1.2)), s[$i$])
  labeledge("B", (rel: (0, 1.2)), s[$j$])
  labeledge("C", (rel: (0, 1.2)), s[$k$])
  labeledge("D", (rel: (0, 1.2)), s[$l$])

  labeledge("A", "B", s[$a$])
  labeledge("B", "C", s[$b$])
  labeledge("C", "D", s[$c$])
}))

#figure(canvas({
  import draw: *
  let s(it) = text(11pt, it)
  content((-1, 0), s[$frac(partial X, partial U_2) = $])
  tensor((0, 0), "A", s[$U_1$])
  circle((1.5, 0), radius: 0.3, name: "B", stroke: none)
  tensor((3, 0), "C", s[$U_3$])
  tensor((4.5, 0), "D", s[$A_4$])
  labeledge("A", (rel: (0, 1.2)), s[$i$])
  labeledge("B", (rel: (0, 1.2)), s[$j$])
  labeledge("C", (rel: (0, 1.2)), s[$k$])
  labeledge("D", (rel: (0, 1.2)), s[$l$])

  labeledge("A", "B", s[$a$])
  labeledge("B", "C", s[$b$])
  labeledge("C", "D", s[$c$])
}))

#figure(canvas({
  import draw: *
  let s(it) = text(11pt, it)
  content((-1.5, 0), s[$overline(X)frac(partial X, partial U_2) = $])
  tensor((0, 0), "A", s[$U_1$])
  circle((1.5, 0), radius: 0.3, name: "B", stroke: none)
  tensor((3, 0), "C", s[$U_3$])
  tensor((4.5, 0), "D", s[$A_4$])
  labeledge("A", (rel: (0, 1.2)), s[$i$])
  labeledge("B", (rel: (0, 1.2)), s[$j$])
  labeledge("C", (rel: (0, 1.2)), s[$k$])
  labeledge("D", (rel: (0, 1.2)), s[$l$])

  labeledge("A", "B", s[$a$])
  labeledge("B", "C", s[$b$])
  labeledge("C", "D", s[$c$])
  rect((-0.5, 1.2), (5, 1.8))
  content((2.25, 1.5), s[$overline(X)$])
}))


== Tensor network differentiation
Let $(Lambda, cal(T), emptyset)$ be a tensor network with scalar output. The gradient of the tensor network contraction with respect to $T_V in cal(T)$ is
$
  frac(partial "contract"(Lambda, cal(T), emptyset), partial T_V) =
  "contract"(Lambda, cal(T) \\ {T_V}, V).
$
That is, the gradient corresponds to the contraction of the tensor network
with the tensor $T_V$ removed and the output label set to $V$.

== Proof
Let $cal(L)$ be a loss function of interest, where its differential form is given by:
$
delta cal(L) = "contract"(V_a, {delta A_(V_a), overline(A)_(V_a)}, emptyset) + "contract"(V_b, {delta B_(V_b), overline(B)_(V_b)}, emptyset)
$ <eq:diffeq>

The goal is to find $overline(A)_(V_a)$ and $overline(B)_(V_b)$ given $overline(C)_(V_c)$.
This can be achieved by using the differential form of tensor contraction, which states that
$
delta C = "contract"(Lambda, {delta A_(V_a), B_(V_b)}, V_c) + "contract"(Lambda, {A_(V_a), delta B_(V_b)}, V_c).
$
By inserting this result into the above equation, we obtain:
$
delta cal(L) = &"contract"(V_a, {delta A_(V_a), overline(A)_(V_a)}, emptyset) + "contract"(V_b, {delta B_(V_b), overline(B)_(V_b)}, emptyset)\
= &"contract"(Lambda, {delta A_(V_a), B_(V_b), overline(C)_(V_c)}, emptyset) + "contract"(Lambda, {A_(V_a), delta B_(V_b), overline(C)_(V_c)}, emptyset).
$
Since $delta A_(V_a)$ and $delta B_(V_b)$ are arbitrary, the above equation immediately implies:

$
overline(A)_(V_a) = "contract"(Lambda, {overline(C)_(V_c), B_(V_b)}, V_a)\
overline(B)_(V_b) = "contract"(Lambda, {A_(V_a), overline(C)_(V_c)}, V_b)
$




== Parameter Update

$
A_(i j) = (sum_(t=1)^(T-1) xi_t (i,j))/(sum_(t=1)^(T-1) sum_(j=1)^N xi_t (i,j))
$

// This equation updates the transition probability matrix $A$. For each pair of states $(i,j)$, we compute the expected number of transitions from state $i$ to state $j$ (numerator) divided by the expected total number of transitions from state $i$ to any state (denominator).

$
B_(i k) = (sum_(t=1)^T eta_t (i,k) dot I(x_t = k))/(sum_(t=1)^T eta_t (i,k))
$

// This equation updates the emission probability matrix $B$. For each state $i$ and observation $k$, we compute the expected number of times the model emits observation $k$ while in state $i$ (numerator) divided by the expected total number of times the model is in state $i$ (denominator). The indicator function $I(x_t = k)$ equals 1 when the observation at time $t$ is $k$, and 0 otherwise.

- _Remark:_ The Baum-Welch algorithm does not guarantee to find the global maximum of the likelihood function.

== Hands-on: Hidden Markov Model

- Run the example code in folder `HiddenMarkovModel` of our demo repository.
  ```bash
  dir=HiddenMarkovModel make init
  dir=HiddenMarkovModel make example
  ```

- Tasks:
  - Watch #link("https://www.youtube.com/watch?v=i3AkTO9HLXo&list=PLM8wYQRetTxBkdvBtz-gw8b9lcVkdXQKV&ab_channel=NormalizedNerd", [YouTube playlist: Markov Chains Clearly Explained!]) and try to understand the code.
  - Complete the second problem in Homework 11.

= Solving spin-glass problem

The partition function of a physical system plays a central role in statistical physics. It is closely related to many physical quantities, such as the free energy, the entropy, and the specific heat.
The spin glass Hamiltonian on a graph $G = (V, E)$ is given by
$
H(bold(s)) = sum_((i, j) in E) J_(i j) s_i s_j + sum_(i in V) h_i s_i,
$
where $s_i$ is the spin of the $i$-th spin.
It partition function is given by
$
Z = sum_(bold(s)) e^(-beta E(bold(s))) = sum_(bold(s)) product_((i, j) in E) e^(-beta J_(i j) s_i s_j) product_(i in V) e^(-beta h_i s_i),
$ <eq:partition-function>
where $beta$ is the inverse temperature.

== Tensor network representation
It corresponds to the following tensor network:
$
  cases(Lambda = {s_i | i in V},
  cal(T) = {e^(-beta J_(i j) s_i s_j) | (i, j) in E} union {e^(-beta h_i s_i) | i in V},
  V_0 = emptyset
  )
$

== Tropical numbers - revisited
It is also possible to compute the ground state energy of the spin glass with tropical algebra@Liu2021. To achieve this, we need to import the `TropicalMinPlus` data type from the `TropicalNumbers` package. This algebra is defined as
$
  &a plus.circle b = min(a, b),\
  &a times.circle b = a + b,\
  &bb(0) = infinity,\
  &bb(1) = 0,
$
where $plus.circle$ and $times.circle$ are the tropical addition and multiplication, respectively, and $bb(0)$ and $bb(1)$ are the zero and unit elements, respectively.
The following code computes the ground state energy of the spin glass.

== Live Coding: Tropical tensor network for spin-glass problem
By comparing with the partition function, we can see that the above code effectively computes the following equation:
$
  min_(bold(s)) sum_(i in V) h_i s_i + sum_((i, j) in E) J_(i j) s_i s_j,
$

==
#bibliography("refs.bib")
