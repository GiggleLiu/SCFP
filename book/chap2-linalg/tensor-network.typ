#import "@preview/cetz:0.2.2": canvas, draw, tree
#import "../book.typ": book-page

#show: book-page.with(title: "Tensor Networks")

#import "@preview/ouset:0.2.0": ouset

#let tensor(location, name, label) = {
  import draw: *
  circle(location, radius: 13pt, name: name)
  content((), text(black, label))
}

#let deltatensor(location, name) = {
  import draw: *
  circle(location, radius: 3pt, name: name, fill:black)
}

#let labeledge(from, to, label) = {
  import draw: *
  line(from, to, name:"line")
  content("line.mid", label, align: center, fill:white, frame:"rect", padding:0.12, stroke: none)
}

#let labelnode(loc, label) = {
  import draw: *
  content(loc, [$#label$], align: center, fill:white, frame:"rect", padding:0.12, stroke: none, name: label)
}

#let infobox(title, body, stroke: blue) = {
  set text(black)
  set align(left)
  rect(
    stroke: stroke,
    inset: 8pt,
    radius: 4pt,
    width: 100%,
    [*#title:*\ #body],
  )
}

= Low-rank tensor decomposition
Let us define a complex matrix $A in CC^(m times n)$, and let its singular value decomposition be
$
A = U S V^dagger
$
where $U$ and $V$ are unitary matrices and $S$ is a diagonal matrix with non-negative real numbers on the diagonal.

== Generalizing to higher dimensions

== CP-decomposition

== Tucker decomposition

== Tensor training decomposition

== Example: 

A rank 3 tensor $A_(i j k)$ can be represented as

#pad(canvas({
  import draw: *
  tensor((1, 1), "A", "A")
  labeledge("A", (rel: (1.5, 0)), "k")
  labeledge("A", (rel: (0, 1.5)), "j")
  labeledge("A", (rel: (-1.5, 0)), "i")
  set-origin((3.5, 0))
  content((0, 1.5), $arrow$)
  set-origin((1, 0))
  content((0, 1.5), `ijk`)
}), x:30pt)

The contraction of two tensors $A_(i j k)$ and $B_(k l)$, i.e. $sum_k A_(i j k) B_(k l)$, can be diagrammatically represented as

#pad(canvas({
  import draw: *
  tensor((1, 1), "A", "A")
  tensor((3, 1), "B", "B")
  labeledge("A", "B", "k")
  labeledge("B", (rel: (1.5, 0)), "l")
  labeledge("A", (rel: (0, 1.5)), "j")
  labeledge("A", (rel: (-1.5, 0)), "i")
  set-origin((5.5, 0))
  content((0, 1.5), $arrow$)
  set-origin((2, 0))
  content((0, 1.5), `ijk,kl->ijl`)
}), x:30pt)

The kronecker product of two matrices $A_(i j)$ and $B_(k l)$, i.e. $A_(i j) times.circle B_(k l)$, can be diagrammatically represented as

#pad(canvas({
  import draw: *
  tensor((1, 1), "A", "A")
  tensor((3, 1), "B", "B")
  labeledge("A", (rel: (0, -1.5)), "j")
  labeledge("A", (rel: (0, 1.5)), "i")
  labeledge("B", (rel: (0, -1.5)), "l")
  labeledge("B", (rel: (0, 1.5)), "k")
  set-origin((5.5, 0))
  content((0, 1), $arrow$)
  set-origin((2, 0))
  content((0, 1), `ij,kl->ijkl`)
}), x:25pt)

The operation $tr(A B C)$ can be diagrammatically represented as

#pad(canvas({
  import draw: *
  tensor((1, 1), "A", "A")
  tensor((3, 1), "B", "B")
  tensor((5, 1), "C", "C")
  labeledge("A", "B", "j")
  labeledge("B", "C", "k")
  bezier("A.north", "C.north", (1, 3), (5, 3), name:"line")
  content("line.mid", "i", align: center, fill:white, frame:"rect", padding:0.1, stroke: none)
  set-origin((7.5, 0))
  content((0, 1.5), $arrow$)
  set-origin((3, 0))
  content((0, 1.5), `ij,jk,ki->`)
}), x:25pt)

From the diagram, we can see the trace permutation rule: $tr(A B C) = tr(C A B) = tr(B C A)$ directly.

#infobox([Definition (Tensor Network)], [A tensor network@Liu2022@Roa2024 is a mathematical framework for defining multilinear maps, which can be represented by a triple $cal(N) = (Lambda, cal(T), V_0)$, where:
- $Lambda$ is the set of variables present in the network $cal(N)$.
- $cal(T) = { T_(V_k) }_(k=1)^K$ is the set of input tensors, where each tensor $T_(V_k)$ is associated with the labels $V_k$.
- $V_0$ specifies the labels of the output tensor.
])
Specifically, each tensor $T_(V_k) in cal(T)$ is labeled by a set of variables $V_k subset.eq Lambda$, where the cardinality $|V_k|$ equals the rank of $T_(V_k)$. The multilinear map, or the *contraction*, applied to this triple is defined as
$
T_(V_0) = "contract"(Lambda, cal(T), V_0) ouset(=, "def") sum_(m in cal(D)_(Lambda without V_0)) product_(T_V in cal(T)) T_(V|M=m),
$
where $M = Lambda without V_0$. $T_(V|M=m)$ denotes a slicing of the tensor $T_V$ with the variables $M$ fixed to the values $m$. The summation runs over all possible configurations of the variables in $M$.

For instance, matrix multiplication can be described as the contraction of a tensor network given by
$
(A B)_({i, k}) = "contract"({i,j,k}, {A_({i, j}), B_({j, k})}, {i, k}),
$
where matrices $A$ and $B$ are input tensors containing the variable sets ${i, j}, {j, k}$, respectively, which are subsets of $Lambda = {i, j, k}$. The output tensor is comprised of variables ${i, k}$ and the summation runs over variables $Lambda without {i, k} = {j}$. The contraction corresponds to
$
(A B)_({i, k}) = sum_j A_({i,j})B_({j, k}).
$

Diagrammatically, a tensor network can be represented as an *open hypergraph*, where each tensor is mapped to a vertex and each variable is mapped to a hyperedge. Two vertices are connected by the same hyperedge if and only if they share a common variable. The diagrammatic representation of the matrix multiplication is given as follows: 

Here, we use different colors to denote different hyperedges. Hyperedges for $i$ and $k$ are left open to denote variables of the output tensor. A slightly more complex example of this is the star contraction:
$
"contract"({i,j,k,l}, {A_({i, l}), B_({j, l}), C_({k, l})}, {i,j,k}) \
= sum_l A_({i,l}) B_({j,l}) C_({k,l}).
$
Note that the variable $l$ is shared by all three tensors, making regular edges, which by definition connect two nodes, insufficient for its representation. This motivates the need for hyperedges, which can connect a single variable to any number of nodes.


= Quantum state

Single qubit states can be represented as a column vector, e.g.

$
|0 angle.r = mat( 1; 0 )\
|1 angle.r = mat( 0; 1 )
$

A multi-qubit state can be represented as a superposition of tensor product of single qubit states, e.g.
$
| psi angle.r &= sum_(i_1, i_2, ..., i_n) c_(i_1 i_2 ... i_n) |i_1 angle.r times.circle |i_2 angle.r times.circle ... times.circle |i_n angle.r \
&= sum_(i_1, i_2, ..., i_n) c_(i_1 i_2 ... i_n) |i_1 i_2 ... i_n angle.r
$


== Schmidt decomposition of a bi-partite state

Schmidt decomposition of a bi-partite state $| psi angle.r_(A B)$ is a decomposition of the state into a sum of product states, which is unique up to a global phase.
$
| psi_(A B) angle.r = sum_i lambda_i |i angle.r_A times.circle |i angle.r_B
$
where $lambda_i$ are non-negative real numbers and $sum_i lambda_i^2 = 1$.
It is easy to verify that the Schmidt coefficients $lambda_i$ correspond to the non-zero elements on the diagonal of $S$.

The Schmidt decomposition can be obtained by the singular value decomposition of the state matrix $A$.
Given a uniform state $| psi angle.r_(A B) = frac(1, 2) (|0 0 angle.r + |01 angle.r + |10 angle.r + |11 angle.r)$, the state matrix $A$ is

//In the $| psi angle.r_("uniform")$ example, the state matrix $A$ is
$
A = frac(1, 2) mat(1, 1; 1, 1)
$
Its singular value decomposition is
$
A = frac(1, sqrt(2)) mat(1, 1; 1, -1) mat(1, 0; 0, 0) frac(1, sqrt(2)) mat(1, 1; 1, -1)^dagger
$
Only one singular value is non-zero, so the state is not entangled.

== Quantum state and tensor network

Task 1: Representing a quantum state with tensor networks
1. How to represent the product state $|0 angle.r times.circle |0 angle.r$ with tensor network diagram?
2. How to represent the GHZ state $ frac(|0 0 1 angle.r + |1 1 1 angle.r, sqrt(2))$ with tensor network diagram?

= Quantum State and Matrix product state (MPS)

== Definition

Tensor network is an efficient data structure for reducing the storage cost of a high dimensional data@Cichocki2014, which is widely used in quantum many-body physics. The matrix product state (MPS) is a special form of tensor network, which is used to represent a quantum state in one-dimensional lattice systems. The MPS is defined as
$
| psi angle.r = sum_(i_1, i_2, ..., i_n) tr(A_1^(i_1) A_2^(i_2) ... A_n^(i_n)) |i_1 i_2 ... i_n angle.r
$
where $A_k^(i_k)$ is a rank-3 tensor, and $i_k$ is the physical index. The bond dimension $chi$ is the number of non-zero elements in the diagonal of the singular value matrix $S$.

#figure(pad(canvas({
  import draw: *
  for (i, name) in ((1, "A1"), (2, "A2"), (3, "A3"), (4, "A4")) {
    tensor((2*i, 1), name, [$A_#i$])
    line(name, (2*i, 2))
    content((2*i, 2.3), $i_#i$)
  }
  for (i, j) in (("A1", "A2"), ("A2", "A3"), ("A3", "A4")) {
      line(i, j, stroke: 3pt)
  }
  content((2.3, 1.8), $d$)
  content((3, 1.4), $chi$)
}), x:20pt),
caption: [A matrix product state (MPS) with virtual bond dimension $chi$ (thick lines) and physical bond dimension $d$.]
)

Task 2: Data compression with matrix product states

What is the data compression ratio of a $n$-site matrix product state with bond dimension $chi$ and local dimension $d$?

== Operations
=== Convert a quantum state to a tensor network

A quantum state can be represented as a hypercubic tensor. For example, a 4 qubits state can be diagrammatically represented as
#pad(canvas({
  import draw: *
  tensor((0, 0), "A", [$A$])
  line("A", (rel: (1.2, 0)))
  line("A", (rel: (-1.2, 0)))
  line("A", (rel: (0, 1.2)))
  line("A", (rel: (0, -1.2)))
  set-origin((2, 0))
  content((0, 0), "=")
  set-origin((2, 0))

  tensor((0, 0), "B", [$A_1$])
  tensor((2, 0), "C", [$A_(2-4)$])
  line("B", (rel: (1.2, 0)))
  line("B", "C", stroke: 3pt)
  line("C", (rel: (1.2, 0)))
  line("C", (rel: (0, -1.2)))
  line("B", (rel: (0, 1.2)))
  line("C", (rel: (0, 1.2)))
 
  set-origin((-2, -3))
  content((0, 0), "=")
  set-origin((0, -1))
  for (i, name) in ((1, "A1"), (2, "A2"), (3, "A3"), (4, "A4")) {
    tensor((2*i, 1), name, [$A_#i$])
    line(name, (2*i, 2))
  }
  for (i, j) in (("A1", "A2"), ("A2", "A3"), ("A3", "A4")) {
      line(i, j, stroke: 3pt)
  }
}), x:20pt)

=== Convert a MPS to a canonical form

A matrix is isometric if its conjugate transpose is its inverse, i.e. $U^dagger U = I$. For tensors, isometry can be defined in different ways. A tensor $A$ in the MPS is left canonical if the following equation holds

#pad(canvas({
  import draw: *
  tensor((0, -1), "A", [$L$])
  tensor((0, 1), "B", [$L^*$])
  line("A", "B")
  line("A", (rel: (1.2, 0)))
  line("B", (rel: (1.2, 0)))
  bezier("A.west", "B.west", (-1.5, -1), (-1.5, 1), name:"line")
  set-origin((1.2, 0))
  content((), "=")
  set-origin((3, 0))
  bezier((0, -1), (0, 1), (-1, -1), (-1, 1), name:"line")
}), x:20pt)

right canonical if

#pad(canvas({
  import draw: *
  tensor((0, -1), "A", [$R$])
  tensor((0, 1), "B", [$R^*$])
  line("A", "B")
  line("A", (rel: (-1.2, 0)))
  line("B", (rel: (-1.2, 0)))
  bezier("A.east", "B.east", (1.5, -1), (1.5, 1), name:"line")
  set-origin((1.2, 0))
  content((), "=")
  set-origin((2.5, 0))
  bezier((0, -1), (0, 1), (1, -1), (1, 1), name:"line")
}), x:20pt)

An MPS can be converted to a canonical form by applying the singular value decomposition to each tensor. The left canonical form is obtained by applying the SVD to each tensor from the left to the right, and the right canonical form is obtained by applying the SVD from the right to the left.

=== Inner product and optimal contraction order

#pad(canvas({
  import draw: *
  for (i, name) in ((1, "A1"), (2, "A2"), (3, "A3"), (4, "A4")) {
    tensor((2*i, 3), name+"*", [$A_#i^*$])
    tensor((2*i, 1), name, [$A_#i$])
    line(name, name+"*")
  }
  for (i, j) in (("A1", "A2"), ("A2", "A3"), ("A3", "A4")) {
      line(i+"*", j+"*", stroke: 3pt)
      line(i, j, stroke: 3pt)
  }
}), x:20pt)

The contraction order can be represented as a contraction tree, where the leaves are the tensors and the internal nodes are the contractions. The goal of contraction order optimization is to minimize the computational cost, including the time complexity, the memory usage and the read/write operations. Multiple algorithms@Kalachev2021@Gray2021 to find optimal contraction orders could be found in #link("https://arrogantgao.github.io/blogs/contractionorder/")[this blog post].

One of the optimal (in space) contraciton order for the inner product of the two states is

#pad(canvas({
  import draw: *
  tree.tree(
    ([], ([], ([], ([], [$A_1$], [$A_1^*$]), [$A_2$]), [$A_2^*$]), ([], ([], ([], [$A_4$], [$A_4^*$]), [$A_3$]), [$A_3^*$])),
    draw-node: (node, ..) => {
      tensor((), "", [#node.content])
    },
    grow: 1.5,
    spread: 1.5
  )
}), x:20pt)

Optimal contraction order of a glued tree tensor network

1. What is the time complexity, memory usage and read/write operations of the above contraction order?
2. Given a tree tensor network, what is the optimal contraction order to compute the inner product of the two states? The inner product of a tree tensor network is diagrammatically represented as
#pad(canvas({
import draw: *
tensor((3, 5), "B1*", [$B_1^*$])
  tensor((3, -1), "B1", [$B_1$])
  tensor((7, 5), "B2*", [$B_2^*$])
  tensor((7, -1), "B2", [$B_2$])
  tensor((5, 7), "C1*", [$C_1^*$])
  tensor((5, -3), "C1", [$C_1$])
  for (i, name) in ((1, "A1"), (2, "A2"), (3, "A3"), (4, "A4")) {
    tensor((2*i, 3), name+"*", [$A_#i^*$])
    tensor((2*i, 1), name, [$A_#i$])
    line(name, name+"*")
  }
  line("A1*", "B1*", stroke: 3pt)
  line("A1", "B1", stroke: 3pt)
  line("A2*", "B1*", stroke: 3pt)
  line("A2", "B1", stroke: 3pt)
  line("A4*", "B2*", stroke: 3pt)
  line("A4", "B2", stroke: 3pt)
  line("A3*", "B2*", stroke: 3pt)
  line("A3", "B2", stroke: 3pt)
  line("B1*", "C1*", stroke: 3pt)
  line("B1", "C1", stroke: 3pt)
  line("B2*", "C1*", stroke: 3pt)
  line("B2", "C1", stroke: 3pt)
}), x:20pt)

Tensor transformation

How to make a transform between the following two tensor networks? i.e. given $A$, $B$ and $C$ on the left, find $X$, $Y$ and $Z$ on the right, and vise versa. (credit: Huan-Hai Zhou)
#pad(canvas({
import draw: *
tensor((0, 0), "A", [$A$])
  tensor((-1, -calc.sqrt(3)), "B", [$B$])
  tensor((1, -calc.sqrt(3)), "C", [$C$])
  line("A", "B")
  line("A", "C")
  line("B", "C")
  line("A", (rel: (0, 1.3)))
  line("B", (rel: (-1.3*calc.cos(calc.pi/6), -1.3*calc.sin(calc.pi/6))))
  line("C", (rel: (1.3*calc.cos(calc.pi/6), -1.3*calc.sin(calc.pi/6))))

  set-origin((1.5, -0.5))
  content((), sym.arrow.l.r.double)
  set-origin((4.5, 0.5))

  tensor((0, 0), "A", [$X$])
  tensor((-1, -calc.sqrt(3)), "B", [$Y$])
  tensor((1, -calc.sqrt(3)), "C", [$Z$])
  let center = (0, -2/calc.sqrt(3))
  line("A", center)
  line("B", center)
  line("C", center)
  line("A", (rel: (0, 1.3)))
  line("B", (rel: (-1.3*calc.cos(calc.pi/6), -1.3*calc.sin(calc.pi/6))))
  line("C", (rel: (1.3*calc.cos(calc.pi/6), -1.3*calc.sin(calc.pi/6))))
  }), x:30pt)

#pagebreak()

#bibliography("refs.bib")
