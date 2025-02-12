#import "@preview/cetz:0.2.2": canvas, draw, tree
#import "@preview/ctheorems:1.1.3": *
#import "@preview/ouset:0.2.0": ouset
#import "../book.typ": book-page

#set math.equation(numbering: "(1)")

#show: book-page.with(title: "Tensor Networks")
#show: thmrules

#let definition = thmbox("definition", "Definition", inset: (x: 1.2em, top: 1em, bottom: 1em), base: none, stroke: black)
#let theorem = thmbox("theorem", "Theorem", base: none, stroke: black)
#let proof = thmproof("proof", "Proof")

#let tensor(location, name, label) = {
  import draw: *
  circle(location, radius: 10pt, name: name)
  content((), text(black, label))
}

#let labeledge(from, to, label) = {
  import draw: *
  line(from, to, name:"line")
  content("line.mid", label, align: center, fill:white, frame:"rect", padding:0.12, stroke: none)
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

= Tensor Networks
Let us define a complex matrix $A in CC^(m times n)$, and let its singular value decomposition be
$
A = U S V^dagger
$
where $U$ and $V$ are unitary matrices and $S$ is a diagonal matrix with non-negative real numbers on the diagonal.

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

  tensor((-1.5, 0), "A", [$U_1$])
  tensor((1.5, 0), "B", [$U_2$])
  tensor((0, -1.5), "C", [$U_3$])
  tensor((0, 1.5), "D", [$U_4$])
  tensor((1, 1), "L", [$Lambda$])
  labeledge("D", (rel: (0, 1.2)), [$i$])
  labeledge("A", (rel: (-1.2, 0)), [$j$])
  labeledge("C", (rel: (0, -1.2)), [$k$])
  labeledge("B", (rel: (1.2, 0)), [$l$])
  content((0, 0), box(inset: 3pt)[$c$], name: "c")
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


== Tensor training decomposition

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

= Tensor networks

Tensor network is a diagrammatic representation of tensor _contractions_. Tensor contraction is a generalization of the multiplication of matrices, which is defined as the summation of the product of the corresponding elements of the two tensors.
In this representation, a tensor is represented as a node, and an index is represented as a hyperedge (a hyperedge connects a single variable to any number of nodes). For example, vectors, matrices and higher order tensors can be represented as

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
  content((rel: (0, -1), to: "A"), [Tensor $A_(i j k)$])
})))

In the same diagram, the tensors associated with the same variable are connected by the same hyperedge. If a variable appears in the output tensor, the hyperedge is left _open_. For example, the diagrammatic representation of the matrix multiplication is given as follows:

#align(center, text(10pt, canvas({
  import draw: *
  tensor((-2, 1), "A", [$A$])
  tensor((0, 1), "B", [$B$])
  labeledge("A", (rel: (-1.5, 0)), [$i$])
  labeledge("A", (rel: (1.5, 0)), [$j$])
  labeledge("B", (rel: (1.5, 0)), [$k$])
  content((-1, -0.5), [$ C_(i k) := sum_j A_(i j) B_(j k) $])
})))

#definition([_(Tensor Network)_ A tensor network@Liu2022@Roa2024 is a mathematical framework for defining multilinear maps, which can be represented by a triple $cal(N) = (Lambda, cal(T), V_0)$, where:
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
(A B)_(i k) = "contract"({i,j,k}, {A_(i j), B_(j k)}, {i, k}),
$
where the input matrices $A$ and $B$ are indexed by the variable sets ${i, j}, {j, k}$, respectively, which are subsets of $Lambda = {i, j, k}$. As a remark of notation, when an set is used as subscripts, we omit the comma and the braces. The output tensor is indexed by variables ${i, k}$ and the summation runs over variables $Lambda without {i, k} = {j}$. The contraction corresponds to
$
(A B)_(i k) = sum_j A_(i j) B_(j k),
$
which is consistent with the matrix multiplication.

== Tensor network differentiation
The differentiation rules for tensor network contraction can be represented as the contraction of the tensor network:
#theorem([_(Tensor network differentiation)_:
    Let $(Lambda, cal(T), emptyset)$ be a tensor network with scalar output. The gradient of the tensor network contraction with respect to $T_V in cal(T)$ is
    $
      frac(partial "contract"(Lambda, cal(T), emptyset), partial T_V) =
      "contract"(Lambda, cal(T) \\ {T_V}, V).
    $
    That is, the gradient corresponds to the contraction of the tensor network
    with the tensor $T_V$ removed and the output label set to $V$.
])

#proof([
Let $cal(L)$ be a loss function of interest, where its differential form is given by:
$
delta cal(L) = "contract"(V_a, {delta A_(V_a), overline(A)_(V_a)}, emptyset) + "contract"(V_b, {delta B_(V_b), overline(B)_(V_b)}, emptyset)
$ <eq:diffeq>

The goal is to find $overline(A)_(V_a)$ and $overline(B)_(V_b)$ given $overline(C)_(V_c)$.
This can be achieved by using the differential form of tensor contraction, which states that
$
delta C = "contract"(Lambda, {delta A_(V_a), B_(V_b)}, V_c) + "contract"(Lambda, {A_(V_a), delta B_(V_b)}, V_c).
$
By inserting this result into @eq:diffeq, we obtain:
$
delta cal(L) = &"contract"(V_a, {delta A_(V_a), overline(A)_(V_a)}, emptyset) + "contract"(V_b, {delta B_(V_b), overline(B)_(V_b)}, emptyset)\
= &"contract"(Lambda, {delta A_(V_a), B_(V_b), overline(C)_(V_c)}, emptyset) + "contract"(Lambda, {A_(V_a), delta B_(V_b), overline(C)_(V_c)}, emptyset).
$
Since $delta A_(V_a)$ and $delta B_(V_b)$ are arbitrary, the above equation immediately implies:

$
overline(A)_(V_a) = "contract"(Lambda, {overline(C)_(V_c), B_(V_b)}, V_a)\
overline(B)_(V_b) = "contract"(Lambda, {A_(V_a), overline(C)_(V_c)}, V_b)
$
])


== Hidden Markov model

(introduce tropical tensor formalism)

== Spin-glass and tensor networks

For example, the contraction of two tensors $A_(i j k)$ and $B_(k l)$, i.e. $sum_k A_(i j k) B_(k l)$, can be diagrammatically represented as

#align(center, canvas({
  import draw: *
  tensor((1, 1), "A", "A")
  tensor((3, 1), "B", "B")
  labeledge("A", "B", "k")
  labeledge("B", (rel: (1.5, 0)), "l")
  labeledge("A", (rel: (0, 1.5)), "j")
  labeledge("A", (rel: (-1.5, 0)), "i")
}))

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


== Born machine

The partition function $Z$ is defined as the following tensor network
#align(center, canvas({
  import draw: *
  let n = 5
  for i in range(n){
    tensor((i*1.5, 0), "A'_" + str(i), [$A'_#i$])
    tensor((i*1.5, -1.5), "A_" + str(i), [$A_#i$])
  }
  for i in range(n){
    line("A'_" + str(i), "A_" + str(i), name: "line_" + str(i))
    content((rel: (-0.3, 0), to: "line_" + str(i) + ".mid"), [$x_#(i+1)$], align: center, fill:white, frame:"rect", padding:0.1, stroke: none)
  }
  for i in range(n - 1){
    line("A_" + str(i), "A_" + str(i+1))
    line("A'_" + str(i), "A'_" + str(i+1))
  }
}))

We use the log-likelihood as the loss function
$
cal(L)=-1/m sum_(bold(x) in "data") ln P(bold(x))
$
where $P(bold(x))$ is the probability of the data $bold(x)$ given the model. It is defined as $p(x)\/Z$, where $p(x) = psi^*(x) psi(x)$ is the unnormalized probability of the data $bold(x)$ and $Z$ is the partition function.

The gradient of the loss function is
$
nabla cal(L)=-2/m sum_(bold(x) in "data") (nabla psi(bold(x)))/psi(bold(x)) + (nabla Z)/Z
$

#pagebreak()

#bibliography("refs.bib")
