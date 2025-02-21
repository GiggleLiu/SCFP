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


=== Exercise: Representing trace operation

Let $A, B$ and $C$ be three square matrices with the same size. Represent the trace operation $tr(A B C)$ with a tensor network diagram.

*Solution*
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

The corresponding einsum notation is `ij,jk,ki->`. From this diagram, we can see the trace permutation rule: $tr(A B C) = tr(C A B) = tr(B C A)$.

== Hidden Markov model

A Hidden Markov Model (HMM)@Bishop2006 is a simple probabilistic graphical model that describes a Markov process with unobserved (hidden) states. The model consists of:

- A sequence of hidden states $z_t$ following a Markov chain with transition probability $P(z_(t+1)|z_t)$
- A sequence of observations $x_t$ that depend only on the current hidden state through emission probability $P(x_t|z_t)$

The joint probability of a sequence of $T+1$ hidden states and $T$ observations can be written as:

$
P(bold(z), bold(x)) = P(z_0) product_(t=1)^T P(z_(t)|z_(t-1))P(x_t|z_t).
$
Note that the conditional probability $P(z_(t)|z_(t-1))$ can be represented as a tensor with two indices. The joint probability $P(bold(z), bold(x))$ can be represented as a tensor network diagram:

#figure(canvas({
  import draw: *
  let s(it) = text(11pt, it)
  
  // Draw transition matrices
  let dx = 2.0
  for i in range(5){
    tensor((dx*i, 0), "A" + str(i), []) 
  }
  for i in range(4){
    line("A" + str(i), "A" + str(i+1), name: "line" + str(i))
    content("line" + str(i) + ".mid", box(inset: 3pt, fill: white, s[$z_#(i+1)$]), name: "z" + str(i))
    tensor((rel: (0, -1)), "B" + str(i), [])
    line("z" + str(i), "B" + str(i))
    line("B" + str(i), (rel: (0, -0.8)))
    content((rel: (0, -0.2)), s[$x_#(i+1)$])
  }
  line("A0", (rel: (-1, 0)))
  content((rel: (-0.3, 0)), s[$z_0$])
}),
caption: [The tensor network representation of a Hidden Markov Model (HMM) with observed variables $x_1, x_2, dots, x_T$ and hidden states $z_0, z_1, dots, z_T$. The circles are conditional probabilities $P(z_t|z_(t-1))$ and $P(x_t|z_t)$.]
)

This is the decoding problem of HMM: Given a sequence of observations $bold(x) = (x_1, x_2, ..., x_T)$, how to find the most likely sequence of hidden states $bold(z)$? The equivalent mathematical formulation is:
$
  arg max_(bold(z)) P(z_0) product_(t=1)^T P(z_(t)|z_(t-1))P(overshell(x)_t|z_t),
$ <eq:decoding>
where $overshell(x)_t$ denotes an observed variable $x_t$ with a fixed value. It is equivalent to contracting the following tensor network:
$
  cases(Lambda = {z_0, z_1, dots, z_T},
  cal(T) = {P(z_0), P(z_1|z_0), dots, P(z_T|z_(T-1)), P(overshell(x)_1|z_1), P(overshell(x)_2|z_2), dots, P(overshell(x)_T|z_T)},
  V_0 = emptyset
  )
$ <eq:decoding-tensor>
Since $overshell(x)_1, overshell(x)_2, dots, overshell(x)_T$ are fixed and not involved in the contraction, $P(overshell(x)_t|z_t)$ is a vector indexed by $z_t$ rather than a matrix.
To solve @eq:decoding, we first convert @eq:decoding-tensor into a tropical tensor network $(Lambda, {log(t) | t in cal(T)}, V_0)$, where $log(t)$ is obtained by taking the logarithm of each element in $t$. Then the contraction of this tropical tensor network is equivalent to
$
  arg max_(bold(z)) sum_(bold(z)) log P(z_0) + sum_(t=1)^T log P(z_t|z_(t-1)) + sum_(t=1)^T log P(overshell(x)_t|z_t),
$
which solves the decoding problem.
Since this tensor network has a chain structure, its contraction is computationally efficient.
This algorithm is equivalent to the Viterbi algorithm.

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

#bibliography("refs.bib")
