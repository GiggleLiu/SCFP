#import "../book.typ": book-page
#import "@preview/physica:0.9.1": *
#import "@preview/jlyfish:0.1.0": *
#import "@preview/cetz:0.2.2": canvas, draw, tree

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


#read-julia-output(json("quantum-simulation-jlyfish.json"))
#let jlc(code, output: true) = [
  #jl-raw(fn: it => if output {
    grid(gutter: 2pt,
      raw(block: true, it.code, lang: "julia"),
      raw(block: true, str(it.result.data), lang: "output")
      )} else {
      raw(block: true, it.code, lang: "julia")
    }
    , code)
]

#show: book-page.with(title: "Quantum simulation")
= Quantum simulation

(Work in progress)

This chapter introduces the basic concepts of quantum simulation, including quantum circuit and tensor network based simulation. The code will be based on the quantum simulator #link("https://github.com/QuantumBFS/Yao.jl")[Yao.jl].

= Exact simulation

We first introduce the basic idea of quantum simulation, and then introduce the package `Yao.jl`.

== Quantum bit and Pauli matrices
A quantum state is represented by a vector in a Hilbert space. e.g. the $+1$ eigenstate of the Pauli-Y operator is represented as a vector of length 2.
#jlc(```julia
y_plus = ComplexF64[1/sqrt(2), im/sqrt(2)]
```
)

We represent a quantum operator as a matrix
$
  sigma_x = mat(0, 1; 1, 0) quad sigma_y = mat(0, -i; i, 0) quad sigma_z = mat(1, 0; 0, -1)
$
#jlc(```julia
id = ComplexF64[1 0; 0 1]
pauli_x = ComplexF64[0 1; 1 0]
pauli_y = ComplexF64[0 -im; im 0]
pauli_z = ComplexF64[1 0; 0 -1]

expected = y_plus' * pauli_y * y_plus
```
)

== Many-body simulation
The exact simulation of a quantum many-body system is based on the Kronecker product of operators. The Hamiltonian of a Heisenberg open chain can be constructed with:
#jlc(```julia
using LinearAlgebra
heisenberg_matrix(n) = sum([pauli_x, pauli_y, pauli_z]) do O
  sum(i->kron(fill(id, i-1)..., O, O, fill(id, n-i-1)...), 1:n-1)
end

h5 = heisenberg_matrix(5)

size(h5)
```)

The similar trick can be applied to quantum states. However, repersenting a quantum operator as dense matrix is not efficient. They are very sparse:

#jlc(```julia
sparsity = count(!iszero, h5)/length(h5)
```)

A simple improvement is to use sparse matrix to store the operator.
#jlc(```julia
using SparseArrays
heisenberg_matrix_sparse(n) = sum([pauli_x, pauli_y, pauli_z]) do O
  sum(i->kron(fill(sparse(id), i-1)..., sparse(O), sparse(O), fill(sparse(id), n-i-1)...), 1:n-1)
end

h5_sparse = heisenberg_matrix_sparse(5)

typeof(h5_sparse), nnz(h5_sparse)/length(h5_sparse)
```)

== Eigenstates and time evolution

The Krylov space method is widely used in quantum simulation. In the following we introduce the package `KrylovKit.jl` to find the eigenstates and time evolution of a quantum system.

== Symmetries

Quantum many-body scar.

= Circuit based simulation

== Noisy quantum systems

Some standard noisy models include

= Tensor network based simulation

SWAP test.

A quantum circuit is a sequence of quantum gates that act on the quantum state.

To store a quantum state in a $n$ qubit system, we need a complex vector of length $2^n$.


= Quantum State and Matrix product state (MPS)

== Definition

Tensor network is an efficient data structure for reducing the storage cost of a high dimensional data@Cichocki2014, which is widely used in quantum many-body physics. The matrix product state (MPS) is a special form of tensor network, which is used to represent a quantum state in one-dimensional lattice systems. The MPS is defined as
$
|psi angle.r = sum_(i_1, i_2, ..., i_n) tr(A_1^(i_1) A_2^(i_2) ... A_n^(i_n)) |i_1 i_2 ... i_n angle.r
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