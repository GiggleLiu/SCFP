#import "../book.typ": book-page
#import "@preview/physica:0.9.1": *
#import "@preview/jlyfish:0.1.0": *
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

= Tensor network based simulation
A quantum circuit is a sequence of quantum gates that act on the quantum state.

To store a quantum state in a $n$ qubit system, we need a complex vector of length $2^n$.