using OMEinsum, ProblemReductions, Graphs

# Define an AFM spin glass on Petersen graph
graph = smallgraph(:petersen)
sg = SpinGlass(graph, ones(Int, ne(graph)), zeros(Int, nv(graph)))

# Initialize tensors and contraction code
β = 1.0
tensors = [[exp(-β * J) exp(β * J); exp(β * J) exp(-β * J)] for J in sg.J]
rawcode = EinCode([[e.src, e.dst] for e in edges(graph)], Int[])

# optimize the contraction code
optcode = optimize_code(rawcode, uniformsize(rawcode, 2), TreeSA())

# Compute the partition function
Z = optcode(tensors...)  # output: 167555.17801582735

# Compute the ground state energy
using TropicalNumbers

tensors = [TropicalMinPlus.([J -J; -J J]) for J in sg.J]
Emin = optcode(tensors...)  # output: -9ₛ
