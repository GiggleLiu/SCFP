using OMEinsum
using LinearAlgebra

struct MPS{T}
    tensors::Vector{Array{T, 3}}
end

# Function to compress a tensor using Tensor Train (TT) decomposition
function tensor_train_decomposition(tensor::AbstractArray, largest_rank::Int; atol=1e-6)
    dims = size(tensor)
    n = length(dims)
    
    # Initialize the cores of the TT decomposition
    tensors = Array{Float64, 3}[]
    
    # Reshape the tensor into a matrix
    rpre = 1
    current_tensor = reshape(tensor, dims[1], :)
    
    # Perform SVD for each core except the last one
    for i in 1:(n-1)
        # Truncate to the specified rank
        U_truncated, S_truncated, V_truncated, r = truncated_svd(current_tensor, largest_rank, atol)

        # Middle cores have shape (largest_rank, dims[i], r)
        push!(tensors, reshape(U_truncated, (rpre, dims[i], r)))
        
        # Prepare the tensor for the next iteration
        current_tensor = S_truncated * V_truncated'
        
        # Reshape for the next SVD
        current_tensor = reshape(current_tensor, r * dims[i+1], :)
        rpre = r
    end
    
    # Add the last core
    push!(tensors, reshape(current_tensor, (rpre, dims[n], 1)))
    
    return MPS(tensors)
end

function truncated_svd(current_tensor::AbstractArray, largest_rank::Int, atol)
    # Compute SVD
    U, S, V = svd(current_tensor)
    r = min(largest_rank, sum(S .> atol))
    S_truncated = Diagonal(S[1:r])
    U_truncated = U[:, 1:r]
    V_truncated = V[:, 1:r]
    return U_truncated, S_truncated, V_truncated, r
end

# Function to contract the TT cores to reconstruct the tensor
function contract(mps::MPS)
    n = length(mps.tensors)
    code = EinCode([[2i-1, 2i, 2i+1] for i in 1:n], Int[2i for i in 1:n])
    size_dict = OMEinsum.get_size_dict(code.ixs, mps.tensors)
    optcode = optimize_code(code, size_dict, GreedyMethod())
    return optcode(mps.tensors...)
end

# Example usage: compressing a uniform tensor of size 2^20
tensor = ones(Float64, fill(2, 20)...);

# Perform TT decomposition
mps = tensor_train_decomposition(tensor, 5)

# Reconstruct the tensor from TT cores
reconstructed_tensor = contract(mps);

# Calculate the relative error
relative_error = norm(tensor - reconstructed_tensor) / norm(tensor)
println("Relative error of reconstruction: ", relative_error)

# Calculate compression ratio
original_size = prod(size(tensor))
compressed_size = sum([prod(size(core)) for core in mps.tensors])
compression_ratio = original_size / compressed_size
println("Compression ratio: ", compression_ratio)

# Print the shapes of the TT cores
println("TT core shapes:")
for (i, core) in enumerate(mps.tensors)
    println("Core $i: $(size(core))")
end
