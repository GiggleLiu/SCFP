//#import "../book.typ": book-page, cross-link, heading-reference
#import "@preview/cetz:0.4.1": *
//#show: book-page.with(title: "GPU Programming with Julia")

#set math.equation(numbering: "(1)")
#let boxed(it, width: 100%) = block(stroke: 1pt, inset: 10pt, radius: 4pt, width: width)[#it]

#show ref: it => {
  let el = it.element
  if el != none and el.func() == heading {
    link(el.location(), el.body)
  } else {
    it
  }
}

#align(center, [= GPU Programming with Julia\
_Jin-Guo Liu_])

#box(fill: rgb("f0f8ff"), inset: 1em, radius: 5pt, width: 100%)[
  *Learning Objectives*
  
  By the end of this lecture, you will be able to:
  - Understand the fundamental differences between CPU and GPU computing
  - Create and manipulate GPU arrays in Julia using CUDA.jl
  - Write and launch custom CUDA kernels for parallel computation
  - Use high-performance GPU libraries (CUBLAS, CUSPARSE, CUFFT)
  - Debug and optimize GPU code for maximum performance
  - Choose appropriate strategies for different GPU computing tasks
]

*Why do we need GPU computing?*

Modern scientific computing and machine learning require processing massive amounts of data. A single CPU, no matter how powerful, can only do so much. GPUs offer thousands of cores working in parallel, providing 10-100× speedups for the right workloads. Whether you're training neural networks, running molecular dynamics simulations, or processing images, GPU computing can dramatically accelerate your work.

= Introduction: The Parallel Computing Revolution

== Understanding the CPU vs GPU Paradigm

Modern GPUs are designed for data-parallel computations, featuring thousands of simple cores optimized for throughput rather than latency @Nickolls2008. This architecture makes them ideal for scientific computing and machine learning workloads.

#box(fill: rgb("fff5ee"), inset: 1em, radius: 5pt, width: 100%)[
  *Intuition First!* 🤔
  
  Imagine you need to paint 10,000 fence posts:
  - *CPU approach*: Hire a few highly skilled painters. Each works very fast, but there are only 8-16 of them.
  - *GPU approach*: Hire 10,000 students with paintbrushes. Each works slower, but they all work simultaneously!
  
  The key insight: Different problems need different solutions!
]

#figure(
  table(
    columns: 4,
    align: center,
    [*Device*], [*Cores*], [*Clock Speed*], [*Best For*],
    [Modern CPU], [8-64], [~3-5 GHz], [Complex logic, branching],
    [Modern GPU], [1000s-10000s], [~1-2 GHz], [Simple, parallel operations],
  ),
  caption: [CPU vs GPU: Different tools for different jobs]
)

*Key Concepts:*
- *Latency* (CPU): Time to complete a single task (microseconds)
- *Throughput* (GPU): Total work completed per second (millions of operations)
- *Parallelism*: Doing many things at once

*🎯 Interactive Question:* If you need to multiply 1 million pairs of numbers, which is better? Think about it before reading on...

*Answer:* GPU! This is a perfectly parallel problem - each multiplication is independent. The GPU can do thousands simultaneously.

== When Should You Use GPU Computing?

#box(fill: rgb("f0fff0"), inset: 1em, radius: 5pt, width: 100%)[
  *The Big Idea* 💡
  
  GPUs excel at *data parallelism* - performing the same operation on many data elements simultaneously.
]

*✓ Excellent for GPUs:*
- Large array operations (element-wise operations)
- Matrix multiplication and linear algebra
- Image and signal processing
- Monte Carlo simulations (parallel random sampling)
- Training neural networks
- Fast Fourier Transforms (FFT)

*✗ Poor for GPUs:*
- Small datasets (< 1000 elements) - overhead dominates
- Complex branching logic (if-else heavy code)
- Sequential algorithms (each step depends on the previous)
- Heavy CPU ↔ GPU data transfers

*⚠️ The Golden Rule:* The speedup from parallel computation must outweigh the overhead of copying data to/from the GPU!

*🎯 Quick Quiz:* Which would benefit more from GPU acceleration?
- A) Computing `sin(x)` for 10 values
- B) Computing `sin(x)` for 10 million values
- C) A recursive Fibonacci function

*Answer:* B! A has too little data (overhead dominates), C is inherently sequential.

== Why Julia for GPU Computing?

Julia provides a unique combination of advantages @Besard2018:

1. *High-level syntax*: Write GPU code that looks like regular Julia
2. *Performance*: Achieve near-CUDA-C performance without low-level programming
3. *Native compilation*: Julia functions compile directly to GPU code
4. *Composability*: GPU arrays work seamlessly with Julia's ecosystem

*The magic:* Julia's compiler can take your high-level code and generate optimized GPU kernels automatically!

= Getting Started with CUDA.jl

CUDA.jl @Besard2019 is the primary Julia package for NVIDIA GPU programming, providing a high-level interface while maintaining performance comparable to native CUDA C code.

== Installation and Verification

#box(fill: rgb("fffacd"), inset: 1em, radius: 5pt, width: 100%)[
  *💻 Live Setup Exercise*
  
  Let's get your GPU working with Julia!
]

*Step 1: Verify your GPU*

First, check that you have an NVIDIA GPU with proper drivers:

```bash
nvidia-smi
```

This command shows:
- GPU model and memory
- Driver version
- Current GPU utilization
- Running processes

*What you should see:* A table showing your GPU information. If you get "command not found", your NVIDIA drivers aren't installed.

*Step 2: Install CUDA.jl*

```julia
using Pkg
Pkg.add("CUDA")
using CUDA

# Test if CUDA is functional
CUDA.functional()  # Should return true
```

*🎯 Troubleshooting:* If `CUDA.functional()` returns `false`, check:
1. Is your GPU NVIDIA? (AMD/Intel won't work with CUDA)
2. Is compute capability ≥ 3.5? Run `CUDA.versioninfo()` to check
3. Are drivers up to date?

*Step 3: Check your setup*

```julia
CUDA.versioninfo()
```

This displays:
- CUDA toolkit version
- Available GPU devices
- Driver compatibility
- Supported features

== Your First GPU Computation

Let's start with the simplest possible example:

```julia
using CUDA

# Create a vector on the CPU
cpu_vector = ones(10)

# Move it to the GPU
gpu_vector = CuArray(cpu_vector)

# Do computation on GPU
result = gpu_vector .+ 1  # Still on GPU!

# Bring result back to CPU
cpu_result = Array(result)
```

*What just happened?*
1. `CuArray()` *uploaded* data from CPU RAM to GPU memory
2. `.+` operation executed on the GPU (all 10 additions in parallel!)
3. `Array()` *downloaded* result back to CPU

*🎯 Think About It:* What if we need to do 100 operations? Should we transfer data after each one?

*Answer:* No! Keep data on GPU, do all operations there, then transfer once at the end.

= Array Programming: The High-Level Approach

== Creating GPU Arrays

#box(fill: rgb("f0f8ff"), inset: 1em, radius: 5pt, width: 100%)[
  *🎯 The High-Level Strategy*
  
  For most tasks, you don't need to write CUDA kernels! Julia's array operations and broadcasting automatically compile to efficient GPU code.
]

```julia
using CUDA

# Create arrays directly on GPU
a = CUDA.zeros(100)           # 100 zeros
b = CUDA.ones(10, 10)         # 10×10 matrix of ones
c = CUDA.randn(1000)          # 1000 random numbers
d = CUDA.fill(3.14, 50)       # 50 elements, all π

# Upload from CPU
cpu_data = randn(1000)
gpu_data = CuArray(cpu_data)

# Download to CPU
result = Array(gpu_data)
```

*Memory Tip:* Creating arrays directly on GPU (e.g., `CUDA.randn()`) is faster than creating on CPU and uploading!

== The Scalar Indexing Trap

#box(fill: rgb("ffe4e1"), inset: 1em, radius: 5pt, width: 100%)[
  *🚨 Critical Warning: Scalar Indexing*
  
  This is the #1 performance killer for beginners!
]

```julia
x = CUDA.randn(1000)

# ❌ This is DISABLED by default (throws error)
# value = x[1]

# ✓ Enable if you really need it (but avoid in performance code!)
CUDA.@allowscalar x[1]

# ✓ Better: Work with entire arrays
sum(x)        # ✓ Fast, runs on GPU
x .+ 1        # ✓ Fast, runs on GPU  
maximum(x)    # ✓ Fast, runs on GPU
x[1] = 5      # ❌ Slow, requires synchronization
```

*Why is scalar indexing so slow?*

Each scalar access requires:
1. CPU stops and waits
2. GPU finishes ALL pending operations (synchronization)
3. Transfer single value over PCIe bus (~16 GB/s vs ~1 TB/s GPU memory)
4. Resume GPU operations

This can be *1000× slower* than processing the entire array!

*🎯 Rule of Thumb:* If you're accessing individual array elements in a loop, you're doing it wrong!

== Broadcasting: Your Secret Weapon

Broadcasting is how you write fast GPU code without writing kernels:

```julia
x = CUDA.randn(10000)

# Simple operations - automatically parallel!
y = x .^ 2                    # Square every element
z = sin.(x) .+ cos.(x)        # Sine plus cosine
w = @. sqrt(abs(x)) + exp(-x^2)  # Complex expression

# Custom functions work too!
function my_function(x)
    if x > 0
        return sqrt(x)
    else
        return 0.0
    end
end

result = my_function.(x)  # Julia compiles this for GPU!
```

*The Magic:* Julia analyzes your function, compiles it to GPU assembly (PTX), and launches it on thousands of threads - automatically!

*Compilation Pipeline:*
```
Julia code → LLVM IR → PTX assembly → GPU execution
```

== 🧪 Hands-on Example: CPU vs GPU Performance

Let's measure the actual speedup:

```julia
using CUDA
using BenchmarkTools
using LinearAlgebra

n = 2000

# Prepare CPU data
a_cpu = randn(n, n)
b_cpu = randn(n, n)
c_cpu = zeros(n, n)

# Prepare GPU data
a_gpu = CuArray(a_cpu)
b_gpu = CuArray(b_gpu)
c_gpu = CUDA.zeros(n, n)

# CPU matrix multiplication
@time mul!(c_cpu, a_cpu, b_cpu)

# GPU matrix multiplication (with synchronization!)
@time CUDA.@sync mul!(c_gpu, a_gpu, b_gpu)

# Verify correctness
@assert c_cpu ≈ Array(c_gpu)
```

*Important:* Always use `CUDA.@sync` for timing! GPU operations are asynchronous - without sync, you'll measure kernel launch time, not execution time.

*🎯 Experiment:* Try different matrix sizes: 100×100, 1000×1000, 5000×5000. At what size does GPU become faster?

*Typical Results:*
- 100×100: CPU faster (overhead dominates)
- 1000×1000: GPU ~2-5× faster
- 5000×5000: GPU ~20-50× faster

== Example: Broadcasting Custom Functions

One of Julia's killer features - broadcast ANY function to GPU:

```julia
using CUDA, BenchmarkTools

# Define pure Julia function
factorial(n) = n == 1 ? 1 : factorial(n-1) * n

function poor_besselj(ν::Int, z::T; atol=eps(T)) where T
    k = 0
    s = (z/2)^ν / factorial(ν)
    out = s
    while abs(s) > atol
        k += 1
        s *= (-1) / k / (k+ν) * (z/2)^2
        out += s
    end
    out
end

# Create test data
x_cpu = randn(10000)
x_gpu = CuArray(x_cpu)

# CPU version
@benchmark poor_besselj.(1, $x_cpu)

# GPU version - SAME CODE!
@benchmark CUDA.@sync poor_besselj.(1, $x_gpu)

# Verify correctness
@assert poor_besselj.(1, x_cpu) ≈ Array(poor_besselj.(1, x_gpu))
```

*🎯 What's happening?*
1. Julia's compiler sees you're broadcasting over a CuArray
2. It compiles `poor_besselj` to GPU code
3. Launches it on thousands of GPU threads
4. Each thread computes one element

No manual CUDA kernel writing needed!

= Writing CUDA Kernels: Taking Full Control

#box(fill: rgb("fff5ee"), inset: 1em, radius: 5pt, width: 100%)[
  *When to Write Kernels* 🤔
  
  Broadcasting handles 80% of use cases. Write custom kernels when:
  - You need fine-grained control over threads and blocks
  - You want to use shared memory for communication
  - You need atomic operations for reductions
  - You're implementing novel parallel algorithms
]

== Understanding GPU Thread Hierarchy

GPUs organize computation in a three-level hierarchy @Nickolls2008:

```
Grid (entire computation)
  ├─ Block 1 (threads that can communicate)
  │   ├─ Thread 1
  │   ├─ Thread 2
  │   └─ ...
  ├─ Block 2
  │   ├─ Thread 1
  │   └─ ...
  └─ ...
```

*Key Concepts:*
- *Thread*: Single execution unit.
- *Warp*: A group of 32 threads executed simultaneously (SIMT).
- *Block*: Group of threads (up to 1024) that can share memory.
- *Grid*: All blocks in a kernel launch.

#box(fill: rgb("fff5ee"), inset: 1em, radius: 5pt, width: 100%)[
  *How GPUs Work: The SIMT Model* 🧠
  
  GPUs use *Single Instruction, Multiple Threads (SIMT)*. Threads are grouped into "warps" (typically 32 threads) that execute the *same instruction* at the same time.
  
  *The Trap:* If threads in a warp diverge (e.g., half go into an `if` block, half go into `else`), the hardware serializes the execution! Both paths are executed for all threads in the warp, with threads masked off when not active.
  
  *Performance Tip:* Avoid branching (if-else) that depends on thread ID or data within a warp!
]

*Why blocks?* Threads in a block can:
- Share fast memory (shared memory)
- Synchronize with each other
- Cooperate on sub-problems

== Your First CUDA Kernel

Let's write a kernel that prints thread IDs:

```julia
using CUDA

function print_kernel()
    # Compute unique thread ID
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    # Print from GPU (debugging only!)
    @cuprintf "Thread %ld in block %ld (global ID: %ld)\n" 
              Int(threadIdx().x) Int(blockIdx().x) Int(i)
    
    # MUST return nothing
    return
end

# Launch kernel: 2 blocks × 4 threads = 8 total threads
CUDA.@sync @cuda threads=4 blocks=2 print_kernel()
```

*Thread ID Formula (Julia 1-based indexing):*
```
global_id = (block_id - 1) × threads_per_block + local_thread_id
```
*Note:* Julia uses 1-based indexing, so we subtract 1 from `blockIdx`. In C/C++, you would use 0-based indexing: `blockIdx.x * blockDim.x + threadIdx.x`.

*🎯 Interactive Question:* With 3 blocks of 8 threads each, what is the global ID of thread 5 in block 2?

*Answer:* $(2-1) times 8 + 5 = 13$

== Example: Fill Array with Indices

A practical kernel that actually does something useful:

```julia
function one2n_kernel(A)
    # Compute global thread index
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    # Critical: Check bounds!
    @inbounds if i <= length(A)
        A[i] = i
    end
    
    return nothing  # Kernels must return nothing
end

# Create array and launch kernel
A = CUDA.zeros(Int, 2000)
@cuda blocks=2 threads=1024 one2n_kernel(A)

# Check result
println(Array(A))  # [1, 2, 3, ..., 2000]
```

*Key Points:*
1. *Always check bounds* - total threads may exceed array size
2. Use `@inbounds` for performance (but only when safe!)
3. Return `nothing` from kernels

== Choosing Block and Thread Counts

#box(fill: rgb("fffacd"), inset: 1em, radius: 5pt, width: 100%)[
  *🎯 Configuration Strategy*
  
  Choosing the right block/thread configuration affects performance!
]

*Rules of Thumb:*
- Threads per block: 128-1024 (typically 256 or 512)
- Must be multiple of 32 (warp size)
- Total threads ≥ array size (extra threads do nothing)

*Why multiples of 32?* GPUs execute threads in groups of 32 called "warps". All threads in a warp execute simultaneously.

```julia
function launch_config(n::Int; threads_per_block=256)
    threads = threads_per_block
    blocks = cld(n, threads)  # Ceiling division: ⌈n/threads⌉
    return (threads=threads, blocks=blocks)
end

# Example: Array of 10,000 elements
config = launch_config(10000)  # (threads=256, blocks=40)
@cuda threads=config.threads blocks=config.blocks my_kernel(data)
```

*🎯 Think About It:* Why not always use the maximum 1024 threads per block?

*Answer:* 
1. *Occupancy:* Each block uses registers and shared memory. If a block uses too many resources, fewer blocks can run simultaneously on a Streaming Multiprocessor (SM).
2. *Granularity:* Smaller blocks can sometimes schedule better across the GPU.
3. *Standard practice:* 256 or 512 threads is a good starting point.

== Kernel Restrictions: What You Can't Do

#box(fill: rgb("ffe4e1"), inset: 1em, radius: 5pt, width: 100%)[
  *🚫 GPU Kernel Limitations*
  
  GPU kernels are restricted environments - not all Julia features work!
]

*❌ Not Allowed:*
- Dynamic memory allocation (`push!`, `append!`, creating arrays)
- I/O operations (printing except `@cuprintf`)
- Most standard library functions
- Recursion (very limited support)
- Calling non-GPU-compatible functions

*✓ Allowed:*
- Basic arithmetic (`+`, `-`, `*`, `/`)
- Math functions (`sin`, `cos`, `exp`, `sqrt`, etc.)
- Control flow (`if`, `while`, `for`)
- Array indexing (fixed-size arrays)
- Thread/block indexing functions

== Debugging: Finding What Went Wrong

```julia
function buggy_kernel(A)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    @inbounds if i <= length(A)
        # ERROR: Dynamic allocation not allowed on GPU!
        A[i:i+1] = randn(2)
    end
    
    return nothing
end

# Debug with type analysis
@device_code_warntype @cuda blocks=2 threads=1024 buggy_kernel(A)
```

*Debugging Tools:*
- `@device_code_warntype`: Check for type instabilities and errors
- `@device_code_llvm`: Inspect LLVM intermediate representation
- `@device_code_ptx`: See actual GPU assembly code
- `@device_code_sass`: See machine code (most detailed)

*🎯 Debugging Strategy:*
1. Start with `@device_code_warntype` - catches most issues
2. Test kernel with small data first (easy to verify)
3. Use `@cuprintf` sparingly (very slow!)
4. Verify against CPU version

= High-Performance GPU Libraries

#box(fill: rgb("f0fff0"), inset: 1em, radius: 5pt, width: 100%)[
  *The Big Idea* 💡
  
  Don't reinvent the wheel! NVIDIA provides highly optimized libraries for common operations. These are often 10-100× faster than naive implementations.
]

== CUBLAS: Linear Algebra Powerhouse

CUBLAS provides GPU-accelerated BLAS (Basic Linear Algebra Subroutines) operations, achieving near-peak performance for dense linear algebra @NVIDIA2023:

```julia
using CUDA
using LinearAlgebra

n = 2000
A = CUDA.randn(n, n)
B = CUDA.randn(n, n)
C = CUDA.zeros(n, n)

# Matrix multiplication (uses CUBLAS automatically)
mul!(C, A, B)  # Calls optimized CUBLAS routine
C = A * B      # Same thing

# Vector operations
v = CUDA.randn(n)
norm(v)           # Vector norm
dot(v, v)         # Dot product  
A * v             # Matrix-vector multiplication

# Matrix operations
A \ v             # Solve linear system (uses CUSOLVER)
```

*Performance:* CUBLAS matrix multiplication achieves near-peak GPU performance (~90% of theoretical maximum).

== CUSOLVER: Matrix Decompositions

Solve linear systems and compute decompositions on GPU:

```julia
using CUDA
using LinearAlgebra

A = CUDA.randn(100, 100)
b = CUDA.randn(100)

# Solve Ax = b (uses CUSOLVER)
x = A \ b

# Verify solution
@assert A * x ≈ b

# Other decompositions
Q, R = qr(A)      # QR decomposition
U, S, V = svd(A)  # Singular Value Decomposition
```

*When to use:* Linear systems, least squares, eigenvalue problems, matrix factorizations.

== CUSPARSE: Sparse Matrix Operations

For matrices with mostly zeros (common in scientific computing):

```julia
using CUDA
using SparseArrays
using LinearAlgebra

# Create sparse matrix on CPU
A_cpu = sprand(1000, 1000, 0.01)  # 1% density (99% zeros)
println("Number of nonzeros: ", nnz(A_cpu))

# Upload to GPU
A_gpu = CUSPARSE.CuSparseMatrixCSC(A_cpu)

# Sparse operations
v = CUDA.randn(1000)
result = A_gpu * v  # Sparse matrix-vector multiplication

# Sparse-sparse multiplication
B_gpu = A_gpu * A_gpu

# Sparse linear systems  
A_sparse = A_gpu + CUDA.I(1000) * 10  # Add diagonal to sparse matrix
x = A_sparse \ v
```

*When to use sparse matrices:*
- Density < 5-10% (rule of thumb)
- Graph algorithms (adjacency matrices)
- Partial differential equations (finite element/difference methods)
- Optimization problems with sparse Jacobians/Hessians

*Trade-off:* Sparse operations save memory but have more complex code. Only worth it for large, sparse matrices.

== CUFFT: Fast Fourier Transform

GPU-accelerated FFT for signal processing and scientific computing:

```julia
using CUDA
using FFTW  # For CPU comparison

# 1D FFT
x = CUDA.randn(10000)
X = CUFFT.fft(x)
x_reconstructed = CUFFT.ifft(X)
@assert x ≈ x_reconstructed

# 2D FFT (images, PDEs)
img = CUDA.randn(1024, 1024)
img_freq = CUFFT.fft(img)

# Power spectrum
power = abs2.(img_freq)

# Filtering in frequency domain
# (Zero out high frequencies)
img_freq[512:end, :] .= 0
img_filtered = CUFFT.ifft(img_freq)
```

*Performance:* 10-50× faster than CPU FFT for large arrays.

*Common Applications:*
- Signal processing (filtering, convolution)
- Image processing
- Solving PDEs (spectral methods)
- Correlation analysis

== 📊 Library Performance Comparison

#figure(
  table(
    columns: 5,
    align: center,
    [*Operation*], [*Library*], [*Array Size*], [*Speedup*], [*Use Case*],
    [Matrix Multiply], [CUBLAS], [2000×2000], [20-50×], [Deep learning, simulation],
    [Linear Solve], [CUSOLVER], [1000×1000], [10-30×], [PDEs, optimization],
    [Sparse MatVec], [CUSPARSE], [10⁶, 1% dense], [5-20×], [Graphs, FEM],
    [FFT], [CUFFT], [10⁶ points], [10-50×], [Signal processing],
  ),
  caption: [Typical GPU speedups vs CPU (depends on hardware)]
)

= Performance Optimization

== Understanding the Memory Bottleneck

#box(fill: rgb("ffe4e1"), inset: 1em, radius: 5pt, width: 100%)[
  *⚠️ The #1 Performance Killer*
  
  CPU-GPU data transfer is ~100× slower than GPU computation!
]

Memory bandwidth is often the limiting factor in GPU performance @Kirk2016. Understanding the memory hierarchy is crucial for optimization.

*Bandwidth Comparison:*
- GPU memory bandwidth: ~1000-2000 GB/s (HBM2/GDDR6X)
- CPU-GPU transfer (PCIe 3.0/4.0/5.0): ~16-64 GB/s
- CPU memory bandwidth: ~50-100 GB/s

*🎯 Golden Rule:* Minimize CPU-GPU transfers! Keep data on GPU as long as possible.

```julia
# ❌ BAD: Transfer after every operation
x_cpu = randn(10000)
for i in 1:100
    x_gpu = CuArray(x_cpu)    # Upload (slow!)
    x_gpu .+= 1
    x_cpu = Array(x_gpu)       # Download (slow!)
end

# ✓ GOOD: Keep data on GPU
x_gpu = CuArray(randn(10000))  # Upload once
for i in 1:100
    x_gpu .+= 1                # All on GPU
end
result = Array(x_gpu)          # Download once
```

*🎯 Performance Impact:* Good version is ~100× faster!

== Kernel Fusion

Broadcasting automatically fuses operations into single kernels:

```julia
x = CUDA.randn(10000)

# ❌ Multiple kernels (slow)
y = x .^ 2          # Kernel 1: square
z = sin.(y)         # Kernel 2: sine
w = z .+ 1          # Kernel 3: add

# ✓ Single fused kernel (fast)
w = @. sin(x^2) + 1  # One kernel does all three!
```

*Why is fusion faster?*
1. Fewer kernel launches (each launch has overhead)
2. Less memory traffic (intermediate results stay in registers)
3. Better instruction-level parallelism

*Speedup:* 2-10× for chains of operations.

== Memory Access Patterns

#box(fill: rgb("fff5ee"), inset: 1em, radius: 5pt, width: 100%)[
  *🎯 Coalesced Memory Access*
  
  GPUs are fastest when adjacent threads access adjacent memory locations @Sanders2010.
]

```julia
# ✓ GOOD: Sequential access (coalesced)
function good_kernel(A, B)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds if i <= length(A)
        B[i] = A[i] * 2  # Thread i accesses element i
    end
    return nothing
end

# ❌ BAD: Strided access (non-coalesced)
function bad_kernel(A, B, stride)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds if i <= length(A)
        B[i] = A[i * stride]  # Threads access non-adjacent elements
    end
    return nothing
end
```

*Performance impact:* Coalesced access is 5-10× faster!

== 🛠️ Profiling GPU Code

```julia
using CUDA
using BenchmarkTools

# Basic timing (with synchronization!)
@time CUDA.@sync operation()

# Detailed benchmarking
@benchmark CUDA.@sync $operation()

# GPU profiler (detailed breakdown)
CUDA.@profile operation()
```

*What to monitor:*
1. *Kernel execution time* - actual GPU computation
2. *Memory transfers* - CPU ↔ GPU data movement
3. *GPU utilization* - is your GPU fully busy?
4. *Memory bandwidth* - are you memory-bound?

= Resource Management and Monitoring

== Monitoring GPU Usage

```julia
using CUDA

# Memory status
CUDA.memory_status()

# Device information
dev = device()
println("Device: ", CUDA.name(dev))
println("Total memory: ", CUDA.totalmem(dev) ÷ (1024^3), " GB")

# Using NVML for detailed monitoring
nvml_dev = NVML.Device(parent_uuid(device()))

# Power consumption
power = NVML.power_usage(nvml_dev)  # milliwatts
println("Power: ", power / 1000, " W")

# Utilization
util = NVML.utilization_rates(nvml_dev)
println("GPU: ", util.compute, "%")
println("Memory: ", util.memory, "%")

# Active processes
procs = NVML.compute_processes(nvml_dev)
for proc in procs
    println("PID: ", proc.pid, " Memory: ", proc.used_gpu_memory ÷ (1024^2), " MB")
end
```

*🎯 Optimization Tip:* If GPU utilization < 50%, you may have:
- Too small workload (increase array sizes)
- Too many CPU-GPU transfers
- Inefficient kernel configuration

== Memory Management

```julia
# Check memory usage
CUDA.memory_status()

# Manual memory management
a = CUDA.rand(1000, 1000)
CUDA.unsafe_free!(a)  # Free immediately (dangerous!)

# Automatic garbage collection
a = CUDA.rand(1000, 1000)
a = nothing
GC.gc()  # Trigger cleanup

# Clear all GPU memory
CUDA.reclaim()  # Force cleanup of all freed memory
```

*🎯 Out of Memory?*
1. Process data in batches
2. Use `CUDA.reclaim()` periodically
3. Reduce array sizes
4. Check for memory leaks (unreleased references)

= Advanced Topics

== Multiple GPUs

If you have multiple GPUs, you can use them all:

```julia
# List all devices
devices = CUDA.devices()
println("Found ", length(devices), " GPUs")

# Select device
CUDA.device!(0)  # Use first GPU
CUDA.device!(1)  # Switch to second GPU

# Use specific device for a block of code
device(0) do
    # This code runs on GPU 0
    x = CUDA.randn(1000)
    y = x .+ 1
end

device(1) do
    # This code runs on GPU 1
    x = CUDA.randn(1000)
    y = x .+ 1
end
```

*Common Pattern:* Split large arrays across GPUs and compute in parallel.

== Atomic Operations

For parallel reductions and synchronization:

```julia
function atomic_sum_kernel(A, result)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    @inbounds if i <= length(A)
        # Atomic add: thread-safe accumulation
        CUDA.@atomic result[1] += A[i]
    end
    
    return nothing
end

# Usage
A = CUDA.randn(10000)
result = CUDA.zeros(1)
@cuda threads=256 blocks=cld(10000, 256) atomic_sum_kernel(A, result)
println("Sum: ", Array(result)[1])
```

*When to use:* Histograms, reductions, counters, parallel accumulation.

== Shared Memory

Fast memory shared within a block for thread cooperation:

```julia
function shared_memory_example(A, B)
    # Allocate shared memory (must be static size)
    tid = threadIdx().x
    bid = blockIdx().x
    i = (bid - 1) * blockDim().x + tid
    
    # Each block has its own shared memory
    shared = @cuStaticSharedMem(Float64, 256)
    
    # Load data to shared memory
    @inbounds if i <= length(A)
        shared[tid] = A[i]
    end
    
    # Wait for all threads in block
    sync_threads()
    
    # Use shared memory (much faster than global memory!)
    @inbounds if i <= length(A)
        B[i] = shared[tid] * 2
    end
    
    return nothing
end
```

*Benefits:* Shared memory is ~100× faster than global memory!

= 🚨 Common Pitfalls and Solutions

#box(fill: rgb("ffe4e1"), inset: 1em, radius: 5pt, width: 100%)[
  *Troubleshooting Guide*
]

#figure(
  table(
    columns: (auto, auto, auto),
    align: left,
    [*Problem*], [*Symptom*], [*Solution*],
    
    [Scalar indexing], [ErrorException or very slow], [Use array operations or `@allowscalar`],
    [Out of memory], [CUDA_ERROR_OUT_OF_MEMORY], [Process in batches, use `CUDA.reclaim()`],
    [Wrong results], [Assertion fails], [Check bounds, verify synchronization],
    [Slow performance], [No speedup vs CPU], [Profile, minimize transfers, check size],
    [Kernel errors], [Launch failures], [Use `@device_code_warntype`],
    [Type instability], [Slow compilation], [Add type annotations],
    [Not using GPU], [Task CPU still shows 100%], [Forgot `CUDA.@sync`?],
  ),
  caption: [Common GPU programming issues and fixes]
)

*🎯 Debugging Checklist:*
1. ✓ Using `CUDA.@sync` for proper timing?
2. ✓ Avoiding scalar indexing?
3. ✓ Array size large enough (> 1000 elements)?
4. ✓ Minimizing CPU-GPU transfers?
5. ✓ Using library functions when available?

= 📊 Method Selection Guide

#box(fill: rgb("f5f5dc"), inset: 1em, radius: 5pt, width: 100%)[
  *🎯 Which Approach Should You Use?*
]

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: left,
    [*Approach*], [*When to Use*], [*Pros*], [*Cons*],
    
    [Broadcasting], [Element-wise ops], [Easy, automatic fusion], [Less control],
    [Custom kernels], [Complex algorithms], [Full control, shared memory], [More complex],
    [CUBLAS], [Matrix operations], [Highly optimized], [Fixed operations],
    [CUSPARSE], [Sparse matrices], [Memory efficient], [Overhead for small matrices],
    [CUFFT], [Fourier transforms], [Very fast], [Specific use case],
  ),
  caption: [Decision guide for different GPU programming approaches]
)

*🌳 Decision Tree:*

1. *Can you use broadcasting?* → Start there (80% of cases)
2. *Is it a standard operation?* → Use library (CUBLAS/CUSPARSE/CUFFT)
3. *Need shared memory or atomics?* → Write custom kernel
4. *Still too slow?* → Profile and optimize

= 🎓 Lecture Summary

#box(fill: rgb("e6f3ff"), inset: 1.2em, radius: 8pt, width: 100%)[
  *🌟 Key Takeaways*
  
  1. *GPUs excel at data parallelism* - same operation on many elements
  2. *Minimize CPU-GPU transfers* - keep data on GPU as long as possible
  3. *Avoid scalar indexing* - work with entire arrays
  4. *Start high-level* - broadcasting works for most tasks
  5. *Use libraries when possible* - CUBLAS, CUSPARSE, CUFFT are highly optimized
  6. *Profile before optimizing* - measure first, optimize second
  7. *Memory is often the bottleneck* - not computation!
]

*Performance Hierarchy (fastest to slowest):*
1. 🚀 Library functions (CUBLAS, CUFFT) - use when possible
2. 🏃 Optimized custom kernels with shared memory
3. 🚶 Broadcasting and array operations
4. 🐌 Scalar indexing and CPU-GPU transfers

= 🏋️‍♂️ Exercises & Hands-On Practice

#box(fill: rgb("fffacd"), inset: 1em, radius: 5pt, width: 100%)[
  *💻 Programming Exercises*
  
  1. *Benchmarking Study:*
     - Compare CPU vs GPU for vector addition with sizes: 10, 100, 1K, 10K, 100K, 1M
     - Plot speedup vs array size
     - Find the "break-even" point where GPU becomes faster
  
  2. *Custom Function Broadcasting:*
     - Implement the logistic function: $f(x) = 1/(1 + e^(-x))$
     - Broadcast it over a GPU array of 1M elements
     - Compare performance with CPU
  
  3. *Kernel Programming:*
     - Write a kernel for SAXPY: $y = a x + y$
     - Compare your kernel with broadcasting: `@. y = a * x + y`
     - Which is faster? Why?
  
  4. *Library Usage:*
     - Solve a linear system $A x = b$ where $A$ is 1000×1000
     - Do it on both CPU and GPU
     - Measure and compare performance
  
  5. *Real-World Application:*
     - Implement parallel Monte Carlo π estimation
     - Generate N random points in unit square
     - Count how many fall inside unit circle
     - Estimate π = 4 × (points in circle / total points)
]

#box(fill: rgb("f0fff0"), inset: 1em, radius: 5pt, width: 100%)[
  *🤔 Conceptual Questions*
  
  1. Why is the GPU "memory-bound" rather than "compute-bound" for most applications?
  2. Explain why scalar indexing is so slow on GPUs.
  3. When would you choose BFGS (CPU) over gradient descent (GPU) for optimization?
  4. Why do we need blocks in addition to threads?
  5. What is the trade-off between using more threads vs more blocks?
]

= 🔍 Further Resources and Next Steps

*Official Documentation:*
1. CUDA.jl Documentation @CUDAjl - Comprehensive guide and API reference
2. JuliaGPU Organization @JuliaGPU - Community hub for GPU computing in Julia
3. JuliaComputing GPU Training @JuliaTraining - Advanced tutorials and examples

*Books and Textbooks:*
- Kirk & Hwu @Kirk2016: "Programming Massively Parallel Processors" - Excellent textbook on GPU programming fundamentals
- Sanders & Kandrot @Sanders2010: "CUDA by Example" - Practical introduction to CUDA programming
- NVIDIA CUDA C Programming Guide @NVIDIACUDA - Comprehensive reference for CUDA

*Research Papers:*
- Besard et al. @Besard2019: "Effective Extensible Programming: Unleashing Julia on GPUs" - Technical details of CUDA.jl
- Besard et al. @Besard2018: "Rapid software prototyping for heterogeneous and distributed platforms" - Julia's approach to GPU computing
- Nickolls & Dally @Nickolls2010: "The GPU Computing Era" - Overview of GPU architecture evolution

*Julia GPU Ecosystem:*
- `CUDA.jl` @Besard2019 - NVIDIA GPU support with native Julia integration
- `AMDGPU.jl` - AMD GPU support
- `KernelAbstractions.jl` @KernelAbstractions - Write portable GPU code
- `Flux.jl` @Innes2018 - Machine learning with automatic GPU support

*Online Resources:*
- NVIDIA Developer Blog - Latest GPU programming techniques and optimizations
- Julia Discourse GPU section - Community Q&A and discussions
- JuliaCon GPU talks - Annual conference presentations on GPU computing

*Next Steps:*
1. Complete the hands-on exercises
2. Try porting your own code to GPU
3. Explore GPU-accelerated machine learning with Flux.jl
4. Learn about distributed computing across multiple GPUs
5. Contribute to the JuliaGPU ecosystem

*🎯 Final Challenge:* Take a computational problem from your research and accelerate it with GPUs. Aim for at least 10× speedup!

= References

#bibliography("refs.bib")
