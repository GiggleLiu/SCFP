#import "@preview/touying:0.6.1": *
#import "@preview/touying-simpl-hkustgz:0.1.2": *
#import "@preview/cetz:0.4.1": canvas, draw, tree, vector, decorations, coordinate
#import "@preview/cetz-plot:0.1.2": plot
#import "@preview/ctheorems:1.1.3": *
#set math.mat(row-gap: 0.1em, column-gap: 0.7em)

#let definition = thmbox("definition", "Definition", inset: (x: 1.2em, top: 1em), base: none)
#let proposition = thmbox("proposition", "Proposition", inset: (x: 1.2em, top: 1em), base: none)
#let theorem = thmbox("theorem", "Theorem", base: none)

#let globalvars = state("t", 0)
#let timecounter(minutes) = [
  #globalvars.update(t => t + minutes)
  #place(dx: 100%, dy: 0%, align(right, text(16pt, red)[#context globalvars.get()min]))
]

#set cite(style: "apa")

#show raw.where(block: true): it=>{
  block(radius:4pt, fill:gray.transparentize(90%), inset:1em, width:99%, text(14pt, it))
}

#show: hkustgz-theme.with(
  config-info(
    title: [GPU Programming with Julia],
    subtitle: [High-Performance Parallel Computing with CUDA.jl],
    author: [Jin-Guo Liu],
    date: datetime.today(),
    institution: [HKUST(GZ) - FUNH - Advanced Materials Thrust],
  ),
)

#let bob(loc, rescale: 1, flip: false, label: none, words: none) = {
  import draw: *
  let r = 0.4 * rescale
  let xr = if flip { -r } else { r }
  circle(loc, radius: (0.8 * r, r), name: "bob")
  circle((rel: (xr * 0.4, 0.2 * r), to: loc), radius: (0.2 * r, 0.18 * r), name: "eye", stroke: none, fill: black)
  line((rel: (-1.5 * xr, -r), to: "bob"), (rel: (-0.6 * xr, -3.5 * r), to: "bob"), (rel: (0.7 * xr, -3.5 * r), to: "bob"), (rel: (1.2 * xr, -r), to: "bob"), stroke: (paint: black, thickness: 1pt), name: "line1", close: true)
  line((anchor: 31%, name: "line1"), (loc.at(0) - 0.5 * xr, loc.at(1) - 5 * r))
  line((anchor: 40%, name: "line1"), (loc.at(0) + 0.5 * xr, loc.at(1) - 5 * r))
  line((anchor: 20%, name: "line1"), (loc.at(0) + 0.5 * xr, loc.at(1) - 2 * r))
  line((anchor: 59%, name: "line1"), (loc.at(0) + 2 * xr, loc.at(1) - 2 * r))
  if label != none {
    content((loc.at(0), loc.at(1) - 1.5 * r), label)
  }
  if words != none {
    content((loc.at(0) + 10 * xr, loc.at(1) - 1.5 * r), box(width: rescale * 270pt, words))
  }
}

#title-slide()
#outline-slide()

= Introduction: Why GPU Computing?

== The Parallel Computing Revolution

Modern scientific computing requires massive parallel processing:
- Training neural networks
- Molecular dynamics simulations  
- Image and signal processing
- Solving large linear systems

*GPU Advantage*: Thousands of cores working in parallel

*Typical Speedup*: 10-100× for the right workloads @Nickolls2008

== CPU vs GPU: Different Philosophies

#figure(
  table(
    columns: 4,
    align: center,
    inset: 10pt,
    [*Device*], [*Cores*], [*Clock Speed*], [*Optimized For*],
    [Modern CPU], [8-64], [3-5 GHz], [Latency, complex logic],
    [Modern GPU], [1000s-10000s], [1-2 GHz], [Throughput, simple ops],
  )
)

*Key Concepts*:
- *CPU*: Few powerful cores, optimized for sequential tasks
- *GPU*: Many simple cores, optimized for parallel tasks
- *SIMT*: Single Instruction, Multiple Threads

== The Analogy

Imagine painting 10,000 fence posts:

*CPU Approach*: 
- Hire 8-16 highly skilled painters
- Each works very fast
- But limited by number of workers

*GPU Approach*:
- Hire 10,000 students with brushes
- Each works slower individually
- All work simultaneously!

*Result*: GPU finishes much faster for this parallel task

== When to Use GPU Computing?

#grid(
  columns: (1fr, 1fr),
  column-gutter: 20pt,
  [
    *✓ Excellent for GPUs:*
    - Large array operations
    - Matrix multiplication
    - Image/signal processing
    - Monte Carlo simulations
    - Deep learning
    - FFT computations
  ],
  [
    *✗ Poor for GPUs:*
    - Small datasets (< 1K elements)
    - Complex branching logic
    - Sequential algorithms
    - Heavy CPU ↔ GPU transfers
    - Small repeated transfers
  ]
)

*Golden Rule*: Speedup from parallelism must exceed transfer overhead!

== Why Julia for GPU Computing?

Julia's unique advantages for GPU programming @Besard2019:

1. *High-level syntax*: GPU code looks like regular Julia
2. *Native compilation*: Julia → LLVM → PTX → GPU
3. *Performance*: Comparable to hand-written CUDA C
4. *Composability*: GPU arrays work with Julia ecosystem
5. *Broadcasting magic*: Any function compiles to GPU automatically

```julia
# This just works on GPU!
x = CUDA.randn(1_000_000)
y = sin.(x) .+ cos.(x)  # Compiled to GPU automatically
```

= Getting Started with CUDA.jl

== Installation

*Step 1: Verify GPU*
```bash
nvidia-smi  # Check GPU and driver
```

*Step 2: Install CUDA.jl*
```julia
using Pkg
Pkg.add("CUDA")
using CUDA

CUDA.functional()  # Should return true
CUDA.versioninfo() # Check installation
```

*Requirements*:
- NVIDIA GPU (compute capability ≥ 3.5)
- Proper NVIDIA drivers
- Julia 1.6+

== Your First GPU Program

```julia
using CUDA

# Create array on CPU
x = ones(1000)

# Move to GPU
x_gpu = CuArray(x)

# Compute on GPU
y_gpu = x_gpu .+ 1  # All on GPU!

# Download result
y = Array(y_gpu)
```

*What happened*?
1. `CuArray()` uploads data to GPU memory
2. `.+` executes on GPU (1000 parallel additions!)
3. `Array()` downloads result back to CPU

== Live Demo: Setup and First Program

#figure(canvas(length: 1.8cm, {
  import draw: *
  bob((0, 0), rescale: 1, flip: true, words: box(stroke: black, inset: 10pt)[*Live coding*: Setting up CUDA.jl and first GPU computation])
}))

== Array Programming on GPU

```julia
using CUDA

# Create directly on GPU (faster!)
a = CUDA.zeros(100)
b = CUDA.ones(10, 10)
c = CUDA.randn(1000)
d = CUDA.fill(3.14, 50)

# Upload from CPU
cpu_data = randn(1000)
gpu_data = CuArray(cpu_data)

# Download to CPU
result = Array(gpu_data)
```

*Pro Tip*: Creating arrays directly on GPU is faster than creating on CPU and uploading!

== The Scalar Indexing Trap

#slide[
#box(fill: rgb("ffe4e1"), inset: 1em, width: 100%)[
  *⚠️ Performance Killer #1*
  
  Scalar indexing is DISABLED by default in CUDA.jl!
]

*Why so slow?* Each access requires CPU-GPU synchronization (~1000× slower!)

][
```julia
x = CUDA.randn(1000)

# ❌ This throws an error
# value = x[1]

# ✓ Enable if needed (but SLOW!)
CUDA.@allowscalar x[1]

# ✓ BETTER: Work with entire arrays
sum(x)        # Fast, runs on GPU
maximum(x)    # Fast, runs on GPU
```
]

== Broadcasting: The Secret Weapon

#slide[Broadcasting automatically parallelizes operations:

```julia
x = CUDA.randn(10000)

# Simple operations
y = x .^ 2
z = sin.(x) .+ cos.(x)
w = @. sqrt(abs(x)) + exp(-x^2)

# Custom functions work too!
f(x) = x > 0 ? sqrt(x) : 0.0
result = f.(x)  # Compiled for GPU automatically!
```
][*Compilation Pipeline*:
```
Julia code → LLVM IR → PTX assembly → GPU execution
```
]

== Performance: CPU vs GPU

#slide[```julia
using CUDA, LinearAlgebra

n = 2000
a_cpu = randn(n, n)
b_cpu = randn(n, n)

# CPU multiplication
@time a_cpu * b_cpu

# GPU multiplication  
a_gpu = CuArray(a_cpu)
b_gpu = CuArray(b_cpu)
@time CUDA.@sync a_gpu * b_gpu
```
][
*Typical Results*:
- 1000×1000: GPU ~2-5× faster
- 5000×5000: GPU ~20-50× faster

*Remember*: Always use `CUDA.@sync` for timing!
]

== Live Demo: Array Programming

#figure(canvas(length: 1.8cm, {
  import draw: *
  bob((0, 0), rescale: 1, flip: true, words: box(stroke: black, inset: 10pt)[*Live coding*: Broadcasting and array operations on GPU])
}))

== Kernel programming: Understanding GPU Architecture

GPUs use *SIMT* (Single Instruction, Multiple Threads):

#figure(canvas(length: 1cm, {
  import draw: *
  
  rect((0, 0), (12, 8), name: "grid")
  content((-1, 7.5), [*Grid*])
  
  // Block 1
  rect((0.5, 4), (3.5, 7.5), stroke: blue, name: "block1")
  content((2, 7), text(10pt, blue)[*Block 1*])
  for i in range(3) {
    for j in range(2) {
      circle((1 + i * 0.8, 4.5 + j * 0.8), radius: 0.2, fill: green)
    }
  }
  
  // Block 2
  rect((4, 4), (7, 7.5), stroke: blue, name: "block2")
  content((5.5, 7), text(10pt, blue)[*Block 2*])
  for i in range(3) {
    for j in range(2) {
      circle((4.5 + i * 0.8, 4.5 + j * 0.8), radius: 0.2, fill: green)
    }
  }
  
  // Warp annotation
  rect((0.7, 4.2), (3, 5.6), stroke: (dash: "dashed", paint: red))
  content((2, 3.8), text(9pt, red)[Warp (32 threads)])
}))

*Hierarchy*: Grid → Block → Warp (32 threads) → Thread

== SIMT and Thread Divergence

#box(fill: rgb("fff5ee"), inset: 1em, width: 100%)[
  *Critical Concept: Warps*
  
  - 32 threads execute *same instruction* simultaneously
  - If threads diverge (different `if` branches), execution serializes!
  - *Impact*: Can slow down by 32× in worst case
]

==
*Example of divergence*:
```julia
function divergent_kernel(A)
    i = thread_id()
    if i % 2 == 0
        A[i] = sin(A[i])    # Half of warp
    else
        A[i] = cos(A[i])    # Other half
    end
end
```

*Avoid*: Branching based on thread ID within a warp!

== Your First CUDA Kernel

```julia
using CUDA

function print_kernel()
    # Compute unique thread ID
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    @cuprintf "Thread %ld\n" Int(i)
    return  # Must return nothing
end

# Launch: 2 blocks × 256 threads = 512 threads
@cuda threads=256 blocks=2 print_kernel()
CUDA.synchronize()
```

==
*Thread ID Formula* (Julia 1-based):
```
global_id = (block_id - 1) × threads_per_block + local_thread_id
```

== Practical Kernel Example

Fill array with indices:

```julia
function one2n_kernel(A)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    # Always check bounds!
    @inbounds if i <= length(A)
        A[i] = i
    end
    
    return nothing
end

A = CUDA.zeros(Int, 10000)
@cuda threads=256 blocks=cld(10000, 256) one2n_kernel(A)

Array(A)  # [1, 2, 3, ..., 10000]
```

*Key Points*:
- Always check bounds
- Return `nothing`
- Use `@inbounds` for performance (when safe)

== Choosing Thread/Block Configuration

*Rules of Thumb*:
- Threads per block: 128-1024 (typically 256 or 512)
- Must be multiple of 32 (warp size)
- Total threads ≥ array size

```julia
function launch_config(n; threads_per_block=256)
    threads = threads_per_block
    blocks = cld(n, threads)  # Ceiling division
    return (threads=threads, blocks=blocks)
end

# Usage
config = launch_config(10000)
@cuda threads=config.threads blocks=config.blocks my_kernel(data)
```

==
*Why not always 1024?* Occupancy! Fewer threads/block → more concurrent blocks

== Kernel Restrictions

#grid(
  columns: (1fr, 1fr),
  column-gutter: 15pt,
  [
    *❌ Not Allowed:*
    - Dynamic allocation
    - I/O (except `@cuprintf`)
    - Most stdlib functions
    - Recursion (limited)
  ],
  [
    *✓ Allowed:*
    - Basic arithmetic
    - Math functions
    - Control flow
    - Array indexing
    - Thread/block functions
  ]
)

*Debugging Tools*:
- `@device_code_warntype` - Type checking
- `@device_code_llvm` - LLVM IR
- `@device_code_ptx` - GPU assembly

== Live Demo: CUDA Kernels

#figure(canvas(length: 1.8cm, {
  import draw: *
  bob((0, 0), rescale: 1, flip: true, words: box(stroke: black, inset: 10pt)[*Live coding*: Writing and launching CUDA kernels])
}))

= High-Performance GPU Libraries

== CUBLAS: Linear Algebra Powerhouse

CUBLAS provides highly optimized BLAS operations @NVIDIA2023:

```julia
using CUDA, LinearAlgebra

n = 2000
A = CUDA.randn(n, n)
B = CUDA.randn(n, n)

# Matrix multiplication (uses CUBLAS)
C = A * B

# Vector operations
v = CUDA.randn(n)
norm(v)      # Vector norm
dot(v, v)    # Dot product
A * v        # Matrix-vector mult
```

*Performance*: Achieves ~90% of theoretical peak GPU performance!

== CUSPARSE: Sparse Matrices

For matrices with mostly zeros:

```julia
using CUDA, SparseArrays

# Create sparse matrix
A_cpu = sprand(10000, 10000, 0.01)  # 1% density
A_gpu = CUSPARSE.CuSparseMatrixCSC(A_cpu)

# Operations
v = CUDA.randn(10000)
result = A_gpu * v  # Sparse mat-vec

# Sparse-sparse
B_gpu = A_gpu * A_gpu
```

*When to use*: Density < 5-10%, graph algorithms, PDEs, FEM

== CUFFT: Fast Fourier Transform

GPU-accelerated FFT @NVIDIACUDA:

```julia
using CUDA, FFTW

# 1D FFT
x = CUDA.randn(100000)
X = CUFFT.fft(x)

# 2D FFT (images)
img = CUDA.randn(1024, 1024)
img_freq = CUFFT.fft(img)

# Inverse
img_reconstructed = CUFFT.ifft(img_freq)
```

*Performance*: 10-50× faster than CPU FFT

*Applications*: Signal processing, image filtering, PDE solvers

== Library Performance Comparison

#figure(
  table(
    columns: 5,
    align: center,
    inset: 8pt,
    [*Operation*], [*Library*], [*Size*], [*Speedup*], [*Use Case*],
    [Matrix Mult], [CUBLAS], [2000×2000], [20-50×], [Deep learning],
    [Linear Solve], [CUSOLVER], [1000×1000], [10-30×], [PDEs],
    [Sparse MatVec], [CUSPARSE], [10⁶, 1%], [5-20×], [Graphs, FEM],
    [FFT], [CUFFT], [10⁶ pts], [10-50×], [Signal proc.],
  )
)

*Rule*: Always use libraries when available - they're highly optimized!

== Live Demo: GPU Libraries

#figure(canvas(length: 1.8cm, {
  import draw: *
  bob((0, 0), rescale: 1, flip: true, words: box(stroke: black, inset: 10pt)[*Live coding*: Using CUBLAS, CUSPARSE, and CUFFT])
}))

= Performance Optimization

== The Memory Bottleneck

#box(fill: rgb("ffe4e1"), inset: 1em, width: 100%)[
  *⚠️ Performance Killer #2*
  
  CPU-GPU data transfer is ~100× slower than GPU computation!
]

*Bandwidth Comparison* @Kirk2016:
- GPU memory: ~1000-2000 GB/s
- PCIe transfer: ~16-64 GB/s
- CPU memory: ~50-100 GB/s

*Golden Rule*: Minimize transfers! Keep data on GPU.

== Memory Transfer Anti-Pattern

```julia
# ❌ BAD: Transfer every iteration
x_cpu = randn(10000)
for i in 1:100
    x_gpu = CuArray(x_cpu)    # Upload
    x_gpu .+= 1
    x_cpu = Array(x_gpu)       # Download
end

# ✓ GOOD: Keep on GPU
x_gpu = CuArray(randn(10000))  # Upload once
for i in 1:100
    x_gpu .+= 1                # All on GPU
end
result = Array(x_gpu)          # Download once
```

*Performance Impact*: Good version is ~100× faster!

== Kernel Fusion

Broadcasting fuses operations automatically:

```julia
x = CUDA.randn(10000)

# ❌ Multiple kernels (3 launches, 3 memory passes)
y = x .^ 2
z = sin.(y)
w = z .+ 1

# ✓ Single fused kernel (1 launch, 1 memory pass)
w = @. sin(x^2) + 1
```

*Why faster*?
1. Fewer kernel launches (overhead)
2. Less memory traffic (intermediate results in registers)
3. Better instruction parallelism

*Speedup*: 2-10× for operation chains

== Coalesced Memory Access

#box(fill: rgb("fff5ee"), inset: 1em, width: 100%)[
  *Memory Access Pattern Matters*
  
  Adjacent threads should access adjacent memory locations @Sanders2010
]

```julia
# ✓ GOOD: Coalesced (thread i → element i)
function good_kernel(A, B)
    i = thread_id()
    @inbounds if i <= length(A)
        B[i] = A[i] * 2
    end
end

# ❌ BAD: Strided access
function bad_kernel(A, B, stride)
    i = thread_id()
    @inbounds if i <= length(A)
        B[i] = A[i * stride]  # Non-sequential!
    end
end
```

*Performance Impact*: Coalesced is 5-10× faster!

== Profiling GPU Code

```julia
using CUDA, BenchmarkTools

# Basic timing
@time CUDA.@sync operation()

# Detailed benchmarking
@benchmark CUDA.@sync $operation()

# GPU profiler
CUDA.@profile operation()
```

*Monitor*:
1. Kernel execution time
2. Memory transfers
3. GPU utilization (should be > 80%)
4. Memory bandwidth utilization

== Optimization Checklist

#box(fill: rgb("e6f3ff"), inset: 1em, width: 100%)[
  *Performance Checklist* ✓
  
  1. Using `CUDA.@sync` for timing?
  2. Avoiding scalar indexing?
  3. Minimizing CPU-GPU transfers?
  4. Using library functions when available?
  5. Fusing operations with broadcasting?
  6. Array size large enough (> 1000)?
  7. Checking GPU utilization?
]

= Advanced Topics

== Multiple GPUs

```julia
# List devices
devices = CUDA.devices()

# Select device
CUDA.device!(0)  # First GPU
CUDA.device!(1)  # Second GPU

# Use specific device
device(0) do
    x = CUDA.randn(1000)
    y = x .+ 1
end

device(1) do
    x = CUDA.randn(1000)
    y = x .+ 1
end
```

*Pattern*: Split data across GPUs, compute in parallel

== Atomic Operations

For parallel reductions:

```julia
function atomic_sum(A, result)
    i = thread_id()
    
    @inbounds if i <= length(A)
        CUDA.@atomic result[1] += A[i]
    end
    
    return nothing
end

A = CUDA.randn(10000)
result = CUDA.zeros(1)
@cuda threads=256 blocks=40 atomic_sum(A, result)
```

*Use cases*: Histograms, counters, parallel accumulation

== Shared Memory

Fast memory within a block:

```julia
function shared_example(A, B)
    tid = threadIdx().x
    i = thread_id()
    
    # Allocate shared memory
    shared = @cuStaticSharedMem(Float64, 256)
    
    # Load to shared memory
    @inbounds if i <= length(A)
        shared[tid] = A[i]
    end
    
    sync_threads()  # Wait for all threads
    
    # Use shared memory (100× faster!)
    @inbounds if i <= length(A)
        B[i] = shared[tid] * 2
    end
end
```

*Benefit*: ~100× faster than global memory!

== Exercise 1: Benchmarking Study

*Task*: Compare CPU vs GPU for vector addition

```julia
using CUDA, BenchmarkTools

function benchmark_sizes()
    sizes = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
    
    for n in sizes
        # CPU
        x_cpu = randn(n)
        t_cpu = @belapsed $x_cpu .+ 1
        
        # GPU
        x_gpu = CUDA.randn(n)
        t_gpu = @belapsed CUDA.@sync $x_gpu .+ 1
        
        println("n=$n: Speedup = $(t_cpu/t_gpu)×")
    end
end
```

*Question*: At what size does GPU become faster?

== Exercise 2: Custom Function Broadcasting

*Task*: Implement logistic function on GPU

```julia
# Logistic function
logistic(x) = 1 / (1 + exp(-x))

# Broadcast over GPU array
x_gpu = CUDA.randn(1_000_000)
y_gpu = logistic.(x_gpu)

# Compare with CPU
x_cpu = Array(x_gpu)
@benchmark logistic.($x_cpu)
@benchmark CUDA.@sync logistic.($x_gpu)
```

*Measure*: CPU vs GPU performance

== Exercise 3: SAXPY Kernel

*Task*: Implement SAXPY ($y = a x + y$) as a kernel

```julia
function saxpy_kernel!(a, x, y)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds if i <= length(x)
        y[i] = a * x[i] + y[i]
    end
    return nothing
end

# Test
n = 10_000
a = 2.0
x = CUDA.randn(n)
y = CUDA.randn(n)

@cuda threads=256 blocks=cld(n,256) saxpy_kernel!(a, x, y)
```

*Compare*: Kernel vs broadcasting `@. y = a * x + y`

== Exercise 4: Monte Carlo π Estimation

*Task*: Estimate π using parallel random sampling

```julia
function estimate_pi_kernel(n_samples, hits)
    i = thread_id()
    @inbounds if i <= n_samples
        x = CUDA.rand()
        y = CUDA.rand()
        if x^2 + y^2 <= 1.0
            CUDA.@atomic hits[1] += 1
        end
    end
    return nothing
end

n = 10_000_000
hits = CUDA.zeros(Int, 1)
# Implement and measure!
```

*Estimate*: π ≈ 4 × (points in circle / total points)

== Live Demo: Hands-On Session

#figure(canvas(length: 1.8cm, {
  import draw: *
  bob((0, 0), rescale: 1, flip: true, words: box(stroke: black, inset: 10pt)[*Live coding*: Working through exercises together])
}))

= Resources and Next Steps

== Further Learning

*Official Documentation*:
- CUDA.jl Documentation @CUDAjl
- JuliaGPU Organization @JuliaGPU
- JuliaComputing Training @JuliaTraining

*Textbooks*:
- Kirk & Hwu @Kirk2016: "Programming Massively Parallel Processors"
- Sanders & Kandrot @Sanders2010: "CUDA by Example"

*Research Papers*:
- Besard et al. @Besard2019: CUDA.jl technical details
- Nickolls et al. @Nickolls2008: GPU architecture

== Julia GPU Ecosystem

*Key Packages*:
- `CUDA.jl` - NVIDIA GPU support
- `AMDGPU.jl` - AMD GPU support
- `KernelAbstractions.jl` - Portable GPU code
- `Flux.jl` @Innes2018 - ML with GPU support

*Community*:
- Julia Discourse (GPU section)
- JuliaCon GPU talks
- NVIDIA Developer Blog

==
#bibliography("refs.bib")

