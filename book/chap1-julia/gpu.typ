#import "../book.typ": book-page, cross-link, heading-reference
#import "@preview/cetz:0.4.1": *
#show: book-page.with(title: "Arrays and GPU Programming")

#set math.equation(numbering: "(1)")
#let boxed(it, width: 100%) = block(stroke: 1pt, inset: 10pt, radius: 4pt, width: width)[#it]

#show ref: it => {
  let el = it.element
  if el != none and el.func() == heading {
    // Override heading references.
    link(el.location(), el.body)
  } else {
    // Other references as usual.
    it
  }
}

#align(center, [= Arrays and GPU Programming\
_Jin-Guo Liu_])

= CUDA programming with Julia

CUDA programming is a parallel computing platform and programming model developed by NVIDIA for performing general-purpose computations on its GPUs (Graphics Processing Units). CUDA stands for Compute Unified Device Architecture.

== Step by step guide
1. Make sure you have a NVIDIA GPU device and its driver is properly installed.
  ```bash
  nvidia-smi
  ```
2. Install the #link("https://github.com/JuliaGPU/CUDA.jl")[CUDA.jl] package, and disable scalar indexing of CUDA arrays.
CUDA.jl @Besard2019 provides wrappers for several CUDA libraries that are part of the CUDA toolkit:

- Driver library: manage the device, launch kernels, etc.
- CUBLAS: linear algebra
- CURAND: random number generation
- CUFFT: fast fourier transform
- CUSPARSE: sparse arrays
- CUSOLVER: decompositions & linear systems

There's also support for a couple of libraries that aren't part of the CUDA toolkit, but are commonly used:

- CUDNN: deep neural networks
- CUTENSOR: linear algebra with tensors

```julia
CUDA.versioninfo()
```

3. Choose a device (if multiple devices are available).

```julia
devices()
dev = CuDevice(0)
```

grid > block > thread

```julia
attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
```

4. Create a CUDA Array

```julia
CUDA.zeros(10)
cuarray1 = CUDA.randn(10)
@test_throws ErrorException cuarray1[3]
CUDA.@allowscalar cuarray1[3] += 10
CuArray(randn(10))   # Upload a CPU Array to GPU
```

5. Compute

Computing a function on GPU Arrays
1. Launch a CUDA job - a few micro seconds
2. Launch more CUDA jobs...
3. Synchronize threads - a few micro seconds

Computing matrix multiplication.

```julia
@elapsed rand(2000,2000) * rand(2000,2000)
@elapsed CUDA.@sync CUDA.rand(2000,2000) * CUDA.rand(2000,2000)
```

*Example: Broadcasting a native Julia function*

Julia -> LLVM (optimized for CUDA) -> CUDA

```julia
factorial(n) = n == 1 ? 1 : factorial(n-1)*n

function poor_besselj(ν::Int, z::T; atol=eps(T)) where T
    k = 0
    s = (z/2)^ν / factorial(ν)
    out = s::T
    while abs(s) > atol
        k += 1
        s *= -(k+ν) * (z/2)^2 / k
        out += s
    end
    out
end
x = CUDA.CuArray(0.0:0.01:10)
poor_besselj.(1, x)
```

6. manage your GPU devices

```julia
nvml_dev = NVML.Device(parent_uuid(device()))

NVML.power_usage(nvml_dev)

NVML.utilization_rates(nvml_dev)

NVML.compute_processes(nvml_dev)
```

== Further reading
1. #link("https://github.com/JuliaComputing/Training")[JuliaComputing/Training]
2. #link("https://github.com/JuliaGPU/CUDA.jl")[JuliaGPU/CUDA.jl]

#bibliography("refs.bib")