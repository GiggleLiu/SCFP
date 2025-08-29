#import "@preview/touying:0.6.1": *
#import "@preview/touying-simpl-hkustgz:0.1.2": *
#import "@preview/cetz:0.4.1": *
#import "../shared/characters.typ": ina, christina, murphy

#let globalvars = state("t", 0)
#let timecounter(minutes) = [
  #globalvars.update(t => t + minutes)
  #place(dx: 100%, dy: 0%, align(right, text(16pt, red)[#context globalvars.get()min]))
]
#let clip(image, top: 0pt, bottom: 0pt, left: 0pt, right: 0pt) = {
  box(clip: true, image, inset: (top: -top, right: -right, left: -left, bottom: -bottom))
}
#set cite(style: "apa")

#let show-graph(vertices, edges, radius:0.2) = {
  import draw: *
  for (k, (i, j)) in vertices.enumerate() {
    circle((i, j), radius:radius, name: str(k), fill:white)
  }
  for (k, l) in edges {
    line(str(k), str(l))
  }
}

#show raw.where(block: true): it=>{
  block(radius:4pt, fill:gray.transparentize(90%), inset:1em, width:99%, text(it))
}

#show: hkustgz-theme.with(
  config-info(
    title: [Julia black magic],
    subtitle: [],
    author: [Jin-Guo Liu],
    date: datetime.today(),
    institution: [HKUST(GZ) - FUNH - Advanced Materials Thrust],
  ),
)

#outline-slide()

= Metaprogramming - Manipulating expressions

== Defining a macro
#box(text(16pt)[```julia
julia> macro myshow(ex)
    quote
        println("$($(QuoteNode(ex))) = $($ex)")
        $ex
    end
end
```])
- `quote ... end` - Quote the expression to avoid evaluation
- `QuoteNode` - Quote the expression to avoid evaluation
- `$` - Evaluate the expression
- _Remark_: Since we evaluate the expression in a string, we need to add two `$(...)` to evaluate the expression twice.

== `@macroexpand`
#box(text(16pt)[```julia
julia> x = 3

julia> @myshow x
x = 3
3

julia> ex = @macroexpand @myshow x
quote
    #= REPL[119]:3 =#
    Main.println("$(:x) = $(Main.x)")
    #= REPL[119]:4 =#
    Main.x
end
```])


== Manipulating AST

Abstract Syntax Tree (AST) is the internal representation of the code in Julia.

#box(text(16pt)[```julia
julia> typeof(ex)
Expr   # The type of expressions in Julia

julia> ex.head
:block

julia> ex.args
4-element Vector{Any}:
 :(#= REPL[131]:3 =#)   # LineNumberNode
 :(Main.println("$(:x) = $(Main.x)"))
 :(#= REPL[131]:4 =#)
 :(Main.x)
```])
- `LineNumberNode` - Line number information, used for debugging

== Examples

#box(text(16pt)[```julia
julia> ex_list = :([1, 2, 3])
:([1, 2, 3])

julia> ex_list.head
:vect

julia> ex_list.args
3-element Vector{Any}:
 1
 2
 3
```])

- `:(...)` - Quote the expression to avoid evaluation, similar to `quote ... end`

== More examples

#box(text(16pt)[```julia
julia> ex_call = :(f(x, y))

julia> ex_if = :(if x > 0; return true; else; return false; end)

julia> ex_for = :(for i in 1:10; println(i); end)

julia> ex_comp = :(x > y)

julia> ex_func = :(function f(x, y); return x + y; end)
```])


== The hygiene problem
An issue that arises in more complex macros is that of hygiene. In short, macros must ensure that the variables they introduce in their returned expressions do not accidentally clash with existing variables in the surrounding code they expand into.

#box(text(16pt)[```julia
julia> @myshow x = x + 1
ERROR: UndefVarError: `#36#x` not defined in `Main`
Suggestion: check for spelling errors or missing imports.
Stacktrace:
 [1] macro expansion
   @ REPL[146]:3 [inlined]
 [2] top-level scope
   @ REPL[148]:1
```])

== Avoid the hygiene problem - `esc`

#box(text(16pt)[```julia
julia> macro myshow(ex)
           quote
              println("$($(QuoteNode(ex))) = $($(esc(ex)))")
              $(esc(ex))
           end
       end
@myshow (macro with 1 method)

julia> @myshow x = x + 1
x = x + 1 = 4
5
```])

== Magic 1: `@nexprs`

#box(text(16pt)[```julia
using Base.Cartesian: @nexprs, @nloops, @nref

@nexprs 3 i -> @show i
```])

== Magic 2: `@nloops`
#box(text(16pt)[```julia
A = rand(10, 10)
s = 0.0
@nloops 2 i A d -> j_d = min(i_d, 5) begin
    s += @nref 2 A j
end
```])

Example: https://github.com/JuliaLang/julia/blob/c9ad04dcc73256bc24eb079f9f6506299b64b8ec/base/multidimensional.jl#L1696



== Implementing a DSL

The difference between syntax and semantics
- Syntax: How we write the code
- Semantics: What the code does

In a DSL, usually we do not want to change the syntax (otherwise, we need to rewrite the token parser!), but we want to change the semantics.

== Example: Syntax change

```julia
function f(x)
    return x + 1
end
```

To:

```c
foo f(x) {    // syntax changed!
  return x + 1
}
```

== Example: Semantics change

```julia
function f(x)
    return x + 1
end
```

To:

```julia
function f(x)
    return x - 1   # semantics changed!
end
```

== How to implement a DSL?

Live coding: Implement a mirror world:
- `+` -> `-`
- `-` -> `+`
- `*` -> `/`
- `/` -> `*`

=== More advanced example
https://github.com/GiggleLiu/ProblemReductions.jl/blob/b84db7fbc22199466c3f203394a10f08d5306fca/src/models/Circuit.jl#L137

== `@eval` for code generation

e.g. Generate similar code for different types
#box(text(16pt)[```julia
for NBIT in [16, 32, 64]
    @eval const $(Symbol(:Tropical, :F, NBIT)) = Tropical{$(Symbol(:Float, NBIT))}
    @eval const $(Symbol(:Tropical, :I, NBIT)) = Tropical{$(Symbol(:Int, NBIT))}
end
```]))

https://github.com/TensorBFS/TropicalNumbers.jl/blob/16c644484f25e5b38dd6a0ed0bc96f58065c2c40/src/TropicalNumbers.jl#L51

== Generated function

https://github.com/under-Peter/OMEinsum.jl/blob/b5fcba8c49bce17f12835ba8b9c1bdf449b0af59/src/Core.jl#L201

#box(text(16pt)[```julia
@inline @generated function subindex(indexer::DynamicEinIndexer{N}, ind::NTuple{N0,Int}) where {N,N0}
    ex = Expr(:call, :+, map(i->i==1 ? :(ind[indexer.locs[$i]]) : :((ind[indexer.locs[$i]]-1) * indexer.cumsize[$i]), 1:N)...)
    :(@inbounds $ex)
end
```])




= String literal and regular expression

== Example: String literal in OMEinsum

```julia
ein"ij,jk->ik"
```
https://github.com/under-Peter/OMEinsum.jl/blob/b5fcba8c49bce17f12835ba8b9c1bdf449b0af59/src/interfaces.jl#L21

== Regular Expressions

Regular expressions (regex) are powerful patterns for matching text.

=== Basic regex matching
#box(text(16pt)[```julia
m = match(r"(\w+) (\d+)", "June 24")
m.captures # ["June", "24"]
```])

- `r"..."` - Creates a regex pattern in Julia
- `\w` - Matches any word character (letters, digits, underscore)
- `\d` - Matches any digit character (0-9)
- `+` - Matches one or more of the preceding element, `*` - Matches zero or more of the preceding element
- `()` - Creates a capture group

==

=== Match an email address

#box(text(16pt)[```julia
email_pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
match(email_pattern, "contact@example.com").match  # "contact@example.com"
```])

- `[]` - Defines a character class (e.g., `[a-z]` matches lowercase letters)
- `{n}` - Matches exactly n occurrences of the preceding element
- `.` - Matches any character except newline

==

=== Match a URL

#box(text(16pt)[```julia
url_pattern = r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
match(url_pattern, "https://julia-lang.org").match  # "https://julia-lang.org"
```])

- `?` - Matches zero or one of the preceding element
- `(?:...)` is a non-capturing group

== Quiz: Match a date in format YYYY-MM-DD
#box(text(16pt)[```julia
# Match a date in format YYYY-MM-DD
date_pattern = r"??????"
date_match = match(date_pattern, "Event date: 2023-09-15")
date_match.captures  # ["2023", "09", "15"]

# Extract all hashtags from text
hashtags = [m.match for m in eachmatch(r"#\w+", "Julia is #fast and #productive")]
# ["#fast", "#productive"]
```])


== Find all matches
#box(text(16pt)[```julia
matches = collect(eachmatch(r"\d+", "10 apples, 20 oranges"))
[m.match for m in matches] # ["10", "20"]

matches = collect(eachmatch(r"\d+ \w+$", "10 apples, 20 oranges"))
[m.match for m in matches] # ["20 oranges"]

matches = collect(eachmatch(r"^\d+ \w+", "10 apples, 20 oranges"))
[m.match for m in matches] # ["10 apples"]
```])

- `^` - Matches the start of a string
- `$` - Matches the end of a string

== Named captures
#box(text(16pt)[```julia
person = match(r"(?<name>\w+) is (?<age>\d+)", "Alice is 29")
person[:name] # "Alice"
person[:age] # "29"
```])

- `(?<name>...)` Creates a named capture group

== Replacing text
#box(text(16pt)[```julia
replaced = replace("Hello World", r"Hello" => "Hi") # "Hi World"
```])


// = Multi-threading the multi-processing

// This section applies to multithreaded Julia code which, in each thread, performs linear algebra operations. Indeed, these linear algebra operations involve BLAS / LAPACK calls, which are themselves multithreaded. In this case, one must ensure that cores aren't oversubscribed due to the two different types of multithreading.

// Julia compiles and uses its own copy of OpenBLAS for linear algebra, whose number of threads is controlled by the environment variable OPENBLAS_NUM_THREADS. It can either be set as a command line option when launching Julia, or modified during the Julia session with BLAS.set_num_threads(N) (the submodule BLAS is exported by using LinearAlgebra). Its current value can be accessed with BLAS.get_num_threads().

// When the user does not specify anything, Julia tries to choose a reasonable value for the number of OpenBLAS threads (e.g. based on the platform, the Julia version, etc.). However, it is generally recommended to check and set the value manually. The OpenBLAS behavior is as follows:

// If OPENBLAS_NUM_THREADS=1, OpenBLAS uses the calling Julia thread(s), i.e. it "lives in" the Julia thread that runs the computation.
// If OPENBLAS_NUM_THREADS=N>1, OpenBLAS creates and manages its own pool of threads (N in total). There is just one OpenBLAS thread pool shared among all Julia threads.
// When you start Julia in multithreaded mode with JULIA_NUM_THREADS=X, it is generally recommended to set OPENBLAS_NUM_THREADS=1. Given the behavior described above, increasing the number of BLAS threads to N>1 can very easily lead to worse performance, in particular when $N<<X$. However this is just a rule of thumb, and the best way to set each number of threads is to experiment on your specific application.

// As an alternative to OpenBLAS, there exist several other backends that can help with linear algebra performance. Prominent examples include #link("https://github.com/JuliaLinearAlgebra/MKL.jl")[MKL.jl] and #link("https://github.com/JuliaMath/AppleAccelerate.jl")[AppleAccelerate.jl].

= CUDA programming
== The GPU

GPUs are specialized hardware designed for parallel processing. Let's compare the performance of NVIDIA A800 GPU with a high-end Intel CPU:

#table(
  columns: (auto, auto),
  inset: 10pt,
  align: horizon,
  [*NVIDIA A800 GPU*], [*Intel Xeon CPU (e.g., Platinum 8380)*],
  [80GB HBM2e memory], [Up to 4TB DDR4 memory],
  [40 TFLOPs FP32 performance], [~2.5 TFLOPs FP32 performance],
  [1,935 GB/s memory bandwidth], [~40 GB/s memory bandwidth per channel],
  [Thousands of CUDA cores (up to 6912)], [Up to 40 CPU cores],
  [Simpler cores, higher quantity], [More sophisticated cores with advanced features],
  [Lower clock speeds (1-2 GHz)], [Higher clock speeds (2-4 GHz)],
  [\$ 10,000 (enterprise)], [\$ 5,000- 10,000 (high-end server CPU)],
)

== Threads, blocks, and grids

In CUDA programming, computation is organized in a hierarchy:

#table(
  columns: (auto, auto),
  inset: 10pt,
  align: horizon,
  [*Concept*], [*Description*],
  [Thread], [Smallest execution unit, runs a single instance of the kernel],
  [Block], [Group of threads that can cooperate via shared memory and synchronization],
  [Grid], [Collection of blocks that execute the same kernel],
)

- `index = (blockIdx().x - 1) * blockDim().x + threadIdx().x`
- `stride = blockDim().x * gridDim().x`

== CuArray

#box(text(16pt)[```julia
using CUDA; CUDA.allowscalar(false)  # forbid scalar indexing

a = CuArray([1, 2, -3.0])
relu(x) = max(x, zero(x))
relu.(a)  # [1.0, 2.0, 0.0]

a[1]  # errors
```])

- One can also use `CUDA.zeros(T, dims...)` to initialize CuArray:

#box(text(16pt)[```julia
CUDA.zeros(Float32, 10, 10)
```])

== Kernel function (device code)

#box(text(16pt)[```julia
# Simple CUDA kernel example
function kernel_add!(y, x, a)
    # Get thread ID
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    # Bounds check
    if i <= length(y)
        @inbounds y[i] = a * x[i] + y[i]
    end
    
    return nothing
end
```])

== Host code

#box(text(16pt)[```julia
# Launch configuration
function saxpy!(y, x, a)
    # Calculate number of threads and blocks
    threads = 256
    blocks = cld(length(y), threads)
    
    # Launch kernel
    @cuda threads=threads blocks=blocks kernel_add!(y, x, a)
    
    return y
end
```])
- `@cuda` - Launch the kernel on the GPU
- `threads` - Number of threads per block
- `blocks` - Number of blocks


== The CUDA version of Einsum
https://gist.github.com/GiggleLiu/57c0fd7d6d775dbe6892a1b2efda40f8

== Final remark

- Do not use macros if not absolutely necessary
- Do not improve performance before profiling

== Hands-on

CUDA implementation of simulated annealing. Just open the demos repo and run the code.

Make sure you have GPU that supports CUDA.

#box(text(16pt)[```bash
$ nvidia-smi
```])

The code is in `Spinglass/examples/cuda.jl`

#box(text(16pt)[```bash
$ make update-Spinglass

$ julia --project=Spinglass/examples Spinglass/examples/cuda.jl
```])