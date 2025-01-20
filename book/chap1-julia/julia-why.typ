#import "../book.typ": book-page, cross-link
#show: book-page.with(title: "Why Julia?")

= Why Julia?

== Introduction to Julia
Julia is a modern, high-performance programming language designed specifically for technical computing. Created at MIT in 2012 and now maintained by JuliaHub Inc., Julia combines the ease of use of Python with the speed of C/C++.

=== Key Advantages
Julia stands out from other programming languages in several important ways:

1. *Open Source*: Unlike MatLab, Julia is completely open source. The source code is maintained on #link("https://github.com/JuliaLang/julia")[GitHub], and its packages are available on #link("https://juliahub.com/ui/Packages")[JuliaHub].

2. *High Performance*: Unlike Python, Julia was designed from the ground up for high performance (#link("https://arxiv.org/abs/1209.5145")[arXiv:1209.5145]). It achieves C-like speeds while maintaining the simplicity of a dynamic language.

3. *Easy to Use*: Unlike C/C++ or Fortran, Julia offers a clean, readable syntax and interactive development environment. Its just-in-time (JIT) compilation provides platform independence while maintaining high performance.

#figure(
  image("images/benchmark.png", width: 400pt),
  caption: [Benchmark comparison of various programming languages normalized to C/C++]
)

=== Growing Adoption
Many prominent scientists and engineers have switched to Julia:

- *Steven G. Johnson*: Creator of #link("http://www.fftw.org/")[FFTW], transitioned from C++
- *Anders Sandvik*: Developer of the Stochastic Series Expansion quantum Monte Carlo method, moved from Fortran (#link("https://physics.bu.edu/~py502/")[Computational Physics course])
- *Miles Stoudenmire*: Creator of #link("https://itensor.org/")[ITensor], switched from C++
- *Jutho Haegeman* and *Chris Rackauckas*: Leading researchers in quantum physics and differential equations

#block(
  fill: rgb("#e7f3fe"),
  inset: 8pt,
  radius: 4pt,
  [
    === Should I Switch to Julia?
    Consider switching to Julia if:
    - Your computations typically run for more than 10 minutes
    - Existing tools don't meet your specific needs
    - You value both performance and code readability
  ]
)

== Understanding Julia's Performance
Let's explore Julia's performance through a simple example: implementing a factorial function.

=== Prerequisites
First, install the required packages:
```julia
pkg> add BenchmarkTools, MethodAnalysis
```

=== A Simple Factorial Implementation
Here's a straightforward implementation of factorial in Julia:

```julia
julia> function jlfactorial(n)
           x = 1
           for i in 1:n
           x = x * i
       end
       return x
   end
jlfactorial (generic function with 1 method)
```

To accurately measure performance, we'll use the `@btime` macro from the `BenchmarkTools` package:
```julia
julia> using BenchmarkTools

julia> @btime jlfactorial(x) setup=(x=5)
2.208 ns (0 allocations: 0 bytes)
120
```

The result shows that computing factorial(5) takes only about 2.2 nanoseconds—approximately 7 CPU clock cycles (with a typical ~0.3ns clock cycle). This demonstrates Julia's impressive performance capabilities.

=== Comparing with C
To provide a fair comparison, let's benchmark against C, which is often considered the gold standard for performance. Julia's seamless C interoperability allows us to make accurate performance comparisons.

First, let's write a C implementation:
```c
// demo.c
#include <stddef.h>
int c_factorial(size_t n) {
    int s = 1;
    for (size_t i=1; i<=n; i++) {
        s *= i;
    }
    return s;
}
```

Compile the C code to a shared library:
```bash
gcc demo.c -fPIC -O3 -shared -o demo.so
```

Now we can call this C function from Julia using the `@ccall` macro (#link("https://docs.julialang.org/en/v1/manual/calling-c-and-fortran-code/")[documentation]):

```julia
julia> using Libdl

julia> c_factorial(x) = Libdl.@ccall "./demo.so".c_factorial(x::Csize_t)::Int

julia> @benchmark c_factorial(5)
BenchmarkTools.Trial: 10000 samples with 1000 evaluations.
 Range (min … max):  7.333 ns … 47.375 ns  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     7.458 ns              ┊ GC (median):    0.00%
 Time  (mean ± σ):   7.764 ns ±  1.620 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%
```

The C implementation takes about 7.3 nanoseconds—remarkably close to Julia's performance.

=== Comparing with Python
Let's examine how Python performs on the same task:

```python
def factorial(n):
    x = 1
    for i in range(1, n+1):
        x = x * i
    return x
```

Using IPython's timing utilities:
```ipython
In [7]: timeit factorial(5)
144 ns ± 0.379 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
```

At 144 nanoseconds, Python is:
- 20 times slower than the C implementation
- 70 times slower than the Julia implementation

However, Python offers an interesting trade-off. While slower, it handles arbitrary-precision integers automatically:
```julia
julia> typemax(Int)
9223372036854775807

julia> jlfactorial(100)
0  # Overflow!

# Meanwhile in Python:
In [8]: factorial(100)
93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000
```

=== Understanding the Performance Gap
Python's flexibility comes at a performance cost. The key differences are:

1. *Type System*: Python's dynamic typing means variables can change types at runtime. This requires storing type information with each value, typically as a `(type, *data)` tuple.

2. *Memory Access*: Python's approach leads to scattered memory access patterns, violating data locality principles. This causes frequent cache misses, forcing slower main memory access.

#figure(
  image("images/data.png", width: 150pt),
  caption: "Impact of data locality on performance"
)

== Understanding Julia's Type System and Multiple Dispatch

=== The Power of Multiple Dispatch
Multiple dispatch is a fundamental feature of Julia that allows functions to be dynamically dispatched based on the runtime types of all their arguments. This is in contrast to single dispatch in object-oriented languages, where method selection is based only on the first argument (the object).

Let's explore this concept through a practical example:

```julia
# Define an abstract type for animals
abstract type AbstractAnimal{L} end
```

The type parameter `L` represents the number of legs. While we could store this as a field, making it a type parameter enables compile-time optimizations.

Now, let's define some concrete animal types:

```julia
# Define concrete types for different animals
struct Dog <: AbstractAnimal{4}
    color::String
end

struct Cat <: AbstractAnimal{4}
    color::String
end

struct Cock <: AbstractAnimal{2}
    gender::Bool
end

struct Human{FT <: Real} <: AbstractAnimal{2}
    height::FT
    function Human(height::T) where T <: Real
        if height <= 0 || height > 300
            error("Human height must be between 0 and 300 cm")
        end
        return new{T}(height)
    end
end
```

Notice how `Human` includes a custom constructor with validation. The `<:` operator indicates a subtype relationship, so `Dog <: AbstractAnimal{4}` means "Dog is a subtype of AbstractAnimal with 4 legs."

=== Implementing Multiple Dispatch
Let's implement a `fight` function to demonstrate multiple dispatch:

```julia
# Default fallback method
fight(a::AbstractAnimal, b::AbstractAnimal) = "draw"

# Specific methods for different combinations
fight(dog::Dog, cat::Cat) = "win"
fight(hum::Human, a::AbstractAnimal) = "win"
fight(hum::Human, a::Union{Dog, Cat}) = "loss"
fight(hum::AbstractAnimal, a::Human) = "loss"
```

Here, `Union{Dog, Cat}` represents a type that can be either `Dog` or `Cat`. This demonstrates how Julia's type system allows for precise method definitions.

However, this implementation has an ambiguity:

```julia
fight(Human(170), Human(180))
ERROR: MethodError: fight(::Human{Int64}, ::Human{Int64}) is ambiguous...
```

The error occurs because two methods could apply: humans win against all animals, but also lose to humans. We can resolve this by adding a specific method for human-vs-human encounters:

```julia
# Resolve ambiguity with a specific method
fight(hum1::Human{T}, hum2::Human{T}) where T<:Real = 
    hum1.height > hum2.height ? "win" : "loss"

# Now we can test various combinations
julia> fight(Cock(true), Cat("red"))
"draw"

julia> fight(Dog("blue"), Cat("white"))
"win"

julia> fight(Human(180), Cat("white"))
"win"

julia> fight(Human(170), Human(180))
"loss"
```

=== Method Instances and Runtime Optimization
Julia creates optimized method instances for each unique combination of argument types:

```julia
using MethodAnalysis
methodinstances(fight)
```

Each method instance represents a specialized version of the function, compiled for specific argument types. This compilation strategy allows Julia to achieve C-like performance while maintaining the flexibility of a dynamic language.

== The Julia Number System: A Case Study in Type Design

=== Type Hierarchy
Julia's type system is organized as a tree, with `Any` as the root type. The `Number` type, a direct subtype of `Any`, serves as the foundation for Julia's entire number system:

```julia
julia> Number <: Any
true
```

The number system's hierarchy is carefully designed to balance flexibility and performance:

```
Number
├─ Complex{T}
├─ Real
│  ├─ AbstractFloat
│  │  ├─ BigFloat
│  │  ├─ Float16
│  │  ├─ Float32
│  │  └─ Float64
│  ├─ AbstractIrrational
│  └─ Integer
      ├─ Bool
      ├─ Signed
      │  ├─ Int8, Int16, Int32, Int64, Int128
      └─ Unsigned
          ├─ UInt8, UInt16, UInt32, UInt64, UInt128
```

Julia provides utilities to explore this type hierarchy:

```julia
julia> using InteractiveUtils

julia> subtypes(Number)
3-element Vector{Any}:
 Base.MultiplicativeInverses.MultiplicativeInverse
 Complex
 Real

julia> supertype(Float64)
AbstractFloat

julia> AbstractFloat <: Real
true
```

=== Primitive Types vs Composite Types
Julia's number system consists of two kinds of types:

1. *Primitive Types*: Built-in types with fixed bit sizes, such as:
```julia
# Floating-point types
primitive type Float16 <: AbstractFloat 16 end
primitive type Float32 <: AbstractFloat 32 end
primitive type Float64 <: AbstractFloat 64 end

# Integer types
primitive type Int64   <: Signed   64 end
primitive type UInt64  <: Unsigned 64 end
```

2. *Composite Types*: Built from other types, like `Complex{T}`:
```julia
struct Complex{T<:Real} <: Number
    re::T
    im::T
end
```

=== Extending the Number System: Multiple Dispatch vs Object-Oriented Approach

Let's compare how Julia and Python handle extending the number system. First, the Python approach:

```python
class X:
    def __init__(self, num):
        self.num = num

    def __add__(self, other):
        return X(self.num + other.num)

    def __radd__(self, other):
        return X(other.num + self.num)

class Y:
    def __init__(self, num):
        self.num = num

    def __add__(self, other):
        return Y(self.num + other.num)
```

This object-oriented approach has limitations:
- Method resolution depends only on the left operand
- Complex interactions between types require careful method ordering
- Adding new types requires modifying existing classes

Now, the Julia approach:

```julia
# Define new number types
struct X{T} <: Number
    num::T
end

struct Y{T} <: Number
    num::T
end

# Define addition operations
Base.:(+)(a::X, b::Y) = X(a.num + b.num)
Base.:(+)(a::Y, b::X) = Y(a.num + b.num)
Base.:(+)(a::X, b::X) = X(a.num + b.num)
Base.:(+)(a::Y, b::Y) = Y(a.num + b.num)
```

Julia's multiple dispatch approach offers several advantages:
- Operations can be defined symmetrically
- New types can be added without modifying existing code
- Type combinations have explicit, clear behavior

=== Compile-Time Computation: A Type System Example

While Julia excels at runtime performance, its type system can also enable compile-time computation. Here's an example using the Fibonacci sequence:

```julia
# Runtime implementation
fib(n::Int) = n <= 2 ? 1 : fib(n-1) + fib(n-2)

julia> @btime fib(40)
  278.066 ms (0 allocations: 0 bytes)
102334155
```

We can leverage Julia's type system to compute Fibonacci numbers at compile time:

```julia
# Compile-time implementation using Val types
fib(::Val{x}) where x = x <= 2 ? Val(1) : addup(fib(Val(x-1)), fib(Val(x-2)))
addup(::Val{x}, ::Val{y}) where {x, y} = Val(x + y)

julia> @btime fib(Val(40))
  0.792 ns (0 allocations: 0 bytes)
Val{102334155}()
```

While this compile-time approach is impressive, it's important to note:
1. It transfers computation cost to compile time
2. It can increase code size
3. It's not recommended for general use
4. It serves as an interesting demonstration of type system capabilities

=== Summary
Julia's type system and multiple dispatch provide:
- A flexible and extensible number system
- Clear and symmetric operation definitions
- Powerful compile-time capabilities
- Superior expressiveness compared to traditional object-oriented approaches