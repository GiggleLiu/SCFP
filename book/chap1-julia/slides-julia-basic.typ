#import "@preview/touying:0.4.2": *
#import "@preview/touying-simpl-hkustgz:0.1.0" as hkustgz-theme
#import "@preview/cetz:0.2.2": *
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

#let m = hkustgz-theme.register()

#show raw.where(block: true): it=>{
  block(radius:4pt, fill:gray.transparentize(90%), inset:1em, width:99%, text(it))
}

// Global information configuration
#let m = (m.methods.info)(
  self: m,
  title: [Julia: A Modern and Efficient Programming Language],
  subtitle: [],
  author: [Jin-Guo Liu],
  date: datetime.today(),
  institution: [HKUST(GZ) - FUNH - Advanced Materials Thrust],
)

// Extract methods
#let (init, slides) = utils.methods(m)
#show: init

// Extract slide functions
#let (slide, empty-slide, title-slide, outline-slide, new-section-slide, ending-slide) = utils.slides(m)
#show: slides.with()

#outline-slide()

= Introduction
== Julia: A Modern and Efficient Programming Language

#link("https://julialang.org/")[Julia] is a modern, high-performance programming language designed for technical computing. Created at MIT in 2012 and now maintained by JuliaHub Inc., Julia combines the ease of use of Python with the speed of C/C++.

1. *Open Source*: Unlike MatLab, Julia is completely open source. The source code is maintained on #link("https://github.com/JuliaLang/julia")[GitHub], and its packages are available on #link("https://juliahub.com/ui/Packages")[JuliaHub].

2. *High Performance*: Unlike Python, Julia was designed from the ground up for high performance (#link("https://arxiv.org/abs/1209.5145")[arXiv:1209.5145]). It achieves C-like speeds while maintaining the simplicity of a dynamic language.

3. *Easy to Use*: Unlike C/C++ or Fortran, Julia offers a clean, readable syntax and interactive development environment. Its just-in-time (JIT) compilation provides platform independence while maintaining high performance.


== Growing Adoption
Many prominent scientists and engineers have switched to Julia:

- *Steven G. Johnson*: Creator of #link("http://www.fftw.org/")[FFTW], transitioned from C++
- *Anders Sandvik*: Developer of the Stochastic Series Expansion quantum Monte Carlo method, moved from Fortran (#link("https://physics.bu.edu/~py502/")[Computational Physics course])
- *Miles Stoudenmire*: Creator of #link("https://itensor.org/")[ITensor], switched from C++
- *Jutho Haegeman* and *Chris Rackauckas*: Leading researchers in quantum physics and differential equations

== Julia is fast
Julia is a just-in-time (JIT) compiled language. It means that the code is compiled to machine code at runtime.
It is as concise (concise $!=$ simple) as Python, but runs much faster!
#figure(
  image("images/benchmark.png", width: 400pt),
)

https://benchmarksgame-team.pages.debian.net/benchmarksgame/fastest/julia-ifx.html

== Just-In-Time (JIT) Compilation
The more you tell the compiler, the more efficient code it can generate.
- The _type_ of a variable is known at compile time. 
- The _value_ of a variable can only be determined at runtime.

#figure(canvas(length: 1.8cm, {
  import draw: *
  content((-4, 0), box(inset: 10pt, radius: 4pt, stroke: black)[program], name: "program")
  content((-0.5, 0), box(inset: 10pt, radius: 4pt, stroke: black)[intermediate\
  representation], name: "intermediate")
  content((4, 0), box(inset: 10pt, radius: 4pt, stroke: black)[binary], name: "binary")
  line("program.east", "intermediate.west", mark: (end: "straight"))
  line("intermediate.east", "binary.west", mark: (end: "straight"), name: "binary-intermediate")
  bezier((rel: (-0.5, 0), to: "intermediate.north"), (rel: (0.5, 0), to: "intermediate.north"), (rel: (-1, 1), to: "intermediate.north"), (rel: (1, 1), to: "intermediate.north"), mark: (end: "straight"))
  content((rel: (0, 1.2), to: "intermediate.north"), [_types_, _constants_, ...])
  content((rel: (0, -0.2), to: "binary-intermediate.mid"), [compile])
  content((rel: (0, 1.5), to: "binary"), box(inset: 5pt)[Input Values], name: "binary-inputs")
  line("binary-inputs", "binary", mark: (end: "stealth"), stroke: 2pt)
  content((rel: (0, -1.5), to: "binary"), box(inset: 5pt)[Output Values], name: "binary-outputs")
  line("binary-outputs", "binary", mark: (start: "stealth"), stroke: 2pt)
})
)

= Julia Type System

== Julia type hierarchy
```
Any      # supertype of all types
├─ Number
│  ├─ Complex{T<:Real}
│  ├─ Real
│  │  ├─ AbstractFloat
│  │  │  ├─ BigFloat
│  │  │  ├─ Float16
│  │  │  ├─ Float32
│  │  │  └─ Float64
│  │  ├─ AbstractIrrational
...
```

== Subtyping operator "`<:`"
#figure(canvas(length: 1.55cm, {
  import draw: *
  circle((0, 0), radius: (4, 3), name: "Any")
  content((-4.5, 0), [Any])
  circle((0.5, 0), radius: (3.5, 2.5), name: "Number")
  content((0.5, 2), [Number])
  circle((2.1, 0), radius: (1.5, 2), name: "Complex")
  content((2.1, 0), [Complex])
  circle((-1.1, 0), radius: (1.5, 2), name: "Real")
  content((-1.1, -1), [Real])
  circle((-1.1, 0.5), radius: (1.3, 1), name: "AbstractFloat")
  content((-1.2, 0.5), [AbstractFloat])
  circle((-0.7, 1), radius: 0.1, name: "Float32", stroke: red)
  circle((-1.5, 1), radius: 0.1, name: "Float64", stroke: red)
  circle((2, 0.5), radius: 0.1, name: "Complex{Float64}", stroke: red)
  content((-1, 3.5), box(inset: 5pt)[Float32], name: "label-Float32")
  line("label-Float32", "Float32", mark: (end: "straight"))

  content((-3, 3.5), box(inset: 5pt)[Float64], name: "label-Float64")
  line("label-Float64", "Float64", mark: (end: "straight"))

  content((3, 3.5), box(inset: 5pt)[Complex{Float64}], name: "label-Complex{Float64}")
  line("label-Complex{Float64}", "Complex{Float64}", mark: (end: "straight"))
  content((8, 0), box(width: 300pt)[
    ```
    Number <: Any
    Complex <: Number
    Real <: Number
    AbstractFloat <: Real

    Float32 <: AbstractFloat
    Float64 <: AbstractFloat
    Complex{Float64} <: Complex
    ```
  ])
})) <fig:type-system>


== Play with types
#box(text(14pt)[```julia
julia> Number <: Any
true
julia> Complex{Float64} <: Complex <: Number
true
julia> Complex{Float64} <: Union{Real, Complex}
true

julia> isabstracttype(Number)  # for deriving new types
true
julia> isconcretetype(Float64) # has fixed memory layout
true
julia> subtypes(Number)
3-element Vector{Any}:
 Base.MultiplicativeInverses.MultiplicativeInverse
 Complex
 Real
julia> supertype(Float64)
AbstractFloat
```
])

== Parameterized types

```julia
struct Complex{T<:Real} <: Number
    re::T
    im::T
end
```
- _type name_: `Complex`
- _type parameters_: `T`
- _type constraint_: `T <: Real`

== Concrete types have fixed memory layout
#box(text(14pt)[```julia
julia> 1.0 + 2im
1.0 + 2.0im

julia> typeof(1.0 + 2im)  # obtain the type of 1.0 + 2im
Complex{Float64}

julia> sizeof(1.0 + 2im)  # memory size in bytes
16

julia> sizeof(Complex{Float64})
16

julia> sizeof(Complex)
ERROR: Argument is an incomplete Complex type and does not have a definite size.

julia> isconcretetype(Complex)
false
```
])

== Experiment: Vector of concrete types

```julia
using BenchmarkTools

x = randn(1000)
typeof(x)
@btime sin.(x)

y = Vector{Real}(randn(1000))
typeof(y)
@btime sin.(y)
```

== Dynamic typing causes cache misses

#figure(scale(150%, text(10pt, canvas(length: 1cm, {
  import draw: *
  let dy = -0.7
  content((-0.5, 0.5), [Stack])
  for j in range(5){
    rect((-2.2, dy * j), (2.0, (j+1) * dy), name: "t" + str(j))
  }
  content((0.5, 1.5 * dy), align(left, box(width:150pt, inset: 3pt, [`(Int, 0x11ff3323)`])))
  content((0.5, 2.5*dy), align(left, box(width: 150pt, inset: 3pt, [`(Float32, 0x11ff3323)`])))
  content((0.5, 3.5*dy), align(left, box(width: 150pt, inset: 3pt, [$dots.v$])))
  content((0.5, 4.5*dy), align(left, box(width: 150pt, inset: 3pt, [`(Int, 0x11ff3322)`])))

  content((5, 0.5), [Memory])
  for j in range(5){
    rect((4, dy * j), (6, (j+1) * dy), name: "a" + str(j))
  }
  content((5, 0.5 * dy), [233])
  content((5, 1.5*dy), [0.4])
  content((5, 2.5*dy), [$dots.v$])
  content((5, 3.5*dy), [9])
  content((5, 4.5*dy), [0.33])

  content((8, dy/2), align(left, box(width:100pt, inset: 3pt, [`0x11ff3322`])))
  content((8, 1.5*dy), align(left, box(width: 100pt, inset: 3pt, [`0x11ff3323`])))
  content((8.5, 2.5*dy), align(left, box(width: 100pt, inset: 3pt, [$dots.v$])))
  content((8, 3.5*dy), align(left, box(width: 100pt, inset: 3pt, [`0x2ef36511`])))
  content((8.5, 4.5*dy), align(left, box(width: 100pt, inset: 3pt, [$dots.v$])))

  line("t1.east", "a3.west", mark: (end: "straight"))
  line("t2.east", "a1.west", mark: (end: "straight"))
  line("t4.east", "a0.west", mark: (end: "straight"))

  line((-3, 0), (-3, -3.5), mark: (end: "straight"))
  content((-4, -1), [visit order])
})))
)



// == Primitive types and composite types
// - _primitive types_: those directly supported by the instruction set architecture.
// - _composite types_: those built from other types.

// === Examples of primitive types
// ```julia
// primitive type Float16 <: AbstractFloat 16 end  # `<:` means subtype
// primitive type Float32 <: AbstractFloat 32 end
// primitive type Float64 <: AbstractFloat 64 end
// ```

// === Examples of composite types
// - The `Complex{T <: Real}` type.

== Abstract Types
Users can define their own _abstract types_ with the `abstract type` keyword. e.g.
```julia
abstract type AbstractTropical{T<:Real} end
```

- cannot be instantiated in memory.
- can derive new types, which is not the case for the concrete types.

== Union of types
```julia
julia> 1.0 isa Float64  # 1.0 is an instance of Float64
true

julia> 1.0 isa Real
true

julia> 1.0 isa Union{Real, Complex}
true

julia> 1.0+2im isa Union{Real, Complex}
true
```

== Understanding Julia's JIT
In the following, we demonstrate the power of Julia's JIT by implementing a factorial function:

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

The result shows that computing factorial(5) takes only about 2.2 nanoseconds—approximately 7 CPU clock cycles (with a typical ~0.3ns clock cycle).

Q: We emphasized that we did not specify the variable type of `n` in the function definition. Then how does the JIT compiler know the types of variables?

== JIT happens when the function is first called

#figure(scale(150%, text(10pt, canvas({
  import draw: *
  content((-3, 0), box(inset: 3pt)[Inputs], name: "inputs")
  content((0, 0), [#box(stroke: black, inset: 10pt, [Call a function], radius: 4pt)], name: "call")
  content((6.5, 0), [#box(stroke: black, inset: 10pt, [Invoke], radius: 4pt)], name: "invoke")
  content((3.5, -2), [#box(stroke: black, inset: 10pt, [JIT Compilation], radius: 4pt)], name: "inference")
  line("inputs", "call.west", mark: (end: "straight"))
  line("call.south", (rel: (0, -1.5)), "inference.west", mark: (end: "straight"))
  line("inference.east", (rel: (0, -2), to: "invoke"), "invoke.south", mark: (end: "straight"))
  line("call.east", "invoke.west", mark: (end: "straight"))
  content((8.8, 0), box(inset: 3pt)[Outputs], name: "outputs")
  line("invoke.east", "outputs", mark: (end: "straight"))
  content((3.5, 0.5), text(green.darken(20%))[Has method instance])
  content((-2, -1.5), text(red.darken(20%))[No method instance])

  content((3.25, -1.1), [slow])
  content((6.5, 0.9), [fast])

  content((3.5, -3.0), [Typed IR $arrow.double.r$ LLVM IR $arrow.double.r$ Binary Code])
})))) <fig:jit>


== Step 1: Infer the types
Given a input type combination, can we infer the types of all variables in the function? It depends.
If all the types are inferred, the function is called *type stable*. Then the function can be compiled to efficient binary code. One can use the `@code_warntype` macro to check if the function is type stable. For example, the `jlfactorial` function with integer input is type stable:

```julia
julia> @code_warntype jlfactorial(10)
??
```

== Type stability
If not all types are inferred, the function is called *type unstable*. Then the function falls back to the _dynamic dispatch_ mode, which can be slow. For example, the following `badcode` function is type unstable:

```julia
julia> badcode(x) = x > 3 ? 1.0 : 3

julia> @code_warntype badcode(4)
??
```

== Type unstable code is slow
```julia
julia> x = rand(1:10, 1000);

julia> typeof(badcode.(x))  # non-concrete element type
Vector{Real} (alias for Array{Real, 1})

julia> @btime badcode.($x)
??
```

In the above example, the "`.`" is the broadcasting operator, it applies the function to each element of the array.

== Type stable code is fast
Instead, if we specify the function in a type stable way, the function can be compiled to efficient binary code:

```julia
julia> stable(x) = x > 3 ? 1.0 : 3.0
stable (generic function with 1 method)

julia> typeof(stable.(x))   # concrete element type
Vector{Float64} (alias for Array{Float64, 1})

julia> @btime stable.($x)
??
```
== Step 2: Generates the LLVM IR

With the typed intermediate representation (IR), the Julia compiler the generates the LLVM IR.
LLVM is a set of compiler and toolchain technologies that can be used to develop a front end for any programming language and a back end for any instruction set architecture. LLVM is the backend of multiple languages, including Julia, Rust, Swift and Kotlin.

In Julia, one can use the `@code_llvm` macro to show the LLVM intermediate representation of a function.

```julia
julia> @code_llvm jlfactorial(10)
??
```

== Step 3: Compiles to binary code

The LLVM IR is then compiled to binary code by the LLVM compiler. The binary code can be printed by the `@code_native` macro.

```julia
julia> @code_native jlfactorial(10)
??
```

== Experiment: analyze the method instances

The method instance is then stored in the method table, and can be analyzed by the `MethodAnalysis` package.

```julia
julia> using MethodAnalysis

julia> methodinstances(jlfactorial)
??

julia> jlfactorial(UInt32(5))
120

julia> methodinstances(jlfactorial)
??
```

== Experiment: Comparing with C and Python

To demonstrate the difference between compiled languages, interpreted languages and JIT compiled languages,

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

Now we can call this C function from Julia using the `@ccall` macro:

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

== Comparing with Python
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

As a remark, when Julia compiler fails to infer the types, it will fall back to the dynamic dispatch mode. Then it also suffers from the problem of cache misses.


= Multiple Dispatch
== The Power of Multiple Dispatch
Multiple dispatch is a fundamental feature of Julia that allows functions to be dynamically dispatched based on the runtime types of all their arguments. This is in contrast to single dispatch in object-oriented languages, where method selection is based only on the first argument (the object).

Let's explore this concept through a practical example:

```julia
# Define an abstract type for animals
abstract type AbstractAnimal end
```

==
Now, let's define some concrete animal types:

#box(text(10pt)[```julia
# Define concrete types for different animals
struct Dog <: AbstractAnimal
    color::String
end

struct Cat <: AbstractAnimal
    color::String
end

struct Cock <: AbstractAnimal
    gender::Bool
end

struct Human{FT <: Real} <: AbstractAnimal
    height::FT
    function Human(height::T) where T <: Real
        if height <= 0 || height > 300
            error("Human height must be between 0 and 300 cm")
        end
        return new{T}(height)
    end
end
```
])

Notice how `Human` includes a custom constructor with validation. The `<:` operator indicates a subtype relationship, so `Dog <: AbstractAnimal` means "Dog is a subtype of AbstractAnimal."

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

== Method Instances and Runtime Optimization
Julia creates optimized method instances for each unique combination of argument types:

```julia
using MethodAnalysis
methodinstances(fight)
```

Each method instance represents a specialized version of the function, compiled for specific argument types. This compilation strategy allows Julia to achieve C-like performance while maintaining the flexibility of a dynamic language.

== Experiment: Comparing with object oriented programming

The Python approach:

#box(text(14pt)[```python
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
])

==
The object-oriented approach has limitations:
- Method resolution depends primarily on the left operand
- Complex interactions between types require careful method ordering
- Adding new types requires modifying existing classes

== The Julia approach

#box(text(14pt)[```julia
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
])

==

Julia's multiple dispatch approach offers several advantages:
- Operations can be defined symmetrically
- New types can be added without modifying existing code
- Type combinations have explicit, clear behavior

== Experiment 2: Compile-Time Computation

While Julia excels at runtime performance, its type system can also enable compile-time computation. Here's an example using the Fibonacci sequence:

```julia
# Runtime implementation
fib(n::Int) = n <= 2 ? 1 : fib(n-1) + fib(n-2)

julia> @btime fib(40)
  278.066 ms (0 allocations: 0 bytes)
102334155
```

== A completely static implementation
We can leverage Julia's type system to compute Fibonacci numbers at compile time:

```julia
# Compile-time implementation using Val types
fib(::Val{x}) where x = x <= 2 ? Val(1) : addup(fib(Val(x-1)), fib(Val(x-2)))
addup(::Val{x}, ::Val{y}) where {x, y} = Val(x + y)

julia> @btime fib(Val(40))
  0.792 ns (0 allocations: 0 bytes)
Val{102334155}()
```

Q: What is happening here?