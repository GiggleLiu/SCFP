#import "../book.typ": book-page, cross-link
#import "@preview/cetz:0.2.2": *
#show: book-page.with(title: "Julia Basic")

= Julia REPL

Julia REPL is the command line interface for Julia. Once you install Julia, you can start it by typing `julia` in your terminal.
It offers four specialized modes:

1. *Julian Mode*: The default mode for executing Julia code.

2. *Shell Mode*: Provide a shell interface. Enter this mode by pressing `;` in Julian mode:
  ```julia
shell> date
Sun Nov  6 10:50:21 PM CST 2022
```

3. *Package Mode*: Provide a package interface. Enter this mode by pressing `]` in Julian mode:
  ```julia
(@v1.8) pkg> st
Status `~/.julia/environments/v1.8/Project.toml`
  [295af30f] Revise v3.4.0
```
  #image("images/Packages.gif")
  To install packages, just type `add <package name>`. For more commands, just type `?` and press `Enter`.
  The information about Julia packages is available at #link("https://juliahub.com/")[JuliaHub].

4. *Help Mode*: Provide a help interface. Enter this mode by pressing `?` in Julian mode:
  ```julia
help> sum
... docstring for sum ...
```

Return to Julian mode from any other mode by pressing `Backspace`. To exit the REPL, press `Ctrl-D`. To stop a running program, press `Ctrl-C`.

= Benchmarking and Profiling
We use Julia for high performance computing. Before we start to introduce how to program with Julia, we first introduce how to benchmark and profile your code.
In Julia, we can use #link("https://github.com/JuliaCI/BenchmarkTools.jl")[`BenchmarkTools`] to measure the performance of your code, which could be installed by `] add BenchmarkTools` in a Julia REPL. It provides two macros `@btime` and `@benchmark`. They run the function for multiple times and record the time, while `@benchmark` provides more detailed information.
```julia
julia> @btime sum($(randn(1000)))
  79.291 ns (0 allocations: 0 bytes)
-1.1417722596480882
```
The "`$`" is the interpolation operator, it is used to interpolate a variable into an expression. In a benchmark, it can be used to exclude the time to initialize the variable.

To find the bottleneck of your code, we need to use a profiler. Profilers are categorized into two types:
- _Event based profiler_, e.g. triggered by the function call event.
- _Statistical profiler_ (used by Julia), e.g. sample the _call stack_ of the running program every 1ms.

Julia provides a statistical profiler in its standard library. Statistical profiler has less overhead, and profiles small functions more accurately.
To use it, we first import the `Profile` module and initialize it (optional):

```julia
julia> using Profile

julia> Profile.init(; delay=0.001)  # How often to sample the call stack
```

Here, we set the delay to 1ms (the same as the default value), which means the profiler will sample the call stack every 1ms.
Then we can use the `@profile` macro to profile the code:

```julia
julia> @profile for _ = 1:10000 sum(randn(1000)) end
```

Since the running time of the code is short, we set the number of iterations to 10000 so that we can get enough snapshots.
After profiling, we can print the profile results:

```julia
julia> Profile.print(; mincount=10)
Overhead ╎ [+additional indent] Count File:Line; Function
=========================================================
  # ... omitted ...
  ╎ 40 @Profile/src/Profile.jl:59; macro expansion
  ╎  40 REPL[39]:1; macro expansion
  ╎   38 @Random/src/normal.jl:278; randn
  ╎    38 @Random/src/normal.jl:272; randn
  ╎     11 @Base/boot.jl:596; Array
  ╎    ╎ 11 @Base/boot.jl:578; Array
11╎    ╎  11 @Base/boot.jl:516; GenericMemory
  ╎     19 @Random/src/normal.jl:260; randn!(rng::Random.TaskLocalRNG, A::Vector{Float64})
Total snapshots: 40. Utilization: 100% across all threads and tasks. Use the `groupby` kwarg to break down by thread and/or task.

julia> Profile.clear()   # Clear the profile results
```
Since the profiling results are quite messy, we only print those with at least 10 snapshots to filter out the function calls accounting for less than 10ms of running time.
The profiling results will not be cleared automatically, so we need to clear it manually. Otherwise, it will be accumulated.


= Just-In-Time (JIT) Compilation

Julia is a just-in-time compiled language. It means that the code is compiled to binary only when needed, which is different from both the static compilation languages like C/C++ and the interpreted languages like Python.
The more information you tell the compiler, the more efficient code it can generate.
Unlike languages without type system, e.g. Python, in Julia, the _type_ of a variable is known at the compile time.
On the other side, the _value_ of a variable can only be determined at runtime, just like any other programming language.

#figure(canvas({
  import draw: *
  let s(it) = text(12pt, it)
  content((-4, 0), box(inset: 5pt, radius: 4pt, stroke: black, s[program]), name: "program")
  content((-0.5, 0), box(inset: 5pt, radius: 4pt, stroke: black, s[intermediate\
  representation]), name: "intermediate")
  content((4, 0), box(inset: 5pt, radius: 4pt, stroke: black, s[binary]), name: "binary")
  line("program.east", "intermediate.west", mark: (end: "straight"))
  line("intermediate.east", "binary.west", mark: (end: "straight"), name: "binary-intermediate")
  bezier((rel: (-0.5, 0), to: "intermediate.north"), (rel: (0.5, 0), to: "intermediate.north"), (rel: (-1, 1), to: "intermediate.north"), (rel: (1, 1), to: "intermediate.north"), mark: (end: "straight"))
  content((rel: (0, 1.2), to: "intermediate.north"), s[_types_, _constants_, ...])
  content((rel: (0, -0.2), to: "binary-intermediate.mid"), s[compile])
  content((rel: (0, 1.5), to: "binary"), box(inset: 5pt, s[Input Values]), name: "binary-inputs")
  line("binary-inputs", "binary", mark: (end: "stealth"), stroke: 2pt)
  content((rel: (0, -1.5), to: "binary"), box(inset: 5pt, s[Output Values]), name: "binary-outputs")
  line("binary-outputs", "binary", mark: (start: "stealth"), stroke: 2pt)
}),
  caption: [A typical compiling pipeline. The compiler makes use of static information (types, constants, etc.) to generate efficient binary code. The input values are only available at runtime.]
)

== Types

Types play a crucial role in the JIT compilation of Julia, which tells the compiler the memory layouts of data.
In Julia, a type is composed of two parts, the _type name_ and the _type parameters_. For example, the `Complex` type is defined as follows:

```julia
struct Complex{T<:Real} <: Number  # `<:` is the subtype operator
    re::T
    im::T
end
```
- _type name_: `Complex`
- _type parameters_: `T`
- _type constraint_: `T <: Real`

If a type has no parameters, the ${}$ is omitted. For example, `Int64` is a type with no parameters.
The operator `<:` is the subtype operator, which means that `Complex` is a subtype of `Number`. *Only* _abstract types_ can be subtyped. To define a new abstract type, we can use the `abstract type` keyword:
```julia
abstract type Number end
```

Abstract types do not have fields, and can not be instantiated in memory. Only _concrete types_ can be instantiated in memory. A type is concrete only if it has all type parameters specified. For example, `Complex` it alone is not concrete. Although the compiler knows `Complex` has a real part and an imaginary part, but it does not know the specific bit size of the real and imaginary parts.
The following example shows how to obtain the type of a value and the memory size of a value:
```julia
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

*_Concrete types_* can be further categorized into *_primitive types_* and *_composite types_*.
Primitive types are those directly supported by the instruction set architecture. A standard list of primitive types includes `Float16`, `Float32`, `Float64`, `Bool`, `Char`, `Int8`, `UInt8`, `Int16`, `UInt16`, `Int32`, `UInt32`, `Int64`, `UInt64`, `Int128`, and `UInt128`.

_Abstract types_ can be used to organize the type system as a type tree. As shown in @fig:type-system, `Any` type is the root, which is the set of everything.
Julia does not support multiple inheritance, so the type tree has a _tree_ topology rather than a _graph_ topology.
In case that one needs a type of both `A` and `B`, one can use the `Union` type:
```julia
julia> const FloatOrComplex = Union{AbstractFloat, Complex{<:AbstractFloat}}
Union{Complex{Float16}, Complex{Float32}, Complex{Float64}}

julia> 1.0 isa FloatOrComplex  # `isa` is the membership operator
true

julia> 1.0 + 2im isa FloatOrComplex
true
```
However, the `Union` type can not be inherited. Actually, the multiple inheritance is usually not a good practice, and is completely forbidden in Julia.

#figure(canvas({
  import draw: *
  let s(it) = text(11pt, it)
  circle((0, 0), radius: (4, 3), name: "Any")
  content((-4.5, 0), s[Any])
  circle((0.5, 0), radius: (3.5, 2.5), name: "Number")
  content((0.5, 2), s[Number])
  circle((2.1, 0), radius: (1.5, 2), name: "Complex")
  content((2.1, 0), s[Complex])
  circle((-1.1, 0), radius: (1.5, 2), name: "Real")
  content((-1.1, -1), s[Real])
  circle((-1.1, 0.5), radius: (1.3, 1), name: "AbstractFloat")
  content((-1.1, 0.5), s[AbstractFloat])
  circle((-0.7, 1), radius: 0.1, name: "Float32", fill: red, stroke: red)
  circle((-1.5, 1), radius: 0.1, name: "Float64", fill: red, stroke: red)
  circle((2, 0.5), radius: 0.1, name: "Complex{Float64}", fill: red, stroke: red)
  content((-1, 3.5), box(inset: 5pt, s[Float32]), name: "label-Float32")
  line("label-Float32", "Float32", mark: (end: "straight"))

  content((-3, 3.5), box(inset: 5pt, s[Float64]), name: "label-Float64")
  line("label-Float64", "Float64", mark: (end: "straight"))

  content((3, 3.5), box(inset: 5pt, s[Complex{Float64}]), name: "label-Complex{Float64}")
  line("label-Complex{Float64}", "Complex{Float64}", mark: (end: "straight"))
  content((7, 0), box(width: 150pt, s[
  - types $arrow.r$ _sets_.
  - `A <: B` $arrow.r$ $A ⊆ B$.
  - `Union{A, B}` $arrow.r$ $A ∪ B$.
  - `a isa A` $arrow.r$ $a ∈ A$.
  ]))
}), caption: [Julia's type system in set theory. The black circles represent abstract types, and red circles represent concrete types.]) <fig:type-system>

```julia
julia> Number <: Any
true

julia> Complex{Float64} <: Complex <: Number
true

julia> Complex{Float64} <: Union{Real, Complex}
true

julia> isabstracttype(Number)
true
julia> subtypes(Number)
3-element Vector{Any}:
 Base.MultiplicativeInverses.MultiplicativeInverse
 Complex
 Real

julia> supertype(Float64)
AbstractFloat
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

The result shows that computing factorial(5) takes only about 2.2 nanoseconds—approximately 7 CPU clock cycles (with a typical ~0.3ns clock cycle). This demonstrates Julia's impressive performance capabilities.

We emphasized that we did not specify the variable type of `n` in the function definition. Then how does the JIT compiler know the types of variables? It turns out that the JIT of Julia happens when the function is *first called* with a specific input types combination.

#figure(canvas({
  import draw: *
  let s(it) = text(11pt, it)
  content((-3, 0), box(inset: 3pt, s[Inputs]), name: "inputs")
  content((0, 0), s[#box(stroke: black, inset: 10pt, s[Call a function], radius: 4pt)], name: "call")
  content((6.5, 0), s[#box(stroke: black, inset: 10pt, s[Invoke], radius: 4pt)], name: "invoke")
  content((3.5, -2), s[#box(stroke: black, inset: 10pt, [JIT Compilation], radius: 4pt)], name: "inference")
  line("inputs", "call.west", mark: (end: "straight"))
  line("call.south", (rel: (0, -1.5)), "inference.west", mark: (end: "straight"))
  line("inference.east", (rel: (0, -2), to: "invoke"), "invoke.south", mark: (end: "straight"))
  line("call.east", "invoke.west", mark: (end: "straight"))
  content((8.8, 0), box(inset: 3pt, s[Outputs]), name: "outputs")
  line("invoke.east", "outputs", mark: (end: "straight"))
  content((3.5, 0.5), text(green.darken(20%), 10pt)[Has method instance])
  content((-2, -1.5), text(red.darken(20%), 10pt)[No method instance])

  content((3.25, -1.1), text(10pt)[slow])
  content((6.5, 0.9), text(10pt)[fast])

  content((3.5, -3.0), s[Typed IR $arrow.double.r$ LLVM IR $arrow.double.r$ Binary Code])
})) <fig:jit>

When calling a Julia function with a specific input, Julia checks if the _method instance_ that associated with the input types has been compiled. If yes, the cached method instance will be invoked.
Otherwise, it will compile the method instance and store it in the method table, and then invoke it.
Depnding on how many different input types combinations the function has been called with, there may be multiple method instances for the same function.
Repeated compilation should be avoided since it can be slow.
We wish the compiled method instance can be reused multiple times, so that the JIT overhead is amortized.
//In some extreme case, the number of input types combinations is too large, causing large JIT compilation time. This is usually caused by the abused use of type system and should be avoided.



=== Step 1: Infer the types
Given a input type combination, can we infer the types of all variables in the function? It depends.
If all the types are inferred, the function is called *type stable*. Then the function can be compiled to efficient binary code. One can use the `@code_warntype` macro to check if the function is type stable. For example, the `jlfactorial` function with integer input is type stable:

```julia
julia> @code_warntype jlfactorial(10)
MethodInstance for jlfactorial(::Int64)
  from jlfactorial(n) @ Main REPL[4]:1
Arguments
  #self#::Core.Const(jlfactorial)
  n::Int64
Locals
  @_3::Union{Nothing, Tuple{Int64, Int64}}
  x::Int64
  i::Int64
Body::Int64
1 ─       (x = 1)
│   %2  = (1:n)::Core.PartialStruct(UnitRange{Int64}, Any[Core.Const(1), Int64])
│         (@_3 = Base.iterate(%2))
│   %4  = (@_3 === nothing)::Bool
│   %5  = Base.not_int(%4)::Bool
└──       goto #4 if not %5
2 ┄ %7  = @_3::Tuple{Int64, Int64}
│         (i = Core.getfield(%7, 1))
│   %9  = Core.getfield(%7, 2)::Int64
│         (x = x * i)
│         (@_3 = Base.iterate(%2, %9))
│   %12 = (@_3 === nothing)::Bool
│   %13 = Base.not_int(%12)::Bool
└──       goto #4 if not %13
3 ─       goto #2
4 ┄       return x
```

We can see that the types of almost all variables are inferred except some warnings.
If not all types are inferred, the function is called *type unstable*. Then the function falls back to the _dynamic dispatch_ mode, which can be slow. For example, the following `badcode` function is type unstable:

```julia
julia> badcode(x) = x > 3 ? 1.0 : 3

julia> @code_warntype badcode(4)
MethodInstance for badcode(::Int64)
  from badcode(x) @ Main REPL[9]:1
Arguments
  #self#::Core.Const(badcode)
  x::Int64
Body::Union{Float64, Int64}
1 ─ %1 = (x > 3)::Bool
└──      goto #3 if not %1
2 ─      return 1.0
3 ─      return 3
```
In this example, the output type `Union{Float64, Int64}` means the return type is either `Float64` or `Int64`. The function is type unstable because the return type is not fixed.
Type unstable code is slow. In the following example, the `badcode` function is ~10 times slower than its type stable version `stable`:

```julia
julia> x = rand(1:10, 1000);

julia> typeof(badcode.(x))  # non-concrete element type
Vector{Real} (alias for Array{Real, 1})

julia> @btime badcode.($x);
  7.091 μs (696 allocations: 26.53 KiB)
```

In the above example, the "`.`" is the broadcasting operator, it applies the function to each element of the array. "`$`" is the interpolation operator, it is used to interpolate a variable into an expression. In a benchmark, it can be used to exclude the time to initialize the variable.


Instead, if we specify the function in a type stable way, the function can be compiled to efficient binary code:

```julia
julia> stable(x) = x > 3 ? 1.0 : 3.0
stable (generic function with 1 method)

julia> typeof(stable.(x))   # concrete element type
Vector{Float64} (alias for Array{Float64, 1})

julia> @btime stable.($x);
  500.644 ns (3 allocations: 7.88 KiB)
```
=== Step 2: Generates the LLVM IR

With the typed intermediate representation (IR), the Julia compiler generates the LLVM IR.
#link("https://llvm.org/")[LLVM] is a compiler infrastructure that can be used to compile programs to different instruction set architectures. It is the backend of multiple languages, including Julia, Rust, Swift and Kotlin.

In Julia, one can use the `@code_llvm` macro to show the LLVM intermediate representation of a function.

```julia
julia> @code_llvm jlfactorial(10)
```

=== Step 3: Compiles to binary code

The LLVM IR is then compiled to binary code by the LLVM compiler. The binary code can be printed by the `@code_native` macro.

```julia
julia> @code_native jlfactorial(10)
```

The method instance is then stored in the method table, and can be analyzed by the `MethodAnalysis` package.

```julia
julia> using MethodAnalysis

julia> jlfactorial(10)
120

julia> methodinstances(jlfactorial)
1-element Vector{Core.MethodInstance}:
 MethodInstance for jlfactorial(::Int64)

julia> jlfactorial(UInt32(5))
120

julia> methodinstances(jlfactorial)
2-element Vector{Core.MethodInstance}:
 MethodInstance for jlfactorial(::Int64)
 MethodInstance for jlfactorial(::UInt32)
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

julia> @btime c_factorial(5);
  7.333 ns (0 allocations: 0 bytes)
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

This is because python's flexibility comes at a performance cost. Python's dynamic typing means variables can change types at runtime. This requires storing type information with each value, typically as a `(type, *data)` tuple. This leads to scattered memory access patterns, causing frequent cache misses.

#figure(canvas(length: 1cm, {
  import draw: *
  let dy = -0.7
  let s(it) = text(11pt, it)
  content((-0.5, 0.5), s[Stack])
  for j in range(5){
    rect((-2.2, dy * j), (2.4, (j+1) * dy), name: "t" + str(j))
  }
  content((0.5, 1.5 * dy), align(left, box(width:150pt, inset: 3pt, s[`(Int, 0x11ff3323)`])))
  content((0.5, 2.5*dy), align(left, box(width: 150pt, inset: 3pt, s[`(Float64, 0x11ff3323)`])))
  content((0.5, 3.5*dy), align(left, box(width: 150pt, inset: 3pt, s[$dots.v$])))
  content((0.5, 4.5*dy), align(left, box(width: 150pt, inset: 3pt, s[`(Int, 0x11ff3322)`])))

  content((5, 0.5), s[Memory])
  for j in range(5){
    rect((4, dy * j), (6, (j+1) * dy), name: "a" + str(j))
  }
  content((5, 0.5 * dy), s[233])
  content((5, 1.5*dy), s[0.4])
  content((5, 2.5*dy), s[$dots.v$])
  content((5, 3.5*dy), s[9])
  content((5, 4.5*dy), s[0.33])

  content((8, dy/2), align(left, box(width:100pt, inset: 3pt, s[`0x11ff3322`])))
  content((8, 1.5*dy), align(left, box(width: 100pt, inset: 3pt, s[`0x11ff3323`])))
  content((8.5, 2.5*dy), align(left, box(width: 100pt, inset: 3pt, s[$dots.v$])))
  content((8, 3.5*dy), align(left, box(width: 100pt, inset: 3pt, s[`0x2ef36511`])))
  content((8.5, 4.5*dy), align(left, box(width: 100pt, inset: 3pt, s[$dots.v$])))

  line("t1.east", "a3.west", mark: (end: "straight"))
  line("t2.east", "a1.west", mark: (end: "straight"))
  line("t4.east", "a0.west", mark: (end: "straight"))

  line((-3, 0), (-3, -3.5), mark: (end: "straight"))
  content((-4.2, -1), s[visit order])
}),
caption: "Dynamic typing causes cache misses"
)

As a remark, when Julia compiler fails to infer the types, it will fall back to the dynamic dispatch mode. Then it also suffers from the problem of cache misses.

= Multiple Dispatch

== The Power of Multiple Dispatch
Multiple dispatch is a fundamental feature of Julia that allows functions to be dynamically dispatched based on the runtime types of all their arguments. This is in contrast to single dispatch in object-oriented languages, where method selection is based only on the first argument (the object).
This feature gives users a superior abstraction power. For example, to define a function named `foo` that takes $k$ arguments, with $m$ possible types for each argument, the number of methods is $m^k$. While for object-oriented languages, the number of methods is only $m$, since each class can have one `foo` method. The superiority will finally reflected on the number of lines of code - Julia code can be super concise.

Let's explore this concept through a practical example:

```julia
# Define an abstract type for animals
abstract type AbstractAnimal end
```

Now, let's define some concrete animal types:

```julia
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

Notice how `Human` includes a custom constructor with validation. The `<:` operator indicates a subtype relationship, so `Dog <: AbstractAnimal` means "Dog is a subtype of AbstractAnimal."

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
julia> fight(hum1::Human{T}, hum2::Human{T}) where T<:Real = 
    hum1.height > hum2.height ? "win" : "loss"

julia> fight(Cock(true), Cat("red"))
"draw"

julia> fight(Dog("blue"), Cat("white"))
"win"

julia> fight(Human(180), Cat("white"))
"win"

julia> fight(Human(170), Human(180))
"loss"
```

Julia creates optimized method instances for each unique combination of argument types:

```julia
julia> methodinstances(fight)
4-element Vector{Core.MethodInstance}:
 MethodInstance for fight(::Dog, ::Cat)
 MethodInstance for fight(::Human{Int64}, ::Human{Int64})
 MethodInstance for fight(::Human{Int64}, ::Cat)
 MethodInstance for fight(::Cock, ::Cat)
```

Each method instance represents a specialized version of the function, compiled for specific argument types. This compilation strategy allows Julia to achieve C-like performance while maintaining the flexibility of a dynamic language.

== Experiment 1: Extending the Number System

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

== Experiment 2: Zero-cost computing?

People like Julia's type system, which is extremely powerful. It is so powerful that it can even enable compile-time computation. Here's an example using the Fibonacci sequence:

```julia
julia> fib(n::Int) = n <= 2 ? 1 : fib(n-1) + fib(n-2);

julia> @btime fib(40)
  278.066 ms (0 allocations: 0 bytes)
102334155
```

We can leverage Julia's type system to compute Fibonacci numbers at compile time:

```julia
julia> fib(::Val{x}) where x = x <= 2 ? Val(1) : addup(fib(Val(x-1)), fib(Val(x-2)));

julia> addup(::Val{x}, ::Val{y}) where {x, y} = Val(x + y);

julia> @btime fib(Val(40))
  0.792 ns (0 allocations: 0 bytes)
Val{102334155}()
```
Amazingly, the compile-time computation completes the computation in 0.792 ns in our benchmark.
BUT, the compile-time computation is not free. It is still a trade-off between the runtime and the compile time. For example, here, the compiler generates $40$ method instances for the `fib` function
```julia
julia> methodinstances(fib) |> length  # `|>` is the pipe operator for single-argument function
40
```

This is because the compiler needs to generate a method instance for each possible argument type, i.e. it creates a table!
In practice, it is not recommended to move everything to compile-time since it may pollute the type system and cause type explosion.