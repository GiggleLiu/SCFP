#import "../book.typ": book-page, cross-link
#import "@preview/cetz:0.2.2": *
#show: book-page.with(title: "Julia Basic")

= Julia Basic
== Julia REPL

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

4. *Help Mode*: Provide a help interface. Enter this mode by pressing `?` in Julian mode:
```julia
help> sum
... docstring for sum ...
```

Return to Julian mode from any other mode by pressing `Backspace`.

== Just-In-Time Compilation

Julia is a just-in-time (JIT) compiled language. It means that the code is compiled to machine code at runtime, which is different from both the static compilation languages like C/C++ and the interpreted languages like Python.

To demonstrate the difference, let's implement a factorial function in Julia:

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

=== Understanding the Performance Gap
Python's flexibility comes at a performance cost. Python's dynamic typing means variables can change types at runtime. This requires storing type information with each value, typically as a `(type, *data)` tuple. This leads to scattered memory access patterns, causing frequent cache misses.

#figure(canvas({
  import draw: *
  let dy = -0.7
  content((-0.5, 0.5), [Stack])
  for j in range(5){
    rect((-3, dy * j), (2.0, (j+1) * dy), name: "t" + str(j))
  }
  content((0, 1.5 * dy), align(left, box(width:150pt, inset: 3pt, [(Int, 0x11ff3323)])))
  content((0, 2.5*dy), align(left, box(width: 150pt, inset: 3pt, [(Float64, 0x11ff3323)])))
  content((0, 3.5*dy), align(left, box(width: 150pt, inset: 3pt, [$dots.v$])))
  content((0, 4.5*dy), align(left, box(width: 150pt, inset: 3pt, [(Int, 0x11ff3322)])))

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

  line((-4, 0), (-4, -3.5), mark: (end: "straight"))
  content((-5, -1), [visit order])
}),
caption: "Dynamic typing causes cache misses"
)

Julia is different, it has just-in-time compilation. Although we did not specify the types of the variables explicitly, Julia can infer the types of variables at the first called and optimize the code for the specific types.

#align(center, canvas({
  import draw: *
  content((0, 0), [#box(stroke: black, inset: 10pt, [Call a function])])
}))

Given a user defined Julia function, the Julia compiler will generate a binary for it at the first called. This binary is called a *method instance*, and it is generated based on the *input types* of the function. The method instance is then stored in the method table, and it will be called when the function is called with the same input types. The method instance is generated by the LLVM compiler, and it is optimized for the input types. The method instance is a binary, and it is as fast as a C/C++ program.

== Step 1: Infer the types
Knowing the types of the variables is key to generate a fast binary. Given the input types, the Julia compiler can infer the types of the variables in the function.

If all the types are inferred, the function is called *type stable*. One can use the `@code_warntype` macro to check if the function is type stable. For example, the `jlfactorial` function with integer input is type stable:

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

If the types are not inferred, the function is called **type unstable**. For example, the `badcode` function is type unstable:

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

julia> @benchmark badcode.($x)
BenchmarkTools.Trial: 10000 samples with 8 evaluations.
 Range (min … max):  2.927 μs … 195.198 μs  ┊ GC (min … max):  0.00% … 96.52%
 Time  (median):     3.698 μs               ┊ GC (median):     0.00%
 Time  (mean ± σ):   4.257 μs ±   7.894 μs  ┊ GC (mean ± σ):  12.43% ±  6.54%

                 ▁▅█▅▃▂                                        
  ▁▃▅▇▇▇▅▃▂▂▂▃▄▆▇███████▇▇▅▄▄▃▃▃▃▃▃▂▂▃▂▂▂▂▁▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▃
  2.93 μs         Histogram: frequency by time        5.44 μs <

 Memory estimate: 26.72 KiB, allocs estimate: 696.

julia> stable(x) = x > 3 ? 1.0 : 3.0
stable (generic function with 1 method)

julia> @benchmark stable.($x)
BenchmarkTools.Trial: 10000 samples with 334 evaluations.
 Range (min … max):  213.820 ns … 25.350 μs  ┊ GC (min … max):  0.00% … 98.02%
 Time  (median):     662.551 ns              ┊ GC (median):     0.00%
 Time  (mean ± σ):   947.978 ns ±  1.187 μs  ┊ GC (mean ± σ):  29.30% ± 21.05%

  ▂▃▅██▇▅▄▃▂▁                                                  ▂
  ████████████▇▅▅▄▄▁▁▁▁▁▁▁▁▁▁▁▁▁▃▅▆▇██████▇▇▇▆█▇▇▇▇▇▇▇▇▆▇▆▆▆▇▇ █
  214 ns        Histogram: log(frequency) by time      6.32 μs <

 Memory estimate: 7.94 KiB, allocs estimate: 1.
```
In the above example:

- "`.`" is the broadcasting operator, it applies the function to each element of the array.
- "`$`" is the interpolation operator, it is used to interpolate a variable into an expression. In a benchmark, it can be used to avoid the overhead of variable initialization.

=== Step 2: Generates the LLVM intermediate representation

LLVM is a set of compiler and toolchain technologies that can be used to develop a front end for any programming language and a back end for any instruction set architecture. LLVM is the backend of multiple languages, including Julia, Rust, Swift and Kotlin.

In Julia, one can use the `@code_llvm` macro to show the LLVM intermediate representation of a function.

```julia
julia> @code_llvm jlfactorial(10)

or any instruction set architecture. LLVM is the backend of multiple languages, including Julia, Rust, Swift and Kotlin.



;  @ /Users/liujinguo/jcode/ModernScientificComputing2024/Lecture2/3.julia.jl#==#d2429055-58e9-4d84-894f-2e639723e078:1 within `jlfactorial`
define i64 @julia_jlfactorial_3677(i64 signext %0) #0 {
top:
;  @ /Users/liujinguo/jcode/ModernScientificComputing2024/Lecture2/3.julia.jl#==#d2429055-58e9-4d84-894f-2e639723e078:3 within `jlfactorial`
; ┌ @ range.jl:5 within `Colon`
; │┌ @ range.jl:403 within `UnitRange`
; ││┌ @ range.jl:414 within `unitrange_last`
     %1 = call i64 @llvm.smax.i64(i64 %0, i64 0)
; └└└
; ┌ @ range.jl:897 within `iterate`
; │┌ @ range.jl:672 within `isempty`
; ││┌ @ operators.jl:378 within `>`
; │││┌ @ int.jl:83 within `<`
      %2 = icmp slt i64 %0, 1
; └└└└
  br i1 %2, label %L32, label %L17.preheader

L17.preheader:                                    ; preds = %top
;  @ /Users/liujinguo/jcode/ModernScientificComputing2024/Lecture2/3.julia.jl#==#d2429055-58e9-4d84-894f-2e639723e078:5 within `jlfactorial`
  %min.iters.check = icmp ult i64 %1, 2
  br i1 %min.iters.check, label %scalar.ph, label %vector.ph

vector.ph:                                        ; preds = %L17.preheader
  %n.vec = and i64 %1, 9223372036854775806
  %ind.end = or i64 %1, 1
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %induction12, %vector.body ]
  %vec.phi = phi i64 [ 1, %vector.ph ], [ %3, %vector.body ]
  %vec.phi11 = phi i64 [ 1, %vector.ph ], [ %4, %vector.body ]
  %offset.idx = or i64 %index, 1
  %induction12 = add i64 %index, 2
;  @ /Users/liujinguo/jcode/ModernScientificComputing2024/Lecture2/3.julia.jl#==#d2429055-58e9-4d84-894f-2e639723e078:4 within `jlfactorial`
; ┌ @ int.jl:88 within `*`
   %3 = mul i64 %vec.phi, %offset.idx
   %4 = mul i64 %vec.phi11, %induction12
   %5 = icmp eq i64 %induction12, %n.vec
   br i1 %5, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
; └
;  @ /Users/liujinguo/jcode/ModernScientificComputing2024/Lecture2/3.julia.jl#==#d2429055-58e9-4d84-894f-2e639723e078:5 within `jlfactorial`
  %bin.rdx = mul i64 %4, %3
  %cmp.n = icmp eq i64 %1, %n.vec
  br i1 %cmp.n, label %L32, label %scalar.ph

scalar.ph:                                        ; preds = %middle.block, %L17.preheader
  %bc.resume.val = phi i64 [ %ind.end, %middle.block ], [ 1, %L17.preheader ]
  %bc.merge.rdx = phi i64 [ %bin.rdx, %middle.block ], [ 1, %L17.preheader ]
  br label %L17

L17:                                              ; preds = %L17, %scalar.ph
  %value_phi4 = phi i64 [ %7, %L17 ], [ %bc.resume.val, %scalar.ph ]
  %value_phi6 = phi i64 [ %6, %L17 ], [ %bc.merge.rdx, %scalar.ph ]
;  @ /Users/liujinguo/jcode/ModernScientificComputing2024/Lecture2/3.julia.jl#==#d2429055-58e9-4d84-894f-2e639723e078:4 within `jlfactorial`
; ┌ @ int.jl:88 within `*`
   %6 = mul i64 %value_phi6, %value_phi4
; └
;  @ /Users/liujinguo/jcode/ModernScientificComputing2024/Lecture2/3.julia.jl#==#d2429055-58e9-4d84-894f-2e639723e078:5 within `jlfactorial`
; ┌ @ range.jl:901 within `iterate`
; │┌ @ promotion.jl:521 within `==`
    %.not = icmp eq i64 %value_phi4, %1
; │└
   %7 = add nuw i64 %value_phi4, 1
; └
  br i1 %.not, label %L32, label %L17

L32:                                              ; preds = %L17, %middle.block, %top
  %value_phi10 = phi i64 [ 1, %top ], [ %bin.rdx, %middle.block ], [ %6, %L17 ]
;  @ /Users/liujinguo/jcode/ModernScientificComputing2024/Lecture2/3.julia.jl#==#d2429055-58e9-4d84-894f-2e639723e078:6 within `jlfactorial`
  ret i64 %value_phi10
}
```

=== Step 3: Compiles to binary code

The LLVM intermediate representation is then compiled to binary code by the LLVM compiler. The binary code can be printed by the `@code_native` macro.

```julia
julia> @code_native jlfactorial(10)
	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 14, 0
	.globl	_julia_jlfactorial_3726         ; -- Begin function julia_jlfactorial_3726
	.p2align	2
_julia_jlfactorial_3726:                ; @julia_jlfactorial_3726
; ┌ @ /Users/liujinguo/jcode/ModernScientificComputing2024/Lecture2/3.julia.jl#==#d2429055-58e9-4d84-894f-2e639723e078:1 within `jlfactorial`
; %bb.0:                                ; %top
; │ @ /Users/liujinguo/jcode/ModernScientificComputing2024/Lecture2/3.julia.jl#==#d2429055-58e9-4d84-894f-2e639723e078:3 within `jlfactorial`
; │┌ @ range.jl:5 within `Colon`
; ││┌ @ range.jl:403 within `UnitRange`
; │││┌ @ range.jl:414 within `unitrange_last`
	cmp	x0, #0
	csel	x9, x0, xzr, gt
; │└└└
	cmp	x0, #1
	b.lt	LBB0_3
; %bb.1:                                ; %L17.preheader
; │ @ /Users/liujinguo/jcode/ModernScientificComputing2024/Lecture2/3.julia.jl#==#d2429055-58e9-4d84-894f-2e639723e078:5 within `jlfactorial`
	cmp	x9, #2
	b.hs	LBB0_4
; %bb.2:
	mov	w8, #1
	mov	w0, #1
	b	LBB0_7
LBB0_3:
	mov	w0, #1
; │ @ /Users/liujinguo/jcode/ModernScientificComputing2024/Lecture2/3.julia.jl#==#d2429055-58e9-4d84-894f-2e639723e078:6 within `jlfactorial`
	ret
LBB0_4:                                 ; %vector.ph
	mov	x12, #0
; │ @ /Users/liujinguo/jcode/ModernScientificComputing2024/Lecture2/3.julia.jl#==#d2429055-58e9-4d84-894f-2e639723e078:5 within `jlfactorial`
	and	x10, x9, #0x7ffffffffffffffe
	orr	x8, x9, #0x1
	mov	w11, #1
	mov	w13, #1
LBB0_5:                                 ; %vector.body
                                        ; =>This Inner Loop Header: Depth=1
; │ @ /Users/liujinguo/jcode/ModernScientificComputing2024/Lecture2/3.julia.jl#==#d2429055-58e9-4d84-894f-2e639723e078:4 within `jlfactorial`
; │┌ @ int.jl:88 within `*`
	madd	x11, x11, x12, x11
; │└
; │ @ /Users/liujinguo/jcode/ModernScientificComputing2024/Lecture2/3.julia.jl#==#d2429055-58e9-4d84-894f-2e639723e078:5 within `jlfactorial`
	add	x14, x12, #2
; │ @ /Users/liujinguo/jcode/ModernScientificComputing2024/Lecture2/3.julia.jl#==#d2429055-58e9-4d84-894f-2e639723e078:4 within `jlfactorial`
; │┌ @ int.jl:88 within `*`
	mul	x13, x13, x14
	mov	x12, x14
	cmp	x10, x14
	b.ne	LBB0_5
; %bb.6:                                ; %middle.block
; │└
; │ @ /Users/liujinguo/jcode/ModernScientificComputing2024/Lecture2/3.julia.jl#==#d2429055-58e9-4d84-894f-2e639723e078:5 within `jlfactorial`
	mul	x0, x13, x11
	cmp	x9, x10
	b.eq	LBB0_9
LBB0_7:                                 ; %L17.preheader15
	add	x9, x9, #1
LBB0_8:                                 ; %L17
                                        ; =>This Inner Loop Header: Depth=1
; │ @ /Users/liujinguo/jcode/ModernScientificComputing2024/Lecture2/3.julia.jl#==#d2429055-58e9-4d84-894f-2e639723e078:4 within `jlfactorial`
; │┌ @ int.jl:88 within `*`
	mul	x0, x0, x8
; │└
; │ @ /Users/liujinguo/jcode/ModernScientificComputing2024/Lecture2/3.julia.jl#==#d2429055-58e9-4d84-894f-2e639723e078:5 within `jlfactorial`
; │┌ @ range.jl:901 within `iterate`
	add	x8, x8, #1
; │└
	cmp	x9, x8
	b.ne	LBB0_8
LBB0_9:                                 ; %L32
; │ @ /Users/liujinguo/jcode/ModernScientificComputing2024/Lecture2/3.julia.jl#==#d2429055-58e9-4d84-894f-2e639723e078:6 within `jlfactorial`
	ret
; └
                                        ; -- End function
.subsections_via_symbols
```

Single function definition may have multiple method instances.

```julia
julia> methods(jlfactorial)
# 1 method for generic function "jlfactorial" from Main:
 [1] jlfactorial(n)
     @ REPL[4]:1
```

Whenever the function is called with a new input type, the Julia compiler will generate a new method instance for the function. The method instance is then stored in the method table, and can be analyzed by the `MethodAnalysis` package.

```julia
julia> using MethodAnalysis

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

When a function is called with multiple arguments, the Julia compiler will invoke the correct method instance according to the type of the arguments. This is called *multiple dispatch*.

== Julia Types and Multiple Dispatch

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