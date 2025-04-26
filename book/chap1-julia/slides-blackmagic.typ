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

#let show-graph(vertices, edges, radius:0.2) = {
  import draw: *
  for (k, (i, j)) in vertices.enumerate() {
    circle((i, j), radius:radius, name: str(k), fill:white)
  }
  for (k, l) in edges {
    line(str(k), str(l))
  }
}

#let m = hkustgz-theme.register()

#show raw.where(block: true): it=>{
  block(radius:4pt, fill:gray.transparentize(90%), inset:1em, width:99%, text(it))
}

// Global information configuration
#let m = (m.methods.info)(
  self: m,
  title: [Julia tips and magic],
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

= Performance Tips

== No Global variables
== Avoid using abstract types in field
```julia
struct MyAmbiguousType
    a
end
```

= Metaprogramming
== `@eval`
== Generated function
https://gist.github.com/GiggleLiu/d72b04fd4d2123a4dba0d024e210da6c

== String literal and regular expression

== MLStyle


= Static arrays

= Multi-threading the multi-processing

This section applies to multithreaded Julia code which, in each thread, performs linear algebra operations. Indeed, these linear algebra operations involve BLAS / LAPACK calls, which are themselves multithreaded. In this case, one must ensure that cores aren't oversubscribed due to the two different types of multithreading.

Julia compiles and uses its own copy of OpenBLAS for linear algebra, whose number of threads is controlled by the environment variable OPENBLAS_NUM_THREADS. It can either be set as a command line option when launching Julia, or modified during the Julia session with BLAS.set_num_threads(N) (the submodule BLAS is exported by using LinearAlgebra). Its current value can be accessed with BLAS.get_num_threads().

When the user does not specify anything, Julia tries to choose a reasonable value for the number of OpenBLAS threads (e.g. based on the platform, the Julia version, etc.). However, it is generally recommended to check and set the value manually. The OpenBLAS behavior is as follows:

If OPENBLAS_NUM_THREADS=1, OpenBLAS uses the calling Julia thread(s), i.e. it "lives in" the Julia thread that runs the computation.
If OPENBLAS_NUM_THREADS=N>1, OpenBLAS creates and manages its own pool of threads (N in total). There is just one OpenBLAS thread pool shared among all Julia threads.
When you start Julia in multithreaded mode with JULIA_NUM_THREADS=X, it is generally recommended to set OPENBLAS_NUM_THREADS=1. Given the behavior described above, increasing the number of BLAS threads to N>1 can very easily lead to worse performance, in particular when $N<<X$. However this is just a rule of thumb, and the best way to set each number of threads is to experiment on your specific application.

As an alternative to OpenBLAS, there exist several other backends that can help with linear algebra performance. Prominent examples include #link("https://github.com/JuliaLinearAlgebra/MKL.jl")[MKL.jl] and #link("https://github.com/JuliaMath/AppleAccelerate.jl")[AppleAccelerate.jl].

= CUDA programming

== CuArrays


== The GPU accelerated simulated annealing

== Calling C and Fortran

== The CUDA version of Einsum
https://gist.github.com/GiggleLiu/57c0fd7d6d775dbe6892a1b2efda40f8