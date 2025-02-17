#import "../book.typ": book-page, cross-link
#show: book-page.with(title: "Setup Julia")

= Setup Julia

== Introduction to Julia
#link("https://julialang.org/")[Julia] is a modern, high-performance programming language designed for technical computing. Created at MIT in 2012 and now maintained by JuliaHub Inc., Julia combines the ease of use of Python with the speed of C/C++.

Julia stands out from other programming languages in several important ways:

1. *Open Source*: Unlike MatLab, Julia is completely open source. The source code is maintained on #link("https://github.com/JuliaLang/julia")[GitHub], and its packages are available on #link("https://juliahub.com/ui/Packages")[JuliaHub].

2. *High Performance*: Unlike Python, Julia was designed from the ground up for high performance (#link("https://arxiv.org/abs/1209.5145")[arXiv:1209.5145]). It achieves C-like speeds while maintaining the simplicity of a dynamic language.

3. *Easy to Use*: Unlike C/C++ or Fortran, Julia offers a clean, readable syntax and interactive development environment. Its just-in-time (JIT) compilation provides platform independence while maintaining high performance.


While we will explore Julia in depth later in this chapter, let's begin by installing and setting up our development environment.

== Step 1: Install Julia 
We recommend using #link("https://github.com/JuliaLang/juliaup")[juliaup] to install Julia. This tool manages Julia versions and installations, allowing you to seamlessly switch between different versions. Just open a terminal and run the following platform-specific command:

```bash
# Linux (including windows subsystem for linux) and macOS
curl -fsSL https://install.julialang.org | sh
```

```bash
# Windows
winget install julia -s msstore
```

=== Select a mirror for faster installation (optional)
If you experience slow download speeds, particularly in regions like China, you can use alternative servers for both Juliaup and Julia packages. Here, we use the mirror from #link("https://mirrors.nju.edu.cn/")[Nanjing University].

```bash
# Linux and macOS
export JULIAUP_SERVER=https://mirror.nju.edu.cn/julia-releases/
export JULIA_PKG_SERVER=https://mirrors.nju.edu.cn/julia
```
```bash
# Windows
env:JULIAUP_SERVER="https://mirror.nju.edu.cn/julia-releases/"
env:JULIA_PKG_SERVER="https://mirrors.nju.edu.cn/julia"
```
You can also download Julia binaries directly from the #link("https://mirror.nju.edu.cn/julia-releases/")[Nanjing University mirror]. After installation, configure your system's PATH to enable launching Julia from the terminal. For detailed instructions, consult the #link("https://julialang.org/downloads/platform/")[platform-specific guide].

=== Verifying Your Installation
To confirm that Julia is installed correctly, open a *new* terminal and run:
```bash
julia
```
This command should launch the Julia REPL (Read-Eval-Print-Loop).
To verify the version and configuration, run this command in the Julia REPL:
```julia
julia> versioninfo()
Julia Version 1.9.2
Commit e4ee485e909 (2023-07-05 09:39 UTC)
Platform Info:
  OS: macOS (arm64-apple-darwin22.4.0)
  CPU: 10 × Apple M2 Pro
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, apple-m1)
  Threads: 1 on 6 virtual cores
Environment:
  JULIA_NUM_THREADS = 1
  JULIA_PROJECT = @.
  JULIA_PKG_SERVER = http://cn-southeast.pkg.juliacn.com/ 
```

== Step 2: Install Revise (strongly recommended)
`Revise` is a package that greatly enhances the Julia development experience. It allows you to edit code and have it automatically loaded into the REPL without restarting the kernel.

1. Install #link("https://github.com/timholy/Revise.jl")[Revise] in the REPL:
  ```julia
  julia> using Pkg; Pkg.add("Revise")
  ```
2. Create a startup configuration file by running:
  ```bash
  mkdir -p ~/.julia/config
  echo 'try
      using Revise
  catch e
      @warn "fail to load Revise."
  end' > ~/.julia/config/startup.jl
  ```
  The startup file executes automatically when you launch Julia, which automatically loads `Revise` for you.

== Step 3: Setting up Editor: VSCode
1. Download and install #link("https://code.visualstudio.com/download")[Visual Studio Code], or its variant #link("https://www.cursor.com/")[Cursor], both are refered as VSCode in the rest of the book.
2. Launch VSCode and open the Extensions sidebar. Search for "julia" and install the official #link("https://github.com/julia-vscode/julia-vscode")[julia-vscode] extension
3. To verify the installation, create a new file in VSCode with the `.jl` extension with the following content:
  ```julia
  println("Hello, world!")
  ```
  To save the file, press `Ctrl+S` (or `Cmd+S` on macOS).
  To run the file, press `Shift+Enter`. If the installation is successful, you should see the output "Hello, world!" in the REPL pop up at the bottom of the VSCode window.

#box(stroke: black, inset: (x: 7pt, y: 5pt), radius: 4pt, [
*Tips: Connect to a remote server in VSCode*

To connect to a remote server in VSCode, you can use the `Remote-SSH` extension.
1. Open the extensions sidebar and search for "Remote-SSH", click on the "Install" button.
2. Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P on macOS) and select "Remote-SSH: Connect to Host...".
3. Enter the hostname or IP address of the remote server and press `Enter`. You will be prompted to enter the username and password of the remote server. Alternatively, you can configure the remote serve in the `.ssh/config` file first, then you will see the remote server in the list of SSH targets.
])
