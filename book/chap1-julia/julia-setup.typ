#import "../book.typ": book-page, cross-link
#show: book-page.with(title: "Setup Julia")

= Setup Julia

#link("https://julialang.org/")[Julia] is a high-level, high-performance, dynamic programming language. Designed from the ground up, Julia addresses the needs of high-performance numerical analysis and computational science without requiring separate compilation to achieve speed.

While we will explore Julia in depth later in this chapter, let's begin by installing and setting up our development environment.

== Step 1: Installing Julia 
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



== Step 2: Package Management
Julia's package ecosystem provides powerful tools for scientific computing. The information about Julia packages is available at #link("https://juliahub.com/")[JuliaHub].
A julia package can be installed in the package mode of the REPL:
1. Enter package mode with `]`
  #image("images/Packages.gif")
2. Install packages with `add <package name>`. For more commands, just type `?` and press `Enter`.
3. Exit package mode with the `backspace` key

== Step 3: Install Revise (strongly recommended)
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

== Step 4: Setting up Editor: VSCode
1. Download and install #link("https://code.visualstudio.com/download")[Visual Studio Code], or its variant #link("https://www.cursor.com/")[Cursor], both are refered as VSCode in the rest of the book.
2. Launch VSCode and open the Extensions sidebar. Search for "julia" and install the official #link("https://github.com/julia-vscode/julia-vscode")[julia-vscode] extension
3. To verify the installation, create a new file in VSCode with the `.jl` extension with the following content:
  ```julia
  println("Hello, world!")
  ```
  To save the file, press `Ctrl+S` (or `Cmd+S` on macOS).
  To run the file, press `Shift+Enter`. If the installation is successful, you should see the output "Hello, world!" in the REPL pop up at the bottom of the VSCode window.