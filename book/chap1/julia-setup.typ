#import "../book.typ": book-page, cross-link
#show: book-page.with(title: "Setup Julia")

= Setup Julia

#let julia = `Julia`

#julia is a high-level, high-performance, dynamic programming language. Designed from the ground up, #julia addresses the needs of high-performance numerical analysis and computational science without requiring separate compilation to achieve speed. It excels not only in scientific computing but also serves effectively as a general-purpose programming language, web development tool, and specification language. As a free and open-source project, #julia benefits from a #link("https://julialang.org/community/")[thriving community] and a #link("https://juliahub.com/")[rich ecosystem] of packages.

While we will explore #julia in depth later in this chapter, let's begin by installing and setting up our development environment.

== Step 1: Installing Julia 
For Linux and macOS users, open a terminal and execute the following command to install #link("https://julialang.org/")[Julia] using #link("https://github.com/JuliaLang/juliaup")[juliaup]. This tool manages Julia versions and installations, allowing you to seamlessly switch between different versions:

```bash
curl -fsSL https://install.julialang.org | sh # Linux and macOS
```

For Windows users, execute this command in a Command Prompt:
```powershell
winget install julia -s msstore # Windows
```
Alternatively, you can install Juliaup directly from the #link("https://www.microsoft.com/store/apps/9NJNWW8PVKMN")[Windows Store].

=== Alternative Installation for Slower Networks
If you experience slow download speeds, particularly in regions like China, you can use alternative servers for both Juliaup and Julia packages. Execute these commands in your terminal before the installation:

*Linux and macOS*
```bash
export JULIAUP_SERVER=https://mirror.nju.edu.cn/julia-releases/
export JULIA_PKG_SERVER=https://mirrors.nju.edu.cn/julia
```
*Windows*
```powershell
env:JULIAUP_SERVER="https://mirror.nju.edu.cn/julia-releases/"
env:JULIA_PKG_SERVER="https://mirrors.nju.edu.cn/julia"
```
You can also download Julia binaries directly from the #link("https://mirror.nju.edu.cn/julia-releases/")[Nanjing University mirror]. After installation, configure your system's PATH to enable launching Julia from the terminal. For detailed instructions, consult the #link("https://julialang.org/downloads/platform/")[platform-specific guide].

=== Verifying Your Installation
To confirm that Julia is installed correctly, open a *new* terminal and run:
```bash
julia
```
This command should launch the Julia REPL (Read-Eval-Print-Loop). For installing specific Julia versions, refer to the #link("https://github.com/JuliaLang/juliaup")[juliaup documentation].

== Step 2: Package Management
#julia's package ecosystem provides powerful tools for scientific computing:
- `Pkg` serves as Julia's built-in package manager
- Enter package mode by pressing `]` in the REPL
#image("images/Packages.gif")
- Look for the environment indicator `(@v1.9)`
- Install packages with `add <package name>`
- Exit package mode with the `backspace` key
- #link("https://pkgdocs.julialang.org/v1/managing-packages/")[Learn more about package management]

== Step 3: Configure the Startup File
Create a startup configuration file by running:

```bash
mkdir -p ~/.julia/config
echo 'try
    using Revise
catch e
    @warn "fail to load Revise."
end' > ~/.julia/config/startup.jl
```

This startup file executes automatically when you launch Julia. Next, install #link("https://github.com/timholy/Revise.jl")[Revise], a package that enhances the Julia development experience:

```julia
julia> using Pkg; Pkg.add("Revise")
```

For more information about startup configuration, visit the #link("https://docs.julialang.org/en/v1/manual/command-line-interface/#Startup-file")[startup file documentation].

=== Additional Packages
Browse the complete collection of Julia packages at #link("https://juliahub.com/")[JuliaHub].

To verify your configuration, run this command in the Julia REPL:
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

== Step 4: Setting Up VSCode
1. Download and install VSCode from #link("https://code.visualstudio.com/download")[Visual Studio Code].
2. Launch VSCode and open the Extensions sidebar
3. Search for "Julia" and install the official #link("https://github.com/julia-vscode/julia-vscode")[julia-vscode] extension

== Understanding the Julia REPL Modes

The Julia REPL offers four specialized modes:

1. *Julian Mode*: The default mode for executing Julia code.

2. *Shell Mode*: Access system commands by pressing `;`:
```julia
shell> date
Sun Nov  6 10:50:21 PM CST 2022
```

3. *Package Mode*: Manage packages by pressing `]`:
```julia
(@v1.8) pkg> st
Status `~/.julia/environments/v1.8/Project.toml`
  [295af30f] Revise v3.4.0
```

4. *Help Mode*: Access documentation by pressing `?`:
```julia
help> sum
... docstring for sum ...
```

Return to Julian mode from any other mode by pressing `Backspace`.

For comprehensive documentation on the REPL, visit the #link("https://docs.julialang.org/en/v1/stdlib/REPL/")[official REPL guide].