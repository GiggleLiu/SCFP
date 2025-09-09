# The Computer Language Benchmarks Game
# https://salsa.debian.org/benchmarksgame-team/benchmarksgame/

# Julia struct definition with type annotations
# The `struct` keyword defines an immutable composite type
struct Node
    # Field declarations with explicit type annotations using :: operator
    # These create self-referential types (Node contains Node fields)
    left::Node
    right::Node
    
    # Inner constructor methods defined within the struct
    # Multiple dispatch allows multiple constructors with different signatures
    function Node(left::Node, right::Node)
        # `new()` is Julia's built-in constructor function
        # Only available inside type definitions
        new(left, right)
    end
    
    # Default constructor with no arguments
    # This creates an uninitialized Node (fields are #undef)
    function Node()
        new()
    end
end

# Recursive function definition without explicit type annotations
# Julia can infer types automatically (type inference)
function make(d)
    # Standard if-else conditional structure
    # `==` is the equality comparison operator
    if d == 0
        # Calling the parameterless constructor
        Node()
    else
        # Recursive function calls - Julia optimizes tail recursion
        # Function calls use parentheses syntax: func(args...)
        Node(make(d-1), make(d-1))
    end
    # Julia functions automatically return the last expression
    # No explicit `return` statement needed
end

# One-line function definition with explicit type annotation
# The :: operator specifies that parameter `t` must be of type Node
# This enables multiple dispatch - different methods for different types
check(t::Node) = 1 + 
    # Ternary operator: condition ? true_value : false_value
    # `isdefined(object, :field)` checks if a field exists/is defined
    # Symbol literals use colon prefix: :left, :right
    (isdefined(t, :left) ? check(t.left) : 0) + 
    (isdefined(t, :right) ? check(t.right) : 0)
    # Field access using dot notation: object.field
    # Parentheses group expressions for proper operator precedence

function loop_depths(d, min_depth, max_depth)
    # For loop with range syntax: start:end creates an inclusive range
    # `div(a, b)` is integer division (equivalent to `a ÷ b`)
    for i = 0:div(max_depth - d, 2)
        # Left bit shift operator: `<<` equivalent to `* 2^n`
        # 1 << n creates 2^n efficiently using bit operations
        niter = 1 << (max_depth - d + min_depth)
        
        # Variable initialization - Julia infers Int type
        c = 0
        
        # Another range: 1:niter creates range from 1 to niter inclusive
        # Range iteration is memory efficient (doesn't create array)
        for j = 1:niter
            # Compound assignment operator: `+=` equivalent to `c = c + ...`
            c += check(make(d)) 
        end
        
        # String interpolation using $ within double quotes
        # Variables/expressions can be embedded directly: $variable
        println("$niter trees of depth $d check: $c")
        
        # Increment using compound assignment
        d += 2
    end
end

# Function with default parameter value using = syntax
# Type annotation :: specifies parameter must be Int type
function perf_binary_trees(N::Int=10)
    # Local variable assignments
    min_depth = 4
    max_depth = N
    stretch_depth = max_depth + 1

    # `let` block creates a new local scope
    # Variables inside `let` don't leak to outer scope
    # Useful for temporary computations
    let c = check(make(stretch_depth))
        println("stretch tree of depth $stretch_depth check: $c")
    end
    # Variable `c` is not accessible here (out of scope)

    # Function call result stored in variable
    long_lived_tree = make(max_depth)

    # Function call with multiple arguments
    loop_depths(min_depth, min_depth, max_depth)
    
    # String interpolation with expression: $(expression)
    # Parentheses are needed for complex expressions in interpolation
    println("long lived tree of depth $max_depth check: $(check(long_lived_tree))")
end

# Direct function call at module level (script execution)
# This executes when the file is run directly
# Julia allows top-level code execution outside functions
perf_binary_trees(21)