using MLStyle

macro foo(ex)
  @match ex begin
    :(function $name($(args...)) $(body...) end) => esc(quote
      function $name($(mirror.(args)...))
        $(mirror.(body)...)
      end
    end)
    ::LineNumberNode => ex
  end
end

function mirror(ex)
  @match ex begin
    :(return $ex) => quote
      return $(mirror(ex))
    end
    :(begin $(body...) end) => quote
      begin $(mirror.(body)...) end
    end
    :($a + $b) => :($(mirror(a)) - $(mirror(b)))
    :($a - $b) => :($(mirror(a)) + $(mirror(b)))
    :($a * $b) => :($(mirror(a)) / $(mirror(b)))
    :($a / $b) => :($(mirror(a)) * $(mirror(b)))
    :($f($(args...))) => :($f($(mirror.(args)...)))
    _ => ex  # default case
  end
end

mirror(:(f(x + y)))
@macroexpand @foo function f(x, y) return x + y end