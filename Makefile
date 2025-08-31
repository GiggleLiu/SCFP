JL = julia --project

default: init

init:
	$(JL) -e 'using Pkg; Pkg.update()'

build:
	cd book && shiroa build

serve:
	cd book && shiroa serve

compile-all:
	cd book && find . -path "*/chap*/*.typ" -exec typst compile --root=. {} \;

clean-pdf:
	cd book && find . -path "*/chap*/*.pdf" -exec rm {} \;

.PHONY: init build serve compile-all clean-pdf
