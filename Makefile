JL = julia --project

default: init

init:
	$(JL) -e 'using Pkg; Pkg.update()'

watch:
	$(JL) -e 'import TypstJlyfish; TypstJlyfish.watch("book/chap6-quantum/quantum-simulation.typ"; typst_args = "--root .")'

build:
	shiroa build

serve:
	shiroa serve

.PHONY: init watch build serve
