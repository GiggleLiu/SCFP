JL = julia --project

default: init

init:
	$(JL) -e 'using Pkg; Pkg.update()'

watch:
	$(JL) -e 'import TypstJlyfish; TypstJlyfish.watch("book/chap6-quantum/quantum-simulation.typ"; typst_args = "--root .")'

build:
	cd book && shiroa build

serve:
	cd book && shiroa serve

.PHONY: init watch build serve
