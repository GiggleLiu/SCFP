JL = julia --project

default: init

init:
	$(JL) -e 'using Pkg; Pkg.update()'

build:
	cd book && shiroa build

serve:
	cd book && shiroa serve

.PHONY: init build serve
