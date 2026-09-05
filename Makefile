SHELL := /usr/bin/env bash

.PHONY: help build test clean-dist

help:
	@printf 'Targets:\n'
	@printf '  make build        Build sdist and wheel\n'
	@printf '  make test         Run the library test suite\n'
	@printf '  make clean-dist   Remove generated distribution files\n'

build:
	python -m build

test:
	python -m pytest -q tests/

clean-dist:
	rm -rf dist/*
