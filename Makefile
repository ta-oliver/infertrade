.ONESHELL:

SHELL := /bin/bash
# Get package name from pwd
PACKAGE_NAME := $(shell basename $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST)))))

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys
print("Please use `make <target>` where <target> is one of\n")
for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		if not target.startswith('--'):
			print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT
export PYTHONWARNINGS=ignore

BROWSER := python3 -c "$$BROWSER_PYSCRIPT"

.PHONY: help
help:
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)


.PHONY: test
test: ## run tests quickly with pytest
	pytest

.PHONY: coverage
coverage: ## check code coverage quickly with pytest
	pytest --cov-report term-missing --cov=$(PACKAGE_NAME)
	

.PHONY: install
install:
	python3.7 -c "import $(PACKAGE_NAME)" >/dev/null 2>&1 || python3 -m pip install . && \
    python3.7 setup.py build_ext --inplace;


.PHONY: venv
venv:  ## create virtualenv environment on local directory.
	@if ! command -v virtualenv >/dev/null 2>&1; then \
		pip install virtualenv;\
	fi && \
	virtualenv ".$(PACKAGE_NAME)_venv" -p python3 -q;

.PHONY: dev
dev: clean ## install the package in development mode including all dependencies
	python3.7 -m pip install .[dev]

.PHONY: dev-venv
dev-venv: venv ## install the package in development mode including all dependencies inside a virtualenv (container).

.PHONY: autoformat
autoformat: ## formats code
	black -l 120 $(PACKAGE_NAME)


	