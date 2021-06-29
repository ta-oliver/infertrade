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
DOCUMENTATION := documentation

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

.PHONY: html
html: ## to make standalone HTML files
	$(MAKE) -C DOCUMENTATION html

.PHONY: dirhtml
dirhtml: ## to make HTML files named index.html in directories
	$(MAKE) -C DOCUMENTATION dirhtml

.PHONY: singlehtml  
singlehtml: ## to make a single large HTML file
	$(MAKE) -C DOCUMENTATION singlehtml

.PHONY: pickle
pickle: ## to make pickle files
	$(MAKE) -C DOCUMENTATION pickle

.PHONY: json
json: ## to make standalone HTML files
	$(MAKE) -C DOCUMENTATION json

.PHONY: htmlhelp
htmlhelp: ## to make HTML files and an HTML help project
	$(MAKE) -C DOCUMENTATION htmlhelp

.PHONY: qthelp
qthelp: ## to make standalone HTML files
	$(MAKE) -C DOCUMENTATION qthelp

.PHONY: devhelp
devhelp: ## to make HTML files and a Devhelp project
	$(MAKE) -C DOCUMENTATION devhelp

.PHONY: epub
epub: ## to make an epub
	$(MAKE) -C DOCUMENTATION epub

.PHONY: latex
latex: ## to make LaTeX files, you can set PAPER=a4 or PAPER=letter
	$(MAKE) -C DOCUMENTATION latex

.PHONY: latexpdf
latexpdf: ## to make LaTeX and PDF files (default pdflatex)
	$(MAKE) -C DOCUMENTATION latexpdf

.PHONY: latexpdfja
latexpdfja: ## to make LaTeX files and run them through platex/dvupdfmx
	$(MAKE) -C DOCUMENTATION latexpdfja

.PHONY: text
text: ## to make LaTeX and PDF files (default pdflatex)
	$(MAKE) -C DOCUMENTATION tex

.PHONY: man
man: ## to make manual pages
	$(MAKE) -C DOCUMENTATION man

.PHONY: texinfo
texinfo: ## to make Texinfo files
	$(MAKE) -C DOCUMENTATION texinfo

.PHONY: info
info: ## to make Texinfo files and run them through makeinfo
	$(MAKE) -C DOCUMENTATION info

.PHONY: gettext
gettext: ## to make PO message catalogs
	$(MAKE) -C DOCUMENTATION gettext

.PHONY: changes 
changes: ## to make an overview of all changed/added/deprecated items
	$(MAKE) -C DOCUMENTATION changes 

.PHONY: xml
xml: ## to make Docutils-native XML files
	$(MAKE) -C DOCUMENTATION xml

.PHONY: pseudoxml
psedoxml: ## to make pseudoxml-XML files for display purposes
	$(MAKE) -C DOCUMENTATION pseudoxml

.PHONY: linkcheck   
linkcheck: ## to check all external links for integrity
	$(MAKE) -C DOCUMENTATION linkcheck

.PHONY: doctest 
doctest: ## to run all doctests embedded in the documentation (if enabled)
	$(MAKE) -C DOCUMENTATION doctest  

.PHONY: doc-coverage 
doc-coverage: ## to run coverage check of the documentation (if enabled)
	$(MAKE) -C DOCUMENTATION coverage