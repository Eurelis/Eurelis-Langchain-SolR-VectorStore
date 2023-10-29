SHELL := /bin/bash
CHDIR_SHELL := $(SHELL)

PYTHON := python

#
# Setup
#
init-venv:
	@echo "***** $@"
	${PYTHON} -m venv ./.venv

update-venv: init-venv
	@echo "***** $@"
	@source .venv/bin/activate &&\
	pip install --upgrade pip &&\
	pip install -r src/requirements.txt

init-project: update-venv

#
# Build
#
build-project: update-venv
	@echo "***** $@"
	@source .venv/bin/activate &&\
	pip install --upgrade build &&\
	python -m build

fast-rebuild:
	@echo "***** $@"
	@source .venv/bin/activate &&\
	python -m build

upload-project:
	@echo "***** $@"
	@source .venv/bin/activate &&\
	pip install --upgrade twine &&\
	twine upload --repository pypi dist/*
