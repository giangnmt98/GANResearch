# Shell used for executing scripts
SHELL := /bin/bash

# Configuration
# Alias for Python executable
PYTHON_EXEC := python3

# Virtual environment directory
VENV_DIR := ganresearch_env

# Directory for storing processed data
CODE_DIRECTORY := ganresearch


# Virtual environment setup
venv:
	# Creates the virtual environment and installs dependencies
	$(PYTHON_EXEC) -m venv $(VENV_DIR) && \
	source $(VENV_DIR)/bin/activate && \
	$(PYTHON_EXEC) -m pip install --upgrade pip setuptools wheel && \
 	$(PYTHON_EXEC) -m pip install -e .[dev]
#  	$(PYTHON_EXEC) -m install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Code stylingso
style:
	# Formats the code with Black
	black ./$(CODE_DIRECTORY)
	# Checks the code style using Flake8
	flake8 ./$(CODE_DIRECTORY)
	# Sorts the imports using isort
	isort -rc ./$(CODE_DIRECTORY)


.PHONY: venv style