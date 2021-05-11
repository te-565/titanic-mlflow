# Based on: https://towardsdatascience.com/create-virtualenv-for-data-science-projects-with-one-command-only-7bec3548419f

# Load in Environment Variables

# Variables
ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
DEACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda deactivate ; conda deactivate
CONDA_ENVIRONMENT_NAME=titanic-mlflow-env
PYTHON_VERSION=3.8

# Help
.DEFAULT_GOAL := help
.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


# Environment Management
.PHONY: create-environment
create-environment: ## Create the env, install packages, create a kernel and write to environment.yaml
	conda create --name $(CONDA_ENVIRONMENT_NAME) --channel conda-forge --yes python=$(PYTHON_VERSION)
	$(ACTIVATE) $(CONDA_ENVIRONMENT_NAME) && \
	conda install --yes --file requirements-conda.txt && \
	pip install -r requirements-pip.txt && \
	conda env export > environment.yaml
	$(DEACTIVATE)

.PHONY: install-all-requirements
install-all-requirements: ## Install packages in requirements-conda & requirements-pip and write to environment.yaml
	$(ACTIVATE) $(CONDA_ENVIRONMENT_NAME) && \
	conda install --yes --file requirements-conda.txt && \
	pip install -r requirements-pip.txt && \
	conda env export > environment.yaml
	$(DEACTIVATE)

.PHONY: install-conda-requirements
install-conda-requirements: ## Install packages in requirements-conda file and write to environment.yaml
	$(ACTIVATE) $(CONDA_ENVIRONMENT_NAME) && \
	conda install --yes --file requirements-conda.txt && \
	conda env export > environment.yaml
	$(DEACTIVATE)


.PHONY: install-pip-requirements
install-pip-requirements: ## Install the packages in the requirements-pip file and write to environment.yaml
	$(ACTIVATE) $(CONDA_ENVIRONMENT_NAME) && \
	pip install -r requirements-pip.txt && \
	conda env export > environment.yaml
	$(DEACTIVATE) 

.PHONY: remove-environment 
remove-environment: ## Remove the environment and any relevant files
	conda env remove --name $(CONDA_ENVIRONMENT_NAME)


# Jupyter kernel management
.PHONY: create-kernel
create-kernel: ## Register the conda environment to Jupyter
	$(ACTIVATE) $(CONDA_ENVIRONMENT_NAME) && \
	ipython kernel install --user --name=$(CONDA_ENVIRONMENT_NAME)
	$(DEACTIVATE) 

.PHONY: remove-kernel
remove-kernel: ## Remove the conda environment from Jupyter
	jupyter kernelspec uninstall $(CONDA_ENVIRONMENT_NAME)

# Execution
.PHONY: run-all
run-all:
	$(ACTIVATE) $(CONDA_ENVIRONMENT_NAME) && \
	python -m main
	$(DEACTIVATE)


# Tests
.PHONY: tests
tests: ## Runs the tests for the application & reports test coverage
	$(ACTIVATE) $(CONDA_ENVIRONMENT_NAME) && \
	coverage run -m pytest -v && \
	coverage report -m
	$(DEACTIVATE)


# MLFlow UI
.PHONY: mlflow-server
mlflow: ## Start the MLFlow webserver
	$(ACTIVATE) $(CONDA_ENVIRONMENT_NAME) && \
	mlflow server \
	--port 5000 \
	--backend-store-uri mlruns/
