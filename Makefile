.PHONY: setup dev test lint format precommit train

setup:
	python -m pip install -U pip
	pip install -e ".[dev]"

dev:
	pip install -e ".[dev,mlops]"

precommit:
	pre-commit install

lint:
	ruff check .

format:
	ruff format .

test:
	pytest -q

train:
	python scripts/train.py --config configs/train.yaml
