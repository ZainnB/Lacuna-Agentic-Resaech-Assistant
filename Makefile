.PHONY: setup services migrate test lint fmt e1 e6 reproduce

setup:
	uv sync --extra dev

services:
	docker compose up -d --wait

migrate: services
	uv run lacuna db migrate

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

# E1: survey recall for the two $0 baselines (keyword search vs citation snowballing)
e1: migrate
	uv run lacuna evals e1

# E6: deep-read retrieval baseline (no LLM)
e6: migrate
	uv run lacuna evals e6 --retrieval-only

reproduce: setup migrate e1 e6
