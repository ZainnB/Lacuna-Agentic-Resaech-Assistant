.PHONY: db data build eval reproduce test

db:
	docker compose up -d --wait

data:
	uv run figrag download

build: db
	uv run figrag build

eval: db
	uv run python eval/run_eval.py

# docker compose up + one command -> the results table
reproduce: data build eval

test:
	uv run pytest
