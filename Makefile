PY ?= python3
export PYTHONPATH := src

.PHONY: install install-extras demo generate ingest ask eval report summary test lint serve docker clean

install:
	$(PY) -m pip install -r requirements-dev.txt

install-extras:
	$(PY) -m pip install -r requirements-extras.txt

## Corpus → index → a few questions → evaluation → README images
demo: clean generate ingest eval ask report

generate:
	$(PY) -m ragqa generate

ingest:
	$(PY) -m ragqa ingest

ask:
	$(PY) -m ragqa ask "What is the hotel limit per night in a Tier 1 city?"
	$(PY) -m ragqa ask "Under the 2023 travel policy, what was the mileage rate for a personal car?"
	$(PY) -m ragqa ask "Who is the CEO of Kestrelway Logistics?"

eval:
	$(PY) -m ragqa eval

report:
	$(PY) -m ragqa report

summary:
	$(PY) -m ragqa summary

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

serve:
	$(PY) -m ragqa serve

docker:
	docker compose up --build

clean:
	rm -rf data reports
