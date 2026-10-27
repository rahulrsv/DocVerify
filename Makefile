.PHONY: install test eval baseline eval-llm data api dashboard up down clean

PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

install:
	$(PY) -m pip install -r requirements-dev.txt

test:
	$(PY) -m pytest -q

eval:
	$(PY) -m app.evals.run --dataset data/golden --fail-on-regression

baseline:
	$(PY) -m app.evals.run --dataset data/golden --update-baseline

eval-llm:
	$(PY) -m app.evals.run --dataset data/golden --provider openai

data:
	$(PY) scripts/generate_golden_dataset.py

api:
	$(PY) -m uvicorn app.main:create_app --factory --reload --port 8100

dashboard:
	API_URL=http://localhost:8100 $(PY) -m streamlit run dashboard/streamlit_app.py --server.port 8502

up:
	docker compose up --build

down:
	docker compose down

clean:
	rm -rf var/*.db reports/eval_*.json reports/latest.*
