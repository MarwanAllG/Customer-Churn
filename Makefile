.PHONY: install format lint api train retrain monitor

install:
	pip install -r requirements.txt

format:
	black .
	ruff --fix .

lint:
	ruff .

api:
	python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

train:
	python Model/model.py customer_churn.json 30

retrain:
	python scripts/retrain.py --data customer_churn.json --window 30

monitor:
	python scripts/monitor.py --ref artifacts/reference_stats.json --data customer_churn.json --window 30


