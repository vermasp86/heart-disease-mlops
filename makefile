.PHONY: help install test train api docker-build docker-run docker-push deploy monitor clean

help:
	@echo "Available commands:"
	@echo "  install     Install dependencies"
	@echo "  test        Run tests"
	@echo "  train       Train models"
	@echo "  api         Run API server"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run Docker container"
	@echo "  docker-push  Push Docker image"
	@echo "  deploy      Deploy to Kubernetes"
	@echo "  monitor     Start monitoring stack"
	@echo "  clean       Clean generated files"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=html

train:
	python src/models/train.py

api:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t heart-disease-api .

docker-run:
	docker-compose up -d

docker-push:
	docker push heart-disease-api:latest

deploy:
	kubectl apply -f kubernetes/deployment.yaml
	kubectl rollout status deployment/heart-disease-api -n heart-disease

monitor:
	docker-compose -f docker-compose.monitoring.yml up -d

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf mlruns
	rm -rf logs/*
	rm -rf reports/*
