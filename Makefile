test:
	python -m pytest tests $(filter-out $@,$(MAKECMDGOALS))

server:
	mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root mlruns
