# Makefile for MLflow CI/CD pipeline setup

# Variables
COMPOSE_FILE=docker-compose.yml
MLFLOW_URL=http://localhost:5000
JENKINS_URL=http://localhost:8080/jenkins

# Default target
all: setup

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt

# Start Docker Compose services
up:
	@echo "Starting Docker Compose services..."
	docker-compose -f $(COMPOSE_FILE) up -d

# Stop Docker Compose services
down:
	@echo "Stopping Docker Compose services..."
	docker-compose -f $(COMPOSE_FILE) down

# Setup: Install dependencies and start services
setup: install up
	@echo "Setup complete. MLflow at $(MLFLOW_URL), Jenkins at $(JENKINS_URL)"

# Simulate webhook trigger
trigger-webhook:
	@echo "Triggering webhook..."
	python scripts/webhook_trigger.py

# Run drift monitoring
monitor-drift:
	@echo "Running drift monitoring..."
	python scripts/monitor_drift.py

# Clean up
clean:
	@echo "Cleaning up..."
	docker-compose -f $(COMPOSE_FILE) down -v
	rm -rf __pycache__

train-model:
    @echo "Training model..."
    python scripts/train_model.py

.PHONY: all install up down setup trigger-webhook monitor-drift clean