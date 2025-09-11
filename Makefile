# GameLens AI Makefile

.PHONY: help install db-upgrade db-downgrade seed test clean

help: ## Show this help message
	@echo "GameLens AI - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

db-upgrade: ## Run database migrations
	alembic upgrade head

db-downgrade: ## Rollback database migrations
	alembic downgrade -1

db-revision: ## Create new migration
	alembic revision --autogenerate -m "$(MSG)"

seed: ## Seed database with demo data (optional)
	@echo "Seeding database with demo data..."
	@python -c "from glai.db import init_database; init_database()"

test: ## Run tests
	python -m pytest tests/ -v

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf build/
	rm -rf dist/

setup: install db-upgrade ## Complete setup (install + migrate)
	@echo "✅ GameLens AI setup complete!"

dev: ## Start development server
	streamlit run pages/2_🚀_Train_Predict_Validate_FAQ.py

# Database management
db-reset: ## Reset database (WARNING: This will delete all data!)
	rm -f gamelens.db
	alembic upgrade head

db-status: ## Show database migration status
	alembic current
	alembic history

# Model management
train-demo: ## Train a demo model
	@echo "Training demo model..."
	@python -c "from glai.train import train_lgbm_quantile; print('Demo training complete')"

# Data management
ingest-demo: ## Ingest demo data
	@echo "Ingesting demo data..."
	@python -c "from glai.ingest import ingest_file; print('Demo ingestion complete')"
