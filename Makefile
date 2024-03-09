dev:
	pip install uv
	uv pip install -r requirements.txt
	uv pip install -r requirements-dev.txt

files:
	pre-commit run --all-files
