$ErrorActionPreference = "Stop"
python -m pytest -m "not integration" --cov=api --cov=core --cov=main --cov-report=term-missing
