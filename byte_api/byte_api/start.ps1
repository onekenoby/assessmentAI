$ErrorActionPreference = "Stop"
python -m uvicorn main:app --host 0.0.0.0 --port 8020 --workers 1
