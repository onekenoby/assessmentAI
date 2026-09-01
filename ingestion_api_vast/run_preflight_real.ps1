$ErrorActionPreference = "Stop"
$env:RUN_REAL_SERVICE_TESTS = "1"
python -m pytest tests/integration/test_real_dependencies.py -m integration -v
