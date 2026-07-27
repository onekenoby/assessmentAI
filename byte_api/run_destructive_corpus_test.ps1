$ErrorActionPreference = "Stop"
if ($env:BYTE_API_RUN_DESTRUCTIVE -ne "1") {
    throw "Impostare BYTE_API_RUN_DESTRUCTIVE=1. Il test inserisce dati reali nel Database A."
}
python -m pytest tests/integration/test_destructive_uploads.py::test_real_corpus_upload_creates_pending_job -m "integration and destructive" -v
