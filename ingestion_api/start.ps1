$env:PYTHONUNBUFFERED = "1"

# Attiva il producer/consumer interno dei documenti.
$env:USE_PRODUCER_CONSUMER = "1"

# Un solo documento preparato in anticipo.
# È il valore originale e più sicuro per RAM, VRAM e file temporanei.
$env:DOC_QUEUE_MAXSIZE = "1"

python -u -m uvicorn main:app `
    --host 0.0.0.0 `
    --port 8010 `
    --workers 1 `
    --log-level info `
    --access-log