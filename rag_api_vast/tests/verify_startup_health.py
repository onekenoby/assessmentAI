#!/usr/bin/env python3
"""Avvia realmente FastAPI/Uvicorn e verifica gli endpoint health.

Il test esegue il lifespan di ``main:app``: vengono quindi inizializzati Ollama,
Qdrant, Neo4j, PostgreSQL, embedder e reranker secondo la configurazione
corrente. Al termine il processo Uvicorn viene arrestato.

Esempi::

    python tests/verify_startup_health.py
    python tests/verify_startup_health.py --startup-timeout 1200
    python tests/verify_startup_health.py --allow-not-ready

Exit code:
    0: liveness OK e readiness conforme all'aspettativa;
    1: startup, health check o shutdown non riusciti.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_FILE = PROJECT_ROOT / "tests" / "startup_health_uvicorn.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avvia Uvicorn e verifica /health/live e /health/ready.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=900.0,
        help="Secondi massimi concessi al lifespan e al caricamento modelli.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=20.0,
        help="Timeout delle singole richieste health.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Intervallo tra i tentativi di liveness.",
    )
    parser.add_argument(
        "--allow-not-ready",
        action="store_true",
        help=(
            "Considera valido anche readiness=503. Utile per verificare soltanto "
            "che il processo resti vivo in POC dopo uno startup degradato/fallito."
        ),
    )
    parser.add_argument(
        "--skip-deep",
        action="store_true",
        help="Non esegue /health/ready?deep=true.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
    )
    return parser.parse_args()


def port_is_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def format_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def read_json_response(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"raw_body": response.text[:4000]}


def tail_file(path: Path, lines: int = 120) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"Impossibile leggere il log: {exc}"
    return "\n".join(content[-lines:])


def print_effective_configuration_warnings() -> None:
    """Mostra soltanto warning non sensibili prima dello startup."""

    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from core.config import settings
    finally:
        try:
            sys.path.remove(str(PROJECT_ROOT))
        except ValueError:
            pass

    print("Configurazione effettiva non sensibile:")
    print(f"  POC_MODE={settings.poc_mode}")
    print(f"  Ollama={settings.ollama_native_chat_url}")
    print(f"  Modello LLM={settings.llm_model_name}")
    print(f"  Qdrant={settings.qdrant_host}:{settings.qdrant_port}")
    print(f"  Collection={settings.qdrant_collection}")
    print(f"  PostgreSQL={settings.pg_host}:{settings.pg_port}/{settings.pg_database}")
    print(f"  Neo4j={settings.neo4j_uri} | enabled={settings.neo4j_enabled}")
    print(f"  Embedder={settings.embedding_model_name}")
    print(f"  Reranker={settings.reranker_model_name}")

    if os.name == "nt":
        for label, raw_path in (
            ("EMBEDDING_MODEL_NAME", settings.embedding_model_name),
            ("RERANKER_MODEL_NAME", settings.reranker_model_name),
        ):
            if str(raw_path).replace("\\", "/").startswith("/workspace/"):
                print(
                    f"ATTENZIONE: {label} usa un percorso Docker ({raw_path}) "
                    "ma il test è eseguito su Windows. Impostare il percorso locale."
                )
            else:
                model_path = Path(str(raw_path))
                if model_path.is_absolute() and not model_path.exists():
                    print(f"ATTENZIONE: {label} non esiste: {model_path}")


def stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def main() -> int:
    args = parse_args()

    if args.port <= 0 or args.port > 65535:
        print("ERRORE: porta non valida", file=sys.stderr)
        return 1

    if port_is_in_use(args.host, args.port):
        print(
            f"ERRORE: {args.host}:{args.port} è già in uso. "
            "Arrestare il processo esistente o scegliere --port.",
            file=sys.stderr,
        )
        return 1

    print("\nRAG API - startup FastAPI e health check")
    print("=" * 78)
    print_effective_configuration_warnings()

    args.log_file = args.log_file.resolve()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "main:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--workers",
        "1",
        "--log-level",
        "info",
    ]

    print("\nComando:")
    print("  " + " ".join(command))
    print(f"Log: {args.log_file}")

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    with args.log_file.open("w", encoding="utf-8", buffering=1) as log_stream:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )

        base_url = f"http://{args.host}:{args.port}"
        live_response: requests.Response | None = None
        deadline = time.monotonic() + args.startup_timeout
        last_error = ""

        try:
            while time.monotonic() < deadline:
                return_code = process.poll()
                if return_code is not None:
                    print(f"\nERRORE: Uvicorn terminato con exit code {return_code}.")
                    print("\nUltime righe del log:")
                    print(tail_file(args.log_file))
                    return 1

                try:
                    response = requests.get(
                        f"{base_url}/health/live",
                        timeout=min(args.request_timeout, 5.0),
                        headers={"Accept": "application/json"},
                    )
                    if response.status_code == 200:
                        live_response = response
                        break
                    last_error = f"HTTP {response.status_code}"
                except requests.RequestException as exc:
                    last_error = str(exc)

                time.sleep(max(0.1, args.poll_interval))

            if live_response is None:
                print(
                    "\nERRORE: liveness non raggiungibile entro "
                    f"{args.startup_timeout:g} secondi. Ultimo errore: {last_error}"
                )
                print("\nUltime righe del log:")
                print(tail_file(args.log_file))
                return 1

            live_payload = read_json_response(live_response)
            print("\n[1] GET /health/live")
            print(f"HTTP {live_response.status_code}")
            print(format_json(live_payload))

            ready_response = requests.get(
                f"{base_url}/health/ready",
                timeout=args.request_timeout,
                headers={"Accept": "application/json"},
            )
            ready_payload = read_json_response(ready_response)
            print("\n[2] GET /health/ready")
            print(f"HTTP {ready_response.status_code}")
            print(format_json(ready_payload))

            deep_response: requests.Response | None = None
            deep_payload: Any = None
            if not args.skip_deep:
                deep_response = requests.get(
                    f"{base_url}/health/ready?deep=true",
                    timeout=max(args.request_timeout, 60.0),
                    headers={"Accept": "application/json"},
                )
                deep_payload = read_json_response(deep_response)
                print("\n[3] GET /health/ready?deep=true")
                print(f"HTTP {deep_response.status_code}")
                print(format_json(deep_payload))

            live_ok = live_response.status_code == 200 and live_payload.get("status") == "ok"
            ready_ok = ready_response.status_code == 200 and ready_payload.get("status") in {
                "ok",
                "degraded",
            }
            deep_ok = True
            if deep_response is not None:
                deep_ok = deep_response.status_code == 200 and deep_payload.get("status") in {
                    "ok",
                    "degraded",
                }

            if args.allow_not_ready:
                overall_ok = live_ok and ready_response.status_code in {200, 503}
                if deep_response is not None:
                    overall_ok = overall_ok and deep_response.status_code in {200, 503}
            else:
                overall_ok = live_ok and ready_ok and deep_ok

            print("\n" + "-" * 78)
            if overall_ok:
                print("ESITO: OK")
                if ready_response.status_code == 503:
                    print(
                        "Nota: il processo è vivo ma non ready; risultato accettato "
                        "perché è stato usato --allow-not-ready."
                    )
                return 0

            print("ESITO: FAIL")
            print("Il processo è vivo, ma una o più dipendenze obbligatorie non sono ready.")
            print("\nUltime righe del log Uvicorn:")
            print(tail_file(args.log_file))
            return 1

        finally:
            stop_process(process)


if __name__ == "__main__":
    raise SystemExit(main())
