"""Batch 13 final verifier for the Multi-Tenant Hybrid-RAG API.

The runner executes the cumulative simulated regression and the non-destructive
real-service/API tests, optionally starting Uvicorn for the HTTP phase.  A
sanitized Markdown and JSON report is always produced.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

from e2e_report import write_batch13_report


PROJECT_ROOT = Path(__file__).resolve().parent

UNIT_FILES = (
    "tests/test_rag_reflex_alignment_batch13_5.py",
    "tests/test_rag_reflex_alignment_batch13_4.py",
    "tests/test_rag_reflex_alignment_batch13_3.py",
    "tests/test_rag_reflex_alignment_batch12.py",
    "tests/test_rag_reflex_alignment_batch11.py",
    "tests/test_rag_reflex_alignment_batch10.py",
    "tests/test_rag_reflex_alignment_batch9.py",
    "tests/test_rag_reflex_alignment_batch8.py",
    "tests/test_rag_reflex_alignment_batch7.py",
    "tests/test_rag_reflex_alignment_batch6.py",
    "tests/test_rag_reflex_alignment_batch5.py",
    "tests/test_retrieval_reflex_alignment_batch4.py",
    "tests/test_retrieval_reflex_alignment_batch3.py",
    "tests/test_retrieval_reflex_alignment_batch2.py",
    "tests/test_rag_reflex_alignment.py",
    "tests/test_retrieval_simulated.py",
    "tests/test_rag_service_simulated.py",
    "tests/test_validation.py",
    "tests/test_reranking.py",
    "tests/test_models.py",
    "tests/test_prompting.py",
    "tests/test_generation_simulated.py",
    "tests/test_routes_rag_simulated.py",
    "tests/test_resources_simulated.py",
    "tests/test_batch13_e2e_support.py",
)

REAL_FILES = (
    "tests/integration/test_postgres_real.py",
    "tests/integration/test_qdrant_real.py",
    "tests/integration/test_neo4j_real.py",
    "tests/integration/test_ollama_real.py",
    "tests/integration/test_organization_isolation_real.py",
    "tests/integration/test_rag_api_e2e_real.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Esegue il collaudo finale Batch 13 e produce il report di allineamento."
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000"),
        help="URL base della RAG API live.",
    )
    parser.add_argument(
        "--start-api",
        action="store_true",
        help="Avvia Uvicorn automaticamente per il collaudo HTTP.",
    )
    parser.add_argument(
        "--api-start-timeout",
        type=int,
        default=180,
        help="Secondi massimi per attendere /health/live.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout di rete per singola operazione reale.",
    )
    parser.add_argument(
        "--api-llm-timeout",
        type=int,
        default=180,
        help=(
            "Timeout della singola chiamata Ollama nell'API avviata dal runner. "
            "Deve essere inferiore al timeout HTTP del collaudo per consentire il fallback."
        ),
    )
    parser.add_argument(
        "--api-llm-num-predict",
        type=int,
        default=512,
        help="Massimo numero di token generati dall'API durante il collaudo E2E.",
    )
    parser.add_argument(
        "--api-llm-max-attempts",
        type=int,
        default=1,
        help="Tentativi Ollama per query nell'API avviata dal runner.",
    )
    parser.add_argument(
        "--allow-empty-data",
        action="store_true",
        help="Consente store reali raggiungibili ma senza dati visibili.",
    )
    parser.add_argument(
        "--require-second-organization",
        action="store_true",
        help="Fallisce se non esistono dati ACCOUNT di almeno una seconda organization_id.",
    )
    parser.add_argument(
        "--capacity-stress",
        action="store_true",
        help="Abilita anche il test HTTP concorrente service_busy.",
    )
    parser.add_argument(
        "--skip-unit",
        action="store_true",
        help="Esegue soltanto i test reali/API.",
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports" / "batch13"),
        help="Directory dei report JSON/Markdown/JUnit.",
    )
    parser.add_argument("--quiet", action="store_true", help="Output pytest compatto.")
    return parser.parse_args()


def _health_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/health/live"


def _readiness_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/health/ready?deep=true"


def _api_is_live(base_url: str, timeout_seconds: float = 2.0) -> bool:
    try:
        import requests

        response = requests.get(_health_url(base_url), timeout=timeout_seconds)
        return response.status_code == 200
    except Exception:
        return False


def _api_readiness(
    base_url: str,
    timeout_seconds: float = 5.0,
) -> tuple[bool, str]:
    """Restituisce readiness e diagnostica pubblica sanitizzata."""

    try:
        import requests

        response = requests.get(
            _readiness_url(base_url),
            timeout=timeout_seconds,
        )
        detail = f"HTTP {response.status_code}: {response.text[:2000]}"
        return response.status_code == 200, detail
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:500]}"


def _resolve_api_launch_plan(
    *,
    start_requested: bool,
    api_live: bool,
    base_url: str,
) -> str:
    """Decide se avviare o riutilizzare l'API senza ambiguità."""

    if start_requested:
        if api_live:
            raise RuntimeError(
                f"La porta di collaudo è già occupata da un'API live su {base_url}. "
                "Il runner non riutilizza più processi preesistenti quando è richiesto "
                "--start-api. Arrestare il processo, scegliere un'altra porta oppure "
                "usare -NoStartApi/omettere --start-api per collaudare deliberatamente "
                "l'istanza esistente."
            )
        return "start"

    if not api_live:
        raise RuntimeError(
            f"RAG API non raggiungibile su {base_url}. "
            "Avviare Uvicorn oppure usare --start-api."
        )
    return "reuse"


def _bounded_e2e_generation_environment(
    env: dict[str, str],
    *,
    network_timeout_seconds: int,
    llm_timeout_seconds: int,
    llm_num_predict: int,
    llm_max_attempts: int,
) -> dict[str, str]:
    """Costruisce il profilo Ollama bounded usato dall'API di collaudo.

    Il timeout Ollama deve terminare prima del timeout HTTP del client pytest,
    altrimenti il fallback Batch 10 non può essere osservato dal collaudo.
    """

    network_timeout = max(5, int(network_timeout_seconds))
    requested_llm_timeout = max(5, int(llm_timeout_seconds))
    safety_margin = min(30, max(5, network_timeout // 10))
    bounded_timeout = min(requested_llm_timeout, max(5, network_timeout - safety_margin))

    attempts = int(llm_max_attempts)
    if attempts < 1 or attempts > 5:
        raise ValueError("api_llm_max_attempts deve essere compreso tra 1 e 5")

    num_predict = int(llm_num_predict)
    if num_predict <= 0:
        raise ValueError("api_llm_num_predict deve essere maggiore di zero")

    child_env = dict(env)
    child_env["LLM_TIMEOUT_S"] = str(bounded_timeout)
    child_env["LLM_NUM_PREDICT"] = str(num_predict)
    child_env["LLM_MAX_ATTEMPTS"] = str(attempts)
    return child_env


def _start_api(
    *,
    base_url: str,
    env: dict[str, str],
    report_dir: Path,
    capacity_stress: bool,
    network_timeout_seconds: int,
    llm_timeout_seconds: int,
    llm_num_predict: int,
    llm_max_attempts: int,
) -> tuple[subprocess.Popen[str], object]:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"base URL non valido: {base_url!r}")
    if parsed.scheme == "https":
        raise ValueError("--start-api supporta soltanto URL http locali")
    if parsed.path not in {"", "/"}:
        raise ValueError("--start-api richiede un base URL senza path")

    host = parsed.hostname
    port = parsed.port or 8000
    if host not in {"127.0.0.1", "localhost", "0.0.0.0"}:
        raise ValueError("--start-api può avviare soltanto un endpoint locale")

    child_env = _bounded_e2e_generation_environment(
        env,
        network_timeout_seconds=network_timeout_seconds,
        llm_timeout_seconds=llm_timeout_seconds,
        llm_num_predict=llm_num_predict,
        llm_max_attempts=llm_max_attempts,
    )
    if capacity_stress:
        # Lo stress deve osservare un 200/fallback e due 503 senza attendere
        # l'intero timeout generativo standard del collaudo.
        child_env["LLM_TIMEOUT_S"] = str(
            min(int(child_env["LLM_TIMEOUT_S"]), 60)
        )
        child_env["RAG_MAX_CONCURRENT_QUERIES"] = "1"
        child_env["RAG_MAX_QUEUED_QUERIES"] = "0"
        child_env["RAG_QUERY_QUEUE_TIMEOUT_S"] = "1"

    report_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (report_dir / "uvicorn_batch13.log").open(
        "w", encoding="utf-8", errors="replace"
    )
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "main:app",
        "--host",
        "127.0.0.1" if host in {"localhost", "0.0.0.0"} else host,
        "--port",
        str(port),
        "--log-level",
        "info",
    ]
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=child_env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, log_handle


def _wait_for_api(
    *,
    process: subprocess.Popen[str],
    base_url: str,
    timeout_seconds: int,
) -> None:
    """Attende liveness e deep readiness dell'istanza appena avviata."""

    deadline = time.monotonic() + max(5, timeout_seconds)
    last_readiness = "readiness non ancora interrogata"

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Uvicorn terminato durante l'avvio con codice {process.returncode}"
            )

        if not _api_is_live(base_url):
            time.sleep(1.0)
            continue

        ready, last_readiness = _api_readiness(base_url)
        if ready:
            return

        # Il lifespan FastAPI termina prima che /health/live possa rispondere.
        # Una liveness disponibile con deep readiness ancora down indica quindi
        # un'inizializzazione fallita o un processo non conforme al collaudo.
        raise RuntimeError(
            "La RAG API risponde alla liveness ma non è deep-ready. "
            f"Dettaglio: {last_readiness}"
        )

    raise TimeoutError(
        f"La RAG API non è diventata pronta su {_readiness_url(base_url)} "
        f"entro {timeout_seconds}s. Ultimo dettaglio: {last_readiness}"
    )


def _stop_api(process: subprocess.Popen[str] | None, log_handle: object | None) -> None:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    if log_handle is not None:
        try:
            log_handle.close()  # type: ignore[attr-defined]
        except Exception:
            pass


def _pytest_command(
    *,
    junit_path: Path,
    quiet: bool,
    skip_unit: bool,
) -> list[str]:
    files: Sequence[str] = REAL_FILES if skip_unit else (*UNIT_FILES, *REAL_FILES)
    return [
        sys.executable,
        "-m",
        "pytest",
        "-q" if quiet else "-v",
        "-s",
        f"--junitxml={junit_path}",
        *files,
    ]


def main() -> int:
    args = parse_args()
    base_url = str(args.base_url).rstrip("/")
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    junit_path = report_dir / "batch13_junit.xml"

    env = os.environ.copy()
    env["RUN_REAL_SERVICE_TESTS"] = "1"
    env["RAG_INTEGRATION_REQUIRE_DATA"] = "0" if args.allow_empty_data else "1"
    env["RAG_INTEGRATION_TIMEOUT_S"] = str(max(5, int(args.timeout)))
    env["RAG_API_BASE_URL"] = base_url
    env["RAG_E2E_REQUIRE_SECOND_ORGANIZATION"] = "1" if args.require_second_organization else "0"
    env["RAG_E2E_RUN_CAPACITY_STRESS"] = "1" if args.capacity_stress else "0"

    process: subprocess.Popen[str] | None = None
    log_handle: object | None = None
    started_api = False
    exit_code = 1
    command: list[str] = []

    try:
        api_live = _api_is_live(base_url)
        launch_plan = _resolve_api_launch_plan(
            start_requested=bool(args.start_api),
            api_live=api_live,
            base_url=base_url,
        )

        if launch_plan == "start":
            process, log_handle = _start_api(
                base_url=base_url,
                env=env,
                report_dir=report_dir,
                capacity_stress=bool(args.capacity_stress),
                network_timeout_seconds=max(5, int(args.timeout)),
                llm_timeout_seconds=max(5, int(args.api_llm_timeout)),
                llm_num_predict=max(1, int(args.api_llm_num_predict)),
                llm_max_attempts=int(args.api_llm_max_attempts),
            )
            started_api = True
            _wait_for_api(
                process=process,
                base_url=base_url,
                timeout_seconds=max(5, int(args.api_start_timeout)),
            )
        else:
            ready, readiness_detail = _api_readiness(base_url)
            if not ready:
                raise RuntimeError(
                    "L'istanza API preesistente è live ma non deep-ready. "
                    f"Dettaglio: {readiness_detail}"
                )

        command = _pytest_command(
            junit_path=junit_path,
            quiet=bool(args.quiet),
            skip_unit=bool(args.skip_unit),
        )
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            check=False,
        )
        exit_code = int(completed.returncode)
    except Exception as exc:
        print(f"Batch 13 runner error: {type(exc).__name__}: {exc}", file=sys.stderr)
        exit_code = 2
    finally:
        _stop_api(process, log_handle)
        json_path, markdown_path = write_batch13_report(
            report_dir=report_dir,
            junit_xml=junit_path,
            command=command,
            exit_code=exit_code,
            base_url=base_url,
            started_api=started_api,
            capacity_stress=bool(args.capacity_stress),
            second_organization_required=bool(args.require_second_organization),
            environ=env,
        )
        print(f"Batch 13 JSON report: {json_path}")
        print(f"Batch 13 Markdown report: {markdown_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
