# Batch 13 — End-to-End Alignment Report

**Overall result:** `PASS`

- Generated at: `2026-07-15T12:47:07.802388+00:00`
- API base URL: `http://127.0.0.1:8013`
- API started by runner: `true`
- Pytest exit code: `0`
- Tests: `189`
- Passed: `188`
- Failed: `0`
- Errors: `0`
- Skipped: `1`
- Duration: `301.23s`

## Alignment matrix

| Area | Evidence | Status |
|---|---|---|
| PostgreSQL schema, RLS and organization visibility | real service integration tests | covered |
| Qdrant collection, vector query and payload filtering | real service integration tests | covered |
| Neo4j connectivity, graph invariants and relationship whitelist | real service integration tests | covered |
| Ollama configured model and native /api/chat response | real service integration tests | covered |
| Live API health, readiness, deterministic and RAG query contracts | HTTP end-to-end tests | covered |
| Document/page scope and public source provenance | HTTP end-to-end tests with a real PostgreSQL chunk | covered |
| Cross-organization leakage checks | real-store isolation tests | required |
| Generation timeout/empty response fallback | Batch 10 deterministic fault simulation | covered |
| Bounded capacity and service_busy | Batch 12 unit tests | covered |

## Interpretation

A PASS result closes the static and executable alignment work through Batch 13. Skipped cross-organization tests are acceptable only when the stores contain ACCOUNT Tier B/C data for a single organization_id; use `--require-second-organization` to require a foreign organization_id for the isolation proof.

The report never records database passwords, API keys, bearer tokens, raw prompts or retrieved document content.
