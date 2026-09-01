from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from core.models import (
    AuditTrail,
    EvaluationVerdict,
    RagEvalResult,
    RagServiceResult,
    RetrievalCandidate,
    RetrievalDebug,
    ScoreStats,
    SourceItem,
)


def test_source_item_normalizes_type_status_and_has_stable_excerpt() -> None:
    source = SourceItem(
        id="s1",
        content="x" * 50,
        filename="evidence.pdf",
        tier="c",
        scope="account",
        organization_id=1234,
        type="tabella",
        status=" ACTIVE ",
    )

    assert source.tier == "C"
    assert source.scope == "ACCOUNT"
    assert source.type == "table"
    assert source.status == "active"
    assert source.excerpt(10) == "xxxxxxxxxx..."
    assert source.audit_reference()["organization_id"] == 1234


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tier": "A", "scope": "ACCOUNT", "organization_id": 1234},
        {"tier": "B", "scope": "GLOBAL", "organization_id": None},
        {"tier": "C", "scope": "ACCOUNT", "organization_id": None},
    ],
)
def test_source_item_rejects_invalid_tenant_provenance(kwargs) -> None:
    with pytest.raises(ValidationError):
        SourceItem(
            id="invalid",
            content="contenuto",
            filename="invalid.pdf",
            **kwargs,
        )


def test_retrieval_candidate_uses_most_advanced_score_and_converts() -> None:
    candidate = RetrievalCandidate(
        id="c1",
        content="contenuto",
        filename="doc.pdf",
        tier="C",
        scope="ACCOUNT",
        organization_id=1234,
        origin="unit-test",
        score_vec=0.3,
        score_rrf=0.1,
        score_rerank=1.2,
        final_score=1.8,
    )

    assert candidate.effective_score == pytest.approx(1.8)

    request_id = str(uuid4())
    source = candidate.to_source_item(request_id=request_id)
    assert source.score == pytest.approx(1.8)
    assert source.db_origin == "unit-test"
    assert source.request_id == request_id


def test_score_stats_calculates_min_max_average() -> None:
    stats = ScoreStats.from_values([0.5, 1.0, 1.5])
    assert stats.minimum == pytest.approx(0.5)
    assert stats.maximum == pytest.approx(1.5)
    assert stats.average == pytest.approx(1.0)


def test_audit_trail_hashes_query_and_does_not_persist_plain_text(source_c) -> None:
    request_id = uuid4()
    audit = AuditTrail.from_sources(
        organization_id=1234,
        user_id="unit-test-user",
        roles=("auditor",),
        request_id=request_id,
        query="Testo sensibile della query",
        sources=(source_c,),
    )

    assert len(audit.query_sha256) == 64
    assert audit.query == "Testo sensibile della query"
    payload = audit.persistent_payload()
    assert payload["query"] == ""
    assert payload["query_sha256"] == audit.query_sha256
    assert payload["retrieved_sources"][0]["filename"] == "evidenza_test.pdf"


def test_eval_result_resolves_verdict_and_factory_methods() -> None:
    result = RagEvalResult(
        faithfulness=0.9,
        answer_relevance=0.8,
        context_support=0.9,
        hallucination_risk=0.1,
    )
    assert result.resolve_verdict(
        minimum_faithfulness=0.75,
        minimum_answer_relevance=0.7,
    ) == EvaluationVerdict.PASS

    assert RagEvalResult.disabled().verdict == EvaluationVerdict.DISABLED
    assert RagEvalResult.error("judge unavailable").verdict == EvaluationVerdict.ERROR


def test_service_result_aligns_retrieval_metrics_and_deduplicates_warnings(source_c) -> None:
    debug = RetrievalDebug()
    result = RagServiceResult(
        answer="Risposta valida",
        sources=(source_c,),
        retrieval=debug,
        warnings=("warning", "warning", "altro"),
    )

    assert result.retrieval.final_sources == 1
    assert result.warnings == ("warning", "altro")
