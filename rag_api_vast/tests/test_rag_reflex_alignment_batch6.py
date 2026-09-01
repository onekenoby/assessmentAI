from __future__ import annotations

from hashlib import sha256
from types import SimpleNamespace

import pytest

from core.audit import AuditSink, AuditSinkOutcome, AuditWriteResult
from core.config import settings
from core.generation import GenerationMetrics, GenerationResult
from core.models import RagEvalResult, RetrievalDebug, SourceItem
from core.prompting import build_tier_context_blocks
from core.rag_service import (
    RagQueryCommand,
    RagService,
    _sources_in_prompt_context,
)


class FakeRetriever:
    def __init__(self, candidates):
        self.candidates = tuple(candidates)

    async def retrieve_candidates(self, **kwargs):
        return self.candidates, RetrievalDebug(
            qdrant_hits=len(self.candidates),
            kept_after_quality_filters=len(self.candidates),
        )

    async def lookup_glossary(self, **kwargs):
        return None


class CapturingGenerator:
    def __init__(self):
        self.prompt = None

    async def generate_async(self, prompt):
        self.prompt = prompt
        content = (
            "**A) Risposta**\n\nLa procedura è documentata.\n\n"
            "**B) Evidenze**\n\n- Evidenza da `first.pdf`.\n\n"
            "**C) Limiti / Conflitti**\n\n- Il contesto è limitato alle fonti inviate.\n\n"
            "**D) Fonti**\n\n- placeholder"
        )
        return GenerationResult(
            content=content,
            model="gemma4:12b",
            request_id="",
            attempts=1,
            elapsed_ms=5,
            response_sha256=sha256(content.encode()).hexdigest(),
            metrics=GenerationMetrics(),
        )


class CapturingAuditor:
    def __init__(self):
        self.last_audit = None

    async def persist_query_audit_async(self, audit, **kwargs):
        self.last_audit = audit
        return AuditWriteResult(
            request_id=str(audit.request_id),
            outcomes=(
                AuditSinkOutcome(
                    sink=AuditSink.QUERY_JSONL,
                    attempted=False,
                    success=True,
                    skipped=True,
                ),
            ),
        )


class CapturingEvaluator:
    def __init__(self):
        self.sources = ()

    async def evaluate_async(self, **kwargs):
        self.sources = tuple(kwargs["sources"])
        return RagEvalResult.disabled()


def _source(identifier: str, *, page: int, content: str) -> SourceItem:
    return SourceItem(
        id=identifier,
        content=content,
        filename=f"{identifier}.pdf",
        page=page,
        page_chunk_index=0,
        doc_id=f"doc-{identifier}",
        type="text",
        tier="C",
        scope="ACCOUNT",
        organization_id=1234,
        status="active",
        corpus_version="v1",
        db_origin="Qdrant",
    )


def test_prompt_context_tracks_stable_dedupe_keys(tenant_context):
    sources = (
        _source("first", page=1, content="A" * 700),
        _source("second", page=2, content="B" * 700),
        _source("third", page=3, content="C" * 700),
    )

    context = build_tier_context_blocks(
        sources,
        max_chars=900,
        context=tenant_context,
    )

    assert context.source_count == 1
    assert context.dropped_sources == 2
    assert context.included_source_ids == ("first",)
    assert context.included_source_keys == (sources[0].dedupe_key,)


def test_prompt_source_selection_uses_dedupe_key_and_prompt_order():
    first = _source("shared", page=1, content="Prima versione")
    same_id_other_chunk = SourceItem(
        **{
            **first.model_dump(exclude={"dedupe_key"}),
            "content": "Stesso ID, pagina differente",
            "page": 2,
            "doc_id": "doc-other",
        }
    )

    selected = _sources_in_prompt_context(
        (first, same_id_other_chunk),
        (same_id_other_chunk.dedupe_key, first.dedupe_key),
    )

    assert [(source.doc_id, source.page) for source in selected] == [
        ("doc-other", 2),
        ("doc-shared", 1),
    ]


@pytest.mark.asyncio
async def test_generated_response_exposes_only_sources_actually_sent_to_llm(
    tenant_context,
    candidate_factory,
):
    long_text = "Contenuto documentale rilevante. " * 40
    candidates = (
        candidate_factory(
            "first",
            filename="first.pdf",
            page=1,
            score_vec=0.99,
            content=long_text + "FIRST",
        ),
        candidate_factory(
            "second",
            filename="second.pdf",
            page=2,
            score_vec=0.80,
            content=long_text + "SECOND",
        ),
        candidate_factory(
            "third",
            filename="third.pdf",
            page=3,
            score_vec=0.70,
            content=long_text + "THIRD",
        ),
    )

    config = settings.model_copy(
        update={
            "max_context_chars": 900,
            "audit_enabled": False,
            "evaluation_enabled": False,
        }
    )
    generator = CapturingGenerator()
    auditor = CapturingAuditor()
    service = RagService(
        config=config,
        resource_manager=SimpleNamespace(get_reranker=lambda: None),
        retriever=FakeRetriever(candidates),
        llm_generator=generator,
        auditor=auditor,
        evaluator=CapturingEvaluator(),
    )

    result = await service.query(
        RagQueryCommand(query="Descrivi la procedura documentata."),
        tenant_context=tenant_context,
    )

    assert generator.prompt is not None
    assert generator.prompt.context.source_count == 1
    assert generator.prompt.context.included_source_ids == ("first",)
    assert "first.pdf" in generator.prompt.user_content
    assert "second.pdf" not in generator.prompt.user_content
    assert "third.pdf" not in generator.prompt.user_content

    assert [source.filename for source in result.sources] == ["first.pdf"]
    assert "first.pdf" in result.answer
    assert "second.pdf" not in result.answer
    assert "third.pdf" not in result.answer

    assert result.retrieval.reranked_sources == 3
    assert result.retrieval.final_sources == 1
    assert result.retrieval.prompt_context_sources == 1
    assert result.retrieval.prompt_dropped_sources == 2

    assert auditor.last_audit is not None
    assert [source.filename for source in auditor.last_audit.retrieved_sources] == [
        "first.pdf",
        "second.pdf",
        "third.pdf",
    ]
    assert "Fonti dopo reranking/diversificazione: **3**" in result.audit_markdown
    assert "Fonti pubbliche finali: **1**" in result.audit_markdown
    assert "Fonti effettivamente inserite nel prompt: **1**" in result.audit_markdown
    assert "Fonti escluse dal prompt per filtri/budget: **2**" in result.audit_markdown


@pytest.mark.asyncio
async def test_evaluation_receives_prompt_sources_not_full_retrieval(
    tenant_context,
    candidate_factory,
    monkeypatch,
):
    long_text = "Evidenza tecnica ripetuta. " * 45
    candidates = (
        candidate_factory(
            "first",
            filename="first.pdf",
            page=1,
            score_vec=0.99,
            content=long_text + "FIRST",
        ),
        candidate_factory(
            "second",
            filename="second.pdf",
            page=2,
            score_vec=0.80,
            content=long_text + "SECOND",
        ),
    )

    config = settings.model_copy(
        update={
            "max_context_chars": 900,
            "audit_enabled": False,
            "evaluation_enabled": False,
        }
    )
    evaluator = CapturingEvaluator()
    evaluation_audit_sources = []

    async def fake_eval_audit(**kwargs):
        evaluation_audit_sources.extend(kwargs["sources"])
        return AuditWriteResult(
            request_id=tenant_context.request_id,
            outcomes=(
                AuditSinkOutcome(
                    sink=AuditSink.EVALUATION_JSONL,
                    attempted=False,
                    success=True,
                    skipped=True,
                ),
            ),
        )

    monkeypatch.setattr(
        "core.rag_service.append_rag_eval_log_async",
        fake_eval_audit,
    )

    service = RagService(
        config=config,
        resource_manager=SimpleNamespace(get_reranker=lambda: None),
        retriever=FakeRetriever(candidates),
        llm_generator=CapturingGenerator(),
        auditor=CapturingAuditor(),
        evaluator=evaluator,
    )

    result = await service.query(
        RagQueryCommand(
            query="Valuta la procedura documentata.",
            include_evaluation=True,
        ),
        tenant_context=tenant_context,
    )

    assert [source.filename for source in evaluator.sources] == ["first.pdf"]
    assert [source.filename for source in evaluation_audit_sources] == ["first.pdf"]
    assert [source.filename for source in result.sources] == ["first.pdf"]
