from __future__ import annotations

import pytest

from core.models import RagIntent
from core.reranking import (
    RankingContext,
    RerankingEngine,
    apply_rrf_scoring,
    diversify_candidates,
    tier_score_delta,
)


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = []

    def predict(self, sentences, **kwargs):
        self.calls.append((list(sentences), dict(kwargs)))
        return self.scores[: len(sentences)]


def test_rrf_combines_vector_bm25_and_graph_rankings(candidate_factory) -> None:
    vector_top = candidate_factory("v", score_vec=0.9)
    bm25_top = candidate_factory("b", score_bm25=5.0)
    graph_top = candidate_factory("g", score_graph=3.0)

    candidates = [vector_top, bm25_top, graph_top]
    apply_rrf_scoring(candidates, k=60)

    assert all(candidate.score_rrf > 0 for candidate in candidates)
    assert vector_top.score_rrf == pytest.approx(bm25_top.score_rrf)
    assert bm25_top.score_rrf == pytest.approx(graph_top.score_rrf)


def test_tier_policy_applies_expected_delta() -> None:
    assert tier_score_delta("A", wants_evidence=False) > 0
    assert tier_score_delta("B", wants_evidence=False) > 0
    assert tier_score_delta("C", wants_evidence=False) < 0
    assert tier_score_delta("C", wants_evidence=True) == pytest.approx(0.0)
    assert tier_score_delta("GRAPH", wants_evidence=True) == pytest.approx(0.0)


def test_diversification_limits_same_page_and_document(candidate_factory) -> None:
    candidates = [
        candidate_factory(
            f"c{i}",
            filename="same.pdf",
            page=1 if i < 3 else 2,
            page_chunk_index=i,
            doc_id="same-doc",
            final_score=10 - i,
        )
        for i in range(5)
    ]

    selected = diversify_candidates(
        candidates,
        max_per_page=1,
        max_per_document=2,
        final_k=5,
    )

    assert len(selected) == 2
    assert {item.page for item in selected} == {1, 2}


def test_engine_uses_cross_encoder_and_does_not_mutate_inputs(candidate_factory) -> None:
    candidates = [
        candidate_factory(
            "a",
            filename="evidenza.pdf",
            page=1,
            score_vec=0.9,
            content="procedura formalizzata",
        ),
        candidate_factory(
            "b",
            filename="altro.pdf",
            page=2,
            score_bm25=4.0,
            content="contenuto secondario",
        ),
    ]
    original_scores = [candidate.final_score for candidate in candidates]
    fake = FakeCrossEncoder([0.2, 1.5])
    engine = RerankingEngine(reranker=fake)

    result = engine.rank(
        candidates,
        context=RankingContext(
            query_text="procedura formalizzata",
            intent=RagIntent.AUDIT,
            wants_evidence=True,
            target_document="evidenza.pdf",
            requested_pages=(1,),
        ),
        final_k=2,
        max_per_page=2,
        max_per_document=2,
    )

    assert result.reranker_used is True
    assert result.final_count == 2
    assert len(fake.calls) == 1
    assert all("ranking_components" in item.metadata for item in result.candidates)
    assert [candidate.final_score for candidate in candidates] == original_scores


def test_engine_falls_back_when_reranker_is_missing(candidate_factory) -> None:
    engine = RerankingEngine(reranker=None)
    result = engine.rank(
        [candidate_factory("a", score_vec=0.8)],
        context=RankingContext(query_text="test"),
        final_k=1,
    )
    assert result.reranker_used is False
    assert result.final_count == 1
