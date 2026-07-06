from __future__ import annotations

import pytest

from core.solvers import (
    SolverName,
    calcolatrice_universale,
    is_calculation_request,
    solve_deterministic_query,
    try_solve_math_query,
)


@pytest.mark.parametrize(
    ("query", "solver_name", "expected_fragment"),
    [
        (
            "Calcola la copertura di una checklist di 20 controlli: 10 implementati e 4 parziali che valgono al 50%.",
            SolverName.CONTROL_COVERAGE,
            "60.00%",
        ),
        (
            "Ordina il rischio degli scenari A 3 x 5 e B 2 x 9 usando probabilità per impatto.",
            SolverName.RISK_PRODUCT,
            "B=18",
        ),
        (
            "Un budget totale di 250.000 euro è allocato al 40% e al 35%. Calcola il restante.",
            SolverName.PERCENTAGE_REMAINDER,
            "62.500,00 euro",
        ),
        (
            "Calcola il tempo cumulativo: 2 incidenti critici, 3 ore per critici; 4 incidenti alti, 10 ore per alti.",
            SolverName.SLA_CUMULATIVE_HOURS,
            "46 ore",
        ),
        (
            "Calcola il ROSI: impatto economico 100.000 euro, probabilità 0,20 che scende a 0,05 e costo annuo 5.000 euro.",
            SolverName.ROSI,
            "200.00%",
        ),
        (
            "Se il rischio inerente Ri = T × V e il rischio residuo Rr <= Ri / 3, esprimi Vm in funzione di V.",
            SolverName.USER_ALGEBRA,
            "Vm ≤ V / 3",
        ),
        (
            "Calcola la scadenza partendo da lunedì ore 08:00 aggiungendo 24 ore.",
            SolverName.DATE_OFFSETS,
            "martedì",
        ),
    ],
)
def test_deterministic_solver_pipeline(query, solver_name, expected_fragment) -> None:
    result = solve_deterministic_query(query)
    assert result is not None
    assert result.solver == solver_name
    assert expected_fragment in result.answer
    assert "**A) Risposta**" in result.answer
    assert "**D) Fonti**" in result.answer


def test_unsupported_query_returns_none() -> None:
    assert solve_deterministic_query("Spiega la politica di sicurezza") is None
    assert try_solve_math_query("Spiega la politica di sicurezza") is None


def test_calculator_accepts_safe_expression_and_rejects_dangerous_input() -> None:
    assert calcolatrice_universale("2 + 3 * 4") == "14,00"
    assert calcolatrice_universale("2 / 0") == ""
    assert calcolatrice_universale("__import__('os').system('dir')") == ""
    assert calcolatrice_universale("2 ** 100") == ""


def test_calculation_request_detection() -> None:
    assert is_calculation_request("Calcola il 20% di 100")
    assert is_calculation_request("Risolvi X > 10")
    assert not is_calculation_request("Quali sono i requisiti normativi?")
