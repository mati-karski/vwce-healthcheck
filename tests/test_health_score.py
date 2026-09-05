"""
Unit tests for the health_score module.
"""

import numpy as np
import pandas as pd
import pytest

from lib.health_score import (
    BREAKPOINTS_5Y,
    BREAKPOINTS_DRAWDOWN,
    BREAKPOINTS_YTD,
    HealthScoreComponent,
    HealthScoreResult,
    _piecewise_score,
    calculate_health_score,
)


class TestPiecewiseScore:
    """Test suite for the piecewise-linear scoring helper."""

    POINTS = [(-20.0, 0.0), (0.0, 50.0), (40.0, 100.0)]

    def test_below_range_clamps_to_min(self):
        assert _piecewise_score(-100.0, self.POINTS) == 0.0

    def test_above_range_clamps_to_max(self):
        assert _piecewise_score(100.0, self.POINTS) == 100.0

    def test_exact_breakpoints(self):
        assert _piecewise_score(-20.0, self.POINTS) == pytest.approx(0.0)
        assert _piecewise_score(0.0, self.POINTS) == pytest.approx(50.0)
        assert _piecewise_score(40.0, self.POINTS) == pytest.approx(100.0)

    def test_midpoint_interpolation(self):
        # A meta' del primo segmento (-20 -> 0) ci si aspetta meta' del salto (0 -> 50)
        assert _piecewise_score(-10.0, self.POINTS) == pytest.approx(25.0)
        # A meta' del secondo segmento (0 -> 40) ci si aspetta meta' del salto (50 -> 100)
        assert _piecewise_score(20.0, self.POINTS) == pytest.approx(75.0)


class TestCalculateHealthScore:
    """
    Test suite per calculate_health_score, con serie sintetiche costruite per
    ottenere percentuali esatte (rendimento 5Y, YTD, drawdown), cosi' da
    verificare sia il piecewise scoring sia la combinazione pesata 40/35/25.
    """

    @staticmethod
    def _build_series(total_return_5y: float, peak_ratio: float, ytd_return: float) -> pd.DataFrame:
        """
        Costruisce 5 anni di prezzi giornalieri (2020-01-01 -> 2024-12-31) tali che:
        - il rendimento sull'intera serie sia esattamente total_return_5y (%)
        - il prezzo massimo della serie sia last_price * peak_ratio
          (drawdown attuale = (1/peak_ratio - 1) * 100)
        - il rendimento nell'ultimo anno solare (2024) sia esattamente ytd_return (%)

        L'ultimo anno e' piatto (prezzo costante) per isolare il rendimento YTD
        dal resto della curva; il picco, se sopra il livello di fine periodo,
        cade prima dell'ultimo anno.
        """
        dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
        is_last_year = dates.year == dates[-1].year
        n_total = len(dates)
        n_last_year = int(is_last_year.sum())
        n_pre = n_total - n_last_year

        first_price = 100.0
        last_price = first_price * (1 + total_return_5y / 100)
        ytd_start_price = last_price / (1 + ytd_return / 100)
        peak = last_price * peak_ratio

        n_rise = int(n_pre * 0.7)
        n_fall = n_pre - n_rise
        rise = np.linspace(first_price, peak, n_rise, endpoint=False)
        fall = np.linspace(peak, ytd_start_price, n_fall, endpoint=False)
        last_year_segment = np.linspace(ytd_start_price, last_price, n_last_year)

        prices = np.concatenate([rise, fall, last_year_segment])
        return pd.DataFrame({"Close": prices}, index=dates)

    def test_mixed_scores_weighted_correctly(self):
        # 5Y +40% (score 100), YTD 0% (score 50), drawdown -25% (score 50)
        # -> 0.40*100 + 0.35*50 + 0.25*50 = 70.0
        data = self._build_series(total_return_5y=40.0, peak_ratio=1 / 0.75, ytd_return=0.0)

        result = calculate_health_score(data)

        assert result.score == pytest.approx(70.0, abs=0.5)
        assert result.interpretation == "Attenzione, monitora: volatilita' normale"

        by_label = {c.label: c for c in result.components}
        assert by_label["Rendimento 5 anni"].pct == pytest.approx(40.0, abs=0.5)
        assert by_label["Rendimento 5 anni"].score == pytest.approx(100.0, abs=0.5)
        assert by_label["Rendimento da inizio anno"].pct == pytest.approx(0.0, abs=0.5)
        assert by_label["Rendimento da inizio anno"].score == pytest.approx(50.0, abs=0.5)
        assert by_label["Distanza dal massimo (5 anni)"].pct == pytest.approx(-25.0, abs=0.5)
        assert by_label["Distanza dal massimo (5 anni)"].score == pytest.approx(50.0, abs=0.5)

    def test_best_case_scores_100(self):
        # Tutti i componenti al cap superiore o oltre -> score 100
        data = self._build_series(total_return_5y=60.0, peak_ratio=1.0, ytd_return=20.0)

        result = calculate_health_score(data)

        assert result.score == pytest.approx(100.0, abs=0.5)
        assert result.interpretation == "Dormi tranquillo, il piano prosegue"

    def test_worst_case_scores_near_zero(self):
        # 5Y e YTD ben sotto i cap inferiori, drawdown severo
        data = self._build_series(total_return_5y=-40.0, peak_ratio=1 / 0.5, ytd_return=-30.0)

        result = calculate_health_score(data)

        assert result.score == pytest.approx(0.0, abs=0.5)
        assert result.interpretation == "Valuta la situazione con calma"

    def test_score_always_within_bounds(self):
        data = self._build_series(total_return_5y=500.0, peak_ratio=1.0, ytd_return=500.0)

        result = calculate_health_score(data)

        assert 0.0 <= result.score <= 100.0

    def test_returns_three_components(self):
        data = self._build_series(total_return_5y=10.0, peak_ratio=1.1, ytd_return=5.0)

        result = calculate_health_score(data)

        assert len(result.components) == 3
        assert all(isinstance(c, HealthScoreComponent) for c in result.components)


class TestHealthScoreExplanation:
    """Test suite per il testo sintetico usato nella tooltip della UI."""

    def test_explanation_lists_all_components_with_weights(self):
        result = HealthScoreResult(
            score=70.0,
            interpretation="Attenzione, monitora: volatilita' normale",
            components=[
                HealthScoreComponent("Rendimento 5 anni", 40.0, 100.0, 0.40),
                HealthScoreComponent("Rendimento da inizio anno", 0.0, 50.0, 0.35),
                HealthScoreComponent("Distanza dal massimo (5 anni)", -25.0, 50.0, 0.25),
            ],
        )

        text = result.explanation()

        assert "Rendimento 5 anni: +40.0% (peso 40%)" in text
        assert "Rendimento da inizio anno: +0.0% (peso 35%)" in text
        assert "Distanza dal massimo (5 anni): -25.0% (peso 25%)" in text

    def test_explanation_empty_components(self):
        result = HealthScoreResult(score=50.0, interpretation="neutro", components=[])

        text = result.explanation()

        assert text == "Media pesata di tre orizzonti temporali:"


class TestBreakpointsConsistency:
    """Verifica che le soglie usate dal modulo restino quelle concordate."""

    def test_breakpoints_5y(self):
        assert BREAKPOINTS_5Y == [(-20.0, 0.0), (0.0, 50.0), (40.0, 100.0)]

    def test_breakpoints_ytd(self):
        assert BREAKPOINTS_YTD == [(-15.0, 0.0), (0.0, 50.0), (15.0, 100.0)]

    def test_breakpoints_drawdown(self):
        assert BREAKPOINTS_DRAWDOWN == [(-50.0, 0.0), (-25.0, 50.0), (0.0, 100.0)]
