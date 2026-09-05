"""
Health score ingenuo: nessuna metrica statistica (Sharpe, volatilita'),
solo percentuali dirette facili da spiegare a chi non mastica finanza.

Score = media pesata di tre percentuali:
- rendimento a 5 anni (40%): il fondo e' cresciuto nel lungo periodo?
- rendimento da inizio anno (35%): il trend dell'anno tiene?
- distanza dal massimo degli ultimi 5 anni (25%): siamo vicini ai massimi
  o in caduta da un picco?
"""

from dataclasses import dataclass, field

import pandas as pd

from .metrics import MetricsCalculator

WEIGHT_5Y = 0.40
WEIGHT_YTD = 0.35
WEIGHT_DRAWDOWN = 0.25

# (percentuale, punteggio) di riferimento; interpolazione lineare tra i punti,
# valore costante fuori dai due estremi.
BREAKPOINTS_5Y = [(-20.0, 0.0), (0.0, 50.0), (40.0, 100.0)]
BREAKPOINTS_YTD = [(-15.0, 0.0), (0.0, 50.0), (15.0, 100.0)]
BREAKPOINTS_DRAWDOWN = [(-50.0, 0.0), (-25.0, 50.0), (0.0, 100.0)]


def _piecewise_score(value: float, points: list[tuple[float, float]]) -> float:
    if value <= points[0][0]:
        return points[0][1]
    if value >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= value <= x1:
            t = (value - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return points[-1][1]


@dataclass
class HealthScoreComponent:
    """Una delle tre percentuali che compongono lo score, gia' pesata."""
    label: str
    pct: float
    score: float
    weight: float


@dataclass
class HealthScoreResult:
    score: float
    interpretation: str
    components: list[HealthScoreComponent] = field(default_factory=list)

    def explanation(self) -> str:
        """Testo sintetico per una tooltip: percentuali reali + peso di ciascuna."""
        lines = ["Media pesata di tre orizzonti temporali:"]
        for c in self.components:
            lines.append(f"- {c.label}: {c.pct:+.1f}% (peso {c.weight * 100:.0f}%)")
        return "\n".join(lines)


def calculate_health_score(data: pd.DataFrame) -> HealthScoreResult:
    """
    Calcola l'health score a partire da una serie storica di almeno 5 anni.

    Args:
        data: DataFrame con colonna 'Close', copertura di 5 anni.

    Returns:
        HealthScoreResult con punteggio 0-100 e dettaglio dei tre componenti.
    """
    prices = data["Close"]
    current_year = prices.index[-1].year
    prices_ytd = prices[prices.index.year == current_year]

    return_5y = MetricsCalculator.calculate_total_return(prices)
    return_ytd = MetricsCalculator.calculate_total_return(prices_ytd)
    drawdown = MetricsCalculator.calculate_current_drawdown(prices)

    components = [
        HealthScoreComponent(
            label="Rendimento 5 anni",
            pct=return_5y,
            score=_piecewise_score(return_5y, BREAKPOINTS_5Y),
            weight=WEIGHT_5Y,
        ),
        HealthScoreComponent(
            label="Rendimento da inizio anno",
            pct=return_ytd,
            score=_piecewise_score(return_ytd, BREAKPOINTS_YTD),
            weight=WEIGHT_YTD,
        ),
        HealthScoreComponent(
            label="Distanza dal massimo (5 anni)",
            pct=drawdown,
            score=_piecewise_score(drawdown, BREAKPOINTS_DRAWDOWN),
            weight=WEIGHT_DRAWDOWN,
        ),
    ]

    score = sum(c.score * c.weight for c in components)
    score = min(max(score, 0.0), 100.0)

    if score >= 75:
        interpretation = "Dormi tranquillo, il piano prosegue"
    elif score >= 50:
        interpretation = "Attenzione, monitora: volatilita' normale"
    else:
        interpretation = "Valuta la situazione con calma"

    return HealthScoreResult(score=score, interpretation=interpretation, components=components)
