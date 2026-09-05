"""
VWCE Health Check - Dashboard semplificata
"""

import logging

import altair as alt
import pandas as pd
import streamlit as st
import yfinance as yf

from lib import DataFetchError, DataFetcher, calculate_health_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

TICKER = "VWCE.DE"

st.set_page_config(
    page_title="VWCE Health Check",
    page_icon="📈",
    layout="centered",
)

st.title("VWCE Health Check")
st.caption("Vanguard FTSE All-World UCITS ETF (EUR) • prospettiva di lungo periodo")


@st.cache_data(ttl=3600)
def load_data(ticker: str) -> pd.DataFrame:
    return DataFetcher().fetch_historical_data(ticker, period="5y", interval="1d")


try:
    data = load_data(TICKER)
except DataFetchError as e:
    st.error(f"Non riesco a scaricare i dati per {TICKER}: {e}")
    st.stop()

if data["Close"].isna().any():
    st.error(
        f"I dati per {TICKER} contengono prezzi mancanti dopo la pulizia "
        "(probabile problema temporaneo sul feed di Yahoo Finance). "
        "Svuota la cache (menu ≡ in alto a destra → 'Clear cache') e ricarica "
        "la pagina; se il problema persiste, guarda il pannello Diagnostica "
        "qui sotto ed i log del terminale."
    )
    st.stop()

result = calculate_health_score(data)

current_price = data["Close"].iloc[-1]
previous_price = data["Close"].iloc[-2]
current_date = data.index[-1]
daily_change_pct = (current_price / previous_price - 1) * 100

col_price, col_health = st.columns(2)

with col_price:
    st.metric(
        label=f"Prezzo attuale (al {current_date:%d/%m})",
        value=f"€{current_price:.2f}",
        delta=f"{daily_change_pct:+.2f}%",
        help=(
            "Variazione percentuale rispetto alla chiusura del giorno di "
            "borsa precedente, non il rendimento a 5 anni del grafico qui "
            "sotto. La data indica l'ultima chiusura disponibile: può "
            "risultare di qualche giorno fa se il feed di Yahoo Finance è "
            "temporaneamente incompleto."
        ),
    )

with col_health:
    st.metric(
        label="Health score",
        value=f"{result.score:.0f}/100",
        delta=result.interpretation,
        delta_color="off",
        help=result.explanation(),
    )

st.divider()

st.subheader("Andamento prezzo (5 anni)")

chart_data = pd.DataFrame({"Prezzo": data["Close"]})
chart_data["SMA 200"] = data["Close"].rolling(200).mean()
chart_data = chart_data.reset_index().melt(
    "Date", var_name="Serie", value_name="Valore"
)

chart = (
    alt.Chart(chart_data)
    .mark_line()
    .encode(
        x=alt.X(
            "Date:T",
            title=None,
            axis=alt.Axis(format="%Y", tickCount="year"),
        ),
        y=alt.Y("Valore:Q", title=None),
        color=alt.Color("Serie:N", title=None),
    )
    .properties(height=400)
)
st.altair_chart(chart, width="stretch")

st.caption(
    "SMA 200",
    help=(
        "Media mobile a 200 giorni: la media del prezzo di chiusura degli "
        "ultimi 200 giorni di borsa (circa 10 mesi). Si muove piu' lentamente "
        "del prezzo e mostra il trend di fondo: se il prezzo sta sopra questa "
        "linea, il trend di lungo periodo e' considerato positivo."
    ),
)

st.divider()
st.caption("Dati: Yahoo Finance • aggiornati ogni ora")

with st.expander("Diagnostica (per segnalare un problema)"):
    st.caption(
        "Se qualcosa non torna, copia questo blocco insieme a una "
        "descrizione di cosa hai visto."
    )
    st.code(
        f"Ticker: {TICKER}\n"
        f"Righe scaricate: {len(data)}\n"
        f"Ultima data: {data.index[-1].date()}\n"
        f"Ultimo prezzo (Close): {current_price}\n"
        f"Prezzo precedente (Close): {previous_price}\n"
        f"Valori NaN residui nel dataset: {int(data.isna().sum().sum())}\n"
        f"yfinance: {yf.__version__}\n"
        f"pandas: {pd.__version__}\n"
        f"streamlit: {st.__version__}\n",
        language="text",
    )
