"""
Coletor do Zillow Observed Rent Index (ZORI) — aluguel de mercado dos EUA.

CSV público, sem chave. Usado como indicador antecedente do CPI Shelter.
"""

import io

import pandas as pd
import requests


class ZillowClient:
    """Baixa o ZORI nacional (suavizado, SA) do CSV público do Zillow."""

    ZORI_URL = (
        "https://files.zillowstatic.com/research/public_csvs/zori/"
        "Metro_zori_uc_sfrcondomfr_sm_sa_month.csv"
    )

    def __init__(self, timeout: int = 60):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "inflation-dashboard/0.1.0"})

    def fetch_us_zori(self) -> pd.DataFrame:
        """Retorna DataFrame ['date', 'value'] com o ZORI nacional mensal."""
        response = self.session.get(self.ZORI_URL, timeout=self.timeout)
        response.raise_for_status()
        df = pd.read_csv(io.BytesIO(response.content))
        us = df[df["RegionName"] == "United States"]
        if us.empty:
            raise RuntimeError("Linha 'United States' não encontrada no CSV do ZORI")
        date_cols = [c for c in df.columns if c[:2] in ("19", "20")]
        series = us.iloc[0][date_cols].astype(float)
        out = pd.DataFrame({
            "date": pd.to_datetime(date_cols).to_period("M").to_timestamp(),
            "value": series.values,
        })
        return out.dropna(subset=["value"]).reset_index(drop=True)
