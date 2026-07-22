"""
Coletor de séries do FRED (Federal Reserve Economic Data) para o painel dos EUA.

A chave da API vem da variável de ambiente FRED_API_KEY ou do arquivo .env
na raiz do projeto (gitignored).
"""

import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_api_key() -> Optional[str]:
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key.strip()
    env_file = _PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("FRED_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


class FREDClient:
    """Coleta séries temporais da API do FRED."""

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    # Séries usadas no painel dos EUA (todas SA, salvo indicação)
    SERIES = {
        # Índices de preço (nível do índice; MoM/YoY calculados no processor)
        "cpi_headline": "CPIAUCSL",       # CPI-U All Items
        "cpi_core": "CPILFESL",           # CPI ex food & energy
        "pce_headline": "PCEPI",          # PCE price index
        "pce_core": "PCEPILFE",           # PCE ex food & energy
        "cpi_core_goods": "CUSR0000SACL1E",   # Commodities less food & energy
        "cpi_shelter": "CUSR0000SAH1",        # Shelter
        "cpi_supercore": "CUSR0000SASL2RS",   # Services less rent of shelter (aprox. supercore)
        "cpi_food": "CPIUFDSL",               # Food
        "cpi_energy": "CPIENGSL",             # Energy
        # Núcleos alternativos (já em taxa, % anualizada ou YoY)
        "median_cpi": "MEDCPIM158SFRBCLE",       # Cleveland Fed Median CPI (MoM anualizado)
        "trimmed_cpi": "TRMMEANCPIM158SFRBCLE",  # Cleveland Fed 16% Trimmed-Mean (MoM anualizado)
        "sticky_cpi": "CORESTICKM159SFRBATL",    # Atlanta Fed Sticky CPI (YoY)
        # Expectativas
        "breakeven_5y": "T5YIE",     # 5-Year Breakeven (diário)
        "breakeven_5y5y": "T5YIFR",  # 5-Year, 5-Year Forward (diário)
        "michigan_1y": "MICH",       # Michigan: expectativa mediana 12m (mensal)
    }

    def __init__(self, api_key: Optional[str] = None, timeout: int = 60):
        self.api_key = api_key or _load_api_key()
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "inflation-dashboard/0.1.0"})

    def fetch_series(self, series_id: str, start_date: str = "2004-01-01") -> pd.DataFrame:
        """Busca uma série; retorna DataFrame com colunas ['date', 'value']."""
        if not self.api_key:
            raise RuntimeError("FRED_API_KEY não configurada (env var ou .env)")
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
        }
        response = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        obs = response.json()["observations"]
        df = pd.DataFrame(obs)[["date", "value"]]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna(subset=["value"]).reset_index(drop=True)

    def fetch_all(self, start_date: str = "2004-01-01") -> Dict[str, pd.DataFrame]:
        """Busca todas as séries do painel. Séries diárias são reduzidas a média mensal."""
        result = {}
        for name, series_id in self.SERIES.items():
            df = self.fetch_series(series_id, start_date=start_date)
            if name.startswith("breakeven"):
                df = (
                    df.set_index("date")["value"]
                    .resample("MS").mean()
                    .dropna().reset_index()
                )
            result[name] = df
        return result
