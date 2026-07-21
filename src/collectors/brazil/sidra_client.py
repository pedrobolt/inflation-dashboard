"""
Cliente genérico para a API SIDRA do IBGE.
"""

import requests
import pandas as pd
from typing import List, Optional


class SidraClient:
    BASE_URL = "https://apisidra.ibge.gov.br/values"

    def __init__(self, timeout: int = 120):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "inflation-dashboard/0.1.0",
            "Accept": "application/json",
        })

    def _build_url(self, tabela: str, parametros: str) -> str:
        return f"{self.BASE_URL}/t/{tabela}/{parametros}"

    def fetch(self, tabela: str, parametros: str) -> List[dict]:
        url = self._build_url(tabela, parametros)
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Erro ao buscar {url}: {e}") from e

    def fetch_to_dataframe(self, tabela: str, parametros: str) -> pd.DataFrame:
        data = self.fetch(tabela, parametros)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data[1:], columns=data[0])
        return df

    @staticmethod
    def parse_period(periodo_str: str) -> pd.Period:
        return pd.Period(periodo_str, freq="M")


def build_period_param(start: str, end: Optional[str] = None) -> str:
    if end:
        return f"{start}-{end}"
    return start
