"""
Coletor de dados do BCB (SGS) para núcleos do IPCA.
"""

from typing import Optional
import requests
import pandas as pd


class BCBClient:
    """Coleta séries temporais do SGS do Banco Central."""

    BASE_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"

    # Códigos conhecidos para núcleos do IPCA
    CORE_CODES = {
        "EX0": 11426,  # IPCA ex alimentos e energia
        "EX1": 11427,  # IPCA ex alimentos, energia e monitorados
        "EX2": 11428,  # IPCA ex alimentos, energia, monitorados e administ
    }

    def __init__(self, timeout: int = 60):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "inflation-dashboard/0.1.0",
            "Accept": "application/json",
        })

    def fetch_series(self, code: int, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Busca série do SGS.

        Args:
            code: código da série no SGS
            start_date: data inicial no formato 'dd/mm/aaaa'
            end_date: data final no formato 'dd/mm/aaaa'
        """
        url = self.BASE_URL.format(code=code)
        params = {"formato": "json", "dataInicial": start_date, "dataFinal": end_date}
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        if df.empty:
            return df
        df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y", errors="coerce")
        df["valor"] = pd.to_numeric(df["valor"].astype(str).str.replace(",", "."), errors="coerce")
        df = df.dropna(subset=["data", "valor"])
        return df

    def fetch_ipca_cores(
        self,
        start_date: str = "01/01/2004",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Busca os três núcleos disponíveis e retorna DataFrame combinado."""
        if end_date is None:
            end_date = pd.Timestamp.now().strftime("%d/%m/%Y")

        parts = []
        for name, code in self.CORE_CODES.items():
            df = self.fetch_series(code, start_date, end_date)
            if df.empty:
                continue
            df = df.rename(columns={"valor": name})
            df["core_name"] = name
            parts.append(df[["data", name]])

        if not parts:
            return pd.DataFrame(columns=["data"])

        merged = parts[0]
        for df in parts[1:]:
            merged = merged.merge(df, on="data", how="outer")
        merged = merged.sort_values("data")
        # Calcula média dos núcleos disponíveis
        core_cols = [c for c in self.CORE_CODES.keys() if c in merged.columns]
        merged["media"] = merged[core_cols].mean(axis=1)
        return merged

    @staticmethod
    def period_to_bcb_date(period_code: str) -> str:
        """Converte '202505' para '01/05/2025'."""
        p = pd.Period(period_code, freq="M")
        return f"01/{p.month:02d}/{p.year}"
