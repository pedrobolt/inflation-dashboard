"""
Coletor de dados do IPCA via API SIDRA do IBGE.
"""

import pandas as pd
from typing import Optional
from .sidra_client import SidraClient, build_period_param


class IBGECollector:
    """Coleta IPCA geral, grupos, subitens e pesos do IBGE SIDRA."""

    def __init__(self, client: Optional[SidraClient] = None):
        self.client = client or SidraClient()

    def fetch_ipca_general(self, start_period: str, end_period: Optional[str] = None) -> pd.DataFrame:
        """Busca IPCA geral (tabela 1737): variação mensal e acumulada 12 meses."""
        period = build_period_param(start_period, end_period)
        params = f"n1/all/v/63,2265/p/{period}"
        df = self.client.fetch_to_dataframe("1737", params)
        return self._normalize_general(df)

    def fetch_ipca_groups_and_subitems(self, start_period: str, end_period: Optional[str] = None) -> pd.DataFrame:
        """Busca IPCA por grupos/subitens com peso (tabela 7060)."""
        period = build_period_param(start_period, end_period)
        params = f"n1/all/v/63,66,2265/p/{period}/c315/all/d/v63%202,v66%202,v2265%202"
        df = self.client.fetch_to_dataframe("7060", params)
        return self._normalize_groups_subitems(df)

    @staticmethod
    def _normalize_general(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df = df.rename(columns={
            "D3C": "periodo_codigo",
            "D3N": "periodo_nome",
            "D2C": "variavel_codigo",
            "D2N": "variavel",
            "V": "valor",
        })
        df["valor"] = (
            df["valor"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .replace({"...": pd.NA, "-": pd.NA, "X": pd.NA})
        )
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df["periodo"] = pd.to_datetime(df["periodo_codigo"].astype(str), format="%Y%m", errors="coerce")
        df_pivot = df.pivot_table(
            index=["periodo", "periodo_codigo", "periodo_nome"],
            columns="variavel",
            values="valor",
            aggfunc="first",
        ).reset_index()
        df_pivot.columns.name = None
        df_pivot = df_pivot.rename(columns={
            "IPCA - Variação mensal": "mom",
            "IPCA - Variação acumulada em 12 meses": "yoy",
        })
        df_pivot["indice"] = "Índice Geral"
        df_pivot["indice_codigo"] = "0"
        return df_pivot

    @staticmethod
    def _normalize_groups_subitems(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df = df.rename(columns={
            "D3C": "periodo_codigo",
            "D3N": "periodo_nome",
            "D2C": "variavel_codigo",
            "D2N": "variavel",
            "D4C": "item_codigo",
            "D4N": "item",
            "V": "valor",
        })
        df["valor"] = (
            df["valor"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .replace({"...": pd.NA, "-": pd.NA, "X": pd.NA})
        )
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df["periodo"] = pd.to_datetime(df["periodo_codigo"].astype(str), format="%Y%m", errors="coerce")
        df_pivot = df.pivot_table(
            index=["periodo", "periodo_codigo", "periodo_nome", "item", "item_codigo"],
            columns="variavel",
            values="valor",
            aggfunc="first",
        ).reset_index()

        df_pivot.columns.name = None
        df_pivot = df_pivot.rename(columns={
            "IPCA - Variação mensal": "mom",
            "IPCA - Peso mensal": "peso",
            "IPCA - Variação acumulada em 12 meses": "yoy",
        })

        # SIDRA 7060 retorna pesos diretamente em percentual do IPCA
        # (ex: Índice Geral = 100.00, Alimentação e bebidas = 21.95)
        return df_pivot
def get_latest_available_period() -> str:
    """Retorna o último mês disponível aproximado (IPCA divulgado com defasagem)."""
    today = pd.Timestamp.now()
    candidate = today - pd.DateOffset(months=2)
    return candidate.strftime("%Y%m")
