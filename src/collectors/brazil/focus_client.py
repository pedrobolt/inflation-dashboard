"""
Coletor de projeções Focus (BCB) via python-bcb.
"""

from typing import Optional
import pandas as pd
from bcb import Expectativas


class FocusClient:
    """Coleta projeções mensais do Focus para IPCA."""

    def __init__(self):
        self.exp = Expectativas()
        self.endpoint = self.exp.get_endpoint('ExpectativasMercadoTop5Mensais')

    def fetch_ipca_projections(
        self,
        start_date: str = "2004-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Busca projeções mensais do IPCA.

        Retorna DataFrame com colunas:
          - data: data da projeção
          - periodo: período de referência (YYYYMM)
          - media: média das projeções (%)
        """
        if end_date is None:
            end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

        # Busca todas as projeções disponíveis.
        # O endpoint tem limitação de 50k/100k por query; fazemos duas buscas para cobrir
        # dados mais antigos e mais recentes, depois combina.
        dfs = []
        for order, limit in [("desc", 100000), ("asc", 50000)]:
            try:
                df_part = self.endpoint.query().filter(
                    f"Indicador eq 'IPCA' and Data ge '{start_date}' and Data le '{end_date}'"
                ).orderby(f"Data {order}").limit(limit).collect()
                if not df_part.empty:
                    dfs.append(df_part)
            except Exception as e:
                print(f"Aviso: falha ao buscar projeções Focus ({order}): {e}")

        if not dfs:
            return pd.DataFrame(columns=["data", "periodo", "media"])

        df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["Data", "DataReferencia"])

        df = df.copy()
        df["Data"] = pd.to_datetime(df["Data"])
        df["ref_ano"] = df["DataReferencia"].str[-4:].astype(int)
        df["ref_mes"] = df["DataReferencia"].str[:2].astype(int)
        df["periodo"] = df["ref_ano"].astype(str) + df["ref_mes"].astype(str).str.zfill(2)
        df["Media"] = pd.to_numeric(df["Media"], errors="coerce")

        # Para cada período de referência, pega a projeção mais recente
        df = df.sort_values("Data")
        latest = df.groupby("periodo").last().reset_index()
        latest = latest.rename(columns={"Data": "data", "Media": "media"})
        return latest[["data", "periodo", "media"]].sort_values("periodo")
