"""
Processador de Surpresa mensal em relação ao Focus.

Como projeções detalhadas por categoria são difíceis de obter publicamente,
este módulo calcula a surpresa para o IPCA geral e usa proxy para categorias.
"""

import pandas as pd
from typing import Dict, List


def process_surpresa(df_general: pd.DataFrame, projections: Dict[str, float] = None) -> Dict[str, List[Dict]]:
    """
    Calcula série de surpresa mensal em bps.
    Surpresa = realizado(t) - projeção média Focus(t).

    Se `projections` não for fornecido, assume projeção igual ao realizado t-1
    (apenas para visualização).
    """
    df = df_general.sort_values("periodo_codigo").copy()
    df["period"] = df["periodo_codigo"].astype(str)
    df["m1"] = df["mom"] * 100  # bps

    # Projeção proxy
    if projections is None:
        df["proj"] = df["m1"].shift(1)
    else:
        df["proj"] = df["period"].map(projections)

    df["surprise"] = df["m1"] - df["proj"]

    # Médias móveis da surpresa
    df["m3"] = df["surprise"].rolling(3, min_periods=1).mean()
    df["m6"] = df["surprise"].rolling(6, min_periods=1).mean()
    df["m12"] = df["surprise"].rolling(12, min_periods=1).mean()

    series = df[["period", "m1", "m3", "m6", "m12"]].to_dict("records")

    # Categorias exibidas no frontend; usam IPCA geral como proxy
    categories = ["IPCA", "Serviços", "Industriais", "Alimentação", "Livres", "Administrados"]
    return {cat: series for cat in categories}
