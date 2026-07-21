"""
Processador da aba Por Grupos (núcleos e categorias especiais).
"""

import pandas as pd
from typing import Dict, Optional
from .resumo import _calc_category_series, _classify_item


def process_grupos(
    df_general: pd.DataFrame,
    df_groups: pd.DataFrame,
    bcb_cores: Optional[pd.DataFrame] = None,
    period: str = None,
) -> Dict[str, Dict]:
    """
    Processa dados para a aba Por Grupos.

    Categorias:
      - BCB: núcleos do BCB (média dos núcleos EX0/EX1/EX2)
      - Serviços, Industriais, Alimentação: calculados a partir do IPCA detalhado
    """
    df_general = df_general.sort_values("periodo_codigo").copy()
    if period is None:
        period = str(df_general["periodo_codigo"].max())

    # Categorias calculadas a partir dos dados detalhados
    categories = {
        "Serviços": _calc_category_series(df_groups, lambda c, n: _classify_item(c, n)["servico"]),
        "Industriais": _calc_category_series(df_groups, lambda c, n: _classify_item(c, n)["industrial"]),
        "Alimentação": _calc_category_series(df_groups, lambda c, n: _classify_item(c, n)["alimentacao_domicilio"]),
    }

    # Para BCB, usa dados reais dos núcleos se disponíveis
    if bcb_cores is not None and not bcb_cores.empty:
        bcb_df = bcb_cores.copy()
        bcb_df["periodo"] = bcb_df["data"].dt.to_period("M").dt.strftime("%Y%m")
        bcb_df["mom"] = bcb_df["media"]
        # Calcula YoY real a partir do índice acumulado em 12 meses
        bcb_df = bcb_df.sort_values("data").reset_index(drop=True)
        bcb_df["index"] = (1 + bcb_df["mom"] / 100).cumprod()
        bcb_df["yoy"] = (bcb_df["index"] / bcb_df["index"].shift(12) - 1) * 100
        bcb_df["yoy"] = bcb_df["yoy"].where(bcb_df["index"].shift(12).notna())
        bcb_df = bcb_df[["periodo", "mom", "yoy"]].drop_duplicates("periodo")
        categories["BCB"] = bcb_df
    else:
        categories["BCB"] = df_general[["periodo", "periodo_codigo", "mom", "yoy"]].copy()
        categories["BCB"] = categories["BCB"].rename(columns={"periodo_codigo": "period"})

    result = {}
    for name, series in categories.items():
        if series.empty:
            continue
        df = series.copy()
        if "periodo_codigo" in df.columns:
            df = df.rename(columns={"periodo_codigo": "period"})
        elif "periodo" in df.columns:
            df = df.rename(columns={"periodo": "period"})
        df = df.sort_values("period")
        df["m3"] = df["mom"].rolling(3, min_periods=1).mean()
        df["m6"] = df["mom"].rolling(6, min_periods=1).mean()
        df["m12"] = df["mom"].rolling(12, min_periods=1).mean()
        df["saar1"] = ((1 + df["mom"] / 100) ** 12 - 1) * 100
        df["saar3"] = ((1 + df["m3"] / 100) ** 12 - 1) * 100
        df["saar6"] = ((1 + df["m6"] / 100) ** 12 - 1) * 100
        df["meta"] = 3.0  # meta BCB

        last = df[df["period"] == period]
        if last.empty:
            continue
        last = last.iloc[-1]

        result[name] = {
            "series": df[["period", "mom", "yoy", "saar1", "saar3", "saar6", "meta"]].to_dict("records"),
            "latest": {
                "saar1": float(last["saar1"]) if pd.notna(last["saar1"]) else None,
                "saar3": float(last["saar3"]) if pd.notna(last["saar3"]) else None,
                "saar6": float(last["saar6"]) if pd.notna(last["saar6"]) else None,
                "yoy": float(last["yoy"]) if pd.notna(last["yoy"]) else None,
            }
        }

    return result
