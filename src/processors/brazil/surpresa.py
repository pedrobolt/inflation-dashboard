"""
Processador de Surpresa mensal em relação ao Focus.
"""

import pandas as pd
from typing import Dict, List, Optional


def process_surpresa(
    df_general: pd.DataFrame,
    df_groups: pd.DataFrame,
    projections: Optional[pd.DataFrame] = None,
) -> Dict[str, List[Dict]]:
    """
    Calcula série de surpresa mensal em bps para IPCA e categorias.

    Surpresa = realizado(t) - projeção Focus(t) (%).

    Para categorias (Serviços, Industriais, Alimentação, Livres, Administrados),
    a surpresa é calculada sobre a variação da categoria, usando a mesma projeção
    do IPCA geral como referência (aproximação, já que o Focus não projeta categorias).
    """
    df = df_general.sort_values("periodo_codigo").copy()
    df["period"] = df["periodo_codigo"].astype(str)
    df["realizado"] = df["mom"]

    # Mapeia projeções para cada período
    if projections is not None and not projections.empty:
        proj_map = projections.set_index("periodo")["media"].to_dict()
    else:
        proj_map = {}

    df["projecao"] = df["period"].map(proj_map)

    # Para períodos sem projeção, usa o realizado t-1 como proxy
    df["projecao"] = df["projecao"].fillna(df["realizado"].shift(1))

    # Surpresa em bps
    df["surprise"] = (df["realizado"] - df["projecao"]) * 100

    # Médias móveis da surpresa
    df["m1"] = df["surprise"]
    df["m3"] = df["surprise"].rolling(3, min_periods=1).mean()
    df["m6"] = df["surprise"].rolling(6, min_periods=1).mean()
    df["m12"] = df["surprise"].rolling(12, min_periods=1).mean()

    ipc_series = df[["period", "m1", "m3", "m6", "m12"]].to_dict("records")

    # Para categorias, calculamos a surpresa usando o realizado da categoria menos projeção IPCA
    from .resumo import _calc_category_series, _classify_item

    categories = {
        "Serviços": _calc_category_series(df_groups, lambda c, n: _classify_item(c, n)["servico"]),
        "Industriais": _calc_category_series(df_groups, lambda c, n: _classify_item(c, n)["industrial"]),
        "Alimentação": _calc_category_series(df_groups, lambda c, n: _classify_item(c, n)["alimentacao_domicilio"]),
        "Livres": _calc_category_series(df_groups, lambda c, n: _classify_item(c, n)["livre"]),
        "Administrados": _calc_category_series(df_groups, lambda c, n: _classify_item(c, n)["administrado"]),
    }

    result = {"IPCA": ipc_series}

    for cat_name, cat_df in categories.items():
        if cat_df.empty:
            continue
        cat_df = cat_df.sort_values("periodo_codigo").copy()
        cat_df["period"] = cat_df["periodo_codigo"].astype(str)
        cat_df["realizado"] = cat_df["mom"]
        cat_df["projecao"] = cat_df["period"].map(proj_map).fillna(cat_df["realizado"].shift(1))
        cat_df["surprise"] = (cat_df["realizado"] - cat_df["projecao"]) * 100
        cat_df["m1"] = cat_df["surprise"]
        cat_df["m3"] = cat_df["surprise"].rolling(3, min_periods=1).mean()
        cat_df["m6"] = cat_df["surprise"].rolling(6, min_periods=1).mean()
        cat_df["m12"] = cat_df["surprise"].rolling(12, min_periods=1).mean()
        result[cat_name] = cat_df[["period", "m1", "m3", "m6", "m12"]].to_dict("records")

    return result
