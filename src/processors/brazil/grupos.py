"""
Processador da aba Por Grupos (núcleos e categorias especiais).
"""

import pandas as pd
from typing import Dict, List, Optional
from .resumo import _calc_category_series, _classify_item


def _calc_yoy_from_mom(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula YoY real a partir do índice acumulado."""
    df = df.sort_values("period").reset_index(drop=True)
    df["index"] = (1 + df["mom"] / 100).cumprod()
    df["yoy"] = (df["index"] / df["index"].shift(12) - 1) * 100
    df["yoy"] = df["yoy"].where(df["index"].shift(12).notna())
    return df


def _calc_saar(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula SAAR 1M e 3M a partir de variação mensal."""
    df = df.sort_values("period").reset_index(drop=True)
    df["m3"] = df["mom"].rolling(3, min_periods=1).mean()
    df["saar1"] = ((1 + df["mom"] / 100) ** 12 - 1) * 100
    df["saar3"] = ((1 + df["m3"] / 100) ** 12 - 1) * 100
    return df


def _subnucleo_row(df: pd.DataFrame, period: str, name: str) -> Optional[Dict]:
    """Monta uma linha da tabela lateral de sub-núcleos."""
    df = _calc_saar(_calc_yoy_from_mom(df))
    prev_period = (pd.Period(period, freq="M") - 1).strftime("%Y%m")
    prev12_period = (pd.Period(period, freq="M") - 12).strftime("%Y%m")

    def get_val(p, col):
        row = df[df["period"] == p]
        if row.empty:
            return None
        v = row[col].iloc[-1]
        return float(v) if pd.notna(v) else None

    def get_row_values(p):
        return {k: get_val(p, k) for k in ["saar1", "saar3", "yoy"]}

    latest = get_row_values(period)
    prev1 = get_row_values(prev_period)
    prev12 = get_row_values(prev12_period)
    if latest["saar1"] is None:
        return None

    return {
        "name": name,
        "saar1": latest["saar1"],
        "saar3": latest["saar3"],
        "yoy": latest["yoy"],
        "prev1_saar1": prev1["saar1"],
        "prev1_saar3": prev1["saar3"],
        "prev1_yoy": prev1["yoy"],
        "prev12_saar1": prev12["saar1"],
        "prev12_saar3": prev12["saar3"],
        "prev12_yoy": prev12["yoy"],
    }


def _bcb_subnuclei(bcb_cores: pd.DataFrame, period: str) -> List[Dict]:
    """Retorna linhas de sub-núcleos para a aba BCB."""
    bcb = bcb_cores.copy()
    bcb["period"] = bcb["data"].dt.to_period("M").dt.strftime("%Y%m")
    rows = []
    for label, col in [
        ("Média dos núcleos do BCB", "media"),
        ("IPCA-EX0", "EX0"),
        ("IPCA-EX3", "EX3"),
        ("IPCA-MS", "MS"),
        ("IPCA-DP", "DP"),
        ("IPCA-P55", "P55"),
    ]:
        if col not in bcb.columns:
            continue
        df = bcb[["period", col]].rename(columns={col: "mom"}).copy()
        row = _subnucleo_row(df, period, label)
        if row:
            rows.append(row)
    return rows


def _category_subnuclei(df_groups: pd.DataFrame, mask_fn, name: str, period: str) -> List[Dict]:
    """Retorna linhas de sub-núcleos para uma categoria (Serviços/Industriais/Alimentação)."""
    series = _calc_category_series(df_groups, mask_fn)
    if series.empty:
        return []
    df = series[["periodo", "mom"]].rename(columns={"periodo": "period"}).copy()
    df["period"] = pd.to_datetime(df["period"]).dt.to_period("M").dt.strftime("%Y%m")
    row = _subnucleo_row(df, period, f"Média dos núcleos de {name}")
    return [row] if row else []


def process_grupos(
    df_general: pd.DataFrame,
    df_groups: pd.DataFrame,
    bcb_cores: Optional[pd.DataFrame] = None,
    period: str = None,
) -> Dict[str, Dict]:
    """
    Processa dados para a aba Por Grupos.

    Categorias:
      - BCB: núcleos do BCB (média EX0/EX3/MS/DP/P55)
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

        if name == "BCB" and bcb_cores is not None and not bcb_cores.empty:
            subnuclei = _bcb_subnuclei(bcb_cores, period)
        else:
            mask_fn = None
            if name == "Serviços":
                mask_fn = lambda c, n: _classify_item(c, n)["servico"]
            elif name == "Industriais":
                mask_fn = lambda c, n: _classify_item(c, n)["industrial"]
            elif name == "Alimentação":
                mask_fn = lambda c, n: _classify_item(c, n)["alimentacao_domicilio"]
            subnuclei = _category_subnuclei(df_groups, mask_fn, name, period) if mask_fn else []

        result[name] = {
            "series": df[["period", "mom", "yoy", "saar1", "saar3", "saar6", "meta"]].to_dict("records"),
            "subnuclei": subnuclei,
            "latest": {
                "saar1": float(last["saar1"]) if pd.notna(last["saar1"]) else None,
                "saar3": float(last["saar3"]) if pd.notna(last["saar3"]) else None,
                "saar6": float(last["saar6"]) if pd.notna(last["saar6"]) else None,
                "yoy": float(last["yoy"]) if pd.notna(last["yoy"]) else None,
            }
        }

    return result
