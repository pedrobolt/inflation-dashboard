"""
Processador da aba Por Grupos (núcleos e categorias especiais).
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional
from .resumo import _calc_category_series, _classify_item, _extract_code_prefix


# Heurísticas para variantes de categorias
_SERVICO_EXCLUDE_PASSAGEM = ["passagem aérea"]
_INDUSTRIAL_EXCLUDE = ["etanol", "fumo", "cigarro", "cosmético", "perfume", "maquiagem",
                       "higiene pessoal", "desodorante", "shampoo", "condicionador", "sabonete"]
_INDUSTRIAL_COSMETIC = ["cosmético", "perfume", "maquiagem", "desodorante", "shampoo",
                        "condicionador", "sabonete", "higiene pessoal"]
_ALIMENTO_IN_NATURA = [
    "arroz", "feijão", "batata", "mandioca", "inhame", "cará", "tomate", "cebola",
    "alface", "cenoura", "chuchu", "abobrinha", "pepino", "berinjela", "quiabo",
    "couve", "brócolis", "espinafre", "repolho", "acelga", "agrião", "rúcula",
    "maçã", "banana", "laranja", "mamão", "manga", "uva", "melão", "melancia",
    "pera", "pêssego", "ameixa", "morango", "abacaxi", "goiaba", "limão",
    "tangerina", "maracujá", "acerola", "açaí", "caju", "cajú",
    "carne bovina", "carne de boi", "carne de vaca", "frango", "peixe",
    "carne suína", "carne de porco", "linguado", "sardinha", "atum", "pescada",
    "ovos", "leite", "café em grão", "café cru",
]


def _name_has_any(name: str, keywords: List[str]) -> bool:
    n = str(name).lower()
    return any(k.lower() in n for k in keywords)


def _is_admin(code: str, name: str) -> bool:
    return _classify_item(code, name)["administrado"]


def _is_in_natura(name: str) -> bool:
    return _name_has_any(name, _ALIMENTO_IN_NATURA)


def _is_cosmetic(name: str) -> bool:
    return _name_has_any(name, _INDUSTRIAL_COSMETIC)


def _is_passagem(name: str) -> bool:
    return _name_has_any(name, _SERVICO_EXCLUDE_PASSAGEM)


def _leaf_items(df_groups: pd.DataFrame) -> pd.DataFrame:
    """Retorna apenas itens folha da tabela de grupos/subitens."""
    df = df_groups.copy()
    df["prefix"] = df["item"].apply(_extract_code_prefix)
    df = df[df["prefix"] != ""]
    all_prefixes = set(df["prefix"].unique())
    leaf_prefixes = {p for p in all_prefixes if not any(
        q != p and q.startswith(p) and len(q) > len(p) for q in all_prefixes
    )}
    return df[df["prefix"].isin(leaf_prefixes)].copy()


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Percentil ponderado usando interpolação linear."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(values) == 0 or weights.sum() == 0:
        return np.nan
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cumsum = np.cumsum(weights) / weights.sum()
    return float(np.interp(q, cumsum, values))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    if weights.sum() == 0:
        return np.nan
    return float(np.average(values, weights=weights))


def _aggregate_group(group: pd.DataFrame, kind: str, lower: float = 0.0, upper: float = 1.0,
                     q: float = 0.5, exclude_mask: Optional[pd.Series] = None) -> float:
    """Agrega variação mensal de um grupo de itens folha."""
    if exclude_mask is not None:
        group = group[~exclude_mask].copy()
    if group.empty or group["peso"].sum() <= 0:
        return np.nan

    vals = group["mom"].values
    w = group["peso"].values

    if kind == "mean":
        return _weighted_mean(vals, w)
    if kind == "trimmed":
        lo = _weighted_percentile(vals, w, lower)
        hi = _weighted_percentile(vals, w, upper)
        mask = (vals >= lo) & (vals <= hi)
        if mask.sum() == 0:
            return np.nan
        return _weighted_mean(vals[mask], w[mask])
    if kind == "percentile":
        return _weighted_percentile(vals, w, q)
    if kind == "dp":
        # Aproximação da dupla ponderação: reduz peso de itens voláteis
        adj_w = w / (1 + np.abs(vals))
        if adj_w.sum() == 0:
            return np.nan
        return float(np.average(vals, weights=adj_w))
    return np.nan


def _category_subnuclei_series(df_groups: pd.DataFrame, mask_fn: Callable[[str, str], bool],
                               name: str) -> pd.DataFrame:
    """Calcula séries mensais dos sub-núcleos de uma categoria."""
    leaves = _leaf_items(df_groups)
    leaves["include"] = leaves.apply(lambda r: mask_fn(str(r["item_codigo"]), str(r["item"])), axis=1)
    cat = leaves[leaves["include"]].copy()
    if cat.empty:
        return pd.DataFrame(columns=["period", "mom"])

    records = []
    for period, g in cat.groupby("periodo_codigo"):
        base_mean = _aggregate_group(g, "mean")
        records.append({
            "period": str(period),
            "mom": base_mean,
        })
    return pd.DataFrame(records)


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


def _category_subnuclei(df_groups: pd.DataFrame, mask_fn: Callable[[str, str], bool],
                        name: str, period: str) -> List[Dict]:
    """Retorna linhas de sub-núcleos para Serviços, Industriais ou Alimentação."""
    leaves = _leaf_items(df_groups)
    leaves["include"] = leaves.apply(lambda r: mask_fn(str(r["item_codigo"]), str(r["item"])), axis=1)
    cat = leaves[leaves["include"]].copy()
    if cat.empty:
        return []

    specs: List[Dict] = []
    if name == "Serviços":
        specs = [
            {"label": "Média dos núcleos de Serviços", "kind": "mean"},
            {"label": "Serviços ex-passagem", "kind": "mean",
             "exclude": lambda n: _is_passagem(n)},
            {"label": "Serviços EX3", "kind": "mean",
             "exclude_admin": True},
            {"label": "Serviços MS (20-80)", "kind": "trimmed", "lower": 0.20, "upper": 0.80},
            {"label": "Serviços DP", "kind": "dp"},
            {"label": "Serviços P58", "kind": "percentile", "q": 0.58},
        ]
    elif name == "Industriais":
        specs = [
            {"label": "Média dos núcleos de Industriais", "kind": "mean"},
            {"label": "Industriais ex-etanol, fumo e cosméticos", "kind": "mean",
             "exclude": lambda n: _name_has_any(n, _INDUSTRIAL_EXCLUDE)},
            {"label": "Industriais EX3 ex-cosméticos", "kind": "mean",
             "exclude_admin": True, "exclude": lambda n: _is_cosmetic(n)},
            {"label": "Industriais MS (20-80)", "kind": "trimmed", "lower": 0.20, "upper": 0.80},
            {"label": "Industriais DP", "kind": "dp"},
            {"label": "Industriais P53", "kind": "percentile", "q": 0.53},
        ]
    elif name == "Alimentação":
        specs = [
            {"label": "Média dos núcleos de Alimentação no Domicílio", "kind": "mean"},
            {"label": "Alimentação ex-in natura", "kind": "mean",
             "exclude": lambda n: _is_in_natura(n)},
            {"label": "Alimentação EX2", "kind": "mean",
             "exclude_admin": True, "exclude": lambda n: _is_in_natura(n)},
            {"label": "Alimentação MS (23-83)", "kind": "trimmed", "lower": 0.23, "upper": 0.83},
            {"label": "Alimentação DP", "kind": "dp"},
            {"label": "Alimentação P55", "kind": "percentile", "q": 0.55},
        ]

    series_by_label: Dict[str, List[Dict]] = {s["label"]: [] for s in specs}
    for period_code, g in cat.groupby("periodo_codigo"):
        for spec in specs:
            exclude_mask = None
            if spec.get("exclude_admin"):
                admin_mask = g.apply(lambda r: _is_admin(r["item_codigo"], r["item"]), axis=1)
                exclude_mask = admin_mask.values
            if "exclude" in spec:
                custom_mask = g["item"].apply(spec["exclude"]).values
                if exclude_mask is None:
                    exclude_mask = custom_mask
                else:
                    exclude_mask = exclude_mask | custom_mask

            val = _aggregate_group(
                g, spec["kind"],
                lower=spec.get("lower", 0.0),
                upper=spec.get("upper", 1.0),
                q=spec.get("q", 0.5),
                exclude_mask=exclude_mask if exclude_mask is not None else None,
            )
            series_by_label[spec["label"]].append({"period": str(period_code), "mom": val})

    rows = []
    for spec in specs:
        df = pd.DataFrame(series_by_label[spec["label"]])
        if df.empty or df["mom"].isna().all():
            continue
        row = _subnucleo_row(df, period, spec["label"])
        if row:
            rows.append(row)
    return rows


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
