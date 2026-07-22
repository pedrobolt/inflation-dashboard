"""
Processador da aba Por Grupos (núcleos e categorias especiais).

Usa séries oficiais do SGS do BCB para as categorias e sub-núcleos,
e a planilha de vetores de agregação para cálculos internos.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set

from .resumo import _extract_code_prefix, _classify_item, _calc_category_series


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


# Mapeamento interno -> nomes das máscaras na planilha de vetores
_VECTOR_MASK_NAMES = {
    "Serviços": "Serviços",
    "Industriais": "Bens industriais",
    "Alimentação": "Alimentação no domicílio",
}

# Ajustes da meta para cada categoria (em pp sobre 3,00%)
_CATEGORY_TARGETS = {
    "BCB": 3.00,
    "Serviços": 4.00,
    "Industriais": 1.25,
    "Alimentação": 3.00,
}


def _name_has_any(name: str, keywords: List[str]) -> bool:
    n = str(name).lower()
    return any(k.lower() in n for k in keywords)


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
                     q: float = 0.5, exclude_mask: Optional[np.ndarray] = None) -> float:
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
        adj_w = w / (1 + np.abs(vals))
        if adj_w.sum() == 0:
            return np.nan
        return float(np.average(vals, weights=adj_w))
    return np.nan


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
    df["m6"] = df["mom"].rolling(6, min_periods=1).mean()
    df["saar1"] = ((1 + df["mom"] / 100) ** 12 - 1) * 100
    df["saar3"] = ((1 + df["m3"] / 100) ** 12 - 1) * 100
    df["saar6"] = ((1 + df["m6"] / 100) ** 12 - 1) * 100
    return df


def _process_official_series(df_mom: pd.DataFrame, meta: float = 3.0) -> pd.DataFrame:
    """Recebe DataFrame com 'period' e 'mom' e retorna série enriquecida."""
    df = df_mom.copy()
    df = df.sort_values("period").reset_index(drop=True)
    df = _calc_saar(_calc_yoy_from_mom(df))
    df["meta"] = meta
    return df


def _clean_records(records: List[Dict]) -> List[Dict]:
    """Substitui NaN por None para serialização JSON segura."""
    for r in records:
        for k in list(r.keys()):
            v = r[k]
            if isinstance(v, float) and pd.isna(v):
                r[k] = None
    return records


def _subnucleo_series(df: pd.DataFrame) -> List[Dict]:
    """Série mensal de mom/yoy para um sub-núcleo."""
    df = _calc_saar(_calc_yoy_from_mom(df.sort_values("period").copy()))
    return _clean_records(df[["period", "mom", "yoy"]].to_dict("records"))


def _subnucleo_row(df: pd.DataFrame, period: str, name: str) -> Optional[Dict]:
    """Monta uma linha da tabela lateral de sub-núcleos."""
    df = _calc_saar(_calc_yoy_from_mom(df))
    prev_period = (pd.Period(period, freq="M") - 1).strftime("%Y%m")
    prev3_period = (pd.Period(period, freq="M") - 3).strftime("%Y%m")
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
    prev3 = get_row_values(prev3_period)
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
        "prev3_saar1": prev3["saar1"],
        "prev3_saar3": prev3["saar3"],
        "prev3_yoy": prev3["yoy"],
        "prev12_saar1": prev12["saar1"],
        "prev12_saar3": prev12["saar3"],
        "prev12_yoy": prev12["yoy"],
        "series": _subnucleo_series(df),
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


def _sgc_subnucleo_row(df_bcb: pd.DataFrame, col: str, period: str, label: str) -> Optional[Dict]:
    """Monta linha de sub-núcleo a partir de uma série do SGS."""
    if df_bcb is None or df_bcb.empty or col not in df_bcb.columns:
        return None
    df = df_bcb[["data", col]].copy()
    df["period"] = df["data"].dt.to_period("M").dt.strftime("%Y%m")
    df = df.rename(columns={col: "mom"}).sort_values("period").reset_index(drop=True)
    row = _subnucleo_row(df, period, label)
    return row


def _heuristic_subnuclei(df_groups: pd.DataFrame, prefixes: Set[str],
                         name: str, period: str) -> List[Dict]:
    """Calcula sub-núcleos heurísticos a partir dos itens oficiais da categoria."""
    leaves = _leaf_items(df_groups)
    leaves = leaves[leaves["prefix"].isin(prefixes)].copy()
    if leaves.empty:
        return []

    if name == "Serviços":
        specs = [
            {"label": "Serviços ex-passagem", "kind": "mean",
             "exclude": lambda n: _is_passagem(n)},
            {"label": "Serviços MS (20-80)", "kind": "trimmed", "lower": 0.20, "upper": 0.80},
            {"label": "Serviços DP", "kind": "dp"},
            {"label": "Serviços P58", "kind": "percentile", "q": 0.58},
        ]
    elif name == "Industriais":
        specs = [
            {"label": "Industriais ex-etanol, fumo e cosméticos", "kind": "mean",
             "exclude": lambda n: _name_has_any(n, _INDUSTRIAL_EXCLUDE)},
            {"label": "Industriais MS (20-80)", "kind": "trimmed", "lower": 0.20, "upper": 0.80},
            {"label": "Industriais DP", "kind": "dp"},
            {"label": "Industriais P53", "kind": "percentile", "q": 0.53},
        ]
    elif name == "Alimentação":
        specs = [
            {"label": "Alimentação ex-in natura", "kind": "mean",
             "exclude": lambda n: _is_in_natura(n)},
            {"label": "Alimentação MS (23-83)", "kind": "trimmed", "lower": 0.23, "upper": 0.83},
            {"label": "Alimentação DP", "kind": "dp"},
            {"label": "Alimentação P55", "kind": "percentile", "q": 0.55},
        ]
    else:
        return []

    series_by_label: Dict[str, List[Dict]] = {s["label"]: [] for s in specs}
    for period_code, g in leaves.groupby("periodo_codigo"):
        for spec in specs:
            exclude_mask = None
            if "exclude" in spec:
                exclude_mask = g["item"].apply(spec["exclude"]).values
            val = _aggregate_group(
                g, spec["kind"],
                lower=spec.get("lower", 0.0),
                upper=spec.get("upper", 1.0),
                q=spec.get("q", 0.5),
                exclude_mask=exclude_mask,
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


def _average_series(rows: List[Dict]) -> pd.DataFrame:
    """Média das séries de sub-núcleos (alinhada por período)."""
    if not rows:
        return pd.DataFrame(columns=["period", "mom"])
    parts = []
    for r in rows:
        if not r.get("series"):
            continue
        df = pd.DataFrame(r["series"])[["period", "mom"]].copy()
        df["period"] = df["period"].astype(str)
        parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["period", "mom"])
    merged = parts[0][["period"]].copy()
    for i, df in enumerate(parts):
        merged = merged.merge(df.rename(columns={"mom": f"mom_{i}"}), on="period", how="outer")
    mom_cols = [c for c in merged.columns if c.startswith("mom_")]
    merged["mom"] = merged[mom_cols].mean(axis=1, skipna=True)
    return merged[["period", "mom"]].dropna(subset=["mom"]).sort_values("period").reset_index(drop=True)


def _average_subnuclei_row(rows: List[Dict], period: str, name: str) -> Optional[Dict]:
    """Calcula a linha 'Média dos núcleos de {categoria}'."""
    if not rows:
        return None
    avg_series = _average_series(rows)
    if avg_series.empty:
        return None
    row = _subnucleo_row(avg_series, period, name)
    if row is None:
        return None
    return row


def _nucleo_order(name: str, cat: str) -> int:
    """Ordem dos sub-núcleos dentro de cada categoria (igual ao protótipo)."""
    order_map = {
        "Serviços": [
            "Serviços ex-passagem",
            "Serviços EX3",
            "Serviços MS (20-80)",
            "Serviços DP",
            "Serviços P58",
        ],
        "Industriais": [
            "Industriais ex-etanol, fumo e cosméticos",
            "Industriais EX3 ex-cosméticos",
            "Industriais MS (20-80)",
            "Industriais DP",
            "Industriais P53",
        ],
        "Alimentação": [
            "Alimentação ex-in natura",
            "Alimentação EX2",
            "Alimentação MS (23-83)",
            "Alimentação DP",
            "Alimentação P55",
        ],
    }
    names = order_map.get(cat, [])
    try:
        return names.index(name)
    except ValueError:
        return 100


def _category_leaf_prefixes(name: str, df_groups: pd.DataFrame) -> Set[str]:
    """Retorna prefixos folha de uma categoria usando classificação do IBGE."""
    leaves = _leaf_items(df_groups)
    key = {"Serviços": "servico", "Industriais": "industrial", "Alimentação": "alimentacao_domicilio"}[name]
    mask = leaves.apply(lambda r: _classify_item(str(r["item_codigo"]), str(r["item"]))[key], axis=1)
    return set(leaves.loc[mask, "prefix"].unique())


def _category_subnuclei(name: str, df_groups: pd.DataFrame, masks: Dict[str, Set[str]],
                        bcb_subnuclei: Optional[pd.DataFrame], period: str) -> List[Dict]:
    """Retorna linhas de sub-núcleos para Serviços, Industriais ou Alimentação."""
    vector_key = _VECTOR_MASK_NAMES.get(name)
    vector_prefixes = masks.get(vector_key, set()) if masks else set()

    leaves = _leaf_items(df_groups)
    prefixes = set(leaves[leaves["prefix"].isin(vector_prefixes)]["prefix"].unique())
    if not prefixes:
        # Fallback para classificação IBGE se a máscara vetorial não cobrir folhas
        prefixes = _category_leaf_prefixes(name, df_groups)
    if not prefixes:
        return []

    rows = _heuristic_subnuclei(df_groups, prefixes, name, period)

    # Substitui/adiciona sub-núcleos oficiais do SGS
    official_map = {
        "Serviços": ("EX3 Serviços", "Serviços EX3"),
        "Industriais": ("EX3 Industriais", "Industriais EX3 ex-cosméticos"),
        "Alimentação": ("EX2", "Alimentação EX2"),
    }
    if name in official_map:
        col, label = official_map[name]
        official = _sgc_subnucleo_row(bcb_subnuclei, col, period, label)
        if official:
            # Remove linha heurística equivalente, se existir
            rows = [r for r in rows if r["name"] != label]
            rows.append(official)

    # Ordena para coincidir com o protótipo
    rows.sort(key=lambda r: _nucleo_order(r["name"], name))

    avg_label = f"Média dos núcleos de {name}"
    if name == "Alimentação":
        avg_label = "Média dos núcleos de Alimentação no Domicílio"
    avg_row = _average_subnuclei_row(rows, period, avg_label)
    if avg_row:
        rows.insert(0, avg_row)
    return rows


def _headline_series(name: str, df_general: pd.DataFrame, bcb_categories: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Série de referência (headline) para a seção Geral."""
    if name == "BCB":
        df = df_general[["periodo_codigo", "mom", "yoy"]].copy()
        df = df.rename(columns={"periodo_codigo": "period"}).sort_values("period").reset_index(drop=True)
        return df
    if bcb_categories is not None and name in bcb_categories.columns:
        df = bcb_categories[["data", name]].copy()
        df["period"] = df["data"].dt.to_period("M").dt.strftime("%Y%m")
        df = df.rename(columns={name: "mom"}).sort_values("period").reset_index(drop=True)
        return df
    return pd.DataFrame(columns=["period", "mom"])


def process_grupos(
    df_general: pd.DataFrame,
    df_groups: pd.DataFrame,
    bcb_cores: Optional[pd.DataFrame] = None,
    bcb_categories: Optional[pd.DataFrame] = None,
    bcb_subnuclei: Optional[pd.DataFrame] = None,
    bcb_vectors: Optional[object] = None,
    period: str = None,
) -> Dict[str, Dict]:
    """
    Processa dados para a aba Por Grupos.

    Categorias:
      - BCB: núcleos do BCB (média EX0/EX3/MS/DP/P55)
      - Serviços, Industriais, Alimentação: séries oficiais do SGS + sub-núcleos
    """
    df_general = df_general.sort_values("periodo_codigo").copy()
    if period is None:
        period = str(df_general["periodo_codigo"].max())

    masks = None
    if bcb_vectors is not None:
        try:
            masks = bcb_vectors.masks(df_groups, period)
        except Exception:
            masks = None

    # Categoria BCB - oficial ou fallback para IPCA geral
    categories = {}
    if bcb_cores is not None and not bcb_cores.empty and "media" in bcb_cores.columns:
        bcb_df = bcb_cores[["data", "media"]].copy()
        bcb_df["period"] = bcb_df["data"].dt.to_period("M").dt.strftime("%Y%m")
        bcb_df = bcb_df.rename(columns={"media": "mom"}).sort_values("period").reset_index(drop=True)
        categories["BCB"] = bcb_df

    # Categorias oficiais do SGS - com fallback para agregação dos itens do IBGE
    official_categories = {}
    for cat in ["Serviços", "Industriais", "Alimentação"]:
        if bcb_categories is not None and cat in bcb_categories.columns:
            df_cat = bcb_categories[["data", cat]].copy()
            df_cat["period"] = df_cat["data"].dt.to_period("M").dt.strftime("%Y%m")
            df_cat = df_cat.rename(columns={cat: "mom"}).sort_values("period").reset_index(drop=True)
            official_categories[cat] = df_cat
        else:
            mask_fn = {
                "Serviços": lambda c, n: _classify_item(c, n)["servico"],
                "Industriais": lambda c, n: _classify_item(c, n)["industrial"],
                "Alimentação": lambda c, n: _classify_item(c, n)["alimentacao_domicilio"],
            }[cat]
            df_cat = _calc_category_series(df_groups, mask_fn)
            if not df_cat.empty:
                df_cat = df_cat.rename(columns={"periodo_codigo": "period"})[["period", "mom"]].copy()
                official_categories[cat] = df_cat

    result = {}
    for name in ["BCB", "Serviços", "Industriais", "Alimentação"]:
        if name == "BCB":
            if "BCB" in categories:
                subnuclei = _bcb_subnuclei(bcb_cores, period)
                main_df = categories["BCB"]
            else:
                # Fallback: IPCA geral sem sub-núcleos
                subnuclei = []
                main_df = df_general[["periodo_codigo", "mom"]].copy()
                main_df = main_df.rename(columns={"periodo_codigo": "period"}).sort_values("period").reset_index(drop=True)
        else:
            if name not in official_categories:
                continue
            subnuclei = _category_subnuclei(name, df_groups, masks, bcb_subnuclei, period)
            if not subnuclei:
                continue
            main_df = _average_series(subnuclei)
            if main_df.empty:
                continue

        target = _CATEGORY_TARGETS.get(name, 3.00)
        main_df = _process_official_series(main_df, meta=target)
        headline_df = _headline_series(name, df_general, bcb_categories)
        headline_df = _process_official_series(headline_df, meta=target) if not headline_df.empty else headline_df

        last = main_df[main_df["period"] == period]
        if last.empty:
            continue
        last = last.iloc[-1]

        result[name] = {
            "series": _clean_records(main_df[["period", "mom", "yoy", "saar1", "saar3", "saar6", "meta"]].to_dict("records")),
            "subnuclei": subnuclei,
            "headline_series": _clean_records(headline_df[["period", "mom", "yoy", "saar1", "saar3", "saar6", "meta"]].to_dict("records")) if not headline_df.empty else [],
            "latest": {
                "saar1": float(last["saar1"]) if pd.notna(last["saar1"]) else None,
                "saar3": float(last["saar3"]) if pd.notna(last["saar3"]) else None,
                "saar6": float(last["saar6"]) if pd.notna(last["saar6"]) else None,
                "yoy": float(last["yoy"]) if pd.notna(last["yoy"]) else None,
                "target": target,
                "target_deviation": float(last["yoy"] - target) if pd.notna(last["yoy"]) else None,
            }
        }

    return result
