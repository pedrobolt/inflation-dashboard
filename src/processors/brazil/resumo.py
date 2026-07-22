"""
Processador de dados do Resumo do IPCA no formato do protótipo.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple


# Grupos principais do IPCA (tabela 7060)
GROUP_CODES = {
    "7169": "Índice Geral",
    "7170": "Alimentação e bebidas",
    "7445": "Habitação",
    "7486": "Artigos de residência",
    "7558": "Vestuário",
    "7625": "Transportes",
    "7660": "Saúde e cuidados pessoais",
    "7712": "Despesas pessoais",
    "7766": "Educação",
    "7786": "Comunicação",
}

# Códigos de subitens considerados administrados (preços regulados/tabelados)
ADMINISTRADOS_CODES = {
    "2202003", "2201004", "2201005",  # energia/gás
    "5104001", "5104002", "5104003",  # combustíveis
    "5101001", "5101002", "5101004", "5101006", "5101007", "5101010", "5101011", "5101026",  # transporte
    "8101003", "8101004", "8101005", "8101006", "8101045",  # educação
    "6203001",  # plano de saúde
    "2101004",  # taxa de água/esgoto
    "2201004",  # gás de botijão (duplicado)
}

# Códigos de combustíveis (para excluir de administrados ex-combustíveis)
COMBUSTIVEIS_CODES = {"5104001", "5104002", "5104003", "2201004", "2201005"}


def _extract_code_prefix(name: str) -> str:
    """Extrai o prefixo hierárquico do nome (ex: '2201.Combustíveis' -> '2201')."""
    name = str(name)
    if "." in name:
        prefix = name.split(".", 1)[0]
        if prefix.isdigit():
            return prefix
    return ""


def _is_service(code_prefix: str) -> bool:
    """Classifica item como serviço pelo prefixo hierárquico."""
    return code_prefix in {"3301", "5101", "6201", "6202", "6203", "7101", "7201", "8101", "8104", "9101"}


def _is_industrial(code_prefix: str) -> bool:
    """Classifica item como bem industrial pelo prefixo hierárquico."""
    return code_prefix in {"3101", "3102", "3103", "3201", "3202", "4101", "4102", "4103",
                           "4201", "4301", "4401", "6101", "6301", "7202"}


def _is_passagem(name: str) -> bool:
    """Identifica itens de passagem aérea."""
    return "passagem aérea" in str(name).lower()


def _classify_item(code: str, name: str) -> Dict[str, bool]:
    prefix = _extract_code_prefix(name)
    prefix4 = prefix[:4] if len(prefix) >= 4 else prefix
    name_lower = str(name).lower()
    # Administrados: preços regulados/tabelados
    is_admin = prefix4 in {"2201", "2202", "5101", "5104", "8101", "8104", "6203", "2101"}
    is_comb = prefix4 in {"2201", "5104"}
    is_service = _is_service(prefix4)
    is_industrial = _is_industrial(prefix4)
    is_alimentacao_domicilio = prefix.startswith("11") and prefix != "1101"
    return {
        "administrado": is_admin,
        "admin_ex_comb": is_admin and not is_comb,
        "livre": not is_admin,
        "alimentacao_domicilio": is_alimentacao_domicilio,
        "servico": is_service,
        "industrial": is_industrial,
    }


def _weighted_mom(df: pd.DataFrame) -> float:
    """Calcula variação mensal agregada por média ponderada de pesos do período anterior."""
    df = df.sort_values("periodo").reset_index(drop=True)
    if len(df) < 2:
        return None
    # Usa peso do mês anterior como proxy para cálculo de contribuição
    df["peso_lag"] = df["peso"].shift(1)
    df["contrib"] = df["mom"] * df["peso_lag"]
    last = df.iloc[-1]
    denom = df["peso_lag"].sum()
    if denom == 0 or pd.isna(denom):
        return None
    return float(df["contrib"].sum() / denom)


def _weighted_yoy(df: pd.DataFrame) -> float:
    """Calcula YoY agregado como média ponderada dos YoY dos subitens no período."""
    last = df.iloc[-1]
    if pd.isna(last.get("peso")) or last["peso"] == 0:
        return None
    total_weight = df["peso"].sum()
    if total_weight == 0:
        return None
    return float((df["yoy"] * df["peso"]).sum() / total_weight)


def _calc_category_series(df_groups: pd.DataFrame, mask_fn) -> pd.DataFrame:
    """Calcula série mensal de MOM e YOY para uma categoria definida por mask_fn.

    Usa apenas itens folha (nível mais baixo) para evitar dupla contagem.
    Um item é considerado folha se não possuir filhos na tabela.
    """
    df = df_groups.copy()
    df["prefix"] = df["item"].apply(_extract_code_prefix)
    df = df[df["prefix"] != ""]

    # Identifica folhas: prefixos que não são prefixo de nenhum outro prefixo
    all_prefixes = set(df["prefix"].unique())
    leaf_prefixes = {p for p in all_prefixes if not any(
        q != p and q.startswith(p) and len(q) > len(p) for q in all_prefixes
    )}
    df = df[df["prefix"].isin(leaf_prefixes)]

    df["include"] = df.apply(lambda r: mask_fn(str(r["item_codigo"]), str(r["item"])), axis=1)
    df = df[df["include"]].copy()
    if df.empty:
        return pd.DataFrame(columns=["periodo", "periodo_codigo", "mom", "yoy"])

    # Agrega por período
    periods = sorted(df["periodo_codigo"].unique())
    rows = []
    for period in periods:
        dfp = df[df["periodo_codigo"] == period].copy()
        total_weight = dfp["peso"].sum()
        if total_weight == 0:
            continue
        mom = (dfp["mom"] * dfp["peso"]).sum() / total_weight
        yoy = None
        if dfp["yoy"].notna().any():
            yoy = (dfp["yoy"].fillna(0) * dfp["peso"]).sum() / total_weight
        rows.append({
            "periodo": dfp["periodo"].iloc[0],
            "periodo_codigo": str(period),
            "mom": float(mom),
            "yoy": float(yoy) if yoy is not None else None,
        })
    return pd.DataFrame(rows)


def _last_period_row(series_df: pd.DataFrame, period: str) -> pd.Series:
    row = series_df[series_df["periodo_codigo"] == period]
    if row.empty:
        return pd.Series({"mom": None, "yoy": None})
    return row.iloc[0]


def _lag_value(series_df: pd.DataFrame, period: str, col: str, lag: int) -> float:
    df = series_df.sort_values("periodo").reset_index(drop=True)
    idx = df[df["periodo_codigo"] == period].index
    if len(idx) == 0:
        return None
    i = idx[0] - lag
    if i < 0 or i >= len(df):
        return None
    val = df.iloc[i][col]
    return float(val) if pd.notna(val) else None


def _core_yoy_series(core_df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Calcula YoY de uma série de núcleo a partir do índice acumulado."""
    df = core_df[["data", col]].copy().rename(columns={col: "mom"})
    df["periodo"] = df["data"].dt.to_period("M").dt.strftime("%Y%m")
    df["periodo_codigo"] = df["periodo"]
    df = df.sort_values("data").reset_index(drop=True)
    df["index"] = (1 + df["mom"] / 100).cumprod()
    df["yoy"] = (df["index"] / df["index"].shift(12) - 1) * 100
    df["yoy"] = df["yoy"].where(df["index"].shift(12).notna())
    return df[["periodo", "periodo_codigo", "mom", "yoy"]]


def process_resumo(
    df_general: pd.DataFrame,
    df_groups: pd.DataFrame,
    bcb_cores: Optional[pd.DataFrame] = None,
    period: str = None,
) -> List[Dict]:
    """
    Processa tabela resumo idêntica ao protótipo.
    Retorna lista de dicionários com: metric, weight, mom_t_12, mom_t_2, mom_t_1, mom,
    mom_bps, yoy_t_12, yoy_t_2, yoy_t_1, yoy.
    """
    df_general = df_general.copy()
    df_general["item_codigo"] = "7169"
    df_general["item"] = "Índice Geral"

    # Prepara grupos principais
    df_groups_main = df_groups[df_groups["item_codigo"].astype(str).isin(GROUP_CODES.keys())].copy()
    general_cols = ["periodo", "periodo_codigo", "item", "item_codigo", "mom", "yoy"]
    groups_cols = ["periodo", "periodo_codigo", "item", "item_codigo", "mom", "peso"]
    if "yoy" in df_groups_main.columns:
        groups_cols.append("yoy")

    df_combined = pd.concat([
        df_general[general_cols],
        df_groups_main[groups_cols],
    ], ignore_index=True)
    df_combined = df_combined.sort_values(["item_codigo", "periodo"])

    if period is None:
        period = str(df_combined["periodo_codigo"].max())

    # Pesos no período alvo (para grupos principais)
    target_weights = (
        df_combined[df_combined["periodo_codigo"] == period]
        .dropna(subset=["peso"])
        .set_index("item_codigo")["peso"]
        .to_dict()
    )

    # Calcula séries especiais
    admin_df = _calc_category_series(df_groups, lambda c, n: _classify_item(c, n)["administrado"])
    admin_ex_df = _calc_category_series(df_groups, lambda c, n: _classify_item(c, n)["admin_ex_comb"])
    livres_df = _calc_category_series(df_groups, lambda c, n: _classify_item(c, n)["livre"])
    alim_df = _calc_category_series(df_groups, lambda c, n: _classify_item(c, n)["alimentacao_domicilio"])
    serv_df = _calc_category_series(df_groups, lambda c, n: _classify_item(c, n)["servico"])
    serv_ex_pass_df = _calc_category_series(
        df_groups, lambda c, n: _classify_item(c, n)["servico"] and not _is_passagem(n)
    )
    ind_df = _calc_category_series(df_groups, lambda c, n: _classify_item(c, n)["industrial"])

    special_series = {
        "Administrados": admin_df,
        "Administrados ex-combustíveis": admin_ex_df,
        "Livres": livres_df,
        "Alimentação no Domicílio": alim_df,
        "Industriais": ind_df,
        "Serviços": serv_df,
        "Serviços ex-passagem": serv_ex_pass_df,
    }

    # Pesos das categorias especiais (soma dos pesos no período alvo, usando folhas)
    special_weights = {}
    for name, s_df in special_series.items():
        if s_df.empty:
            special_weights[name] = 0.0
            continue
        # pega pesos do período alvo usando a mesma lógica de folhas
        df_p = df_groups[df_groups["periodo_codigo"] == period].copy()
        df_p["prefix"] = df_p["item"].apply(_extract_code_prefix)
        df_p = df_p[df_p["prefix"] != ""]
        all_prefixes = set(df_p["prefix"].unique())
        leaf_prefixes = {p for p in all_prefixes if not any(
            q != p and q.startswith(p) and len(q) > len(p) for q in all_prefixes
        )}
        df_p = df_p[df_p["prefix"].isin(leaf_prefixes)]
        if name == "Administrados":
            mask = df_p.apply(lambda r: _classify_item(r["item_codigo"], r["item"])["administrado"], axis=1)
        elif name == "Administrados ex-combustíveis":
            mask = df_p.apply(lambda r: _classify_item(r["item_codigo"], r["item"])["admin_ex_comb"], axis=1)
        elif name == "Livres":
            mask = df_p.apply(lambda r: _classify_item(r["item_codigo"], r["item"])["livre"], axis=1)
        elif name == "Alimentação no Domicílio":
            mask = df_p.apply(lambda r: _classify_item(r["item_codigo"], r["item"])["alimentacao_domicilio"], axis=1)
        elif name == "Industriais":
            mask = df_p.apply(lambda r: _classify_item(r["item_codigo"], r["item"])["industrial"], axis=1)
        elif name == "Serviços":
            mask = df_p.apply(lambda r: _classify_item(r["item_codigo"], r["item"])["servico"], axis=1)
        elif name == "Serviços ex-passagem":
            mask = df_p.apply(
                lambda r: _classify_item(r["item_codigo"], r["item"])["servico"] and not _is_passagem(r["item"]),
                axis=1,
            )
        else:
            mask = pd.Series(False, index=df_p.index)
        special_weights[name] = float(df_p.loc[mask, "peso"].sum()) if mask.any() else 0.0

    # Núcleos do BCB (linhas sem peso / sem BPS)
    core_series: Dict[str, pd.DataFrame] = {}
    if bcb_cores is not None and not bcb_cores.empty:
        core_cols = [c for c in ["EX0", "EX3", "MS", "DP", "P55"] if c in bcb_cores.columns]
        for col in core_cols:
            core_series[f"IPCA-{col}"] = _core_yoy_series(bcb_cores, col)
        if "media" in bcb_cores.columns:
            core_series["Média dos núcleos"] = _core_yoy_series(bcb_cores, "media")

    # Ordem das linhas no protótipo
    ordered_metrics = [
        "Índice Geral",
        "Administrados",
        "Administrados ex-combustíveis",
        "Livres",
        "Alimentação no Domicílio",
        "Industriais",
        "Serviços",
        "Serviços ex-passagem",
        "Média dos núcleos",
        "IPCA-EX0",
        "IPCA-EX3",
        "IPCA-MS",
        "IPCA-DP",
        "IPCA-P55",
    ]

    result = []
    for name in ordered_metrics:
        is_core = name in core_series
        is_special = name in special_series
        is_general = name == "Índice Geral"

        if is_general:
            df_item = df_combined[df_combined["item_codigo"] == "7169"].sort_values("periodo").reset_index(drop=True)
            weight = 100.0
        elif is_special:
            df_item = special_series[name].copy()
            weight = special_weights[name]
        elif is_core:
            df_item = core_series[name].copy()
            weight = None
        else:
            continue

        if df_item.empty:
            continue

        idx_target = df_item[df_item["periodo_codigo"] == period].index
        if len(idx_target) == 0:
            continue
        idx = idx_target[0]

        mom = float(df_item.loc[idx, "mom"]) if pd.notna(df_item.loc[idx, "mom"]) else None
        yoy = float(df_item.loc[idx, "yoy"]) if pd.notna(df_item.loc[idx, "yoy"]) else None

        mom_t_1 = _lag_value(df_item, period, "mom", 1)
        mom_t_2 = _lag_value(df_item, period, "mom", 2)
        mom_t_12 = _lag_value(df_item, period, "mom", 12)
        yoy_t_1 = _lag_value(df_item, period, "yoy", 1)
        yoy_t_2 = _lag_value(df_item, period, "yoy", 2)
        yoy_t_12 = _lag_value(df_item, period, "yoy", 12)

        # BPS no IPCA geral = mom * 100 (para chegar em bps)
        mom_bps = mom * 100 if (mom is not None and weight is not None) else None

        result.append({
            "metric": name,
            "weight": round(weight, 1) if weight is not None else None,
            "mom_t_12": mom_t_12,
            "mom_t_2": mom_t_2,
            "mom_t_1": mom_t_1,
            "mom": mom,
            "mom_bps": mom_bps,
            "yoy_t_12": yoy_t_12,
            "yoy_t_2": yoy_t_2,
            "yoy_t_1": yoy_t_1,
            "yoy": yoy,
        })

    return result


def format_period_label(period_code: str) -> str:
    """Converte '202504' para 'abr/26'."""
    p = pd.Period(period_code, freq="M")
    months = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
    return f"{months[p.month - 1]}/{str(p.year)[-2:]}"
