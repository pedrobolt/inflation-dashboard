"""
Processador de dados dos Destaques do IPCA no formato do protótipo.

Calcula contribuição de cada subitem em pontos-base (bps):
    contribuicao_bps = peso * variacao_mom
"""

import pandas as pd
from typing import Dict, List


def _clean_item_name(name: str) -> str:
    name = str(name)
    if "." in name and name[0].isdigit():
        name = name.split(".", 1)[1]
    return name.strip()


def _extract_parent(name: str) -> str:
    """Extrai o grupo imediato do subitem (texto antes do último ponto)."""
    name = str(name)
    if "." in name and name[0].isdigit():
        parts = name.split(".")
        # ex: 1101002.Arroz -> parent 'Cereais...' não direto; simplifica com o texto antes do último ponto
        if len(parts) >= 2:
            return parts[-2]
    return ""


from .resumo import _extract_code_prefix


def _is_leaf_item(df: pd.DataFrame) -> pd.Series:
    """Retorna máscara com itens folha (sem filhos na tabela)."""
    prefixes = df["item"].apply(_extract_code_prefix)
    valid = prefixes != ""
    all_prefixes = set(prefixes[valid].unique())
    leaf_prefixes = {p for p in all_prefixes if not any(
        q != p and q.startswith(p) and len(q) > len(p) for q in all_prefixes
    )}
    return prefixes.isin(leaf_prefixes) & valid


def process_destaques(df_groups: pd.DataFrame, top_n: int = 10) -> Dict[str, List[Dict]]:
    df = df_groups.copy().reset_index(drop=True)
    # Usa apenas itens folha (subitens) para evitar duplicação
    df = df[_is_leaf_item(df).values]
    # Exclui o índice geral
    df = df[df["item_codigo"].astype(str) != "7169"]
    df = df.dropna(subset=["peso", "mom"])
    df["contribuicao_bps"] = df["peso"] * df["mom"]

    positive = df[df["contribuicao_bps"] > 0].nlargest(top_n, "contribuicao_bps")
    negative = df[df["contribuicao_bps"] < 0].nsmallest(top_n, "contribuicao_bps")

    def format_item(row):
        full = str(row["item"])
        return {
            "item_codigo": str(row["item_codigo"]),
            "name": _clean_item_name(full),
            "parent": _clean_item_name(_extract_parent(full)),
            "weight": round(float(row["peso"]), 2),
            "mom": round(float(row["mom"]), 2),
            "bps": round(float(row["contribuicao_bps"]), 1),
        }

    return {
        "positive": [format_item(row) for _, row in positive.iterrows()],
        "negative": [format_item(row) for _, row in negative.iterrows()],
    }
