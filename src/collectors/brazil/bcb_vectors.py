"""
Coletor dos vetores de agregação oficiais do BCB para séries analíticas do IPCA.

A planilha Vetores_NT_57.xlsx define quais componentes do IPCA compõem cada
série analítica (EX0, EX3, Serviços, Bens industriais, etc.).
Fonte: Nota Técnica 57 do BCB.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import requests


VECTOR_URL = "https://www.bcb.gov.br/content/publicacoes/notastecnicas_arq/Vetores_NT_57.xlsx"

# Mapeamento dos períodos de vigência das estruturas do IPCA para as abas da planilha
_SHEET_PERIODS = [
    ("jan20-presente", 202001),
    ("jan12-dez19", 201201),
    ("jul06-dez11", 200607),
    ("jan06-jun06", 200601),
    ("ago99-dez05", 199908),
    ("jan91-jul99", 199101),
]

# Colunas de séries que queremos expor
_SERIES_COLUMNS = [
    "Administrados",
    "Livres",
    "Alimentação no domicílio",
    "Serviços",
    "Bens industriais",
    "Comercializáveis",
    "Não comercializáveis",
    "Bens não duráveis",
    "Bens semiduráveis",
    "Bens duráveis",
    "Núcleo EX-FE",
    "Núcleo EX0",
    "Núcleo EX1",
    "Núcleo EX2",
    "Núcleo EX3",
    "EX3 Serviços",
    "EX3 Industriais",
]


def _find_sheet(period_code: str) -> str:
    """Escolhe a aba da planilha mais adequada para o período informado."""
    p = int(period_code)
    for sheet, start in _SHEET_PERIODS:
        if p >= start:
            return sheet
    return _SHEET_PERIODS[-1][0]


def _component_prefix(name: str) -> str:
    """Extrai o prefixo numérico antes do ponto (ex: '1101002.Arroz' -> '1101002')."""
    name = str(name)
    if "." not in name:
        return ""
    parts = name.split(".", 1)
    if parts[0].isdigit():
        return parts[0]
    return ""


def download_vector(cache_path: Path) -> None:
    """Baixa a planilha de vetores do BCB para o caminho informado."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(VECTOR_URL, timeout=120)
    response.raise_for_status()
    cache_path.write_bytes(response.content)


def load_vector_df(cache_path: Path, period_code: str) -> pd.DataFrame:
    """Carrega a planilha de vetores e retorna apenas as colunas de interesse."""
    if not cache_path.exists():
        download_vector(cache_path)
    sheet = _find_sheet(period_code)
    df = pd.read_excel(cache_path, sheet_name=sheet)
    # A primeira coluna contém o nome do componente
    name_col = df.columns[0]
    df = df.rename(columns={name_col: "componente"})
    df["prefix"] = df["componente"].apply(_component_prefix)
    # Descarta colunas não utilizadas
    keep_cols = ["componente", "prefix"] + [c for c in _SERIES_COLUMNS if c in df.columns]
    return df[keep_cols]


def _normalize_name(col: str) -> str:
    """Normaliza nome de série para uso interno (sem acentos e iniciais minúsculas)."""
    col = col.strip()
    mapping = {
        "Núcleo EX0": "EX0",
        "Núcleo EX1": "EX1",
        "Núcleo EX2": "EX2",
        "Núcleo EX3": "EX3",
        "Núcleo EX-FE": "EX-FE",
    }
    return mapping.get(col, col)


class BCBVectors:
    """Vetores de agregação do BCB para séries analíticas do IPCA."""

    def __init__(self, cache_path: Optional[Path] = None):
        if cache_path is None:
            self.cache_path = Path(__file__).resolve().parents[3] / "data" / "bcb_vectors.xlsx"
        else:
            self.cache_path = Path(cache_path)
        self._df: Optional[pd.DataFrame] = None
        self._period_code: Optional[str] = None

    def load(self, period_code: str) -> None:
        self._df = load_vector_df(self.cache_path, period_code)
        self._period_code = period_code

    def _ensure_loaded(self, period_code: str) -> pd.DataFrame:
        if self._df is None or self._period_code != period_code:
            self.load(period_code)
        return self._df

    def weights(self, df_groups: pd.DataFrame, period_code: str) -> Dict[str, float]:
        """Retorna pesos oficiais das séries analíticas no período."""
        df_vec = self._ensure_loaded(period_code)
        # Prepara grupos do IBGE com prefixo do item
        df_groups = df_groups[df_groups["periodo_codigo"] == period_code].copy()
        df_groups["prefix"] = df_groups["item"].apply(_component_prefix)
        # Considera apenas linhas com prefixo válido
        df_groups = df_groups[df_groups["prefix"] != ""]
        # Faz merge com os vetores (todos os componentes, não apenas folhas)
        merged = df_groups[["prefix", "peso"]].merge(df_vec, on="prefix", how="inner")
        result = {}
        for col in _SERIES_COLUMNS:
            if col not in df_vec.columns:
                continue
            w = merged.loc[merged[col] == 1, "peso"].sum()
            result[_normalize_name(col)] = float(w)
        return result

    def masks(self, df_groups: pd.DataFrame, period_code: str) -> Dict[str, Set[str]]:
        """Retorna conjunto de prefixos de itens pertencentes a cada série."""
        df_vec = self._ensure_loaded(period_code)
        df_groups = df_groups[df_groups["periodo_codigo"] == period_code].copy()
        df_groups["prefix"] = df_groups["item"].apply(_component_prefix)
        prefixes = set(df_groups["prefix"].dropna())
        result = {}
        for col in _SERIES_COLUMNS:
            if col not in df_vec.columns:
                continue
            series_prefixes = set(df_vec.loc[df_vec[col] == 1, "prefix"].dropna())
            result[_normalize_name(col)] = series_prefixes & prefixes
        return result
