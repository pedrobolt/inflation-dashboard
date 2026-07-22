"""
Processador do painel de inflação dos EUA.

Converte as séries do FRED/Zillow para o mesmo formato de registros
({'period': 'YYYYMM', 'mom', 'yoy', 'saar1', 'saar3', 'saar6', 'meta'})
que os geradores de gráfico do painel Brasil já consomem.
"""

from typing import Dict, List, Optional

import pandas as pd

US_TARGET = 2.0  # Meta do Fed: 2% (PCE); usada como referência em todos os gráficos


def _clean(records: List[Dict]) -> List[Dict]:
    for r in records:
        for k, v in list(r.items()):
            if isinstance(v, float) and pd.isna(v):
                r[k] = None
    return records


def _reindex_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Preenche meses ausentes no calendário (ex.: divulgação pulada por shutdown do governo)
    com NaN, para que .shift()/.rolling() comparem sempre o mês certo, não a posição na lista."""
    s = df.sort_values("date").set_index("date")
    full_range = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    return s.reindex(full_range).rename_axis("date").reset_index()


def _index_to_series(df: pd.DataFrame, meta: Optional[float] = US_TARGET) -> List[Dict]:
    """Nível do índice ['date','value'] -> registros com mom/yoy/saar1/saar3/saar6/meta."""
    s = _reindex_monthly(df)
    s["period"] = s["date"].dt.to_period("M").dt.strftime("%Y%m")
    s["mom"] = (s["value"] / s["value"].shift(1) - 1) * 100
    s["yoy"] = (s["value"] / s["value"].shift(12) - 1) * 100
    s["m3"] = s["mom"].rolling(3, min_periods=3).mean()
    s["m6"] = s["mom"].rolling(6, min_periods=6).mean()
    s["saar1"] = ((1 + s["mom"] / 100) ** 12 - 1) * 100
    s["saar3"] = ((1 + s["m3"] / 100) ** 12 - 1) * 100
    s["saar6"] = ((1 + s["m6"] / 100) ** 12 - 1) * 100
    s["meta"] = meta
    return _clean(s[["period", "mom", "yoy", "saar1", "saar3", "saar6", "meta"]].to_dict("records"))


def _rate_to_series(df: pd.DataFrame, smooth: int = 0) -> List[Dict]:
    """Série já em taxa (%) -> registros com a taxa na chave 'yoy' (p/ multiline)."""
    s = _reindex_monthly(df)
    s["period"] = s["date"].dt.to_period("M").dt.strftime("%Y%m")
    s["yoy"] = s["value"].rolling(smooth, min_periods=smooth).mean() if smooth else s["value"]
    return _clean(s[["period", "yoy"]].to_dict("records"))


def _merge_comparison(a: List[Dict], b: List[Dict], meta: Optional[float] = US_TARGET) -> List[Dict]:
    """Duas séries com yoy/saar3 -> formato index_/core_ do build_comparison_chart."""
    da = pd.DataFrame(a)[["period", "yoy", "saar3"]].rename(
        columns={"yoy": "index_yoy", "saar3": "index_saar3"})
    db = pd.DataFrame(b)[["period", "yoy", "saar3"]].rename(
        columns={"yoy": "core_yoy", "saar3": "core_saar3"})
    merged = da.merge(db, on="period", how="outer").sort_values("period")
    merged["meta"] = meta
    return _clean(merged.to_dict("records"))


def _latest(series: List[Dict], key: str) -> Optional[float]:
    for r in reversed(series):
        if r.get(key) is not None:
            return round(float(r[key]), 2)
    return None


def process_us(fred: Dict[str, pd.DataFrame], zori: Optional[pd.DataFrame]) -> Dict:
    """Monta os dados das 4 abas do painel EUA."""
    idx = {name: _index_to_series(df) for name, df in fred.items()
           if name.startswith(("cpi_", "pce_"))}

    # ── Resumo: momentum do core CPI + CPI vs PCE ──
    momentum = idx["cpi_core"]
    cpi_vs_pce_headline = _merge_comparison(idx["cpi_headline"], idx["pce_headline"])
    cpi_vs_pce_core = _merge_comparison(idx["cpi_core"], idx["pce_core"])

    # ── Composição: os "três baldes" + food & energy ──
    composicao = [
        {"name": "Core goods", "series": idx["cpi_core_goods"]},
        {"name": "Shelter", "series": idx["cpi_shelter"]},
        {"name": "Supercore (serv. ex-shelter)", "series": idx["cpi_supercore"]},
        {"name": "Food", "series": idx["cpi_food"]},
        {"name": "Energy", "series": idx["cpi_energy"]},
    ]

    # ── Shelter: CPI Shelter vs aluguel de mercado (ZORI) ──
    shelter_cmp = []
    if zori is not None and not zori.empty:
        shelter_cmp = _merge_comparison(_index_to_series(zori), idx["cpi_shelter"])

    # ── Expectativas: breakevens + amplitude ──
    expectativas = [
        {"name": "Breakeven 5A", "series": _rate_to_series(fred["breakeven_5y"])},
        {"name": "Breakeven 5A5A forward", "series": _rate_to_series(fred["breakeven_5y5y"])},
        {"name": "Michigan 12m", "series": _rate_to_series(fred["michigan_1y"])},
    ]
    # ponytail: amplitude via núcleos alternativos; "% da cesta acima de 3%" exigiria
    # puxar todos os componentes do CPI — adicionar se os núcleos não bastarem
    amplitude = [
        {"name": "Median CPI (3M MA)", "series": _rate_to_series(fred["median_cpi"], smooth=3)},
        {"name": "Trimmed-Mean 16% (3M MA)", "series": _rate_to_series(fred["trimmed_cpi"], smooth=3)},
        {"name": "Sticky CPI (YoY)", "series": _rate_to_series(fred["sticky_cpi"])},
        {"name": "Core CPI (YoY)", "series": [
            {"period": r["period"], "yoy": r["yoy"]} for r in idx["cpi_core"]]},
    ]

    latest_period = max(r["period"] for r in idx["cpi_core"] if r.get("yoy") is not None)

    return {
        "latest_period": latest_period,
        "momentum": momentum,
        "momentum_latest": {k: _latest(momentum, k) for k in ["saar1", "saar3", "saar6", "yoy"]},
        "cpi_vs_pce_headline": cpi_vs_pce_headline,
        "cpi_vs_pce_core": cpi_vs_pce_core,
        "cpi_pce_latest": {
            "cpi": _latest(idx["cpi_headline"], "yoy"),
            "pce": _latest(idx["pce_headline"], "yoy"),
            "cpi_core": _latest(idx["cpi_core"], "yoy"),
            "pce_core": _latest(idx["pce_core"], "yoy"),
        },
        "composicao": composicao,
        "composicao_latest": [
            {"name": item["name"],
             "yoy": _latest(item["series"], "yoy"),
             "saar3": _latest(item["series"], "saar3")}
            for item in composicao
        ],
        "shelter_cmp": shelter_cmp,
        "shelter_latest": {
            "zori": _latest(shelter_cmp, "index_yoy") if shelter_cmp else None,
            "cpi_shelter": _latest(idx["cpi_shelter"], "yoy"),
        },
        "expectativas": expectativas,
        "expectativas_latest": [
            {"name": item["name"], "value": _latest(item["series"], "yoy")}
            for item in expectativas
        ],
        "amplitude": amplitude,
        "amplitude_latest": [
            {"name": item["name"], "value": _latest(item["series"], "yoy")}
            for item in amplitude
        ],
    }
