"""
Gerador do site estático no formato do protótipo.

Busca dados, processa e gera index.html com gráficos Plotly embutidos.
"""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..collectors.brazil.ibge_client import IBGECollector, get_latest_available_period
from ..collectors.brazil.bcb_client import BCBClient
from ..collectors.brazil.focus_client import FocusClient
from ..processors.brazil.resumo import process_resumo, format_period_label
from ..processors.brazil.destaques import process_destaques
from ..processors.brazil.surpresa import process_surpresa
from ..processors.brazil.grupos import process_grupos
from ..processors.brazil import charts as chart_builder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SITE_DIR = PROJECT_ROOT / "site"
DATA_DIR = SITE_DIR / "data" / "brazil"
TEMPLATE_DIR = PROJECT_ROOT / "src" / "builders" / "templates"


def bps_style(value: float) -> str:
    """Gera estilo de fundo para células BPS."""
    if value is None:
        return ""
    v = abs(value)
    alpha = round(min(0.45, 0.05 + v / 30), 3)
    if value >= 0:
        return f"background:rgba(255,139,143,{alpha});color:#1c1c1c"
    return f"background:rgba(139,154,194,{alpha});color:#1c1c1c"


def heatmap_style(value: float) -> str:
    """Gera estilo de fundo para células YoY."""
    if value is None:
        return ""
    # normaliza entre 0 e 8%
    v = max(0, min(abs(value), 8))
    alpha = round(v / 10, 3)
    return f"background:rgba(255,139,143,{alpha});color:#1c1c1c"


def delta_style(current: float, previous: float) -> str:
    if current is None or previous is None:
        return ""
    if current > previous:
        return "color:#EA243E"
    if current < previous:
        return "color:#445486"
    return ""


def delta_str(current: float, previous: float) -> str:
    if current is None or previous is None:
        return "–"
    d = current - previous
    if abs(d) < 0.01:
        return "–"
    arrow = "↑" if d > 0 else "↓"
    return f"{arrow} {abs(d):.2f}"


def _fetch_in_chunks(collector: IBGECollector, start_period: str, end_period: str, chunk_years: int = 2):
    """Busca dados do IBGE em chunks para evitar erro 400 por URL muito longa."""
    start_dt = pd.to_datetime(start_period, format="%Y%m")
    end_dt = pd.to_datetime(end_period, format="%Y%m")

    general_parts = []
    groups_parts = []
    current = start_dt
    while current <= end_dt:
        chunk_end = min(current + pd.DateOffset(years=chunk_years) - pd.DateOffset(months=1), end_dt)
        sp = current.strftime("%Y%m")
        ep = chunk_end.strftime("%Y%m")
        print(f"  Buscando chunk {sp} - {ep}...")
        general_parts.append(collector.fetch_ipca_general(sp, ep))
        groups_parts.append(collector.fetch_ipca_groups_and_subitems(sp, ep))
        current = chunk_end + pd.DateOffset(months=1)

    df_general = pd.concat(general_parts, ignore_index=True)
    df_groups = pd.concat(groups_parts, ignore_index=True)
    return df_general, df_groups


def build_data(start_period: str, end_period: str, collector: IBGECollector = None) -> Dict:
    collector = collector or IBGECollector()
    print(f"Buscando dados de {start_period} a {end_period}...")
    df_general, df_groups = _fetch_in_chunks(collector, start_period, end_period)
    print(f"Dados carregados: {len(df_general)} registros gerais, {len(df_groups)} registros de grupos/subitens")

    latest_period = str(df_general["periodo_codigo"].max())

    # Busca núcleos do BCB
    print("Buscando núcleos do BCB...")
    bcb_client = BCBClient()
    try:
        start_bcb = pd.to_datetime(start_period, format="%Y%m").strftime("%d/%m/%Y")
        end_bcb = pd.to_datetime(end_period, format="%Y%m").strftime("%d/%m/%Y")
        bcb_cores = bcb_client.fetch_ipca_cores(start_date=start_bcb, end_date=end_bcb)
        print(f"Núcleos carregados: {len(bcb_cores)} registros")
    except Exception as e:
        print(f"Aviso: não foi possível carregar núcleos do BCB: {e}")
        bcb_cores = None

    # Busca projeções Focus
    print("Buscando projeções Focus...")
    focus_client = FocusClient()
    try:
        start_focus = pd.to_datetime(start_period, format="%Y%m").strftime("%Y-%m-%d")
        end_focus = pd.to_datetime(end_period, format="%Y%m").strftime("%Y-%m-%d")
        focus_proj = focus_client.fetch_ipca_projections(start_date=start_focus, end_date=end_focus)
        print(f"Projeções carregadas: {len(focus_proj)} períodos")
    except Exception as e:
        print(f"Aviso: não foi possível carregar projeções Focus: {e}")
        focus_proj = None

    resumo = process_resumo(df_general, df_groups, period=latest_period)
    destaques = process_destaques(df_groups[df_groups["periodo_codigo"] == latest_period], top_n=10)
    surpresa = process_surpresa(df_general, df_groups, projections=focus_proj)
    grupos = process_grupos(df_general, df_groups, bcb_cores=bcb_cores, period=latest_period)

    general_series = (
        df_general[["periodo_codigo", "mom", "yoy"]]
        .dropna(subset=["periodo_codigo"])
        .sort_values("periodo_codigo")
        .rename(columns={"periodo_codigo": "period"})
        .to_dict("records")
    )

    # Constrói gráficos de Surpresa
    surpresa_charts = {}
    for cat, series in surpresa.items():
        full_df = pd.DataFrame(series)
        detail_df = full_df.tail(36)
        surpresa_charts[cat] = {
            "main": chart_builder.build_surprise_chart(series),
            "detail": chart_builder.build_detail_chart(detail_df.to_dict("records"), title="DETALHE · ÚLT. 36M"),
        }

    # Constrói gráficos de Grupos
    grupos_data = {}
    for cat, data in grupos.items():
        series = data["series"]
        full_df = pd.DataFrame(series)
        detail_df = full_df.tail(36)

        # Encontra valores anteriores para tabela
        latest = data["latest"]
        prev = {}
        prev12 = {}
        p = pd.Period(latest_period, freq="M")
        prev_period = (p - 1).strftime("%Y%m")
        prev12_period = (p - 12).strftime("%Y%m")
        for key in ["saar1", "saar3", "yoy"]:
            val = full_df[full_df["period"] == prev_period]
            val12 = full_df[full_df["period"] == prev12_period]
            prev[key] = float(val[key].iloc[-1]) if not val.empty and pd.notna(val[key].iloc[-1]) else None
            prev12[key] = float(val12[key].iloc[-1]) if not val12.empty and pd.notna(val12[key].iloc[-1]) else None

        grupos_data[cat] = {
            "latest": latest,
            "prev": prev,
            "prev12": prev12,
            "charts": {
                "main": chart_builder.build_group_chart(series, title=None),
                "detail": chart_builder.build_detail_chart(detail_df.to_dict("records"), title="DETALHE · ÚLT. 36M"),
            }
        }

    return {
        "period": latest_period,
        "period_label": format_period_label(latest_period),
        "period_label_upper": format_period_label(latest_period).upper(),
        "resumo": resumo,
        "destaques": destaques,
        "surpresa_categories": list(surpresa.keys()),
        "surpresa_charts": surpresa_charts,
        "grupos_categories": list(grupos.keys()),
        "grupos_data": grupos_data,
        "general_series": general_series,
        "metadata": {
            "source": "IBGE SIDRA",
            "last_update": pd.Timestamp.now().isoformat(),
            "start_period": start_period,
            "end_period": end_period,
        },
    }


def save_data(data: Dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "latest.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Dados salvos em {DATA_DIR / 'latest.json'}")


def build_static_site(data: Dict) -> None:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    env.filters["bps_style"] = bps_style
    env.filters["heatmap_style"] = heatmap_style
    env.globals["bps_style"] = bps_style
    env.globals["heatmap_style"] = heatmap_style
    env.globals["delta_style"] = delta_style
    env.globals["delta_str"] = delta_str

    template = env.get_template("index.html")
    html = template.render(**data)

    with open(SITE_DIR / "index.html", "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Site gerado em {SITE_DIR / 'index.html'}")


def main():
    end_period = get_latest_available_period()
    start_dt = pd.to_datetime(end_period, format="%Y%m") - pd.DateOffset(years=22)
    start_period = start_dt.strftime("%Y%m")

    data = build_data(start_period, end_period)
    save_data(data)
    build_static_site(data)
    print("Build concluído.")


if __name__ == "__main__":
    main()
