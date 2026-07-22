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
from ..collectors.brazil.bcb_vectors import BCBVectors
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


def _fmt_bps(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    return f"{int(round(float(v))):+d}"


def _category_display(cat: str) -> str:
    return {"Alimentação": "Alimentação no domicílio"}.get(cat, cat)


def _target_legend(target: float, yoy: float, show_deviation: bool = True) -> str:
    if not show_deviation or yoy is None or (isinstance(yoy, float) and pd.isna(yoy)):
        return f"Meta BCB ({target:.2f}%)"
    diff = yoy - target
    sign = "+" if diff >= 0 else "−"
    return f"Meta BCB {sign}{abs(diff):.2f}pp ({target:.2f}%)"


def _short_nucleo_label(name: str, cat: str) -> str:
    short = name
    for prefix in ["Serviços ", "Industriais ", "Alimentação ", "IPCA-"]:
        if short.startswith(prefix):
            short = short[len(prefix):]
    short = short.replace("MS (20-80)", "MS").replace("MS (23-83)", "MS")
    if short.lower().startswith("ex-"):
        short = short[0].upper() + short[1:]
    if cat == "Industriais" and "ex-etanol" in short.lower():
        short = "Ex-etanol, fumo & cosméticos"
    return short


def _nucleos_used_text(cat: str, names: List[str]) -> str:
    if cat == "BCB":
        labels = ["EX0", "EX3", "MS", "DP", "P55"]
    else:
        labels = [_short_nucleo_label(n, cat) for n in names]
    if len(labels) > 1:
        return ", ".join(labels[:-1]) + " e " + labels[-1]
    return labels[0] if labels else ""


def _subnucleo_color(i: int) -> str:
    colors = [
        chart_builder.COLORS["primary"],
        chart_builder.COLORS["green"],
        chart_builder.COLORS["orange"],
        chart_builder.COLORS["pink"],
        chart_builder.COLORS["dark_grey"],
    ]
    return colors[i % len(colors)]


def _merge_comparison_series(headline: List[Dict], core: List[Dict]) -> List[Dict]:
    """Combina série headline (índice/categoria) e série núcleo em uma única série."""
    hl = pd.DataFrame(headline)
    cr = pd.DataFrame(core)
    if hl.empty or cr.empty:
        return []
    hl = hl[["period", "yoy"]].rename(columns={"yoy": "index"})
    cr = cr[["period", "yoy", "meta"]].rename(columns={"yoy": "core"})
    merged = hl.merge(cr, on="period", how="outer").sort_values("period")
    return merged.to_dict("records")


# Paletas de cores para o heatmap do Resumo (mesmas do protótipo)
_LOW_COLOR = (139, 154, 194)
_CENTER_COLOR = (226, 228, 235)
_HIGH_COLOR = (255, 139, 143)
_BPS_START = (246, 242, 244)


def _interpolate(c1: tuple, c2: tuple, t: float) -> tuple:
    """Interpolação linear entre duas cores RGB."""
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _heatmap_style(value: float, cap: float = 8.0) -> str:
    """Gera estilo de fundo para células YoY (divergente, centrado em 2.5%)."""
    if value is None:
        return ""
    if value <= 0:
        color = _LOW_COLOR
    elif value < 2.5:
        color = _interpolate(_LOW_COLOR, _CENTER_COLOR, value / 2.5)
    elif value < cap:
        color = _interpolate(_CENTER_COLOR, _HIGH_COLOR, (value - 2.5) / (cap - 2.5))
    else:
        color = _HIGH_COLOR
    return f"background:rgb{color};color:#1c1c1c"


def _bps_style(value: float, cap: float = 70.0) -> str:
    """Gera estilo de fundo para células BPS (sequencial vermelho/azul)."""
    if value is None:
        return ""
    if value >= 0:
        color = _interpolate(_BPS_START, _HIGH_COLOR, min(value, cap) / cap)
    else:
        color = _interpolate(_BPS_START, _LOW_COLOR, min(abs(value), cap) / cap)
    return f"background:rgb{color};color:#1c1c1c"


def _resumo_row_class(metric: str) -> str:
    """Classe da linha na tabela Resumo, conforme protótipo."""
    if metric in ("Índice Geral", "Média dos núcleos"):
        return "lvl0 header" if metric == "Índice Geral" else "lvl0"
    return "lvl1 top-sep"


def _resumo_padding(metric: str) -> int:
    """Indentação da célula de métrica na tabela Resumo."""
    if metric in ("Índice Geral", "Média dos núcleos"):
        return 16
    if metric in ("Administrados", "Livres") or metric.startswith("IPCA-"):
        return 34
    if metric in ("Serviços ex-passagem",):
        return 70
    return 52


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
        bcb_categories = bcb_client.fetch_ipca_categories(start_date=start_bcb, end_date=end_bcb)
        print(f"Categorias oficiais carregadas: {len(bcb_categories)} registros")
        bcb_subnuclei = bcb_client.fetch_ipca_subnuclei(start_date=start_bcb, end_date=end_bcb)
        print(f"Sub-núcleos oficiais carregados: {len(bcb_subnuclei)} registros")
    except Exception as e:
        print(f"Aviso: não foi possível carregar dados do BCB: {e}")
        bcb_cores = None
        bcb_categories = None
        bcb_subnuclei = None

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

    # Vetores oficiais de agregação (pesos das séries analíticas)
    bcb_vectors = BCBVectors()
    try:
        core_weights = bcb_vectors.weights(df_groups, latest_period)
        print(f"Pesos oficiais calculados: { {k: round(v, 1) for k, v in core_weights.items() if k in ('EX0', 'EX3')} }")
    except Exception as e:
        print(f"Aviso: não foi possível carregar vetores de agregação: {e}")
        core_weights = {}

    resumo = process_resumo(
        df_general, df_groups, bcb_cores=bcb_cores, core_weights=core_weights, period=latest_period
    )
    destaques = process_destaques(df_groups[df_groups["periodo_codigo"] == latest_period], top_n=10)
    surpresa = process_surpresa(df_general, df_groups, projections=focus_proj)
    grupos = process_grupos(
        df_general,
        df_groups,
        bcb_cores=bcb_cores,
        bcb_categories=bcb_categories,
        bcb_subnuclei=bcb_subnuclei,
        bcb_vectors=bcb_vectors,
        period=latest_period,
    )

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
        df_surp = pd.DataFrame(series)
        latest_surp = df_surp.sort_values("period").iloc[-1] if not df_surp.empty else {}
        surpresa_charts[cat] = {
            "main": chart_builder.build_surprise_chart(series),
            "detail": chart_builder.build_surprise_chart(df_surp.tail(36).to_dict("records")),
            "legend": {
                "m1": _fmt_bps(latest_surp.get("m1")),
                "m3": _fmt_bps(latest_surp.get("m3")),
                "m6": _fmt_bps(latest_surp.get("m6")),
                "m12": _fmt_bps(latest_surp.get("m12")),
            },
        }

    # Constrói gráficos de Grupos
    grupos_data = {}
    for cat, data in grupos.items():
        series = data["series"]
        subnuclei = data["subnuclei"]
        latest = data["latest"]
        target = latest.get("target", 3.0)
        target_text = _target_legend(target, latest.get("yoy"), show_deviation=(cat != "BCB"))

        main_chart = chart_builder.build_group_chart(series, title=None)

        # Legendas dos núcleos usados
        used_nuclei = [n["name"] for n in subnuclei[1:]] if len(subnuclei) > 1 else [n["name"] for n in subnuclei]
        nucleos_text = _nucleos_used_text(cat, used_nuclei)
        main_legend = {
            "nucleos_text": nucleos_text,
            "saar1": latest.get("saar1"),
            "saar3": latest.get("saar3"),
            "saar6": latest.get("saar6"),
            "yoy": latest.get("yoy"),
            "target_text": target_text,
        }

        # Sazonalidade
        seasonal = chart_builder.calc_seasonality(series)
        seasonal_chart = chart_builder.build_seasonal_chart(
            months=seasonal["months"],
            current=seasonal["current"],
            previous=seasonal["previous"],
            p10=seasonal["p10"],
            p90=seasonal["p90"],
            median=seasonal["median"],
            title=None,
        )
        mom_detail = chart_builder.build_mom_detail_chart(series)

        # Geral: headline vs média dos núcleos
        headline_series = data.get("headline_series", [])
        comparison_series = _merge_comparison_series(headline_series, series)
        label_index = "Índice geral" if cat == "BCB" else _category_display(cat)
        label_core = subnuclei[0]["name"] if subnuclei else f"Média dos núcleos de {cat}"
        geral_headline = {
            "chart": chart_builder.build_comparison_chart(
                comparison_series, label_index=label_index, label_core=label_core, title=None
            ),
            "detail": chart_builder.build_detail_comparison_chart(
                comparison_series, label_index=label_index, label_core=label_core
            ),
            "legend": {
                "index": {"label": label_index, "color": chart_builder.COLORS["pink"]},
                "core": {"label": label_core, "color": chart_builder.COLORS["primary"]},
                "target_text": target_text,
            },
        }

        # Geral: sub-núcleos individuais
        individual_series = [
            {"name": n["name"], "series": n["series"]}
            for n in subnuclei[1:]
        ]
        geral_subnuclei = {
            "chart": chart_builder.build_multiline_chart(individual_series, title=None, meta=target),
            "detail": chart_builder.build_detail_multiline_chart(individual_series, title="DETALHE · ÚLT. 36M", meta=target),
            "legend": [
                {"label": n["name"], "color": _subnucleo_color(i)}
                for i, n in enumerate(subnuclei[1:])
            ],
            "target_text": target_text,
        }

        grupos_data[cat] = {
            "subnuclei": subnuclei,
            "main_chart": main_chart,
            "main_legend": main_legend,
            "seasonal_chart": seasonal_chart,
            "seasonal_detail": mom_detail,
            "geral_headline": geral_headline,
            "geral_subnuclei": geral_subnuclei,
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

    # Caps dinâmicos baseados nos dados do Resumo para manter o contraste do heatmap
    bps_values = [r["mom_bps"] for r in data["resumo"] if r.get("mom_bps") is not None]
    bps_cap = max(70.0, max(abs(v) for v in bps_values) if bps_values else 70.0)

    # O protótipo usa uma escala fixa de -1% a +8% para YoY
    yoy_cap = 8.0

    def bps_style(value: float) -> str:
        return _bps_style(value, cap=bps_cap)

    def heatmap_style(value: float) -> str:
        return _heatmap_style(value, cap=yoy_cap)

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    env.filters["bps_style"] = bps_style
    env.filters["heatmap_style"] = heatmap_style
    env.globals["bps_style"] = bps_style
    env.globals["heatmap_style"] = heatmap_style
    env.globals["delta_style"] = delta_style
    env.globals["delta_str"] = delta_str
    env.globals["resumo_row_class"] = _resumo_row_class
    env.globals["resumo_padding"] = _resumo_padding

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
