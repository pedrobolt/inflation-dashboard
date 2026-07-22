"""
Geradores de gráficos Plotly no formato do protótipo.
"""

import json
import uuid
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder


# Paleta do protótipo
COLORS = {
    "primary": "#0072B2",
    "accent": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "pink": "#CC79A7",
    "grey": "#E5E7EB",
    "dark_grey": "#94a3b8",
    "text": "#1c1c1c",
    "grid": "rgba(0,0,0,0.04)",
    "grid_y": "rgba(0,0,0,0.06)",
    "zero_line": "rgba(0,0,0,0.18)",
}


def _new_id() -> str:
    return str(uuid.uuid4())


def _detail_annotation(text: str = "DETALHE · ÚLT. 36M") -> Dict:
    """Anotação de rótulo usada nos gráficos de detalhe, no formato do protótipo."""
    return {
        "font": {"color": "#64748b", "family": "'Inter', sans-serif", "size": 13},
        "showarrow": False,
        "text": f"<b><span style='letter-spacing:0.14em'>{text}</span></b>",
        "x": 0.0,
        "xanchor": "left",
        "xref": "paper",
        "xshift": -32,
        "y": 1.0,
        "yanchor": "bottom",
        "yref": "paper",
        "yshift": 28,
    }


def _apply_detail_style(chart: Dict, title: str = "DETALHE · ÚLT. 36M") -> Dict:
    """Aplica fundo cinza, margem superior e anotação de rótulo de um gráfico de detalhe."""
    fig = json.loads(chart["json"])
    fig["layout"]["plot_bgcolor"] = "#F4F6F8"
    fig["layout"]["margin"]["t"] = 58
    fig["layout"].pop("title", None)
    fig["layout"]["annotations"] = fig["layout"].get("annotations", []) + [_detail_annotation(title)]
    return {"div_id": chart["div_id"], "json": json.dumps(fig)}


def _to_json(fig: go.Figure) -> str:
    """Serializa figura Plotly para JSON com arrays como listas Python puras."""
    plotly_json = fig.to_plotly_json()
    clean = _convert_to_json_safe(plotly_json)
    return json.dumps(clean, ensure_ascii=False)


def _convert_to_json_safe(obj):
    """Converte recursivamente numpy/pandas/datetime/base64 para tipos JSON nativos."""
    import base64
    import datetime
    import numpy as np

    if isinstance(obj, dict) and "dtype" in obj and "bdata" in obj and isinstance(obj["bdata"], str):
        return _decode_bdata(obj)
    if isinstance(obj, np.ndarray):
        return [_convert_to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Series, pd.Index)):
        return [_convert_to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, pd.Period, datetime.datetime, datetime.date, np.datetime64)):
        return str(obj)
    if isinstance(obj, np.generic):
        if pd.isna(obj):
            return None
        return obj.item()
    if isinstance(obj, float):
        if pd.isna(obj):
            return None
        return obj
    if isinstance(obj, (list, tuple)):
        return [_convert_to_json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _convert_to_json_safe(v) for k, v in obj.items()}
    return obj


def _decode_bdata(obj: dict) -> list:
    """Decodifica array Plotly base64 para lista Python."""
    import base64
    import numpy as np

    try:
        raw = base64.b64decode(obj["bdata"])
        arr = np.frombuffer(raw, dtype=np.dtype(obj["dtype"]))
    except Exception:
        return obj

    if np.issubdtype(arr.dtype, np.floating):
        return [None if (isinstance(v, float) and np.isnan(v)) else float(v) for v in arr.tolist()]
    if np.issubdtype(arr.dtype, np.bool_):
        return [bool(v) for v in arr.tolist()]
    return [int(v) if np.issubdtype(arr.dtype, np.integer) else v for v in arr.tolist()]


def _parse_period(period_code: str) -> pd.Timestamp:
    return pd.to_datetime(period_code, format="%Y%m")


def _base_layout(height: int = 450, title: str = None, margin_top: int = 18) -> Dict:
    layout = {
        "font": {"family": "'Source Sans 3', sans-serif", "color": COLORS["text"]},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(250,250,250,0.5)",
        "height": height,
        "margin": {"t": margin_top, "b": 40, "l": 50, "r": 30},
        "showlegend": False,
        "xaxis": {
            "gridcolor": COLORS["grid"],
            "zeroline": False,
            "showline": True,
            "linecolor": COLORS["grid_y"],
            "tickfont": {"size": 11, "color": COLORS["dark_grey"], "family": "Inter, sans-serif"},
        },
        "yaxis": {
            "gridcolor": COLORS["grid_y"],
            "zeroline": False,
            "showline": True,
            "linecolor": COLORS["grid_y"],
            "tickfont": {"size": 11, "color": COLORS["dark_grey"], "family": "Inter, sans-serif"},
            "ticksuffix": "%",
            "tickformat": ".1f",
            "hoverformat": ".2f",
        },
        "hoverlabel": {
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "rgba(0,0,0,0.10)",
            "font": {"family": "'Source Sans 3', sans-serif", "size": 12},
        },
        "shapes": [{
            "type": "line",
            "xref": "paper",
            "x0": 0,
            "x1": 1,
            "yref": "y",
            "y0": 0,
            "y1": 0,
            "line": {"color": COLORS["zero_line"], "width": 1},
        }],
    }
    if title:
        layout["title"] = {
            "text": title,
            "font": {"size": 11, "color": "#64748b", "family": "Inter, sans-serif"},
            "x": 0,
            "xanchor": "left",
        }
    return layout


def build_general_history_chart(series: List[Dict]) -> Dict:
    """Gráfico histórico de MOM e YoY do IPCA geral."""
    df = pd.DataFrame(series)
    if df.empty:
        return {"div_id": _new_id(), "json": _to_json(go.Figure())}
    df = df.sort_values("period")
    df["date"] = df["period"].apply(lambda x: _parse_period(str(x)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["mom"],
        mode="lines",
        name="MoM",
        line={"color": COLORS["accent"], "width": 2},
        hovertemplate="<b>MoM</b> %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["yoy"],
        mode="lines",
        name="YoY",
        line={"color": COLORS["green"], "width": 2},
        hovertemplate="<b>YoY</b> %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(_base_layout(height=420, margin_top=18))
    return {"div_id": _new_id(), "json": _to_json(fig)}


def build_surprise_chart(series: List[Dict]) -> Dict:
    """Gráfico de surpresa (1M, 3M, 6M, 12M) em bps."""
    df = pd.DataFrame(series)
    if df.empty:
        return {"div_id": _new_id(), "json": _to_json(go.Figure())}
    df = df.sort_values("period")
    df["date"] = df["period"].apply(lambda x: _parse_period(str(x)))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["date"], y=df["m1"],
        name="1M",
        marker_color=COLORS["grey"],
        opacity=0.85,
        hovertemplate="<b>1M</b> %{y:.0f}<extra></extra>",
    ))
    for col, color, label in [("m3", COLORS["accent"], "3M"),
                              ("m6", COLORS["primary"], "6M"),
                              ("m12", COLORS["green"], "12M")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col],
                mode="lines",
                name=label,
                line={"color": color, "width": 2},
                hovertemplate=f"<b>{label}</b> %{{y:.0f}}<extra></extra>",
            ))
    fig.update_layout(_base_layout(height=450, margin_top=18))
    fig.update_layout(yaxis={"ticksuffix": "", "tickformat": ".0f", "hoverformat": ".0f"})
    return {"div_id": _new_id(), "json": _to_json(fig)}


def build_group_chart(series: List[Dict], title: str = None) -> Dict:
    """Gráfico de núcleos (1M SAAR, 3M SAAR, 6M SAAR, YoY)."""
    df = pd.DataFrame(series)
    if df.empty:
        return {"div_id": _new_id(), "json": _to_json(go.Figure())}
    df = df.sort_values("period")
    df["date"] = df["period"].apply(lambda x: _parse_period(str(x)))

    fig = go.Figure()
    # meta BCB tracejada
    if "meta" in df.columns and df["meta"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["meta"],
            mode="lines",
            name="Meta BCB",
            line={"color": "rgba(204,121,167,0.30)", "dash": "dot", "width": 1.4},
            hoverinfo="skip",
        ))
    fig.add_trace(go.Bar(
        x=df["date"], y=df.get("saar1", df.get("mom", [])),
        name="1M SAAR",
        marker_color=COLORS["grey"],
        opacity=0.85,
        hovertemplate="<b>1M SAAR</b> %{y:.2f}%<extra></extra>",
    ))
    for col, color, label in [("saar3", COLORS["accent"], "3M SAAR"),
                              ("saar6", COLORS["primary"], "6M SAAR"),
                              ("yoy", COLORS["green"], "YoY")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col],
                mode="lines",
                name=label,
                line={"color": color, "width": 2},
                hovertemplate=f"<b>{label}</b> %{{y:.2f}}%<extra></extra>",
            ))
    fig.update_layout(_base_layout(height=450, title=title, margin_top=18))
    return {"div_id": _new_id(), "json": _to_json(fig)}


def build_detail_chart(series: List[Dict], title: str = "DETALHE · ÚLT. 36M") -> Dict:
    """Gráfico de detalhe dos últimos 36 meses com fundo #F4F6F8."""
    chart = build_group_chart(series, title=None)
    return _apply_detail_style(chart, title)


def build_detail_surprise_chart(series: List[Dict], title: str = "DETALHE · ÚLT. 36M") -> Dict:
    """Gráfico de surpresa de detalhe (últimos 36 meses) com fundo #F4F6F8."""
    chart = build_surprise_chart(series)
    return _apply_detail_style(chart, title)


def build_seasonal_chart(months: List[int], current: List[float], previous: List[float],
                         p10: List[float], p90: List[float], median: List[float],
                         title: str = None) -> Dict:
    """Gráfico de sazonalidade: faixa P10-P90, mediana, ano anterior e ano corrente."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=p90,
        mode="lines",
        line={"color": "rgba(0,0,0,0)", "width": 0},
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=months, y=p10,
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(0,0,0,0.06)",
        line={"color": "rgba(0,0,0,0)", "width": 0},
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=months, y=median,
        mode="lines",
        line={"color": "rgba(0,0,0,0.45)", "dash": "dot", "width": 1.4},
        name="Mediana",
        hovertemplate="<b>Mediana</b> %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=months, y=previous,
        mode="lines+markers",
        line={"color": "#FAB6C1", "width": 1.5},
        marker={"size": 5, "color": "#FAB6C1"},
        name="Anterior",
        hovertemplate="<b>Anterior</b> %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=months, y=current,
        mode="lines+markers",
        line={"color": "#EA243E", "width": 2},
        marker={"size": 5, "color": "#EA243E"},
        name="Corrente",
        hovertemplate="<b>Corrente</b> %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(_base_layout(height=320, title=title, margin_top=52))
    return {"div_id": _new_id(), "json": _to_json(fig)}


def build_comparison_chart(series: List[Dict], label_index: str = "Índice geral",
                           label_core: str = "Média dos núcleos", title: str = None) -> Dict:
    """Gráfico de comparação headline vs média dos núcleos (YoY)."""
    df = pd.DataFrame(series)
    if df.empty:
        return {"div_id": _new_id(), "json": _to_json(go.Figure())}
    df = df.sort_values("period")
    df["date"] = df["period"].apply(lambda x: _parse_period(str(x)))

    fig = go.Figure()
    if "index" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["index"],
            mode="lines",
            name=label_index,
            line={"color": COLORS["pink"], "width": 1.5},
            hovertemplate=f"<b>{label_index}</b> %{{y:.2f}}%<extra></extra>",
        ))
    if "core" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["core"],
            mode="lines",
            name=label_core,
            line={"color": COLORS["primary"], "width": 2},
            hovertemplate=f"<b>{label_core}</b> %{{y:.2f}}%<extra></extra>",
        ))
    if "meta" in df.columns and df["meta"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["meta"],
            mode="lines",
            name="Meta BCB",
            line={"color": "rgba(204,121,167,0.30)", "dash": "dot", "width": 1.4},
            hoverinfo="skip",
        ))
    fig.update_layout(_base_layout(height=320, title=title, margin_top=52))
    return {"div_id": _new_id(), "json": _to_json(fig)}


def build_detail_comparison_chart(series: List[Dict], label_index: str = "Índice geral",
                                  label_core: str = "Média dos núcleos",
                                  title: str = "DETALHE · ÚLT. 36M") -> Dict:
    """Versão detalhada (últimos 36 meses) da comparação headline vs núcleos."""
    chart = build_comparison_chart(series, label_index=label_index, label_core=label_core, title=None)
    return _apply_detail_style(chart, title)


def build_multiline_chart(series_list: List[Dict], title: str = None, meta: float = None) -> Dict:
    """Gráfico com múltiplas séries de linha (sub-núcleos individuais, YoY)."""
    colors = [COLORS["primary"], COLORS["green"], COLORS["orange"], COLORS["pink"], COLORS["dark_grey"]]
    fig = go.Figure()
    for i, item in enumerate(series_list):
        df = pd.DataFrame(item["series"]).sort_values("period")
        if df.empty:
            continue
        df["date"] = df["period"].apply(lambda x: _parse_period(str(x)))
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["yoy"],
            mode="lines",
            name=item["name"],
            line={"color": color, "width": 1.8},
            hovertemplate=f"<b>{item['name']}</b> %{{y:.2f}}%<extra></extra>",
        ))
    if meta is not None:
        fig.add_hline(y=meta, line={"color": "rgba(204,121,167,0.30)", "dash": "dot", "width": 1.4})
    fig.update_layout(_base_layout(height=320, title=title, margin_top=52))
    return {"div_id": _new_id(), "json": _to_json(fig)}


def build_detail_multiline_chart(series_list: List[Dict], title: str = "DETALHE · ÚLT. 36M",
                                 meta: float = None) -> Dict:
    """Versão detalhada (últimos 36 meses) do gráfico de múltiplas linhas."""
    chart = build_multiline_chart(series_list, title=None, meta=meta)
    return _apply_detail_style(chart, title)


def build_mom_detail_chart(series: List[Dict], title: str = "DETALHE · ÚLT. 36M") -> Dict:
    """Gráfico de barras da variação mensal bruta (MoM) dos últimos 36 meses."""
    df = pd.DataFrame(series)
    if df.empty:
        return {"div_id": _new_id(), "json": _to_json(go.Figure())}
    df = df.sort_values("period").tail(36).reset_index(drop=True)
    df["date"] = df["period"].apply(lambda x: _parse_period(str(x)))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["date"], y=df["mom"],
        marker_color=COLORS["accent"],
        opacity=0.85,
        hovertemplate="<b>MoM</b> %{y:.2f}%<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(_base_layout(height=320, title=None, margin_top=58))
    fig.update_layout(plot_bgcolor="#F4F6F8")
    fig.add_annotation(**_detail_annotation(title))
    return {"div_id": _new_id(), "json": _to_json(fig)}


def calc_seasonality(series: List[Dict]) -> Dict:
    """
    Calcula padrão sazonal mensal a partir de uma série de MoM.

    Retorna mediana, P10 e P90 de anos de referência (excluindo o ano corrente)
    e os valores do ano corrente e do ano anterior.
    """
    df = pd.DataFrame(series)
    if df.empty:
        return {"months": [], "median": [], "p10": [], "p90": [],
                "current": [], "previous": [], "current_year": None, "previous_year": None}
    df = df.dropna(subset=["period", "mom"]).copy()
    df["period"] = df["period"].astype(str)
    df["date"] = df["period"].apply(lambda x: _parse_period(x))
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    latest_year = int(df["year"].max())
    previous_year = latest_year - 1
    ref_years = [y for y in df["year"].unique() if y != latest_year]

    months = list(range(1, 13))
    median, p10, p90, current, previous = [], [], [], [], []
    for m in months:
        ref_vals = df[(df["month"] == m) & (df["year"].isin(ref_years))]["mom"].tolist()
        if len(ref_vals) > 0:
            median.append(float(np.median(ref_vals)))
            p10.append(float(np.percentile(ref_vals, 10)))
            p90.append(float(np.percentile(ref_vals, 90)))
        else:
            median.append(None)
            p10.append(None)
            p90.append(None)

        cur_row = df[(df["month"] == m) & (df["year"] == latest_year)]
        current.append(float(cur_row["mom"].iloc[-1]) if not cur_row.empty else None)

        prev_row = df[(df["month"] == m) & (df["year"] == previous_year)]
        previous.append(float(prev_row["mom"].iloc[-1]) if not prev_row.empty else None)

    return {
        "months": months,
        "median": median,
        "p10": p10,
        "p90": p90,
        "current": current,
        "previous": previous,
        "current_year": latest_year,
        "previous_year": previous_year,
    }


def calc_moving_averages(series_df: pd.DataFrame, col: str = "mom") -> pd.DataFrame:
    """Calcula médias móveis 3, 6 e 12 meses."""
    df = series_df.sort_values("period").copy()
    df["m3"] = df[col].rolling(3, min_periods=1).mean()
    df["m6"] = df[col].rolling(6, min_periods=1).mean()
    df["m12"] = df[col].rolling(12, min_periods=1).mean()
    return df


def calc_saar(series_df: pd.DataFrame, col: str = "mom") -> pd.DataFrame:
    """Calcula taxa anualizada sazonalmente (SAAR) a partir de variação mensal."""
    df = series_df.sort_values("period").copy()
    df["saar1"] = ((1 + df[col] / 100) ** 12 - 1) * 100
    df["saar3"] = ((1 + df["m3"] / 100) ** 12 - 1) * 100
    df["saar6"] = ((1 + df["m6"] / 100) ** 12 - 1) * 100
    return df
