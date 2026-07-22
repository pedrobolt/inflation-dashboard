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

MONTHS_ABBR = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]

# Marcações de ano no eixo x (Dez/16 de cada ano, rótulo com 2 dígitos), igual ao protótipo
_YEAR_TICKVALS = [f"{y}-12-16T00:00:00" for y in range(1979, 2030)]
_YEAR_TICKTEXT = [f"{y % 100:02d}" for y in range(1980, 2031)]


def _new_id() -> str:
    return str(uuid.uuid4())


def _label_annotation(text: str) -> Dict:
    """Anotação de rótulo no canto superior esquerdo, no formato do protótipo (usada em
    gráficos de detalhe 'DETALHE · ÚLT. 36M' e nos rótulos de métrica/série dos pares lado a lado)."""
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
    fig["layout"]["paper_bgcolor"] = "#F4F6F8"
    fig["layout"]["margin"]["t"] = 58
    fig["layout"]["annotations"] = fig["layout"].get("annotations", []) + [_label_annotation(title)]
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


def _base_layout(height: int = 450, margin_top: int = 18, margin_left: int = 50,
                 margin_bottom: int = 40, margin_right: int = 30,
                 y_rangemode: Optional[str] = None) -> Dict:
    layout = {
        "font": {"family": "'Source Sans 3', sans-serif", "color": COLORS["text"], "size": 12},
        "paper_bgcolor": "rgba(250,250,250,0.5)",
        "plot_bgcolor": "rgba(250,250,250,0.5)",
        "height": height,
        "margin": {"t": margin_top, "b": margin_bottom, "l": margin_left, "r": margin_right},
        "showlegend": False,
        "hovermode": "x unified",
        "xaxis": {
            "gridcolor": COLORS["grid"],
            "zeroline": False,
            "showgrid": True,
            "tickfont": {"size": 13, "color": COLORS["dark_grey"], "family": "'Inter', sans-serif"},
            "showspikes": True,
            "spikethickness": 1,
            "spikedash": "dot",
            "spikecolor": "rgba(0,0,0,0.25)",
            "spikemode": "across",
            "hoverformat": "%b/%Y",
            "tickvals": _YEAR_TICKVALS,
            "ticktext": _YEAR_TICKTEXT,
        },
        "yaxis": {
            "gridcolor": COLORS["grid_y"],
            "zeroline": False,
            "tickfont": {"size": 13, "color": COLORS["dark_grey"], "family": "'Inter', sans-serif"},
            "ticksuffix": "%",
            "tickformat": ".1f",
            "hoverformat": ".2f",
        },
        "hoverlabel": {
            "bgcolor": "rgba(255,255,255,0.72)",
            "bordercolor": "rgba(0,0,0,0.10)",
            "font": {"family": "'Source Sans 3', sans-serif", "size": 12},
            "align": "left",
        },
        "shapes": [{
            "type": "line",
            "xref": "x domain",
            "x0": 0,
            "x1": 1,
            "yref": "y",
            "y0": 0,
            "y1": 0,
            "line": {"color": COLORS["zero_line"], "width": 1},
        }],
    }
    if y_rangemode:
        layout["yaxis"]["rangemode"] = y_rangemode
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
        line={"color": COLORS["accent"], "width": 1.8},
        hovertemplate="<b>MoM</b> %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["yoy"],
        mode="lines",
        name="YoY",
        line={"color": COLORS["green"], "width": 2.5},
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
    for col, color, width, label in [("m3", COLORS["accent"], 1.8, "3M"),
                                     ("m6", COLORS["primary"], 2.2, "6M"),
                                     ("m12", COLORS["green"], 2.5, "12M")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col],
                mode="lines",
                name=label,
                line={"color": color, "width": width},
                hovertemplate=f"<b>{label}</b> %{{y:.0f}}<extra></extra>",
            ))
    fig.update_layout(_base_layout(height=450, margin_top=18, margin_left=40, y_rangemode="tozero"))
    fig.update_layout(yaxis={"ticksuffix": "", "tickformat": ".0f", "hoverformat": ".0f"})
    return {"div_id": _new_id(), "json": _to_json(fig)}


def build_detail_surprise_chart(series: List[Dict], title: str = "DETALHE · ÚLT. 36M") -> Dict:
    """Gráfico de surpresa de detalhe (últimos 36 meses) com fundo #F4F6F8."""
    chart = build_surprise_chart(series)
    return _apply_detail_style(chart, title)


def build_group_chart(series: List[Dict]) -> Dict:
    """Gráfico de núcleos (1M SAAR, 3M SAAR, 6M SAAR, YoY)."""
    df = pd.DataFrame(series)
    if df.empty:
        return {"div_id": _new_id(), "json": _to_json(go.Figure())}
    df = df.sort_values("period")
    df["date"] = df["period"].apply(lambda x: _parse_period(str(x)))

    fig = go.Figure()
    # meta BCB tracejada (linha em degraus)
    if "meta" in df.columns and df["meta"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["meta"],
            mode="lines",
            line_shape="hv",
            name="Meta BCB",
            line={"color": "rgba(204,121,167,0.85)", "dash": "dash", "width": 1.4},
            hovertemplate="<b>Meta BCB</b> %{y:.2f}%<extra></extra>",
        ))
    fig.add_trace(go.Bar(
        x=df["date"], y=df.get("saar1", df.get("mom", [])),
        name="1M SAAR",
        marker_color=COLORS["grey"],
        opacity=0.85,
        hovertemplate="<b>1M SAAR</b> %{y:.2f}%<extra></extra>",
    ))
    for col, color, width, label in [("saar3", COLORS["accent"], 1.8, "3M SAAR"),
                                     ("saar6", COLORS["primary"], 2.2, "6M SAAR"),
                                     ("yoy", COLORS["green"], 2.5, "YoY")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col],
                mode="lines",
                name=label,
                line={"color": color, "width": width},
                hovertemplate=f"<b>{label}</b> %{{y:.2f}}%<extra></extra>",
            ))
    fig.update_layout(_base_layout(height=450, margin_top=18, margin_left=55, y_rangemode="tozero"))
    return {"div_id": _new_id(), "json": _to_json(fig)}


def build_detail_chart(series: List[Dict], title: str = "DETALHE · ÚLT. 36M") -> Dict:
    """Gráfico de detalhe dos últimos 36 meses com fundo #F4F6F8."""
    chart = build_group_chart(series)
    return _apply_detail_style(chart, title)


def build_seasonal_chart(months: List[int], current: List[float], previous: List[float],
                         p10: List[float], p90: List[float], median: List[float],
                         current_year: int = None, previous_year: int = None,
                         label: str = None,
                         color_previous: str = None, color_current: str = None) -> Dict:
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
        line={"color": "rgba(0,0,0,0.45)", "dash": "dash", "width": 1.4},
        name="Mediana hist.",
        hovertemplate="<b>Mediana hist.</b> %{y:.2f}%<extra></extra>",
    ))
    color_previous = color_previous or COLORS["pink"]
    color_current = color_current or COLORS["primary"]
    fig.add_trace(go.Scatter(
        x=months, y=previous,
        mode="lines+markers",
        line={"color": color_previous, "width": 1.5},
        marker={"size": 7, "color": color_previous, "line": {"color": "white", "width": 1.5}},
        name=f"Ano anterior ({previous_year})" if previous_year else "Ano anterior",
        hovertemplate="<b>Anterior</b> %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=months, y=current,
        mode="lines+markers",
        line={"color": color_current, "width": 2.0},
        marker={"size": 7, "color": color_current, "line": {"color": "white", "width": 1.5}},
        name=f"Ano corrente ({current_year})" if current_year else "Ano corrente",
        hovertemplate="<b>Corrente</b> %{y:.2f}%<extra></extra>",
    ))
    layout = _base_layout(height=450, margin_top=52, margin_left=42, margin_bottom=22, margin_right=10)
    layout["xaxis"].update({
        "tickmode": "array",
        "tickvals": list(range(1, 13)),
        "ticktext": MONTHS_ABBR,
        "range": [0.5, 12.5],
    })
    if label:
        layout["annotations"] = [_label_annotation(label.upper())]
    fig.update_layout(layout)
    return {"div_id": _new_id(), "json": _to_json(fig)}


def build_comparison_chart(series: List[Dict], label_index: str = "Índice geral",
                           label_core: str = "Média dos núcleos", metric: str = "yoy",
                           label: str = None) -> Dict:
    """Gráfico de comparação headline vs média dos núcleos (YoY ou 3M SAAR)."""
    df = pd.DataFrame(series)
    if df.empty:
        return {"div_id": _new_id(), "json": _to_json(go.Figure())}
    df = df.sort_values("period")
    df["date"] = df["period"].apply(lambda x: _parse_period(str(x)))

    index_col = f"index_{metric}"
    core_col = f"core_{metric}"

    fig = go.Figure()
    if "meta" in df.columns and df["meta"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["meta"],
            mode="lines",
            line_shape="hv",
            name="Meta BCB",
            line={"color": "rgba(0,0,0,0.45)", "dash": "dash", "width": 1.4},
            hovertemplate="<b>Meta BCB</b> %{y:.2f}%<extra></extra>",
        ))
    if index_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[index_col],
            mode="lines",
            name=label_index,
            line={"color": COLORS["accent"], "width": 1.8},
            hovertemplate=f"<b>{label_index}</b> %{{y:.2f}}%<extra></extra>",
        ))
    if core_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[core_col],
            mode="lines",
            name=label_core,
            line={"color": COLORS["pink"], "width": 2.5},
            hovertemplate=f"<b>{label_core}</b> %{{y:.2f}}%<extra></extra>",
        ))
    layout = _base_layout(height=450, margin_top=52, margin_left=35, margin_bottom=40, margin_right=20,
                          y_rangemode="tozero")
    if label:
        layout["annotations"] = [_label_annotation(label.upper())]
    fig.update_layout(layout)
    return {"div_id": _new_id(), "json": _to_json(fig)}


def build_multiline_chart(series_list: List[Dict], metric: str = "yoy", meta: float = None,
                          label: str = None) -> Dict:
    """Gráfico com múltiplas séries de linha (sub-núcleos individuais, YoY ou 3M SAAR)."""
    colors = [COLORS["orange"], COLORS["accent"], COLORS["primary"], COLORS["green"], COLORS["pink"]]
    fig = go.Figure()
    if meta is not None:
        all_dates = sorted({p["period"] for item in series_list for p in item["series"]})
        if all_dates:
            meta_dates = [_parse_period(p) for p in all_dates]
            fig.add_trace(go.Scatter(
                x=meta_dates, y=[meta] * len(meta_dates),
                mode="lines",
                line_shape="hv",
                name="Meta BCB",
                line={"color": "rgba(0,0,0,0.45)", "dash": "dash", "width": 1.4},
                hovertemplate="<b>Meta BCB</b> %{y:.2f}%<extra></extra>",
            ))
    for i, item in enumerate(series_list):
        df = pd.DataFrame(item["series"]).sort_values("period")
        if df.empty or metric not in df.columns:
            continue
        df["date"] = df["period"].apply(lambda x: _parse_period(str(x)))
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[metric],
            mode="lines",
            name=item["name"],
            line={"color": color, "width": 1.8},
            hovertemplate=f"<b>{item['name']}</b> %{{y:.2f}}%<extra></extra>",
        ))
    layout = _base_layout(height=450, margin_top=52, margin_left=35, margin_bottom=40, margin_right=20,
                          y_rangemode="tozero")
    if label:
        layout["annotations"] = [_label_annotation(label.upper())]
    fig.update_layout(layout)
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
