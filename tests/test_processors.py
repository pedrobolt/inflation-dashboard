import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from processors.brazil.resumo import process_resumo
from processors.brazil.destaques import process_destaques
from processors.brazil.grupos import process_grupos


def test_process_resumo():
    df_general = pd.DataFrame({
        "periodo": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "periodo_codigo": ["202401", "202402", "202403"],
        "mom": [0.5, 0.6, 0.7],
        "yoy": [4.0, 4.1, 4.2],
    })

    df_groups = pd.DataFrame({
        "periodo": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"] * 2),
        "periodo_codigo": ["202401", "202402", "202403"] * 2,
        "item": ["1.Alimentação e bebidas", "1.Alimentação e bebidas", "1.Alimentação e bebidas",
                 "2.Habitação", "2.Habitação", "2.Habitação"],
        "item_codigo": ["7170"] * 3 + ["7445"] * 3,
        "mom": [1.0, 1.1, 1.2, -0.5, -0.4, -0.3],
        "yoy": [5.0, 5.1, 5.2, 3.0, 3.1, 3.2],
        "peso": [20.0, 20.0, 20.0, 15.0, 15.0, 15.0],
    })

    bcb_cores = pd.DataFrame({
        "data": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "EX0": [0.4, 0.5, 0.6],
        "EX3": [0.3, 0.4, 0.5],
        "MS": [0.35, 0.45, 0.55],
        "DP": [0.45, 0.55, 0.65],
        "P55": [0.5, 0.6, 0.7],
        "media": [0.4, 0.5, 0.6],
    })

    result = process_resumo(df_general, df_groups, bcb_cores=bcb_cores, period="202403")
    assert len(result) > 0

    geral = next((r for r in result if r["metric"] == "Índice Geral"), None)
    assert geral is not None
    assert geral["mom"] == 0.7
    assert geral["mom_t_1"] == 0.6

    # O novo Resumo segue o protótipo: só Índice Geral, categorias especiais e núcleos
    assert next((r for r in result if r["metric"] == "Alimentação e bebidas"), None) is None
    assert next((r for r in result if r["metric"] == "IPCA-EX0"), None) is not None


def test_process_destaques():
    df = pd.DataFrame({
        "periodo": pd.to_datetime(["2024-03-01"] * 4),
        "periodo_codigo": ["202403"] * 4,
        "item": ["1101.Item A", "1102.Item B", "1103.Item C", "1104.Item D"],
        "item_codigo": ["11010", "11020", "11030", "11040"],
        "mom": [10.0, 5.0, -8.0, -3.0],
        "peso": [1.0, 2.0, 1.0, 2.0],
    })

    result = process_destaques(df, top_n=2)
    assert len(result["positive"]) == 2
    assert len(result["negative"]) == 2
    assert result["positive"][0]["bps"] == 10.0  # 1.0 * 10.0
    assert result["negative"][0]["bps"] == -8.0  # 1.0 * -8.0


def test_process_grupos():
    periods = pd.date_range("2023-01-01", periods=24, freq="MS")
    period_codes = [d.strftime("%Y%m") for d in periods]
    df_general = pd.DataFrame({
        "periodo": periods,
        "periodo_codigo": period_codes,
        "mom": [0.5 + 0.01 * i for i in range(24)],
        "yoy": [4.0 + 0.05 * i for i in range(24)],
    })

    # Itens folha para cada uma das três categorias (códigos compatíveis com _classify_item)
    items = [
        ("3301001.Serviço A", "33010", 1.0),
        ("3101001.Industrial A", "31010", 1.0),
        ("1102001.Alimento A", "11020", 1.0),
    ]
    rows = []
    for p, c in zip(periods, period_codes):
        for name, code, weight in items:
            rows.append({
                "periodo": p,
                "periodo_codigo": c,
                "item": name,
                "item_codigo": code,
                "mom": 0.4,
                "yoy": 3.5,
                "peso": weight,
            })
    df_groups = pd.DataFrame(rows)

    bcb_cores = pd.DataFrame({
        "data": periods,
        "EX0": [0.4] * 24,
        "EX3": [0.45] * 24,
        "MS": [0.42] * 24,
        "DP": [0.43] * 24,
        "P55": [0.44] * 24,
        "media": [0.43] * 24,
    })
    bcb_categories = pd.DataFrame({
        "data": periods,
        "Serviços": [0.5] * 24,
        "Industriais": [0.4] * 24,
        "Alimentação": [0.6] * 24,
    })

    result = process_grupos(
        df_general, df_groups,
        bcb_cores=bcb_cores,
        bcb_categories=bcb_categories,
        bcb_subnuclei=None,
        bcb_vectors=None,
        period="202412",
    )

    assert "BCB" in result
    assert "Serviços" in result
    assert "Industriais" in result
    assert "Alimentação" in result
    for cat, data in result.items():
        assert len(data["series"]) > 0
        assert len(data["subnuclei"]) > 0
        assert "latest" in data


if __name__ == "__main__":
    test_process_resumo()
    test_process_destaques()
    test_process_grupos()
    print("Todos os testes passaram.")
