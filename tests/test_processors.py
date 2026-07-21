import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from processors.brazil.resumo import process_resumo
from processors.brazil.destaques import process_destaques


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

    result = process_resumo(df_general, df_groups, period="202403")
    assert len(result) > 0

    geral = next((r for r in result if r["metric"] == "Índice Geral"), None)
    assert geral is not None
    assert geral["mom"] == 0.7
    assert geral["mom_t_1"] == 0.6

    alimentacao = next((r for r in result if r["metric"] == "Alimentação e bebidas"), None)
    assert alimentacao is not None
    assert alimentacao["mom"] == 1.2
    assert alimentacao["weight"] == 20.0


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


if __name__ == "__main__":
    test_process_resumo()
    test_process_destaques()
    print("Todos os testes passaram.")
