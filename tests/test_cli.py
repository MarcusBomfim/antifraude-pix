from datetime import date
from pathlib import Path

import polars as pl
import pytest

from antifraude_pix.cli import principal
from antifraude_pix.variaveis.catalogo import NOMES


def test_simular_grava_os_dados_e_conta_o_que_saiu(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    saida = principal(
        ["simular", "--clientes", "80", "--dias", "20", "--golpes-por-mil", "120", "--saida", str(tmp_path)]
    )
    impresso = capsys.readouterr().out

    assert saida == 0
    assert impresso.startswith("80 clientes, 20 dias a partir de 01/07/2025, semente 2026.\n")
    assert "Pix;" in impresso
    assert f"Gravado em {tmp_path}" in impresso
    assert (tmp_path / "banco" / "transacoes.parquet").exists()


def test_validar_confere_o_que_simular_gravou(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    principal(["simular", "--clientes", "30", "--dias", "10", "--saida", str(tmp_path)])
    capsys.readouterr()

    assert principal(["validar", str(tmp_path)]) == 0
    assert capsys.readouterr().out.endswith("Os dados seguem o contrato.\n")


def test_a_configuracao_invalida_e_recusada_com_o_motivo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert principal(["simular", "--clientes", "0", "--saida", str(tmp_path)]) == 2
    assert capsys.readouterr().err == "antifraude-pix simular: O número de clientes tem de ser positivo.\n"
    assert not (tmp_path / "banco").exists()


def test_validar_a_pasta_que_nao_existe_falha_com_o_motivo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert principal(["validar", str(tmp_path / "nenhuma")]) == 1
    assert "antifraude-pix validar: não consegui ler" in capsys.readouterr().err


def test_o_comando_que_nao_existe_e_recusado_pelo_leitor() -> None:
    with pytest.raises(SystemExit) as saida:
        principal(["treinar"])

    assert saida.value.code == 2


def test_variaveis_le_o_banco_e_grava_uma_linha_por_pix(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    principal(["simular", "--clientes", "40", "--dias", "15", "--saida", str(tmp_path)])
    capsys.readouterr()

    assert principal(["variaveis", str(tmp_path)]) == 0
    impresso = capsys.readouterr().out
    variaveis = pl.read_parquet(tmp_path / "variaveis.parquet")
    pix = pl.read_parquet(tmp_path / "banco" / "transacoes.parquet")

    assert impresso.startswith(f"33 variáveis de {pix.height} Pix, gravadas em {tmp_path / 'variaveis.parquet'}")
    assert variaveis["id"].equals(pix["id"])
    assert variaveis.columns == ["id", *NOMES]


def test_variaveis_grava_onde_foi_pedido(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dados, destino = tmp_path / "dados", tmp_path / "outra" / "pasta" / "variaveis-de-teste.parquet"
    principal(["simular", "--clientes", "30", "--dias", "10", "--saida", str(dados)])

    assert principal(["variaveis", str(dados), "--saida", str(destino)]) == 0
    assert f"gravadas em {destino}" in capsys.readouterr().out
    assert pl.read_parquet(destino).columns == ["id", *NOMES]
    assert not (dados / "variaveis.parquet").exists()


def test_variaveis_da_pasta_que_nao_existe_falha_com_o_motivo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert principal(["variaveis", str(tmp_path / "nenhuma")]) == 1
    assert "antifraude-pix variaveis: não consegui ler" in capsys.readouterr().err


def test_avaliar_treina_mes_a_mes_e_mede_contra_a_verdade(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # De 1º de julho a 2 de novembro: aquecimento, agosto de treino, setembro de validação, outubro e novembro de teste.
    principal(["simular", "--clientes", "150", "--dias", "125", "--golpes-por-mil", "200", "--saida", str(tmp_path)])
    capsys.readouterr()

    assert principal(["avaliar", str(tmp_path)]) == 0
    impresso = capsys.readouterr().out
    pontuacoes = pl.read_parquet(tmp_path / "avaliacao.parquet")

    assert impresso.startswith("Teste de 01/10/2025 a 02/11/2025, em 2 meses: ")
    assert all(modelo in impresso for modelo in ("as regras de hoje", "a regressão logística", "o LightGBM"))
    assert pontuacoes["modelo"].unique(maintain_order=True).to_list() == ["regras", "logistica", "lightgbm"]
    assert pontuacoes["corte"].unique().sort().to_list() == [date(2025, 10, 1), date(2025, 11, 1)]


def test_avaliar_dados_curtos_demais_falha_com_o_motivo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    principal(["simular", "--clientes", "30", "--dias", "60", "--saida", str(tmp_path)])
    capsys.readouterr()

    assert principal(["avaliar", str(tmp_path)]) == 1
    assert "não chegam a um mês de teste" in capsys.readouterr().err
    assert not (tmp_path / "avaliacao.parquet").exists()


def test_avaliar_a_pasta_que_nao_existe_falha_com_o_motivo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert principal(["avaliar", str(tmp_path / "nenhuma")]) == 1
    assert "antifraude-pix avaliar: não consegui ler" in capsys.readouterr().err


def test_decidir_decide_explica_e_mede_o_custo_e_os_erros_por_idade(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    principal(["simular", "--clientes", "150", "--dias", "125", "--golpes-por-mil", "200", "--saida", str(tmp_path)])
    capsys.readouterr()

    assert principal(["decidir", str(tmp_path)]) == 0
    impresso = capsys.readouterr().out
    decisoes = pl.read_parquet(tmp_path / "decisoes.parquet")
    nao_aprovados = decisoes.filter(pl.col("acao") != "aprovar")

    assert impresso.startswith("Decisões de 01/10/2025 a 02/11/2025, em 2 meses: ")
    assert all(
        politica in impresso
        for politica in ("aprovar tudo", "bloquear de 50% para cima", "segurar o alerta de 0,5%", "decidir pelo custo")
    )
    assert all(faixa in impresso for faixa in ("18 a 34", "35 a 54", "55 a 64", "65 ou mais"))
    assert decisoes.columns == [
        "id",
        "corte",
        "modelo",
        "pontuacao",
        "probabilidade",
        "valor",
        "idade",
        "acao",
        "explicacao",
        "golpe",
    ]
    assert decisoes["corte"].unique().sort().to_list() == [date(2025, 10, 1), date(2025, 11, 1)]
    assert nao_aprovados.height > 0
    assert nao_aprovados["explicacao"].str.contains(" de chance de golpe: ").all()


def test_decidir_dados_curtos_demais_falha_com_o_motivo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    principal(["simular", "--clientes", "30", "--dias", "60", "--saida", str(tmp_path)])
    capsys.readouterr()

    assert principal(["decidir", str(tmp_path)]) == 1
    assert "antifraude-pix decidir: os dados de" in capsys.readouterr().err
    assert not (tmp_path / "decisoes.parquet").exists()


def test_decidir_a_pasta_que_nao_existe_falha_com_o_motivo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert principal(["decidir", str(tmp_path / "nenhuma")]) == 1
    assert "antifraude-pix decidir: não consegui ler" in capsys.readouterr().err
