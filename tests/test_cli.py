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
