import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from mlflow.models import Model

from antifraude_pix.avaliacao.no_tempo import avaliar
from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.modelos.rotulo import Dobra
from antifraude_pix.rastreio.registro import DESAFIANTE, registrar, treinar_o_final
from antifraude_pix.tempo_real.pacote import FICHA, Pacote, _so_o_nome_do_codigo, abrir, empacotar
from antifraude_pix.variaveis.calculo import calcular
from tests.apoio import simulacao_media

SETEMBRO = Dobra(date(2025, 9, 1), date(2025, 9, 29))


@pytest.fixture(scope="module")
def variaveis() -> pl.DataFrame:
    return calcular(VisaoDoBanco.da(simulacao_media()))


@pytest.fixture(scope="module")
def pontuacoes(variaveis: pl.DataFrame) -> pl.DataFrame:
    return avaliar(simulacao_media(), variaveis, (SETEMBRO,), modelos=("lightgbm",))


@pytest.fixture(scope="module")
def registro(tmp_path_factory: pytest.TempPathFactory, variaveis: pl.DataFrame, pontuacoes: pl.DataFrame) -> Path:
    """Um registro com o campeão de 1º de setembro e o desafiante de duas semanas depois."""
    pasta = tmp_path_factory.mktemp("dados")
    registrar(simulacao_media(), variaveis, pontuacoes, pasta, corte=date(2025, 9, 1))
    registrar(simulacao_media(), variaveis, pontuacoes, pasta, corte=date(2025, 9, 15), apelido=DESAFIANTE)

    return pasta


def _em_texto(tabela: pl.DataFrame) -> list[dict[str, object]]:
    return tabela.with_columns(pl.col("iniciacao", "chave").cast(pl.String)).to_dicts()


def test_o_pacote_decide_sem_o_registro(variaveis: pl.DataFrame, pontuacoes: pl.DataFrame, tmp_path: Path) -> None:
    # O pacote vai para outra máquina, onde o registro não existe: aqui, somem os artefatos para onde ele aponta.
    dados = tmp_path / "dados"
    registrar(simulacao_media(), variaveis, pontuacoes, dados, corte=date(2025, 9, 1))
    pacote = empacotar(dados, tmp_path / "campeao")
    shutil.rmtree(dados / "mlflow" / "artefatos")
    decisor, aberto = abrir(tmp_path / "campeao")
    modelo, _ = treinar_o_final(simulacao_media(), variaveis, date(2025, 9, 1))
    amostra = variaveis.tail(50)
    decididos = [decisor.decidir_um(linha) for linha in _em_texto(amostra)]

    assert pacote == aberto == Pacote("antifraude-pix", "1", "campeao", "2025-09-01")
    assert json.loads((tmp_path / "campeao" / FICHA).read_text(encoding="utf-8"))["versao"] == "1"
    assert np.allclose(
        [decidido["probabilidade"] for decidido in decididos], modelo.probabilidade(amostra), rtol=0, atol=1e-12
    )


def test_empacotar_de_novo_troca_o_pacote_inteiro(registro: Path, tmp_path: Path) -> None:
    destino = tmp_path / "campeao"
    empacotar(registro, destino)
    (destino / "sobra.txt").write_text("do pacote antigo", encoding="utf-8")

    assert empacotar(registro, destino, DESAFIANTE) == Pacote("antifraude-pix", "2", DESAFIANTE, "2025-09-15")
    assert abrir(destino)[1].versao == "2"
    assert not (destino / "sobra.txt").exists()
    # Nada da troca fica para trás ao lado do pacote.
    assert sorted(caminho.name for caminho in tmp_path.iterdir()) == ["campeao"]


def test_o_mlmodel_do_pacote_acha_o_codigo_pelo_nome_em_qualquer_maquina(registro: Path, tmp_path: Path) -> None:
    # No registro, o arquivo do modelo como código está pelo caminho absoluto de quem registrou; no Linux, o MLflow
    # emendaria um caminho do Windows inteiro na pasta do pacote.
    empacotar(registro, tmp_path / "campeao")
    codigo = Model.load(str(tmp_path / "campeao" / "MLmodel")).flavors["python_function"]["model_code_path"]

    assert codigo == "modelo_do_pix.py"
    assert (tmp_path / "campeao" / codigo).is_file()


@pytest.mark.parametrize(
    "registrado_em",
    [
        r"C:\Users\quem\antifraude-pix\src\antifraude_pix\rastreio\modelo_do_pix.py",
        "/home/quem/rastreio/modelo_do_pix.py",
    ],
    ids=["Windows", "Linux"],
)
def test_o_caminho_do_codigo_vira_o_nome_venha_de_que_maquina_vier(registrado_em: str, tmp_path: Path) -> None:
    Model(flavors={"python_function": {"model_code_path": registrado_em}}).save(str(tmp_path / "MLmodel"))
    _so_o_nome_do_codigo(tmp_path)

    assert Model.load(str(tmp_path / "MLmodel")).flavors["python_function"]["model_code_path"] == "modelo_do_pix.py"


def test_a_pasta_que_nao_e_pacote_diz_como_fazer_um(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="antifraude-pix empacotar"):
        abrir(tmp_path)
