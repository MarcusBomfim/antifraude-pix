from datetime import date
from pathlib import Path
from types import SimpleNamespace

import mlflow
import numpy as np
import polars as pl
import pytest
from mlflow.tracking import MlflowClient

from antifraude_pix.avaliacao.metricas import resumo
from antifraude_pix.avaliacao.no_tempo import avaliar
from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.decisao.custo import Custos, decidir
from antifraude_pix.modelos.preparo import ENTRADA
from antifraude_pix.modelos.rotulo import Dobra, janela_de_teste
from antifraude_pix.rastreio.experimentos import endereco
from antifraude_pix.rastreio.modelo_do_pix import SAIDA, ModeloDoPix, da_tabela
from antifraude_pix.rastreio.registro import (
    CAMPEAO,
    NOME,
    Registrado,
    gravar_os_artefatos,
    para_a_tabela,
    registrar,
    treinar_o_final,
)
from antifraude_pix.variaveis.calculo import calcular
from tests.apoio import simulacao_media

SETEMBRO = Dobra(date(2025, 9, 1), date(2025, 9, 29))


@pytest.fixture(scope="module")
def variaveis() -> pl.DataFrame:
    return calcular(VisaoDoBanco.da(simulacao_media()))


@pytest.fixture(scope="module")
def setembro(variaveis: pl.DataFrame) -> pl.DataFrame:
    return janela_de_teste(VisaoDoBanco.da(simulacao_media()), variaveis, SETEMBRO)


@pytest.fixture(scope="module")
def pontuacoes(variaveis: pl.DataFrame) -> pl.DataFrame:
    return avaliar(simulacao_media(), variaveis, (SETEMBRO,), modelos=("lightgbm",))


@pytest.fixture(scope="module")
def pasta(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("dados")


@pytest.fixture(scope="module")
def registrado(pasta: Path, variaveis: pl.DataFrame, pontuacoes: pl.DataFrame) -> Registrado:
    return registrar(simulacao_media(), variaveis, pontuacoes, pasta)


def test_o_modelo_e_registrado_com_o_apelido_de_campeao_e_a_ficha(pasta: Path, registrado: Registrado) -> None:
    cliente = MlflowClient(endereco(pasta))
    versao = cliente.get_model_version_by_alias(NOME, CAMPEAO)

    assert str(versao.version) == registrado.versao == "1"
    assert versao.run_id == registrado.corrida
    assert versao.description == registrado.ficha
    # No MLflow 3, o modelo fica à parte, ligado à corrida que o gravou; a corrida guarda a ficha.
    assert "ficha.md" in {artefato.path for artefato in cliente.list_artifacts(registrado.corrida)}


def test_o_campeao_carregado_decide_e_explica_como_o_modelo_treinado(
    pasta: Path, registrado: Registrado, variaveis: pl.DataFrame, setembro: pl.DataFrame
) -> None:
    del registrado
    mlflow.set_tracking_uri(endereco(pasta))
    carregado = mlflow.pyfunc.load_model(f"models:/{NOME}@{CAMPEAO}")
    saida = pl.from_pandas(carregado.predict(para_a_tabela(setembro)))
    modelo, _ = treinar_o_final(simulacao_media(), variaveis)
    decididos = decidir(setembro.with_columns(pl.Series("probabilidade", modelo.probabilidade(setembro))))

    assert saida.columns == list(SAIDA)
    assert saida.height == setembro.height
    assert np.allclose(saida["probabilidade"].to_numpy(), modelo.probabilidade(setembro), rtol=0, atol=1e-12)
    assert saida["acao"].to_list() == decididos["acao"].cast(pl.String).to_list()
    assert saida.filter(pl.col("acao") != "aprovar")["explicacao"].str.contains(" de chance de golpe: ").all()


def test_a_assinatura_e_a_entrada_sem_a_idade_e_a_saida_explicada(pasta: Path, registrado: Registrado) -> None:
    mlflow.set_tracking_uri(endereco(pasta))
    del registrado
    assinatura = mlflow.models.get_model_info(f"models:/{NOME}@{CAMPEAO}").signature
    vazias = {coluna.name for coluna in assinatura.inputs.inputs if not coluna.required}

    assert assinatura.inputs.input_names() == [variavel.nome for variavel in ENTRADA]
    assert assinatura.outputs.input_names() == list(SAIDA)
    # A variável que pode faltar, como o aparelho de sempre, que não tem minutos desde que entrou, não é obrigatória.
    assert {"chave", "minutos_desde_o_aparelho_novo", "mediana_de_90_dias"} <= vazias
    assert "valor" not in vazias


def test_o_treino_final_vai_ate_o_ultimo_dia_dos_dados(variaveis: pl.DataFrame) -> None:
    _, treino = treinar_o_final(simulacao_media(), variaveis)

    assert treino.corte == date(2025, 9, 29)
    assert 0 < treino.golpes_sabidos < treino.pix
    assert 0 < treino.na_calibracao < treino.pix


def test_a_ficha_traz_as_medidas_da_avaliacao(registrado: Registrado, pontuacoes: pl.DataFrame) -> None:
    precisao = resumo(pontuacoes)["precisao_media"][0]

    for secao in ("## O que ele faz", "## Para que ele não serve", "## A entrada", "## A decisão", "## Onde ele erra"):
        assert secao in registrado.ficha

    assert f"| {precisao:.3f}".replace(".", ",") in registrado.ficha
    assert "Fica de fora a idade (ADR-0011)" in registrado.ficha
    assert "minutos_desde_o_aparelho_novo, outros_clientes_no_aparelho.\n" in registrado.ficha
    assert "| 65 ou mais |" in registrado.ficha
    assert f'mlflow.pyfunc.load_model("models:/{NOME}@{CAMPEAO}")' in registrado.ficha


def test_o_modelo_carregado_le_as_arvores_a_calibracao_os_custos_e_os_tipos(
    variaveis: pl.DataFrame, setembro: pl.DataFrame, tmp_path: Path
) -> None:
    modelo, _ = treinar_o_final(simulacao_media(), variaveis)
    custos = Custos(confirmar=40.0, bloquear_um_honesto=400.0, confirmacao_para=0.5)
    contexto = SimpleNamespace(artifacts=gravar_os_artefatos(modelo, custos, variaveis, tmp_path))
    carregado = ModeloDoPix()
    carregado.load_context(contexto)
    saida = pl.from_pandas(carregado.predict(None, para_a_tabela(setembro)))
    esperado = decidir(setembro.with_columns(pl.Series("probabilidade", modelo.probabilidade(setembro))), custos)

    assert carregado.custos == custos
    assert carregado.tipos == _tipos(variaveis)
    assert np.allclose(saida["probabilidade"].to_numpy(), modelo.probabilidade(setembro), rtol=0, atol=1e-12)
    assert saida["acao"].to_list() == esperado["acao"].cast(pl.String).to_list()


def test_o_modelo_que_nao_foi_carregado_avisa(setembro: pl.DataFrame) -> None:
    with pytest.raises(RuntimeError, match="ainda não foi carregado"):
        ModeloDoPix().predict(None, para_a_tabela(setembro.head(2)))


def _tipos(variaveis: pl.DataFrame) -> dict[str, pl.DataType]:
    return {variavel.nome: variaveis.schema[variavel.nome] for variavel in ENTRADA}


def test_faltando_variavel_o_modelo_diz_qual(setembro: pl.DataFrame, variaveis: pl.DataFrame) -> None:
    tabela = para_a_tabela(setembro.head(3)).drop(columns=["valor", "hora"])

    with pytest.raises(ValueError, match="Faltam variáveis para decidir o Pix: valor, hora"):
        da_tabela(tabela, _tipos(variaveis))


def test_a_tabela_volta_com_os_tipos_do_treino(setembro: pl.DataFrame, variaveis: pl.DataFrame) -> None:
    de_volta = da_tabela(para_a_tabela(setembro), _tipos(variaveis))

    assert de_volta.equals(setembro.select(variavel.nome for variavel in ENTRADA))
