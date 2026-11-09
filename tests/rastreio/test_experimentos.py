from datetime import date
from pathlib import Path

import polars as pl
import pytest
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from antifraude_pix.avaliacao.decisoes import balanco
from antifraude_pix.avaliacao.metricas import resumo
from antifraude_pix.avaliacao.no_tempo import Rotulo, avaliar
from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.modelos.preparo import ENTRADA
from antifraude_pix.modelos.rotulo import Dobra
from antifraude_pix.rastreio.experimentos import EXPERIMENTO, VARIANTES, endereco, experimento, rastrear
from antifraude_pix.variaveis.calculo import calcular
from antifraude_pix.variaveis.catalogo import CATALOGO
from tests.apoio import simulacao_media

SETEMBRO = (Dobra(date(2025, 9, 1), date(2025, 9, 29)),)


@pytest.fixture(scope="module")
def variaveis() -> pl.DataFrame:
    return calcular(VisaoDoBanco.da(simulacao_media()))


@pytest.fixture(scope="module")
def pasta(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("dados")


@pytest.fixture(scope="module")
def corridas(pasta: Path, variaveis: pl.DataFrame) -> list[Run]:
    ids = rastrear(VARIANTES[0], simulacao_media(), variaveis, SETEMBRO, pasta)
    cliente = MlflowClient(endereco(pasta))

    return [cliente.get_run(id_) for id_ in ids]


def test_cada_modelo_da_variante_vira_uma_corrida_com_o_nome_e_o_porque(pasta: Path, corridas: list[Run]) -> None:
    cliente = MlflowClient(endereco(pasta))
    do_pix = cliente.get_experiment_by_name(EXPERIMENTO)

    assert do_pix is not None
    assert [corrida.data.tags["modelo"] for corrida in corridas] == ["regras", "logistica", "lightgbm"]
    assert {corrida.data.tags["variante"] for corrida in corridas} == {"padrao"}
    assert {corrida.info.experiment_id for corrida in corridas} == {do_pix.experiment_id}
    assert corridas[2].info.run_name == "padrao · lightgbm"
    assert "rótulo do dia" in corridas[2].data.tags["mlflow.note.content"]


def test_a_corrida_guarda_os_dados_o_treino_o_modelo_e_a_decisao(corridas: list[Run]) -> None:
    _, logistica, lightgbm = (corrida.data.params for corrida in corridas)

    # A configuração da simulação identifica os dados: a mesma semente dá os mesmos dados, byte a byte.
    assert lightgbm["dados.clientes"] == "1500"
    assert lightgbm["dados.inicio"] == "2025-07-01"
    assert lightgbm["dados.semente"] == "2026"
    assert lightgbm["rotulo"] == "do_dia"
    assert lightgbm["espera_em_dias"] == "0"
    assert lightgbm["entrada.variaveis"] == str(len(ENTRADA))
    assert lightgbm["entrada.sem"] == "idade"
    assert lightgbm["lightgbm.num_leaves"] == "15"
    assert lightgbm["lightgbm.um_legitimo_em"] == "10"
    assert lightgbm["custos.confirmar"] == "10.0"
    assert lightgbm["teste.cortes"] == "2025-09-01"
    assert logistica["logistica.C"] == "0.1"
    assert not any(chave.startswith("lightgbm.") for chave in logistica)


def test_as_metricas_sao_as_da_avaliacao(pasta: Path, corridas: list[Run], variaveis: pl.DataFrame) -> None:
    lightgbm = corridas[2]
    pontuacoes = avaliar(simulacao_media(), variaveis, SETEMBRO, modelos=("lightgbm",))
    esperado = resumo(pontuacoes).row(0, named=True)
    medidas = lightgbm.data.metrics
    cliente = MlflowClient(endereco(pasta))

    assert medidas["precisao_media"] == pytest.approx(esperado["precisao_media"])
    assert medidas["recall_no_alerta"] == pytest.approx(esperado["recall_no_alerta"])
    assert medidas["golpes"] == esperado["golpes"]
    assert {"precisao_media.falso_parente", "decisao.custo", "idade.65_ou_mais.honestos_bloqueados"} <= set(medidas)
    pelo_custo = balanco(pontuacoes).filter(pl.col("politica") == "custo").row(0, named=True)
    assert medidas["decisao.custo"] == pytest.approx(pelo_custo["custo"])
    assert medidas["decisao.honestos_bloqueados"] == pelo_custo["honestos_bloqueados"]
    assert [ponto.step for ponto in cliente.get_metric_history(lightgbm.info.run_id, "mes.precisao_media")] == [1]
    assert {artefato.path for artefato in cliente.list_artifacts(lightgbm.info.run_id)} == {
        "resumo.json",
        "por_golpe.json",
        "balanco.json",
        "por_idade.json",
    }


def test_o_experimento_e_criado_uma_vez_e_fica_ao_lado_dos_dados(tmp_path: Path) -> None:
    primeiro, segundo = experimento(tmp_path), experimento(tmp_path)
    criado = MlflowClient(endereco(tmp_path)).get_experiment(primeiro)

    assert primeiro == segundo
    assert criado.artifact_location == (tmp_path / "mlflow" / "artefatos").resolve().as_uri()
    assert (tmp_path / "mlflow" / "mlflow.db").exists()
    assert endereco(tmp_path) == f"sqlite:///{(tmp_path / 'mlflow' / 'mlflow.db').resolve().as_posix()}"


def test_as_variantes_sao_as_medidas_das_adrs() -> None:
    por_nome = {variante.nome: variante for variante in VARIANTES}

    assert list(por_nome) == ["padrao", "com_a_verdade", "esperando_14_dias", "esperando_28_dias", "com_a_idade"]
    assert por_nome["padrao"].modelos == ("regras", "logistica", "lightgbm")
    assert por_nome["com_a_verdade"].rotulo is Rotulo.VERDADE
    assert [por_nome[nome].espera_em_dias for nome in ("esperando_14_dias", "esperando_28_dias")] == [14, 28]
    assert por_nome["com_a_idade"].entrada == CATALOGO
    assert all(variante.entrada == ENTRADA for variante in VARIANTES if variante.nome != "com_a_idade")
    assert all(variante.descricao.endswith(").") for variante in VARIANTES)


@pytest.mark.parametrize("nome", ["com_a_verdade", "esperando_14_dias", "com_a_idade"])
def test_cada_variante_chega_a_avaliacao_com_o_que_ela_muda(nome: str, tmp_path: Path, variaveis: pl.DataFrame) -> None:
    variante = next(variante for variante in VARIANTES if variante.nome == nome)
    (id_,) = rastrear(variante, simulacao_media(), variaveis, SETEMBRO, tmp_path)
    corrida = MlflowClient(endereco(tmp_path)).get_run(id_)
    esperado = avaliar(
        simulacao_media(),
        variaveis,
        SETEMBRO,
        modelos=("lightgbm",),
        rotulo=variante.rotulo,
        espera_em_dias=variante.espera_em_dias,
        entrada=variante.entrada,
    )

    assert corrida.data.metrics["precisao_media"] == pytest.approx(resumo(esperado)["precisao_media"][0])
    assert corrida.data.params["rotulo"] == variante.rotulo.value
    assert corrida.data.params["espera_em_dias"] == str(variante.espera_em_dias)
    assert corrida.data.params["entrada.variaveis"] == str(len(variante.entrada))
    assert corrida.data.params["entrada.sem"] == ("nenhuma" if variante.entrada == CATALOGO else "idade")
