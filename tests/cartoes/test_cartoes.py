import hashlib
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest
from mlflow.tracking import MlflowClient

from antifraude_pix.avaliacao.decisoes import balanco
from antifraude_pix.avaliacao.metricas import resumo
from antifraude_pix.cartoes.base import ARQUIVO, COMPONENTES, MD5, baixar, ler
from antifraude_pix.cartoes.processo import (
    ARVORES,
    DIA,
    MODELOS,
    ArvoresDosCartoes,
    LogisticaDosCartoes,
    avaliar,
    com_as_variaveis,
    dividir,
    separar,
    treinar,
)
from antifraude_pix.cartoes.rastreio import EXPERIMENTO, rastrear
from antifraude_pix.modelos.treino import LIGHTGBM, SEMENTE
from antifraude_pix.rastreio.experimentos import endereco
from tests.apoio import COMPRAS, cartoes_inventados


@pytest.fixture(scope="module")
def arquivo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return cartoes_inventados(tmp_path_factory.mktemp("origem"))


@pytest.fixture(scope="module")
def compras(arquivo: Path) -> pl.DataFrame:
    return ler(arquivo)


@pytest.fixture(scope="module")
def pontuacoes(compras: pl.DataFrame) -> pl.DataFrame:
    return avaliar(compras)


def test_a_base_lida_tem_o_tempo_as_componentes_o_valor_e_a_fraude(compras: pl.DataFrame) -> None:
    assert compras.columns == ["id", "segundo", *COMPONENTES, "valor", "fraude"]
    assert compras.height == COMPRAS
    assert compras["id"].to_list() == list(range(COMPRAS))
    assert compras.schema["fraude"] == pl.Boolean
    assert 0.01 < compras["fraude"].mean() < 0.05  # type: ignore[operator]
    assert compras["valor"].min() > 0  # type: ignore[operator]


def test_o_numero_em_notacao_cientifica_e_lido_como_numero(tmp_path: Path) -> None:
    # O arquivo do OpenML escreve o segundo 100.000 como "1e+05", depois de milhares de segundos inteiros.
    arquivo = cartoes_inventados(tmp_path, linhas=3)
    linhas = arquivo.read_text(encoding="utf-8").splitlines()
    dados = linhas.index("@data") + 1
    linhas[dados + 2] = "1e+05" + linhas[dados + 2][linhas[dados + 2].index(",") :]
    arquivo.write_text("\n".join(linhas) + "\n", encoding="utf-8")

    assert ler(arquivo)["segundo"][2] == 100_000.0


def test_o_arquivo_com_outras_colunas_e_recusado(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="não tem as colunas da base de cartões"):
        ler(cartoes_inventados(tmp_path, linhas=5, colunas=("Time", "V1", "Amount")))


def test_o_download_confere_o_md5_e_nao_baixa_de_novo(arquivo: Path, tmp_path: Path) -> None:
    md5 = hashlib.md5(arquivo.read_bytes(), usedforsecurity=False).hexdigest()
    destino = baixar(tmp_path, arquivo.as_uri(), md5)

    assert destino == tmp_path / ARQUIVO
    assert destino.read_bytes() == arquivo.read_bytes()
    # Já está lá, e confere: nem tenta o endereço, que agora não existe.
    assert baixar(tmp_path, (tmp_path / "nao-existe.arff").as_uri(), md5) == destino


def test_o_arquivo_que_nao_confere_e_jogado_fora(arquivo: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="não é o publicado"):
        baixar(tmp_path, arquivo.as_uri(), "0" * 32)

    assert list(tmp_path.iterdir()) == []


def test_o_treino_e_o_primeiro_dia_e_o_teste_o_segundo() -> None:
    compras = pl.DataFrame({"segundo": [0.0, DIA - 0.5, float(DIA), 2 * DIA - 1.0], "fraude": [False] * 4})
    treino, teste = dividir(com_as_variaveis(compras))

    assert treino["segundo"].to_list() == [0.0, DIA - 0.5]
    assert teste["segundo"].to_list() == [float(DIA), 2 * DIA - 1.0]
    assert com_as_variaveis(compras)["hora"].to_list() == pytest.approx([0.0, 24 - 0.5 / 3_600, 0.0, 24 - 1 / 3_600])


def test_cada_modelo_pontua_cada_compra_do_segundo_dia_no_formato_do_pix(
    compras: pl.DataFrame, pontuacoes: pl.DataFrame
) -> None:
    _, teste = dividir(compras)

    assert pontuacoes.columns == ["id", "corte", "modelo", "pontuacao", "probabilidade", "valor", "golpe"]
    assert pontuacoes["modelo"].unique(maintain_order=True).to_list() == list(MODELOS)

    for (_,), do_modelo in pontuacoes.group_by("modelo"):
        assert do_modelo["id"].to_list() == teste["id"].to_list()
        assert do_modelo.filter(pl.col("golpe") == "fraude")["id"].to_list() == teste.filter("fraude")["id"].to_list()

    assert pontuacoes["probabilidade"].is_between(0, 1).all()


def test_as_metricas_e_a_decisao_do_pix_medem_os_cartoes(pontuacoes: pl.DataFrame) -> None:
    precisao = {linha["modelo"]: linha["precisao_media"] for linha in resumo(pontuacoes).iter_rows(named=True)}
    custos = balanco(pontuacoes).filter(pl.col("modelo") == "lightgbm")
    custo = dict(zip(custos["politica"], custos["custo"], strict=True))

    # Na base inventada, a fraude está em V1 e V3: os modelos que aprendem acham, e o valor sozinho, não.
    assert precisao["lightgbm"] > 0.8
    assert precisao["logistica"] > 0.8
    assert precisao["valor"] < 0.2
    assert custo["custo"] < custo["aprovar_tudo"]


def test_o_mesmo_treino_da_a_mesma_probabilidade(compras: pl.DataFrame) -> None:
    treino, teste = dividir(com_as_variaveis(compras))

    for nome in MODELOS:
        assert np.array_equal(treinar(nome, treino).probabilidade(teste), treinar(nome, treino).probabilidade(teste))


def test_a_calibracao_e_um_sorteio_que_o_ajuste_nao_ve(compras: pl.DataFrame) -> None:
    treino, _ = dividir(com_as_variaveis(compras))
    ajuste, calibracao = separar(treino, np.random.default_rng(1))

    assert ajuste.height + calibracao.height == treino.height
    assert ajuste.join(calibracao, on="id", how="semi").is_empty()
    assert 0.15 < calibracao.height / treino.height < 0.25


def test_o_modelo_aprende_so_com_o_ajuste(compras: pl.DataFrame) -> None:
    # O treino sorteia a calibração primeiro, com a mesma semente: a padronização é a do ajuste, sem a calibração.
    treino, _ = dividir(com_as_variaveis(compras))
    ajuste, _ = separar(treino, np.random.default_rng(SEMENTE))
    logistica = treinar("logistica", treino).pontuador
    componentes = ajuste.select(*COMPONENTES, pl.col("valor").log1p()).to_numpy()

    assert isinstance(logistica, LogisticaDosCartoes)
    assert np.allclose(logistica.media, componentes.mean(axis=0))
    assert not np.allclose(
        logistica.media, treino.select(*COMPONENTES, pl.col("valor").log1p()).to_numpy().mean(axis=0)
    )


def test_o_peso_devolve_as_compras_legitimas_a_proporcao_que_elas_tem(compras: pl.DataFrame) -> None:
    # As árvores veem uma compra legítima em cada dez. Sem o peso dez, a compra comum sairia com cara de fraude:
    # na base inventada, a probabilidade típica fica quinze vezes maior sem ele.
    treino, _ = dividir(com_as_variaveis(compras))
    ajuste, _ = separar(treino, np.random.default_rng(SEMENTE))
    arvores = treinar("lightgbm", treino).pontuador
    rotulo = ajuste["rotulo"].to_numpy()
    amostra = rotulo | (np.random.default_rng(3).random(rotulo.size) < 0.1)
    matriz = ajuste.select(ARVORES).to_numpy()
    sem_peso = lgb.LGBMClassifier(
        **LIGHTGBM, random_state=1, n_jobs=4, deterministic=True, force_col_wise=True, verbose=-1
    )
    sem_peso.fit(matriz[amostra], rotulo[amostra])

    def tipica(modelo: lgb.LGBMClassifier) -> float:
        return float(np.median(np.asarray(modelo.predict_proba(matriz), dtype=np.float64)[:, 1]))

    assert isinstance(arvores, ArvoresDosCartoes)
    assert tipica(arvores.modelo) < tipica(sem_peso) / 3


def test_a_compra_media_do_ajuste_pontua_o_intercepto_da_logistica(compras: pl.DataFrame) -> None:
    # Padronizada pelas contas do ajuste, a compra média vira o zero de cada coluna, e sobra só o intercepto.
    treino, _ = dividir(com_as_variaveis(compras))
    logistica = treinar("logistica", treino).pontuador
    assert isinstance(logistica, LogisticaDosCartoes)
    media = pl.DataFrame(
        {
            **{componente: [logistica.media[posicao]] for posicao, componente in enumerate(COMPONENTES)},
            "valor": [float(np.expm1(logistica.media[len(COMPONENTES)]))],
        }
    )

    assert logistica.pontuar(media)[0] == pytest.approx(float(logistica.modelo.intercept_[0]))


def test_o_modelo_tem_de_existir(compras: pl.DataFrame) -> None:
    treino, _ = dividir(com_as_variaveis(compras))

    with pytest.raises(ValueError, match="Não conheço o modelo 'regras'"):
        treinar("regras", treino)


def test_cada_modelo_vira_uma_corrida_no_experimento_dos_cartoes(pontuacoes: pl.DataFrame, tmp_path: Path) -> None:
    ids = rastrear(pontuacoes, tmp_path)
    cliente = MlflowClient(endereco(tmp_path))
    corridas = [cliente.get_run(id_) for id_ in ids]
    esperado = resumo(pontuacoes.filter(pl.col("modelo") == "lightgbm"))["precisao_media"][0]
    dos_cartoes = cliente.get_experiment_by_name(EXPERIMENTO)

    assert dos_cartoes is not None
    assert [corrida.data.tags["modelo"] for corrida in corridas] == list(MODELOS)
    assert {corrida.info.experiment_id for corrida in corridas} == {dos_cartoes.experiment_id}
    assert corridas[2].data.params["dados.md5"] == MD5
    assert corridas[2].data.params["lightgbm.num_leaves"] == "15"
    assert corridas[2].data.metrics["precisao_media"] == pytest.approx(esperado)
    assert "decisao.custo" in corridas[2].data.metrics
