"""O modelo que vai para o banco, registrado no MLflow com a sua ficha (ADR-0014).

O modelo registrado é o LightGBM treinado com tudo o que o banco sabia no corte,
por padrão o fim dos dados, com a probabilidade calibrada, a decisão pelo custo e a explicação
num pacote só (`modelo_do_pix.py`). É ele que a API da Parte 6 carrega, pelo
nome e pelo apelido, e não por um arquivo solto.

O registro guarda junto:
- o que o modelo aprendeu, em formatos que se leem sem executar nada: as
  árvores no texto do LightGBM, a calibração e os custos em JSON, e os tipos
  da entrada numa tabela vazia;
- a ficha do modelo, com as medidas da avaliação no tempo, como descrição da versão;
- a assinatura: as variáveis de entrada, com o tipo e o vazio de cada uma, e as colunas de saída;
- o código do pacote e as versões das bibliotecas, para o modelo carregar igual em outra máquina;
- o apelido `campeao`, ou o `desafiante` do retreino, que fica com o de
  campeão quando ganha a comparação (ADR-0021).
"""

import json
import logging
import tempfile
import warnings
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pandas.errors import Pandas4Warning

import antifraude_pix
from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.decisao.custo import Custos
from antifraude_pix.modelos.preparo import ENTRADA
from antifraude_pix.modelos.rotulo import janela_de_treino
from antifraude_pix.modelos.treino import SEMENTE, Arvores, Calibrado, separar, treinar
from antifraude_pix.rastreio import modelo_do_pix
from antifraude_pix.rastreio.experimentos import VARIANTES, endereco, experimento, metricas, parametros
from antifraude_pix.rastreio.ficha import Treino, ficha
from antifraude_pix.rastreio.modelo_do_pix import ModeloDoPix
from antifraude_pix.simulacao.simulador import Simulacao

NOME = "antifraude-pix"
"""O nome do modelo no registro do MLflow."""

CAMPEAO = "campeao"
"""O apelido da versão que decide: a API carrega `models:/antifraude-pix@campeao`."""

DESAFIANTE = "desafiante"
"""O apelido da última versão treinada para disputar com o campeão (ADR-0021)."""

VEREDITO = "veredito"
"""A etiqueta da versão do desafiante com o resultado da comparação: `promovido` ou `recusado`."""

BIBLIOTECAS = ("lightgbm", "mlflow", "numpy", "pandas", "polars", "pyarrow", "scikit-learn")
"""O que o modelo precisa para carregar, com a versão exata de quem o registrou."""

ARQUIVOS = {"arvores": "arvores.txt", "ajustes": "ajustes.json", "tipos": "tipos.parquet"}
"""Os artefatos do modelo, cada um num formato que se lê sem executar nada."""


@dataclass(frozen=True, slots=True)
class Registrado:
    """O modelo registrado: a versão, a corrida que o gravou, e a ficha."""

    versao: str
    corrida: str
    ficha: str


@dataclass(frozen=True, slots=True)
class Carregado:
    """Um modelo do registro, pelo apelido: a versão, o corte com que foi treinado, o modelo e os custos da decisão."""

    versao: str
    corte: date
    modelo: Any
    """O modelo calibrado, da cópia do código guardada com ele: pontua e dá a probabilidade, como o `Calibrado`."""
    custos: Custos


def para_a_tabela(variaveis: pl.DataFrame) -> pd.DataFrame:
    """As variáveis da entrada como o MLflow as recebe: uma tabela do pandas, com as categorias em texto."""
    nomes = [variavel.nome for variavel in ENTRADA]
    categorias = [nome for nome in nomes if isinstance(variaveis.schema[nome], pl.Enum)]

    return variaveis.select(nomes).with_columns(pl.col(categorias).cast(pl.String)).to_pandas()


def exemplo(variaveis: pl.DataFrame) -> pl.DataFrame:
    """Uns poucos Pix que mostram cada variável cheia e, quando ela pode ser vazia, vazia.

    A assinatura sai deste exemplo: a variável que nunca aparece vazia nele vira obrigatória.
    """
    vazios = [variaveis.filter(pl.col(variavel.nome).is_null()).head(1) for variavel in ENTRADA]

    return pl.concat([variaveis.head(3), *vazios]).unique("id", keep="first", maintain_order=True)


def treinar_o_final(
    simulacao: Simulacao, variaveis: pl.DataFrame, corte: date | None = None
) -> tuple[Calibrado, Treino]:
    """O LightGBM treinado com tudo o que o banco sabia no corte: por padrão, o dia seguinte ao último Pix."""
    visao = VisaoDoBanco.da(simulacao)

    if corte is None:
        ultimo = visao.transacoes["momento"].max()

        if not isinstance(ultimo, datetime):
            raise ValueError("A simulação não tem nenhum Pix para treinar o modelo.")

        corte = ultimo.date() + timedelta(days=1)

    janela = janela_de_treino(visao, variaveis, corte)
    modelo = treinar("lightgbm", janela, corte)
    # O treino separa a calibração com o mesmo sorteio: a mesma semente dá os mesmos Pix.
    _, calibracao = separar(janela, corte, np.random.default_rng(SEMENTE))

    return modelo, Treino(corte, janela.height, int(janela["rotulo"].sum()), calibracao.height)


def gravar_os_artefatos(modelo: Calibrado, custos: Custos, variaveis: pl.DataFrame, pasta: Path) -> dict[str, str]:
    """As árvores, os ajustes e os tipos da entrada, gravados na pasta. Devolve o caminho de cada um."""
    if not isinstance(modelo.pontuador, Arvores):
        raise TypeError(f"O registro é das árvores, e não de {modelo.nome}.")

    arvores = modelo.pontuador
    booster = arvores.modelo if isinstance(arvores.modelo, lgb.Booster) else arvores.modelo.booster_
    caminhos = {nome: pasta / arquivo for nome, arquivo in ARQUIVOS.items()}
    booster.save_model(caminhos["arvores"])
    ajustes = {
        "inclinacao": modelo.inclinacao,
        "intercepto": modelo.intercepto,
        "entrada": [variavel.nome for variavel in arvores.entrada],
        "custos": asdict(custos),
    }
    caminhos["ajustes"].write_text(json.dumps(ajustes, ensure_ascii=False, indent=2), encoding="utf-8")
    variaveis.select(variavel.nome for variavel in arvores.entrada).clear().write_parquet(caminhos["tipos"])

    return {nome: str(caminho) for nome, caminho in caminhos.items()}


def registrar(
    simulacao: Simulacao,
    variaveis: pl.DataFrame,
    pontuacoes: pl.DataFrame,
    pasta_dos_dados: Path,
    *,
    custos: Custos | None = None,
    corte: date | None = None,
    apelido: str = CAMPEAO,
) -> Registrado:
    """Treina o modelo com o que o banco sabia no corte, grava a corrida com a ficha e registra a versão com o apelido.

    As `pontuacoes` são as do LightGBM nos meses de teste, da avaliação no tempo: são elas que a ficha mostra.
    Sem o corte, o modelo sabe tudo o que os dados têm; o apelido, sem outro, é o de campeão.
    """
    custos = custos or Custos()
    # O ambiente do uv não tem o pip, e o MLflow avisa que não achou a versão dele: as bibliotecas vão fixadas.
    logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
    modelo, treino = treinar_o_final(simulacao, variaveis, corte)
    idades = variaveis.select("id", "idade")
    texto = ficha(simulacao.config, treino, pontuacoes, idades, custos=custos)
    entrada = para_a_tabela(exemplo(variaveis))
    saida = ModeloDoPix(modelo, custos, {nome: variaveis.schema[nome] for nome in entrada.columns}).predict(
        None, entrada
    )
    id_do_experimento = experimento(pasta_dos_dados)
    cortes = pontuacoes["corte"].unique().sort()

    with (
        mlflow.start_run(experiment_id=id_do_experimento, run_name="registro · lightgbm") as corrida,
        tempfile.TemporaryDirectory() as temporaria,
        warnings.catch_warnings(),
    ):
        # Dois avisos do MLflow sobre o próprio código: ele compara as colunas do exemplo com um range, que o
        # pandas 3 vai mudar, e lembra que inteiro não guarda vazio; aqui, nenhuma variável inteira fica vazia.
        warnings.filterwarnings("ignore", "Operation with range is deprecated", Pandas4Warning)
        warnings.filterwarnings("ignore", ".*Inferred schema contains integer column", UserWarning)
        mlflow.set_tags(
            {"variante": "registro", "modelo": "lightgbm", "apelido": apelido, "mlflow.note.content": texto}
        )
        mlflow.log_params(
            parametros(VARIANTES[0], "lightgbm", simulacao, (), custos)
            | {
                "teste.cortes": ", ".join(f"{corte:%Y-%m-%d}" for corte in cortes),
                "treino.corte": f"{treino.corte:%Y-%m-%d}",
                "treino.pix": treino.pix,
                "treino.golpes_sabidos": treino.golpes_sabidos,
            }
        )
        mlflow.log_metrics(metricas(pontuacoes, idades, custos))
        mlflow.log_text(texto, "ficha.md")
        gravado = mlflow.pyfunc.log_model(
            name="modelo",
            python_model=modelo_do_pix.__file__,
            artifacts=gravar_os_artefatos(modelo, custos, variaveis, Path(temporaria)),
            signature=infer_signature(entrada, saida),
            input_example=entrada,
            code_paths=[str(Path(antifraude_pix.__file__).parent)],
            pip_requirements=[f"{biblioteca}=={version(biblioteca)}" for biblioteca in BIBLIOTECAS],
        )

    registrado = mlflow.register_model(gravado.model_uri, NOME)
    cliente = MlflowClient()
    cliente.update_model_version(NOME, registrado.version, description=texto)
    cliente.set_registered_model_alias(NOME, apelido, registrado.version)

    return Registrado(str(registrado.version), corrida.info.run_id, texto)


def carregar(pasta_dos_dados: Path, apelido: str = CAMPEAO) -> Carregado:
    """O modelo com o apelido no registro do MLflow da pasta, com a versão e o corte com que foi treinado."""
    mlflow.set_tracking_uri(endereco(pasta_dos_dados))
    cliente = MlflowClient()
    versao = cliente.get_model_version_by_alias(NOME, apelido)
    corte = date.fromisoformat(cliente.get_run(str(versao.run_id)).data.params["treino.corte"])
    por_dentro = mlflow.pyfunc.load_model(f"models:/{NOME}@{apelido}").unwrap_python_model()

    return Carregado(str(versao.version), corte, por_dentro.modelo, por_dentro.custos)


def dar_o_veredito(pasta_dos_dados: Path, versao: str, *, promovido: bool) -> None:
    """Grava o veredito na versão do desafiante e, se ele ganhou, passa a ela o apelido de campeão."""
    mlflow.set_tracking_uri(endereco(pasta_dos_dados))
    cliente = MlflowClient()
    cliente.set_model_version_tag(NOME, versao, VEREDITO, "promovido" if promovido else "recusado")

    if promovido:
        cliente.set_registered_model_alias(NOME, CAMPEAO, versao)
