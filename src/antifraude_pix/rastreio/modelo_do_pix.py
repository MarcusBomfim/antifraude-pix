"""O modelo registrado, como código: o MLflow carrega este arquivo, e não um objeto guardado em pickle (ADR-0014).

O que o modelo aprendeu vai em formatos que se leem sem executar nada:
- as árvores, no texto em que o próprio LightGBM as grava;
- a reta da calibração, os custos da decisão e as variáveis da entrada, num JSON;
- o tipo de cada variável, numa tabela vazia em Parquet, que guarda até as categorias.

Ao carregar, o MLflow executa este arquivo, que aponta o modelo com `set_model`,
e entrega os artefatos ao `load_context`. Entram as variáveis de um Pix; saem a
pontuação, a probabilidade, a decisão e a explicação.
"""

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from mlflow.pyfunc.model import PythonModel

from antifraude_pix.decisao.custo import Acao, Custos, acao_de, decidir
from antifraude_pix.decisao.explicacao import explicar, texto
from antifraude_pix.modelos.treino import Arvores, Calibrado
from antifraude_pix.variaveis.catalogo import CATALOGO

SAIDA = ("pontuacao", "probabilidade", "acao", "explicacao")
"""As colunas que o modelo devolve, uma linha por Pix."""


class ModeloDoPix(PythonModel):
    """O LightGBM calibrado, a decisão pelo custo e a explicação: entram as variáveis, sai a decisão explicada."""

    def __init__(
        self,
        modelo: Calibrado | None = None,
        custos: Custos | None = None,
        tipos: dict[str, pl.DataType] | None = None,
    ) -> None:
        self.modelo = modelo
        self.custos = custos or Custos()
        self.tipos = tipos or {}

    def load_context(self, context: Any) -> None:
        """Lê as árvores, a calibração, os custos e os tipos dos artefatos gravados no registro."""
        ajustes = json.loads(Path(context.artifacts["ajustes"]).read_text(encoding="utf-8"))
        entrada = tuple(variavel for variavel in CATALOGO if variavel.nome in ajustes["entrada"])
        arvores = Arvores(lgb.Booster(model_file=context.artifacts["arvores"]), entrada)
        self.modelo = Calibrado("lightgbm", arvores, ajustes["inclinacao"], ajustes["intercepto"])
        self.custos = Custos(**ajustes["custos"])
        self.tipos = dict(pl.read_parquet(context.artifacts["tipos"]).schema)

    def predict(self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """A decisão explicada de cada Pix, na ordem em que os Pix vieram."""
        del context, params

        return self.decidir(da_tabela(model_input, self.tipos)).to_pandas()

    def decidir(self, variaveis: pl.DataFrame) -> pl.DataFrame:
        """A decisão explicada de cada Pix, direto do Polars: o caminho do serviço, sem passar pelo pandas.

        As variáveis vêm com os nomes da entrada; as categorias podem vir em texto, e viram as do treino.
        """
        if self.modelo is None:
            raise RuntimeError("O modelo ainda não foi carregado.")

        com_os_tipos = variaveis.select(pl.col(nome).cast(tipo) for nome, tipo in self.tipos.items())
        pontuacao = self.modelo.pontuar(com_os_tipos)
        decididos = decidir(
            com_os_tipos.with_columns(
                pl.Series("pontuacao", pontuacao), pl.Series("probabilidade", self.modelo.probabilidade_da(pontuacao))
            ),
            self.custos,
        )

        return explicar(self.modelo, decididos).select(
            "pontuacao", "probabilidade", pl.col("acao").cast(pl.String), "explicacao"
        )

    def decidir_um(self, variaveis: dict[str, Any]) -> dict[str, Any]:
        """A decisão explicada de um Pix só: o caminho do serviço, com as contas em NumPy e em Python (ADR-0018).

        Numa tabela de uma linha, o Polars gasta mais em preparar a conta do que em fazê-la. Aqui, a linha das
        árvores sai direto das variáveis, o LightGBM pontua com um fio só, e a decisão é a mesma conta de `decidir`,
        na mesma ordem: o resultado é o mesmo até o último bit, e um teste confere. Só o Pix segurado ou bloqueado,
        que tem razões a explicar, passa pelo caminho da tabela.
        """
        if self.modelo is None:
            raise RuntimeError("O modelo ainda não foi carregado.")

        arvores = self.modelo.pontuador

        if not isinstance(arvores, Arvores):
            return dict(self.decidir(pl.DataFrame([variaveis])).row(0, named=True))

        booster = arvores.modelo if isinstance(arvores.modelo, lgb.Booster) else arvores.modelo.booster_
        linha = np.array([[self._numero(variavel.nome, variaveis[variavel.nome]) for variavel in arvores.entrada]])
        pontuacao = np.asarray(booster.predict(linha, raw_score=True, num_threads=1), dtype=np.float64)
        probabilidade = float(self.modelo.probabilidade_da(pontuacao)[0])
        saida = acao_de(probabilidade, variaveis["valor"], self.custos)

        if saida is not Acao.APROVAR:
            return dict(self.decidir(pl.DataFrame([variaveis])).row(0, named=True))

        return {
            "pontuacao": float(pontuacao[0]),
            "probabilidade": probabilidade,
            "acao": saida.value,
            "explicacao": texto(saida, probabilidade, ()),
        }

    def _numero(self, nome: str, valor: Any) -> float:
        """O valor como as árvores o recebem: o vazio vira NaN, o sim ou não vira 1 ou 0, e a categoria, o código."""
        if valor is None:
            return float("nan")

        tipo = self.tipos[nome]

        if isinstance(tipo, pl.Enum):
            return float(tipo.categories.to_list().index(valor))

        return float(valor)


def da_tabela(tabela: pd.DataFrame, tipos: dict[str, pl.DataType]) -> pl.DataFrame:
    """A tabela do pandas de volta às variáveis do Polars, cada uma com o tipo com que o modelo aprendeu."""
    faltam = [nome for nome in tipos if nome not in tabela.columns]

    if faltam:
        raise ValueError(f"Faltam variáveis para decidir o Pix: {', '.join(faltam)}.")

    return pl.from_pandas(tabela[list(tipos)]).select(pl.col(nome).cast(tipo) for nome, tipo in tipos.items())


mlflow.models.set_model(ModeloDoPix())
