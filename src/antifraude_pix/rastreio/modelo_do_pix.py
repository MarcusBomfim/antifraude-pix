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
import pandas as pd
import polars as pl
from mlflow.pyfunc.model import PythonModel

from antifraude_pix.decisao.custo import Custos, decidir
from antifraude_pix.decisao.explicacao import explicar
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

        if self.modelo is None:
            raise RuntimeError("O modelo ainda não foi carregado.")

        variaveis = da_tabela(model_input, self.tipos)
        pontuacao = self.modelo.pontuar(variaveis)
        decididos = decidir(
            variaveis.with_columns(
                pl.Series("pontuacao", pontuacao), pl.Series("probabilidade", self.modelo.probabilidade_da(pontuacao))
            ),
            self.custos,
        )

        return (
            explicar(self.modelo, decididos)
            .select("pontuacao", "probabilidade", pl.col("acao").cast(pl.String), "explicacao")
            .to_pandas()
        )


def da_tabela(tabela: pd.DataFrame, tipos: dict[str, pl.DataType]) -> pl.DataFrame:
    """A tabela do pandas de volta às variáveis do Polars, cada uma com o tipo com que o modelo aprendeu."""
    faltam = [nome for nome in tipos if nome not in tabela.columns]

    if faltam:
        raise ValueError(f"Faltam variáveis para decidir o Pix: {', '.join(faltam)}.")

    return pl.from_pandas(tabela[list(tipos)]).select(pl.col(nome).cast(tipo) for nome, tipo in tipos.items())


mlflow.models.set_model(ModeloDoPix())
