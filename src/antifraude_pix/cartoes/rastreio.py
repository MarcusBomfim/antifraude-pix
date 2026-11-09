"""As corridas da base de cartões no MLflow, num experimento só delas (ADR-0015)."""

from pathlib import Path

import mlflow
import polars as pl

from antifraude_pix.avaliacao.decisoes import balanco
from antifraude_pix.avaliacao.metricas import ALERTA, resumo
from antifraude_pix.cartoes.base import ENDERECO, MD5
from antifraude_pix.cartoes.processo import NOMES
from antifraude_pix.decisao.custo import Custos
from antifraude_pix.modelos.treino import CALIBRACAO, LIGHTGBM, LOGISTICA, SEMENTE, UM_LEGITIMO_EM
from antifraude_pix.rastreio.experimentos import experimento

EXPERIMENTO = "cartoes-ulb"


def rastrear(pontuacoes: pl.DataFrame, pasta_dos_dados: Path, custos: Custos | None = None) -> list[str]:
    """Uma corrida por modelo, com a base de onde veio, os parâmetros, e as métricas do segundo dia."""
    custos = custos or Custos()
    id_do_experimento = experimento(pasta_dos_dados, EXPERIMENTO)
    corridas = []

    for (modelo,), do_modelo in pontuacoes.group_by("modelo", maintain_order=True):
        nome = str(modelo)
        parametros: dict[str, object] = {
            "dados.origem": ENDERECO,
            "dados.md5": MD5,
            "modelo": nome,
            "treino": "o primeiro dia",
            "teste": "o segundo dia",
            "treino.semente": SEMENTE,
            "calibracao.parte": CALIBRACAO,
            "alerta": ALERTA,
            "custos.confirmar": custos.confirmar,
            "custos.bloquear_um_honesto": custos.bloquear_um_honesto,
            "custos.confirmacao_para": custos.confirmacao_para,
        }

        if nome == "lightgbm":
            parametros |= {f"lightgbm.{chave}": valor for chave, valor in LIGHTGBM.items()}
            parametros["lightgbm.um_legitimo_em"] = UM_LEGITIMO_EM
        elif nome == "logistica":
            parametros |= {f"logistica.{chave}": valor for chave, valor in LOGISTICA.items()}

        medidas = {
            chave: float(valor)
            for chave, valor in resumo(do_modelo).drop("modelo").row(0, named=True).items()
            if valor is not None
        }
        pelo_custo = balanco(do_modelo, custos).filter(pl.col("politica") == "custo").row(0, named=True)
        medidas |= {
            f"decisao.{chave}": float(pelo_custo[chave])
            for chave in ("custo", "perdido", "golpes_parados", "honestos_segurados", "honestos_bloqueados")
        }

        with mlflow.start_run(experiment_id=id_do_experimento, run_name=f"cartoes · {nome}") as corrida:
            mlflow.set_tags({"modelo": nome, "mlflow.note.content": f"{NOMES[nome]}, na base da ULB."})
            mlflow.log_params(parametros)
            mlflow.log_metrics(medidas)
            corridas.append(corrida.info.run_id)

    return corridas
