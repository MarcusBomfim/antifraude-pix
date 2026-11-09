"""Os experimentos rastreados no MLflow (ADR-0013).

Cada medida que decidiu alguma coisa nas Partes 3 e 4 vira uma corrida no MLflow:
- os três modelos como ficaram;
- o LightGBM com a verdade, que mede o preço do rótulo atrasado;
- o LightGBM esperando o rótulo amadurecer, 14 e 28 dias;
- o LightGBM com a idade, que a Parte 4 tirou do modelo.

Cada corrida guarda o que é preciso para refazê-la e para compará-la:
- os parâmetros: a configuração da simulação, que dá os mesmos dados byte a
  byte, os parâmetros do modelo, o rótulo, a espera, a entrada e os custos;
- as métricas de todos os meses de teste juntos, e a de cada mês, como passo;
- as tabelas da avaliação, em JSON;
- o commit do código, que o MLflow anota sozinho quando roda num repositório Git.

O rastreio fica ao lado dos dados, num banco SQLite e numa pasta de artefatos,
e não vai para o Git: sai da mesma semente, como os dados.
"""

import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import mlflow
import polars as pl
from mlflow.tracking import MlflowClient

from antifraude_pix.avaliacao.decisoes import balanco
from antifraude_pix.avaliacao.justica import por_idade
from antifraude_pix.avaliacao.metricas import ALERTA, por_golpe, resumo
from antifraude_pix.avaliacao.no_tempo import Rotulo, avaliar
from antifraude_pix.decisao.custo import Custos, decidir
from antifraude_pix.modelos.preparo import ENTRADA, PROTEGIDAS
from antifraude_pix.modelos.rotulo import AQUECIMENTO, VALIDACAO, Dobra
from antifraude_pix.modelos.treino import (
    CALIBRACAO,
    LIGHTGBM,
    LOGISTICA,
    MADURO_EM_DIAS,
    MODELOS,
    SEMENTE,
    UM_LEGITIMO_EM,
)
from antifraude_pix.simulacao.simulador import Simulacao
from antifraude_pix.variaveis.catalogo import CATALOGO, Variavel

PASTA = "mlflow"
"""A pasta do rastreio, dentro da pasta dos dados: o banco `mlflow.db` e os `artefatos`."""

EXPERIMENTO = "antifraude-pix"
"""O experimento do MLflow em que caem as corridas do Pix."""


@dataclass(frozen=True, slots=True)
class Variante:
    """Um jeito de treinar, com o que ele muda no padrão, e os modelos que ele treina."""

    nome: str
    descricao: str
    modelos: tuple[str, ...] = ("lightgbm",)
    rotulo: Rotulo = Rotulo.DO_DIA
    espera_em_dias: int = 0
    entrada: tuple[Variavel, ...] = ENTRADA


VARIANTES: tuple[Variante, ...] = (
    Variante(
        "padrao",
        "Os três modelos como ficaram: o rótulo do dia, sem esperar, e sem a idade (ADRs 0007, 0008 e 0011).",
        modelos=tuple(MODELOS),
    ),
    Variante(
        "com_a_verdade",
        "O LightGBM aprendendo com a verdade, que nenhum banco tem: o preço do rótulo atrasado (ADR-0007).",
        rotulo=Rotulo.VERDADE,
    ),
    Variante(
        "esperando_14_dias",
        "O LightGBM esperando 14 dias para o rótulo amadurecer, e perdendo os golpes mais recentes (ADR-0007).",
        espera_em_dias=14,
    ),
    Variante(
        "esperando_28_dias",
        "O LightGBM esperando 28 dias para o rótulo amadurecer (ADR-0007).",
        espera_em_dias=28,
    ),
    Variante(
        "com_a_idade",
        "O LightGBM recebendo a idade do cliente, como na Parte 3, antes de ela sair do modelo (ADR-0011).",
        entrada=CATALOGO,
    ),
)
"""As variantes medidas: o padrão, e cada desvio dele que uma ADR mediu antes de decidir."""


def endereco(pasta_dos_dados: Path) -> str:
    """O endereço do rastreio: o banco SQLite dentro da pasta do rastreio, que é criada se não existir."""
    pasta = (pasta_dos_dados / PASTA).resolve()
    pasta.mkdir(parents=True, exist_ok=True)

    return f"sqlite:///{(pasta / 'mlflow.db').as_posix()}"


def experimento(pasta_dos_dados: Path, nome: str = EXPERIMENTO) -> str:
    """Aponta o MLflow para o rastreio da pasta dos dados, e devolve o id do experimento, criado se preciso."""
    # O MLflow anuncia cada passo no registro; aqui, só o que for aviso.
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    mlflow.set_tracking_uri(endereco(pasta_dos_dados))
    cliente = MlflowClient()
    existente = cliente.get_experiment_by_name(nome)

    if existente is not None:
        return str(existente.experiment_id)

    artefatos = (pasta_dos_dados / PASTA / "artefatos").resolve()

    return str(cliente.create_experiment(nome, artifact_location=artefatos.as_uri()))


def parametros(
    variante: Variante, modelo: str, simulacao: Simulacao, dobras: tuple[Dobra, ...], custos: Custos
) -> dict[str, Any]:
    """Os parâmetros de uma corrida: os dados, o treino, o modelo, a decisão e os meses de teste."""
    dados = {f"dados.{chave}": _texto(valor) for chave, valor in asdict(simulacao.config).items()}
    do_modelo: dict[str, Any] = {}

    if modelo == "lightgbm":
        do_modelo = {f"lightgbm.{chave}": valor for chave, valor in LIGHTGBM.items()}
        do_modelo["lightgbm.um_legitimo_em"] = UM_LEGITIMO_EM
    elif modelo == "logistica":
        do_modelo = {f"logistica.{chave}": valor for chave, valor in LOGISTICA.items()}

    return {
        **dados,
        "modelo": modelo,
        "variante": variante.nome,
        "rotulo": variante.rotulo.value,
        "espera_em_dias": variante.espera_em_dias,
        "entrada.variaveis": len(variante.entrada),
        "entrada.sem": ", ".join(variavel.nome for variavel in CATALOGO if variavel not in variante.entrada)
        or "nenhuma",
        "entrada.protegidas": ", ".join(PROTEGIDAS),
        "treino.aquecimento_em_dias": AQUECIMENTO,
        "treino.meses_de_validacao": VALIDACAO,
        "treino.semente": SEMENTE,
        "calibracao.parte": CALIBRACAO,
        "calibracao.maduro_em_dias": MADURO_EM_DIAS,
        **do_modelo,
        "alerta": ALERTA,
        "custos.confirmar": custos.confirmar,
        "custos.bloquear_um_honesto": custos.bloquear_um_honesto,
        "custos.confirmacao_para": custos.confirmacao_para,
        "teste.cortes": ", ".join(f"{dobra.corte:%Y-%m-%d}" for dobra in dobras),
    }


def metricas(pontuacoes: pl.DataFrame, idades: pl.DataFrame, custos: Custos) -> dict[str, float]:
    """As métricas de um modelo, com os meses de teste juntos: a fila, cada golpe, a decisão e cada idade."""
    medidas: dict[str, float] = {}

    for chave, valor in resumo(pontuacoes).drop("modelo").row(0, named=True).items():
        if valor is not None:
            medidas[chave] = float(valor)

    for linha in por_golpe(pontuacoes).iter_rows(named=True):
        medidas[f"precisao_media.{linha['golpe']}"] = linha["precisao_media"]
        medidas[f"recall_no_alerta.{linha['golpe']}"] = linha["recall_no_alerta"]

    pelo_custo = balanco(pontuacoes, custos).filter(pl.col("politica") == "custo").row(0, named=True)

    for chave in ("custo", "perdido", "golpes_parados", "honestos_segurados", "honestos_bloqueados"):
        medidas[f"decisao.{chave}"] = float(pelo_custo[chave])

    decididos = decidir(pontuacoes, custos).join(idades, on="id")

    for linha in por_idade(decididos).iter_rows(named=True):
        faixa = str(linha["faixa"]).replace(" ", "_")

        for chave in ("recall", "honestos_segurados", "honestos_bloqueados"):
            medidas[f"idade.{faixa}.{chave}"] = float(linha[chave])

    return medidas


def rastrear(
    variante: Variante,
    simulacao: Simulacao,
    variaveis: pl.DataFrame,
    dobras: tuple[Dobra, ...],
    pasta_dos_dados: Path,
    *,
    custos: Custos | None = None,
) -> list[str]:
    """Avalia a variante no tempo e grava uma corrida por modelo no MLflow. Devolve os ids das corridas."""
    custos = custos or Custos()
    id_do_experimento = experimento(pasta_dos_dados)
    pontuacoes = avaliar(
        simulacao,
        variaveis,
        dobras,
        modelos=variante.modelos,
        rotulo=variante.rotulo,
        espera_em_dias=variante.espera_em_dias,
        entrada=variante.entrada,
    )
    idades = variaveis.select("id", "idade")
    corridas = []

    for modelo in variante.modelos:
        do_modelo = pontuacoes.filter(pl.col("modelo") == modelo)

        with mlflow.start_run(experiment_id=id_do_experimento, run_name=f"{variante.nome} · {modelo}") as corrida:
            mlflow.set_tags({"variante": variante.nome, "modelo": modelo, "mlflow.note.content": variante.descricao})
            mlflow.log_params(parametros(variante, modelo, simulacao, dobras, custos))
            mlflow.log_metrics(metricas(do_modelo, idades, custos))

            # Cada mês de teste é um passo: o MLflow desenha a precisão média mês a mês.
            for passo, (_, do_mes) in enumerate(do_modelo.group_by("corte", maintain_order=True), start=1):
                # O mês sem nenhum golpe, como o pedaço de mês no fim de uma simulação curta, não tem fila a medir.
                if not do_mes["golpe"].is_not_null().any():
                    continue

                mes = resumo(do_mes).row(0, named=True)
                mlflow.log_metrics(
                    {"mes.precisao_media": mes["precisao_media"], "mes.recall_no_alerta": mes["recall_no_alerta"]},
                    step=passo,
                )

            mlflow.log_dict(_tabela(resumo(do_modelo)), "resumo.json")
            mlflow.log_dict(_tabela(por_golpe(do_modelo)), "por_golpe.json")
            mlflow.log_dict(_tabela(balanco(do_modelo, custos)), "balanco.json")
            mlflow.log_dict(_tabela(por_idade(decidir(do_modelo, custos).join(idades, on="id"))), "por_idade.json")
            corridas.append(corrida.info.run_id)

    return corridas


def _tabela(tabela: pl.DataFrame) -> dict[str, Any]:
    """A tabela como JSON: as categorias e as datas viram texto."""
    textos = [nome for nome, tipo in tabela.schema.items() if isinstance(tipo, pl.Enum) or tipo == pl.Date]

    return {"linhas": tabela.with_columns(pl.col(textos).cast(pl.String)).to_dicts()}


def _texto(valor: object) -> object:
    return f"{valor:%Y-%m-%d}" if isinstance(valor, date) else valor
