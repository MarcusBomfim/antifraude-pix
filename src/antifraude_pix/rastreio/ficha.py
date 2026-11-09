"""A ficha do modelo: o que ele faz, com que dados, como foi medido, e onde ele erra (ADR-0014).

A ficha sai das próprias medidas, e não de um texto escrito à mão: os números
dela são os da avaliação no tempo, com o mesmo processo que treinou o modelo.
Ela vai junto com o modelo registrado no MLflow, como a descrição da versão,
para quem for usá-lo, revisá-lo ou trocá-lo por outro.
"""

from dataclasses import dataclass
from datetime import date

import polars as pl

from antifraude_pix.avaliacao.decisoes import NOMES as NOMES_DAS_POLITICAS
from antifraude_pix.avaliacao.decisoes import balanco
from antifraude_pix.avaliacao.justica import por_idade
from antifraude_pix.avaliacao.metricas import ALERTA, por_golpe, resumo
from antifraude_pix.decisao.custo import Custos, decidir
from antifraude_pix.modelos.preparo import ENTRADA, PROTEGIDAS
from antifraude_pix.modelos.treino import LIGHTGBM, UM_LEGITIMO_EM
from antifraude_pix.simulacao.configuracao import Configuracao
from antifraude_pix.simulacao.golpes import NOMES as NOMES_DOS_GOLPES
from antifraude_pix.simulacao.golpes import Golpe
from antifraude_pix.variaveis.catalogo import Grupo


@dataclass(frozen=True, slots=True)
class Treino:
    """O que o treino do modelo registrado viu: até quando, quantos Pix, e quantos golpes o banco já sabia."""

    corte: date
    pix: int
    golpes_sabidos: int
    na_calibracao: int


def ficha(
    config: Configuracao,
    treino: Treino,
    pontuacoes: pl.DataFrame,
    idades: pl.DataFrame,
    *,
    custos: Custos | None = None,
    nome: str = "antifraude-pix",
) -> str:
    """A ficha em Markdown, com as medidas do LightGBM nos meses de teste (`pontuacoes`, de um modelo só)."""
    custos = custos or Custos()
    meses = pontuacoes["corte"].unique().sort().to_list()
    linha = resumo(pontuacoes).row(0, named=True)
    golpes = int(linha["golpes"])
    partes = [
        f"# A ficha do modelo: {nome}",
        "",
        "## O que ele faz",
        "",
        "Para cada Pix, dá a chance de golpe, calibrada, e decide pelo menor custo esperado: aprovar, segurar "
        "para confirmar com o cliente ou bloquear. Cada Pix segurado ou bloqueado sai com as três razões que mais "
        "pesaram, em português (ADRs 0010 e 0012).",
        "",
        "Entram as variáveis de um Pix; saem a pontuação, a probabilidade, a decisão e a explicação.",
        "",
        "## Para que ele não serve",
        "",
        "- **Para um banco de verdade, como está.** Ele aprendeu com um simulador; num banco, o processo é o "
        "mesmo, mas o treino é com os dados do banco.",
        "- **Para julgar o cliente.** Ele olha o Pix, o hábito e o recebedor. Não é um modelo de crédito, "
        "nem de quem a pessoa é.",
        "- **Para decidir sozinho os casos de dúvida.** A confirmação com o cliente e a revisão do analista fazem "
        "parte da decisão.",
        "",
        "## Os dados",
        "",
        f"- A simulação padrão: {_numero(config.clientes)} clientes, {_numero(config.dias)} dias a partir de "
        f"{config.inicio:%d/%m/%Y}, semente {config.semente}. A mesma semente dá os mesmos dados, byte a byte.",
        f"- O treino vai até a véspera de {treino.corte:%d/%m/%Y}: {_numero(treino.pix)} Pix, dos quais "
        f"{_numero(treino.golpes_sabidos)} o banco já sabia que eram golpe, pela contestação (o rótulo do dia, "
        "ADR-0007).",
        f"- A probabilidade é calibrada pelo método de Platt em {_numero(treino.na_calibracao)} Pix maduros, que "
        "o modelo não viu (ADR-0008).",
        "",
        "## A entrada",
        "",
        f"{len(ENTRADA)} variáveis, calculadas só com o que o banco sabia antes de cada Pix (ADR-0006):",
        "",
        *_grupos(),
        "",
        f"Fica de fora a {', '.join(PROTEGIDAS)} (ADR-0011): o banco tem o dado, mas a decisão não depende dele.",
        "",
        "## O modelo",
        "",
        f"O LightGBM, com {_numero(LIGHTGBM['n_estimators'])} árvores de {LIGHTGBM['num_leaves']} folhas e taxa de "
        f"aprendizado de {_decimal(LIGHTGBM['learning_rate'], 2)}. Aprende com todos os golpes e um Pix legítimo "
        f"em cada {UM_LEGITIMO_EM}, com peso {UM_LEGITIMO_EM}.",
        "",
        "## A medida, mês a mês",
        "",
        f"O mesmo processo, refeito em cada mês de teste: o modelo treinado com o que o banco sabia na véspera, e "
        f"medido contra a verdade do simulador. De {meses[0]:%m/%Y} a {meses[-1]:%m/%Y}, com "
        f"{_numero(pontuacoes.height)} Pix e {_numero(golpes)} golpes:",
        "",
        "| Precisão média | Golpes pegos no alerta | Dinheiro salvo no alerta | Golpes previstos |",
        "|--:|--:|--:|--:|",
        f"| {_decimal(linha['precisao_media'], 3)} | {_porcentagem(linha['recall_no_alerta'])} | "
        f"{_porcentagem(linha['recall_em_reais'])} | {_numero(round(linha['golpes_previstos']))} de "
        f"{_numero(golpes)} |",
        "",
        f"O alerta são os {_porcentagem(ALERTA, 1)} de Pix de cada mês com mais cara de golpe (ADR-0009).",
        "",
        "| Golpe | Golpes | Precisão média | Pegos no alerta |",
        "|---|--:|--:|--:|",
        *(
            f"| {NOMES_DOS_GOLPES[Golpe(g['golpe'])]} | {_numero(g['golpes'])} | {_decimal(g['precisao_media'], 3)} "
            f"| {_porcentagem(g['recall_no_alerta'])} |"
            for g in por_golpe(pontuacoes).iter_rows(named=True)
        ),
        "",
        "## A decisão",
        "",
        f"Confirmar custa {_reais(custos.confirmar)}, bloquear um cliente honesto custa "
        f"{_reais(custos.bloquear_um_honesto)}, e a confirmação para {_porcentagem(custos.confirmacao_para)} "
        "dos golpes. São suposições, escritas na ADR-0010.",
        "",
        "| | Custo | Golpes parados | Honestos segurados | Honestos bloqueados |",
        "|---|--:|--:|--:|--:|",
        *(
            f"| {NOMES_DAS_POLITICAS[p['politica']]} | {_reais(p['custo'])} | {_numero(round(p['golpes_parados']))} "
            f"| {_numero(p['honestos_segurados'])} | {_numero(p['honestos_bloqueados'])} |"
            for p in balanco(pontuacoes, custos).iter_rows(named=True)
        ),
        "",
        "## Os erros por idade",
        "",
        "Pela decisão do custo, em cada faixa de idade (ADR-0011):",
        "",
        "| Idade | Golpes pegos | Honestos segurados a cada 10 mil | Honestos bloqueados a cada 10 mil |",
        "|---|--:|--:|--:|",
        *(
            f"| {f['faixa']} | {_porcentagem(f['recall'])} | {_decimal(f['honestos_segurados'], 1)} "
            f"| {_decimal(f['honestos_bloqueados'], 2)} |"
            for f in por_idade(decidir(pontuacoes, custos).join(idades, on="id")).iter_rows(named=True)
        ),
        "",
        "## Onde ele erra",
        "",
        "- **O golpe que o banco demora a saber.** A venda falsa é contestada em semanas, e o modelo aprende "
        "menos com ela (ADR-0007).",
        "- **O primeiro golpe para uma laranja nova.** O recebedor já contestado é a razão que mais pesa; antes "
        "da primeira contestação, sobra o hábito do cliente (ADR-0012).",
        "- **As pontas da probabilidade.** Acima de 10%, o modelo é tímido: foi golpe mais vezes do que ele "
        "previu, e ele bloqueia um pouco menos do que devia (ADR-0008).",
        "- **O cliente honesto de 65 anos ou mais** é segurado para confirmar mais vezes que os outros, porque o "
        "golpe contra ele é mais comum (ADR-0011).",
        "- **Os números são de um simulador**, e um banco de verdade fica abaixo deles.",
        "",
        "## Como usar",
        "",
        "```python",
        "import mlflow",
        "",
        f'modelo = mlflow.pyfunc.load_model("models:/{nome}@campeao")',
        "decisoes = modelo.predict(variaveis)  # pontuacao, probabilidade, acao e explicacao",
        "```",
        "",
    ]

    return "\n".join(partes)


def _numero(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")


def _decimal(valor: float, casas: int) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def _porcentagem(fracao: float, casas: int = 0) -> str:
    return f"{fracao:.{casas}%}".replace(".", ",")


def _reais(valor: float) -> str:
    return f"R$ {_numero(round(valor))}"


def _grupos() -> list[str]:
    """Uma linha por grupo do catálogo, com as variáveis que o modelo recebe; a última termina em ponto."""
    linhas = [
        f"- **{grupo.value}**: {', '.join(v.nome for v in ENTRADA if v.grupo is grupo)};"
        for grupo in Grupo
        if any(v.grupo is grupo for v in ENTRADA)
    ]

    return [*linhas[:-1], linhas[-1].removesuffix(";") + "."]
