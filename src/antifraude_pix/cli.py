"""A linha de comando do projeto.

- `simular` gera os dados;
- `validar` os confere;
- `variaveis` calcula as variáveis de cada Pix;
- `avaliar` treina os modelos mês a mês com o rótulo do dia, e os mede contra a verdade;
- `decidir` decide cada Pix pelo custo, explica cada decisão em português, e mede o custo e os erros por idade;
- `experimentos` refaz no MLflow as medidas que decidiram o desenho dos modelos;
- `registrar` treina o modelo final e o registra no MLflow, com a ficha, como o campeão;
- `cartoes` roda o mesmo processo na base pública de cartões da ULB.
"""

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
from mlflow.tracking import MlflowClient

from antifraude_pix.avaliacao.decisoes import NOMES as NOMES_DAS_POLITICAS
from antifraude_pix.avaliacao.decisoes import balanco
from antifraude_pix.avaliacao.justica import por_idade
from antifraude_pix.avaliacao.metricas import ALERTA, resumo
from antifraude_pix.avaliacao.no_tempo import avaliar, com_a_verdade, treinados
from antifraude_pix.cartoes import base as base_dos_cartoes
from antifraude_pix.cartoes.processo import NOMES as NOMES_NOS_CARTOES
from antifraude_pix.cartoes.processo import avaliar as avaliar_os_cartoes
from antifraude_pix.cartoes.rastreio import rastrear as rastrear_os_cartoes
from antifraude_pix.dados.banco import VisaoDoBanco
from antifraude_pix.dados.gravacao import gravar, ler, ler_o_banco
from antifraude_pix.dados.validacao import problemas
from antifraude_pix.decisao.custo import Acao, decidir
from antifraude_pix.decisao.explicacao import explicar
from antifraude_pix.modelos.rotulo import Dobra, dobras, primeiro_corte
from antifraude_pix.modelos.treino import NOMES as NOMES_DOS_MODELOS
from antifraude_pix.rastreio.experimentos import PASTA as PASTA_DO_RASTREIO
from antifraude_pix.rastreio.experimentos import VARIANTES, rastrear
from antifraude_pix.rastreio.registro import CAMPEAO, NOME, registrar
from antifraude_pix.simulacao.configuracao import Configuracao
from antifraude_pix.simulacao.golpes import NOMES, Golpe
from antifraude_pix.simulacao.simulador import Simulacao, simular
from antifraude_pix.variaveis.calculo import calcular
from antifraude_pix.variaveis.catalogo import CATALOGO


def principal(argumentos: Sequence[str] | None = None) -> int:
    """O comando `antifraude-pix`.

    Devolve 0 quando dá certo, 1 quando os dados têm problema, e 2 com o pedido errado.
    """
    pedido = _leitor().parse_args(argumentos)
    comandos: dict[str, Callable[[], int]] = {
        "simular": lambda: _simular(pedido),
        "validar": lambda: _validar(pedido.pasta),
        "variaveis": lambda: _variaveis(pedido.pasta, pedido.saida),
        "avaliar": lambda: _avaliar(pedido.pasta, pedido.saida),
        "decidir": lambda: _decidir(pedido.pasta, pedido.saida),
        "experimentos": lambda: _experimentos(pedido.pasta),
        "registrar": lambda: _registrar(pedido.pasta, pedido.ficha),
        "cartoes": lambda: _cartoes(pedido.pasta),
    }

    return comandos[pedido.comando]()


def _leitor() -> argparse.ArgumentParser:
    padrao = Configuracao()
    leitor = argparse.ArgumentParser(prog="antifraude-pix", description="Fraude em Pix: o simulador e os modelos.")
    comandos = leitor.add_subparsers(dest="comando", required=True)

    simular_ = comandos.add_parser("simular", help="gera os dados de uma simulação e os grava em Parquet")
    simular_.add_argument(
        "--clientes", type=int, default=padrao.clientes, help="quantos clientes (padrão: %(default)s)"
    )
    simular_.add_argument("--dias", type=int, default=padrao.dias, help="quantos dias (padrão: %(default)s)")
    simular_.add_argument(
        "--inicio",
        type=date.fromisoformat,
        default=padrao.inicio,
        help="o primeiro dia, AAAA-MM-DD (padrão: %(default)s)",
    )
    simular_.add_argument(
        "--semente", type=int, default=padrao.semente, help="a semente do sorteio (padrão: %(default)s)"
    )
    simular_.add_argument(
        "--golpes-por-mil",
        type=float,
        default=padrao.golpes_por_mil_clientes_ao_mes,
        help="ataques por mês a cada mil clientes (padrão: %(default)s)",
    )
    simular_.add_argument("--saida", type=Path, default=Path("dados"), help="a pasta dos dados (padrão: %(default)s)")

    validar = comandos.add_parser("validar", help="confere os dados gravados numa pasta")
    validar.add_argument("pasta", type=Path, nargs="?", default=Path("dados"), help="a pasta dos dados (padrão: dados)")

    variaveis = comandos.add_parser(
        "variaveis", help="calcula as variáveis de cada Pix, só com o que o banco sabia antes dele"
    )
    variaveis.add_argument(
        "pasta", type=Path, nargs="?", default=Path("dados"), help="a pasta dos dados (padrão: dados)"
    )
    variaveis.add_argument(
        "--saida", type=Path, help="o arquivo das variáveis (padrão: variaveis.parquet, na pasta dos dados)"
    )

    avaliar_ = comandos.add_parser(
        "avaliar", help="treina os modelos mês a mês com o rótulo do dia, e os mede contra a verdade do simulador"
    )
    avaliar_.add_argument(
        "pasta", type=Path, nargs="?", default=Path("dados"), help="a pasta dos dados (padrão: dados)"
    )
    avaliar_.add_argument(
        "--saida", type=Path, help="o arquivo das pontuações (padrão: avaliacao.parquet, na pasta dos dados)"
    )

    decidir_ = comandos.add_parser(
        "decidir",
        help="decide cada Pix de teste pelo custo, explica cada decisão, e mede o custo e os erros por idade",
    )
    decidir_.add_argument(
        "pasta", type=Path, nargs="?", default=Path("dados"), help="a pasta dos dados (padrão: dados)"
    )
    decidir_.add_argument(
        "--saida", type=Path, help="o arquivo das decisões (padrão: decisoes.parquet, na pasta dos dados)"
    )

    experimentos = comandos.add_parser(
        "experimentos", help="refaz no MLflow as medidas que decidiram o desenho dos modelos, uma corrida por modelo"
    )
    experimentos.add_argument(
        "pasta", type=Path, nargs="?", default=Path("dados"), help="a pasta dos dados (padrão: dados)"
    )

    registrar_ = comandos.add_parser(
        "registrar", help="treina o modelo final e o registra no MLflow, com a ficha, como o campeão"
    )
    registrar_.add_argument(
        "pasta", type=Path, nargs="?", default=Path("dados"), help="a pasta dos dados (padrão: dados)"
    )
    registrar_.add_argument(
        "--ficha", type=Path, help="onde gravar a ficha do modelo (padrão: ficha-do-modelo.md, na pasta dos dados)"
    )

    cartoes = comandos.add_parser(
        "cartoes", help="roda o mesmo processo na base pública de cartões da ULB, baixada do OpenML se faltar"
    )
    cartoes.add_argument("pasta", type=Path, nargs="?", default=Path("dados"), help="a pasta dos dados (padrão: dados)")

    return leitor


def _simular(pedido: argparse.Namespace) -> int:
    try:
        config = Configuracao(
            clientes=pedido.clientes,
            dias=pedido.dias,
            inicio=pedido.inicio,
            semente=pedido.semente,
            golpes_por_mil_clientes_ao_mes=pedido.golpes_por_mil,
        )
    except ValueError as erro:
        print(f"antifraude-pix simular: {erro}", file=sys.stderr)
        return 2

    comeco = time.perf_counter()
    simulacao = simular(config)
    encontrados = problemas(simulacao)

    if encontrados:
        _mostrar_problemas(encontrados)
        return 1

    gravar(simulacao, pedido.saida)
    print(_resumo(simulacao))
    print(f"Gravado em {pedido.saida}, em {_numero(round(time.perf_counter() - comeco))} s.")

    return 0


def _validar(pasta: Path) -> int:
    try:
        simulacao = ler(pasta)
    except (OSError, ValueError, KeyError) as erro:
        print(f"antifraude-pix validar: não consegui ler {pasta}: {erro}", file=sys.stderr)
        return 1

    encontrados = problemas(simulacao)

    if encontrados:
        _mostrar_problemas(encontrados)
        return 1

    print(_resumo(simulacao))
    print("Os dados seguem o contrato.")

    return 0


def _variaveis(pasta: Path, saida: Path | None) -> int:
    """Lê só o que o banco vê, calcula as variáveis e as grava em Parquet, ao lado dos dados."""
    try:
        visao = ler_o_banco(pasta)
    except (OSError, ValueError, KeyError) as erro:
        print(f"antifraude-pix variaveis: não consegui ler {pasta}: {erro}", file=sys.stderr)
        return 1

    comeco = time.perf_counter()
    destino = saida or pasta / "variaveis.parquet"
    destino.parent.mkdir(parents=True, exist_ok=True)
    calcular(visao).write_parquet(destino, compression="zstd")

    print(
        f"{_numero(len(CATALOGO))} variáveis de {_numero(visao.transacoes.height)} Pix, "
        f"gravadas em {destino}, em {_numero(round(time.perf_counter() - comeco))} s."
    )

    return 0


def _avaliar(pasta: Path, saida: Path | None) -> int:
    """Treina em cada corte com o que o banco sabia, pontua o mês do corte, e mede contra a verdade (ADR-0009)."""
    try:
        simulacao = ler(pasta)
    except (OSError, ValueError, KeyError) as erro:
        print(f"antifraude-pix avaliar: não consegui ler {pasta}: {erro}", file=sys.stderr)
        return 1

    comeco = time.perf_counter()
    visao = VisaoDoBanco.da(simulacao)
    as_dobras = _as_dobras(visao)

    if not as_dobras:
        _sem_mes_de_teste("avaliar", pasta)
        return 1

    pontuacoes = avaliar(simulacao, calcular(visao), as_dobras)
    destino = saida or pasta / "avaliacao.parquet"
    destino.parent.mkdir(parents=True, exist_ok=True)
    pontuacoes.write_parquet(destino, compression="zstd")

    print(_quadro(pontuacoes, as_dobras))
    print(f"Pontuações gravadas em {destino}, em {_numero(round(time.perf_counter() - comeco))} s.")

    return 0


def _decidir(pasta: Path, saida: Path | None) -> int:
    """Decide pelo custo cada Pix dos meses de teste, com o LightGBM de cada mês, e explica cada decisão (ADR-0010).

    Mede contra a verdade o custo de cada jeito de decidir e os erros em cada faixa de idade (ADR-0011).
    """
    try:
        simulacao = ler(pasta)
    except (OSError, ValueError, KeyError) as erro:
        print(f"antifraude-pix decidir: não consegui ler {pasta}: {erro}", file=sys.stderr)
        return 1

    comeco = time.perf_counter()
    visao = VisaoDoBanco.da(simulacao)
    as_dobras = _as_dobras(visao)

    if not as_dobras:
        _sem_mes_de_teste("decidir", pasta)
        return 1

    variaveis = calcular(visao)
    partes = []

    for dobra, teste, modelos in treinados(simulacao, variaveis, as_dobras, modelos=("lightgbm",)):
        modelo = modelos["lightgbm"]
        pontuacao = modelo.pontuar(teste)
        decididos = decidir(
            teste.with_columns(
                pl.Series("pontuacao", pontuacao), pl.Series("probabilidade", modelo.probabilidade_da(pontuacao))
            )
        )
        partes.append(
            explicar(modelo, decididos).select(
                "id",
                pl.lit(dobra.corte).alias("corte"),
                pl.lit("lightgbm").alias("modelo"),
                "pontuacao",
                "probabilidade",
                "valor",
                "idade",
                "acao",
                "explicacao",
            )
        )

    decisoes = com_a_verdade(simulacao, pl.concat(partes))
    destino = saida or pasta / "decisoes.parquet"
    destino.parent.mkdir(parents=True, exist_ok=True)
    decisoes.write_parquet(destino, compression="zstd")

    print(_quadro_das_decisoes(decisoes, as_dobras))
    print(f"Decisões gravadas em {destino}, em {_numero(round(time.perf_counter() - comeco))} s.")

    return 0


def _experimentos(pasta: Path) -> int:
    """Refaz cada variante no tempo e grava as corridas no MLflow, na pasta do rastreio (ADR-0013)."""
    try:
        simulacao = ler(pasta)
    except (OSError, ValueError, KeyError) as erro:
        print(f"antifraude-pix experimentos: não consegui ler {pasta}: {erro}", file=sys.stderr)
        return 1

    comeco = time.perf_counter()
    visao = VisaoDoBanco.da(simulacao)
    as_dobras = _as_dobras(visao)

    if not as_dobras:
        _sem_mes_de_teste("experimentos", pasta)
        return 1

    variaveis = calcular(visao)
    print("As variantes, com a precisão média de cada modelo nos meses de teste:")

    for variante in VARIANTES:
        # A precisão média de cada modelo sai da própria corrida gravada, e não de outra avaliação.
        corridas = [MlflowClient().get_run(id_) for id_ in rastrear(variante, simulacao, variaveis, as_dobras, pasta)]
        precisoes = ", ".join(
            f"{NOMES_DOS_MODELOS[corrida.data.tags['modelo']]} "
            f"{_decimal(corrida.data.metrics['precisao_media'], casas=3)}"
            for corrida in corridas
        )
        print(f"- {variante.nome}: {precisoes}")

    print(
        f"Corridas gravadas no MLflow, em {pasta / PASTA_DO_RASTREIO}, em "
        f"{_numero(round(time.perf_counter() - comeco))} s. Para vê-las:"
    )
    print(f"uv run mlflow ui --backend-store-uri sqlite:///{(pasta / PASTA_DO_RASTREIO / 'mlflow.db').as_posix()}")

    return 0


def _registrar(pasta: Path, destino_da_ficha: Path | None) -> int:
    """Mede o LightGBM no tempo, treina o final com tudo, e o registra no MLflow com a ficha (ADR-0014)."""
    try:
        simulacao = ler(pasta)
    except (OSError, ValueError, KeyError) as erro:
        print(f"antifraude-pix registrar: não consegui ler {pasta}: {erro}", file=sys.stderr)
        return 1

    comeco = time.perf_counter()
    visao = VisaoDoBanco.da(simulacao)
    as_dobras = _as_dobras(visao)

    if not as_dobras:
        _sem_mes_de_teste("registrar", pasta)
        return 1

    variaveis = calcular(visao)
    pontuacoes = avaliar(simulacao, variaveis, as_dobras, modelos=("lightgbm",))
    registrado = registrar(simulacao, variaveis, pontuacoes, pasta)
    destino = destino_da_ficha or pasta / "ficha-do-modelo.md"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(registrado.ficha, encoding="utf-8")

    print(f"O modelo {NOME}, versão {registrado.versao}, registrado no MLflow com o apelido {CAMPEAO}.")
    print(f'Para carregá-lo: mlflow.pyfunc.load_model("models:/{NOME}@{CAMPEAO}").')
    print(f"Ficha gravada em {destino}, em {_numero(round(time.perf_counter() - comeco))} s.")

    return 0


def _cartoes(pasta: Path) -> int:
    """Baixa a base de cartões, se faltar, e roda nela o mesmo processo do Pix (ADR-0015)."""
    pasta_da_base = pasta / "cartoes"
    comeco = time.perf_counter()

    if not (pasta_da_base / base_dos_cartoes.ARQUIVO).exists():
        print(f"Baixando a base de cartões da ULB do OpenML (151 MB) para {pasta_da_base}...")

    try:
        # O endereço e o MD5 vêm do módulo na hora: os testes os trocam por uma base inventada.
        arquivo = base_dos_cartoes.baixar(pasta_da_base, base_dos_cartoes.ENDERECO, base_dos_cartoes.MD5)
        compras = base_dos_cartoes.ler(arquivo)
    except (OSError, ValueError) as erro:
        print(f"antifraude-pix cartoes: não consegui a base de cartões: {erro}", file=sys.stderr)
        return 1

    pontuacoes = avaliar_os_cartoes(compras)
    rastrear_os_cartoes(pontuacoes, pasta)
    um_modelo = pontuacoes.filter(pl.col("modelo") == pontuacoes["modelo"][0])
    fraudes = int(um_modelo["golpe"].is_not_null().sum())
    linhas = [
        f"{_numero(compras.height)} compras com cartão em dois dias, {_numero(int(compras['fraude'].sum()))} "
        "fraudes. O treino é o primeiro dia; o teste, o segundo, com "
        f"{_numero(um_modelo.height)} compras e {_numero(fraudes)} fraudes.",
        f"Segurando os {_porcentagem(ALERTA, casas=1)} de compras com mais cara de fraude:",
        "",
        _linha(
            ("", 24, "<"),
            ("precisão média", 16, ">"),
            ("fraudes pegas", 15, ">"),
            ("dinheiro salvo", 16, ">"),
            ("fraudes previstas", 19, ">"),
            ("custo pelo custo", 18, ">"),
        ),
    ]
    custos = balanco(pontuacoes).filter(pl.col("politica") == "custo")
    custo = dict(zip(custos["modelo"], custos["custo"], strict=True))

    for linha in resumo(pontuacoes).iter_rows(named=True):
        linhas.append(
            _linha(
                (NOMES_NOS_CARTOES[linha["modelo"]], 24, "<"),
                (_decimal(linha["precisao_media"]), 16, ">"),
                (_porcentagem(linha["recall_no_alerta"], casas=0), 15, ">"),
                (_porcentagem(linha["recall_em_reais"], casas=0), 16, ">"),
                (_numero(round(linha["golpes_previstos"])), 19, ">"),
                (f"€ {_numero(round(custo[linha['modelo']]))}", 18, ">"),
            )
        )

    print("\n".join(linhas))
    print(
        f"Corridas gravadas no MLflow, no experimento dos cartões, em {_numero(round(time.perf_counter() - comeco))} s."
    )

    return 0


def _as_dobras(visao: VisaoDoBanco) -> tuple[Dobra, ...]:
    ultimo = visao.transacoes["momento"].max()

    return dobras(primeiro_corte(visao), ultimo.date()) if isinstance(ultimo, datetime) else ()


def _sem_mes_de_teste(comando: str, pasta: Path) -> None:
    print(
        f"antifraude-pix {comando}: os dados de {pasta} não chegam a um mês de teste. Antes dele, são precisos "
        "o aquecimento, um mês de treino e um mês de validação.",
        file=sys.stderr,
    )


def _quadro_das_decisoes(decisoes: pl.DataFrame, as_dobras: tuple[Dobra, ...]) -> str:
    """O resumo das decisões em texto: o custo de cada jeito de decidir, os erros por idade e uma explicação de cada."""
    golpe = pl.col("golpe").is_not_null()
    golpes = decisoes.filter(golpe)
    ultimo_dia = as_dobras[-1].fim - timedelta(days=1)
    meses = "mês" if len(as_dobras) == 1 else "meses"
    acoes = pl.col("acao").cast(pl.String)
    segurados = decisoes.filter(acoes == Acao.SEGURAR.value).height
    bloqueados = decisoes.filter(acoes == Acao.BLOQUEAR.value).height
    linhas = [
        f"Decisões de {as_dobras[0].corte:%d/%m/%Y} a {ultimo_dia:%d/%m/%Y}, em {len(as_dobras)} {meses}: "
        f"{_numero(decisoes.height)} Pix, {_numero(golpes.height)} de golpe, "
        f"que somam {_reais(float(golpes['valor'].sum()))}.",
        "Em cada mês, o LightGBM treinado com o que o banco sabia na véspera; em cada Pix, a saída de menor custo "
        "esperado.",
        f"Pelo custo, o banco segurou {_numero(segurados)} Pix para confirmar "
        f"({_porcentagem(segurados / decisoes.height)}) e bloqueou {_numero(bloqueados)} "
        f"({_porcentagem(bloqueados / decisoes.height)}).",
        "",
        _linha(
            ("", 28, "<"),
            ("custo", 14, ">"),
            ("perdido em golpes", 19, ">"),
            ("golpes parados", 16, ">"),
            ("honestos segurados", 20, ">"),
            ("honestos bloqueados", 21, ">"),
        ),
    ]

    for linha in balanco(decisoes.drop("acao")).iter_rows(named=True):
        linhas.append(
            _linha(
                (NOMES_DAS_POLITICAS[linha["politica"]], 28, "<"),
                (_reais(linha["custo"]), 14, ">"),
                (_reais(linha["perdido"]), 19, ">"),
                (_numero(round(linha["golpes_parados"])), 16, ">"),
                (_numero(linha["honestos_segurados"]), 20, ">"),
                (_numero(linha["honestos_bloqueados"]), 21, ">"),
            )
        )

    linhas += [
        "",
        "Os erros por idade, pelo custo:",
        "",
        _linha(
            ("", 14, "<"),
            ("golpes pegos", 14, ">"),
            ("honestos segurados a cada 10 mil", 34, ">"),
            ("honestos bloqueados a cada 10 mil", 35, ">"),
        ),
    ]

    for linha in por_idade(decisoes).iter_rows(named=True):
        linhas.append(
            _linha(
                (linha["faixa"], 14, "<"),
                (_porcentagem(linha["recall"], casas=0), 14, ">"),
                (_decimal(linha["honestos_segurados"], casas=1), 34, ">"),
                (_decimal(linha["honestos_bloqueados"], casas=2), 35, ">"),
            )
        )

    exemplos = (
        ("Um golpe bloqueado", golpe & (acoes == Acao.BLOQUEAR.value)),
        ("Um golpe segurado", golpe & (acoes == Acao.SEGURAR.value)),
        ("Um Pix honesto segurado", ~golpe & (acoes == Acao.SEGURAR.value)),
    )
    linhas.append("")

    for titulo, filtro in exemplos:
        escolhidos = decisoes.filter(filtro)

        if escolhidos.height:
            linhas.append(f"{titulo}: {escolhidos['explicacao'][0]}")

    return "\n".join(linhas)


def _linha(*colunas: tuple[str, int, str]) -> str:
    return "".join(f"{texto:{lado}{largura}}" for texto, largura, lado in colunas)


def _quadro(pontuacoes: pl.DataFrame, as_dobras: tuple[Dobra, ...]) -> str:
    """O resumo da avaliação em texto: uma linha por modelo."""
    um_modelo = pontuacoes.filter(pl.col("modelo") == pontuacoes["modelo"][0])
    golpes = int(um_modelo["golpe"].is_not_null().sum())
    ultimo_dia = as_dobras[-1].fim - timedelta(days=1)
    meses = "mês" if len(as_dobras) == 1 else "meses"
    cabecalho = "".join(
        f"{titulo:>{largura}}"
        for titulo, largura in (
            ("", 24),
            ("precisão média", 16),
            ("golpes pegos", 14),
            ("dinheiro salvo", 16),
            ("honestos por golpe", 20),
            ("golpes previstos", 18),
        )
    )
    linhas = [
        f"Teste de {as_dobras[0].corte:%d/%m/%Y} a {ultimo_dia:%d/%m/%Y}, em {len(as_dobras)} {meses}: "
        f"{_numero(um_modelo.height)} Pix, {_numero(golpes)} de golpe.",
        "O treino usa só o rótulo do dia; a medida, a verdade do simulador.",
        f"Segurando, em cada mês, os {_porcentagem(ALERTA, casas=1)} de Pix com mais cara de golpe:",
        "",
        cabecalho,
    ]

    for linha in resumo(pontuacoes).iter_rows(named=True):
        honestos = linha["legitimos_por_golpe_pego"]
        colunas = (
            (NOMES_DOS_MODELOS[linha["modelo"]], 24, "<"),
            (_decimal(linha["precisao_media"]), 16, ">"),
            (_porcentagem(linha["recall_no_alerta"], casas=0), 14, ">"),
            (_porcentagem(linha["recall_em_reais"], casas=0), 16, ">"),
            ("—" if honestos is None else _decimal(honestos, casas=1), 20, ">"),
            (_numero(round(linha["golpes_previstos"])), 18, ">"),
        )
        linhas.append("".join(f"{texto:{lado}{largura}}" for texto, largura, lado in colunas))

    return "\n".join(linhas)


def _mostrar_problemas(encontrados: list[str]) -> None:
    print("Os dados têm problemas:", file=sys.stderr)

    for problema in encontrados:
        print(f"- {problema}", file=sys.stderr)


def _resumo(simulacao: Simulacao) -> str:
    config = simulacao.config
    pix = simulacao.transacoes.height
    fraudes = simulacao.verdade.fraudes
    contestadas = simulacao.contestacoes.join(fraudes, on="transacao", how="semi").height
    por_golpe = ", ".join(
        f"{NOMES[Golpe(golpe)]} {_numero(quantos)}"
        for golpe, quantos in fraudes.group_by("golpe").len().sort("golpe").iter_rows()
    )

    return (
        f"{_numero(config.clientes)} clientes, {_numero(config.dias)} dias "
        f"a partir de {config.inicio:%d/%m/%Y}, semente {config.semente}.\n"
        f"{_numero(pix)} Pix; {_numero(fraudes.height)} de golpe ({_porcentagem(fraudes.height / max(pix, 1))}), "
        f"em {_numero(simulacao.verdade.ataques.height)} ataques: {por_golpe}.\n"
        f"{_numero(contestadas)} Pix de golpe contestados ({_porcentagem(contestadas / max(fraudes.height, 1))}), "
        f"e {_numero(simulacao.contestacoes.height - contestadas)} Pix legítimos contestados mesmo assim."
    )


def _numero(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")


def _porcentagem(fracao: float, casas: int = 2) -> str:
    return f"{fracao:.{casas}%}".replace(".", ",")


def _decimal(valor: float, casas: int = 2) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def _reais(valor: float) -> str:
    return f"R$ {_numero(round(valor))}"


if __name__ == "__main__":
    sys.exit(principal())
