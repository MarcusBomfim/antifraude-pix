"""A linha de comando do projeto.

- `simular` gera os dados, com ou sem a virada da falsa central;
- `validar` os confere;
- `variaveis` calcula as variáveis de cada Pix;
- `avaliar` treina os modelos mês a mês com o rótulo do dia, e os mede contra a verdade;
- `decidir` decide cada Pix pelo custo, explica cada decisão em português, e mede o custo e os erros por idade;
- `experimentos` refaz no MLflow as medidas que decidiram o desenho dos modelos;
- `registrar` treina o modelo final, ou o de um corte, e o registra no MLflow, com a ficha, como o campeão;
- `cartoes` roda o mesmo processo na base pública de cartões da ULB;
- `servir` põe de pé a API que decide cada Pix em tempo real, com o campeão do registro;
- `latencia` mede quanto o serviço leva para decidir cada Pix, por dentro e pela API;
- `monitorar` acompanha o campeão semana a semana: a deriva, as saídas e a perda que o rótulo mostra;
- `retreinar` treina um desafiante, o compara com o campeão nos mesmos Pix, e o promove se ele ganhar.
"""

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import uvicorn
from mlflow.exceptions import MlflowException
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
from antifraude_pix.modelos.rotulo import Dobra, dobras, janela_de_teste, primeiro_corte
from antifraude_pix.modelos.treino import NOMES as NOMES_DOS_MODELOS
from antifraude_pix.monitoramento.deriva import MUDOU, nivel
from antifraude_pix.monitoramento.retreino import JANELA, Veredito, comparar
from antifraude_pix.monitoramento.rotulo import ALARME, conhecidos
from antifraude_pix.monitoramento.semanas import REFERENCIA, semana_a_semana
from antifraude_pix.rastreio.experimentos import PASTA as PASTA_DO_RASTREIO
from antifraude_pix.rastreio.experimentos import VARIANTES, rastrear
from antifraude_pix.rastreio.registro import CAMPEAO, DESAFIANTE, NOME, Carregado, carregar, dar_o_veredito, registrar
from antifraude_pix.simulacao.configuracao import Configuracao
from antifraude_pix.simulacao.golpes import NOMES, Golpe
from antifraude_pix.simulacao.simulador import Simulacao, simular
from antifraude_pix.tempo_real.estado import Estado
from antifraude_pix.tempo_real.latencia import Evento, eventos, medir_pela_api, medir_por_dentro, resumir
from antifraude_pix.tempo_real.servico import carregar_o_campeao, criar
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
        "registrar": lambda: _registrar(pedido.pasta, pedido.ficha, pedido.corte),
        "cartoes": lambda: _cartoes(pedido.pasta),
        "servir": lambda: _servir(pedido.pasta, pedido.porta),
        "latencia": lambda: _latencia(pedido.pasta, pedido.pix),
        "monitorar": lambda: _monitorar(pedido.pasta, pedido.ate, pedido.saida),
        "retreinar": lambda: _retreinar(pedido.pasta, pedido.ate),
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
    simular_.add_argument(
        "--virada",
        type=date.fromisoformat,
        help="o dia em que a falsa central passa a pedir um Pix por dia, AAAA-MM-DD (padrão: sem virada)",
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
    registrar_.add_argument(
        "--corte",
        type=date.fromisoformat,
        help="treina com o que o banco sabia na meia-noite deste dia, AAAA-MM-DD (padrão: tudo o que os dados têm)",
    )

    cartoes = comandos.add_parser(
        "cartoes", help="roda o mesmo processo na base pública de cartões da ULB, baixada do OpenML se faltar"
    )
    cartoes.add_argument("pasta", type=Path, nargs="?", default=Path("dados"), help="a pasta dos dados (padrão: dados)")

    servir = comandos.add_parser(
        "servir", help="põe de pé a API que decide cada Pix em tempo real, com o campeão do registro do MLflow"
    )
    servir.add_argument("pasta", type=Path, nargs="?", default=Path("dados"), help="a pasta dos dados (padrão: dados)")
    servir.add_argument("--porta", type=int, default=8000, help="a porta da API (padrão: %(default)s)")

    latencia = comandos.add_parser(
        "latencia", help="mede quanto o serviço leva para decidir cada Pix do último mês, por dentro e pela API"
    )
    latencia.add_argument(
        "pasta", type=Path, nargs="?", default=Path("dados"), help="a pasta dos dados (padrão: dados)"
    )
    latencia.add_argument("--pix", type=int, default=5_000, help="quantos Pix em cada medida (padrão: %(default)s)")

    ajuda_do_ate = "o dia de hoje, AAAA-MM-DD: vale o que o banco sabia na meia-noite dele (padrão: o fim dos dados)"
    monitorar = comandos.add_parser(
        "monitorar", help="acompanha o campeão semana a semana: a deriva, as saídas e a perda que o rótulo mostra"
    )
    monitorar.add_argument(
        "pasta", type=Path, nargs="?", default=Path("dados"), help="a pasta dos dados (padrão: dados)"
    )
    monitorar.add_argument("--ate", type=date.fromisoformat, help=ajuda_do_ate)
    monitorar.add_argument(
        "--saida", type=Path, help="o arquivo das semanas (padrão: monitoramento.parquet, na pasta dos dados)"
    )

    retreinar = comandos.add_parser(
        "retreinar", help="treina um desafiante, o compara com o campeão nos mesmos Pix e o promove se ele ganhar"
    )
    retreinar.add_argument(
        "pasta", type=Path, nargs="?", default=Path("dados"), help="a pasta dos dados (padrão: dados)"
    )
    retreinar.add_argument("--ate", type=date.fromisoformat, help=ajuda_do_ate)

    return leitor


def _simular(pedido: argparse.Namespace) -> int:
    try:
        config = Configuracao(
            clientes=pedido.clientes,
            dias=pedido.dias,
            inicio=pedido.inicio,
            semente=pedido.semente,
            golpes_por_mil_clientes_ao_mes=pedido.golpes_por_mil,
            virada=pedido.virada,
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


def _registrar(pasta: Path, destino_da_ficha: Path | None, corte: date | None) -> int:
    """Mede o LightGBM no tempo, treina o final com o que o banco sabia no corte, e o registra com a ficha (ADR-0014).

    Com o corte, a ficha mostra só os meses de teste que terminam até ele: o modelo não sabe do que vem depois.
    """
    try:
        simulacao = ler(pasta)
    except (OSError, ValueError, KeyError) as erro:
        print(f"antifraude-pix registrar: não consegui ler {pasta}: {erro}", file=sys.stderr)
        return 1

    comeco = time.perf_counter()
    visao = VisaoDoBanco.da(simulacao)
    as_dobras = _ate_o_corte(_as_dobras(visao), corte)

    if not as_dobras:
        _sem_mes_de_teste("registrar", pasta, corte)
        return 1

    variaveis = calcular(visao)
    pontuacoes = avaliar(simulacao, variaveis, as_dobras, modelos=("lightgbm",))
    registrado = registrar(simulacao, variaveis, pontuacoes, pasta, corte=corte)
    destino = destino_da_ficha or pasta / "ficha-do-modelo.md"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(registrado.ficha, encoding="utf-8")

    print(f"O modelo {NOME}, versão {registrado.versao}, registrado no MLflow com o apelido {CAMPEAO}.")

    if corte is not None:
        print(f"Treinado com o que o banco sabia na meia-noite de {corte:%d/%m/%Y}.")

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


def _servir(pasta: Path, porta: int) -> int:
    """Sobe o estado com toda a história do banco, carrega o campeão e serve a API, até alguém parar (ADR-0017)."""
    try:
        visao = ler_o_banco(pasta)
        decisor, versao = carregar_o_campeao(pasta)
    except (OSError, ValueError, KeyError, TypeError, MlflowException) as erro:
        print(f"antifraude-pix servir: não consegui subir o serviço com {pasta}: {erro}", file=sys.stderr)
        return 1

    comeco = time.perf_counter()
    estado = Estado.da_historia(visao)
    print(
        f"O estado do banco subiu com {_numero(estado.pix)} Pix em {_numero(round(time.perf_counter() - comeco))} s. "
        f"Quem decide é o {NOME}, versão {versao}."
    )
    print(f"A API está em http://127.0.0.1:{porta}, e a documentação, em http://127.0.0.1:{porta}/docs.")
    uvicorn.run(criar(estado, decisor, versao), host="127.0.0.1", port=porta, log_level="warning")

    return 0


def _latencia(pasta: Path, quantos: int) -> int:
    """Mede o serviço no último mês dos dados: `quantos` Pix por dentro, e os seguintes pela API (ADR-0018)."""
    try:
        visao = ler_o_banco(pasta)
        decisor, versao = carregar_o_campeao(pasta)
    except (OSError, ValueError, KeyError, TypeError, MlflowException) as erro:
        print(f"antifraude-pix latencia: não consegui medir com {pasta}: {erro}", file=sys.stderr)
        return 1

    ultimo = visao.transacoes["momento"].max()

    if not isinstance(ultimo, datetime):
        print(f"antifraude-pix latencia: os dados de {pasta} não têm nenhum Pix.", file=sys.stderr)
        return 1

    inicio = ultimo.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    comeco = time.perf_counter()
    estado = Estado.da_historia(visao.ate(inicio.replace(tzinfo=None)))
    subida, na_subida = time.perf_counter() - comeco, estado.pix
    do_mes = eventos(visao, inicio, ultimo + timedelta(seconds=1))
    corte = _depois_de(do_mes, quantos)
    por_dentro = medir_por_dentro(estado, decisor, do_mes[:corte])
    pela_api = medir_pela_api(
        criar(estado, decisor, versao), do_mes[corte : corte + _depois_de(do_mes[corte:], quantos)]
    )
    nao_aprovados = por_dentro.filter(pl.col("acao") != "aprovar")
    medidas = {
        "as variáveis": por_dentro["variaveis"],
        "a decisão": por_dentro["decisao"],
        "a decisão do Pix segurado ou bloqueado": nao_aprovados["decisao"],
        "dentro do serviço": pela_api["servico"],
        "a ida e volta pela API": pela_api["ida_e_volta"],
    }
    linhas = [
        f"O estado do banco subiu com {_numero(na_subida)} Pix em {_numero(round(subida))} s. "
        f"Quem decide é o {NOME}, versão {versao}.",
        f"Os Pix de {inicio:%d/%m/%Y} em diante, na ordem do tempo, em milissegundos:",
        "",
        _linha(
            ("", 40, "<"),
            ("Pix", 8, ">"),
            ("mediana", 10, ">"),
            ("p95", 10, ">"),
            ("p99", 10, ">"),
            ("máximo", 10, ">"),
        ),
    ]

    for linha in resumir(medidas).iter_rows(named=True):
        linhas.append(
            _linha(
                (linha["medida"], 40, "<"),
                (_numero(linha["pix"]), 8, ">"),
                (_decimal(linha["p50"]), 10, ">"),
                (_decimal(linha["p95"]), 10, ">"),
                (_decimal(linha["p99"]), 10, ">"),
                (_decimal(linha["maximo"]), 10, ">"),
            )
        )

    print("\n".join(linhas))

    return 0


def _depois_de(os_eventos: list[Evento], quantos: int) -> int:
    """A posição logo depois do `quantos`-ésimo Pix: os eventos até ali levam esse número de Pix."""
    vistos = 0

    for posicao, evento in enumerate(os_eventos):
        vistos += evento.tipo == "pix"

        if vistos == quantos:
            return posicao + 1

    return len(os_eventos)


def _as_dobras(visao: VisaoDoBanco) -> tuple[Dobra, ...]:
    ultimo = visao.transacoes["momento"].max()

    return dobras(primeiro_corte(visao), ultimo.date()) if isinstance(ultimo, datetime) else ()


def _ate_o_corte(as_dobras: tuple[Dobra, ...], corte: date | None) -> tuple[Dobra, ...]:
    """Os meses de teste que terminam até o corte: os que o modelo treinado nele poderia ter visto medidos."""
    return as_dobras if corte is None else tuple(dobra for dobra in as_dobras if dobra.fim <= corte)


def _o_fim(visao: VisaoDoBanco) -> date:
    """O dia seguinte ao último Pix: o monitor e o retreino olham, por padrão, tudo o que os dados têm."""
    ultimo = visao.transacoes["momento"].max()

    if not isinstance(ultimo, datetime):
        raise ValueError("Os dados não têm nenhum Pix.")

    return ultimo.date() + timedelta(days=1)


def _sem_mes_de_teste(comando: str, pasta: Path, corte: date | None = None) -> None:
    ate = "" if corte is None else f" antes de {corte:%d/%m/%Y}"
    print(
        f"antifraude-pix {comando}: os dados de {pasta} não chegam a um mês de teste{ate}. Antes dele, são "
        "precisos o aquecimento, um mês de treino e um mês de validação.",
        file=sys.stderr,
    )


def _monitorar(pasta: Path, ate: date | None, destino: Path | None) -> int:
    """Acompanha o campeão desde o corte dele, semana a semana, com o que o banco sabia em `ate` (ADR-0020)."""
    try:
        simulacao = ler(pasta)
        campeao = carregar(pasta)
    except (OSError, ValueError, KeyError, MlflowException) as erro:
        print(f"antifraude-pix monitorar: não consegui ler os dados e o campeão de {pasta}: {erro}", file=sys.stderr)
        return 1

    comeco = time.perf_counter()
    visao = VisaoDoBanco.da(simulacao)
    ate = ate or _o_fim(visao)

    try:
        semanas = semana_a_semana(visao, calcular(visao), campeao.modelo, campeao.custos, corte=campeao.corte, ate=ate)
    except ValueError as erro:
        print(f"antifraude-pix monitorar: {erro}", file=sys.stderr)
        return 1

    destino = destino or pasta / "monitoramento.parquet"
    destino.parent.mkdir(parents=True, exist_ok=True)
    semanas.write_parquet(destino)
    print(_quadro_do_monitor(semanas, campeao, ate))
    print("")

    if semanas["razao_acumulada"][-1] >= ALARME:
        print(
            "Alarme: o golpe passa mais do que o campeão espera, e o rótulo mostra o que a deriva não mostra. "
            f"Hora de treinar um desafiante: antifraude-pix retreinar {pasta} --ate {ate:%Y-%m-%d}."
        )
    else:
        print(f"Dentro do esperado: abaixo de {_decimal(ALARME, casas=0)} vezes, o golpe passa como o campeão espera.")

    print(f"Semanas gravadas em {destino}, em {_numero(round(time.perf_counter() - comeco))} s.")

    return 0


def _quadro_do_monitor(semanas: pl.DataFrame, campeao: Carregado, ate: date) -> str:
    """O monitor em texto: uma linha por semana, e o que ele diz no fim."""
    primeira, ultima = semanas["semana"][0], semanas["semana"][-1] + timedelta(days=6)
    larguras = (
        ("semana", 12, "<"),
        ("Pix", 9, ">"),
        ("segurados", 11, ">"),
        ("bloqueados", 12, ">"),
        ("PSI prob.", 11, ">"),
        ("perda esperada", 16, ">"),
        ("perda conhecida", 17, ">"),
        ("razão", 8, ">"),
    )
    linhas = [
        f"O campeão é o {NOME}, versão {campeao.versao}, treinado com o que o banco sabia em {campeao.corte:%d/%m/%Y}.",
        f"Cada semana inteira de {primeira:%d/%m/%Y} a {ultima:%d/%m/%Y}, contra as {REFERENCIA // 7} semanas antes "
        f"do corte, com o rótulo que o banco tinha em {ate:%d/%m/%Y}:",
        "",
        _linha(*larguras) + "  variável que mais mudou (PSI)",
    ]

    for linha in semanas.iter_rows(named=True):
        colunas = (
            (f"{linha['semana']:%d/%m/%Y}", 12, "<"),
            (_numero(linha["pix"]), 9, ">"),
            (_porcentagem(linha["parte_segurada"]), 11, ">"),
            (_porcentagem(linha["parte_bloqueada"]), 12, ">"),
            (_decimal(linha["psi_da_probabilidade"], casas=3), 11, ">"),
            (_reais(linha["esperada"]), 16, ">"),
            (_reais(linha["conhecida"]), 17, ">"),
            ("—" if linha["razao"] is None else _decimal(linha["razao"]), 8, ">"),
        )
        linhas.append(f"{_linha(*colunas)}  {linha['variavel_que_mais_mudou']} ({_decimal(linha['psi_dela'])})")

    maior = float(semanas["psi_da_probabilidade"].max())  # type: ignore[arg-type]
    mudaram = semanas.filter(pl.col("psi_dela") >= MUDOU)["variavel_que_mais_mudou"].unique(maintain_order=True)
    linhas += [
        "",
        f"A probabilidade do campeão: {nivel(maior)} em todas as semanas, com o PSI de no máximo "
        f"{_decimal(maior, casas=3)}."
        if nivel(maior) == "estável"
        else f"A probabilidade do campeão mudou: o PSI chegou a {_decimal(maior, casas=3)}.",
    ]

    if mudaram.len():
        linhas.append(
            f"As variáveis que mudaram, com o PSI acima de {_decimal(MUDOU)}: {', '.join(mudaram.to_list())}."
        )

    linhas.append(
        f"Desde o corte, a perda conhecida nos Pix aprovados é {_decimal(semanas['razao_acumulada'][-1])} "
        "vezes a que o campeão esperava."
    )

    return "\n".join(linhas)


def _retreinar(pasta: Path, ate: date | None) -> int:
    """Treina o desafiante, o compara com o campeão nos mesmos Pix, e passa a ele o apelido se ele ganhar (ADR-0021)."""
    try:
        simulacao = ler(pasta)
        campeao = carregar(pasta)
    except (OSError, ValueError, KeyError, MlflowException) as erro:
        print(f"antifraude-pix retreinar: não consegui ler os dados e o campeão de {pasta}: {erro}", file=sys.stderr)
        return 1

    comeco = time.perf_counter()
    visao = VisaoDoBanco.da(simulacao)
    ate = ate or _o_fim(visao)
    corte = ate - timedelta(days=JANELA)

    if corte <= campeao.corte:
        print(
            f"antifraude-pix retreinar: o campeão foi treinado em {campeao.corte:%d/%m/%Y}, e o desafiante seria "
            f"treinado em {corte:%d/%m/%Y}, {JANELA} dias antes de {ate:%d/%m/%Y}: ele não saberia nada a mais.",
            file=sys.stderr,
        )
        return 1

    as_dobras = _ate_o_corte(_as_dobras(visao), corte)

    if not as_dobras:
        _sem_mes_de_teste("retreinar", pasta, corte)
        return 1

    variaveis = calcular(visao)
    pontuacoes = avaliar(simulacao, variaveis, as_dobras, modelos=("lightgbm",))
    registrar(simulacao, variaveis, pontuacoes, pasta, corte=corte, apelido=DESAFIANTE)
    desafiante = carregar(pasta, DESAFIANTE)
    pix = janela_de_teste(visao, variaveis, Dobra(corte, ate))
    rotulo = conhecidos(visao, ate)

    def pontuacoes_de(modelo: Carregado) -> pl.DataFrame:
        probabilidade = modelo.modelo.probabilidade_da(modelo.modelo.pontuar(pix))

        return pix.select("id", "valor", pl.Series("probabilidade", probabilidade)).join(rotulo, on="id", how="left")

    try:
        veredito = comparar(
            pontuacoes_de(campeao), pontuacoes_de(desafiante), variaveis.select("id", "idade"), campeao.custos
        )
    except ValueError as erro:
        print(f"antifraude-pix retreinar: {erro}", file=sys.stderr)
        return 1

    dar_o_veredito(pasta, desafiante.versao, promovido=veredito.promover)
    print(_quadro_do_retreino(veredito, campeao, desafiante, pix.height, ate))
    print(f"Em {_numero(round(time.perf_counter() - comeco))} s.")

    return 0


def _quadro_do_retreino(veredito: Veredito, campeao: Carregado, desafiante: Carregado, pix: int, ate: date) -> str:
    """A comparação em texto: as medidas lado a lado, e o veredito."""
    um, outro = veredito.campeao, veredito.desafiante
    linhas = [
        f"O campeão: {NOME}, versão {campeao.versao}, treinado com o que o banco sabia em {campeao.corte:%d/%m/%Y}.",
        f"O desafiante: versão {desafiante.versao}, treinado com o que o banco sabia em "
        f"{desafiante.corte:%d/%m/%Y}, registrado com o apelido {DESAFIANTE}.",
        f"Os dois decidiram os mesmos {_numero(pix)} Pix, de {desafiante.corte:%d/%m/%Y} a "
        f"{ate - timedelta(days=1):%d/%m/%Y}, medidos com o rótulo que o banco tinha em {ate:%d/%m/%Y}: "
        f"{_numero(um.golpes)} contestados.",
        "",
        _linha(("", 30, "<"), ("campeão", 14, ">"), ("desafiante", 14, ">")),
    ]

    for nome, deste, daquele in (
        ("precisão média", _decimal(um.precisao_media, casas=3), _decimal(outro.precisao_media, casas=3)),
        ("custo das decisões", _reais(um.custo), _reais(outro.custo)),
        (
            "contestados pegos",
            f"{_numero(um.golpes_pegos)} de {_numero(um.golpes)}",
            f"{_numero(outro.golpes_pegos)} de {_numero(outro.golpes)}",
        ),
        (
            "honestos de 65+ bloqueados",
            _numero(um.idosos_honestos_bloqueados),
            _numero(outro.idosos_honestos_bloqueados),
        ),
    ):
        linhas.append(_linha((nome, 30, "<"), (deste, 14, ">"), (daquele, 14, ">")))

    linhas.append("")

    if veredito.promover:
        linhas.append(
            f"O desafiante ganhou: a versão {desafiante.versao} é o novo {CAMPEAO}, e a API passa a carregá-la."
        )
    else:
        linhas.append(f"O campeão continua, e a versão {desafiante.versao} fica como {DESAFIANTE}. Ela não passou:")
        linhas.extend(f"- {razao};" if razao != veredito.razoes[-1] else f"- {razao}." for razao in veredito.razoes)

    return "\n".join(linhas)


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
