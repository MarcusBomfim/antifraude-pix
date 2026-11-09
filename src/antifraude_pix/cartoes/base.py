"""A base pública de cartões: as compras com cartão de clientes europeus, em dois dias de setembro de 2013 (ADR-0015).

São 284.807 compras e 492 fraudes (0,17%), coletadas pela Worldline com a
Universidade Livre de Bruxelas (ULB). Para proteger os clientes, as
variáveis originais viraram 28 componentes principais, de V1 a V28, sem
nome; só o tempo, em segundos desde a primeira compra, e o valor vêm como
são. O rótulo é o final: a base não diz quando a fraude foi descoberta.

O arquivo vem do OpenML (conjunto 1597), e é conferido pelo MD5 publicado
lá. Ele fica na pasta dos dados, e não no repositório.

Dal Pozzolo, Caelen, Johnson e Bontempi. Calibrating Probability with
Undersampling for Unbalanced Classification. IEEE CIDM, 2015.
"""

import hashlib
import urllib.request
from pathlib import Path

import polars as pl

ENDERECO = "https://www.openml.org/data/v1/download/1673544/creditcard.arff"
MD5 = "178bcf9bb1f31a3dfe12d0e577884add"
"""O MD5 que o OpenML publica para o arquivo: o que chega tem de ser este."""

ARQUIVO = "creditcard.arff"
COMPONENTES: tuple[str, ...] = tuple(f"V{numero}" for numero in range(1, 29))
"""As 28 componentes principais, sem nome: o banco não pôde mostrar as variáveis de onde elas saíram."""


def baixar(pasta: Path, endereco: str = ENDERECO, md5: str = MD5) -> Path:
    """O arquivo da base na pasta, baixado se ainda não estiver lá, e conferido pelo MD5.

    O download vai para um arquivo provisório, que só ganha o nome certo depois de conferido.
    """
    destino = pasta / ARQUIVO

    if destino.exists() and _md5(destino) == md5:
        return destino

    pasta.mkdir(parents=True, exist_ok=True)
    provisorio = destino.with_suffix(".parte")

    with urllib.request.urlopen(endereco, timeout=60) as resposta, provisorio.open("wb") as arquivo:
        while bloco := resposta.read(1 << 20):
            arquivo.write(bloco)

    chegou = _md5(provisorio)

    if chegou != md5:
        provisorio.unlink()
        raise ValueError(f"O arquivo baixado de {endereco} não é o publicado: o MD5 é {chegou}, e não {md5}.")

    provisorio.replace(destino)

    return destino


def ler(arquivo: Path) -> pl.DataFrame:
    """As compras da base, na ordem do arquivo: `id`, `segundo`, V1 a V28, `valor` (em euros) e `fraude`.

    O arquivo é ARFF: um cabeçalho com o nome de cada coluna, e as linhas depois de `@data`, separadas por vírgula.
    """
    nomes: list[str] = []
    linhas_do_cabecalho = 0

    with arquivo.open(encoding="utf-8") as texto:
        for linha in texto:
            linhas_do_cabecalho += 1

            if linha.lower().startswith("@attribute"):
                nomes.append(linha.split()[1].strip("'"))
            elif linha.lower().startswith("@data"):
                break

    esperado = ["Time", *COMPONENTES, "Amount", "Class"]

    if nomes != esperado:
        raise ValueError(f"O arquivo {arquivo} não tem as colunas da base de cartões: tem {', '.join(nomes)}.")

    # Cada coluna com o tipo dela, sem adivinhar: o arquivo escreve o segundo 100.000 como "1e+05".
    compras = pl.read_csv(
        arquivo,
        skip_rows=linhas_do_cabecalho,
        has_header=False,
        schema=dict.fromkeys(nomes[:-1], pl.Float64) | {"Class": pl.String},
        quote_char="'",
    )

    return compras.select(
        pl.int_range(pl.len(), dtype=pl.Int64).alias("id"),
        pl.col("Time").cast(pl.Float64).alias("segundo"),
        *(pl.col(componente).cast(pl.Float64) for componente in COMPONENTES),
        pl.col("Amount").cast(pl.Float64).alias("valor"),
        (pl.col("Class") == "1").alias("fraude"),
    )


def _md5(arquivo: Path) -> str:
    resumo = hashlib.md5(usedforsecurity=False)

    with arquivo.open("rb") as conteudo:
        while bloco := conteudo.read(1 << 20):
            resumo.update(bloco)

    return resumo.hexdigest()
