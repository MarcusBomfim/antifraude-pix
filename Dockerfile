# syntax=docker/dockerfile:1
#
# A imagem do serviço que decide cada Pix (ADR-0025): o código e as
# dependências do uv.lock, sem os dados. A história do banco e o pacote do
# campeão entram como volume, só para leitura, na pasta /dados.
#
# As imagens de base são fixadas pelo digest, além da tag: a tag pode ser
# movida, e o digest não. A tag diz a versão, para quem for atualizar.

FROM ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 AS uv

# A construção: o ambiente do uv.lock, sem as ferramentas de desenvolvimento,
# e o pacote instalado nele, sem depender da pasta src.
FROM python:3.13.18-slim-trixie@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS construcao

COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# As dependências primeiro, numa camada que só muda quando o uv.lock muda.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project

COPY LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-editable

# A imagem que roda: o mesmo Python, o ambiente pronto, e um usuário sem poder.
FROM python:3.13.18-slim-trixie@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# O LightGBM usa o OpenMP, que a imagem slim não traz.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 antifraude
COPY --from=construcao /app/.venv /app/.venv

# De onde veio a imagem: o CI passa o repositório e o commit.
ARG FONTE=""
ARG REVISAO=""
LABEL org.opencontainers.image.title="antifraude-pix" \
      org.opencontainers.image.description="A API que decide cada Pix: aprovar, segurar para confirmar ou bloquear." \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="${FONTE}" \
      org.opencontainers.image.revision="${REVISAO}"

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1
USER antifraude
WORKDIR /home/antifraude
EXPOSE 8000

# O estado do banco leva uns 30 s para subir com dois milhões de Pix: a saúde só conta depois disso.
HEALTHCHECK --interval=10s --timeout=3s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/saude', timeout=2)"]

ENTRYPOINT ["antifraude-pix"]
CMD ["servir", "/dados", "--pacote", "/dados/campeao", "--host", "0.0.0.0", "--porta", "8000"]
