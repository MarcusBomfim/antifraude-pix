#!/usr/bin/env bash
# A fumaça da imagem (ADR-0025): com dados pequenos, a imagem sobe, fica
# saudável e decide pela API os Pix dos dois últimos dias, que a reprodução
# manda. Roda no CI a cada commit, e em qualquer máquina com o Docker e o uv:
#
#   bash implantacao/fumaca.sh [imagem]
#
# FONTE e REVISAO, se existirem no ambiente, vão para os rótulos da imagem.
set -euo pipefail

IMAGEM=${1:-antifraude-pix:fumaca}
NOME=antifraude-pix-fumaca
PORTA=${PORTA:-18000}
DADOS=$(mktemp -d)

limpar() {
  docker rm --force "$NOME" > /dev/null 2>&1 || true
  rm -rf "$DADOS"
}
trap limpar EXIT

echo "== Os dados: 150 clientes por 125 dias, o campeão registrado e empacotado"
uv run antifraude-pix simular --clientes 150 --dias 125 --golpes-por-mil 200 --saida "$DADOS" > /dev/null
uv run antifraude-pix registrar "$DADOS" > /dev/null
uv run antifraude-pix empacotar "$DADOS"
# O usuário da imagem não é o dono da pasta: ela tem de ser legível por todos.
chmod -R a+rX "$DADOS"

echo "== A imagem"
docker build --tag "$IMAGEM" --build-arg "FONTE=${FONTE:-}" --build-arg "REVISAO=${REVISAO:-}" .

echo "== O contêiner, com a história até 01/11/2025"
MONTAGEM=$DADOS

# No Git Bash do Windows, o Docker quer o caminho do Windows, e o Bash não pode reescrever o /dados. Só neste
# comando: o uv, nos outros, precisa que o Bash converta o caminho da pasta temporária.
if command -v cygpath > /dev/null; then
  MONTAGEM=$(cygpath --windows "$DADOS")
fi

MSYS_NO_PATHCONV=1 docker run --detach --name "$NOME" --publish "127.0.0.1:$PORTA:8000" --volume "$MONTAGEM:/dados:ro" "$IMAGEM" \
  servir /dados --pacote /dados/campeao --host 0.0.0.0 --ate 2025-11-01

for _ in $(seq 1 60); do
  SAUDE=$(docker inspect --format '{{.State.Health.Status}}' "$NOME")

  if [ "$SAUDE" = healthy ]; then
    break
  fi

  if [ "$(docker inspect --format '{{.State.Running}}' "$NOME")" != true ]; then
    break
  fi

  sleep 3
done

if [ "$SAUDE" != healthy ]; then
  echo "A imagem não ficou saudável ($SAUDE):"
  docker logs "$NOME"
  exit 1
fi

docker logs "$NOME"

echo "== A reprodução dos dois últimos dias pela API"
uv run antifraude-pix reproduzir "http://127.0.0.1:$PORTA" "$DADOS" --de 2025-11-01 --velocidade 0
echo "== A fumaça passou"
