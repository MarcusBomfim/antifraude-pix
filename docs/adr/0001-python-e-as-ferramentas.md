# ADR-0001 — Python, e as ferramentas que tratam o código de dados como código

**Status:** aceita

## Contexto

O projeto é de machine learning: dados, variáveis, modelos, métricas e um serviço que decide em milissegundos se um Pix sai. O ecossistema dessas peças é Python. É lá que estão a NumPy, as tabelas (pandas, Polars), os modelos (scikit-learn, LightGBM, XGBoost), a explicação (SHAP), o rastreio dos experimentos (MLflow) e o serviço (FastAPI). Também é Python o que as vagas de ciência de dados e de engenharia de ML pedem.

O código de ciência de dados costuma ser tratado como rascunho: notebook sem teste, dependência sem versão, número mágico no meio. Este projeto é feito com o mesmo cuidado dos outros, em C#. Aqui, o modelo vai decidir o dinheiro de alguém.

## Decisão

- **Python 3.13**, a versão que todas as bibliotecas do projeto já suportam, fixada na `.python-version`.
- **O uv** cuida do Python, do ambiente e das dependências. O `uv.lock` fixa cada pacote, inclusive os indiretos, e o CI instala exatamente o mesmo (`uv sync --locked`).
- **As versões são fixadas**, e só entram versões com pelo menos duas semanas. A mesma semente tem de dar os mesmos dados hoje e daqui a um ano.
  - O Polars fica na 1.44: a 2.0, uma versão maior, tem quatro semanas.
- **O Polars cuida das tabelas, e a NumPy do sorteio.**
  - O Polars é rápido, tem tipo em cada coluna e lê e grava Parquet sem intermediário.
  - Onde um modelo pedir pandas ou NumPy, a conversão acontece na fronteira, e não no meio do caminho.
- **O ruff formata e confere o estilo, e o mypy confere os tipos no modo estrito**, nos testes também. Aviso é erro, como nos outros projetos.
- **Os testes são em pytest, com o hypothesis para as propriedades.**
  - Exemplos conferidos à mão onde há resposta conhecida, como a data da Páscoa.
  - Propriedades sorteadas onde a regra vale para qualquer entrada, como "o quinto dia útil é útil e tem quatro dias úteis antes".
  - No CI, o hypothesis sorteia sempre os mesmos exemplos.
- **O código fala português**, como nos outros projetos: `Cliente`, `Laranja`, `quinto_dia_util`. As palavras da biblioteca ficam como a biblioteca as chama.

## Consequências

- **Rodar o projeto é instalar o uv e chamar `uv sync`.** O uv baixa o Python certo sozinho.
- **Atualizar uma biblioteca é uma decisão**, com o `uv.lock` mudando num commit que diz por quê.
- **O mypy estrito pede tipo em tudo**, inclusive nos vetores da NumPy. Custa um pouco de cerimônia, e pega o vetor de números com vírgula passado onde se esperava o de inteiros.
