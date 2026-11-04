# ADR-0005 — A visão do banco, e o corte no tempo: o que o banco sabia em cada instante

**Status:** aceita

## Contexto

A Parte 1 grava o que o banco vê numa pasta e a verdade do simulador em outra (ADR-0004). Separar as pastas não basta. O código que aprende recebe, quase sempre, um objeto com tudo dentro, e basta um `simulacao.verdade` distraído para o modelo aprender com o que nenhum banco tem.

Também não basta ler só o que o banco vê, porque o banco sabe as coisas na hora em que elas acontecem:
- a contestação de um Pix de segunda-feira só existe na quarta;
- o aparelho novo só existe depois de entrar na conta.

Um modelo treinado no dia 1º de outubro só pode usar o que existia no dia 1º de outubro, e com o rótulo daquele dia. Ainda não sabe do golpe que vai ser contestado no dia 5.

## Decisão

- **A visão do banco é um tipo próprio**, `VisaoDoBanco`, com o cadastro, os Pix, os aparelhos novos e as contestações, e nenhum campo para a verdade. As variáveis, os modelos e o serviço recebem a visão, e não a simulação.
- **Ler o banco do disco não abre a pasta da verdade.** `ler_o_banco` confere o SHA-256 só dos arquivos de `banco/` e lê só eles. A leitura funciona com a pasta da verdade apagada, e o teste confere isso.
- **O corte num instante** (`visao.ate(instante)`) devolve o banco daquele momento:
  - os Pix e os aparelhos novos de antes do instante;
  - as contestações que já tinham chegado. A de um Pix antigo, feita depois do instante, ainda não existe nele;
  - o cadastro inteiro, porque os clientes já estavam no banco.
- **O instante é no horário de Brasília, sem fuso**, como as datas da configuração. Um instante com fuso é recusado, em vez de convertido em silêncio.

## Consequências

- **O treino da Parte 3 corta no tempo**, e o rótulo do treino é o que o banco tinha no dia do corte. O golpe recente, ainda sem contestação, entra como legítimo, como entra no banco.
- **O vazamento vira teste.** Uma variável que não olha o futuro dá o mesmo valor calculada com o banco inteiro e com o banco cortado em qualquer instante depois do Pix (ADR-0006).
- **A verdade continua disponível para medir**, pela simulação inteira, e só pelos testes e pelas análises que dizem por que a usam.
