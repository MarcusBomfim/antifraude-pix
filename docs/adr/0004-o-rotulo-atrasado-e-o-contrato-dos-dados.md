# ADR-0004 — O rótulo atrasado e sujo, o contrato dos dados, e a verdade em outra pasta

**Status:** aceita

## Contexto

Na vida real, o banco não sabe na hora que um Pix foi golpe. Sabe quando a vítima reclama, e cada golpe tem a sua demora:
- na invasão de conta, a vítima percebe em horas;
- no falso parente, quando fala com o filho de verdade;
- na venda falsa, quando o produto não chega, dias depois.

Parte das vítimas nunca reclama. E há quem conteste um Pix legítimo.

O modelo treinado com a verdade do simulador seria melhor do que qualquer modelo de banco consegue ser. O avaliado sem a demora pareceria melhor do que é, porque na hora de treinar os golpes da última semana ainda não teriam sido contestados.

## Decisão

- **O rótulo do banco é a contestação**, com a chance e a demora de cada golpe:

  | Golpe | Contestado | Demora típica |
  |---|---|---|
  | Invasão de conta | 95% | 7 horas |
  | Falsa central | 90% | 10 horas |
  | Falso parente | 85% | um dia e meio |
  | Venda falsa | 60% | oito dias |

  - A vítima contesta o ataque inteiro, de uma vez, depois do último Pix dele.
  - Ninguém contesta antes de dez minutos, e ninguém depois do prazo do MED, de 80 dias.
  - Cinquenta Pix legítimos a cada milhão são contestados mesmo assim.
- **A verdade fica em outra pasta.**
  - O que o banco vê vai para `banco/`: o cadastro, os Pix, os aparelhos novos e as contestações.
  - O que só o simulador sabe vai para `verdade/`: o golpe de cada Pix, os ataques e as contas laranja.
  - O código que treina lê de `banco/`. Um `verdade/` nele chama a atenção em qualquer revisão.
- **O contrato dos dados é escrito e conferido.**
  - Cada tabela tem colunas e tipos fixos; as categorias são tipos fechados, e o dinheiro é inteiro, em centavos.
  - Uma validação de fora confere o que tem de valer, sem ler o código do simulador:
    - os Pix em ordem, no período, dentro do saldo e do limite noturno;
    - cada aparelho existindo antes do Pix que sai dele;
    - cada contestação depois do Pix e dentro do prazo do MED;
    - cada fraude apontando para um Pix e um ataque que existem.
  - Os problemas vêm todos de uma vez, em frases. Dados que não passam não são gravados.
- **O manifesto diz de onde os dados vieram**: a configuração, a versão do simulador, e as linhas e o SHA-256 de cada arquivo. A leitura confere o SHA-256, e o arquivo mexido depois de gravado é recusado.

## Consequências

- **O treino da Parte 3 terá de cortar no tempo.** Só conta como golpe o que foi contestado até o dia do corte. O golpe recente, ainda sem contestação, fica como legítimo no treino, como fica no banco.
- **A distância entre o rótulo e a verdade vira medida.** Quanto do golpe o banco nunca soube, quanto o modelo pega do que nunca foi contestado, quanto o rótulo sujo custa.
- **Os mesmos dados dão os mesmos arquivos, byte a byte**, para a mesma versão do Polars. Trocar a versão pode mudar os bytes sem mudar os dados, e é a leitura, e não o hash, que diz se os dados são os mesmos.
