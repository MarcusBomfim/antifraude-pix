"""Os parâmetros da simulação inteira, cada um com o seu porquê (ADR-0002).

Aqui mora o que vale para o mundo todo: quantos clientes, quantos dias,
quantos golpes por mês, o limite noturno, o prazo do MED. O que é hábito de
cada cliente ou mecânica de cada golpe tem nome e porquê ao lado do código
que o usa, na rotina e nos golpes. A mesma configuração com a mesma semente
dá sempre os mesmos dados.
"""

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True, slots=True, kw_only=True)
class Configuracao:
    """O mundo que o simulador monta."""

    clientes: int = 20_000
    """Os clientes do banco cujo Pix de saída o modelo vai vigiar."""

    inicio: date = date(2025, 7, 1)
    """O primeiro dia simulado.

    De julho a dezembro, a simulação passa pela Black Friday e pelo Natal, quando a venda falsa cresce.
    """

    dias: int = 184
    """Quantos dias a simulação cobre."""

    semente: int = 2026
    """A semente de todo o sorteio: a mesma semente dá os mesmos dados, byte a byte."""

    golpes_por_mil_clientes_ao_mes: float = 10.0
    """Quantos ataques começam por mês, a cada mil clientes.

    É mais do que a fraude real, de propósito: com vinte mil clientes e a
    taxa de verdade, sobrariam poucas centenas de fraudes, e nenhum modelo
    aprende com isso. A proporção continua rara — perto de 0,1% das
    transações —, e a taxa é um parâmetro, para quem quiser a mais baixa.
    """

    limite_noturno: int = 100_000
    """O limite do Pix noturno, em centavos: R$ 1.000, o padrão do Banco Central para pessoa física."""

    inicio_da_noite: int = 20
    """A hora em que começa o período noturno do Pix."""

    fim_da_noite: int = 6
    """A hora em que o período noturno termina."""

    prazo_do_med: int = 80
    """Os dias que a vítima tem para pedir a devolução pelo MED, o Mecanismo Especial de Devolução do Pix."""

    virada: date | None = None
    """O dia em que a quadrilha da falsa central muda de tática, ou nenhum (ADR-0019).

    Dali em diante, a "central" ensina a vítima a mandar o dinheiro aos
    poucos: um Pix por dia, para a mesma "conta segura", sem pressa. É o golpe
    novo que o monitoramento tem de perceber. Sem virada, a simulação é a de
    sempre, byte a byte.
    """

    def __post_init__(self) -> None:
        problemas = [
            problema
            for problema in (
                "O número de clientes tem de ser positivo." if self.clientes <= 0 else None,
                "A semente não pode ser negativa." if self.semente < 0 else None,
                "O número de dias tem de ser positivo." if self.dias <= 0 else None,
                "A taxa de golpes não pode ser negativa." if self.golpes_por_mil_clientes_ao_mes < 0 else None,
                "O limite noturno tem de ser positivo." if self.limite_noturno <= 0 else None,
                "O prazo do MED tem de ser positivo." if self.prazo_do_med <= 0 else None,
                "A virada tem de cair dentro da simulação."
                if self.virada is not None and not self.inicio <= self.virada < self.fim
                else None,
            )
            if problema is not None
        ]

        if problemas:
            raise ValueError(" ".join(problemas))

    @property
    def fim(self) -> date:
        """O dia seguinte ao último simulado."""
        return self.inicio + timedelta(days=self.dias)

    def noturno(self, hora: int) -> bool:
        """A hora está no período noturno do Pix, das 20h às 6h."""
        return hora >= self.inicio_da_noite or hora < self.fim_da_noite
