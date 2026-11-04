import os

from hypothesis import settings

# No CI, o hypothesis sorteia sempre os mesmos exemplos: a falha que aparece lá
# aparece de novo na máquina de quem for investigar, e o verde não depende de
# sorte. Na máquina de quem desenvolve, os exemplos mudam a cada rodada, e
# acham o que os do CI não acharam.
settings.register_profile("ci", derandomize=True, deadline=None, print_blob=True)
settings.register_profile("local", deadline=None)
settings.load_profile("ci" if os.environ.get("CI") else "local")
