from __future__ import annotations

import math
import heapq
import os
from dataclasses import dataclass
from numbers import Real
from typing import Iterable, TypeAlias

import numpy as np


Cell: TypeAlias = tuple[int, int]
WorldPoint: TypeAlias = tuple[float, float]


class PlanejadorGlobal:
    """Constroi e mantem o grid de ocupacao estatico do ambiente.

    Args:
        img: Imagem ``H x W`` monocromatica ou ``H x W x 3/4`` RGB/RGBA. Valores
            escuros representam obstaculos; a entrada pode estar em ``[0, 1]``
            ou ``[0, 255]``.
        escala: Metros por pixel, como escalar ou tupla ``(sx, sy)``.
        origem: Coordenada mundial ``(x, y)`` do canto inferior esquerdo da
            imagem. A linha zero da imagem permanece no topo, como na Aula 13.
        tamanho_celula_m: Lado, em metros, de cada celula do grid.
        raio_robo_m: Raio de colisao usado na inflacao das paredes.
        margem_seguranca_m: Margem adicional de seguranca usada na inflacao.

    Attributes:
        ocupacao_estatica: Matriz booleana ``(linha, coluna)`` com paredes ja
            infladas. ``True`` significa que a celula nao pode ser percorrida.
        ocupacao: Copia que sera atualizada pelos bloqueios dinamicos. Nesta
            etapa ela ainda e identica a ``ocupacao_estatica``.
    """

    def __init__(
        self,
        img: np.ndarray,
        escala: float | tuple[float, float],
        origem: tuple[float, float],
        *,
        tamanho_celula_m: float = 0.10,
        raio_robo_m: float = 0.26,
        margem_seguranca_m: float = 0.15,
        raio_inflacao_dinamica_m: float | None = None,
    ) -> None:
        self.escala_x_m_por_pixel, self.escala_y_m_por_pixel = (
            self._normalizar_escala(escala)
        )
        self.origem = self._normalizar_origem(origem)
        self.tamanho_celula_m = self._validar_positivo(
            tamanho_celula_m, "tamanho_celula_m"
        )
        self.raio_robo_m = self._validar_nao_negativo(raio_robo_m, "raio_robo_m")
        self.margem_seguranca_m = self._validar_nao_negativo(
            margem_seguranca_m, "margem_seguranca_m"
        )

        pixels_ocupados = self._binarizar_imagem(img)
        self.altura_pixels, self.largura_pixels = pixels_ocupados.shape
        self.altura_m = self.altura_pixels * self.escala_y_m_por_pixel
        self.largura_m = self.largura_pixels * self.escala_x_m_por_pixel
        self.rows = math.ceil(self.altura_m / self.tamanho_celula_m)
        self.cols = math.ceil(self.largura_m / self.tamanho_celula_m)

        ocupacao_base = self._agrupar_pixels_em_grid(pixels_ocupados)
        self.ocupacao_base = ocupacao_base.copy()
        raio_inflacao = self.raio_robo_m + self.margem_seguranca_m
        self.ocupacao_estatica = self._inflar_obstaculos(
            ocupacao_base, raio_inflacao
        )
        self.ocupacao_estatica_parede = self._inflar_obstaculos(
            ocupacao_base, self.tamanho_celula_m * 1.45
        )
        self.ocupacao_dinamica = np.zeros_like(self.ocupacao_estatica, dtype=bool)
        self.ocupacao = self.ocupacao_estatica.copy()
        self._contagem_bloqueio_dinamico = np.zeros_like(
            self.ocupacao_estatica, dtype=np.int32
        )
        if raio_inflacao_dinamica_m is not None:
            self.raio_inflacao_dinamica_m = float(raio_inflacao_dinamica_m)
        else:
            self.raio_inflacao_dinamica_m = float(
                self.raio_robo_m + min(self.margem_seguranca_m, 0.12)
            )
        self._deltas_inflacao_dinamica: list[tuple[int, int]] = (
            self._precomputar_deltas_inflacao(self.raio_inflacao_dinamica_m)
        )

        # Estado D* Lite. Os dicionarios sao esparsos: ausencia equivale a inf.
        self.g: dict[Cell, float] = {}
        self.rhs: dict[Cell, float] = {}
        self.U: list[tuple[float, float, int, Cell]] = []
        self.km = 0.0
        self.goal: Cell | None = None
        self.goal_xy: WorldPoint | None = None
        self.start: Cell | None = None
        self.last_start: Cell | None = None
        self._chaves_atuais: dict[Cell, tuple[float, float]] = {}
        self._contador_fila = 0
        self._celulas_dinamicas_pendentes: set[Cell] = set()
        self._replanejamento_pendente = False

    @staticmethod
    def _validar_positivo(valor: float, nome: str) -> float:
        """Valida e normaliza uma distancia estritamente positiva em metros."""
        if not isinstance(valor, Real) or not math.isfinite(float(valor)) or valor <= 0:
            raise ValueError(f"{nome} deve ser um numero finito maior que zero")
        return float(valor)

    @staticmethod
    def _validar_nao_negativo(valor: float, nome: str) -> float:
        """Valida e normaliza uma distancia nao negativa em metros."""
        if not isinstance(valor, Real) or not math.isfinite(float(valor)) or valor < 0:
            raise ValueError(f"{nome} deve ser um numero finito maior ou igual a zero")
        return float(valor)

    @classmethod
    def _normalizar_escala(
        cls, escala: float | tuple[float, float]
    ) -> tuple[float, float]:
        """Converte escala escalar ou ``(sx, sy)`` para metros por pixel."""
        if isinstance(escala, Real):
            valor = cls._validar_positivo(float(escala), "escala")
            return valor, valor

        if len(escala) != 2:
            raise ValueError("escala deve ser um numero ou uma tupla (sx, sy)")
        return (
            cls._validar_positivo(escala[0], "escala[0]"),
            cls._validar_positivo(escala[1], "escala[1]"),
        )

    @staticmethod
    def _normalizar_origem(origem: tuple[float, float]) -> WorldPoint:
        """Valida a coordenada mundial do canto inferior esquerdo do mapa."""
        if len(origem) != 2 or not all(
            isinstance(valor, Real) and math.isfinite(float(valor)) for valor in origem
        ):
            raise ValueError("origem deve ser uma tupla finita (x, y)")
        return float(origem[0]), float(origem[1])

    @staticmethod
    def _binarizar_imagem(img: np.ndarray) -> np.ndarray:
        """Retorna matriz booleana na qual ``True`` significa pixel ocupado.

        O limiar de luminosidade e ``0,5`` apos normalizacao, equivalente a
        inverter preto/branco e binarizar no notebook da Aula 13.
        """
        imagem = np.asarray(img)
        if imagem.ndim == 2:
            luminancia = imagem.astype(float)
        elif imagem.ndim == 3 and imagem.shape[2] in (3, 4):
            rgb = imagem[..., :3].astype(float)
            luminancia = (
                0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
            )
        else:
            raise ValueError("img deve ter formato HxW, HxWx3 ou HxWx4")

        if luminancia.size == 0 or not np.all(np.isfinite(luminancia)):
            raise ValueError("img deve conter ao menos um pixel finito")

        if float(np.max(luminancia)) > 1.0:
            luminancia = luminancia / 255.0
        luminancia = np.clip(luminancia, 0.0, 1.0)
        return luminancia < 0.5

    def _agrupar_pixels_em_grid(self, pixels_ocupados: np.ndarray) -> np.ndarray:
        """Agrupa pixels por celula; qualquer pixel ocupado bloqueia o bloco.

        Esta e a mesma decisao de ocupacao usada no notebook de Aula 13, onde
        o somatorio do bloco e binarizado acima de zero.
        """
        grade = np.zeros((self.rows, self.cols), dtype=bool)
        for linha in range(self.rows):
            y_inicio = int(
                math.floor(
                    linha * self.tamanho_celula_m / self.escala_y_m_por_pixel
                )
            )
            y_fim = min(
                self.altura_pixels,
                int(
                    math.ceil(
                        (linha + 1)
                        * self.tamanho_celula_m
                        / self.escala_y_m_por_pixel
                    )
                ),
            )
            for coluna in range(self.cols):
                x_inicio = int(
                    math.floor(
                        coluna * self.tamanho_celula_m / self.escala_x_m_por_pixel
                    )
                )
                x_fim = min(
                    self.largura_pixels,
                    int(
                        math.ceil(
                            (coluna + 1)
                            * self.tamanho_celula_m
                            / self.escala_x_m_por_pixel
                        )
                    ),
                )
                if y_inicio < y_fim and x_inicio < x_fim:
                    grade[linha, coluna] = bool(
                        np.any(pixels_ocupados[y_inicio:y_fim, x_inicio:x_fim])
                    )
        return grade

    def _inflar_obstaculos(
        self, ocupacao_base: np.ndarray, raio_m: float
    ) -> np.ndarray:
        """Dilata paredes com um disco de raio ``raio_m`` sem SciPy."""
        if raio_m == 0:
            return ocupacao_base.copy()

        alcance = math.ceil(raio_m / self.tamanho_celula_m)
        inflada = ocupacao_base.copy()
        for delta_linha in range(-alcance, alcance + 1):
            for delta_coluna in range(-alcance, alcance + 1):
                distancia = math.hypot(delta_linha, delta_coluna) * self.tamanho_celula_m
                if distancia > raio_m + 1e-12:
                    continue

                destino_linha_inicio = max(0, delta_linha)
                destino_linha_fim = min(self.rows, self.rows + delta_linha)
                destino_coluna_inicio = max(0, delta_coluna)
                destino_coluna_fim = min(self.cols, self.cols + delta_coluna)
                origem_linha_inicio = max(0, -delta_linha)
                origem_linha_fim = self.rows - max(0, delta_linha)
                origem_coluna_inicio = max(0, -delta_coluna)
                origem_coluna_fim = self.cols - max(0, delta_coluna)

                inflada[
                    destino_linha_inicio:destino_linha_fim,
                    destino_coluna_inicio:destino_coluna_fim,
                ] |= ocupacao_base[
                    origem_linha_inicio:origem_linha_fim,
                    origem_coluna_inicio:origem_coluna_fim,
                ]
        return inflada

    def _precomputar_deltas_inflacao(self, raio_m: float) -> list[tuple[int, int]]:
        """Precomputa deltas relativos (dr, dc) para inflacao circular de raio informado."""
        alcance = math.ceil(raio_m / self.tamanho_celula_m)
        deltas: list[tuple[int, int]] = []
        for dr in range(-alcance, alcance + 1):
            for dc in range(-alcance, alcance + 1):
                if math.hypot(dr, dc) * self.tamanho_celula_m <= raio_m + 1e-12:
                    deltas.append((dr, dc))
        return deltas

    def mundo_para_celula(self, ponto_xy: WorldPoint) -> Cell | None:
        """Converte ``(x, y)`` mundial para ``(linha, coluna)`` ou ``None``.

        A linha e invertida porque a imagem possui origem no canto superior,
        enquanto o eixo ``y`` do CoppeliaSim cresce para cima.
        """
        x, y = float(ponto_xy[0]), float(ponto_xy[1])
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        coluna = math.floor((x - self.origem[0]) / self.tamanho_celula_m)
        linha_mundo = math.floor((y - self.origem[1]) / self.tamanho_celula_m)
        if not (0 <= coluna < self.cols and 0 <= linha_mundo < self.rows):
            return None
        return self.rows - 1 - linha_mundo, coluna

    def celula_para_mundo(self, celula: Cell) -> WorldPoint:
        """Retorna o centro mundial da celula ``(linha, coluna)`` informada."""
        linha, coluna = celula
        if not (0 <= linha < self.rows and 0 <= coluna < self.cols):
            raise ValueError("celula fora dos limites do grid")
        return (
            self.origem[0] + (coluna + 0.5) * self.tamanho_celula_m,
            self.origem[1] + (self.rows - linha - 0.5) * self.tamanho_celula_m,
        )

    def eh_obstaculo_estatico(self, ponto_xy: WorldPoint) -> bool:
        """Informa se a coordenada mundial (x, y) coincide com obstaculo estatico (parede).

        Considera estaticos os pontos que caem dentro de paredes do mapa original
        ou em sua vizinhanca imediata (~1 celula de tolerancia de discretizacao),
        alem de pontos fora dos limites do mapa.
        """
        celula = self.mundo_para_celula(ponto_xy)
        if celula is None:
            return True
        linha, coluna = celula
        return bool(self.ocupacao_estatica_parede[linha, coluna])

    def definir_goal(self, goal_xy: WorldPoint) -> None:
        """Reinicializa D* Lite para o novo goal mundial ``(x, y)``.

        O goal precisa estar dentro de uma celula livre do mapa estatico. A
        Bloqueios dinamicos posteriores podem torna-lo indisponivel; nesse
        caso, ``caminho_completo`` retorna uma lista vazia ate o desbloqueio.
        """
        goal = self.mundo_para_celula(goal_xy)
        if goal is None:
            raise ValueError("goal esta fora dos limites do mapa")
        if self.ocupacao_estatica[goal]:
            raise ValueError("goal pertence a uma celula ocupada no mapa estatico")

        self.g.clear()
        self.rhs.clear()
        self.U.clear()
        self._chaves_atuais.clear()
        self.km = 0.0
        self.goal = goal
        self.goal_xy = (float(goal_xy[0]), float(goal_xy[1]))
        self.start = None
        self.last_start = None
        self.rhs[goal] = 0.0
        self._inserir_fila(goal)
        self._replanejamento_pendente = True

    def marcar_bloqueio(
        self, celulas: list[Cell], bloqueado: bool, inflar: bool = True
    ) -> None:
        """Marca ou libera celulas dinamicas no grid com inflacao e contagem de referencia.

        Args:
            celulas: Lista de indices ``(linha, coluna)`` do grid. Celulas fora
                do mapa sao ignoradas para tornar a fronteira com sensores
                segura.
            bloqueado: ``True`` adiciona um obstaculo dinamico; ``False`` o
                remove e decrementa contagens de referencia. Paredes estaticas
                infladas nunca sao removidas.
            inflar: Se ``True``, infla pelo raio de colisao do robo para fechar
                frestas estreitas e intransitaveis no grid do D*.
        """
        if not isinstance(bloqueado, (bool, np.bool_)):
            raise ValueError("bloqueado deve ser booleano")

        deltas = self._deltas_inflacao_dinamica if inflar else [(0, 0)]
        celulas_afetadas: set[Cell] = set()

        for celula_centro in celulas:
            if not self._celula_valida(celula_centro):
                continue
            cr, cc = celula_centro
            for dr, dc in deltas:
                r, c = cr + dr, cc + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    celulas_afetadas.add((r, c))

        for linha, coluna in celulas_afetadas:
            # Uma parede estatica ja esta permanentemente ocupada.
            if self.ocupacao_estatica[linha, coluna]:
                continue

            if bloqueado:
                # Nao bloqueia a celula onde o robo ou o goal estao agora
                if (linha, coluna) == self.start or (linha, coluna) == self.goal:
                    continue
                if self._contagem_bloqueio_dinamico[linha, coluna] == 0:
                    self.ocupacao_dinamica[linha, coluna] = True
                    self.ocupacao[linha, coluna] = True
                    self._celulas_dinamicas_pendentes.add((linha, coluna))
                    self._replanejamento_pendente = True
                self._contagem_bloqueio_dinamico[linha, coluna] += 1
            else:
                if self._contagem_bloqueio_dinamico[linha, coluna] > 0:
                    self._contagem_bloqueio_dinamico[linha, coluna] -= 1
                    if self._contagem_bloqueio_dinamico[linha, coluna] == 0:
                        self.ocupacao_dinamica[linha, coluna] = False
                        self.ocupacao[linha, coluna] = bool(
                            self.ocupacao_estatica[linha, coluna]
                        )
                        self._celulas_dinamicas_pendentes.add((linha, coluna))
                        self._replanejamento_pendente = True

    def limpar_obstaculos_dinamicos(self) -> None:
        """Remove todos os obstaculos dinamicos e restaura ocupacao estatica."""
        linhas, colunas = np.where(self.ocupacao_dinamica)
        for r, c in zip(linhas, colunas):
            self.ocupacao_dinamica[r, c] = False
            self.ocupacao[r, c] = bool(self.ocupacao_estatica[r, c])
            self._celulas_dinamicas_pendentes.add((int(r), int(c)))
        self._contagem_bloqueio_dinamico.fill(0)
        if len(linhas) > 0:
            self._replanejamento_pendente = True

    def atualizar(self, pose_atual: tuple[float, float, float]) -> None:
        """Propaga apenas alteracoes de start ou de custo desde o ultimo ciclo.

        Args:
            pose_atual: Pose mundial ``(x, y, theta)``. Somente ``x`` e ``y``
                determinam a celula de inicio do D* Lite.
        """
        if self.goal is None:
            return
        if len(pose_atual) < 2:
            raise ValueError("pose_atual deve conter ao menos x e y")

        start = self.mundo_para_celula((pose_atual[0], pose_atual[1]))
        if start is None:
            self.start = None
            return

        start_mudou = start != self.last_start
        if self.last_start is not None and start_mudou:
            self.km += self._heuristica(self.last_start, start)
        self.start = start
        self.last_start = start

        if not (start_mudou or self._replanejamento_pendente):
            return

        for celula in self._celulas_dinamicas_pendentes:
            self._atualizar_vertice(celula)
            for predecessor in self._vizinhos(celula):
                self._atualizar_vertice(predecessor)
        self._celulas_dinamicas_pendentes.clear()
        self._compute_shortest_path()
        self._replanejamento_pendente = False

    def _encontrar_celula_livre_proxima(
        self, celula_origem: Cell, max_raio: int = 6
    ) -> Cell | None:
        """Busca por BFS a celula livre mais proxima caso start caia na margem inflada."""
        from collections import deque

        fila = deque([celula_origem])
        visitados = {celula_origem}
        while fila:
            r, c = fila.popleft()
            if not self.ocupacao[r, c] and math.isfinite(self._g((r, c))):
                return (r, c)
            for dr, dc in [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.rows
                    and 0 <= nc < self.cols
                    and (nr, nc) not in visitados
                    and abs(nr - celula_origem[0]) <= max_raio
                    and abs(nc - celula_origem[1]) <= max_raio
                ):
                    visitados.add((nr, nc))
                    fila.append((nr, nc))
        return None

    def caminho_completo(
        self, pose_atual: tuple[float, float, float]
    ) -> list[WorldPoint]:
        """Retorna caminho mundial suavizado entre pose e goal, ou ``[]``.

        A lista inclui o centro da celula inicial e termina na coordenada exata
        passada a ``definir_goal``. Se o robo tangenciou momentaneamente a
        margem inflada de uma parede, busca a celula livre mais proxima para
        restituir o vetor de navegacao para fora da quina.
        """
        if self.goal is None or self.goal_xy is None:
            return []
        self.atualizar(pose_atual)
        if self.start is None or self.ocupacao[self.goal]:
            return []

        start_planejamento = self.start
        estava_em_margem = False
        if self.ocupacao[start_planejamento]:
            start_livre = self._encontrar_celula_livre_proxima(start_planejamento)
            if start_livre is not None:
                start_planejamento = start_livre
                estava_em_margem = True
            else:
                return []

        if math.isinf(self._g(start_planejamento)):
            return []

        caminho_celulas = self._extrair_caminho_celulas(start_planejamento)
        if not caminho_celulas:
            return []
        caminho_suavizado = self._suavizar_caminho(caminho_celulas)
        caminho = [self.celula_para_mundo(celula) for celula in caminho_suavizado]
        if estava_em_margem:
            caminho.insert(0, (float(pose_atual[0]), float(pose_atual[1])))
        caminho[-1] = self.goal_xy
        return caminho

    def _celula_valida(self, celula: object) -> bool:
        """Indica se o objeto representa indice de celula dentro da grade."""
        if not isinstance(celula, tuple) or len(celula) != 2:
            return False
        linha, coluna = celula
        return (
            isinstance(linha, (int, np.integer))
            and isinstance(coluna, (int, np.integer))
            and 0 <= int(linha) < self.rows
            and 0 <= int(coluna) < self.cols
        )

    def _g(self, celula: Cell) -> float:
        """Retorna ``g(celula)``; valores nao visitados representam infinito."""
        return self.g.get(celula, math.inf)

    def _rhs(self, celula: Cell) -> float:
        """Retorna ``rhs(celula)``; valores nao visitados representam infinito."""
        return self.rhs.get(celula, math.inf)

    @staticmethod
    def _quase_igual(primeiro: float, segundo: float) -> bool:
        """Compara custos finitos e infinitos sem ruido de ponto flutuante."""
        return math.isclose(primeiro, segundo, rel_tol=1e-12, abs_tol=1e-12)

    def _heuristica(self, origem: Cell, destino: Cell) -> float:
        """Distancia octil admissivel para a vizinhanca 8-conexa."""
        delta_linha = abs(origem[0] - destino[0])
        delta_coluna = abs(origem[1] - destino[1])
        menor = min(delta_linha, delta_coluna)
        return max(delta_linha, delta_coluna) + (math.sqrt(2.0) - 1.0) * menor

    def _calcular_chave(self, celula: Cell) -> tuple[float, float]:
        """Calcula a chave lexicografica D* Lite da celula fornecida."""
        if self.start is None:
            raise RuntimeError("start deve existir antes de calcular chave D* Lite")
        menor = min(self._g(celula), self._rhs(celula))
        return menor + self._heuristica(self.start, celula) + self.km, menor

    def _inserir_fila(self, celula: Cell) -> None:
        """Insere ou atualiza uma chave usando invalidacao preguiçosa no heap."""
        # Antes da primeira pose, o goal ainda deve ser enfileirado. A chave
        # usa o proprio goal como referencia temporaria; ela e corrigida quando
        # ``atualizar`` recebe o primeiro start.
        if self.start is None:
            menor = min(self._g(celula), self._rhs(celula))
            chave = menor, menor
        else:
            chave = self._calcular_chave(celula)
        self._chaves_atuais[celula] = chave
        self._contador_fila += 1
        heapq.heappush(self.U, (*chave, self._contador_fila, celula))

    def _topo_valido(self) -> tuple[tuple[float, float], Cell] | None:
        """Descarta entradas antigas e le a menor chave ainda valida."""
        while self.U:
            primeira, segunda, _, celula = self.U[0]
            chave = (primeira, segunda)
            if self._chaves_atuais.get(celula) == chave:
                return chave, celula
            heapq.heappop(self.U)
        return None

    def _remover_topo_valido(self) -> tuple[tuple[float, float], Cell] | None:
        """Extrai a menor entrada atual do heap, ignorando entradas obsoletas."""
        topo = self._topo_valido()
        if topo is None:
            return None
        _, celula = topo
        primeira, segunda, _, celula_extraida = heapq.heappop(self.U)
        assert celula_extraida == celula
        self._chaves_atuais.pop(celula, None)
        return (primeira, segunda), celula

    def _vizinhos(self, celula: Cell) -> list[Cell]:
        """Lista todos os vizinhos 8-conexos internos, inclusive ocupados."""
        linha, coluna = celula
        vizinhos: list[Cell] = []
        for delta_linha in (-1, 0, 1):
            for delta_coluna in (-1, 0, 1):
                if delta_linha == 0 and delta_coluna == 0:
                    continue
                candidato = linha + delta_linha, coluna + delta_coluna
                if self._celula_valida(candidato):
                    vizinhos.append(candidato)
        return vizinhos

    def _custo(self, origem: Cell, destino: Cell) -> float:
        """Retorna custo de aresta ou infinito se o movimento for inviavel."""
        if self.ocupacao[origem] or self.ocupacao[destino]:
            return math.inf
        delta_linha = destino[0] - origem[0]
        delta_coluna = destino[1] - origem[1]
        if abs(delta_linha) > 1 or abs(delta_coluna) > 1 or (
            delta_linha == 0 and delta_coluna == 0
        ):
            return math.inf
        if delta_linha != 0 and delta_coluna != 0:
            # Impede cortar o canto entre duas paredes ortogonais.
            if self.ocupacao[origem[0], destino[1]] or self.ocupacao[destino[0], origem[1]]:
                return math.inf
            return math.sqrt(2.0)
        return 1.0

    def _atualizar_vertice(self, celula: Cell) -> None:
        """Aplica ``updateVertex`` do D* Lite a uma celula."""
        if self.goal is None:
            return
        if celula != self.goal:
            melhor = min(
                (self._custo(celula, vizinho) + self._g(vizinho))
                for vizinho in self._vizinhos(celula)
            )
            self.rhs[celula] = melhor

        self._chaves_atuais.pop(celula, None)
        if not self._quase_igual(self._g(celula), self._rhs(celula)):
            self._inserir_fila(celula)

    def _compute_shortest_path(self) -> None:
        """Executa ``computeShortestPath`` ate consistencia do start atual."""
        if self.start is None or self.goal is None:
            return

        while True:
            topo = self._topo_valido()
            precisa_processar = (
                topo is not None and topo[0] < self._calcular_chave(self.start)
            ) or not self._quase_igual(self._rhs(self.start), self._g(self.start))
            if not precisa_processar:
                break
            if topo is None:
                break

            chave_antiga, vertice = self._remover_topo_valido()  # type: ignore[misc]
            chave_nova = self._calcular_chave(vertice)
            if chave_antiga < chave_nova:
                self._inserir_fila(vertice)
            elif self._g(vertice) > self._rhs(vertice):
                self.g[vertice] = self._rhs(vertice)
                for predecessor in self._vizinhos(vertice):
                    self._atualizar_vertice(predecessor)
            else:
                self.g[vertice] = math.inf
                self._atualizar_vertice(vertice)
                for predecessor in self._vizinhos(vertice):
                    self._atualizar_vertice(predecessor)

    def _extrair_caminho_celulas(self, inicio: Cell | None = None) -> list[Cell]:
        """Segue o sucessor de menor ``custo + g`` do start (ou inicio) ate o goal."""
        celula_inicial = self.start if inicio is None else inicio
        if celula_inicial is None or self.goal is None:
            return []
        atual = celula_inicial
        caminho = [atual]
        visitadas = {atual}
        while atual != self.goal:
            opcoes = [
                (self._custo(atual, vizinho) + self._g(vizinho), vizinho)
                for vizinho in self._vizinhos(atual)
            ]
            opcoes = [opcao for opcao in opcoes if math.isfinite(opcao[0])]
            if not opcoes:
                return []
            _, proximo = min(opcoes, key=lambda opcao: (opcao[0], opcao[1]))
            if proximo in visitadas:
                return []
            caminho.append(proximo)
            visitadas.add(proximo)
            atual = proximo
            if len(caminho) > self.rows * self.cols:
                return []
        return caminho

    def _linha_livre(self, origem: Cell, destino: Cell) -> bool:
        """Verifica linha de visada por todas as celulas cruzadas pelo segmento."""
        linha, coluna = origem
        destino_linha, destino_coluna = destino
        if self.ocupacao[origem] or self.ocupacao[destino]:
            return False

        delta_coluna = destino_coluna - coluna
        delta_linha = destino_linha - linha
        passo_coluna = 0 if delta_coluna == 0 else (1 if delta_coluna > 0 else -1)
        passo_linha = 0 if delta_linha == 0 else (1 if delta_linha > 0 else -1)
        t_delta_coluna = math.inf if delta_coluna == 0 else 1.0 / abs(delta_coluna)
        t_delta_linha = math.inf if delta_linha == 0 else 1.0 / abs(delta_linha)
        t_max_coluna = math.inf if delta_coluna == 0 else 0.5 / abs(delta_coluna)
        t_max_linha = math.inf if delta_linha == 0 else 0.5 / abs(delta_linha)

        while (linha, coluna) != destino:
            if math.isclose(t_max_coluna, t_max_linha, abs_tol=1e-12):
                # O segmento cruza um canto: ambos os lados precisam estar livres.
                lados = ((linha, coluna + passo_coluna), (linha + passo_linha, coluna))
                if any(
                    not self._celula_valida(lado) or self.ocupacao[lado]
                    for lado in lados
                ):
                    return False
                linha += passo_linha
                coluna += passo_coluna
                t_max_linha += t_delta_linha
                t_max_coluna += t_delta_coluna
            elif t_max_coluna < t_max_linha:
                coluna += passo_coluna
                t_max_coluna += t_delta_coluna
            else:
                linha += passo_linha
                t_max_linha += t_delta_linha
            if not self._celula_valida((linha, coluna)) or self.ocupacao[linha, coluna]:
                return False
        return True

    def _suavizar_caminho(self, caminho: list[Cell]) -> list[Cell]:
        """Remove zig-zags mantendo somente segmentos com linha de visada livre."""
        if len(caminho) <= 2:
            return caminho
        suavizado = [caminho[0]]
        indice_origem = 0
        while indice_origem < len(caminho) - 1:
            indice_destino = len(caminho) - 1
            while indice_destino > indice_origem + 1:
                if self._linha_livre(caminho[indice_origem], caminho[indice_destino]):
                    break
                indice_destino -= 1
            suavizado.append(caminho[indice_destino])
            indice_origem = indice_destino
        return suavizado


class HokuyoSensorSim:
    """Leitor do ``fastHokuyo`` modelado pelos dois vision sensors do template.

    Args:
        sim: Objeto da API remota do CoppeliaSim.
        base_name: Caminho absoluto do sensor, por exemplo
            ``"/PioneerP3DX[0]/fastHokuyo"``.
        is_range_data: Quando ``True`` (padrao), retorna ``N x 2`` com angulo
            em radianos e distancia em metros. Quando ``False``, retorna
            ``N x 3`` com pontos no referencial do Hokuyo.

    A classe nao descarta leituras por distancia nem associa pontos a robos.
    A interpretacao das leituras cabe a estrategia de navegacao do agente.
    """

    _vision_sensor_name_template = "{}/sensor{}"
    _angle_min = -120.0 * math.pi / 180.0
    _angle_increment = (240.0 / 684.0) * math.pi / 180.0

    def __init__(self, sim, base_name: str, is_range_data: bool = True) -> None:
        if "fastHokuyo" not in base_name:
            raise ValueError(
                "base_name deve conter 'fastHokuyo', por exemplo "
                "'/PioneerP3DX/fastHokuyo'"
            )
        self._sim = sim
        self._base_name = base_name
        self._is_range_data = bool(is_range_data)
        self._base_obj = sim.getObject(base_name)
        if self._base_obj == -1:
            raise ValueError(f"objeto Hokuyo invalido: {base_name}")

        self._vision_sensors_obj = [
            sim.getObject(self._vision_sensor_name_template.format(base_name, indice))
            for indice in (1, 2)
        ]
        if any(objeto == -1 for objeto in self._vision_sensors_obj):
            raise ValueError(f"vision sensor ausente abaixo de {base_name}")

    def get_is_range_data(self) -> bool:
        """Informa se ``getSensorData`` devolvera pares angulo-distancia."""
        return self._is_range_data

    def set_is_range_data(self, is_range_data: bool) -> None:
        """Seleciona a saida de range ``N x 2`` ou de pontos ``N x 3``."""
        self._is_range_data = bool(is_range_data)

    def getSensorData(self) -> np.ndarray:
        """Le dados brutos dos sensores e devolve uma matriz NumPy.

        Returns:
            ``float[N, 2]`` de ``(angulo_rad, distancia_m)`` para modo range,
            ou ``float[N, 3]`` de pontos no referencial do laser. Ausencia de
            leituras retorna uma matriz vazia com numero de colunas correto.
        """
        angulo = self._angle_min
        dados: list[list[float]] = []

        for vision_sensor in self._vision_sensors_obj:
            resposta = self._sim.readVisionSensor(vision_sensor)
            # A API remota pode retornar apenas o status inteiro enquanto o
            # vision sensor ainda nao publicou o pacote auxiliar.
            if not isinstance(resposta, (tuple, list)) or len(resposta) < 3:
                continue
            _, _, pacote = resposta
            if not pacote or len(pacote) < 2:
                continue
            largura, altura = int(pacote[0]), int(pacote[1])
            if largura <= 0 or altura <= 0:
                continue

            transformacao_relativa = None
            if not self._is_range_data:
                matriz_sensor = self._sim.getObjectMatrix(vision_sensor)
                matriz_base_inversa = self._sim.getMatrixInverse(
                    self._sim.getObjectMatrix(self._base_obj)
                )
                transformacao_relativa = self._sim.multiplyMatrices(
                    matriz_base_inversa, matriz_sensor
                )

            pontos_sensor: list[list[float]] = []
            for linha in range(altura):
                for coluna in range(largura):
                    indice = 2 + 4 * (linha * largura + coluna)
                    if indice + 3 >= len(pacote):
                        continue
                    leitura = pacote[indice : indice + 4]
                    angulo += self._angle_increment
                    if self._is_range_data:
                        dados.append([angulo, float(leitura[3])])
                    else:
                        pontos_sensor.append(
                            [float(leitura[0]), float(leitura[1]), float(leitura[2])]
                        )

            if pontos_sensor:
                assert transformacao_relativa is not None
                matriz = np.asarray(transformacao_relativa, dtype=float).reshape(3, 4)
                pontos = np.asarray(pontos_sensor, dtype=float)
                pontos_no_laser = pontos @ matriz[:, :3].T + matriz[:, 3]
                dados.extend(pontos_no_laser.tolist())

        colunas = 2 if self._is_range_data else 3
        return np.asarray(dados, dtype=float).reshape((-1, colunas))


class Sensoriamento:
    """Le pose mundial e laser cru do robo sem tomar decisoes reativas."""

    def __init__(self) -> None:
        self._hokuyos: dict[tuple[int, str], HokuyoSensorSim] = {}

    @staticmethod
    def _caminho_robo(robotname: str) -> str:
        """Normaliza nome relativo ou absoluto para um caminho CoppeliaSim."""
        nome = robotname.strip()
        if not nome:
            raise ValueError("robotname nao pode ser vazio")
        return "/" + nome.lstrip("/")

    def obter_pose(self, sim, robotname: str) -> tuple[float, float, float]:
        """Retorna pose mundial ``(x_m, y_m, theta_rad)`` do robo solicitado."""
        handle = sim.getObject(self._caminho_robo(robotname))
        if handle == -1:
            raise ValueError(f"robo nao encontrado: {robotname}")
        posicao = sim.getObjectPosition(handle, -1)
        orientacao = sim.getObjectOrientation(handle, -1)
        return float(posicao[0]), float(posicao[1]), float(orientacao[2])

    def obter_laser(self, sim, robotname: str) -> np.ndarray:
        """Retorna laser cru ``N x 2`` nas colunas ``(angulo_rad, distancia_m)``."""
        caminho = self._caminho_robo(robotname)
        chave = (id(sim), caminho)
        leitor = self._hokuyos.get(chave)
        if leitor is None:
            leitor = HokuyoSensorSim(sim, f"{caminho}/fastHokuyo", is_range_data=True)
            self._hokuyos[chave] = leitor
        return leitor.getSensorData()


class Controlador:
    """Converte vetor de velocidade no mundo em comando diferencial ``(v, w)``.

    O controlador aplica a transformacao de Desai et al. (1998). Ele nao sabe
    se o vetor recebido veio de um waypoint global, de campos potenciais ou da
    soma de ambos, portanto pode ser reutilizado por qualquer estrategia de
    navegacao que produza um vetor de velocidade no mundo.
    """

    def __init__(
        self,
        *,
        distancia_controle_m: float = 0.20,
        v_max_m_s: float = 0.65,
        w_max_rad_s: float = 1.80,
    ) -> None:
        if distancia_controle_m <= 0:
            raise ValueError("distancia_controle_m deve ser maior que zero")
        if v_max_m_s <= 0 or w_max_rad_s <= 0:
            raise ValueError("limites de velocidade devem ser maiores que zero")
        self.distancia_controle_m = float(distancia_controle_m)
        self.v_max_m_s = float(v_max_m_s)
        self.w_max_rad_s = float(w_max_rad_s)

    def controlar(
        self,
        pose_atual: tuple[float, float, float],
        vetor_xy_desejado: tuple[float, float],
    ) -> tuple[float, float]:
        """Transforma vetor mundial ``(vx, vy)`` em ``(v_m_s, w_rad_s)``.

        A equacao usada e ``v=cos(theta)vx+sin(theta)vy`` e
        ``w=(-sin(theta)vx+cos(theta)vy)/d``. A saida e saturada para evitar
        comandos irreais ao PioneerP3DX.
        """
        if len(pose_atual) < 3 or len(vetor_xy_desejado) != 2:
            raise ValueError("pose requer (x, y, theta) e vetor requer (vx, vy)")
        theta = float(pose_atual[2])
        vx, vy = float(vetor_xy_desejado[0]), float(vetor_xy_desejado[1])
        if not all(math.isfinite(valor) for valor in (theta, vx, vy)):
            raise ValueError("pose e vetor devem conter valores finitos")
        if math.hypot(vx, vy) < 1e-12:
            return 0.0, 0.0

        cosseno = math.cos(theta)
        seno = math.sin(theta)
        v = cosseno * vx + seno * vy
        w = (-seno * vx + cosseno * vy) / self.distancia_controle_m

        # Modulacao de velocidade em curvas para evitar subestercamento contra quinas
        angulo_desejado = math.atan2(vy, vx)
        diff_ang = abs(
            (angulo_desejado - theta + math.pi) % (2.0 * math.pi) - math.pi
        )
        if diff_ang > math.radians(20.0):
            fator_curva = max(0.12, math.cos(diff_ang) ** 2)
            v *= fator_curva

        return (
            float(np.clip(v, -self.v_max_m_s, self.v_max_m_s)),
            float(np.clip(w, -self.w_max_rad_s, self.w_max_rad_s)),
        )

    @staticmethod
    def para_velocidades_roda(v: float, w: float, L: float, r: float) -> tuple[float, float]:
        """Converte ``(v_m_s, w_rad_s)`` em ``(wl_rad_s, wr_rad_s)``.

        Args:
            v: Velocidade linear do centro do eixo.
            w: Velocidade angular do chassi.
            L: Distancia entre rodas, em metros.
            r: Raio de cada roda, em metros e estritamente positivo.
        """
        if not math.isfinite(float(v)) or not math.isfinite(float(w)):
            raise ValueError("v e w devem ser finitos")
        if not math.isfinite(float(L)) or L < 0:
            raise ValueError("L deve ser finito e nao negativo")
        if not math.isfinite(float(r)) or r <= 0:
            raise ValueError("r deve ser finito e maior que zero")
        wl = v / r - (w * L) / (2.0 * r)
        wr = v / r + (w * L) / (2.0 * r)
        return float(wl), float(wr)


class DetectorDeChegada:
    """Verifica chegada apenas pela distancia euclidiana no plano mundial."""

    def chegou(
        self,
        pose_atual: tuple[float, float, float],
        goal: WorldPoint,
        tol: float,
    ) -> bool:
        """Retorna ``True`` quando ``||(x, y) - goal|| <= tol``."""
        if len(pose_atual) < 2 or len(goal) != 2:
            raise ValueError("pose e goal devem conter coordenadas x e y")
        if not math.isfinite(float(tol)) or tol < 0:
            raise ValueError("tol deve ser finita e nao negativa")
        distancia = math.hypot(
            float(pose_atual[0]) - float(goal[0]),
            float(pose_atual[1]) - float(goal[1]),
        )
        tolerancia = float(tol)
        return distancia < tolerancia or math.isclose(
            distancia, tolerancia, rel_tol=0.0, abs_tol=1e-12
        )


@dataclass
class TelemetriaDecisao:
    """Registra o estado instantaneo das decisoes do planejador e camada reativa."""

    pose: tuple[float, float, float]
    caminho: list[WorldPoint]
    goal: WorldPoint
    lookahead: WorldPoint | None
    laser_bruto: np.ndarray
    laser_dinamico: np.ndarray
    v_att: tuple[float, float]
    f_rep: tuple[float, float]
    f_tan: tuple[float, float]
    f_sym: tuple[float, float]
    v_tot: tuple[float, float]
    v_desejado: tuple[float, float] | None
    estado_recuperacao: int
    v_linear: float = 0.0
    w_angular: float = 0.0
    wl: float = 0.0
    wr: float = 0.0


class CamadaReativa:
    """Camada reativa dinamica para desvio de obstaculos e navegacao local.

    Combina lookahead no caminho global D* Lite com repulsao por campos
    potenciais a partir do laser Hokuyo, deteccao e persistencia de obstaculos
    dinamicos, histerese/expiracao de celulas bloqueadas, quebra de simetria
    a direita e deteccao/recuperacao de estagnacao (recuo e giro).
    """

    ESTADO_NORMAL = 0
    ESTADO_RECUO = 1
    ESTADO_GIRO = 2

    def __init__(
        self,
        *,
        distancia_lookahead_m: float = 0.60,
        raio_influencia_laser_m: float = 0.90,
        distancia_saturacao_m: float = 0.32,
        ganho_repulsao: float = 0.35,
        ganho_tangencial: float = 0.30,
        ganho_simetria: float = 0.30,
        f_rep_max: float = 0.65,
        angulo_frontal_rad: float = math.radians(30.0),
        alcance_dinamico_m: float = 3.0,
        passos_persistencia: int = 3,
        passos_expiracao: int = 15,
        janela_estagnacao_passos: int = 20,
        limiar_deslocamento_estagnacao_m: float = 0.05,
        limiar_angulo_estagnacao_rad: float = 0.20,
        passos_recuo: int = 22,
        passos_giro: int = 15,
        velocidade_desejada_m_s: float = 0.50,
        distancia_controle_m: float = 0.20,
        tolerancia_goal_m: float = 0.10,
        raio_robo_m: float = 0.26,
        distancia_critica_parede_m: float = 0.35,
        distancia_gap_impassavel_m: float = 0.75,
    ) -> None:
        self.distancia_lookahead_m = float(distancia_lookahead_m)
        self.raio_influencia_laser_m = float(raio_influencia_laser_m)
        self.distancia_saturacao_m = float(distancia_saturacao_m)
        self.ganho_repulsao = float(ganho_repulsao)
        self.ganho_tangencial = float(ganho_tangencial)
        self.ganho_simetria = float(ganho_simetria)
        self.f_rep_max = float(f_rep_max)
        self.angulo_frontal_rad = float(angulo_frontal_rad)
        self.alcance_dinamico_m = float(alcance_dinamico_m)
        self.passos_persistencia = int(passos_persistencia)
        self.passos_expiracao = int(passos_expiracao)
        self.janela_estagnacao_passos = int(janela_estagnacao_passos)
        self.limiar_deslocamento_estagnacao_m = float(
            limiar_deslocamento_estagnacao_m
        )
        self.limiar_angulo_estagnacao_rad = float(limiar_angulo_estagnacao_rad)
        self.passos_recuo = int(passos_recuo)
        self.passos_giro = int(passos_giro)
        self.velocidade_desejada_m_s = float(velocidade_desejada_m_s)
        self.distancia_controle_m = float(distancia_controle_m)
        self.tolerancia_goal_m = float(tolerancia_goal_m)
        self.raio_robo_m = float(raio_robo_m)
        self.distancia_critica_parede_m = float(distancia_critica_parede_m)
        self.distancia_gap_impassavel_m = float(distancia_gap_impassavel_m)

        # Estado interno de rastreamento de obstaculos dinamicos
        self._passo_atual = 0
        self._contagens_persistencia: dict[Cell, int] = {}
        self._ultimo_passo_visto: dict[Cell, int] = {}
        self._celulas_bloqueadas: set[Cell] = set()

        # Estado interno de estagnacao e recuperacao
        self._estado_recuperacao = self.ESTADO_NORMAL
        self._passos_em_recuperacao = 0
        self._historico_poses: list[tuple[int, float, float, float]] = []

        # Telemetria para visualizacao e auditoria de decisoes
        self.ultima_telemetria: TelemetriaDecisao | None = None

    def resetar_recuperacao(self) -> None:
        """Reinicia o monitor de estagnacao e a maquina de recuperacao."""
        self._estado_recuperacao = self.ESTADO_NORMAL
        self._passos_em_recuperacao = 0
        self._historico_poses.clear()

    def resetar_obstaculos_dinamicos(
        self, planejador: PlanejadorGlobal | None = None
    ) -> None:
        """Libera todas as celulas dinamicamente bloqueadas e limpa historico."""
        if planejador is not None:
            if hasattr(planejador, "limpar_obstaculos_dinamicos"):
                planejador.limpar_obstaculos_dinamicos()
            elif self._celulas_bloqueadas:
                planejador.marcar_bloqueio(list(self._celulas_bloqueadas), False)
        self._celulas_bloqueadas.clear()
        self._contagens_persistencia.clear()
        self._ultimo_passo_visto.clear()

    def selecionar_lookahead(
        self,
        pose: tuple[float, float, float],
        caminho: list[WorldPoint],
        planejador: PlanejadorGlobal | None = None,
    ) -> WorldPoint | None:
        """Seleciona waypoint a frente no caminho global a distancia de lookahead.

        Se planejador for fornecido, verifica a linha de visada desobstruida
        para evitar que a atracao corte quinas de paredes ou obstaculos.
        """
        if not caminho:
            return None
        if len(caminho) == 1:
            return caminho[0]
        xr, yr = float(pose[0]), float(pose[1])
        idx_mais_proximo = min(
            range(len(caminho)),
            key=lambda i: math.hypot(caminho[i][0] - xr, caminho[i][1] - yr),
        )
        candidato_idx = len(caminho) - 1
        for i in range(idx_mais_proximo, len(caminho)):
            dist = math.hypot(caminho[i][0] - xr, caminho[i][1] - yr)
            if dist >= self.distancia_lookahead_m:
                candidato_idx = i
                break

        if planejador is not None:
            celula_robo = planejador.mundo_para_celula((xr, yr))
            if celula_robo is not None and not planejador.ocupacao[celula_robo]:
                for i in range(candidato_idx, idx_mais_proximo, -1):
                    celula_wp = planejador.mundo_para_celula(caminho[i])
                    if celula_wp is not None and planejador._linha_livre(celula_robo, celula_wp):
                        return caminho[i]

        return caminho[candidato_idx]

    def calcular_atracao(
        self, pose: tuple[float, float, float], waypoint: WorldPoint
    ) -> tuple[float, float]:
        """Calcula vetor atrativo mundial em direcao ao waypoint de lookahead."""
        delta_x = float(waypoint[0]) - float(pose[0])
        delta_y = float(waypoint[1]) - float(pose[1])
        dist = math.hypot(delta_x, delta_y)
        if dist < 1e-9:
            return 0.0, 0.0
        fator = self.velocidade_desejada_m_s / dist
        return delta_x * fator, delta_y * fator

    def calcular_repulsao(
        self, pose: tuple[float, float, float], laser: np.ndarray
    ) -> tuple[float, float]:
        """Calcula forca repulsiva resultante por campos potenciais."""
        if laser is None or len(laser) == 0:
            return 0.0, 0.0
        dados = np.asarray(laser, dtype=float)
        if dados.ndim != 2 or dados.shape[1] < 2:
            return 0.0, 0.0

        angulos = dados[:, 0]
        distancias = dados[:, 1]
        mask = (
            np.isfinite(angulos)
            & np.isfinite(distancias)
            & (distancias > 0.02)
            & (distancias < self.raio_influencia_laser_m)
        )
        if not np.any(mask):
            return 0.0, 0.0

        angulos_validos = angulos[mask]
        dist_validas = distancias[mask]

        # Agrupamento em setores angulares para evitar viés de contagem de feixes
        num_setores = 24
        ang_min, ang_max = -math.radians(120.0), math.radians(120.0)
        largura_setor = (ang_max - ang_min) / num_setores

        setor_indices = np.clip(
            np.floor((angulos_validos - ang_min) / largura_setor).astype(int),
            0,
            num_setores - 1,
        )

        theta = float(pose[2])
        fx_total, fy_total = 0.0, 0.0

        for s in range(num_setores):
            em_setor = setor_indices == s
            if not np.any(em_setor):
                continue
            idx_min = np.argmin(dist_validas[em_setor])
            d_min = float(dist_validas[em_setor][idx_min])
            ang_obs = float(angulos_validos[em_setor][idx_min])

            d_efetiva = max(d_min, self.distancia_saturacao_m)
            magnitude = (
                self.ganho_repulsao
                * (
                    (1.0 / d_efetiva)
                    - (1.0 / self.raio_influencia_laser_m)
                )
                * (1.0 / d_efetiva)
            )

            phi = theta + ang_obs
            fx_total += -magnitude * math.cos(phi)
            fy_total += -magnitude * math.sin(phi)

        norma = math.hypot(fx_total, fy_total)
        if norma > self.f_rep_max and norma > 1e-9:
            escala = self.f_rep_max / norma
            fx_total *= escala
            fy_total *= escala

        return fx_total, fy_total

    def filtrar_laser_dinamico(
        self,
        pose: tuple[float, float, float],
        laser: np.ndarray,
        planejador: PlanejadorGlobal | None = None,
        incluir_paredes_criticas: bool = False,
    ) -> np.ndarray:
        """Filtra leituras do laser mantendo apenas obstaculos dinamicos (e paredes criticas se solicitado).

        Se planejador for None, considera todas as leituras do laser.
        Se planejador for fornecido, descarta leituras cujas coordenadas mundiais
        coincidem com obstaculos estaticos (paredes do mapa original infladas),
        garantindo que o robo acredite no caminho D* para a geometria estatica e
        aplique campos potenciais exclusivamente contra obstaculos dinamicos.
        Se incluir_paredes_criticas for True, preserva leituras de paredes que
        estejam a distancia critica (dist <= distancia_critica_parede_m) para servir
        de buffer de seguranca contra colisoes de quina.
        """
        if laser is None or len(laser) == 0:
            return np.empty((0, 2), dtype=float)
        dados = np.asarray(laser, dtype=float)
        if dados.ndim != 2 or dados.shape[1] < 2:
            return np.empty((0, 2), dtype=float)
        if planejador is None:
            return dados

        angulos = dados[:, 0]
        distancias = dados[:, 1]
        xr, yr, theta = float(pose[0]), float(pose[1]), float(pose[2])

        indices_mantidos: list[int] = []
        for i, (ang, dist) in enumerate(zip(angulos, distancias)):
            if not (math.isfinite(ang) and math.isfinite(dist) and dist > 0.02):
                continue
            xw = xr + dist * math.cos(theta + ang)
            yw = yr + dist * math.sin(theta + ang)
            eh_estatico = planejador.eh_obstaculo_estatico((xw, yw))
            if not eh_estatico:
                indices_mantidos.append(i)
            elif incluir_paredes_criticas and dist <= self.distancia_critica_parede_m:
                indices_mantidos.append(i)

        if not indices_mantidos:
            return np.empty((0, 2), dtype=float)
        return dados[indices_mantidos]

    def filtrar_laser_combinado(
        self,
        pose: tuple[float, float, float],
        laser: np.ndarray,
        planejador: PlanejadorGlobal | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filtra o laser em uma unica passada, retornando (laser_dinamico, laser_reativo).

        Elimina a duplicacao de calculos trigonometricos e consultas estaticas.
        """
        if laser is None or len(laser) == 0:
            vazio = np.empty((0, 2), dtype=float)
            return vazio, vazio
        dados = np.asarray(laser, dtype=float)
        if dados.ndim != 2 or dados.shape[1] < 2:
            vazio = np.empty((0, 2), dtype=float)
            return vazio, vazio
        if planejador is None:
            return dados, dados

        angulos = dados[:, 0]
        distancias = dados[:, 1]
        xr, yr, theta = float(pose[0]), float(pose[1]), float(pose[2])
        d_crit = self.distancia_critica_parede_m

        indices_dinamicos: list[int] = []
        indices_reativos: list[int] = []
        for i, (ang, dist) in enumerate(zip(angulos, distancias)):
            if not (math.isfinite(ang) and math.isfinite(dist) and dist > 0.02):
                continue
            xw = xr + dist * math.cos(theta + ang)
            yw = yr + dist * math.sin(theta + ang)
            eh_estatico = planejador.eh_obstaculo_estatico((xw, yw))
            if not eh_estatico:
                indices_dinamicos.append(i)
                indices_reativos.append(i)
            elif dist <= d_crit:
                indices_reativos.append(i)

        laser_din = (
            dados[indices_dinamicos]
            if indices_dinamicos
            else np.empty((0, 2), dtype=float)
        )
        laser_reat = (
            dados[indices_reativos]
            if indices_reativos
            else np.empty((0, 2), dtype=float)
        )
        return laser_din, laser_reat

    def calcular_repulsao_tangencial(
        self,
        pose: tuple[float, float, float],
        laser: np.ndarray,
        vetor_atracao: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        """Calcula forca repulsiva tangencial (circulacao) em torno de obstaculos.

        A forca tangencial e perpendicular a forca repulsiva radial e atua para
        contornar obstaculos rapidamente. A direcao tangencial (horaria ou
        anti-horaria) e selecionada de modo a maximizar o alinhamento com o
        vetor de atracao ao goal/lookahead. Em caso de simetria (obstaculo
        frontal colinear), prioriza o desvio para a direita.
        """
        if laser is None or len(laser) == 0:
            return 0.0, 0.0
        dados = np.asarray(laser, dtype=float)
        if dados.ndim != 2 or dados.shape[1] < 2:
            return 0.0, 0.0

        angulos = dados[:, 0]
        distancias = dados[:, 1]
        mask = (
            np.isfinite(angulos)
            & np.isfinite(distancias)
            & (distancias > 0.02)
            & (distancias < self.raio_influencia_laser_m)
        )
        if not np.any(mask):
            return 0.0, 0.0

        angulos_validos = angulos[mask]
        dist_validas = distancias[mask]

        num_setores = 24
        ang_min, ang_max = -math.radians(120.0), math.radians(120.0)
        largura_setor = (ang_max - ang_min) / num_setores

        setor_indices = np.clip(
            np.floor((angulos_validos - ang_min) / largura_setor).astype(int),
            0,
            num_setores - 1,
        )

        theta = float(pose[2])
        fx_tan_total, fy_tan_total = 0.0, 0.0

        v_att_x = 0.0
        v_att_y = 0.0
        tem_atracao = False
        if vetor_atracao is not None and len(vetor_atracao) >= 2:
            v_att_x, v_att_y = float(vetor_atracao[0]), float(vetor_atracao[1])
            tem_atracao = math.hypot(v_att_x, v_att_y) > 1e-6

        for s in range(num_setores):
            em_setor = setor_indices == s
            if not np.any(em_setor):
                continue
            idx_min = np.argmin(dist_validas[em_setor])
            d_min = float(dist_validas[em_setor][idx_min])
            ang_obs = float(angulos_validos[em_setor][idx_min])

            d_efetiva = max(d_min, self.distancia_saturacao_m)
            magnitude = (
                self.ganho_tangencial
                * (
                    (1.0 / d_efetiva)
                    - (1.0 / self.raio_influencia_laser_m)
                )
                * (1.0 / d_efetiva)
            )

            phi = theta + ang_obs
            u_tan_dir_x = math.sin(phi)
            u_tan_dir_y = -math.cos(phi)

            u_tan_esq_x = -math.sin(phi)
            u_tan_esq_y = math.cos(phi)

            if tem_atracao:
                dot_dir = v_att_x * u_tan_dir_x + v_att_y * u_tan_dir_y
                dot_esq = v_att_x * u_tan_esq_x + v_att_y * u_tan_esq_y
                if dot_esq > dot_dir + 1e-9:
                    u_escolhido_x, u_escolhido_y = u_tan_esq_x, u_tan_esq_y
                else:
                    u_escolhido_x, u_escolhido_y = u_tan_dir_x, u_tan_dir_y
            else:
                u_escolhido_x, u_escolhido_y = u_tan_dir_x, u_tan_dir_y

            fx_tan_total += magnitude * u_escolhido_x
            fy_tan_total += magnitude * u_escolhido_y

        norma = math.hypot(fx_tan_total, fy_tan_total)
        if norma > self.f_rep_max and norma > 1e-9:
            escala = self.f_rep_max / norma
            fx_tan_total *= escala
            fy_tan_total *= escala

        return fx_tan_total, fy_tan_total

    def calcular_quebra_simetria(
        self, pose: tuple[float, float, float], laser: np.ndarray
    ) -> tuple[float, float]:
        """Aplica regra deterministica de desvio a direita em encontro frontal."""
        if laser is None or len(laser) == 0:
            return 0.0, 0.0
        dados = np.asarray(laser, dtype=float)
        if dados.ndim != 2 or dados.shape[1] < 2:
            return 0.0, 0.0

        angulos = dados[:, 0]
        distancias = dados[:, 1]
        mask_frontal = (
            np.isfinite(angulos)
            & np.isfinite(distancias)
            & (np.abs(angulos) <= self.angulo_frontal_rad)
            & (distancias > 0.02)
            & (distancias < self.raio_influencia_laser_m)
        )
        if not np.any(mask_frontal):
            return 0.0, 0.0

        d_frontal_min = float(np.min(distancias[mask_frontal]))
        peso = (
            self.raio_influencia_laser_m - d_frontal_min
        ) / self.raio_influencia_laser_m
        magnitude = self.ganho_simetria * peso

        theta = float(pose[2])
        # Vetor unitario a direita do chassi no referencial mundial
        u_direita_x = math.sin(theta)
        u_direita_y = -math.cos(theta)

        return magnitude * u_direita_x, magnitude * u_direita_y

    def processar_obstaculos_dinamicos(
        self,
        laser: np.ndarray,
        pose: tuple[float, float, float],
        planejador: PlanejadorGlobal,
    ) -> None:
        """Detecta obstaculos no espaco livre estatico com persistencia, vedacao de vaos e expiracao."""
        self._passo_atual += 1

        celulas_vistas: set[Cell] = set()
        if laser is not None and len(laser) > 0:
            dados = np.asarray(laser, dtype=float)
            if dados.ndim == 2 and dados.shape[1] >= 2:
                angulos = dados[:, 0]
                distancias = dados[:, 1]
                xr, yr, theta = float(pose[0]), float(pose[1]), float(pose[2])
                n_feixes = len(angulos)

                pontos_mundo: list[tuple[float, float]] = []
                eh_obstaculo: list[bool] = []
                eh_dinamico: list[bool] = []

                for ang, dist in zip(angulos, distancias):
                    if not (math.isfinite(ang) and math.isfinite(dist) and dist > 0.05):
                        pontos_mundo.append((math.nan, math.nan))
                        eh_obstaculo.append(False)
                        eh_dinamico.append(False)
                        continue
                    xw = xr + dist * math.cos(theta + ang)
                    yw = yr + dist * math.sin(theta + ang)
                    pontos_mundo.append((xw, yw))

                    no_alcance = dist <= self.alcance_dinamico_m
                    parede_estatica = (
                        hasattr(planejador, "eh_obstaculo_estatico")
                        and planejador.eh_obstaculo_estatico((xw, yw))
                    )
                    eh_obs = bool(no_alcance and math.isfinite(xw))
                    eh_obstaculo.append(eh_obs)

                    eh_din = bool(eh_obs and not parede_estatica)
                    eh_dinamico.append(eh_din)

                    if eh_din:
                        celula = planejador.mundo_para_celula((xw, yw))
                        if (
                            celula is not None
                            and celula != planejador.start
                            and celula != planejador.goal
                        ):
                            celulas_vistas.add(celula)

                # Deteccao e vedacao de vaos intransponiveis (Gap Bridging)
                ultimo_idx_obs: int | None = None
                for i in range(n_feixes):
                    if not eh_obstaculo[i]:
                        continue
                    if ultimo_idx_obs is not None and (i - ultimo_idx_obs > 1):
                        if eh_dinamico[ultimo_idx_obs] or eh_dinamico[i]:
                            d_ant = distancias[ultimo_idx_obs]
                            d_i = distancias[i]
                            min_d = min(d_ant, d_i)
                            tem_abertura = any(
                                (not math.isfinite(distancias[k]))
                                or (distancias[k] > min_d + 0.30)
                                for k in range(ultimo_idx_obs + 1, i)
                            )
                            if tem_abertura:
                                pA = pontos_mundo[ultimo_idx_obs]
                                pB = pontos_mundo[i]
                                d_gap = math.hypot(pA[0] - pB[0], pA[1] - pB[1])
                                if 0.10 < d_gap <= self.distancia_gap_impassavel_m:
                                    passos_ponte = max(
                                        2,
                                        int(
                                            math.ceil(
                                                d_gap
                                                / (planejador.tamanho_celula_m * 0.5)
                                            )
                                        ),
                                    )
                                    for s in range(1, passos_ponte):
                                        t = s / passos_ponte
                                        px = pA[0] + t * (pB[0] - pA[0])
                                        py = pA[1] + t * (pB[1] - pA[1])
                                        celula_ponte = planejador.mundo_para_celula(
                                            (px, py)
                                        )
                                        if (
                                            celula_ponte is not None
                                            and celula_ponte != planejador.start
                                            and celula_ponte != planejador.goal
                                        ):
                                            celulas_vistas.add(celula_ponte)
                    ultimo_idx_obs = i

        novas_bloqueadas: list[Cell] = []
        for celula in celulas_vistas:
            self._contagens_persistencia[celula] = (
                self._contagens_persistencia.get(celula, 0) + 1
            )
            self._ultimo_passo_visto[celula] = self._passo_atual
            if (
                self._contagens_persistencia[celula] >= self.passos_persistencia
                and celula not in self._celulas_bloqueadas
            ):
                novas_bloqueadas.append(celula)
                self._celulas_bloqueadas.add(celula)

        if novas_bloqueadas:
            planejador.marcar_bloqueio(novas_bloqueadas, True)

        novas_desbloqueadas: list[Cell] = []
        for celula in list(self._celulas_bloqueadas):
            ultimo_passo = self._ultimo_passo_visto.get(celula, 0)
            if self._passo_atual - ultimo_passo > self.passos_expiracao:
                novas_desbloqueadas.append(celula)
                self._celulas_bloqueadas.remove(celula)
                self._contagens_persistencia.pop(celula, None)
                self._ultimo_passo_visto.pop(celula, None)

        if novas_desbloqueadas:
            planejador.marcar_bloqueio(novas_desbloqueadas, False)

        for celula in list(self._contagens_persistencia.keys()):
            if celula not in self._celulas_bloqueadas:
                if self._passo_atual - self._ultimo_passo_visto.get(celula, 0) > 2:
                    self._contagens_persistencia.pop(celula, None)
                    self._ultimo_passo_visto.pop(celula, None)

    def monitorar_estagnacao(
        self, pose: tuple[float, float, float], distancia_ao_goal: float
    ) -> bool:
        """Detecta travamento sem confundir curva lenta e conduz recuperacao."""
        if self._estado_recuperacao == self.ESTADO_RECUO:
            self._passos_em_recuperacao += 1
            if self._passos_em_recuperacao >= self.passos_recuo:
                self._estado_recuperacao = self.ESTADO_GIRO
                self._passos_em_recuperacao = 0
            return True

        if self._estado_recuperacao == self.ESTADO_GIRO:
            self._passos_em_recuperacao += 1
            if self._passos_em_recuperacao >= self.passos_giro:
                self._estado_recuperacao = self.ESTADO_NORMAL
                self._passos_em_recuperacao = 0
                self._historico_poses.clear()
                return False
            return True

        xr, yr, theta = float(pose[0]), float(pose[1]), float(pose[2])
        self._historico_poses.append((self._passo_atual, xr, yr, theta))
        if len(self._historico_poses) > self.janela_estagnacao_passos:
            self._historico_poses.pop(0)

        if len(self._historico_poses) == self.janela_estagnacao_passos:
            _, x_antigo, y_antigo, theta_antigo = self._historico_poses[0]
            deslocamento = math.hypot(xr - x_antigo, yr - y_antigo)
            diff_ang = abs(
                (theta - theta_antigo + math.pi) % (2.0 * math.pi) - math.pi
            )

            if (
                deslocamento < self.limiar_deslocamento_estagnacao_m
                and diff_ang < self.limiar_angulo_estagnacao_rad
                and distancia_ao_goal > self.tolerancia_goal_m
            ):
                self._estado_recuperacao = self.ESTADO_RECUO
                self._passos_em_recuperacao = 0
                self._historico_poses.clear()
                return True

        return False

    def calcular_vetor_desejado(
        self,
        pose: tuple[float, float, float],
        caminho: list[WorldPoint],
        laser: np.ndarray,
        goal: WorldPoint,
        planejador: PlanejadorGlobal | None = None,
    ) -> tuple[float, float] | None:
        """Combina lookahead, repulsao radial/tangencial e quebra de simetria em vetor (vx, vy).

        Se planejador for informado, os campos potenciais e ajustes de velocidade
        atuam exclusivamente sobre obstaculos dinamicos, fazendo com que o robo
        siga fielmente o caminho D* calculado para os obstaculos estaticos.
        """
        xr, yr, theta = float(pose[0]), float(pose[1]), float(pose[2])
        dist_goal = math.hypot(goal[0] - xr, goal[1] - yr)

        em_recuperacao = self.monitorar_estagnacao(pose, dist_goal)
        if em_recuperacao:
            d_c = self.distancia_controle_m
            laser_din = self.filtrar_laser_dinamico(pose, laser, planejador)
            if self._estado_recuperacao == self.ESTADO_RECUO:
                v_rec = -0.25
                w_rec = -0.20
                vx = math.cos(theta) * v_rec - d_c * math.sin(theta) * w_rec
                vy = math.sin(theta) * v_rec + d_c * math.cos(theta) * w_rec
            elif self._estado_recuperacao == self.ESTADO_GIRO:
                v_giro = 0.0
                w_giro = -1.20
                vx = -d_c * math.sin(theta) * w_giro
                vy = d_c * math.cos(theta) * w_giro
            else:
                vx, vy = 0.0, 0.0

            self.ultima_telemetria = TelemetriaDecisao(
                pose=pose,
                caminho=list(caminho),
                goal=goal,
                lookahead=None,
                laser_bruto=(
                    np.asarray(laser, dtype=float)
                    if laser is not None and len(laser) > 0
                    else np.empty((0, 2), dtype=float)
                ),
                laser_dinamico=laser_din,
                v_att=(0.0, 0.0),
                f_rep=(0.0, 0.0),
                f_tan=(0.0, 0.0),
                f_sym=(0.0, 0.0),
                v_tot=(vx, vy),
                v_desejado=(vx, vy),
                estado_recuperacao=self._estado_recuperacao,
            )
            return vx, vy

        wp = self.selecionar_lookahead(pose, caminho, planejador=planejador)
        laser_bruto_arr = (
            np.asarray(laser, dtype=float)
            if laser is not None and len(laser) > 0
            else np.empty((0, 2), dtype=float)
        )
        if wp is None:
            laser_din = self.filtrar_laser_dinamico(pose, laser, planejador)
            if not caminho and dist_goal > self.tolerancia_goal_m:
                self._estado_recuperacao = self.ESTADO_RECUO
                self._passos_em_recuperacao = 0
                d_c = self.distancia_controle_m
                v_rec = -0.25
                w_rec = -0.20
                vx = math.cos(theta) * v_rec - d_c * math.sin(theta) * w_rec
                vy = math.sin(theta) * v_rec + d_c * math.cos(theta) * w_rec
                self.ultima_telemetria = TelemetriaDecisao(
                    pose=pose,
                    caminho=list(caminho),
                    goal=goal,
                    lookahead=None,
                    laser_bruto=laser_bruto_arr,
                    laser_dinamico=laser_din,
                    v_att=(0.0, 0.0),
                    f_rep=(0.0, 0.0),
                    f_tan=(0.0, 0.0),
                    f_sym=(0.0, 0.0),
                    v_tot=(vx, vy),
                    v_desejado=(vx, vy),
                    estado_recuperacao=self._estado_recuperacao,
                )
                return vx, vy
            self.ultima_telemetria = TelemetriaDecisao(
                pose=pose,
                caminho=list(caminho),
                goal=goal,
                lookahead=None,
                laser_bruto=laser_bruto_arr,
                laser_dinamico=laser_din,
                v_att=(0.0, 0.0),
                f_rep=(0.0, 0.0),
                f_tan=(0.0, 0.0),
                f_sym=(0.0, 0.0),
                v_tot=(0.0, 0.0),
                v_desejado=None,
                estado_recuperacao=self._estado_recuperacao,
            )
            return None

        # Filtra laser em passada unica para obter dinamico e reativo (com buffer critico)
        laser_dinamico, laser_reativo = self.filtrar_laser_combinado(
            pose, laser, planejador
        )

        v_att_x, v_att_y = self.calcular_atracao(pose, wp)
        f_rep_x, f_rep_y = self.calcular_repulsao(pose, laser_reativo)
        f_tan_x, f_tan_y = self.calcular_repulsao_tangencial(
            pose, laser_dinamico, (v_att_x, v_att_y)
        )
        f_sym_x, f_sym_y = self.calcular_quebra_simetria(pose, laser_dinamico)

        v_tot_x = v_att_x + f_rep_x + f_tan_x + f_sym_x
        v_tot_y = v_att_y + f_rep_y + f_tan_y + f_sym_y

        norma = math.hypot(v_tot_x, v_tot_y)
        if norma < 1e-9:
            self.ultima_telemetria = TelemetriaDecisao(
                pose=pose,
                caminho=list(caminho),
                goal=goal,
                lookahead=wp,
                laser_bruto=laser_bruto_arr,
                laser_dinamico=laser_dinamico,
                v_att=(v_att_x, v_att_y),
                f_rep=(f_rep_x, f_rep_y),
                f_tan=(f_tan_x, f_tan_y),
                f_sym=(f_sym_x, f_sym_y),
                v_tot=(v_tot_x, v_tot_y),
                v_desejado=None,
                estado_recuperacao=self._estado_recuperacao,
            )
            return None

        v_alvo = self.velocidade_desejada_m_s
        if laser_reativo is not None and len(laser_reativo) > 0:
            dados = np.asarray(laser_reativo, dtype=float)
            if dados.ndim == 2 and dados.shape[1] >= 2:
                ang = dados[:, 0]
                dst = dados[:, 1]
                mask_f = (
                    np.abs(ang) <= self.angulo_frontal_rad
                ) & (dst > 0.02) & np.isfinite(dst)
                if np.any(mask_f):
                    d_min_f = float(np.min(dst[mask_f]))
                    if d_min_f < 0.65:
                        v_alvo = max(
                            0.12, self.velocidade_desejada_m_s * (d_min_f / 0.65)
                        )

        escala = v_alvo / norma
        v_des_x = v_tot_x * escala
        v_des_y = v_tot_y * escala

        self.ultima_telemetria = TelemetriaDecisao(
            pose=pose,
            caminho=list(caminho),
            goal=goal,
            lookahead=wp,
            laser_bruto=laser_bruto_arr,
            laser_dinamico=laser_dinamico,
            v_att=(v_att_x, v_att_y),
            f_rep=(f_rep_x, f_rep_y),
            f_tan=(f_tan_x, f_tan_y),
            f_sym=(f_sym_x, f_sym_y),
            v_tot=(v_tot_x, v_tot_y),
            v_desejado=(v_des_x, v_des_y),
            estado_recuperacao=self._estado_recuperacao,
        )
        return v_des_x, v_des_y



class VisualizadorNavegacao:
    """Janela grafica interativa para visualizacao em tempo real do planejamento e telemetria."""

    def __init__(
        self,
        titulo: str = "Pioneer P3-DX - Visão e Decisões de Navegação em Tempo Real",
        tamanho_figura: tuple[float, float] = (14.0, 7.0),
        intervalo_passos: int = 1,
    ) -> None:
        import matplotlib

        try:
            if matplotlib.get_backend().lower() == "agg":
                matplotlib.use("TkAgg")
        except Exception:
            pass

        import matplotlib.gridspec as gridspec
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        self.plt = plt
        self.patches = patches
        self.titulo = titulo
        self.intervalo_passos = max(1, int(intervalo_passos))
        self._contador_passos = 0

        self.plt.ion()
        self.fig = self.plt.figure(figsize=tamanho_figura)
        try:
            self.fig.canvas.manager.set_window_title(self.titulo)
        except Exception:
            pass

        gs = gridspec.GridSpec(
            2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.3, 0.7]
        )
        self.ax_global = self.fig.add_subplot(gs[:, 0])
        self.ax_local = self.fig.add_subplot(gs[0, 1])
        self.ax_hud = self.fig.add_subplot(gs[1, 1])
        self.ax_hud.axis("off")

        self._im_global = None
        (self._linha_rota_global,) = self.ax_global.plot(
            [], [], "--", color="#ea580c", lw=1.8, alpha=0.9, label="Rota D* Lite"
        )
        (self._linha_trajetoria_global,) = self.ax_global.plot(
            [],
            [],
            "-",
            color="#2563eb",
            lw=2.2,
            alpha=0.85,
            label="Trajetória percorrida",
        )
        self._scatter_goals_pendentes = self.ax_global.scatter(
            [],
            [],
            s=110,
            marker="*",
            color="#ef4444",
            edgecolors="white",
            lw=0.8,
            zorder=6,
            label="Goals pendentes",
        )
        self._scatter_goals_alcancados = self.ax_global.scatter(
            [],
            [],
            s=120,
            marker="*",
            color="#16a34a",
            edgecolors="white",
            lw=0.8,
            zorder=6,
            label="Goals alcançados",
        )
        self._scatter_robo_global = self.ax_global.scatter(
            [],
            [],
            s=80,
            marker="o",
            color="#7c3aed",
            edgecolors="white",
            lw=1.0,
            zorder=7,
            label="Robô",
        )
        self._anotacoes_goals: list[object] = []

        self._im_local = None
        self._patch_casco = self.patches.Circle(
            (0, 0),
            radius=0.26,
            facecolor="#93c5fd",
            edgecolor="#1d4ed8",
            lw=1.5,
            alpha=0.65,
            zorder=4,
        )
        self._patch_margem = self.patches.Circle(
            (0, 0),
            radius=0.41,
            facecolor="none",
            edgecolor="#64748b",
            lw=1.2,
            linestyle="--",
            alpha=0.7,
            zorder=3,
        )
        self.ax_local.add_patch(self._patch_casco)
        self.ax_local.add_patch(self._patch_margem)

        (self._linha_rota_local,) = self.ax_local.plot(
            [], [], "--", color="#ea580c", lw=2.0, alpha=0.9, zorder=3
        )
        self._scatter_laser_estatico = self.ax_local.scatter(
            [], [], s=10, color="#94a3b8", alpha=0.5, zorder=2
        )
        self._scatter_laser_dinamico = self.ax_local.scatter(
            [], [], s=28, color="#ef4444", edgecolors="#fef08a", lw=0.6, zorder=5
        )
        self._scatter_lookahead = self.ax_local.scatter(
            [],
            [],
            s=90,
            marker="X",
            color="#06b6d4",
            edgecolors="black",
            lw=0.8,
            zorder=6,
        )

        self._q_att = self.ax_local.quiver(
            [0],
            [0],
            [0],
            [0],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#16a34a",
            width=0.012,
            zorder=7,
        )
        self._q_rep = self.ax_local.quiver(
            [0],
            [0],
            [0],
            [0],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#dc2626",
            width=0.012,
            zorder=7,
        )
        self._q_tan = self.ax_local.quiver(
            [0],
            [0],
            [0],
            [0],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#c026d3",
            width=0.012,
            zorder=7,
        )
        self._q_sym = self.ax_local.quiver(
            [0],
            [0],
            [0],
            [0],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#d97706",
            width=0.010,
            zorder=7,
        )
        self._q_tot = self.ax_local.quiver(
            [0],
            [0],
            [0],
            [0],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#1e3a8a",
            width=0.016,
            zorder=8,
        )
        self._q_orientacao = self.ax_local.quiver(
            [0],
            [0],
            [0],
            [0],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#1e293b",
            width=0.010,
            zorder=5,
        )

        self.ax_local.plot([], [], color="#16a34a", lw=2.0, label="v_att (Atração D*)")
        self.ax_local.plot(
            [], [], color="#dc2626", lw=2.0, label="F_rep (Repulsão Radial)"
        )
        self.ax_local.plot(
            [], [], color="#c026d3", lw=2.0, label="F_tan (Repulsão Tangencial)"
        )
        self.ax_local.plot(
            [], [], color="#d97706", lw=1.5, label="F_sym (Quebra Simetria)"
        )
        self.ax_local.plot(
            [], [], color="#1e3a8a", lw=2.5, label="v_des (Comando Resultante)"
        )
        self.ax_local.scatter(
            [], [], s=20, color="#ef4444", label="Laser Dinâmico"
        )
        self.ax_local.scatter(
            [], [], s=12, color="#94a3b8", label="Laser Parede Estática"
        )
        self.ax_local.legend(
            loc="upper right", fontsize=7.5, framealpha=0.9, facecolor="#ffffff"
        )

        self._hud_text = self.ax_hud.text(
            0.02,
            0.95,
            "Inicializando telemetria...",
            transform=self.ax_hud.transAxes,
            fontsize=8.0,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#0f172a",
                edgecolor="#334155",
            ),
            color="#f8fafc",
        )

        self.ax_global.set_title(
            "Visão Global do Robô — Mapa D* Lite & Trajetória",
            fontsize=10,
            weight="bold",
        )
        self.ax_global.set_xlabel("x (m)", fontsize=9)
        self.ax_global.set_ylabel("y (m)", fontsize=9)
        self.ax_global.grid(True, alpha=0.25)
        self.ax_global.legend(loc="upper right", fontsize=8, framealpha=0.9)

        self.ax_local.set_title(
            "Visão Local do Robô — Decisões & Forças Reativas",
            fontsize=10,
            weight="bold",
        )
        self.ax_local.set_xlabel("x (m)", fontsize=9)
        self.ax_local.set_ylabel("y (m)", fontsize=9)
        self.ax_local.grid(True, alpha=0.3)
        self.ax_local.set_aspect("equal", adjustable="box")

        self.plt.tight_layout()
        self.plt.show(block=False)
        self.fig.canvas.flush_events()

    def atualizar(
        self,
        *,
        telemetria: TelemetriaDecisao | None,
        planejador: PlanejadorGlobal | None,
        goals: list[tuple[float, float]],
        indice_goal: int,
        trajetoria: list[tuple[float, float, float]] | None = None,
    ) -> None:
        """Atualiza os graficos em tempo real com estado e telemetria atuais."""
        self._contador_passos += 1
        if self._contador_passos % self.intervalo_passos != 0:
            return
        if not self.plt.fignum_exists(self.fig.number):
            return
        if telemetria is None or planejador is None:
            return

        try:
            xr, yr, theta = telemetria.pose
            extent = (
                planejador.origem[0],
                planejador.origem[0] + planejador.largura_m,
                planejador.origem[1],
                planejador.origem[1] + planejador.altura_m,
            )

            rows, cols = planejador.rows, planejador.cols
            rgb_mapa = np.full((rows, cols, 3), 245, dtype=np.uint8)
            rgb_mapa[planejador.ocupacao_estatica] = [200, 205, 210]
            if hasattr(planejador, "ocupacao_base"):
                rgb_mapa[planejador.ocupacao_base] = [35, 40, 50]
            if hasattr(planejador, "ocupacao_dinamica"):
                rgb_mapa[planejador.ocupacao_dinamica] = [220, 38, 38]

            img_rgb_flip = np.flipud(rgb_mapa)

            if self._im_global is None:
                self._im_global = self.ax_global.imshow(
                    img_rgb_flip, extent=extent, origin="lower", zorder=1
                )
                self._im_local = self.ax_local.imshow(
                    img_rgb_flip, extent=extent, origin="lower", zorder=1
                )
                self.ax_global.set_xlim(extent[0], extent[1])
                self.ax_global.set_ylim(extent[2], extent[3])
                self.ax_global.set_aspect("equal", adjustable="box")
            else:
                self._im_global.set_data(img_rgb_flip)
                self._im_local.set_data(img_rgb_flip)

            if telemetria.caminho and len(telemetria.caminho) > 1:
                pts_caminho = np.asarray(telemetria.caminho)
                self._linha_rota_global.set_data(
                    pts_caminho[:, 0], pts_caminho[:, 1]
                )
                self._linha_rota_local.set_data(
                    pts_caminho[:, 0], pts_caminho[:, 1]
                )
            else:
                self._linha_rota_global.set_data([], [])
                self._linha_rota_local.set_data([], [])

            if trajetoria and len(trajetoria) > 1:
                pts = (
                    trajetoria[:: max(1, len(trajetoria) // 1000)]
                    if len(trajetoria) > 1000
                    else trajetoria
                )
                pts_traj = np.asarray([(x, y) for x, y, _ in pts])
                self._linha_trajetoria_global.set_data(
                    pts_traj[:, 0], pts_traj[:, 1]
                )

            pts_alcancados = goals[:indice_goal]
            pts_pendentes = goals[indice_goal:]
            if pts_alcancados:
                self._scatter_goals_alcancados.set_offsets(
                    np.asarray(pts_alcancados)
                )
            else:
                self._scatter_goals_alcancados.set_offsets(np.empty((0, 2)))

            if pts_pendentes:
                self._scatter_goals_pendentes.set_offsets(
                    np.asarray(pts_pendentes)
                )
            else:
                self._scatter_goals_pendentes.set_offsets(np.empty((0, 2)))

            if len(self._anotacoes_goals) != len(goals):
                for a in self._anotacoes_goals:
                    try:
                        a.remove()
                    except Exception:
                        pass
                self._anotacoes_goals.clear()
                for idx, (gx, gy) in enumerate(goals, start=1):
                    anotacao = self.ax_global.annotate(
                        f"G{idx}",
                        (gx, gy),
                        xytext=(5, 5),
                        textcoords="offset points",
                        color="#0f172a",
                        weight="bold",
                        fontsize=8.5,
                    )
                    self._anotacoes_goals.append(anotacao)

            self._scatter_robo_global.set_offsets([[xr, yr]])

            zoom = 2.5
            self.ax_local.set_xlim(xr - zoom, xr + zoom)
            self.ax_local.set_ylim(yr - zoom, yr + zoom)
            self._patch_casco.set_center((xr, yr))
            self._patch_margem.set_center((xr, yr))

            tam_orient = 0.35
            self._q_orientacao.set_offsets([[xr, yr]])
            self._q_orientacao.set_UVC(
                [tam_orient * math.cos(theta)], [tam_orient * math.sin(theta)]
            )

            pontos_estaticos: list[tuple[float, float]] = []
            pontos_dinamicos: list[tuple[float, float]] = []
            if len(telemetria.laser_bruto) > 0:
                for ang, dist in telemetria.laser_bruto:
                    if not (
                        math.isfinite(ang)
                        and math.isfinite(dist)
                        and 0.02 < dist < 4.5
                    ):
                        continue
                    xw = xr + dist * math.cos(theta + ang)
                    yw = yr + dist * math.sin(theta + ang)
                    if (
                        abs(xw - xr) <= zoom + 0.5
                        and abs(yw - yr) <= zoom + 0.5
                    ):
                        if (
                            hasattr(planejador, "eh_obstaculo_estatico")
                            and planejador.eh_obstaculo_estatico((xw, yw))
                        ):
                            pontos_estaticos.append((xw, yw))
                        else:
                            pontos_dinamicos.append((xw, yw))

            if pontos_estaticos:
                self._scatter_laser_estatico.set_offsets(
                    np.asarray(pontos_estaticos)
                )
            else:
                self._scatter_laser_estatico.set_offsets(np.empty((0, 2)))

            if pontos_dinamicos:
                self._scatter_laser_dinamico.set_offsets(
                    np.asarray(pontos_dinamicos)
                )
            else:
                self._scatter_laser_dinamico.set_offsets(np.empty((0, 2)))

            if telemetria.lookahead is not None:
                self._scatter_lookahead.set_offsets(
                    [[telemetria.lookahead[0], telemetria.lookahead[1]]]
                )
            else:
                self._scatter_lookahead.set_offsets(np.empty((0, 2)))

            self._q_att.set_offsets([[xr, yr]])
            self._q_att.set_UVC([telemetria.v_att[0]], [telemetria.v_att[1]])

            self._q_rep.set_offsets([[xr, yr]])
            self._q_rep.set_UVC([telemetria.f_rep[0]], [telemetria.f_rep[1]])

            self._q_tan.set_offsets([[xr, yr]])
            self._q_tan.set_UVC([telemetria.f_tan[0]], [telemetria.f_tan[1]])

            self._q_sym.set_offsets([[xr, yr]])
            self._q_sym.set_UVC([telemetria.f_sym[0]], [telemetria.f_sym[1]])

            v_des_x, v_des_y = (
                telemetria.v_desejado
                if telemetria.v_desejado is not None
                else (0.0, 0.0)
            )
            self._q_tot.set_offsets([[xr, yr]])
            self._q_tot.set_UVC([v_des_x], [v_des_y])

            nomes_estados = {
                0: "NORMAL (Seguindo D* + APF Dinâmico)",
                1: "⚠️ RECUO ATIVO (Recuperação de Estagnação)",
                2: "⚠️ GIRO ATIVO (Alinhamento de Rota)",
            }
            nome_estado = nomes_estados.get(
                telemetria.estado_recuperacao, "DESCONHECIDO"
            )
            dist_meta = math.hypot(
                telemetria.goal[0] - xr, telemetria.goal[1] - yr
            )

            norm_att = math.hypot(*telemetria.v_att)
            norm_rep = math.hypot(*telemetria.f_rep)
            norm_tan = math.hypot(*telemetria.f_tan)
            norm_sym = math.hypot(*telemetria.f_sym)
            norm_tot = math.hypot(v_des_x, v_des_y)

            num_din_laser = len(pontos_dinamicos)
            num_bloq_grid = (
                int(np.count_nonzero(planejador.ocupacao_dinamica))
                if hasattr(planejador, "ocupacao_dinamica")
                else 0
            )
            num_wp = len(telemetria.caminho)

            texto = (
                f"╔═══════════════════ PAINEL DE DECISÕES & TELEMETRIA ═══════════════════╗\n"
                f"║ Estado:         {nome_estado:<48} ║\n"
                f"║ Meta Ativa:     G{indice_goal + 1}/{len(goals)} em ({telemetria.goal[0]:.2f}, {telemetria.goal[1]:.2f}) m | Distância: {dist_meta:.2f} m{'':<10} ║\n"
                f"║ Velocidades:    v = {telemetria.v_linear:+.2f} m/s | w = {telemetria.w_angular:+.2f} rad/s (wl={telemetria.wl:+.1f}, wr={telemetria.wr:+.1f}){'':<4} ║\n"
                f"╠─────────────────── FORÇAS DA CAMADA REATIVA ──────────────────────────╣\n"
                f"║ • Atração D* (v_att):        {norm_att:.2f} m/s [verde]{'':<26} ║\n"
                f"║ • Repulsão Radial (F_rep):   {norm_rep:.2f} m/s [vermelho - obst. dinâmicos]{'':<10} ║\n"
                f"║ • Repulsão Tangencial (F_tan):{norm_tan:.2f} m/s [magenta - circulação rápida]{'':<8} ║\n"
                f"║ • Quebra Simetria (F_sym):   {norm_sym:.2f} m/s [ouro - desvio à direita]{'':<13} ║\n"
                f"║ • Vetor Resultante (v_des):  {norm_tot:.2f} m/s [azul escuro]{'':<23} ║\n"
                f"╠─────────────────── ESTADO DO MAPA & SENSORES ─────────────────────────╣\n"
                f"║ • Feixes Laser Dinâmicos:    {num_din_laser} pontos detectados no campo local{'':<13} ║\n"
                f"║ • Células Bloqueadas D*:     {num_bloq_grid} células dinâmicas ativas no grid{'':<12} ║\n"
                f"║ • Caminho D* Lite:           {num_wp} waypoints até o destino{'':<21} ║\n"
                f"╚════════════════════════════════════════════════════════════════════════╝"
            )
            self._hud_text.set_text(texto)

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.plt.pause(0.001)
        except Exception:
            pass

    def fechar(self) -> None:
        """Fecha a janela do visualizador com seguranca."""
        try:
            if self.plt.fignum_exists(self.fig.number):
                self.plt.close(self.fig)
        except Exception:
            pass


__all__ = [
    "Agent",
    "CamadaReativa",
    "Cell",
    "Controlador",
    "DetectorDeChegada",
    "HokuyoSensorSim",
    "PlanejadorGlobal",
    "Sensoriamento",
    "TelemetriaDecisao",
    "VisualizadorNavegacao",
    "WorldPoint",
]


class Agent:
    """Executa a navegacao deliberativa e reativa de um PioneerP3DX.

    A classe mantem estado entre ciclos para planejar a rota D* Lite,
    desviar reativamente de obstaculos com base no sensor laser Hokuyo,
    detectar e contornar obstaculos dinamicos, converter o vetor de
    velocidade em comandos de roda e avancar para os proximos destinos.
    """

    MAPA_LARGURA_M = 30.0
    MAPA_ALTURA_M = 30.0
    MAPA_ORIGEM = (-15.0, -15.0)
    TAMANHO_CELULA_M = 0.10
    RAIO_ROBO_M = 0.26
    MARGEM_SEGURANCA_M = 0.15
    TOLERANCIA_GOAL_M = 0.10
    VELOCIDADE_DESEJADA_M_S = 0.50
    LOOKAHEAD_M = 0.60

    def __init__(self, id_robo: int, *, visualizar: bool = False):
        """Inicializa o agente sem alterar a assinatura exigida pelo TP2.

        Args:
            id_robo: Identificador um-baseado recebido pelo servidor. O objeto
                CoppeliaSim correspondente e ``PioneerP3DX[id_robo - 1]``.
            visualizar: Se ``True``, abre janela em tempo real exibindo mapa D*,
                visão ego local, vetores de força reativa e telemetria. Pode
                também ser habilitado via variável de ambiente ``TP2_VISUALIZAR=1``
                ou pelo método ``agente.habilitar_visualizacao()``.
        """
        self.id = id_robo
        self.robotname = f"PioneerP3DX[{self.id - 1}]"
        self.L = 0.381
        self.r = 0.0975

        self._planejador: PlanejadorGlobal | None = None
        self._sensoriamento = Sensoriamento()
        self._controlador = Controlador()
        self._detector_chegada = DetectorDeChegada()
        self._camada_reativa = CamadaReativa(
            distancia_lookahead_m=self.LOOKAHEAD_M,
            velocidade_desejada_m_s=self.VELOCIDADE_DESEJADA_M_S,
            distancia_controle_m=self._controlador.distancia_controle_m,
            tolerancia_goal_m=self.TOLERANCIA_GOAL_M,
            raio_robo_m=self.RAIO_ROBO_M,
        )
        self._assinatura_goals: tuple[tuple[float, float], ...] | None = None
        self._indice_goal = 0
        self._goal_planejado: tuple[float, float] | None = None
        self._historico_trajetoria: list[tuple[float, float, float]] = []

        visualizar_env = os.environ.get("TP2_VISUALIZAR", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._visualizacao_ativa = bool(visualizar or visualizar_env)
        self._visualizador: VisualizadorNavegacao | None = None
        if self._visualizacao_ativa:
            self.habilitar_visualizacao(True)

    def habilitar_visualizacao(
        self,
        ativo: bool = True,
        *,
        titulo: str | None = None,
        intervalo_passos: int = 1,
    ) -> None:
        """Habilita ou desabilita a janela de visualizacao grafica em tempo real."""
        self._visualizacao_ativa = bool(ativo)
        if self._visualizacao_ativa:
            if self._visualizador is None:
                nome_janela = (
                    titulo
                    or f"Pioneer P3-DX #{self.id} — Navegação & Decisões em Tempo Real"
                )
                self._visualizador = VisualizadorNavegacao(
                    titulo=nome_janela, intervalo_passos=intervalo_passos
                )
        else:
            self.fechar_visualizacao()

    def fechar_visualizacao(self) -> None:
        """Fecha a janela grafica do visualizador se estiver aberta."""
        if self._visualizador is not None:
            self._visualizador.fechar()
            self._visualizador = None
        self._visualizacao_ativa = False

    def movement(self, sim, img, goal_list):
        """Executa um ciclo e informa se o goal corrente foi alcancado."""
        goals = self._normalizar_goals(goal_list)
        if not goals:
            self._parar_rodas(sim)
            return False

        self._inicializar_planejador(img)
        assert self._planejador is not None
        self._sincronizar_sequencia_goals(goals)
        if self._indice_goal >= len(goals):
            self._parar_rodas(sim)
            return False

        goal_atual = goals[self._indice_goal]
        if goal_atual != self._goal_planejado:
            self._planejador.definir_goal(goal_atual)
            self._goal_planejado = goal_atual

        pose = self._sensoriamento.obter_pose(sim, self.robotname)
        self._historico_trajetoria.append(pose)

        if self._detector_chegada.chegou(
            pose, goal_atual, tol=self.TOLERANCIA_GOAL_M
        ):
            self._parar_rodas(sim)
            self._indice_goal += 1
            self._goal_planejado = None
            self._camada_reativa.resetar_recuperacao()
            if self._indice_goal < len(goals):
                proximo_goal = goals[self._indice_goal]
                self._planejador.definir_goal(proximo_goal)
                self._goal_planejado = proximo_goal
            if self._visualizacao_ativa and self._visualizador is not None:
                self._visualizador.atualizar(
                    telemetria=self._camada_reativa.ultima_telemetria,
                    planejador=self._planejador,
                    goals=list(goals),
                    indice_goal=self._indice_goal,
                    trajetoria=self._historico_trajetoria,
                )
            return True

        laser = self._sensoriamento.obter_laser(sim, self.robotname)
        self._camada_reativa.processar_obstaculos_dinamicos(
            laser, pose, self._planejador
        )

        self._planejador.atualizar(pose)
        caminho = self._planejador.caminho_completo(pose)

        vetor = self._camada_reativa.calcular_vetor_desejado(
            pose, caminho, laser, goal_atual, planejador=self._planejador
        )
        if vetor is None:
            self._parar_rodas(sim)
            if self._visualizacao_ativa and self._visualizador is not None:
                self._visualizador.atualizar(
                    telemetria=self._camada_reativa.ultima_telemetria,
                    planejador=self._planejador,
                    goals=list(goals),
                    indice_goal=self._indice_goal,
                    trajetoria=self._historico_trajetoria,
                )
            return False

        v, w = self._controlador.controlar(pose, vetor)
        wl, wr = self._controlador.para_velocidades_roda(v, w, self.L, self.r)
        self._enviar_velocidades_roda(sim, wl, wr)

        if self._camada_reativa.ultima_telemetria is not None:
            self._camada_reativa.ultima_telemetria.v_linear = float(v)
            self._camada_reativa.ultima_telemetria.w_angular = float(w)
            self._camada_reativa.ultima_telemetria.wl = float(wl)
            self._camada_reativa.ultima_telemetria.wr = float(wr)

        if self._visualizacao_ativa and self._visualizador is not None:
            self._visualizador.atualizar(
                telemetria=self._camada_reativa.ultima_telemetria,
                planejador=self._planejador,
                goals=list(goals),
                indice_goal=self._indice_goal,
                trajetoria=self._historico_trajetoria,
            )

        return False

    def _inicializar_planejador(self, img: np.ndarray) -> None:
        """Constroi a grade apenas na primeira chamada de movement."""
        if self._planejador is not None:
            return
        imagem = np.asarray(img)
        if imagem.ndim < 2 or imagem.shape[0] == 0 or imagem.shape[1] == 0:
            raise ValueError("img deve conter altura e largura positivas")

        escala = (
            self.MAPA_LARGURA_M / imagem.shape[1],
            self.MAPA_ALTURA_M / imagem.shape[0],
        )
        self._planejador = PlanejadorGlobal(
            imagem,
            escala=escala,
            origem=self.MAPA_ORIGEM,
            tamanho_celula_m=self.TAMANHO_CELULA_M,
            raio_robo_m=self.RAIO_ROBO_M,
            margem_seguranca_m=self.MARGEM_SEGURANCA_M,
        )

    def _sincronizar_sequencia_goals(
        self, goals: tuple[tuple[float, float], ...]
    ) -> None:
        """Reinicia o indice somente quando o servidor entrega lista nova."""
        if goals != self._assinatura_goals:
            self._assinatura_goals = goals
            self._indice_goal = 0
            self._goal_planejado = None
            self._camada_reativa.resetar_recuperacao()

    @staticmethod
    def _normalizar_goals(
        goal_list: Iterable[Iterable[float]],
    ) -> tuple[tuple[float, float], ...]:
        """Converte goals de entrada para tuplas finitas de coordenadas."""
        goals: list[tuple[float, float]] = []
        for indice, goal in enumerate(goal_list):
            try:
                x, y = float(goal[0]), float(goal[1])  # type: ignore[index]
            except (IndexError, TypeError, ValueError) as erro:
                raise ValueError(
                    f"goal_list[{indice}] deve conter coordenadas (x, y)"
                ) from erro
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"goal_list[{indice}] deve conter valores finitos")
            goals.append((x, y))
        return tuple(goals)

    @staticmethod
    def _primeiro_waypoint_a_frente(
        caminho: list[tuple[float, float]]
    ) -> tuple[float, float] | None:
        """Seleciona o proximo waypoint da rota global."""
        if not caminho:
            return None
        return caminho[1] if len(caminho) > 1 else caminho[0]

    def _vetor_para_waypoint(
        self,
        pose: tuple[float, float, float],
        waypoint: tuple[float, float],
    ) -> tuple[float, float] | None:
        """Retorna vetor mundial de modulo controlado, ou None se nulo."""
        delta_x = waypoint[0] - pose[0]
        delta_y = waypoint[1] - pose[1]
        norma = math.hypot(delta_x, delta_y)
        if norma < 1e-12:
            return None
        escala = self.VELOCIDADE_DESEJADA_M_S / norma
        return delta_x * escala, delta_y * escala

    def _parar_rodas(self, sim) -> None:
        """Envia velocidade zero aos dois motores do robo deste agente."""
        self._enviar_velocidades_roda(sim, 0.0, 0.0)

    def _enviar_velocidades_roda(self, sim, wl: float, wr: float) -> None:
        """Envia velocidades angulares aos motores esquerdo e direito."""
        motor_esquerdo = sim.getObject(f"/{self.robotname}/leftMotor")
        motor_direito = sim.getObject(f"/{self.robotname}/rightMotor")
        if motor_esquerdo == -1 or motor_direito == -1:
            raise ValueError(f"motores nao encontrados para {self.robotname}")
        sim.setJointTargetVelocity(motor_esquerdo, float(wl))
        sim.setJointTargetVelocity(motor_direito, float(wr))
