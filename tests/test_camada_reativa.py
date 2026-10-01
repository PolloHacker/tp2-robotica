from __future__ import annotations

import math
import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

# Adiciona o diretorio src ao path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Agente1 import (
    Agent,
    CamadaReativa,
    Controlador,
    DetectorDeChegada,
    PlanejadorGlobal,
    Sensoriamento,
    TelemetriaDecisao,
    VisualizadorNavegacao,
)


class TestPotencialRepulsao(unittest.TestCase):
    """Testes para o calculo de repulsao por campos potenciais a partir do laser."""

    def setUp(self) -> None:
        self.camada = CamadaReativa(
            raio_influencia_laser_m=0.90,
            distancia_saturacao_m=0.32,
            ganho_repulsao=0.35,
            f_rep_max=0.65,
        )

    def test_sem_leituras_laser(self) -> None:
        """Sem laser ou com laser vazio, a forca repulsiva deve ser nula."""
        pose = (0.0, 0.0, 0.0)
        fx, fy = self.camada.calcular_repulsao(pose, np.empty((0, 2)))
        self.assertEqual(fx, 0.0)
        self.assertEqual(fy, 0.0)

        fx_none, fy_none = self.camada.calcular_repulsao(pose, None)  # type: ignore[arg-type]
        self.assertEqual(fx_none, 0.0)
        self.assertEqual(fy_none, 0.0)

    def test_obstaculo_alem_raio_influencia(self) -> None:
        """Obstaculos fora do raio de influencia (>= 0.9m) nao devem gerar forca."""
        pose = (0.0, 0.0, 0.0)
        # Laser com leituras a 1.5m e 3.0m
        laser = np.array([[0.0, 1.5], [math.radians(30), 2.0], [-math.radians(30), 3.0]])
        fx, fy = self.camada.calcular_repulsao(pose, laser)
        self.assertEqual(fx, 0.0)
        self.assertEqual(fy, 0.0)

    def test_repulsao_frontal(self) -> None:
        """Obstaculo frontal a 0.5m deve empurrar o robo para tras (direcao -x)."""
        pose = (0.0, 0.0, 0.0)  # Robo virado para +x
        laser = np.array([[0.0, 0.50]])
        fx, fy = self.camada.calcular_repulsao(pose, laser)
        # Como o obstaculo esta a frente em +x, a repulsa deve ter componente x negativo e y nulo
        self.assertLess(fx, -0.05)
        self.assertAlmostEqual(fy, 0.0, places=6)

    def test_repulsao_lateral_esquerda(self) -> None:
        """Obstaculo a esquerda (+45 graus) deve empurrar para tras e para a direita (-y)."""
        pose = (0.0, 0.0, 0.0)  # Robo virado para +x
        laser = np.array([[math.radians(45.0), 0.50]])
        fx, fy = self.camada.calcular_repulsao(pose, laser)
        self.assertLess(fx, 0.0)  # Empurra para tras
        self.assertLess(fy, 0.0)  # Empurra para a direita (-y)

    def test_repulsao_lateral_direita(self) -> None:
        """Obstaculo a direita (-45 graus) deve empurrar para tras e para a esquerda (+y)."""
        pose = (0.0, 0.0, 0.0)  # Robo virado para +x
        laser = np.array([[-math.radians(45.0), 0.50]])
        fx, fy = self.camada.calcular_repulsao(pose, laser)
        self.assertLess(fx, 0.0)  # Empurra para tras
        self.assertGreater(fy, 0.0)  # Empurra para a esquerda (+y)

    def test_saturacao_muito_proximo(self) -> None:
        """Obstaculo extremamente proximo nao deve gerar forca infinita nem NaN; deve saturar em f_rep_max."""
        pose = (0.0, 0.0, 0.0)
        # Laser com distancia quase nula
        laser = np.array([[0.0, 0.05]])
        fx, fy = self.camada.calcular_repulsao(pose, laser)
        self.assertTrue(math.isfinite(fx))
        self.assertTrue(math.isfinite(fy))
        norma = math.hypot(fx, fy)
        self.assertLessEqual(norma, self.camada.f_rep_max + 1e-6)
        self.assertGreater(norma, 0.1)


class TestQuebraDeSimetria(unittest.TestCase):
    """Testes para a regra deterministica de quebra de simetria em encontros frontais."""

    def setUp(self) -> None:
        self.camada = CamadaReativa(
            raio_influencia_laser_m=0.90,
            ganho_simetria=0.30,
            angulo_frontal_rad=math.radians(30.0),
        )

    def test_encontro_frontal_prioriza_direita(self) -> None:
        """Obstaculo frontal direto a 0.4m deve produzir forca para a direita do robo."""
        # Robo com theta = 0 (apontando para +x); direita e -y
        pose = (0.0, 0.0, 0.0)
        laser = np.array([[0.0, 0.40]])
        fx, fy = self.camada.calcular_quebra_simetria(pose, laser)
        self.assertAlmostEqual(fx, 0.0, places=6)
        self.assertLess(fy, -0.05)  # Componente para a direita (-y)

    def test_encontro_frontal_orientacao_diferente(self) -> None:
        """Quando o robo esta com theta = pi/2 (apontando para +y), a direita e +x."""
        pose = (0.0, 0.0, math.pi / 2.0)
        laser = np.array([[0.0, 0.40]])
        fx, fy = self.camada.calcular_quebra_simetria(pose, laser)
        self.assertGreater(fx, 0.05)  # Componente para a direita (+x)
        self.assertAlmostEqual(fy, 0.0, places=6)

    def test_obstaculo_fora_do_cone_frontal(self) -> None:
        """Obstaculo lateral (fora dos 30 graus frontais) nao deve ativar quebra de simetria."""
        pose = (0.0, 0.0, 0.0)
        laser = np.array([[math.radians(60.0), 0.40]])
        fx, fy = self.camada.calcular_quebra_simetria(pose, laser)
        self.assertEqual(fx, 0.0)
        self.assertEqual(fy, 0.0)


class TestLookaheadEAtracao(unittest.TestCase):
    """Testes para selecao de lookahead no caminho global e vetor de atracao."""

    def setUp(self) -> None:
        self.camada = CamadaReativa(
            distancia_lookahead_m=0.60,
            velocidade_desejada_m_s=0.50,
        )

    def test_selecao_lookahead(self) -> None:
        """Verifica que o waypoint escolhido fica a pelo menos lookahead de distancia a frente."""
        pose = (0.0, 0.0, 0.0)
        # Caminho ao longo do eixo x: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
        caminho = [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (0.6, 0.0), (0.8, 0.0), (1.0, 0.0)]
        wp = self.camada.selecionar_lookahead(pose, caminho)
        self.assertIsNotNone(wp)
        self.assertAlmostEqual(wp[0], 0.6, places=5)
        self.assertAlmostEqual(wp[1], 0.0, places=5)

    def test_lookahead_proximo_ao_goal(self) -> None:
        """Quando o robo esta a menos que lookahead do goal, o lookahead deve ser o proprio goal."""
        pose = (0.8, 0.0, 0.0)
        caminho = [(0.0, 0.0), (0.5, 0.0), (0.8, 0.0), (1.0, 0.0)]
        wp = self.camada.selecionar_lookahead(pose, caminho)
        self.assertEqual(wp, (1.0, 0.0))

    def test_caminho_vazio(self) -> None:
        """Caminho vazio deve retornar None."""
        pose = (0.0, 0.0, 0.0)
        self.assertIsNone(self.camada.selecionar_lookahead(pose, []))

    def test_calculo_atracao(self) -> None:
        """Calcula vetor unitario escalado para a velocidade desejada."""
        pose = (0.0, 0.0, 0.0)
        wp = (1.0, 0.0)
        vx, vy = self.camada.calcular_atracao(pose, wp)
        self.assertAlmostEqual(vx, 0.50, places=5)
        self.assertAlmostEqual(vy, 0.0, places=5)


class TestDeteccaoObstaculosDinamicos(unittest.TestCase):
    """Testes para deteccao, persistencia, bloqueio e expiracao (histerese) de celulas."""

    def setUp(self) -> None:
        # Cria um mapa sintético livre de 50x50 pixels (5.0m x 5.0m)
        img_branca = np.ones((50, 50), dtype=float)  # Todo branco = livre
        self.planejador = PlanejadorGlobal(
            img_branca,
            escala=0.10,
            origem=(-2.5, -2.5),
            tamanho_celula_m=0.10,
            raio_robo_m=0.26,
            margem_seguranca_m=0.05,
        )
        self.camada = CamadaReativa(
            passos_persistencia=3,
            passos_expiracao=5,
            alcance_dinamico_m=3.0,
        )

    def test_persistencia_necessaria_para_bloqueio(self) -> None:
        """Uma leitura isolada (ruido) nao deve bloquear a celula. Somente apos 3 passos consecutivos."""
        pose = (0.0, 0.0, 0.0)
        # Laser detecta obstaculo a 1.0m na frente (x=1.0, y=0.0)
        laser = np.array([[0.0, 1.0]])
        celula_esperada = self.planejador.mundo_para_celula((1.0, 0.0))
        self.assertIsNotNone(celula_esperada)

        # Passo 1: detectado 1 vez
        self.camada.processar_obstaculos_dinamicos(laser, pose, self.planejador)
        self.assertFalse(self.planejador.ocupacao_dinamica[celula_esperada])
        self.assertNotIn(celula_esperada, self.camada._celulas_bloqueadas)

        # Passo 2: detectado 2 vezes
        self.camada.processar_obstaculos_dinamicos(laser, pose, self.planejador)
        self.assertFalse(self.planejador.ocupacao_dinamica[celula_esperada])

        # Passo 3: detectado 3 vezes (atinge limiar_persistencia = 3)
        self.camada.processar_obstaculos_dinamicos(laser, pose, self.planejador)
        self.assertTrue(self.planejador.ocupacao_dinamica[celula_esperada])
        self.assertIn(celula_esperada, self.camada._celulas_bloqueadas)

    def test_expiracao_e_desbloqueio(self) -> None:
        """Apos bloqueado, se o obstaculo deixar de ser observado por mais de passos_expiracao, a celula deve ser desbloqueada."""
        pose = (0.0, 0.0, 0.0)
        laser = np.array([[0.0, 1.0]])
        celula = self.planejador.mundo_para_celula((1.0, 0.0))

        # Passos 1 a 3 para atingir persistencia
        for _ in range(3):
            self.camada.processar_obstaculos_dinamicos(laser, pose, self.planejador)
        self.assertTrue(self.planejador.ocupacao_dinamica[celula])

        # Agora o obstaculo desaparece (laser sem deteccao na celula)
        laser_vazio = np.empty((0, 2))
        for _ in range(self.camada.passos_expiracao):
            self.camada.processar_obstaculos_dinamicos(laser_vazio, pose, self.planejador)
            # Durante os passos de histerese ainda permanece bloqueado
            self.assertTrue(self.planejador.ocupacao_dinamica[celula])

        # Mais um passo excedendo passos_expiracao: deve ser desbloqueado!
        self.camada.processar_obstaculos_dinamicos(laser_vazio, pose, self.planejador)
        self.assertFalse(self.planejador.ocupacao_dinamica[celula])
        self.assertNotIn(celula, self.camada._celulas_bloqueadas)

    def test_replanejamento_apos_bloqueio_e_desbloqueio(self) -> None:
        """Bloqueio persistente forca o D* Lite a desviar, e o desbloqueio por expiracao restaura a rota original."""
        pose = (-1.0, 0.0, 0.0)
        goal = (1.0, 0.0)
        self.planejador.definir_goal(goal)
        caminho_inicial = self.planejador.caminho_completo(pose)
        self.assertTrue(len(caminho_inicial) > 0)

        # Celula intermediaria na rota direta
        celula_meio = self.planejador.mundo_para_celula((0.0, 0.0))
        self.assertIsNotNone(celula_meio)

        # O laser detecta um obstaculo dinamico a 1.0m de distancia a frente (em 0.0, 0.0)
        laser = np.array([[0.0, 1.0]])
        for _ in range(self.camada.passos_persistencia):
            self.camada.processar_obstaculos_dinamicos(laser, pose, self.planejador)

        # Verifica que a celula foi bloqueada
        self.assertTrue(self.planejador.ocupacao_dinamica[celula_meio])
        # Atualiza o planejador com a nova pose e gera novo caminho
        self.planejador.atualizar(pose)
        caminho_desvio = self.planejador.caminho_completo(pose)
        self.assertTrue(len(caminho_desvio) > 0)

        # Assegura que nenhuma coordenada do caminho de desvio esta na celula bloqueada
        for pt in caminho_desvio[:-1]:  # Exceto o goal final
            c = self.planejador.mundo_para_celula(pt)
            self.assertNotEqual(c, celula_meio)

        # Agora o obstaculo expira
        laser_vazio = np.empty((0, 2))
        for _ in range(self.camada.passos_expiracao + 1):
            self.camada.processar_obstaculos_dinamicos(laser_vazio, pose, self.planejador)

        self.assertFalse(self.planejador.ocupacao_dinamica[celula_meio])
        self.planejador.atualizar(pose)
        caminho_restaurado = self.planejador.caminho_completo(pose)
        self.assertTrue(len(caminho_restaurado) > 0)


class TestEstagnacaoERecuperacao(unittest.TestCase):
    """Testes para deteccao de estagnacao e maquina de estados de recuperacao (recuo e giro)."""

    def setUp(self) -> None:
        self.camada = CamadaReativa(
            janela_estagnacao_passos=5,
            limiar_deslocamento_estagnacao_m=0.05,
            limiar_angulo_estagnacao_rad=0.20,
            passos_recuo=3,
            passos_giro=3,
            distancia_controle_m=0.20,
        )

    def test_deteccao_estagnacao_e_recuperacao(self) -> None:
        """Robo parado no mesmo ponto por 5 ciclos longe do goal deve acionar recuperacao."""
        pose = (0.0, 0.0, 0.0)
        goal = (5.0, 5.0)

        # Ciclos 1 a 4: ainda enchendo a janela
        for _ in range(4):
            vetor = self.camada.calcular_vetor_desejado(pose, [(1.0, 0.0)], np.empty((0, 2)), goal)
            self.assertEqual(self.camada._estado_recuperacao, CamadaReativa.ESTADO_NORMAL)

        # Ciclo 5: janela cheia com deslocamento nulo -> entra em ESTADO_RECUO
        vetor_recuo = self.camada.calcular_vetor_desejado(pose, [(1.0, 0.0)], np.empty((0, 2)), goal)
        self.assertEqual(self.camada._estado_recuperacao, CamadaReativa.ESTADO_RECUO)
        self.assertIsNotNone(vetor_recuo)
        # No recuo com v=-0.20 e theta=0, vx deve ser negativo
        self.assertLess(vetor_recuo[0], 0.0)

        # Executa passos de recuo
        for _ in range(self.camada.passos_recuo - 1):
            self.camada.calcular_vetor_desejado(pose, [(1.0, 0.0)], np.empty((0, 2)), goal)
            self.assertEqual(self.camada._estado_recuperacao, CamadaReativa.ESTADO_RECUO)

        # Proximo ciclo deve transicionar para ESTADO_GIRO
        self.camada.calcular_vetor_desejado(pose, [(1.0, 0.0)], np.empty((0, 2)), goal)
        self.assertEqual(self.camada._estado_recuperacao, CamadaReativa.ESTADO_GIRO)

        # Executa passos de giro
        for _ in range(self.camada.passos_giro):
            self.camada.calcular_vetor_desejado(pose, [(1.0, 0.0)], np.empty((0, 2)), goal)

        # Apos concluir o giro, retorna ao ESTADO_NORMAL
        self.assertEqual(self.camada._estado_recuperacao, CamadaReativa.ESTADO_NORMAL)

    def test_nao_confunde_curva_ou_giro_com_estagnacao(self) -> None:
        """Se o robo estiver mudando a orientacao (fazendo curva ou giro no eixo), nao deve considerar travamento."""
        goal = (5.0, 5.0)
        # Robo na mesma posicao x, y mas girando 0.3 rad a cada ciclo
        for i in range(10):
            pose = (0.0, 0.0, i * 0.30)
            vetor = self.camada.calcular_vetor_desejado(pose, [(1.0, 0.0)], np.empty((0, 2)), goal)
            self.assertEqual(self.camada._estado_recuperacao, CamadaReativa.ESTADO_NORMAL)

    def test_nao_ativa_estagnacao_se_estiver_no_goal(self) -> None:
        """Se o robo estiver na vizinhanca imediata do goal, nao deve acionar estagnacao."""
        pose = (0.0, 0.0, 0.0)
        goal = (0.05, 0.0)  # Menor que tolerancia_goal_m (0.10)
        for _ in range(10):
            vetor = self.camada.calcular_vetor_desejado(pose, [goal], np.empty((0, 2)), goal)
            self.assertEqual(self.camada._estado_recuperacao, CamadaReativa.ESTADO_NORMAL)


class TestCombinacaoVetorDesejado(unittest.TestCase):
    """Testes para a combinacao de lookahead, atracao, repulsao e quebra de simetria em um unico vetor (vx, vy)."""

    def setUp(self) -> None:
        self.camada = CamadaReativa(
            distancia_lookahead_m=0.60,
            velocidade_desejada_m_s=0.50,
            raio_influencia_laser_m=0.90,
            ganho_repulsao=0.35,
            ganho_simetria=0.30,
        )

    def test_combinacao_livre_segue_lookahead(self) -> None:
        """Sem obstaculos, o vetor desejado deve ser puramente o vetor atrativo em direcao ao lookahead."""
        pose = (0.0, 0.0, 0.0)
        caminho = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        goal = (2.0, 0.0)
        vetor = self.camada.calcular_vetor_desejado(pose, caminho, np.empty((0, 2)), goal)
        self.assertIsNotNone(vetor)
        self.assertAlmostEqual(vetor[0], 0.50, places=5)
        self.assertAlmostEqual(vetor[1], 0.0, places=5)

    def test_combinacao_com_obstaculo_lateral_esquerda(self) -> None:
        """Com obstaculo a esquerda, o vetor resultante deve ser desviado para a direita (vy < 0)."""
        pose = (0.0, 0.0, 0.0)
        caminho = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        goal = (2.0, 0.0)
        laser = np.array([[math.radians(45.0), 0.50]])
        vetor = self.camada.calcular_vetor_desejado(pose, caminho, laser, goal)
        self.assertIsNotNone(vetor)
        # Deve manter componente de avanco (vx > 0) e desviar para a direita (vy < 0)
        self.assertGreater(vetor[0], 0.0)
        self.assertLess(vetor[1], -0.05)
        # Modulo deve respeitar a velocidade desejada
        self.assertAlmostEqual(math.hypot(vetor[0], vetor[1]), 0.50, places=4)

    def test_combinacao_frontal_aplica_desvio_direita(self) -> None:
        """Com obstaculo frontal, a quebra de simetria adiciona componente para a direita (vy < 0)."""
        pose = (0.0, 0.0, 0.0)
        caminho = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        goal = (2.0, 0.0)
        laser = np.array([[0.0, 0.40]])
        vetor = self.camada.calcular_vetor_desejado(pose, caminho, laser, goal)
        self.assertIsNotNone(vetor)
        # Deve desviar para a direita (vy < 0)
        self.assertLess(vetor[1], -0.05)

    def test_modulacao_velocidade_obstaculo_muito_proximo(self) -> None:
        """Quando o obstaculo frontal esta muito proximo (< 0.5m), a velocidade resultante deve ser reduzida."""
        pose = (0.0, 0.0, 0.0)
        caminho = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        goal = (2.0, 0.0)
        laser = np.array([[0.0, 0.25]])
        vetor = self.camada.calcular_vetor_desejado(pose, caminho, laser, goal)
        self.assertIsNotNone(vetor)
        norma = math.hypot(vetor[0], vetor[1])
        # A velocidade alvo deve ser reduzida em relacao a velocidade_desejada_m_s (0.50)
        self.assertLess(norma, 0.50)
        self.assertGreaterEqual(norma, 0.18 - 1e-5)


class TestIntegracaoAgent(unittest.TestCase):
    def test_assinaturas_exigidas_pelo_template(self) -> None:
        self.assertEqual(list(inspect.signature(Agent.__init__).parameters), ["self", "id_robo"])
        self.assertEqual(
            list(inspect.signature(Agent.movement).parameters),
            ["self", "sim", "img", "goal_list"],
        )

    """Testes de integracao com a classe Agent e o ciclo movement."""

    def test_inicializacao_agente(self) -> None:
        """Agent deve instanciar a CamadaReativa com os parametros corretos."""
        agente = Agent(id_robo=1)
        self.assertEqual(agente.id, 1)
        self.assertEqual(agente.robotname, "PioneerP3DX[0]")
        self.assertIsInstance(agente._camada_reativa, CamadaReativa)
        self.assertEqual(agente._camada_reativa.distancia_lookahead_m, agente.LOOKAHEAD_M)

    def test_ciclo_movement_com_mock(self) -> None:
        """Testa o pipeline completo de movement com objetos simulados."""
        agente = Agent(id_robo=1)
        sim_mock = MagicMock()

        # Mock de pose e motores
        sim_mock.getObject.return_value = 100
        sim_mock.getObjectPosition.return_value = [-0.32, -0.62, 0.15]
        sim_mock.getObjectOrientation.return_value = [0.0, 0.0, 0.0]
        # Mock de laser retornando status vazio (sem colisoes no momento)
        sim_mock.readVisionSensor.return_value = 0

        # Imagem sintetica de 50x50 pixels branca (livre)
        img = np.ones((50, 50), dtype=float)
        goals = [(1.0, 1.0), (2.0, 2.0)]

        # Executa um ciclo de movement
        chegou = agente.movement(sim_mock, img, goals)
        self.assertFalse(chegou)

        # Verifica se as velocidades de roda foram enviadas aos motores
        self.assertEqual(sim_mock.setJointTargetVelocity.call_count, 2)

    def test_chegada_ao_goal_retorna_true(self) -> None:
        """Quando a pose esta dentro da tolerancia do goal, movement deve retornar True."""
        agente = Agent(id_robo=1)
        sim_mock = MagicMock()
        sim_mock.getObject.return_value = 100
        # Pose exatamente no goal (1.0, 1.0)
        sim_mock.getObjectPosition.return_value = [1.0, 1.0, 0.15]
        sim_mock.getObjectOrientation.return_value = [0.0, 0.0, 0.0]
        sim_mock.readVisionSensor.return_value = 0

        img = np.ones((50, 50), dtype=float)
        goals = [(1.0, 1.0), (2.0, 2.0)]

        chegou = agente.movement(sim_mock, img, goals)
        self.assertTrue(chegou)
        # Velocidades zeradas na parada
        sim_mock.setJointTargetVelocity.assert_any_call(100, 0.0)


class TestRepulsaoTangencial(unittest.TestCase):
    """Testes para o calculo de forca repulsiva tangencial (circulacao)."""

    def setUp(self) -> None:
        self.camada = CamadaReativa(
            raio_influencia_laser_m=0.90,
            distancia_saturacao_m=0.32,
            ganho_repulsao=0.35,
            ganho_tangencial=0.30,
            f_rep_max=0.65,
        )

    def test_sem_leituras_laser(self) -> None:
        """Sem laser ou com laser vazio, a forca tangencial deve ser nula."""
        pose = (0.0, 0.0, 0.0)
        fx, fy = self.camada.calcular_repulsao_tangencial(pose, np.empty((0, 2)))
        self.assertEqual(fx, 0.0)
        self.assertEqual(fy, 0.0)

        fx_none, fy_none = self.camada.calcular_repulsao_tangencial(pose, None)  # type: ignore[arg-type]
        self.assertEqual(fx_none, 0.0)
        self.assertEqual(fy_none, 0.0)

    def test_obstaculo_alem_raio_influencia(self) -> None:
        """Obstaculos fora do raio de influencia nao devem gerar forca tangencial."""
        pose = (0.0, 0.0, 0.0)
        laser = np.array([[0.0, 1.5], [math.radians(30), 2.0]])
        fx, fy = self.camada.calcular_repulsao_tangencial(pose, laser)
        self.assertEqual(fx, 0.0)
        self.assertEqual(fy, 0.0)

    def test_obstaculo_frontal_desvia_direita(self) -> None:
        """Obstaculo frontal direto com goal a frente deve produzir forca tangencial para a direita (-y)."""
        pose = (0.0, 0.0, 0.0)  # Robo virado para +x
        laser = np.array([[0.0, 0.50]])
        v_att = (0.50, 0.0)
        fx, fy = self.camada.calcular_repulsao_tangencial(pose, laser, v_att)
        self.assertAlmostEqual(fx, 0.0, places=6)
        self.assertLess(fy, -0.05)  # Componente para a direita (-y)

    def test_obstaculo_lateral_esquerda_circula_direita(self) -> None:
        """Obstaculo a esquerda (+45 graus) com goal a frente deve empurrar para frente e direita."""
        pose = (0.0, 0.0, 0.0)
        laser = np.array([[math.radians(45.0), 0.50]])
        v_att = (0.50, 0.0)
        fx, fy = self.camada.calcular_repulsao_tangencial(pose, laser, v_att)
        self.assertGreater(fx, 0.0)  # Avanco para frente (+x)
        self.assertLess(fy, -0.05)  # Curvatura para a direita (-y)

    def test_obstaculo_lateral_direita_circula_esquerda(self) -> None:
        """Obstaculo a direita (-45 graus) com goal a frente deve empurrar para frente e esquerda."""
        pose = (0.0, 0.0, 0.0)
        laser = np.array([[-math.radians(45.0), 0.50]])
        v_att = (0.50, 0.0)
        fx, fy = self.camada.calcular_repulsao_tangencial(pose, laser, v_att)
        self.assertGreater(fx, 0.0)  # Avanco para frente (+x)
        self.assertGreater(fy, 0.05)  # Curvatura para a esquerda (+y)

    def test_saturacao_f_rep_max(self) -> None:
        """Obstaculo muito proximo deve ter forca tangencial finita saturada em f_rep_max."""
        pose = (0.0, 0.0, 0.0)
        laser = np.array([[0.0, 0.05]])
        fx, fy = self.camada.calcular_repulsao_tangencial(pose, laser)
        self.assertTrue(math.isfinite(fx))
        self.assertTrue(math.isfinite(fy))
        norma = math.hypot(fx, fy)
        self.assertLessEqual(norma, self.camada.f_rep_max + 1e-6)
        self.assertGreater(norma, 0.1)


class TestFiltroEstaticoEAcreditarDStar(unittest.TestCase):
    """Testes para filtragem de obstaculos estaticos e preservacao fiel do caminho D*."""

    def setUp(self) -> None:
        # Cria imagem com uma parede vertical em x entre 1.0 e 1.5m
        # Grid 50x50, 0.1m/pixel, escala 0.1, origem (-2.5, -2.5)
        # Linha 0 topo -> y=2.5, Linha 49 base -> y=-2.5
        # Colunas 0..49 -> x=-2.5..2.5. Parede em x=1.0..1.5 -> colunas 35..40
        self.img = np.ones((50, 50), dtype=float)
        self.img[:, 35:40] = 0.0  # Parede vertical preta
        self.planejador = PlanejadorGlobal(
            self.img,
            escala=0.10,
            origem=(-2.5, -2.5),
            tamanho_celula_m=0.10,
            raio_robo_m=0.26,
            margem_seguranca_m=0.15,
        )
        self.camada = CamadaReativa(
            distancia_lookahead_m=0.60,
            velocidade_desejada_m_s=0.50,
            raio_influencia_laser_m=0.90,
        )

    def test_reconhecimento_obstaculo_estatico(self) -> None:
        """Verifica que pontos sobre a parede sao identificados como estaticos e espaco livre como dinamico."""
        # Ponto sobre a parede em x=1.2, y=0.0
        self.assertTrue(self.planejador.eh_obstaculo_estatico((1.2, 0.0)))
        # Ponto livre em x=0.0, y=0.0
        self.assertFalse(self.planejador.eh_obstaculo_estatico((0.0, 0.0)))
        # Ponto fora do mapa
        self.assertTrue(self.planejador.eh_obstaculo_estatico((10.0, 10.0)))

    def test_filtrar_laser_descarta_parede_estatica(self) -> None:
        """Leituras do laser que incidem sobre paredes estaticas sao filtradas."""
        pose = (0.5, 0.0, 0.0)  # A 0.7m da parede que comeca em x=1.2
        # Laser apontando para a parede: dist 0.7m, angulo 0
        laser = np.array([[0.0, 0.70]])
        laser_dinamico = self.camada.filtrar_laser_dinamico(pose, laser, self.planejador)
        self.assertEqual(len(laser_dinamico), 0)

    def test_filtrar_laser_preserva_obstaculo_dinamico(self) -> None:
        """Leituras do laser em espaco livre (outro robo) sao mantidas."""
        pose = (0.0, 0.0, 0.0)
        # Laser detectando outro robo em x=0.5, y=0.0 (espaco livre)
        laser = np.array([[0.0, 0.50]])
        laser_dinamico = self.camada.filtrar_laser_dinamico(pose, laser, self.planejador)
        self.assertEqual(len(laser_dinamico), 1)
        self.assertAlmostEqual(laser_dinamico[0, 1], 0.50)

    def test_robo_acredita_no_caminho_dstar_junto_a_parede(self) -> None:
        """Com apenas paredes estaticas por perto, o vetor desejado segue fielmente o lookahead D* sem repulsao da parede."""
        pose = (0.5, 0.0, math.pi / 2.0)  # Virado para o norte (+y), com a parede a sua direita (+x)
        # Caminho paralelo a parede indo para o norte: (0.5, 0.0) -> (0.5, 0.6) -> (0.5, 1.2)
        caminho = [(0.5, 0.0), (0.5, 0.6), (0.5, 1.2)]
        goal = (0.5, 1.2)
        # Laser detecta a parede a direita (em x=1.25, distancia ~0.75m a -90 graus)
        laser_parede = np.array([[-math.pi / 2.0, 0.75]])

        # Com planejador informado: deve acreditar no D* e nao sofrer repulsa da parede
        vetor = self.camada.calcular_vetor_desejado(pose, caminho, laser_parede, goal, self.planejador)
        self.assertIsNotNone(vetor)
        # O lookahead esta em (0.5, 0.6), portanto o vetor deve apontar puramente para +y
        self.assertAlmostEqual(vetor[0], 0.0, places=4)
        self.assertAlmostEqual(vetor[1], 0.50, places=4)

    def test_desvio_ocorre_somente_para_obstaculo_dinamico(self) -> None:
        """Quando surge um obstaculo dinamico no caminho, o vetor desvia; para a parede estatica, nao."""
        pose = (0.5, 0.0, math.pi / 2.0)  # Virado para +y
        caminho = [(0.5, 0.0), (0.5, 0.6), (0.5, 1.2)]
        goal = (0.5, 1.2)

        # Laser com duas leituras: parede estatica a direita e obstaculo dinamico a frente
        # Parede estatica a x=1.25 (a direita do robo, angulo -90 deg no referencial do laser)
        # Obstaculo dinamico a frente em (0.5, 0.5) (distancia 0.5m, angulo 0 deg)
        laser_misto = np.array([
            [-math.pi / 2.0, 0.75],  # Parede estatica
            [0.0, 0.50],            # Obstaculo dinamico frontal
        ])

        vetor = self.camada.calcular_vetor_desejado(pose, caminho, laser_misto, goal, self.planejador)
        self.assertIsNotNone(vetor)
        # O obstaculo dinamico frontal deve causar desvio a direita (que e +x para theta = pi/2)
        self.assertGreater(vetor[0], 0.05)


class TestTelemetriaEVisualizador(unittest.TestCase):
    """Testes para o registro de telemetria e o visualizador em tempo real."""

    def setUp(self) -> None:
        self.camada = CamadaReativa(
            distancia_lookahead_m=0.60,
            raio_influencia_laser_m=0.90,
            velocidade_desejada_m_s=0.50,
        )
        img = np.ones((50, 50))
        self.planejador = PlanejadorGlobal(
            img, escala=0.10, origem=(-2.5, -2.5), tamanho_celula_m=0.10
        )
        self.planejador.definir_goal((2.0, 2.0))

    def test_telemetria_preenchida_apos_calculo_vetor(self) -> None:
        """Apos calcular o vetor desejado, a camada reativa deve registrar TelemetriaDecisao."""
        pose = (0.0, 0.0, 0.0)
        caminho = [(0.0, 0.0), (0.6, 0.0), (1.2, 0.0)]
        goal = (1.2, 0.0)
        laser = np.array([[0.0, 0.50]])

        vetor = self.camada.calcular_vetor_desejado(
            pose, caminho, laser, goal, self.planejador
        )
        self.assertIsNotNone(vetor)
        self.assertIsNotNone(self.camada.ultima_telemetria)

        tele = self.camada.ultima_telemetria
        self.assertIsInstance(tele, TelemetriaDecisao)
        self.assertEqual(tele.pose, pose)
        self.assertEqual(tele.goal, goal)
        self.assertIsNotNone(tele.lookahead)
        self.assertIsNotNone(tele.v_desejado)
        self.assertEqual(tele.estado_recuperacao, CamadaReativa.ESTADO_NORMAL)
        self.assertGreater(math.hypot(*tele.v_att), 0.0)
        self.assertGreater(math.hypot(*tele.f_rep), 0.0)

    def test_telemetria_em_recuperacao(self) -> None:
        """Em estado de recuperacao, telemetria deve indicar estado != ESTADO_NORMAL."""
        self.camada._estado_recuperacao = CamadaReativa.ESTADO_RECUO
        pose = (0.0, 0.0, 0.0)
        caminho = [(0.6, 0.0)]
        goal = (2.0, 0.0)

        vetor = self.camada.calcular_vetor_desejado(
            pose, caminho, np.empty((0, 2)), goal, self.planejador
        )
        self.assertIsNotNone(vetor)
        self.assertIsNotNone(self.camada.ultima_telemetria)
        self.assertEqual(
            self.camada.ultima_telemetria.estado_recuperacao,
            CamadaReativa.ESTADO_RECUO,
        )

    def test_visualizador_ciclo_vida_e_atualizacao(self) -> None:
        """Visualizador deve inicializar, atualizar e fechar sem erros."""
        viz = VisualizadorNavegacao(titulo="Teste Unitario Visualizador")
        try:
            pose = (0.0, 0.0, 0.0)
            caminho = [(0.0, 0.0), (0.6, 0.0), (1.2, 0.0)]
            goal = (1.2, 0.0)
            laser = np.array([[0.0, 0.50], [math.pi / 2, 1.0]])

            self.camada.calcular_vetor_desejado(
                pose, caminho, laser, goal, self.planejador
            )
            tele = self.camada.ultima_telemetria
            self.assertIsNotNone(tele)

            # Atualiza o visualizador com dados validos
            viz.atualizar(
                telemetria=tele,
                planejador=self.planejador,
                goals=[goal, (2.0, 2.0)],
                indice_goal=0,
                trajetoria=[(0.0, 0.0, 0.0)],
            )
        finally:
            viz.fechar()

    def test_agent_habilitar_e_fechar_visualizacao(self) -> None:
        """Agent deve permitir habilitar e fechar a visualizacao dinamicamente."""
        agente = Agent(1)
        self.assertFalse(agente._visualizacao_ativa)
        self.assertIsNone(agente._visualizador)

        agente.habilitar_visualizacao(True, titulo="Teste Agente")
        self.assertTrue(agente._visualizacao_ativa)
        self.assertIsNotNone(agente._visualizador)

        agente.fechar_visualizacao()
        self.assertFalse(agente._visualizacao_ativa)
        self.assertIsNone(agente._visualizador)

    def test_agent_movement_atualiza_trajetoria_e_telemetria(self) -> None:
        """movement deve alimentar o historico de trajetoria e telemetria de controle."""
        agente = Agent(1)
        sim_mock = MagicMock()
        sim_mock.getObject.return_value = 1
        sim_mock.getStringProperty.return_value = "PioneerP3DX"

        # Mock de pose em (0.0, 0.0, 0.0)
        sim_mock.getObjectPosition.return_value = [0.0, 0.0, 0.138]
        sim_mock.getObjectOrientation.return_value = [0.0, 0.0, 0.0]
        # Mock de laser sem obstaculos proximos
        sim_mock.readCustomDataBlock.return_value = None

        img = np.ones((50, 50))
        goals = [(1.0, 1.0)]

        chegou = agente.movement(sim_mock, img, goals)
        self.assertFalse(chegou)
        self.assertEqual(len(agente._historico_trajetoria), 1)
        self.assertIsNotNone(agente._camada_reativa.ultima_telemetria)
        self.assertGreater(
            abs(agente._camada_reativa.ultima_telemetria.v_linear), 0.0
        )


class TestInflacaoEContagemReferencia(unittest.TestCase):
    """Testes para inflacao circular de obstaculos dinamicos e contagem de referencia."""

    def setUp(self) -> None:
        img_branca = np.ones((50, 50), dtype=float)
        self.planejador = PlanejadorGlobal(
            img_branca,
            escala=0.10,
            origem=(-2.5, -2.5),
            tamanho_celula_m=0.10,
            raio_robo_m=0.26,
            margem_seguranca_m=0.05,
        )

    def test_marcar_bloqueio_com_inflacao_e_contagem(self) -> None:
        """Marca celula com inflacao e verifica contagem de referencia."""
        celula_a = (25, 25)
        celula_b = (25, 26)

        self.assertEqual(np.count_nonzero(self.planejador.ocupacao_dinamica), 0)

        # Bloqueia A com inflacao
        self.planejador.marcar_bloqueio([celula_a], True, inflar=True)
        num_bloqueadas_a = np.count_nonzero(self.planejador.ocupacao_dinamica)
        self.assertGreater(num_bloqueadas_a, 1)

        # Bloqueia B (sobreposicao de footprint)
        self.planejador.marcar_bloqueio([celula_b], True, inflar=True)
        self.assertGreaterEqual(self.planejador._contagem_bloqueio_dinamico[25, 25], 2)

        # Desbloqueia A: celula comum entre A e B ainda deve permanecer ocupada
        self.planejador.marcar_bloqueio([celula_a], False, inflar=True)
        self.assertTrue(self.planejador.ocupacao_dinamica[25, 25])
        self.assertEqual(self.planejador._contagem_bloqueio_dinamico[25, 25], 1)

        # Desbloqueia B: todas as celulas devem ser completamente liberadas
        self.planejador.marcar_bloqueio([celula_b], False, inflar=True)
        self.assertEqual(np.count_nonzero(self.planejador.ocupacao_dinamica), 0)
        self.assertEqual(np.count_nonzero(self.planejador._contagem_bloqueio_dinamico), 0)
        np.testing.assert_array_equal(
            self.planejador.ocupacao, self.planejador.ocupacao_estatica
        )

    def test_desbloqueio_preserva_pegada_de_outra_observacao(self) -> None:
        """Mudanca do start nao pode liberar celula pertencente a outro obstaculo."""
        centro_a, centro_b = (25, 25), (25, 26)
        celula_comum = (25, 25)
        self.planejador.start = celula_comum
        self.planejador.marcar_bloqueio([centro_a], True)
        self.assertFalse(self.planejador.ocupacao_dinamica[celula_comum])

        self.planejador.start = (10, 10)
        self.planejador.marcar_bloqueio([centro_b], True)
        self.assertTrue(self.planejador.ocupacao_dinamica[celula_comum])
        self.planejador.marcar_bloqueio([centro_a], False)
        self.assertTrue(self.planejador.ocupacao_dinamica[celula_comum])
        self.planejador.marcar_bloqueio([centro_b], False)
        self.assertFalse(self.planejador.ocupacao_dinamica[celula_comum])

    def test_liberacao_completa_apos_expiracao(self) -> None:
        """Camada reativa garante liberacao de todas as celulas infladas apos expiracao."""
        camada = CamadaReativa(passos_persistencia=2, passos_expiracao=4)
        pose = (0.0, 0.0, 0.0)
        laser = np.array([[0.0, 1.0]])

        # Persistencia: 2 passos com deteccao
        for _ in range(2):
            camada.processar_obstaculos_dinamicos(laser, pose, self.planejador)

        self.assertGreater(np.count_nonzero(self.planejador.ocupacao_dinamica), 0)

        # Expiracao: 5 passos sem deteccao
        laser_vazio = np.empty((0, 2))
        for _ in range(5):
            camada.processar_obstaculos_dinamicos(laser_vazio, pose, self.planejador)

        # Todas as celulas infladas devem estar completamente liberadas
        self.assertEqual(np.count_nonzero(self.planejador.ocupacao_dinamica), 0)
        self.assertEqual(np.count_nonzero(self.planejador._contagem_bloqueio_dinamico), 0)
        self.assertEqual(len(camada._celulas_bloqueadas), 0)


class TestBufferCriticoParede(unittest.TestCase):
    """Testes para o buffer critico de seguranca contra paredes na camada reativa."""

    def setUp(self) -> None:
        img = np.ones((50, 50), dtype=float)
        img[:, 35:] = 0.0
        self.planejador = PlanejadorGlobal(
            img,
            escala=0.10,
            origem=(-2.5, -2.5),
            tamanho_celula_m=0.10,
            raio_robo_m=0.26,
            margem_seguranca_m=0.05,
        )
        self.camada = CamadaReativa(distancia_critica_parede_m=0.35)

    def test_parede_distante_filtrada_parede_proxima_mantida_no_buffer(self) -> None:
        """Parede longe (>0.35m) e filtrada; parede criticamente proxima (<=0.35m) e mantida para repulsao."""
        pose_longe = (0.50, 0.0, 0.0)
        laser_longe = np.array([[0.0, 0.50]])  # Atinge parede em x=1.00 (dist 0.50m)

        pose_perto = (0.75, 0.0, 0.0)
        laser_perto = np.array([[0.0, 0.25]])  # Atinge parede em x=1.00 (dist 0.25m)

        # Sem buffer critico (estritamente dinamico): ambos descartados
        res_longe_estrito = self.camada.filtrar_laser_dinamico(
            pose_longe, laser_longe, self.planejador, incluir_paredes_criticas=False
        )
        self.assertEqual(len(res_longe_estrito), 0)

        res_perto_estrito = self.camada.filtrar_laser_dinamico(
            pose_perto, laser_perto, self.planejador, incluir_paredes_criticas=False
        )
        self.assertEqual(len(res_perto_estrito), 0)

        # Com buffer critico: o distante e descartado, mas o proximo e preservado
        res_longe_crit = self.camada.filtrar_laser_dinamico(
            pose_longe, laser_longe, self.planejador, incluir_paredes_criticas=True
        )
        self.assertEqual(len(res_longe_crit), 0)

        res_perto_crit = self.camada.filtrar_laser_dinamico(
            pose_perto, laser_perto, self.planejador, incluir_paredes_criticas=True
        )
        self.assertEqual(len(res_perto_crit), 1)
        self.assertAlmostEqual(res_perto_crit[0, 1], 0.25)


class TestModulacaoVelocidadeCurva(unittest.TestCase):
    """Testes para modulacao de velocidade em curvas no Controlador."""

    def setUp(self) -> None:
        self.controlador = Controlador(
            distancia_controle_m=0.20, v_max_m_s=0.50, w_max_rad_s=1.0
        )

    def test_velocidade_reduz_em_curvas_acentuadas(self) -> None:
        """Em curvas acentuadas (>20 deg), v linear deve ser reduzida para evitar subestercamento."""
        pose_reta = (0.0, 0.0, 0.0)
        vetor_reto = (0.50, 0.0)
        v_reta, _ = self.controlador.controlar(pose_reta, vetor_reto)

        theta_curva = math.radians(60.0)
        vetor_curva = (0.50 * math.cos(theta_curva), 0.50 * math.sin(theta_curva))
        v_curva, _ = self.controlador.controlar(pose_reta, vetor_curva)

        self.assertAlmostEqual(v_reta, 0.50, places=3)
        self.assertLess(v_curva, v_reta * 0.50)
        self.assertGreater(v_curva, 0.0)


class TestDeteccaoVaoIntransponivelECircunavegacao(unittest.TestCase):
    """Testes para deteccao de vao entre duas pessoas, gap bridging e circunavegacao."""

    def setUp(self) -> None:
        img_branca = np.ones((80, 80), dtype=float)
        self.planejador = PlanejadorGlobal(
            img_branca,
            escala=0.10,
            origem=(-4.0, -4.0),
            tamanho_celula_m=0.10,
            raio_robo_m=0.26,
            margem_seguranca_m=0.15,
        )
        self.camada = CamadaReativa(
            passos_persistencia=2,
            passos_expiracao=5,
            distancia_gap_impassavel_m=0.75,
        )

    def test_deteccao_e_vedacao_de_vao_entre_duas_pessoas(self) -> None:
        """Laser que passa entre duas pessoas separadas por 0.45m detecta e veda o vao."""
        pose = (-1.5, 0.0, 0.0)  # Robô olhando para +x
        # Pessoa 1 em y = -0.25 (ang ~ -9.5 deg), Pessoa 2 em y = +0.25 (ang ~ +9.5 deg)
        # No meio (ang 0), feixe passa e vai a 5.0m
        angulos = np.radians([-15, -12, -9, -5, 0, 5, 9, 12, 15])
        distancias = np.array([1.5, 1.5, 1.5, 5.0, 5.0, 5.0, 1.5, 1.5, 1.5])
        laser = np.column_stack([angulos, distancias])

        for _ in range(self.camada.passos_persistencia):
            self.camada.processar_obstaculos_dinamicos(laser, pose, self.planejador)

        # Célula central no vão (x=0.0, y=0.0):
        celula_centro_vao = self.planejador.mundo_para_celula((0.0, 0.0))
        self.assertIsNotNone(celula_centro_vao)
        # O vão deve estar ocupado no grid dinâmico!
        self.assertTrue(self.planejador.ocupacao_dinamica[celula_centro_vao])

    def test_circunavegacao_de_duas_pessoas_no_dstar(self) -> None:
        """Duas pessoas com vão intransponível forçam o D* Lite a contornar por fora."""
        pose = (-2.0, 0.0, 0.0)
        goal = (2.0, 0.0)
        self.planejador.definir_goal(goal)

        # Simula laser detectando as duas pessoas com vão intransponível
        angulos = np.radians([-15, -12, -9, -5, 0, 5, 9, 12, 15])
        distancias = np.array([2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 2.0, 2.0, 2.0])
        laser = np.column_stack([angulos, distancias])

        for _ in range(self.camada.passos_persistencia):
            self.camada.processar_obstaculos_dinamicos(laser, pose, self.planejador)

        self.planejador.atualizar(pose)
        caminho = self.planejador.caminho_completo(pose)
        self.assertTrue(len(caminho) > 0)

        # Verifica que o caminho D* Lite circunavega: nenhum ponto atravessa o miolo do vão
        # entre as pessoas em x in [-0.3, 0.3] com |y| < 0.35m
        for pt in caminho:
            if abs(pt[0]) < 0.25:
                # Deve contornar pelo flanco externo
                self.assertGreaterEqual(abs(pt[1]), 0.40)

    def test_deteccao_curta_distancia_sem_ponto_cego(self) -> None:
        """Obstáculo a 0.15m (menor que o raio do robô 0.26m) deve ser detectado e bloqueado."""
        pose = (0.0, 0.0, 0.0)
        laser = np.array([[0.0, 0.15]])  # 15 cm a frente
        celula_esperada = self.planejador.mundo_para_celula((0.15, 0.0))

        for _ in range(self.camada.passos_persistencia):
            self.camada.processar_obstaculos_dinamicos(laser, pose, self.planejador)

        self.assertTrue(self.planejador.ocupacao_dinamica[celula_esperada])

    def test_obstaculo_dinamico_proximo_a_parede_estatica(self) -> None:
        """Pessoa em pé a 30cm de parede estática deve ser rastreada e inflada no grid livre."""
        img = np.ones((80, 80), dtype=float)
        img[:, :20] = 0.0  # Parede na esquerda
        planejador = PlanejadorGlobal(
            img,
            escala=0.10,
            origem=(-4.0, -4.0),
            tamanho_celula_m=0.10,
            raio_robo_m=0.26,
            margem_seguranca_m=0.15,
        )
        camada = CamadaReativa(passos_persistencia=2)
        # Parede acaba em x = -2.0. Pessoa em x = -1.7 (a 30cm da parede)
        pose = (0.0, 0.0, math.pi)  # Olhando para -x
        laser = np.array([[0.0, 1.70]])  # Atinge a pessoa em x = -1.70
        celula_pessoa = planejador.mundo_para_celula((-1.70, 0.0))

        for _ in range(camada.passos_persistencia):
            camada.processar_obstaculos_dinamicos(laser, pose, planejador)

        # Pessoa deve ser rastreada e bloqueada
        self.assertIn(celula_pessoa, camada._celulas_bloqueadas)
        # Células no espaço livre adjacente devem ser infladas dinamicamente
        self.assertGreater(np.count_nonzero(planejador.ocupacao_dinamica), 0)
        celula_livre_inflada = planejador.mundo_para_celula((-1.45, 0.05))
        self.assertIsNotNone(celula_livre_inflada)
        self.assertTrue(planejador.ocupacao_dinamica[celula_livre_inflada])

    def test_filtrar_laser_combinado_equivalencia(self) -> None:
        """filtrar_laser_combinado deve retornar arrays identicos a filtrar_laser_dinamico."""
        pose = (0.50, 0.0, 0.0)
        laser = np.array([[0.0, 0.50], [math.radians(30), 0.25], [math.radians(-30), 1.0]])

        din_esperado = self.camada.filtrar_laser_dinamico(
            pose, laser, self.planejador, incluir_paredes_criticas=False
        )
        reat_esperado = self.camada.filtrar_laser_dinamico(
            pose, laser, self.planejador, incluir_paredes_criticas=True
        )

        din_comb, reat_comb = self.camada.filtrar_laser_combinado(
            pose, laser, self.planejador
        )
        np.testing.assert_array_equal(din_comb, din_esperado)
        np.testing.assert_array_equal(reat_comb, reat_esperado)

    def test_expiracao_completa_de_vao_vedado(self) -> None:
        """Após pessoas saírem, todo o vão vedado e as células infladas expiram e são liberadas."""
        pose = (-1.5, 0.0, 0.0)
        angulos = np.radians([-15, -12, -9, -5, 0, 5, 9, 12, 15])
        distancias = np.array([1.5, 1.5, 1.5, 5.0, 5.0, 5.0, 1.5, 1.5, 1.5])
        laser = np.column_stack([angulos, distancias])

        for _ in range(self.camada.passos_persistencia):
            self.camada.processar_obstaculos_dinamicos(laser, pose, self.planejador)

        self.assertGreater(np.count_nonzero(self.planejador.ocupacao_dinamica), 0)

        # Pessoas saem
        laser_vazio = np.empty((0, 2))
        for _ in range(self.camada.passos_expiracao + 1):
            self.camada.processar_obstaculos_dinamicos(laser_vazio, pose, self.planejador)

        self.assertEqual(np.count_nonzero(self.planejador.ocupacao_dinamica), 0)
        self.assertEqual(np.count_nonzero(self.planejador._contagem_bloqueio_dinamico), 0)
        self.assertEqual(len(self.camada._celulas_bloqueadas), 0)


if __name__ == "__main__":
    unittest.main()
