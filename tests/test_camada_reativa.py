from __future__ import annotations

import math
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


if __name__ == "__main__":
    unittest.main()
