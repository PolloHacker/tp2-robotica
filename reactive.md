# Camada Reativa Dinâmica — Documentação Técnica (Pessoa B)

> Nota: este texto registra uma etapa anterior da camada reativa. As assinaturas
> públicas, parâmetros e resultados finais estão atualizados em
> [docs/RELATORIO.md](docs/RELATORIO.md).

**Data:** 30/09/2026  
**Escopo:** Implementação, modelagem matemática, calibração e validação da camada reativa dinâmica do Pioneer P3-DX no Trabalho Prático 2 de Robótica Móvel (UFMG).  
**Arquivo principal:** [`src/Agente1.py`](src/Agente1.py)  
**Suíte de testes:** [`tests/test_camada_reativa.py`](tests/test_camada_reativa.py)  

---

## 1. Visão Geral da Arquitetura

A arquitetura de navegação do robô combina planejamento global deliberativo (**Pessoa A**) com controle reativo dinâmico (**Pessoa B**):

```
                                +---------------------------+
                                |    Sensoriamento (sim)    |
                                +-------------+-------------+
                                              |
                          +-------------------+-------------------+
                          | (pose: x, y, θ)                       | (laser: α, d)
                          v                                       v
+-------------------------+---+               +-------------------+--------------------+
|  PlanejadorGlobal (D* Lite) |               |        CamadaReativa (Pessoa B)        |
+-------------------------+---+               +-------------------+--------------------+
| - Grid estático inflado     |               | 1. Deteccao obstaculos dinamicos       |
| - Bloqueios dinâmicos       | <-------------+    - Persistência e expiração          |
| - Atualização incremental   |  marcar_      | 2. Filtragem de obstaculos estaticos   |
| - Caminho suavizado         |  bloqueio     |    - Confianca no caminho D* estatico  |
+-------------+---------------+               | 3. Seleção de lookahead global         |
              | caminho                       | 4. Vetor atrativo                      |
              +-----------------------------> | 5. Campos potenciais repulsivos:       |
                                              |    - Repulsao radial                   |
                                              |    - Repulsao tangencial (circulacao)  |
                                              | 6. Quebra determinística de simetria   |
                                              | 7. Detecção e recuperação de estagnação|
                                              +-------------------+--------------------+
                                                                  |
                                                                  | vetor mundial (vx, vy)
                                                                  v
                                              +-------------------+--------------------+
                                              |     Controlador (Desai et al.)         |
                                              +-------------------+--------------------+
                                              | - (vx, vy) -> (v, w)                   |
                                              | - (v, w)   -> (wl, wr)                 |
                                              +-------------------+--------------------+
                                                                  |
                                                                  v
                                              +-------------------+--------------------+
                                              |    Motores CoppeliaSim (Left/Right)    |
                                              +----------------------------------------+
```

A **Camada Reativa** consome `Sensoriamento.obter_laser`, `PlanejadorGlobal.caminho_completo` e alimenta `Controlador.controlar`, garantindo que:
1. O robô confie e siga estritamente o caminho ótimo do D* Lite contra obstáculos estáticos, sem sofrer repulsões desnecessárias de paredes conhecidas.
2. Obstáculos dinâmicos no campo do laser acionem campos potenciais repulsivos radiais e tangenciais para desvio rápido e contorno ágil.
3. Não haja colisões frontais por indecisão (quebra de simetria à direita e circulação tangencial orientada ao goal).
4. Obstáculos dinâmicos observados com consistência sejam inseridos no grid D* Lite para desvio deliberativo de longo alcance.
5. Obstáculos dinâmicos que desocupem a área sejam liberados por histerese/expiração.
6. Travamentos em cantos ou becos sejam detectados e desfeitos por manobras de recuperação (recuo e giro), sem confundir curvas lentas com estagnação.

---

## 2. Modelagem Matemática e Componentes

### 2.1. Lookahead no Caminho Global e Força Atrativa

Em vez de perseguir o waypoint imediatamente adjacente (distância de $0,10\text{ m}$), a estratégia emprega um ponto de **lookahead** à frente:

1. Localiza o waypoint mais próximo da pose atual do robô $(x_r, y_r)$.
2. Varre o caminho para a frente até encontrar o primeiro waypoint $P_{wp} = (x_{wp}, y_{wp})$ cuja distância seja:
   $$\|P_{wp} - P_r\| \ge d_{lookahead} \quad (d_{lookahead} = 0,60\text{ m})$$
3. Se todos os waypoints restantes estiverem mais próximos que $d_{lookahead}$ (aproximação do goal), o próprio goal final $P_{goal}$ é selecionado.

O vetor de atração mundial aponta para o lookahead normalizado com a velocidade desejada $v_{des}$:
$$\vec{v}_{att} = v_{des} \cdot \frac{P_{wp} - P_r}{\|P_{wp} - P_r\|}$$

### 2.2. Repulsão por Campos Potenciais Artificiais (APF)

O sensor Hokuyo retorna leituras em coordenadas polares relativas $(\alpha_i, d_i)$, com $\alpha_i \in [-120^\circ, +120^\circ]$.

Para evitar viés de densidade de feixes (uma parede plana de 100 feixes geraria 100x mais força do que um robô pontual), as leituras válidas ($d_i > 0,02\text{ m}$ e $d_i < d_{inf}$) são agrupadas em **24 setores angulares** de $10^\circ$. Para cada setor $s$, extrai-se a distância mínima $d_{s}$ e o ângulo $\alpha_s$:

1. **Saturação de proximidade:**
   $$d_{ef} = \max(d_s, d_{sat}) \quad (d_{sat} = 0,32\text{ m})$$
   Garante que leituras muito próximas ao casco ($R_{robo} = 0,26\text{ m}$) não causem singularidade numérica nem divergência.

2. **Magnitude do potencial repulsivo:**
   $$F_{s} = k_{rep} \left( \frac{1}{d_{ef}} - \frac{1}{d_{inf}} \right) \cdot \frac{1}{d_{ef}} \quad (k_{rep} = 0,35, \; d_{inf} = 0,90\text{ m})$$

3. **Direção mundial do feixe repulsivo:**
   Sendo $\theta$ a orientação mundial do robô:
   $$\phi_s = \theta + \alpha_s$$
   $$\vec{F}_{rep, s} = -F_s \cdot \begin{bmatrix} \cos\phi_s \\ \sin\phi_s \end{bmatrix}$$

4. **Soma e saturação:**
   $$\vec{F}_{rep} = \sum_{s} \vec{F}_{rep, s}$$
   Se $\|\vec{F}_{rep}\| > f_{rep\_max}$ ($0,65\text{ m/s}$), o vetor é saturado:
   $$\vec{F}_{rep} \leftarrow f_{rep\_max} \cdot \frac{\vec{F}_{rep}}{\|\vec{F}_{rep}\|}$$

### 2.3. Repulsão Tangencial (Circulação em Torno de Obstáculos Dinâmicos)

A repulsão radial clássica atua apenas empurrando o robô para trás quando o obstáculo está à sua frente, o que pode reduzir a velocidade a quase zero ou causar oscilação prolongada.

Para permitir que o robô **contorne obstáculos dinâmicos muito mais rápido**, implementou-se a **repulsão tangencial** (campo circulatório/vórtice):

1. **Direções ortogonais:** Para cada setor com ângulo mundial $\phi_s$, os vetores unitários perpendiculares à direção do obstáculo são:
   $$\hat{u}_{tan, dir} = \begin{bmatrix} \sin\phi_s \\ -\cos\phi_s \end{bmatrix} \quad (\text{girado } -90^\circ, \text{à direita do obstáculo})$$
   $$\hat{u}_{tan, esq} = \begin{bmatrix} -\sin\phi_s \\ \cos\phi_s \end{bmatrix} \quad (\text{girado } +90^\circ, \text{à esquerda do obstáculo})$$

2. **Orientação alinhada ao goal:** A direção $\hat{u}_{tan}$ é selecionada maximizando o produto escalar com o vetor de atração $\vec{v}_{att}$ do lookahead:
   $$\text{Se } \vec{v}_{att} \cdot \hat{u}_{tan, esq} > \vec{v}_{att} \cdot \hat{u}_{tan, dir} \implies \hat{u}_{tan} = \hat{u}_{tan, esq}$$
   $$\text{Caso contrário (incluindo empate/colineariedade frontal)} \implies \hat{u}_{tan} = \hat{u}_{tan, dir}$$

3. **Magnitude e acúmulo:**
   $$F_{tan, s} = k_{tan} \left( \frac{1}{d_{ef}} - \frac{1}{d_{inf}} \right) \cdot \frac{1}{d_{ef}} \quad (k_{tan} = 0,30)$$
   $$\vec{F}_{tan} = \sum_{s} F_{tan, s} \cdot \hat{u}_{tan, s}$$
   Saturado em $f_{rep\_max}$ caso a norma exceda o limite.

### 2.4. Filtragem de Obstáculos Estáticos e Confiança no Caminho D*

O planejador deliberativo D* Lite calcula a rota ótima com folga de segurança inflada ($R_{robo} + M_{seg} = 0,41\text{ m}$) contra todas as paredes estáticas da imagem do mapa. Se o campo potencial atuasse também contra paredes estáticas, o robô seria empurrado lateralmente em corredores ou impedido de dobrar esquinas planejadas pelo D*.

Portanto, a camada reativa implementa a regra: **o robô acredita no caminho D* para obstáculos estáticos, realizando alterações apenas para obstáculos dinâmicos**.

1. O método [`PlanejadorGlobal.eh_obstaculo_estatico`](src/Agente1.py) verifica se uma coordenada mundial $(x_w, y_w)$ incide sobre uma parede estática ou sua vizinhança imediata de discretização ($\le 0,145\text{ m}$).
2. O método [`CamadaReativa.filtrar_laser_dinamico`](src/Agente1.py) descarta todos os feixes refletidos em paredes estáticas, isolando apenas os pontos correspondentes a robôs móveis ou caixas dinâmicas.
3. As forças repulsiva radial $\vec{F}_{rep}$, tangencial $\vec{F}_{tan}$, de quebra de simetria $\vec{F}_{sym}$ e a modulação frontal de velocidade atuam **exclusivamente** sobre leituras dinâmicas. Na ausência de obstáculos dinâmicos, o robô segue a rota global com desvio zero.

### 2.5. Quebra Determinística de Simetria (Desvio à Direita)

Em encontros frontais colineares (dois robôs se aproximando frente a frente), $\vec{v}_{att}$ e $\vec{F}_{rep}$ tornam-se opostos e colineares, criando um ponto de sela / deadlock.

Adota-se uma regra determinística baseada na convenção náutica/aérea (COLREGS): **priorizar desvio para a direita do robô**.

1. Filtram-se obstáculos no cone frontal:
   $$|\alpha| \le \alpha_{frontal} \quad (\alpha_{frontal} = 30^\circ)$$
2. Extrai-se a menor distância frontal $d_{front\_min}$. Se $d_{front\_min} < d_{inf}$:
   $$w_{sym} = \frac{d_{inf} - d_{front\_min}}{d_{inf}}$$
   $$\vec{F}_{sym} = k_{sym} \cdot w_{sym} \cdot \hat{u}_{direita} \quad (k_{sym} = 0,30)$$
   Onde $\hat{u}_{direita}$ é o vetor unitário ortogonal à direita do chassi no mundo:
   $$\hat{u}_{direita} = \begin{bmatrix} \sin\theta \\ -\cos\theta \end{bmatrix}$$

### 2.6. Combinação dos Vetores de Velocidade

Os quatro vetores são somados no espaço de estados mundial:
$$\vec{v}_{tot} = \vec{v}_{att} + \vec{F}_{rep} + \vec{F}_{tan} + \vec{F}_{sym}$$

Se houver obstáculo dinâmico frontal a menos de $0,50\text{ m}$, a velocidade desejada é modulada para dar tempo à rotação:
$$v_{alvo} = \max\left(0,18, \; v_{des} \cdot \frac{d_{front\_min}}{0,50}\right)$$

O vetor resultante final é normalizado para $v_{alvo}$:
$$\vec{v}_{desejado} = v_{alvo} \cdot \frac{\vec{v}_{tot}}{\|\vec{v}_{tot}\|}$$

Este vetor é fornecido a [`Controlador.controlar`](src/Agente1.py#L769), que converte para comandos diferenciais $(v, w)$ e velocidades de roda $(w_l, w_r)$ segundo o modelo de Desai et al. (1998).

---

## 3. Gestão Dinâmica de Obstáculos no Grid D* Lite

A camada reativa integra o laser com o [`PlanejadorGlobal`](src/Agente1.py#L15) para que robôs ou caixas móveis se tornem obstáculos no mapa de custo incremental:

### 3.1. Transformação de Coordenadas
Para cada leitura $(\alpha, d)$ com $R_{robo} < d \le 3,0\text{ m}$:
$$x_{obs} = x_r + d \cos(\theta + \alpha)$$
$$y_{obs} = y_r + d \sin(\theta + \alpha)$$
$$c = \text{mundo\_para\_celula}((x_{obs}, y_{obs}))$$

### 3.2. Filtro de Espaço Livre e Proteções
- Se $c$ já pertence a `ocupacao_estatica` (parede estática inflada): **ignora** (já tratado na inicialização).
- Se $c$ coincide com o start atual ou o goal atual: **ignora** (evita bloquear o próprio robô ou inviabilizar a chegada).

### 3.3. Persistência (Filtro Anti-Ruído)
Um obstáculo detectado em um único frame não pode acionar o replanejamento D* Lite.
- Mantém contador de ocorrências $C[c]$.
- Quando $C[c] \ge N_{persist}$ ($N_{persist} = 3$ ciclos consecutivos):
  - Marca célula como bloqueada: `planejador.marcar_bloqueio([c], True)`.
  - Registra tempo do último avistamento $T_{last}[c]$.

### 3.4. Histerese e Expiração (Desbloqueio Automático)
Quando o obstáculo móvel sai do local:
- Se $ciclo_{atual} - T_{last}[c] > N_{expira}$ ($N_{expira} = 15$ ciclos $\approx 1,5\text{ s}$):
  - Desbloqueia célula: `planejador.marcar_bloqueio([c], False)`.
  - Remove dos registros de persistência.
- O D* Lite propaga a remoção do custo e restaura a rota ótima mais curta.

---

## 4. Detecção de Estagnação e Recuperação

Para contornar mínimos locais severos ou aprisionamento por múltiplos agentes:

### 4.1. Monitoramento por Janela Deslizante
Armazena as últimas $N = 20$ poses $(x_k, y_k, \theta_k)$.
A estagnação é declarada se e somente se:
1. O deslocamento linear for menor que o limiar:
   $$\Delta p = \sqrt{(x_k - x_0)^2 + (y_k - y_0)^2} < 0,05\text{ m}$$
2. A variação angular for menor que o limiar:
   $$|\Delta\theta| = |(\theta_k - \theta_0 + \pi) \pmod{2\pi} - \pi| < 0,20\text{ rad} \; (\approx 11,5^\circ)$$
3. A distância até o goal atual for maior que a tolerância de chegada ($0,10\text{ m}$).

> **Diferencial Crítico:** Se o robô estiver fazendo uma curva lenta ou girando no próprio eixo, $|\Delta\theta| \gg 0,20\text{ rad}$. O algoritmo detecta que há rotação ativa e **não** dispara falso alarme de travamento.

### 4.2. Máquina de Estados de Recuperação
Ao detectar estagnação real:

```
[ESTADO_NORMAL] 
       | (estagnação confirmada: Δp < 0.05m e |Δθ| < 0.20 rad)
       v
 [ESTADO_RECUO]  ---> v = -0.20 m/s, w = -0.30 rad/s (12 passos)
       | (passos concluídos)
       v
  [ESTADO_GIRO]  ---> v = 0.00 m/s, w = -1.00 rad/s para a direita (10 passos)
       | (passos concluídos)
       v
[ESTADO_NORMAL]  ---> Janela reiniciada, replanejamento D* Lite retoma condução
```

Para manter interface unificada com o `Controlador`, as velocidades de recuperação $(v, w)$ são convertidas diretamente em vetor mundial $(v_x, v_y)$ pela transformação inversa de Desai et al.:
$$v_x = \cos\theta \cdot v - d_c \sin\theta \cdot w$$
$$v_y = \sin\theta \cdot v + d_c \cos\theta \cdot w$$

---

## 5. Tabela de Parâmetros Calibrados

| Parâmetro | Valor | Justificativa Física |
| :--- | :---: | :--- |
| `distancia_lookahead_m` | `0,60 m` | Suaviza o rastreamento do caminho sem cortar quinas fechadas (~2,3 raios do robô). |
| `raio_influencia_laser_m` | `0,90 m` | Obstáculos a mais de 90 cm não interferem na trajetória em corredores normais. |
| `distancia_saturacao_m` | `0,32 m` | Folga ligeiramente superior ao raio do casco ($0,26\text{ m}$) para evitar forças infinitas. |
| `ganho_repulsao` | `0,35` | Balanço ideal entre desvio de paredes e atração ao goal. |
| `ganho_tangencial` | `0,30` | Força tangencial circulatória para contorno rápido de obstáculos dinâmicos. |
| `ganho_simetria` | `0,30` | Força lateral suficiente para deflexão preventiva à direita em encontros frontais. |
| `f_rep_max` | `0,65 m/s` | Casado com a velocidade linear máxima $v_{max}$ do controlador. |
| `angulo_frontal_rad` | `±30°` | Setor frontal crítico para detecção de colisões iminentes. |
| `alcance_dinamico_m` | `3,0 m` | Limite de alcance confiável do fastHokuyo para inserção de células no grid. |
| `passos_persistencia` | `3` | Filtra ruídos espúrios; confirmação em ~0,15–0,30 s de observação contínua. |
| `passos_expiracao` | `15` | Histerese de ~1,5 s antes de liberar células após saída do obstáculo. |
| `janela_estagnacao_passos` | `20` | Janela temporal de ~1,0–2,0 s para diagnóstico de travamento. |
| `limiar_deslocamento_estagnacao_m` | `0,05 m` | Deslocamento residual acumulado indicativo de robô parado. |
| `limiar_angulo_estagnacao_rad` | `0,20 rad` | Limiar angular para não confundir curva lenta ou manobra de giro com travamento. |
| `passos_recuo` / `passos_giro` | `12` / `10` | Duração das manobras de desengajamento antes de reiniciar busca de rota. |

---

## 6. Validação por Testes Automatizados

A suíte em [`tests/test_camada_reativa.py`](tests/test_camada_reativa.py) contém **37 testes unitários e de integração**, executáveis via:

```bash
uv run python -m unittest discover tests
```

### 6.1. Casos de Teste Cobertos

1. **Campos Potenciais Repulsivos Radiais (`TestPotencialRepulsao`):**
   - `test_sem_leituras_laser`: força nula se laser vazio ou None.
   - `test_obstaculo_alem_raio_influencia`: ausência de força para $d \ge 0,90\text{ m}$.
   - `test_repulsao_frontal`: empurrão exclusivamente para trás (direção $-x$) com obstáculo em $\alpha = 0$.
   - `test_repulsao_lateral_esquerda`: empurrão para trás e para a direita ($-y$) com obstáculo a $+45^\circ$.
   - `test_repulsao_lateral_direita`: empurrão para trás e para a esquerda ($+y$) com obstáculo a $-45^\circ$.
   - `test_saturacao_muito_proximo`: magnitude finita e limitada a `f_rep_max` para distâncias quase nulas.

2. **Repulsão Tangencial Circulatória (`TestRepulsaoTangencial`):**
   - `test_sem_leituras_laser`: força nula se laser vazio.
   - `test_obstaculo_alem_raio_influencia`: ausência de força fora de $0,90\text{ m}$.
   - `test_obstaculo_frontal_desvia_direita`: empurrão tangencial prioritário à direita ($-y$) para obstáculo frontal.
   - `test_obstaculo_lateral_esquerda_circula_direita`: empurrão com avanço ($+x$) e curvatura à direita ($-y$) contornando obstáculo à esquerda.
   - `test_obstaculo_lateral_direita_circula_esquerda`: empurrão com avanço ($+x$) e curvatura à esquerda ($+y$) contornando obstáculo à direita.
   - `test_saturacao_f_rep_max`: magnitude saturada em `f_rep_max` em distâncias muito pequenas.

3. **Filtragem de Estáticos e Confiança no D* (`TestFiltroEstaticoEAcreditarDStar`):**
   - `test_reconhecimento_obstaculo_estatico`: mapeamento correto de pontos sobre paredes como estáticos e áreas livres como dinâmicos.
   - `test_filtrar_laser_descarta_parede_estatica`: descarte de feixes refletidos em paredes estáticas conhecidas.
   - `test_filtrar_laser_preserva_obstaculo_dinamico`: preservação integral de feixes em espaço livre (outro robô móvel).
   - `test_robo_acredita_no_caminho_dstar_junto_a_parede`: robô seguindo rota D* ao lado de parede estática não sofre desvio repulsivo, mantendo direção exata do lookahead D*.
   - `test_desvio_ocorre_somente_para_obstaculo_dinamico`: cenário com parede e obstáculo dinâmico em que o desvio reativo ocorre exclusivamente em resposta ao obstáculo dinâmico.

4. **Quebra de Simetria (`TestQuebraDeSimetria`):**
   - `test_encontro_frontal_prioriza_direita`: geração de componente lateral à direita ($-y$) com $\theta = 0$.
   - `test_encontro_frontal_orientacao_diferente`: geração de componente lateral à direita ($+x$) com $\theta = \pi/2$.
   - `test_obstaculo_fora_do_cone_frontal`: sem viés lateral para obstáculos fora dos $\pm 30^\circ$.

5. **Lookahead e Atração (`TestLookaheadEAtracao`):**
   - `test_selecao_lookahead`: seleção do waypoint à distância $\ge 0,60\text{ m}$ à frente no caminho.
   - `test_lookahead_proximo_ao_goal`: seleção do próprio goal quando a menos de $0,60\text{ m}$ do fim.
   - `test_caminho_vazio`: retorno `None` para caminhos vazios.
   - `test_calculo_atracao`: cálculo de vetor atrativo unitário escalado para $v_{desejada}$.

6. **Detecção Dinâmica e Histerese (`TestDeteccaoObstaculosDinamicos`):**
   - `test_persistencia_necessaria_para_bloqueio`: não bloqueia nos passos 1 e 2; bloqueia no passo 3.
   - `test_expiracao_e_desbloqueio`: mantém bloqueado durante histerese e libera automaticamente no passo 16.
   - `test_replanejamento_apos_bloqueio_e_desbloqueio`: verifica que o D* Lite encontra desvio contornando a célula bloqueada dinamicamente e restabelece a rota curta original assim que a célula expira.

7. **Estagnação e Recuperação (`TestEstagnacaoERecuperacao`):**
   - `test_deteccao_estagnacao_e_recuperacao`: ativação no 5º ciclo parado, sequência `RECUO` ($v < 0$) $\to$ `GIRO` ($w < 0$) $\to$ retorno a `NORMAL`.
   - `test_nao_confunde_curva_ou_giro_com_estagnacao`: rotação ativa com $\Delta\theta > 0,20\text{ rad}$ não dispara estagnação.
   - `test_nao_ativa_estagnacao_se_estiver_no_goal`: robô dentro da tolerância do goal não entra em recuperação.

8. **Composição Vetorial (`TestCombinacaoVetorDesejado`):**
   - `test_combinacao_livre_segue_lookahead`: vetor idêntico ao lookahead em espaço livre.
   - `test_combinacao_com_obstaculo_lateral_esquerda`: deflexão lateral à direita com componente de avanço preservado.
   - `test_combinacao_frontal_aplica_desvio_direita`: quebra de simetria desloca vetor para a direita em encontro frontal.
   - `test_modulacao_velocidade_obstaculo_muito_proximo`: redução automática de velocidade em distâncias $< 0,50\text{ m}$.

9. **Integração com `Agent` (`TestIntegracaoAgent`):**
   - `test_inicializacao_agente`: instanciação correta dos módulos na classe `Agent`.
   - `test_ciclo_movement_com_mock`: ciclo completo de sensoriamento, planejamento, reatividade e envio de comandos de roda aos motores.
   - `test_chegada_ao_goal_retorna_true`: validação do contrato booleano e parada das rodas ao alcançar o objetivo.

### 6.2. Resultado da Execução
```text
Ran 42 tests in 1.460s
OK
```
---
## 7. Visualizador em Tempo Real & Painel de Telemetria de Decisões
Para atender ao requisito de transparência e inspeção em tempo real das decisões de planejamento e navegação do robô, foram introduzidas as classes [`TelemetriaDecisao`](src/Agente1.py) e [`VisualizadorNavegacao`](src/Agente1.py).
### 7.1. Como Habilitar a Visualização
A janela separada pode ser ativada de três formas transparentes:
1. **Parâmetro de inicialização:** `agente = Agent(id_robo=1, visualizar=True)`
2. **Método dinâmico:** `agente.habilitar_visualizacao(True)`
3. **Variável de ambiente:** `export TP2_VISUALIZAR=1` (permite visualizar qualquer execução sem alterar código)
### 7.2. Painéis da Janela Gráfica
A janela interativa possui layout 2x2 otimizado (sub-15 ms por quadro, não bloqueante para a simulação):
1. **Painel Esquerdo — Visão Global (30 m x 30 m):**
- Grid de ocupação estático inflado do D* Lite (paredes conhecidas em cinza escuro/médio).
- Células dinamicamente bloqueadas exibidas em vermelho vivo (`#dc2626`).
- Rota global ótima do D* Lite (linha tracejada laranja `#ea580c`).
- Trajetória percorrida pelo robô (linha contínua azul `#2563eb`).
- Goals pendentes (estrelas vermelhas numeradas G1, G2, etc.) e goals alcançados (estrelas verdes).
- Marcador de pose do robô em roxo (`#7c3aed`).
2. **Painel Superior Direito — Visão Ego Local do Robô (Zoom 5 m x 5 m):**
- Casco circular do Pioneer P3-DX ($R = 0,26\text{ m}$) e margem de segurança inflada ($R = 0,41\text{ m}$).
- Vetor de orientação da proa ($\theta$).
- Feixes do sensor laser Hokuyo classificados:
- **Cinza:** leituras coincidentes com paredes estáticas (ignoradas pelo APF para manter fidelidade ao D*).
- **Vermelho vivo:** leituras correspondentes a obstáculos dinâmicos ativos.
- Waypoint de lookahead selecionado no caminho (cruz ciano).
- **Vetores de Força em Tempo Real (Quivers):**
- $\vec{v}_{att}$ (Verde): Atração ao waypoint do caminho D*.
- $\vec{F}_{rep}$ (Vermelho): Repulsão radial contra obstáculos dinâmicos.
- $\vec{F}_{tan}$ (Magenta): Repulsão tangencial de circulação rápida (contorno ágil).
- $\vec{F}_{sym}$ (Ouro): Força determinística de quebra de simetria à direita.
- $\vec{v}_{des}$ (Azul Escuro): Comando vetorial total enviado ao controlador de rastreamento.
3. **Painel Inferior Direito — HUD de Telemetria e Diagnóstico:**
- Terminal monocromático estilizado de alto contraste (`#0f172a`).
- Estado da máquina de navegação (`NORMAL`, `RECUO`, `GIRO`).
- Meta ativa ($G_k$), coordenadas mundiais e distância euclidiana instantânea.
- Velocidades em tempo real: $v$ linear, $\omega$ angular e comandos de motor $w_l, w_r$.
- Intensidades exatas de todas as forças reativas ativas ($\|v_{att}\|, \|F_{rep}\|, \|F_{tan}\|, \|F_{sym}\|, \|v_{des}\| drive$).
- Contagem de feixes dinâmicos, células bloqueadas ativas no D* e número de waypoints restantes.
