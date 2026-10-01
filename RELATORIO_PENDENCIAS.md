# Relatorio de pendencias do TP2

**Data:** 29/09/2026  
**Escopo:** estado da implementacao da Pessoa A e itens restantes para uma
execucao completa, robusta e experimentalmente validada no CoppeliaSim.

## Entregue nesta etapa

Estao implementados e cobertos por testes automatizados:

- conversao da imagem do mapa em grid de ocupacao, inflacao de obstaculos e
  conversoes mundo <-> celula;
- D* Lite incremental, incluindo bloqueio e desbloqueio de celulas dinamicas;
- leitura de pose e encapsulamento dos dados brutos do Hokuyo;
- conversao de vetor desejado mundial para `(v, w)` e para velocidades das
  rodas do Pioneer;
- verificacao de chegada e sequenciamento de uma **lista completa de goals**;
- sorteio reprodutivel de goals livres, alcancaveis e separados entre si;
- testes unitarios dos componentes e smoke test pela ZeroMQ Remote API.

O comando abaixo executa uma sequencia de goals aleatorios validos; ele passa
a mesma lista integral para `Agent.movement(...)` a cada ciclo:

```powershell
..\.venv\Scripts\python.exe teste_coppelia.py --seed 42 --quantidade 5 --separacao 1.0 --duracao 120
```

## Pendencias funcionais

### 1. Camada reativa dinamica (Pessoa B) — [IMPLEMENTADO]

Implementada na classe `CamadaReativa` em `src/Agente1.py` e integrada ao ciclo
`Agent.movement(...)`, consumindo `Sensoriamento.obter_laser`,
`PlanejadorGlobal.caminho_completo` e `Controlador.controlar`. Totalmente coberta
por 42 testes automatizados em `tests/test_camada_reativa.py`.

- **Visualizador em tempo real e telemetria de decisões:** `VisualizadorNavegacao`
  e `TelemetriaDecisao` fornecem uma janela interativa separada exibindo em tempo
  real a visão global (grid D*, rota planejada, trajetória e goals), a visão ego local
  (casco do robô, feixes laser dinâmicos e estáticos, e quivers de todas as forças APF
  atuantes: atração, repulsão radial, repulsão tangencial e quebra de simetria) e um HUD
  completo de telemetria em monospace. Ativável por `visualizar=True`, `habilitar_visualizacao()`
  ou `TP2_VISUALIZAR=1`.

- **Confiança no caminho D* e filtragem estática:** `filtrar_laser_dinamico` e
  `eh_obstaculo_estatico` identificam e descartam leituras incidentes em paredes do
  mapa original, garantindo que o robô siga fielmente a rota D* Lite sem repulsão
  de paredes estáticas, realizando desvios reativos exclusivamente para obstáculos dinâmicos.
- **Repulsão radial e tangencial por campos potenciais:**
  - Repulsão radial via `calcular_repulsao` (`ganho_repulsao = 0.35`).
  - Repulsão tangencial circulatória via `calcular_repulsao_tangencial`
    (`ganho_tangencial = 0.30`), orientada para maximizar o alinhamento com a
    atração ao goal/lookahead, permitindo contornar robôs e obstáculos móveis
    com muito mais rapidez e sem perda desnecessária de velocidade linear.
- **Lookahead e combinação:** `selecionar_lookahead` seleciona waypoint à frente
  no caminho global (`distancia_lookahead_m = 0.60 m`). Atração, repulsão radial,
  repulsão tangencial e quebra de simetria são combinados em `calcular_vetor_desejado`
  em um único vetor `(vx, vy)` enviado a `Controlador.controlar`.
- **Detecção e persistência de obstáculos dinâmicos:** `processar_obstaculos_dinamicos`
  converte coordenadas de impacto no espaço livre estático para células do grid e
  chama `planejador.marcar_bloqueio(celulas, True)` após persistência (`passos_persistencia = 3`).
- **Histerese e expiração:** células bloqueadas dinamicamente são automaticamente
  liberadas via `planejador.marcar_bloqueio(celulas, False)` quando não mais
  observadas após tempo de expiração (`passos_expiracao = 15`).
- **Detecção de estagnação e recuperação:** `monitorar_estagnacao` monitora o
  deslocamento e variação angular em uma janela temporal (`janela_estagnacao_passos = 20`);
  detecta travamento sem confundir curvas lentas ou giros no eixo (distinguindo
  pela taxa de variação de theta), acionando máquina de estados de recuperação
  composta por recuo (`ESTADO_RECUO`, v < 0) e giro (`ESTADO_GIRO`, w < 0).
- **Quebra determinística de simetria:** `calcular_quebra_simetria` adiciona
  componente lateral à direita (`u_direita = (sin theta, -cos theta)`) em
  encontros frontais (|ângulo| <= 30 deg), prevenindo deadlocks colineares.

### 2. Calibracao na cena real

Os valores abaixo sao valores iniciais centralizados em `Agent`; precisam ser
confirmados visualmente e, quando houver, contra a validacao do servidor:

| Parametro | Valor inicial | O que medir |
| --- | ---: | --- |
| `RAIO_ROBO_M` | 0,26 m | folga real entre casco e paredes |
| `MARGEM_SEGURANCA_M` | 0,15 m | colisao versus perda de corredores viaveis |
| `TOLERANCIA_GOAL_M` | 0,10 m | erro final aceito pelo servidor |
| `VELOCIDADE_DESEJADA_M_S` | 0,50 m/s | estabilidade, derrapagem e tempo de missao |
| limites de `v` e `w` | classe `Controlador` | oscilacao, raio de curva e saturacao |
| `TAMANHO_CELULA_M` | 0,10 m | tempo de replanejamento versus detalhe do mapa |

Em especial, a leitura de `2 rad/s` observada na probe da cena apenas mostra
que houve comando de roda. Ela nao substitui a calibracao da velocidade nem a
medicao de deslocamento do robo.

### 3. Validacao experimental obrigatoria

Os testes unitarios validam a logica, mas nao substituem os experimentos no
CoppeliaSim. Ainda devem ser registrados, para cada rodada: seed, lista de
goals, pose inicial, goals concluidos, tempo por goal, distancia final,
colisoes e observacoes.

1. **Travessia longa:** usar pelo menos tres goals em regioes distintas e uma
   orientacao inicial desfavoravel. Avaliar convergencia, curvas, seguranca e
   erro final.
2. **Replanejamento incremental:** bloquear uma celula sobre a rota, medir
   `PlanejadorGlobal.atualizar`, liberar a celula e confirmar o retorno da
   rota curta. Comparar com uma reinicializacao completa usando mesma pose e
   mesmo goal.
3. **Cenario dinamico:** colocar dois robos em rota de colisao e um robo
   parado sobre o caminho. Medir acionamentos de bloqueio, tempo de resposta,
   recuperacoes e eventuais deadlocks.

### 4. Integracao final com o servidor da disciplina

- Confirmar o formato exato de `goal_list` entregue pelo servidor; a
  implementacao espera um iteravel de pares `(x, y)` em metros no referencial
  mundial da cena.
- Confirmar se o retorno `True` deve sinalizar cada goal individual ou o fim
  de toda a lista. Atualmente ele sinaliza a chegada ao **goal corrente**,
  conforme o contrato recebido, e avanca o indice interno.
- Executar uma partida completa na configuracao oficial, sem alterar as
  assinaturas exigidas de `Agent.__init__(id_robo)` e
  `movement(sim, img, goal_list)`.
- Salvar log de resultados e, se permitido pelo enunciado, imagens ou video
  das trajetorias para anexar ao relatorio final.

## Criterio de conclusao

O TP2 pode ser considerado pronto quando a lista de goals for concluida na
cena oficial sem colisao estatica, o erro final respeitar a tolerancia do
servidor, o replanejamento dinamico nao produzir oscilacao excessiva e os tres
experimentos acima tiverem resultados registrados e reproduziveis.
