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
por 26 testes automatizados em `tests/test_camada_reativa.py`.

- **Repulsao por campos potenciais:** calculada via `calcular_repulsao` a partir
  de `(angulo, distancia)` do laser, agrupada em setores angulares para evitar
  viés de contagem de feixes, com raio de influencia (`raio_influencia_laser_m = 0.90 m`)
  e saturacao proxima ao robo (`distancia_saturacao_m = 0.32 m`, saturada em `f_rep_max`).
- **Lookahead e combinacao:** `selecionar_lookahead` seleciona waypoint a frente
  no caminho global (`distancia_lookahead_m = 0.60 m`). Atracao, repulsao e
  desvio lateral sao combinados em `calcular_vetor_desejado` em um unico vetor
  `(vx, vy)` enviado a `Controlador.controlar`.
- **Deteccao e persistencia de obstaculos dinamicos:** `processar_obstaculos_dinamicos`
  converte coordenadas de impacto no espaco livre estatico para celulas do grid e
  chama `planejador.marcar_bloqueio(celulas, True)` apos persistencia (`passos_persistencia = 3`).
- **Histerese e expiracao:** celulas bloqueadas dinamicamente sao automaticamente
  liberadas via `planejador.marcar_bloqueio(celulas, False)` quando nao mais
  observadas apos tempo de expiracao (`passos_expiracao = 15`).
- **Deteccao de estagnacao e recuperacao:** `monitorar_estagnacao` monitora o
  deslocamento e variacao angular em uma janela temporal (`janela_estagnacao_passos = 20`);
  detecta travamento sem confundir curvas lentas ou giros no eixo (distinguindo
  pela taxa de variacao de theta), acionando maquina de estados de recuperacao
  composta por recuo (`ESTADO_RECUO`, v < 0) e giro (`ESTADO_GIRO`, w < 0).
- **Quebra deterministica de simetria:** `calcular_quebra_simetria` adiciona
  componente lateral a direita (`u_direita = (sin theta, -cos theta)`) em
  encontros frontais (|angulo| <= 30 deg), prevenindo deadlocks colineares.

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
