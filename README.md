# TP2 — Planejamento e Navegação

Implementação em Python para navegar com um Pioneer P3-DX pelo mapa de 30 × 30 m do trabalho. O agente combina D* Lite em um grid inflado, tratamento de obstáculos móveis pelo Hokuyo e controle diferencial. O método `Agent.movement(sim, img, goal_list)` retorna `True` quando o goal corrente está a até 0,10 m do robô; o servidor deve manter a lista na ordem da rodada.

## Integrantes

- Nome completo: Isaac Reyes Alves de Abreu — matrícula: 2025050342
- Nome completo: Pedro Luiz Silva — matrícula: 2024006129

## Execução

1. Abra `cenas/MapaTP2.ttt` no CoppeliaSim, com o add-on ZMQ Remote API ativo, e deixe a simulação parada.
2. Instale as dependências com `uv sync`.
3. Execute `uv run python main.py` (ou `.venv/bin/python main.py`). O script conecta ao simulador, gera cinco goals reproduzíveis e navega. A visualização interativa do agente pode ser habilitada exportando a variável de ambiente `TP2_VISUALIZAR=1`.

> Caso prefira, pode rodar tudo via `execucao_coppelia.ipynb` também.
