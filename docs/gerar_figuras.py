"""Gera figuras e tempos offline, sem depender do CoppeliaSim.

Execute da raiz do projeto: .venv/bin/python docs/gerar_figuras.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from Agente1 import PlanejadorGlobal  # noqa: E402


def vizinhos_anteriores(planejador: PlanejadorGlobal, celula: tuple[int, int]):
    """Implementacao anterior, mantida somente para comparacao de tempo."""
    linha, coluna = celula
    vizinhos = []
    for delta_linha in (-1, 0, 1):
        for delta_coluna in (-1, 0, 1):
            if delta_linha == 0 and delta_coluna == 0:
                continue
            candidato = linha + delta_linha, coluna + delta_coluna
            if planejador._celula_valida(candidato):
                vizinhos.append(candidato)
    return vizinhos


def construir_planejador(imagem: np.ndarray) -> PlanejadorGlobal:
    return PlanejadorGlobal(
        imagem,
        escala=(30.0 / imagem.shape[1], 30.0 / imagem.shape[0]),
        origem=(-15.0, -15.0),
    )


def medir_rota(imagem, pose, goal, anterior):
    planejador = construir_planejador(imagem)
    if anterior:
        planejador._vizinhos = lambda celula: vizinhos_anteriores(planejador, celula)
    planejador.definir_goal(goal)
    inicio = time.perf_counter()
    rota = planejador.caminho_completo(pose)
    tempo = time.perf_counter() - inicio
    if not rota:
        raise RuntimeError(f"Sem rota para {pose} -> {goal}")
    return tempo, rota


def main() -> None:
    imagem = np.asarray(Image.open(RAIZ / "images" / "tp2mapa.png").convert("RGB"))
    cenarios = [
        {"nome": "A", "pose": (8.03, 14.28, 3.05), "goal": (-13.15, 11.95)},
        {"nome": "B", "pose": (8.0, -12.0, math.pi / 2), "goal": (-13.35, 11.95)},
    ]
    resultados = []
    for cenario in cenarios:
        tempos = {"anterior": [], "atual": []}
        rotas = {}
        for repeticao in range(2):
            for anterior in (repeticao == 0, repeticao != 0):
                rotulo = "anterior" if anterior else "atual"
                tempo, rota = medir_rota(
                    imagem, cenario["pose"], cenario["goal"], anterior
                )
                tempos[rotulo].append(tempo)
                rotas[rotulo] = rota
        if rotas["anterior"] != rotas["atual"]:
            raise RuntimeError("A otimizacao mudou a rota do planejador")
        resultados.append(
            {
                **cenario,
                "tempos_s": tempos,
                "medianas_s": {k: statistics.median(v) for k, v in tempos.items()},
                "waypoints": len(rotas["atual"]),
                "comprimento_rota_m": sum(
                    math.dist(a, b) for a, b in zip(rota, rota[1:])
                ),
                "rota": rotas["atual"],
            }
        )

    fig, eixos = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    extent = (-15, 15, -15, 15)
    for eixo, resultado in zip(eixos, resultados):
        eixo.imshow(np.flipud(imagem), origin="lower", extent=extent, alpha=0.8)
        rota = np.asarray(resultado["rota"])
        eixo.plot(rota[:, 0], rota[:, 1], color="#e76f00", linewidth=2, label="rota D* Lite")
        x, y, theta = resultado["pose"]
        gx, gy = resultado["goal"]
        eixo.scatter([x], [y], color="#6b37b7", s=85, label="pose inicial")
        eixo.arrow(x, y, 0.8 * math.cos(theta), 0.8 * math.sin(theta),
                   color="#6b37b7", width=0.07, head_width=0.4)
        eixo.scatter([gx], [gy], marker="*", color="#ce334c", s=180, label="goal")
        eixo.set(xlim=(-15, 15), ylim=(-15, 15), xlabel="x (m)", ylabel="y (m)",
                 title=f"Cenário {resultado['nome']}: {resultado['waypoints']} waypoints")
        eixo.set_aspect("equal")
        eixo.legend(loc="lower right", fontsize=8)
    fig.savefig(RAIZ / "docs" / "rotas_offline.svg")
    plt.close(fig)

    fig, eixo = plt.subplots(figsize=(7, 4), constrained_layout=True)
    posicoes = np.arange(len(resultados))
    largura = 0.34
    eixo.bar(posicoes - largura / 2,
             [r["medianas_s"]["anterior"] for r in resultados], largura,
             label="vizinhos anteriores", color="#9aa4b2")
    eixo.bar(posicoes + largura / 2,
             [r["medianas_s"]["atual"] for r in resultados], largura,
             label="vizinhos atuais", color="#376fa8")
    eixo.set(xticks=posicoes, xticklabels=[f"Cenário {r['nome']}" for r in resultados],
             ylabel="tempo do primeiro planejamento (s)",
             title="Busca D* Lite inicial no mapa de 300 × 300 células")
    eixo.legend()
    fig.savefig(RAIZ / "docs" / "tempo_planejamento.svg")
    plt.close(fig)

    # Tempos transcritos literalmente da saída preservada em
    # execucao_coppelia.ipynb (primeira execução medida no simulador).
    tempos_goals = [99.20, 162.70, 216.05, 225.30, 260.25]
    fig, eixo = plt.subplots(figsize=(7, 4), constrained_layout=True)
    eixo.step([0.0, *tempos_goals], list(range(6)), where="post",
              color="#1764a0", linewidth=2)
    eixo.scatter(tempos_goals, list(range(1, 6)), color="#1764a0", zorder=3)
    eixo.set(xlabel="tempo simulado (s)", ylabel="goals alcançados",
             xlim=(0, 275), ylim=(-0.1, 5.3),
             title="Experimento 1: progressão medida no CoppeliaSim")
    eixo.grid(alpha=0.25)
    fig.savefig(RAIZ / "docs" / "progresso_coppelia_1.svg")
    plt.close(fig)

    dados = [
        {chave: valor for chave, valor in resultado.items() if chave != "rota"}
        for resultado in resultados
    ]
    (RAIZ / "docs" / "metricas_offline.json").write_text(
        json.dumps(dados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(dados, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
