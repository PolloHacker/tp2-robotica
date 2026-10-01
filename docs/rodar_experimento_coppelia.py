"""Executa o segundo experimento no CoppeliaSim aberto e registra dados brutos.

Requer a cena MapaTP2.ttt parada e o add-on ZMQ Remote API ativo.
Execute da raiz: .venv/bin/python docs/rodar_experimento_coppelia.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from Agente1 import Agent, Sensoriamento  # noqa: E402

POSE_INICIAL = (8.0, -12.0, math.pi / 2)
GOALS = [(-13.35, 11.95), (10.0, -10.0)]
DURACAO_MAXIMA_S = 350.0


def main() -> None:
    imagem = np.asarray(Image.open(RAIZ / "images" / "tp2mapa.png").convert("RGB"))
    cliente = RemoteAPIClient()
    sim = cliente.require("sim")
    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError("A simulacao precisa estar parada antes do experimento")

    agente = Agent(1)
    sensor = Sensoriamento()
    robo = sim.getObject("/PioneerP3DX[0]")
    pose_original = sim.getObjectPose(robo, sim.handle_world)
    x0, y0, theta0 = POSE_INICIAL
    pose_experimento = [
        x0, y0, pose_original[2], 0.0, 0.0,
        math.sin(theta0 / 2), math.cos(theta0 / 2),
    ]
    registros: list[tuple[float, float, float, float, int, int, int]] = []
    chegadas: list[dict[str, float | int]] = []
    indice_goal = 0
    iniciada = False
    cliente.setStepping(True)
    try:
        sim.setObjectPose(robo, pose_experimento, sim.handle_world)
        pose_real = sensor.obter_pose(sim, agente.robotname)
        print(f"Pose posicionada: {pose_real}", flush=True)
        sim.startSimulation()
        iniciada = True
        proximo_aviso = 20.0

        while True:
            tempo = float(sim.getSimulationTime())
            if tempo >= DURACAO_MAXIMA_S:
                break
            pose = sensor.obter_pose(sim, agente.robotname)
            bloqueios = len(agente._camada_reativa._celulas_bloqueadas)
            recuperacao = int(
                agente._camada_reativa._estado_recuperacao
                != agente._camada_reativa.ESTADO_NORMAL
            )
            registros.append((tempo, *pose, indice_goal, bloqueios, recuperacao))
            if agente.movement(sim, imagem, GOALS):
                distancia = math.dist(pose[:2], GOALS[indice_goal])
                indice_goal += 1
                chegadas.append({"goal": indice_goal, "tempo_s": tempo, "erro_m": distancia})
                print(f"Goal {indice_goal}/{len(GOALS)}: t={tempo:.2f}s, erro={distancia:.3f}m", flush=True)
                if indice_goal == len(GOALS):
                    break
            cliente.step()
            if tempo >= proximo_aviso:
                print(f"t={tempo:.1f}s; goals={indice_goal}; pose=({pose[0]:.2f},{pose[1]:.2f})", flush=True)
                proximo_aviso += 20.0

        tempo_final = float(sim.getSimulationTime())
        pose_final = sensor.obter_pose(sim, agente.robotname)
        registros.append((tempo_final, *pose_final, indice_goal,
                          len(agente._camada_reativa._celulas_bloqueadas),
                          int(agente._camada_reativa._estado_recuperacao
                              != agente._camada_reativa.ESTADO_NORMAL)))
    finally:
        if iniciada:
            agente._parar_rodas(sim)
            sim.stopSimulation()
            for _ in range(40):
                if sim.getSimulationState() == sim.simulation_stopped:
                    break
                cliente.step()
        if sim.getSimulationState() == sim.simulation_stopped:
            sim.setObjectPose(robo, pose_original, sim.handle_world)
        else:
            print("AVISO: simulacao nao parou; pose original nao restaurada", flush=True)

    saida = RAIZ / "docs"
    with (saida / "experimento_coppelia_2.csv").open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(("tempo_s", "x_m", "y_m", "theta_rad", "goals_alcancados", "bloqueios", "recuperacao"))
        escritor.writerows(registros)

    pontos = np.asarray(registros, dtype=float)
    distancia_m = float(np.linalg.norm(np.diff(pontos[:, 1:3], axis=0), axis=1).sum())
    metricas = {
        "pose_inicial_solicitada": POSE_INICIAL,
        "goals": GOALS,
        "tempo_final_s": tempo_final,
        "goals_alcancados": indice_goal,
        "chegadas": chegadas,
        "pose_final": pose_final,
        "distancia_percorrida_m": distancia_m,
        "amostras": len(registros),
        "passos_recuperacao": int(np.sum(pontos[:, 6])),
        "max_bloqueios_dinamicos": int(np.max(pontos[:, 5])),
    }
    (saida / "experimento_coppelia_2.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fig, eixos = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    eixo = eixos[0]
    eixo.imshow(np.flipud(imagem), origin="lower", extent=(-15, 15, -15, 15), alpha=0.8)
    eixo.plot(pontos[:, 1], pontos[:, 2], color="#1764a0", linewidth=1.5, label="trajetoria medida")
    eixo.scatter([x0], [y0], color="#6b37b7", s=80, label="inicio")
    for i, goal in enumerate(GOALS, 1):
        eixo.scatter(*goal, color="#1f9d55" if i <= indice_goal else "#d63c3c", marker="*", s=150)
        eixo.annotate(f"G{i}", goal, xytext=(5, 5), textcoords="offset points")
    eixo.set(xlabel="x (m)", ylabel="y (m)", title="Experimento 2: trajetória no CoppeliaSim")
    eixo.set_aspect("equal")
    eixo.legend()
    eixo = eixos[1]
    eixo.step(pontos[:, 0], pontos[:, 4], where="post", color="#1764a0", label="goals alcancados")
    eixo.set(xlabel="tempo simulado (s)", ylabel="goals alcancados", ylim=(-0.1, len(GOALS) + 0.3),
             title="Progresso e obstáculos dinâmicos")
    eixo2 = eixo.twinx()
    eixo2.plot(pontos[:, 0], pontos[:, 5], color="#d27316", alpha=0.65,
               label="celulas dinamicas bloqueadas")
    eixo2.set_ylabel("celulas dinamicas")
    fig.savefig(saida / "experimento_coppelia_2.svg")
    plt.close(fig)
    print(json.dumps(metricas, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
