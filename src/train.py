"""Entrena un agente DQN de implementación propia por cada combinación de
versión de la recompensa y semilla aleatoria.

Protocolo:
  * El agente aprende utilizando la partición de ENTRENAMIENTO mediante
    episodios aleatorios de 30 días.

  * Cada `eval_freq` pasos, el agente recorre toda la partición de VALIDACIÓN
    utilizando una política voraz, sin exploración (epsilon = 0).

    Se guarda el checkpoint que obtiene la mayor recompensa de validación.
    El entrenamiento se detiene después de `patience` evaluaciones consecutivas
    sin mejora.

    La partición de PRUEBA nunca se utiliza durante este proceso.

Uso:
  python -m src.train
  # Entrena todas las versiones de la recompensa con todas las semillas.

  python -m src.train --reward dd_turnover --seed 0
  # Ejecuta un único entrenamiento con la recompensa `dd_turnover` y la
  # semilla 0.

Archivos generados por cada ejecución en
models/<reward>_seed<seed>/:

  best_model.pt
      Pesos del checkpoint que obtuvo el mejor resultado en validación.

  val_curve.csv
      Registro del paso de entrenamiento, la recompensa de validación,
      el valor de epsilon y la pérdida promedio. Este archivo contiene la
      curva de aprendizaje utilizada en el informe.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent import DQNAgent
from src.data import load_config, prepare
from src.environment import make_env


def linear_eps(step, p):
    """Epsilon decays linearly from initial to final over the first exploration_fraction of steps."""
    frac = min(1.0, step / (p["exploration_fraction"] * p["total_timesteps"]))
    return p["exploration_initial_eps"] + frac * (p["exploration_final_eps"] - p["exploration_initial_eps"])


def run_validation(env, agent):
    """Greedy pass over the whole validation split; returns the total reward."""
    obs, _ = env.reset()
    total, done = 0.0, False
    while not done:
        obs, reward, terminated, truncated, _ = env.step(agent.act(obs, eps=0.0))
        total += reward
        done = terminated or truncated
    return total


def train_one(splits, cfg, reward, seed):
    p = cfg["dqn"]
    out = Path("models") / f"{reward}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)

    train_env = make_env(splits["train"], cfg, reward, train=True)
    val_env = make_env(splits["val"], cfg, reward)
    agent = DQNAgent(train_env.observation_space.shape[0], train_env.action_space.n, cfg, seed)

    obs, _ = train_env.reset(seed=seed)
    best, no_improve, losses, curve = -np.inf, 0, [], []
    for step in range(1, p["total_timesteps"] + 1):
        eps = linear_eps(step, p)
        action = agent.act(obs, eps)
        next_obs, r, terminated, truncated, _ = train_env.step(action)
        # Time-limit truncation is not a real terminal state: keep bootstrapping (done = terminated only).
        agent.buffer.add(obs, action, r, next_obs, float(terminated))
        obs = next_obs
        if terminated or truncated:
            obs, _ = train_env.reset()

        if step > p["learning_starts"] and step % p["train_freq"] == 0:
            losses.append(agent.update())
        if step % p["target_update_interval"] == 0:
            agent.sync_target()

        if step % p["eval_freq"] == 0:
            val_reward = run_validation(val_env, agent)
            curve.append({"step": step, "val_reward": val_reward, "eps": eps,
                          "mean_loss": float(np.mean(losses)) if losses else np.nan})
            losses = []
            if val_reward > best:
                best, no_improve = val_reward, 0
                agent.save(out / "best_model.pt")
            else:
                no_improve += 1
            print(f"[{reward} | seed {seed}] step {step:>7}  val {val_reward:8.2f}  best {best:8.2f}  eps {eps:.2f}")
            if no_improve >= p["patience"]:
                print(f"Early stop: {p['patience']} evaluations without improvement.")
                break

    pd.DataFrame(curve).to_csv(out / "val_curve.csv", index=False)
    print(f"[{reward} | seed {seed}] done. best val reward {best:.2f} -> {out}/best_model.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--reward", help="train only this reward version")
    ap.add_argument("--seed", type=int, help="train only this seed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    splits = prepare(cfg)
    rewards = [args.reward] if args.reward else list(cfg["rewards"])
    seeds = [args.seed] if args.seed is not None else cfg["seeds"]
    for reward in rewards:
        for seed in seeds:
            train_one(splits, cfg, reward, seed)


if __name__ == "__main__":
    main()
