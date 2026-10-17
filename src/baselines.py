"""Políticas de referencia. Utilizan la misma interfaz que el agente de RL: policy(obs) -> action.

Ambas políticas se ejecutan en el mismo entorno que el agente, por lo que pagan los mismos costos de transacción y se evalúan durante el mismo período.
"""
import numpy as np


def buy_and_hold(n_actions):
    """Always choose the largest allocation (100%). Trades once, at the first bar."""
    return lambda obs: n_actions - 1


def random_agent(n_actions, seed):
    """Uniform random allocation at every bar (reference for 'no skill')."""
    rng = np.random.default_rng(seed)
    return lambda obs: int(rng.integers(n_actions))
