"""Entorno de inversión en un único activo, compatible con la API de Gymnasium.

Convención temporal, sin uso de información futura:
  1. En la vela t, el agente observa las variables calculadas con precios de cierre hasta t, inclusive, y la fracción actual del portafolio invertida.

  2. El agente selecciona una asignación objetivo del capital (0 %, 50 % o 100 %). Si la acción es diferente de la anterior, 
  el portafolio se rebalancea al precio de cierre close[t] y paga:

         comisión * |valor nominal negociado|

     Si la acción se repite, no se realiza ninguna operación, no se genera un nuevo costo y el peso de la posición varía con el movimiento del precio.

  3. El precio cambia desde close[t] hasta close[t+1] y el portafolio se revaloriza.

  4. La recompensa se calcula utilizando una única fórmula. 
  Las versiones difieren únicamente en los pesos asignados a cada penalización:

         r = log(V[t+1] / V[t])
             - dd_penalty * incremento_del_drawdown
             - turnover_penalty * |dw|

     v1, log_return:
         ambas penalizaciones son iguales a 0.

     v2, drawdown_penalty:
         dd_penalty = 2.

     v3, dd_turnover:
         dd_penalty = 0.5 y turnover_penalty = 0.001.

El agente nunca observa close[t+1] antes de tomar la decisión.
"""
import gymnasium as gym
import numpy as np

from src.data import FEATURES


class TradingEnv(gym.Env):
    def __init__(self, df, positions, fee, initial_value, dd_penalty=0.0,
                 turnover_penalty=0.0, reward_scale=1.0, episode_length=None):
        self.features = df[FEATURES].to_numpy(np.float32)
        self.prices = df["close"].to_numpy(np.float64)
        self.dates = df.index
        self.positions = positions
        self.fee = fee
        self.initial_value = initial_value
        self.dd_penalty = dd_penalty
        self.turnover_penalty = turnover_penalty
        self.reward_scale = reward_scale
        self.episode_length = episode_length      # None = run the whole split (evaluation)

        self.action_space = gym.spaces.Discrete(len(positions))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (len(FEATURES) + 1,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        last = len(self.prices) - 1
        if self.episode_length and self.episode_length < last:   # training: random window
            self.t = int(self.np_random.integers(0, last - self.episode_length))
            self.end = self.t + self.episode_length
        else:                                                    # evaluation: full period
            self.t, self.end = 0, last
        self.cash, self.units = float(self.initial_value), 0.0
        self.value = self.peak = float(self.initial_value)
        self.prev_action = None
        return self._obs(), {}

    def _obs(self):
        weight = self.units * self.prices[self.t] / self.value
        return np.append(self.features[self.t], weight).astype(np.float32)

    def step(self, action):
        action = int(action)
        price, value_before = self.prices[self.t], self.value

        # 1) Rebalance at close[t] only if the decision changed.
        cost, turnover, traded = 0.0, 0.0, action != self.prev_action
        if traded:
            target = self.positions[action]
            turnover = abs(target - self.units * price / self.value)   # |change in exposure|
            cost = self.fee * abs(target * self.value - self.units * price)
            self.value -= cost
            self.units = target * self.value / price
            self.cash = self.value - self.units * price
            self.prev_action = action
        weight = self.units * price / self.value

        # 2) Price moves to close[t+1]; revalue the portfolio.
        self.t += 1
        self.value = self.cash + self.units * self.prices[self.t]

        # 3) Reward: log growth (net of costs) minus optional risk / churn penalties.
        log_ret = np.log(self.value / value_before)
        old_dd = 1 - value_before / self.peak
        self.peak = max(self.peak, self.value)
        dd_increase = max(0.0, (1 - self.value / self.peak) - old_dd)
        reward = log_ret - self.dd_penalty * dd_increase - self.turnover_penalty * turnover

        truncated = self.t >= self.end
        info = {"date": self.dates[self.t], "action": action, "weight": weight,
                "traded": traded, "cost": cost, "value": self.value, "price": self.prices[self.t]}
        return self._obs(), float(reward * self.reward_scale), False, truncated, info


def make_env(df, cfg, reward, train=False):
    """Build the environment for a reward version. train=True -> random 30-day episodes."""
    e, r = cfg["env"], cfg["rewards"][reward]
    return TradingEnv(
        df, e["positions"], e["fee"], e["initial_value"],
        dd_penalty=r.get("dd_penalty", 0.0),
        turnover_penalty=r.get("turnover_penalty", 0.0),
        reward_scale=e["reward_scale"],
        episode_length=e["episode_length"] if train else None,
    )