"""DQN implementado desde cero en PyTorch, siguiendo a Mnih et al. (2015).

Componentes:
  * QNetwork      Red neuronal multicapa que transforma el estado en los
                  valores Q(s, a) correspondientes a cada acción.
  * ReplayBuffer  Almacena transiciones (s, a, r, s', done) y extrae
                  minilotes aleatorios, lo que reduce la correlación entre
                  horas consecutivas.
  * DQNAgent      Selección de acciones mediante una estrategia epsilon-greedy
                  y actualización por diferencia temporal (TD) utilizando
                  una red objetivo.

Regla de actualización, correspondiente a un paso de gradiente sobre un minilote:
    y      = r + gamma * (1 - done) * max_a' Q_target(s', a')
    loss   = Huber(Q(s, a) - y)

La red objetivo es una copia congelada de la red principal. Se actualiza cada
`target_update_interval` pasos para evitar que el valor objetivo de la regresión
cambie en cada paso de entrenamiento.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """MLP: obs_dim -> hidden layers (ReLU) -> one Q-value per action."""

    def __init__(self, obs_dim, n_actions, net_arch):
        super().__init__()
        layers, last = [], obs_dim
        for size in net_arch:
            layers += [nn.Linear(last, size), nn.ReLU()]
            last = size
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Fixed-size circular buffer. The oldest transitions are overwritten."""

    def __init__(self, size, obs_dim, rng):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.next_obs = np.zeros((size, obs_dim), np.float32)
        self.actions = np.zeros(size, np.int64)
        self.rewards = np.zeros(size, np.float32)
        self.dones = np.zeros(size, np.float32)
        self.size, self.pos, self.full, self.rng = size, 0, False, rng

    def add(self, obs, action, reward, next_obs, done):
        i = self.pos
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.dones[i] = next_obs, done
        self.pos = (self.pos + 1) % self.size
        self.full = self.full or self.pos == 0

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size if self.full else self.pos, batch_size)
        to_t = torch.as_tensor
        return (to_t(self.obs[idx]), to_t(self.actions[idx]), to_t(self.rewards[idx]),
                to_t(self.next_obs[idx]), to_t(self.dones[idx]))


class DQNAgent: # acá se implementa la política epsilon-greedy y la actualización de la red Q por TD.
    def __init__(self, obs_dim, n_actions, cfg, seed):
        p = cfg["dqn"]
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.n_actions, self.gamma, self.batch_size = n_actions, p["gamma"], p["batch_size"]
        self.q = QNetwork(obs_dim, n_actions, p["net_arch"])
        self.q_target = QNetwork(obs_dim, n_actions, p["net_arch"])
        self.sync_target()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=p["learning_rate"])
        self.buffer = ReplayBuffer(p["buffer_size"], obs_dim, self.rng)

    def act(self, obs, eps=0.0):
        """Epsilon-greedy: random action with probability eps, otherwise argmax Q."""
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            return int(self.q(torch.as_tensor(obs).unsqueeze(0)).argmax(1).item())
# el paso de actualización de la red Q se realiza en el método update, que toma un minilote aleatorio del buffer de repetición y realiza un paso de gradiente utilizando la pérdida Huber.
    def update(self):
        """One gradient step on a random mini-batch. Returns the loss."""
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        with torch.no_grad():
            next_q = self.q_target(next_obs).max(1).values
            target = rewards + self.gamma * (1 - dones) * next_q
        q_sa = self.q(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_sa, target)            # Huber loss: robust to fat-tailed rewards
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        self.q_target.load_state_dict(self.q.state_dict())

    def save(self, path):
        torch.save(self.q.state_dict(), path)


def load_policy(path, obs_dim, n_actions, cfg): # función que nos permite cargar una red Q entrenada y devolver una política codiciosa obs -> acción.
    """Load a saved Q-network and return a greedy policy obs -> action."""
    q = QNetwork(obs_dim, n_actions, cfg["dqn"]["net_arch"])
    q.load_state_dict(torch.load(path))
    q.eval()

    def policy(obs):
        with torch.no_grad():
            return int(q(torch.as_tensor(obs).unsqueeze(0)).argmax(1).item())
    return policy


def load_ensemble_policy(paths, obs_dim, n_actions, cfg): # función que nos permite tener un conjunto de redes Q entrenadas y promediar sus valores Q para tomar decisiones de manera más robusta.
    """Average the Q-values of several trained networks and act greedily."""
    nets = []
    for path in paths:
        q = QNetwork(obs_dim, n_actions, cfg["dqn"]["net_arch"])
        q.load_state_dict(torch.load(path))
        q.eval()
        nets.append(q)

    def policy(obs): # obs -> action
        x = torch.as_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            return int(torch.stack([q(x) for q in nets]).mean(0).argmax(1).item())
    return policy