"""Evaluate RL agents and baselines on one split, under identical conditions.

  python -m src.evaluate --split val    # compare reward versions and seeds -> choose final_reward
  python -m src.evaluate --split test   # FINAL held-out evaluation. Run once, change nothing after.

On validation all reward versions are evaluated; on test only `final_reward`
from the config (the decision must be taken before seeing the test results).

Outputs (results/):
  metrics_<split>.csv           cumulative return, max drawdown, costs, trades, exposure
  equity_<split>.png            equity curves: RL vs buy & hold vs random
  positions_<split>.png         price and invested fraction of each RL agent (first seed)
  history_<split>_<name>.csv    step-by-step log (for anomaly analysis)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.agent import load_ensemble_policy, load_policy
from src.baselines import buy_and_hold, random_agent
from src.data import load_config, prepare
from src.environment import make_env
import numpy as np

N_RANDOM_RUNS = 20
OUT = Path("results")


def run_episode(env, policy):
    """Run a policy over the whole split; return the step log indexed by date."""
    obs, _ = env.reset()
    rows, done = [], False
    while not done:
        obs, _, terminated, truncated, info = env.step(policy(obs))
        rows.append(info)
        done = terminated or truncated
    return pd.DataFrame(rows).set_index("date")


BARS_PER_YEAR = 24 * 365   # hourly crypto bars (24/7); change if timeframe changes


def metrics(hist, initial_value):
    equity = pd.concat([pd.Series([initial_value]), hist["value"].reset_index(drop=True)])
    rets = equity.pct_change().dropna()
    cum = equity.iloc[-1] / initial_value - 1
    mdd = (equity / equity.cummax() - 1).min()
    return {
        "cum_return_%": 100 * cum,
        "max_drawdown_%": 100 * mdd,
        "sharpe": rets.mean() / rets.std() * np.sqrt(BARS_PER_YEAR) if rets.std() > 0 else 0.0,
        "return_over_dd": cum / abs(mdd) if mdd < 0 else 0.0,   # higher = better reward per unit of pain
        "costs_$": hist["cost"].sum(),
        "n_trades": int((hist["cost"] > 0).sum()),
        "avg_exposure_%": 100 * hist["weight"].mean(),
    }


def plot_equity(histories, random_curve, split):
    fig, ax = plt.subplots(figsize=(12, 5))
    histories["Buy & hold"]["value"].plot(ax=ax, color="black", lw=2, label="Buy & hold")
    random_curve.plot(ax=ax, color="gray", lw=1.5, label=f"Random (mean of {N_RANDOM_RUNS})")
    for name, h in histories.items():
        if name.startswith("RL"):
            h["value"].plot(ax=ax, lw=1, label=name)
    ax.set(title=f"Equity curve - {split}", ylabel="Portfolio value ($)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"equity_{split}.png", dpi=120)


def plot_positions(histories, rl_names, split):
    fig, axes = plt.subplots(len(rl_names) + 1, 1, figsize=(12, 2.5 * (len(rl_names) + 1)), sharex=True)
    price = histories["Buy & hold"]["price"]
    axes[0].plot(price.index, price, color="black")
    axes[0].set(title="Price (log)", yscale="log")
    for ax, name in zip(axes[1:], rl_names):
        ax.step(histories[name].index, histories[name]["weight"], where="post", lw=0.8)
        ax.set(title=f"{name}: invested fraction", ylim=(-0.05, 1.05))
    fig.tight_layout()
    fig.savefig(OUT / f"positions_{split}.png", dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = prepare(cfg)[args.split]
    initial, n_actions = cfg["env"]["initial_value"], len(cfg["env"]["positions"])
    rewards = list(cfg["rewards"]) if args.split == "val" else [cfg["final_reward"]]
    if args.split == "test":
        print(f"HELD-OUT EVALUATION with final_reward='{cfg['final_reward']}'. Do not tune after this.")
    OUT.mkdir(exist_ok=True)

    # Baselines (the reward type does not affect their actions or portfolio value).
    base_env = make_env(df, cfg, "log_return")
    histories = {"Buy & hold": run_episode(base_env, buy_and_hold(n_actions))}
    random_runs = [run_episode(base_env, random_agent(n_actions, s)) for s in range(N_RANDOM_RUNS)]

    # RL agents: best validation checkpoint of every (reward, seed).
    rl_names = []
    for reward in rewards:
        for seed in cfg["seeds"]:
            path = Path("models") / f"{reward}_seed{seed}" / "best_model.pt"
            if not path.exists():
                print(f"Missing {path} - run training first. Skipped.")
                continue
            name = f"RL {reward} s{seed}"
            env = make_env(df, cfg, reward)
            policy = load_policy(path, env.observation_space.shape[0], env.action_space.n, cfg)
            histories[name] = run_episode(env, policy)
            rl_names.append(name)


            # Ensemble of all seeds for the final reward (average Q-values).
            final = cfg["final_reward"]
            paths = [Path("models") / f"{final}_seed{s}" / "best_model.pt" for s in cfg["seeds"]]
            env = make_env(df, cfg, final)
            histories[f"RL {final} ensemble"] = run_episode(
                env, load_ensemble_policy(paths, env.observation_space.shape[0], env.action_space.n, cfg))

    # Metrics table: one row per run + mean rows (seed average is the headline number).
    rows = {name: metrics(h, initial) for name, h in histories.items()}
    rows[f"Random (mean of {N_RANDOM_RUNS})"] = pd.DataFrame([metrics(h, initial) for h in random_runs]).mean().to_dict()
    for reward in rewards:
        seeds_rows = [rows[n] for n in rl_names if n.startswith(f"RL {reward} ")]
        if seeds_rows:
            rows[f"RL {reward} (mean of seeds)"] = pd.DataFrame(seeds_rows).mean().to_dict()
    table = pd.DataFrame(rows).T.round(2)
    table.to_csv(OUT / f"metrics_{args.split}.csv")
    print(table.to_string())

    for name, h in histories.items():
        h.to_csv(OUT / f"history_{args.split}_{name.replace(' ', '_').replace('&', 'and')}.csv")
    random_curve = pd.concat([h["value"] for h in random_runs], axis=1).mean(axis=1)
    plot_equity(histories, random_curve, args.split)
    if rl_names:
        first_seed = [n for n in rl_names if n.endswith(f"s{cfg['seeds'][0]}")]
        plot_positions(histories, first_seed, args.split)
    print(f"Saved metrics and plots to {OUT}/")


if __name__ == "__main__":
    main()
