# Estes es un bench 100% descriptivo. 

# 25% del capital en BTC se compra y mantiene, 75 en cash.
# Esto no afecta ninguna decisión del modelo. Reutiliza la corrida de buy & hold, por lo que el tiempo y los costos son idénticos.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = "results"
SPLIT = "test"
INITIAL = 100_000
WEIGHT = 0.25
HOURS_PER_YEAR = 24 * 365


def compute_metrics(values, exposure, costs, n_trades):
    returns = values.pct_change().dropna()
    drawdown = values / values.cummax() - 1
    cum_return = values.iloc[-1] / INITIAL - 1
    max_dd = drawdown.min()
    return {
        "cum_return_%": round(100 * cum_return, 2),
        "max_drawdown_%": round(100 * max_dd, 2),
        "sharpe": round(returns.mean() / returns.std() * np.sqrt(HOURS_PER_YEAR), 2),
        "return_over_dd": round(cum_return / abs(max_dd), 2),
        "costs_$": round(costs, 2),
        "n_trades": n_trades,
        "avg_exposure_%": round(100 * exposure.mean(), 2),
    }


bh = pd.read_csv(f"{RESULTS}/history_{SPLIT}_Buy_and_hold.csv", parse_dates=["date"]).set_index("date")

# Buy & hold scales linearly with capital: 25 % of its value + 75 % in cash
bench_value = (1 - WEIGHT) * INITIAL + WEIGHT * bh["value"]
bench_exposure = WEIGHT * bh["value"] / bench_value  # drifts as the BTC price moves

# Sanity check: buy & hold metrics must match metrics_test.csv
official = pd.read_csv(f"{RESULTS}/metrics_{SPLIT}.csv", index_col=0)
print("Buy & hold (recalculated):", compute_metrics(bh["value"], bh["weight"], bh["cost"].sum(), 1))
print("Buy & hold (official):    ", official.loc["Buy & hold"].to_dict())

bench = compute_metrics(bench_value, bench_exposure, WEIGHT * bh["cost"].sum(), 1)
table = pd.concat([official, pd.DataFrame([bench], index=["Fixed 25 % BTC (descriptive)"])])
table.to_csv(f"{RESULTS}/metrics_{SPLIT}_with_benchmark.csv")  # does not overwrite the official file
print(table)

# Equity curve: final agent (seeds) vs. benchmark
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(bh.index, bh["value"], color="black", lw=1.2, label="Buy & hold")
ax.plot(bench_value.index, bench_value, color="gray", ls="--", lw=1.2, label="Fixed 25 % BTC")
for seed in [0, 1, 2]:
    h = pd.read_csv(f"{RESULTS}/history_{SPLIT}_RL_dd_light_s{seed}.csv", parse_dates=["date"]).set_index("date")
    ax.plot(h.index, h["value"], lw=0.9, label=f"RL dd_light s{seed}")
ax.set_title(f"Equity curve - {SPLIT} (with equal-exposure benchmark)")
ax.set_ylabel("Portfolio value ($)")
ax.legend(fontsize=8)
plt.savefig(f"{RESULTS}/equity_{SPLIT}_with_benchmark.png", dpi=150, bbox_inches="tight")
plt.show()