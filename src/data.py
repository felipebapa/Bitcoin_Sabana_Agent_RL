"""Data download, feature engineering, temporal split and normalization.

Lookahead rules enforced here:
  * Every feature at bar t uses only closes up to and including bar t
    (rolling windows look backwards only, no centered windows, no bfill).
  * Normalization statistics (mean/std) are computed on the TRAIN split only
    and then applied unchanged to validation and test.
  * Features are computed on the full series BEFORE splitting. This is safe
    because they are backward-looking: the first validation bars use train-period
    history, which was already known at that moment.

Run `python -m src.data` to download the dataset and run the lookahead check.
"""
import datetime

import numpy as np
import pandas as pd
import yaml

FEATURES = ["ret_1", "ret_short", "ma_gap", "volatility", "rsi"]


def load_config(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def download(cfg):
    """Download OHLCV candles with the gym-trading-env downloader (saved to data/)."""
    from gym_trading_env.downloader import download as gte_download

    d = cfg["data"]
    gte_download(
        exchange_names=[d["exchange"]],
        symbols=[d["symbol"]],
        timeframe=d["timeframe"],
        dir="data",
        since=datetime.datetime.fromisoformat(d["since"]),
    )


def load_prices(path):
    """Load OHLCV data from .pkl (downloader) or .csv (needs a 'timestamp' column)."""
    if path.endswith(".csv"):
        df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    else:
        df = pd.read_pickle(path)
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    return df[~df.index.duplicated(keep="first")]


def add_features(df, short_window, long_window, rsi_window):
    """Add backward-looking, scale-free features. Drops the warm-up rows."""
    df = df.copy()
    close = df["close"]
    log_ret = np.log(close).diff()

    df["ret_1"] = log_ret                                                    # last-bar move
    df["ret_short"] = np.log(close / close.shift(short_window))              # 1-day momentum
    df["ma_gap"] = close / close.rolling(long_window).mean() - 1             # distance to weekly trend
    df["volatility"] = log_ret.rolling(short_window).std()                   # recent risk

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_window).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_window).mean()
    df["rsi"] = (100 * gain / (gain + loss)).fillna(50) / 50 - 1             # RSI rescaled to [-1, 1]

    return df.dropna()


def split(df, train_end, val_end):
    """Strict temporal split: train < validation < test, no overlap."""
    return {
        "train": df.loc[:train_end],
        "val": df.loc[pd.Timestamp(train_end) + pd.Timedelta(days=1): val_end],
        "test": df.loc[pd.Timestamp(val_end) + pd.Timedelta(days=1):],
    }


def normalize(splits):
    """Z-score features with TRAIN statistics only; clip extreme values."""
    mean = splits["train"][FEATURES].mean()
    std = splits["train"][FEATURES].std()
    out = {}
    for name, part in splits.items():
        part = part.copy()
        part[FEATURES] = ((part[FEATURES] - mean) / std).clip(-5, 5)
        out[name] = part
    return out


def prepare(cfg):
    """Full pipeline: load -> features -> split -> normalize."""
    f = cfg["features"]
    df = add_features(load_prices(cfg["data"]["path"]), f["short_window"], f["long_window"], f["rsi_window"])
    return normalize(split(df, cfg["data"]["train_end"], cfg["data"]["val_end"]))


def check_no_lookahead(raw, cfg, n_checks=5):
    """Features at bar t must not change when all data after t is removed."""
    f = cfg["features"]
    full = add_features(raw, f["short_window"], f["long_window"], f["rsi_window"])
    rng = np.random.default_rng(0)
    for cut in rng.integers(len(raw) // 2, len(raw), n_checks):
        truncated = add_features(raw.iloc[:cut], f["short_window"], f["long_window"], f["rsi_window"])
        t = truncated.index[-1]
        assert np.allclose(truncated.loc[t, FEATURES], full.loc[t, FEATURES]), f"Lookahead at {t}"
    print(f"Lookahead check passed ({n_checks} random cut points).")


if __name__ == "__main__":
    cfg = load_config()
    download(cfg)
    raw = load_prices(cfg["data"]["path"])
    check_no_lookahead(raw, cfg)
    for name, part in prepare(cfg).items():
        print(f"{name:5s}: {part.index[0]} -> {part.index[-1]}  ({len(part)} bars)")
