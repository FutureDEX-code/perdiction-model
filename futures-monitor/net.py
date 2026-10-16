#!/usr/bin/env python3
"""One-hidden-layer network for the next-session direction.

Same 10 features and same up/down label as train.py. The daily fit stays the
logistic regression in train.py. Edit this file to change the network.

Running it only prints a three-year holdout. It does not write the contract
model files and does not touch the paper book.
"""

from __future__ import annotations

import json
import math
import random

import train

HIDDEN = 8
EPOCHS = 300
LR = 0.05
L2 = 0.05
SEED = 0


def sigmoid(z: float) -> float:
    if z > 30:
        return 1.0
    if z < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def fit(
    xs: list[list[float]],
    ys: list[int],
    hidden: int = HIDDEN,
    epochs: int = EPOCHS,
    lr: float = LR,
    l2: float = L2,
    seed: int = SEED,
) -> tuple[list[list[float]], list[float], list[float], float]:
    """Return w1, b1, w2, b2. w1 is hidden by feature, w2 is one weight per hidden unit."""

    rng = random.Random(seed)
    dim = len(xs[0])
    w1 = [[(rng.random() - 0.5) * 0.2 for _ in range(dim)] for _ in range(hidden)]
    b1 = [0.0] * hidden
    w2 = [(rng.random() - 0.5) * 0.2 for _ in range(hidden)]
    b2 = 0.0
    n = len(xs)
    for _ in range(epochs):
        gw1 = [[0.0] * dim for _ in range(hidden)]
        gb1 = [0.0] * hidden
        gw2 = [0.0] * hidden
        gb2 = 0.0
        for x, y in zip(xs, ys):
            h = [sigmoid(b1[a] + sum(w1[a][j] * x[j] for j in range(dim))) for a in range(hidden)]
            p = sigmoid(b2 + sum(w2[a] * h[a] for a in range(hidden)))
            err = p - y
            gb2 += err
            for a in range(hidden):
                gw2[a] += err * h[a]
                dh = err * w2[a] * h[a] * (1.0 - h[a])
                gb1[a] += dh
                for j in range(dim):
                    gw1[a][j] += dh * x[j]
        for a in range(hidden):
            for j in range(dim):
                w1[a][j] -= lr * (gw1[a][j] / n + l2 * w1[a][j])
            b1[a] -= lr * gb1[a] / n
            w2[a] -= lr * (gw2[a] / n + l2 * w2[a])
        b2 -= lr * gb2 / n
    return w1, b1, w2, b2


def predict(
    model: tuple[list[list[float]], list[float], list[float], float],
    x: list[float],
) -> float:
    """Probability that the next close is above this close."""

    w1, b1, w2, b2 = model
    hidden = len(w2)
    h = [sigmoid(b1[a] + sum(w1[a][j] * x[j] for j in range(len(x)))) for a in range(hidden)]
    return sigmoid(b2 + sum(w2[a] * h[a] for a in range(hidden)))


def _cached_rows() -> list[dict]:
    series = {}
    for key in ("A", "M", "RM", "S", "SM", "BO"):
        bars = json.loads((train.DATA / f"{key}.json").read_text())
        series[key] = {day: float(close) for day, close in bars}
    news = {str(k): float(v) for k, v in json.loads((train.DATA / "news.json").read_text()).items()}
    return [row for row in train.samples(series, news) if row["day"] >= "2017-01-01"]


def _score(key: str, model, rows: list[dict], mean: list[float], scale: list[float]) -> tuple[int, float, float]:
    wins = 0
    aligned = 0.0
    for row in rows:
        prob = predict(model, train.apply_scale(row["x"], mean, scale))
        side = 1 if prob >= 0.5 else -1
        move = row["move"][key]
        wins += int((side == 1 and move > 0) or (side == -1 and move < 0))
        aligned += side * move
    n = len(rows)
    return n, wins / n, aligned / n


def main() -> None:
    rows = _cached_rows()
    last = rows[-1]["day"]
    start = f"{int(last[:4]) - 3}{last[4:]}"
    fit_rows = [row for row in rows if row["day"] < start]
    test_rows = [row for row in rows if row["day"] >= start]
    mean, scale = train.standardize([row["x"] for row in fit_rows])
    xs = [train.apply_scale(row["x"], mean, scale) for row in fit_rows]
    print(f"神经网络  拟合 {fit_rows[0]['day']} → {fit_rows[-1]['day']}  对答案 {test_rows[0]['day']} → {last}")
    print("每天都判断涨跌。这份结果不写进 model.json。")
    for key in ("A", "M", "RM"):
        model = fit(xs, [row["y"][key] for row in fit_rows])
        n, hit, aligned = _score(key, model, test_rows, mean, scale)
        print(f"  {train.NAMES[key]} {n} 天  胜率 {hit * 100:.1f}%  顺向 {aligned:+.3f}%")


if __name__ == "__main__":
    main()
