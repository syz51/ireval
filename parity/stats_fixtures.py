# /// script
# requires-python = "==3.12.*"
# dependencies = ["scipy==1.18.1", "numpy==2.5.4", "statsmodels==0.15.0"]
# ///
"""Reference values for ireval's statistics, from SciPy, NumPy and statsmodels.

Writes tests/fixtures/stats/stats.json, read by tests/stats_parity.rs.

Usage (from the repository root):
    uv run --script parity/stats_fixtures.py --out tests/fixtures/stats/stats.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import special, stats
from statsmodels.stats.multitest import multipletests


def num(x):
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def exact_sign_flip_p(d: list[float]) -> float:
    """Exact two-sided sign-flip p-value in rational arithmetic: the share of
    the 2^m sign assignments of the non-zero differences whose |sum| is at
    least the observed |sum|. Differences are rounded to 12 decimals first,
    recovering the intended values (0.1 - 0.2 + 0.2 - 0.1 is 0, not the
    binary-rounding residue that a literal float sum leaves)."""
    nz = [Fraction(repr(round(x, 12))) for x in d if round(x, 12) != 0]
    observed = abs(sum(nz))
    hits = 0
    for signs in itertools.product((1, -1), repeat=len(nz)):
        if abs(sum(s * x for s, x in zip(signs, nz))) >= observed:
            hits += 1
    return hits / 2 ** len(nz)


def paired_samples(rng: np.random.Generator):
    cases = []
    for n in [2, 3, 4, 5, 6, 8, 10, 12, 13, 20, 30, 50, 100, 250]:
        for style in ["grid", "continuous", "binary", "shifted"]:
            if style == "grid":  # metric values like P@10: multiples of 0.1
                a = rng.integers(0, 11, n) / 10
                b = np.clip(a + rng.integers(-3, 4, n) / 10, 0, 1)
            elif style == "continuous":
                a = rng.uniform(0, 1, n)
                b = np.clip(a + rng.normal(0.03, 0.1, n), 0, 1)
            elif style == "binary":  # Success@k
                a = rng.integers(0, 2, n).astype(float)
                b = rng.integers(0, 2, n).astype(float)
            else:
                a = rng.beta(2, 5, n)
                b = rng.beta(2.5, 5, n)
            cases.append((a, b))
    # Degenerate cases: identical runs, constant non-zero difference.
    a = rng.uniform(0, 1, 10)
    cases.append((a, a.copy()))
    cases.append((np.full(6, 0.25), np.full(6, 0.5)))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261107)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    t_dist = []
    for df in [1, 2, 3, 4, 5, 9, 10, 29, 30, 49, 99, 249, 1000, 5000]:
        for t in [0.0, 0.05, 0.5, 1.0, 1.96, 2.5, 3.0, 5.0, 10.0, 30.0, 100.0, -2.0]:
            t_dist.append({"t": t, "df": df, "p": num(2 * stats.t.sf(abs(t), df))})

    ln_gamma = [{"x": x, "value": num(special.gammaln(x))} for x in
                [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.7, 10.0, 15.5, 50.0, 124.5, 1000.0, 2500.5]]

    paired = []
    for a, b in paired_samples(rng):
        d = b - a
        tt = stats.ttest_rel(b, a)
        sd = np.std(d, ddof=1)
        case = {
            "a": [float(x) for x in a],
            "b": [float(x) for x in b],
            "t": num(tt.statistic),
            "p_t": num(tt.pvalue),
            "mean_diff": num(np.mean(d)),
            "cohens_d": num(np.mean(d) / sd) if sd > 0 else "nan",
        }
        m = int(np.count_nonzero(d))
        if m <= 14:
            case["p_exact"] = exact_sign_flip_p([float(x) for x in d])
            res = stats.permutation_test(
                (b, a),
                lambda x, y: np.mean(x - y),
                permutation_type="samples",
                alternative="two-sided",
                n_resamples=np.inf,
            )
            case["p_exact_scipy"] = num(res.pvalue)
        elif len(d) <= 100:
            res = stats.permutation_test(
                (b, a),
                lambda x, y, axis: np.mean(x - y, axis=axis),
                permutation_type="samples",
                alternative="two-sided",
                n_resamples=200_000,
                vectorized=True,
                rng=rng,
            )
            case["p_mc_scipy"] = num(res.pvalue)
            case["p_mc_scipy_resamples"] = 200_000
        # Percentile intervals are only comparable between two Monte Carlo
        # runs when the bootstrap distribution is effectively continuous: on
        # grid data the quantile jumps between atoms.
        if len(d) >= 10 and len(np.unique(np.round(d, 9))) == len(d):
            boot = stats.bootstrap(
                (d,),
                np.mean,
                method="percentile",
                n_resamples=200_000,
                confidence_level=0.95,
                rng=rng,
            )
            case["ci_scipy"] = [num(boot.confidence_interval.low), num(boot.confidence_interval.high)]
            case["ci_scipy_resamples"] = 200_000
        paired.append(case)

    quantiles = []
    for n in [1, 2, 3, 7, 10, 101]:
        x = np.sort(rng.normal(0, 1, n))
        for q in [0.0, 0.025, 0.1, 0.5, 0.9, 0.975, 1.0]:
            quantiles.append({"x": [float(v) for v in x], "q": q, "value": num(np.quantile(x, q))})

    corrections = []
    for m in [1, 2, 3, 5, 10, 50]:
        for _ in range(3):
            p = rng.uniform(0, 0.2, m)
            if m >= 3:
                p[1] = p[0]  # ties
            p = [float(round(v, 6)) for v in p]
            corrections.append({
                "p": p,
                "holm": [num(v) for v in multipletests(p, method="holm")[1]],
                "bh": [num(v) for v in multipletests(p, method="fdr_bh")[1]],
            })

    out = {
        "versions": {p: importlib.metadata.version(p) for p in ["scipy", "numpy", "statsmodels"]},
        "seed": args.seed,
        "t_dist": t_dist,
        "ln_gamma": ln_gamma,
        "paired": paired,
        "quantiles": quantiles,
        "corrections": corrections,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
