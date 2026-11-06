# Parity fixtures

`cargo test` compares ireval with reference implementations through committed fixtures, so it needs no Python or network access. This directory holds the generators that produce them. Rerun a generator only to change the fixtures.

| Generator | Output | References |
| --- | --- | --- |
| `metrics_fixtures.py` | `tests/fixtures/parity/` | `trec_eval` 9.0.8 and 10.0-rc3 (built from source), pytrec-eval-terrier 0.5.11, ir-measures 0.4.3, ranx 0.3.21 |
| `stats_fixtures.py` | `tests/fixtures/stats/stats.json` | SciPy 1.18.1, NumPy 2.5.4, statsmodels 0.15.0 |

Both scripts declare their dependencies inline and run with `uv run --script`. `tests/fixtures/parity/manifest.json` records the package versions and the `trec_eval -v` output.

## Building trec_eval

`trec_eval` prints values with `%6.4f`, which is too coarse for floating-point parity. Build both versions with the format changed to `%.17g`, which round-trips a double:

```sh
work=$(mktemp -d)
git clone --branch v9.0.8 --depth 1 https://github.com/usnistgov/trec_eval "$work/te9"
git clone --depth 1 https://github.com/usnistgov/trec_eval "$work/te10"
for d in "$work/te9" "$work/te10"; do
  sed -i.bak 's/%6\.4f/%.17g/g' "$d/meas_print_single.c" "$d/meas_print_final.c"
  make -C "$d"
done
```

Versions used for the committed fixtures:

- v9.0.8: tag commit `d95ca64e14a47d763ae349fb65e6d8cde4141dbd`. Its `-v` output says 9.0.7 because the tag did not bump the version string.
- 10.0-rc3: `main` at `35cf096ba6d5ae08ca4803c056b31ed61a56a6d4`, 2026-10-15.

## Regenerating

```sh
uv run --script parity/metrics_fixtures.py \
  --trec-eval-9 "$work/te9/trec_eval" --trec-eval-10 "$work/te10/trec_eval" \
  --out tests/fixtures/parity            # --cases 100 --seed 20261107 by default
uv run --script parity/stats_fixtures.py --out tests/fixtures/stats/stats.json
cargo test --release --test parity --test stats_parity -- --nocapture
```

## How the metric fixtures are built

Each case has 1–14 queries. The generator randomizes:

- docid styles (`d12`, `D7`, `doc-003`, `42`, `x_9`), mixed so that byte-order tie-breaking matters;
- graded labels up to 4, occasional negative labels, and queries with no relevant document;
- queries missing from the run and run queries missing from the qrels;
- rankings of 1–60 documents with judged and unjudged documents;
- scores drawn from coarse grids, which produce ties, from integers, or from continuous values.

The rank column is deliberately wrong, so ignoring it is tested too. Every tenth case is written as JSON Lines.

Each case is evaluated in three configurations:

1. level 1, `skip`;
2. `-c` with level 1 or 2;
3. random level (0–3), missing mode and `-M`.

How each reference is run:

- **`trec_eval`** runs with `-q` and every cutoff measure at cutoffs 1, 3, 5, 10, 20 and 50. RR@k is `recip_rank` under `-M k`.
- **10.0-rc3** runs only where it does not abort: with `-c`, or when no qrels query is missing from the run.
- **pytrec_eval** is skipped for level 0, which it rejects. The ranking is truncated in Python for `-M`.
- **ir_measures (Judged@k) and ranx (nDCG, nDCG_exp)** receive a run whose scores encode `trec_eval`'s order, because their tie-breaking differs. Negative labels are dropped, since ireval treats them as unjudged. In `skip` mode, ir_measures' zero rows for absent queries are dropped.
- **`trec_eval -c` aggregate NumRel** is not compared when the level is not 1 (see the README's divergences).

Every expected file stores, for each metric and query (`all` is the aggregate), the value and the references that produced it. The generator prints any disagreement between references. The committed fixtures have none beyond one unit in the last place, between pytrec_eval/ranx and the locally built `trec_eval`.

## How the statistics fixtures are built

- **Paired samples.** n ranges from 2 to 250, in four shapes: P@10-like grids, continuous, binary and beta. There are also two degenerate samples: identical runs, and a constant difference.
- **Exact sign-flip p-values** come from rational enumeration of the differences rounded to 12 decimals. Rounding recovers the intended values, so 0.3 − 0.2 is 0.1 rather than 0.09999999999999998. SciPy's exact `permutation_test` is recorded too.
- **Monte Carlo p-values** use SciPy with 200,000 resamples.
- **Bootstrap intervals** use `scipy.stats.bootstrap(method="percentile")` with 200,000 resamples. They are compared only on samples with distinct differences, because on grid data the percentile jumps between atoms.
