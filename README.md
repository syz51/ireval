# ireval

Information-retrieval evaluation with `trec_eval` semantics, and paired significance tests across runs.

- Every metric reproduces a named `trec_eval` or `ir_measures` measure. Randomized parity tests against `trec_eval` 9.0.8 and 10.0-rc3, `pytrec_eval`, `ir_measures` and `ranx` agree to within 1.2e-16 ([Parity](#parity)).
- Paired t-test, randomization (sign-flip) test, bootstrap confidence interval, Cohen's d, and Holm or Benjamini–Hochberg correction. These are checked against SciPy, NumPy and statsmodels.
- Data problems that silently change conclusions are reported as `rf.finding/1` records: unjudged documents, queries missing from a run, duplicate documents, ties at a cutoff, and tied scores whose written order the ranking changes.

```sh
cargo install --path .
ireval eval qrels.txt bm25.run dense.run -m nDCG@10,AP,R@100
ireval eval -q -c --format csv qrels.txt bm25.run          # per query; missing queries score 0
ireval compare qrels.txt bm25.run dense.run hybrid.run -m nDCG@10 --seed 1
ireval check --format jsonl qrels.txt *.run               # findings only; exit 1 on errors
```

## Inputs

| Kind | TREC format | JSON Lines |
| --- | --- | --- |
| Qrels | `qid iter docid rel` | `{"qid": "q1", "docid": "d7", "rel": 2}` |
| Run | `qid Q0 docid rank score tag` | `{"qid": "q1", "docid": "d7", "score": 12.5}` (optional `"tag"`) |

- The format is detected from the first record (`{` means JSON Lines). `--qrels-format` and `--run-format` override the detection.
- TREC fields are separated by whitespace. Blank lines and lines starting with `#` are skipped, and a leading UTF-8 byte order mark is ignored.
- In JSON, `qid` and `docid` may be strings or integers, and unknown fields are ignored.
- The `iter`, `Q0` and `rank` columns are ignored, as in `trec_eval`: ranks always come from the scores. An integer rank, and an integer `rank` field in JSON, is read only to report tied scores that the ranking reorders (`tie-reorders-run`).
- `rel` must be an integer. A score that is not a number, or is NaN, is an error.
- A repeated `(qid, docid)` in the qrels is an error. Errors name the file and line.

## Evaluation semantics

**Ranking.** Each query's documents are sorted by score descending, with ties broken by docid descending in byte order. This is `trec_eval`'s `comp_sim_docno` in `form_res_rels.c`; the parity fixtures contain 228 ranked lists with tied scores. The written rank order of tied documents is not kept: a run that needs its order must give strictly decreasing scores, such as `score = N + 1 - rank`. When the tie-break changes the written order, the `tie-reorders-run` finding reports it. `-M/--max-docs N` then keeps the top N documents (`trec_eval -M`).

**Judgments.** A retrieved document is judged when the qrels hold a label of 0 or more for it. A document missing from the qrels, or carrying a negative label, is unjudged. Negative labels are `trec_eval`'s "in pool but unjudged". An unjudged document is never relevant.

**Relevance.** A document is relevant when its label is at least the relevance level: `-l/--relevance-level`, default 1, as `trec_eval -l`. R, the query's relevant count, counts judged documents at or above that level. nDCG ignores the level, as in `trec_eval`.

**Query set**, chosen with `--missing`:

| Mode | Queries averaged | Reference |
| --- | --- | --- |
| `skip` (default) | Queries present in both the qrels and the run | `trec_eval` default |
| `zero` (`-c`) | Every qrels query. A query absent from the run is evaluated as an empty ranking: 0 on every metric except NumRel. | `trec_eval -c` |

In both modes, queries in a run but not in the qrels are ignored and reported. A qrels query with no relevant document is still averaged in, as `trec_eval` does, and is reported (`query-without-relevant`). A run that shares no query with the qrels is an error in both modes (exit 2), as in `trec_eval` 9.0.8, which stops with "No queries with both results and relevance info" even under `-c`.

**Duplicates.** A document listed twice for one query makes `trec_eval` abort, and `ireval` refuses the run the same way (exit 2, naming both lines). `--duplicates first` keeps the higher-ranked copy and reports a `run-duplicate-doc` finding. Like `trec_eval`, which checks only the queries it evaluates, `ireval` accepts duplicates in run queries that have no judgments; they are reported as a `run-duplicate-doc` warning with `evaluated: false`.

**Aggregation.** Over the query set, metrics are averaged and the counts NumRet, NumRel, NumRelRet and NumQ are summed. Both rules are `trec_eval`'s.

## Metrics

Notation for one query: the ranking is `r_1 … r_n` after `-M`. `rel(i)` is 1 when `r_i` is relevant. `R` is the number of relevant judged documents, and `N` the number of judged non-relevant documents (labels `0 ≤ l < level`). `g(l)` is the gain of label `l`, counted only when `l > 0`. All logarithms are base 2. A metric whose denominator would be 0 scores 0.

Names are case-insensitive. Output uses the first name in each row.

| Metric | Also accepted | Definition | Reference (verified) |
| --- | --- | --- | --- |
| `P@k` | `P_k`, `P.k`, `precision@k` | `Σ_{i≤min(k,n)} rel(i) / k`: a short ranking is padded with non-relevant documents | `trec_eval` `P_k`; pytrec_eval |
| `R@k` | `recall_k`, `recall@k` | `Σ_{i≤min(k,n)} rel(i) / R` | `trec_eval` `recall_k`; pytrec_eval |
| `AP` | `map` | `Σ_i rel(i)·(Σ_{j≤i} rel(j))/i / R` | `trec_eval` `map`; pytrec_eval |
| `AP@k` | `map_cut_k`, `map@k` | As AP over `i ≤ k`, still divided by `R` (not `min(k, R)`) | `trec_eval` `map_cut_k`; pytrec_eval |
| `Rprec` | `R-prec` | `Σ_{i≤min(R,n)} rel(i) / R` | `trec_eval` `Rprec`; pytrec_eval |
| `RR` | `recip_rank`, `MRR` | `1 / (rank of the first relevant document)`, or 0 if none is retrieved | `trec_eval` `recip_rank`; pytrec_eval |
| `RR@k` | `MRR@k`, `recip_rank@k` | RR over the top k only | `trec_eval -M k -m recip_rank` |
| `nDCG` | `ndcg` | `DCG / IDCG` with `DCG = Σ_i g(l_i)/log(i+1)`, `g(l) = l`. IDCG is the DCG of all judged labels in descending order. Ignores `-l`. | `trec_eval` `ndcg`; pytrec_eval; ranx `ndcg` |
| `nDCG@k` | `ndcg_cut_k`, `ndcg@k` | DCG over the top k / IDCG over the top k ideal | `trec_eval` `ndcg_cut_k`; pytrec_eval; ranx `ndcg@k` |
| `nDCG_exp`, `nDCG_exp@k` | `nDCG(dcg='exp-log2')@k`, `ndcg_burges` | As nDCG with `g(l) = 2^l − 1` | ranx `ndcg_burges` (ir_measures `nDCG(dcg='exp-log2')`) |
| `Success@k` | `success_k` | 1 if any relevant document is in the top k | `trec_eval` `success_k`; pytrec_eval |
| `Bpref` | `bpref` | `(1/R) Σ_{relevant retrieved r} (1 − min(n_r, R)/min(N, R))`, where `n_r` counts judged non-relevant documents ranked above `r`. The term is 1 when `n_r = 0`. Unjudged documents are skipped. | `trec_eval` `bpref`; pytrec_eval |
| `Judged@k` | `judged_k` | Judged documents in the top `min(k,n)` / `min(k,n)` | ir_measures `Judged@k` (see divergences) |
| `NumRet` | `num_ret` | `n` (summed) | `trec_eval` `num_ret` |
| `NumRel` | `num_rel` | `R` (summed) | `trec_eval` `num_rel` |
| `NumRelRet` | `num_rel_ret` | `Σ_i rel(i)` (summed) | `trec_eval` `num_rel_ret` |
| `NumQ` | `num_q` | Number of queries averaged | `trec_eval` `num_q` |

`P`, `R`, `Success` and `Judged` need a cutoff. When one is written without `@k`, it expands to one metric per entry of `[ireval] cutoffs`. If no cutoffs are configured, that is an error. `AP`, `RR`, `nDCG` and `nDCG_exp` without `@k` use the whole ranking. With no `-m` and no configured metrics, `eval` prints AP, nDCG@10, P@10, R@100, RR, Bpref, NumQ, NumRet, NumRel and NumRelRet.

### Known divergences from reference tools

- **`trec_eval -c` and NumRel.** With `-c`, `trec_eval` sums `num_rel` over all qrels counting labels `> 0`, whatever `-l` is. `ireval` sums the per-query R. The two differ only with `-c` and `-l` greater than 1.
- **ir_measures `Judged@k`.** ir_measures breaks score ties by docid *ascending*. It counts a negative label as judged and divides by zero on an empty ranking. It also reports 0 for qrels queries absent from the run even without `-c`. `ireval` uses `trec_eval`'s order and judgments, and the parity tests feed ir_measures a tie-free, negative-free equivalent.
- **`trec_eval` 10 `unj@k`.** This measure is the unjudged fraction with denominator `k`: a short ranking is padded with judged documents. `Judged@k` divides by `min(k, n)`.
- **ranx.** ranx's tie order is not `trec_eval`'s. It agrees on nDCG and `ndcg_burges` once ties are removed.
- **`trec_eval` 10.0-rc3** aborts with "result qid … not found" when a qrels query is missing from the run and `-c` is not given. Version 9.0.8 skips such queries, and so does `ireval`.
- **Parsing.** `trec_eval` reads `rel` with `atol`, so `1.7` becomes 1; `ireval` rejects it. `trec_eval` checks duplicates after `-M` truncation; `ireval` checks the whole list.

## Comparing runs

`ireval compare qrels A B [C …] -m METRIC…` compares each run against the first run. `--all-pairs` compares every pair instead. For each metric and pair, the paired sample is the per-query values on the queries evaluated for both runs, and `d = b − a` per query.

| Output | Definition | Checked against |
| --- | --- | --- |
| `diff` | `mean(d)` | NumPy |
| `cohens_d` | `mean(d) / sd(d)` with `n − 1` (d_z, Cohen's d for paired samples). Undefined (null) when the differences are constant (see `zero-variance`). | NumPy |
| `t`, `p_t` | Paired t-test, two-sided, `n − 1` degrees of freedom. `p = I_{ν/(ν+t²)}(ν/2, 1/2)` (regularized incomplete beta, Lentz continued fraction, Lanczos `ln Γ`). Constant non-zero differences give `t = ±∞, p = 0`, as in SciPy; differences count as constant when their range is at most 1e-9 times their largest magnitude, so binary rounding noise does not produce a huge finite t. | `scipy.stats.ttest_rel`, `scipy.stats.t.sf`: relative error ≤ 3.2e-12 on the distribution grid |
| `p_randomization` | Two-sided sign-flip test of `S = Σ d`. An assignment counts when `\|S*\| ≥ \|S\| − 1e-9·Σ\|d\|`; this tolerance absorbs binary rounding, so 0.1 − 0.2 + 0.2 − 0.1 ties with 0. Zero differences are dropped. With `m` non-zero differences and `2^m ≤ --trials`, all assignments are enumerated (exact). Otherwise `--trials` random assignments give `p = (1 + hits)/(1 + trials)`. | Exact rational enumeration (39 fixtures). `scipy.stats.permutation_test` exact (38 of 39 equal; the other is a binary-rounding tie SciPy misses) and Monte Carlo (within 5 standard errors). |
| `ci_low`, `ci_high` | Paired percentile bootstrap: `--bootstrap` resamples of the queries with replacement. The interval is given by the `(1 ± confidence)/2` quantiles of the resampled means, interpolated linearly (type 7). | `scipy.stats.bootstrap(method="percentile")`, within Monte Carlo error; `numpy.quantile` exactly |
| `p_adjusted` | The primary test's p-value (`--test randomization` (default) or `t`), corrected across all comparisons of the invocation (every metric and pair): `--correction holm` (default), `bh` or `none`. | `statsmodels` `multipletests` `holm` and `fdr_bh` |
| `significant` | `p_adjusted ≤ --alpha` (default 0.05) | |

Random numbers come from xoshiro256++ seeded by SplitMix64; both are implemented in `src/rng.rs` and tested against the reference C code. Each comparison draws from its own stream, seeded with `splitmix64(seed ^ fnv1a(fnv1a(metric, paired qids and values), test))`. The stream depends only on the metric and the paired data, not on file names or on the other runs of the invocation, so renaming a run or adding a third one leaves a pair's p-value and interval unchanged, and a seed reproduces them in every version. Bounded integers use Lemire's unbiased method. Defaults: `--seed 0`, `--trials 10000`, `--bootstrap 10000`, `--confidence 0.95`.

Without `-c`, a query that one run lacks is dropped from that pair's sample and reported (`comparison-queries-differ`). `-c` scores it 0 instead. A comparison with fewer than `--min-queries` (default 20) pairs is reported as `small-sample`.

## Output

- `eval`: `--format table` (default), `csv`, `json` or `jsonl`, with `-q/--per-query` for values per query.
- `compare`: the same formats.
- `check`: `text` or `jsonl`.
- Tables round to `--precision` (default 4) places. CSV and JSON print full precision, which round-trips.
- In table and CSV formats, findings go to standard error.
- `json` embeds findings in the document.
- `jsonl` writes only `rf.finding/1` records. Metric values become `info` findings with rule `measurement`, and comparisons rule `comparison`.

Output to a closed pipe (for example `| head`) ends quietly. Exit codes: 0 on success, 1 when a finding has severity `error` (only `check` produces these), and 2 for usage, input or I/O errors.

### Findings

| Rule | Severity | Meaning |
| --- | --- | --- |
| `query-missing-from-run` | warn | Qrels queries absent from a run, and how they were treated (`skip` or `zero`) |
| `query-not-in-qrels` | info | Run queries with no judgments (ignored) |
| `query-without-relevant` | info | Qrels queries with no document at the relevance level (still averaged) |
| `unjudged-in-top-k` | warn | Queries whose unjudged fraction in the top `--unjudged-k` (10) exceeds `--unjudged-threshold` (0.1) |
| `tie-at-cutoff` | warn or info | The documents at ranks k and k+1 have the same score, for a cutoff k in use. Severity is warn when the tied documents have different judgments, because then the docid tie-break decides the metric value. |
| `tie-reorders-run` | warn or info | Documents with the same score are written in an order, by integer rank or else by line, that the docid tie-break changes. Groups below `-M` are not counted. Severity is warn when a reordered group has different judgments, because then the tie-break, not the written order, decides the metric value. The finding counts the queries and groups and names the first group. |
| `run-duplicate-doc` | warn (`eval --duplicates first`, or a query without judgments), error (`check`, judged queries) | A document listed more than once for a query |
| `comparison-queries-differ` | warn | Queries left out of a paired comparison because one run lacks them |
| `small-sample` | warn | A comparison uses fewer than `--min-queries` queries |
| `zero-variance` | warn | All paired differences are equal (to within relative 1e-9), so t and d are undefined or infinite |
| `query-named-all` | warn | The qrels contain a query with id `all`, which collides with the aggregate row of `trec_eval`-style output |
| `measurement`, `comparison` | info | Values in `jsonl` output |

Paths in findings are relative to the directory of `research.toml` when they lie inside it. `SOURCE_DATE_EPOCH` (integer seconds) fixes the `at` timestamp; a malformed value is a usage error.

## Configuration

`ireval` reads defaults from the `[ireval]` table of the nearest `research.toml`, searching upward from the current directory, or from `--config <path>`. Flags override the table, and unknown keys are reported on standard error and otherwise ignored. Known keys are validated when the file is read: an unknown `missing`, `test` or `correction` value, or an empty `metrics` list, is an error.

```toml
[ireval]
metrics = ["nDCG@10", "AP", "P", "R"]   # P and R expand over cutoffs
cutoffs = [10, 100]
relevance_level = 1
missing = "skip"            # or "zero" (trec_eval -c)
max_docs = 1000             # trec_eval -M
test = "randomization"      # or "t"
correction = "holm"         # or "bh", "none"
seed = 20261107
trials = 10000
bootstrap = 10000
confidence = 0.95
alpha = 0.05
min_queries = 20
unjudged_k = 10
unjudged_threshold = 0.1
```

## Parity

`cargo test` runs offline against committed fixtures. [`parity/`](parity/README.md) holds the generators and the steps to rebuild the reference binaries.

**Metrics** (`tests/parity.rs`). The 100 randomized cases contain:

- 699 qrels queries and 6,542 run lines;
- 228 ranked lists with tied scores, 117 single-document lists and 2,992 unjudged retrieved documents;
- 278 negative labels, 215 queries with no relevant document, and graded labels up to 4;
- 110 qrels queries missing from runs and 88 run queries missing from qrels;
- mixed docid styles, to exercise byte-order tie-breaking;
- every tenth case in JSON Lines.

Each case is evaluated in three configurations: the default; `-c` with level 1 or 2; and a random combination of level (0–3), missing mode and `-M`.

| Metric | Reference values compared | Max \|ireval − reference\| |
| --- | --- | --- |
| P, R, Success @1,3,5,10,20,50 | `trec_eval` 9.0.8: 12,402 each; 10.0-rc3: 9,090; pytrec_eval: 9,972 | ≤ 1.2e-16 |
| AP, AP@k, nDCG, nDCG@k | `trec_eval` 9.0.8: 14,469; 10.0-rc3: 10,605; pytrec_eval: 11,634 | ≤ 1.2e-16 |
| nDCG, nDCG@k | ranx `ndcg`: 8,694 | ≤ 1.2e-16 |
| nDCG_exp, nDCG_exp@k | ranx `ndcg_burges`: 8,694 | ≤ 1.2e-16 |
| RR, RR@k | `trec_eval` 9.0.8: 14,469 (RR@k via `-M`); 10.0-rc3: 10,605; pytrec_eval: 1,662 | ≤ 5.6e-17 |
| Rprec, Bpref | `trec_eval` 9.0.8: 2,067 each; 10.0-rc3: 1,515; pytrec_eval: 1,662 | ≤ 1.2e-16 |
| Judged@k | ir_measures 0.4.3: 11,658 | ≤ 1.2e-16 |
| NumRet, NumRel, NumRelRet, NumQ | `trec_eval` 9.0.8 and 10.0-rc3 | 0 |

The largest difference, 1.1e-16, is one unit in the last place. The references differ from each other by the same amount: for nDCG, pytrec_eval and ranx sometimes produce one value and the locally built `trec_eval` another, one ulp apart. Aggregates add summation-order differences of the same size.

**Statistics** (`tests/stats_parity.rs`), against SciPy 1.18.1, NumPy 2.5.4 and statsmodels 0.15.0:

- Student t tail: 168 points, relative error ≤ 3.2e-12.
- Paired t-test: 58 samples, relative error ≤ 3.0e-11. The worst case is an n = 2 sample with nearly constant differences, where SciPy itself warns of catastrophic cancellation.
- Exact sign-flip p-values: equal to rational enumeration in all 39 cases.
- Monte Carlo p-values: 15 comparisons, and bootstrap intervals: 16 comparisons, both within Monte Carlo error.
- Quantiles, Holm and BH adjustments: exact to 1e-14.

## Dependencies

- `clap`: command line.
- `serde`, `serde_json` (with `preserve_order`, so JSON keeps metric order) and `toml`: input and output formats and `research.toml`.

The random-number generator and the statistical functions are implemented in this crate. Each is a few dozen lines, and owning them pins seeded results to a documented algorithm independent of other crates' release choices.

## Versioning

Every pull request that changes the tool's behaviour raises the version in `Cargo.toml` and `Cargo.lock`.

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your option.
