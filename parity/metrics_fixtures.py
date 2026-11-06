# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "pytrec-eval-terrier==0.5.11",
#     "ir-measures==0.4.3",
#     "ranx==0.3.21",
#     "numpy",
# ]
# ///
"""Generate randomized qrels/runs and reference metric values.

Writes TREC (and some JSON Lines) fixtures plus one expected-values file per
case into tests/fixtures/parity/. `cargo test` compares ireval against these
values offline. Reference implementations:

- trec_eval 9.0.8 and 10.0-rc3 built from source, with the print format
  changed from %6.4f to %.17g so values round-trip (see parity/README.md);
- pytrec_eval (pytrec-eval-terrier), which embeds trec_eval's C code;
- ir_measures' own Judged@k provider;
- ranx's ndcg (Järvelin) and ndcg_burges (2^rel - 1 gains).

Usage (from the repository root):
    uv run --script parity/metrics_fixtures.py \
        --trec-eval-9 /path/to/trec_eval-9.0.8 --trec-eval-10 /path/to/trec_eval-10 \
        --cases 100 --seed 20261107 --out tests/fixtures/parity
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import subprocess
import tempfile
from pathlib import Path

CUTOFFS = [1, 3, 5, 10, 20, 50]


def trec_order(docs: dict[str, float]) -> list[str]:
    """Score descending, docid descending (byte order), as trec_eval sorts."""
    by_doc = sorted(docs, key=lambda d: d.encode(), reverse=True)
    return sorted(by_doc, key=lambda d: -docs[d])


def doc_universe(rng: random.Random) -> list[str]:
    styles = rng.sample(["d{}", "D{}", "doc-{:03d}", "{}", "x_{}"], k=rng.randint(1, 3))
    names = set()
    while len(names) < 90:
        names.add(rng.choice(styles).format(rng.randint(0, 400)))
    return sorted(names)


def score_fn(rng: random.Random):
    style = rng.choice(["grid", "int", "continuous", "mixed"])
    grid = [0.1, 0.2, 0.25, 0.5, 1.0, 2.0, -1.0, 0.0]

    def score() -> float:
        s = style if style != "mixed" else rng.choice(["grid", "continuous"])
        if s == "grid":
            return rng.choice(grid)
        if s == "int":
            return float(rng.randint(1, 6))
        return round(rng.uniform(-3, 40), rng.randint(1, 12))

    return score


def generate_case(rng: random.Random):
    universe = doc_universe(rng)
    qid_pool = ["1", "2", "3", "10", "101", "q1", "q2", "q10", "Q1", "topic-7", "t_5", "200", "9", "q9", "a"]
    qids = rng.sample(qid_pool, k=rng.randint(1, 14))
    max_grade = rng.choice([1, 1, 2, 3, 4])
    allow_negative = rng.random() < 0.25
    score = score_fn(rng)
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    for qid in qids:
        pool = rng.sample(universe, k=rng.randint(1, 30))
        p_rel = rng.choice([0.0, 0.1, 0.3, 0.6, 0.9])
        labels = {}
        for d in pool:
            if allow_negative and rng.random() < 0.1:
                labels[d] = -1
            elif rng.random() < p_rel:
                labels[d] = rng.randint(1, max_grade)
            else:
                labels[d] = 0
        qrels[qid] = labels
        # Query missing from the run (but keep at least one query in common:
        # trec_eval and ireval both refuse a run with none).
        if rng.random() < 0.15 and (run or qid != qids[-1]):
            continue
        m = rng.choice([1, 2, 3, rng.randint(1, 40), rng.randint(20, 60)])
        p_judged = rng.choice([0.2, 0.5, 0.8, 1.0])
        chosen: set[str] = set()
        for _ in range(m * 3):
            if len(chosen) >= m:
                break
            src = pool if rng.random() < p_judged else universe
            chosen.add(rng.choice(src))
        run[qid] = {d: score() for d in sorted(chosen)}
    for qid in rng.sample(["extra1", "999", "q77"], k=rng.choice([0, 0, 1, 2])):
        run[qid] = {d: score() for d in rng.sample(universe, k=rng.randint(1, 10))}
    return qrels, run


def write_trec(qrels, run, qpath: Path, rpath: Path, rng: random.Random):
    qlines = [f"{q} 0 {d} {l}" for q in qrels for d, l in qrels[q].items()]
    rng.shuffle(qlines)
    qpath.write_text("\n".join(qlines) + "\n")
    rlines = []
    for q, docs in run.items():
        for i, (d, s) in enumerate(docs.items()):
            # The rank column is deliberately not the true rank: it is ignored.
            rlines.append(f"{q} Q0 {d} {i + 1} {s!r} tag")
    rng.shuffle(rlines)
    rpath.write_text("\n".join(rlines) + "\n")


def write_jsonl(qrels, run, qpath: Path, rpath: Path):
    with qpath.open("w") as f:
        for q in qrels:
            for d, l in qrels[q].items():
                f.write(json.dumps({"qid": q, "docid": d, "rel": l}) + "\n")
    with rpath.open("w") as f:
        for q, docs in run.items():
            for d, s in docs.items():
                f.write(json.dumps({"qid": q, "docid": d, "score": s, "tag": "tag"}) + "\n")


TREC_NAMES = {
    "P": "P@{}",
    "recall": "R@{}",
    "map_cut": "AP@{}",
    "ndcg_cut": "nDCG@{}",
    "success": "Success@{}",
}
TREC_SCALARS = {
    "map": "AP",
    "Rprec": "Rprec",
    "recip_rank": "RR",
    "ndcg": "nDCG",
    "bpref": "Bpref",
    "num_ret": "NumRet",
    "num_rel": "NumRel",
    "num_rel_ret": "NumRelRet",
    "num_q": "NumQ",
}


def trec_name(name: str) -> str | None:
    if name in TREC_SCALARS:
        return TREC_SCALARS[name]
    base, _, k = name.rpartition("_")
    if base in TREC_NAMES and k.isdigit():
        return TREC_NAMES[base].format(k)
    return None


def run_trec_eval(binary, qpath, rpath, level, complete, max_docs, measures):
    args = [binary, "-q", "-l", str(level)]
    if complete:
        args.append("-c")
    if max_docs is not None:
        args += ["-M", str(max_docs)]
    for m in measures:
        args += ["-m", m]
    out = subprocess.run(args + [str(qpath), str(rpath)], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{args}: {out.stderr}")
    for line in out.stdout.splitlines():
        name, qid, value = line.split()
        if name == "runid":
            continue
        yield name, qid, float(value)


def trec_reference(binary, qpath, rpath, cfg):
    cut = ",".join(map(str, CUTOFFS))
    measures = [f"P.{cut}", f"recall.{cut}", f"map_cut.{cut}", f"ndcg_cut.{cut}", f"success.{cut}"]
    measures += list(TREC_SCALARS)
    per_query: dict[str, dict[str, float]] = {}
    aggregate: dict[str, float] = {}
    complete = cfg["missing"] == "zero"
    for name, qid, value in run_trec_eval(
        binary, qpath, rpath, cfg["relevance_level"], complete, cfg["max_docs"], measures
    ):
        metric = trec_name(name)
        if metric is None:
            continue
        (aggregate if qid == "all" else per_query.setdefault(qid, {}))[metric] = value
    # RR@k is recip_rank over the top k only: trec_eval -M k.
    for k in CUTOFFS:
        depth = k if cfg["max_docs"] is None else min(k, cfg["max_docs"])
        for name, qid, value in run_trec_eval(
            binary, qpath, rpath, cfg["relevance_level"], complete, depth, ["recip_rank"]
        ):
            if name == "recip_rank":
                (aggregate if qid == "all" else per_query.setdefault(qid, {}))[f"RR@{k}"] = value
    return {"aggregate": aggregate, "per_query": per_query}


def truncated(run, max_docs):
    out = {}
    for q, docs in run.items():
        order = trec_order(docs)
        if max_docs is not None:
            order = order[:max_docs]
        out[q] = order
    return out


def pytrec_reference(qrels, run, cfg):
    import pytrec_eval

    cut = ",".join(map(str, CUTOFFS))
    measures = {f"P.{cut}", f"recall.{cut}", f"map_cut.{cut}", f"ndcg_cut.{cut}", f"success.{cut}"}
    measures |= set(TREC_SCALARS) - {"num_q"}
    order = truncated(run, cfg["max_docs"])
    trun = {q: {d: run[q][d] for d in docs} for q, docs in order.items()}
    ev = pytrec_eval.RelevanceEvaluator(qrels, measures, relevance_level=cfg["relevance_level"])
    res = ev.evaluate(trun)
    per_query = {}
    for qid, vals in res.items():
        row = {}
        for name, value in vals.items():
            metric = trec_name(name)
            if metric is not None:
                row[metric] = value
        per_query[qid] = row
    return {"per_query": per_query}


def tie_free(run, max_docs):
    """Run with unique scores that encode trec_eval's order (for libraries
    that break ties differently)."""
    order = truncated(run, max_docs)
    return {q: {d: float(len(docs) - i) for i, d in enumerate(docs)} for q, docs in order.items()}


def ir_measures_reference(qrels, run, cfg):
    import ir_measures
    from ir_measures import Judged

    # ireval counts a document as judged when its label is >= 0; negative
    # labels mean "in pool but unjudged" (trec_eval). Drop them here.
    jqrels = {q: {d: l for d, l in labels.items() if l >= 0} for q, labels in qrels.items()}
    jqrels = {q: v for q, v in jqrels.items() if v}
    trun = tie_free(run, cfg["max_docs"])
    measures = [Judged @ k for k in CUTOFFS]
    per_query: dict[str, dict[str, float]] = {}
    provider = ir_measures.judged
    for m in provider.iter_calc(measures, jqrels, trun):
        per_query.setdefault(m.query_id, {})[f"Judged@{m.measure['cutoff']}"] = m.value
    if cfg["missing"] == "skip":
        # ir_measures reports 0 for qrels queries absent from the run (like
        # trec_eval -c); in skip mode those queries are not evaluated.
        per_query = {q: v for q, v in per_query.items() if q in run}
    return {"per_query": per_query}


def ranx_reference(qrels, run, cfg):
    from ranx import Qrels, Run, evaluate

    trun = tie_free(run, cfg["max_docs"])
    # Gains come from positive labels only (ireval and trec_eval treat
    # negative labels as unjudged, and label 0 has no gain).
    pos = {q: {d: l for d, l in labels.items() if l > 0} for q, labels in qrels.items()}
    qids = sorted(q for q in trun if q in pos and pos[q])
    if not qids:
        return {"per_query": {}}
    rq = Qrels({q: pos[q] for q in qids})
    rr = Run({q: trun[q] for q in qids})
    metrics = ["ndcg_burges", "ndcg"] + [f"ndcg_burges@{k}" for k in CUTOFFS] + [f"ndcg@{k}" for k in CUTOFFS]
    evaluate(rq, rr, metrics, return_mean=False, make_comparable=False)
    per_query = {q: {} for q in qids}
    for metric in metrics:
        name = metric.replace("ndcg_burges", "nDCG_exp").replace("ndcg", "nDCG")
        for q in qids:
            per_query[q][name] = float(rr.scores[metric][q])
    return {"per_query": per_query}


SHORT = {
    "trec_eval-9.0.8": "t9",
    "trec_eval-10.0-rc3": "t10",
    "pytrec_eval": "py",
    "ir_measures": "irm",
    "ranx": "rx",
}


def consensus(refs, label):
    """Columnar expected values: {"qids": [...], "values": {metric: {refs:
    [value or None per qid]}}}, where `refs` lists (space-separated, sorted)
    the reference codes that produced exactly those values. "all" is the
    aggregate row."""
    grouped: dict[tuple[str, str], dict[float, list[str]]] = {}
    for ref, data in refs.items():
        rows = dict(data.get("per_query", {}))
        if "aggregate" in data:
            rows["all"] = data["aggregate"]
        for qid, row in rows.items():
            for metric, value in row.items():
                grouped.setdefault((metric, qid), {}).setdefault(value, []).append(SHORT[ref])
    qids = sorted({q for _, q in grouped if q != "all"}) + ["all"]
    pos = {q: i for i, q in enumerate(qids)}
    values: dict[str, dict[str, list]] = {}
    for (metric, qid), by_value in sorted(grouped.items()):
        if len(by_value) > 1:
            print(f"references disagree: {label} {qid} {metric} {by_value}")
        for value, rs in by_value.items():
            col = values.setdefault(metric, {}).setdefault(" ".join(sorted(rs)), [None] * len(qids))
            col[pos[qid]] = value
    return {"qids": qids, "values": values}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trec-eval-9", required=True)
    ap.add_argument("--trec-eval-10", required=True)
    ap.add_argument("--cases", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    versions = {
        p: importlib.metadata.version(p)
        for p in ["pytrec-eval-terrier", "ir-measures", "ranx", "numpy"]
    }
    for label, binary in (("trec_eval-9.0.8", args.trec_eval_9), ("trec_eval-10.0-rc3", args.trec_eval_10)):
        v = subprocess.run([binary, "-v"], capture_output=True, text=True)
        versions[label] = (v.stdout + v.stderr).strip()
    manifest = {
        "seed": args.seed,
        "cases": args.cases,
        "cutoffs": CUTOFFS,
        "versions": versions,
        "reference_codes": {v: k for k, v in SHORT.items()},
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i in range(args.cases):
            name = f"case_{i:03d}"
            qrels, run = generate_case(rng)
            qpath, rpath = tmp / f"{name}.qrels", tmp / f"{name}.run"
            write_trec(qrels, run, qpath, rpath, rng)
            if i % 10 == 9:
                fq, fr = f"{name}.qrels.jsonl", f"{name}.run.jsonl"
                write_jsonl(qrels, run, args.out / fq, args.out / fr)
            else:
                fq, fr = f"{name}.qrels", f"{name}.run"
                (args.out / fq).write_text(qpath.read_text())
                (args.out / fr).write_text(rpath.read_text())
            missing_any = any(q not in run for q in qrels)
            configs = [
                {"relevance_level": 1, "missing": "skip", "max_docs": None},
                {"relevance_level": rng.choice([1, 2]), "missing": "zero", "max_docs": None},
                {
                    "relevance_level": rng.choice([0, 1, 2, 3]),
                    "missing": rng.choice(["skip", "zero"]),
                    "max_docs": rng.choice([None, rng.randint(1, 25)]),
                },
            ]
            evaluations = []
            for cfg in configs:
                refs = {"trec_eval-9.0.8": trec_reference(args.trec_eval_9, qpath, rpath, cfg)}
                if cfg["missing"] == "zero" or not missing_any:
                    # 10.0-rc3 exits with an error for qrels queries missing
                    # from the run unless -c is given.
                    refs["trec_eval-10.0-rc3"] = trec_reference(args.trec_eval_10, qpath, rpath, cfg)
                if cfg["relevance_level"] >= 1:  # pytrec_eval rejects level 0
                    refs["pytrec_eval"] = pytrec_reference(qrels, run, cfg)
                refs["ir_measures"] = ir_measures_reference(qrels, run, cfg)
                refs["ranx"] = ranx_reference(qrels, run, cfg)
                if cfg["missing"] == "zero" and cfg["relevance_level"] != 1:
                    # trec_eval -c sums num_rel over all qrels counting labels
                    # > 0, ignoring -l; ireval sums the per-query NumRel
                    # (documented divergence, tested separately).
                    for r in ("trec_eval-9.0.8", "trec_eval-10.0-rc3"):
                        refs.get(r, {}).get("aggregate", {}).pop("NumRel", None)
                evaluations.append({**cfg, "expected": consensus(refs, f"{name} {cfg}")})
            (args.out / f"{name}.expected.json").write_text(
                json.dumps({"qrels": fq, "run": fr, "evaluations": evaluations}, sort_keys=True) + "\n"
            )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
