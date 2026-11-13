//! Evaluation of runs against qrels: ranking, query sets and aggregation.

use std::collections::{BTreeMap, HashMap};

use serde_json::json;

use crate::findings::{Evidence, Finding, Severity};
use crate::input::{Error, Qrels, Result, Run};
use crate::metrics::{self, Judgment, Metric, QueryData};

/// Which queries are averaged. In `research.toml`: `"skip"` or `"zero"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Missing {
    /// trec_eval default: only queries present in both the qrels and the run.
    Skip,
    /// trec_eval `-c`: every qrels query; one absent from the run is scored as
    /// an empty ranking (0 for every metric except NumRel).
    Zero,
}

/// What to do when a run lists the same document twice for a query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Duplicates {
    /// Refuse the run, as trec_eval does.
    Error,
    /// Keep the highest-ranked occurrence and report a finding.
    First,
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub relevance_level: i64,
    pub missing: Missing,
    pub max_docs: Option<usize>,
    pub duplicates: Duplicates,
    /// Depth and threshold for the unjudged-documents warning.
    pub unjudged_k: usize,
    pub unjudged_threshold: f64,
}

/// A retrieved document after ranking.
#[derive(Clone, Debug)]
struct Ranked {
    score: f64,
    judgment: Judgment,
}

/// Per-query qrels statistics that do not depend on the run.
#[derive(Clone, Debug)]
struct QrelsStats {
    num_rel: usize,
    num_judged_nonrel: usize,
    ideal: Vec<i64>,
}

fn qrels_stats(qrels: &Qrels, level: i64) -> BTreeMap<&str, QrelsStats> {
    qrels
        .queries
        .iter()
        .map(|(qid, q)| {
            let labels = q.labels.values().map(|(l, _)| *l).filter(|&l| l >= 0);
            let mut ideal: Vec<i64> = labels.clone().filter(|&l| l > 0).collect();
            ideal.sort_unstable_by(|a, b| b.cmp(a));
            let num_rel = labels.clone().filter(|&l| l >= level).count();
            let num_judged_nonrel = labels.filter(|&l| l < level).count();
            (
                qid.as_str(),
                QrelsStats {
                    num_rel,
                    num_judged_nonrel,
                    ideal,
                },
            )
        })
        .collect()
}

/// One evaluated query.
#[derive(Clone, Debug)]
pub struct QueryResult {
    pub qid: String,
    /// True when the query was absent from the run and scored as empty.
    pub missing: bool,
    pub values: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct RunResult {
    pub name: String,
    pub path: String,
    pub tag: Option<String>,
    pub queries: Vec<QueryResult>,
    pub aggregate: Vec<f64>,
}

impl RunResult {
    /// Per-query values of metric `index`, keyed by qid.
    pub fn by_qid(&self, index: usize) -> BTreeMap<&str, f64> {
        self.queries
            .iter()
            .map(|q| (q.qid.as_str(), q.values[index]))
            .collect()
    }
}

/// Sorts entries as trec_eval does: score descending, then docid descending
/// (byte order).
fn trec_order(entries: &mut [&crate::input::RunEntry]) {
    entries.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .expect("scores are never NaN")
            .then_with(|| b.docid.as_bytes().cmp(a.docid.as_bytes()))
    });
}

/// Tie groups in a run whose written order differs from [`trec_order`].
#[derive(Default)]
struct Reorder<'a> {
    groups: usize,
    qids: Vec<&'a str>,
    /// Queries where a reordered group holds different judgments.
    mixed: Vec<&'a str>,
    example: Option<ReorderExample>,
}

struct ReorderExample {
    line: usize,
    qid: String,
    score: f64,
    written: Vec<String>,
    evaluated: Vec<String>,
}

/// Evaluates one run. `name` labels it in output; findings about the run are
/// appended to `findings`. Fails on invalid metrics or settings, on a
/// duplicate document under [`Duplicates::Error`], and when no qrels query
/// appears in the run (as `trec_eval` 9.0.8 does, with or without `-c`).
pub fn evaluate(
    qrels: &Qrels,
    run: &Run,
    name: &str,
    metrics: &[Metric],
    settings: &Settings,
    findings: &mut Vec<Finding>,
) -> Result<RunResult> {
    for m in metrics {
        m.validate().map_err(Error)?;
    }
    if settings.max_docs == Some(0) || settings.unjudged_k == 0 {
        return Err(Error(
            "max_docs and unjudged_k must be positive".to_string(),
        ));
    }
    let stats = qrels_stats(qrels, settings.relevance_level);
    let level = settings.relevance_level;
    let mut queries = Vec::new();
    let mut missing_qids = Vec::new();
    let mut duplicate_count = 0usize;
    let mut duplicate_example = None;
    let mut unjudged_over: Vec<&str> = Vec::new();
    let mut unjudged_sum = 0.0;
    let mut unjudged_n = 0usize;
    let mut tie_cutoffs: Vec<usize> = metrics
        .iter()
        .filter_map(|m| m.cutoff)
        .map(|k| settings.max_docs.map_or(k, |d| k.min(d)))
        .collect();
    if let Some(d) = settings.max_docs {
        tie_cutoffs.push(d);
    }
    tie_cutoffs.sort_unstable();
    tie_cutoffs.dedup();
    // cutoff → (queries with a boundary tie, queries where the tie mixes judgments)
    let mut ties: BTreeMap<usize, (Vec<&str>, Vec<&str>)> = BTreeMap::new();
    let mut reorder = Reorder::default();

    for (qid, qrel) in &qrels.queries {
        let st = &stats[qid.as_str()];
        let Some(entries) = run.queries.get(qid) else {
            missing_qids.push(qid.as_str());
            if settings.missing == Missing::Zero {
                let data = QueryData {
                    ranked: &[],
                    level,
                    num_rel: st.num_rel,
                    num_judged_nonrel: st.num_judged_nonrel,
                    ideal: &st.ideal,
                };
                queries.push(QueryResult {
                    qid: qid.clone(),
                    missing: true,
                    values: metrics
                        .iter()
                        .map(|&m| metrics::compute(m, &data))
                        .collect(),
                });
            }
            continue;
        };
        let mut sorted: Vec<&crate::input::RunEntry> = entries.iter().collect();
        trec_order(&mut sorted);
        // Duplicate documents: trec_eval refuses the run.
        let mut seen: HashMap<&str, usize> = HashMap::new();
        let mut kept = Vec::with_capacity(sorted.len());
        for e in sorted {
            if let Some(&first) = seen.get(e.docid.as_str()) {
                if settings.duplicates == Duplicates::Error {
                    return Err(Error(format!(
                        "{}:{}: document {:?} appears twice for query {qid:?} (also line {first}); \
                         trec_eval rejects such runs, pass --duplicates first to keep the higher-ranked copy",
                        run.path, e.line, e.docid
                    )));
                }
                duplicate_count += 1;
                duplicate_example.get_or_insert((e.line, e.docid.clone(), qid.clone(), first));
                continue;
            }
            seen.insert(&e.docid, e.line);
            kept.push(e);
        }
        let ranked_all: Vec<Ranked> = kept
            .iter()
            .map(|e| Ranked {
                score: e.score,
                judgment: match qrel.labels.get(&e.docid) {
                    Some(&(l, _)) if l >= 0 => Judgment::Label(l),
                    _ => Judgment::Unjudged,
                },
            })
            .collect();
        let depth = settings
            .max_docs
            .map_or(ranked_all.len(), |d| d.min(ranked_all.len()));
        let judgments: Vec<Judgment> = ranked_all[..depth].iter().map(|r| r.judgment).collect();

        // Tied scores whose written order the docid tie-break changes. The
        // written order is the rank column when every tied entry has an
        // integer rank, else the line order.
        let mut query_reordered = false;
        let mut query_mixed = false;
        let mut start = 0;
        while start < depth {
            let score = kept[start].score;
            let end = start
                + kept[start..]
                    .iter()
                    .take_while(|e| e.score == score)
                    .count();
            let group = &kept[start..end];
            let mut written = group.to_vec();
            if written.iter().all(|e| e.rank.is_some()) {
                written.sort_by_key(|e| (e.rank, e.line));
            } else {
                written.sort_by_key(|e| e.line);
            }
            if written.iter().zip(group).any(|(w, g)| w.docid != g.docid) {
                reorder.groups += 1;
                query_reordered = true;
                let first = ranked_all[start].judgment;
                query_mixed |= ranked_all[start..end].iter().any(|r| r.judgment != first);
                reorder.example.get_or_insert_with(|| ReorderExample {
                    line: written[0].line,
                    qid: qid.clone(),
                    score,
                    written: written.iter().take(5).map(|e| e.docid.clone()).collect(),
                    evaluated: group.iter().take(5).map(|e| e.docid.clone()).collect(),
                });
            }
            start = end;
        }
        if query_reordered {
            reorder.qids.push(qid);
            if query_mixed {
                reorder.mixed.push(qid);
            }
        }

        // Boundary ties: the document at rank k ties with the one at k + 1.
        for &k in &tie_cutoffs {
            if ranked_all.len() > k && ranked_all[k - 1].score == ranked_all[k].score {
                let s = ranked_all[k].score;
                let inside: Vec<Judgment> = ranked_all[..k]
                    .iter()
                    .filter(|r| r.score == s)
                    .map(|r| r.judgment)
                    .collect();
                let mixed = ranked_all[k..]
                    .iter()
                    .take_while(|r| r.score == s)
                    .any(|r| inside.iter().any(|&j| j != r.judgment));
                let entry = ties.entry(k).or_default();
                entry.0.push(qid);
                if mixed {
                    entry.1.push(qid);
                }
            }
        }

        let data = QueryData {
            ranked: &judgments,
            level,
            num_rel: st.num_rel,
            num_judged_nonrel: st.num_judged_nonrel,
            ideal: &st.ideal,
        };
        let unjudged = 1.0
            - metrics::compute(
                Metric {
                    kind: metrics::Kind::Judged,
                    cutoff: Some(settings.unjudged_k),
                },
                &data,
            );
        unjudged_sum += unjudged;
        unjudged_n += 1;
        if unjudged > settings.unjudged_threshold {
            unjudged_over.push(qid);
        }
        queries.push(QueryResult {
            qid: qid.clone(),
            missing: false,
            values: metrics
                .iter()
                .map(|&m| metrics::compute(m, &data))
                .collect(),
        });
    }

    let subject = run.path.clone();
    let evidence = |locator: String| {
        vec![Evidence {
            path: run.path.clone(),
            locator,
        }]
    };
    if let Some((line, docid, qid, first)) = duplicate_example {
        findings.push(Finding::new(
            "run-duplicate-doc",
            Severity::Warn,
            format!("{}:{line}", run.path),
            format!(
                "{duplicate_count} duplicate document entries removed (kept the higher-ranked copy); first: {docid:?} in query {qid:?}, also line {first}"
            ),
            evidence(format!("line {line}")),
            Some(json!({"run": name, "duplicates": duplicate_count})),
        ));
    }
    if !missing_qids.is_empty() {
        let treatment = match settings.missing {
            Missing::Skip => "left out of the averages (trec_eval default)",
            Missing::Zero => "scored 0 and averaged in (trec_eval -c)",
        };
        findings.push(Finding::new(
            "query-missing-from-run",
            Severity::Warn,
            subject.clone(),
            format!(
                "{} of {} qrels queries are absent from the run and were {treatment}",
                missing_qids.len(),
                qrels.queries.len()
            ),
            evidence("whole file".into()),
            Some(json!({
                "run": name,
                "missing": missing_qids.len(),
                "qrels_queries": qrels.queries.len(),
                "treatment": match settings.missing { Missing::Skip => "skip", Missing::Zero => "zero" },
                "qids": first_n(&missing_qids),
            })),
        ));
    }
    let extra: Vec<&str> = run
        .queries
        .keys()
        .filter(|q| !qrels.queries.contains_key(*q))
        .map(String::as_str)
        .collect();
    // trec_eval only checks the queries it evaluates, so duplicates in
    // unjudged queries never refuse a run; they are still reported.
    let mut unjudged_duplicates = 0usize;
    let mut unjudged_example = None;
    for qid in &extra {
        let mut seen: HashMap<&str, usize> = HashMap::new();
        for e in &run.queries[*qid] {
            if let Some(&first) = seen.get(e.docid.as_str()) {
                unjudged_duplicates += 1;
                unjudged_example.get_or_insert((e.line, e.docid.clone(), qid.to_string(), first));
            } else {
                seen.insert(&e.docid, e.line);
            }
        }
    }
    if let Some((line, docid, qid, first)) = unjudged_example {
        findings.push(Finding::new(
            "run-duplicate-doc",
            Severity::Warn,
            format!("{}:{line}", run.path),
            format!(
                "{unjudged_duplicates} duplicate document entries in queries without judgments (not evaluated); first: {docid:?} in query {qid:?}, also line {first}"
            ),
            evidence(format!("line {line}")),
            Some(json!({"run": name, "duplicates": unjudged_duplicates, "evaluated": false})),
        ));
    }
    if !extra.is_empty() {
        findings.push(Finding::new(
            "query-not-in-qrels",
            Severity::Info,
            subject.clone(),
            format!(
                "{} run queries have no judgments and were ignored",
                extra.len()
            ),
            evidence("whole file".into()),
            Some(json!({"run": name, "count": extra.len(), "qids": first_n(&extra)})),
        ));
    }
    if !unjudged_over.is_empty() {
        let k = settings.unjudged_k;
        findings.push(Finding::new(
            "unjudged-in-top-k",
            Severity::Warn,
            subject.clone(),
            format!(
                "{} of {unjudged_n} queries have more than {:.0}% unjudged documents in the top {k}; mean unjudged fraction {:.3}",
                unjudged_over.len(),
                settings.unjudged_threshold * 100.0,
                unjudged_sum / unjudged_n as f64
            ),
            evidence(format!("top {k} per query")),
            Some(json!({
                "run": name,
                "k": k,
                "threshold": settings.unjudged_threshold,
                "queries_over_threshold": unjudged_over.len(),
                "queries": unjudged_n,
                "mean_unjudged_fraction": unjudged_sum / unjudged_n as f64,
                "qids": first_n(&unjudged_over),
            })),
        ));
    }
    for (k, (tied, mixed)) in &ties {
        let severity = if mixed.is_empty() {
            Severity::Info
        } else {
            Severity::Warn
        };
        findings.push(Finding::new(
            "tie-at-cutoff",
            severity,
            subject.clone(),
            format!(
                "{} queries have a score tie across rank {k}/{}; in {} of them the tied documents have different judgments, so the docid tie-break decides the value",
                tied.len(),
                k + 1,
                mixed.len()
            ),
            evidence(format!("rank {k}")),
            Some(json!({
                "run": name,
                "cutoff": k,
                "queries_tied": tied.len(),
                "queries_tie_changes_judgments": mixed.len(),
                "qids": first_n(mixed),
            })),
        ));
    }

    if let Some(ex) = &reorder.example {
        let severity = if reorder.mixed.is_empty() {
            Severity::Info
        } else {
            Severity::Warn
        };
        findings.push(Finding::new(
            "tie-reorders-run",
            severity,
            format!("{}:{}", run.path, ex.line),
            format!(
                "{} queries have tied scores whose written order the docid tie-break changes ({} tie groups); in {} of them the tied documents have different judgments, so the tie-break, not the written order, decides the values; first: query {:?}, score {}, written {:?}, evaluated {:?}",
                reorder.qids.len(),
                reorder.groups,
                reorder.mixed.len(),
                ex.qid,
                ex.score,
                ex.written,
                ex.evaluated
            ),
            evidence(format!("line {}", ex.line)),
            Some(json!({
                "run": name,
                "queries_reordered": reorder.qids.len(),
                "groups_reordered": reorder.groups,
                "queries_reorder_changes_judgments": reorder.mixed.len(),
                "qids": first_n(&reorder.qids),
                "qids_judgments_differ": first_n(&reorder.mixed),
            })),
        ));
    }

    if queries.iter().all(|q| q.missing) {
        return Err(Error(format!(
            "{}: no query has both results and judgments in {}",
            run.path, qrels.path
        )));
    }
    let n = queries.len() as f64;
    let aggregate = metrics
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let sum: f64 = queries.iter().map(|q| q.values[i]).sum();
            if m.kind.is_count() { sum } else { sum / n }
        })
        .collect();
    Ok(RunResult {
        name: name.to_string(),
        path: run.path.clone(),
        tag: run.tag.clone(),
        queries,
        aggregate,
    })
}

/// Queries in the qrels with no document at or above the relevance level.
pub fn queries_without_relevant(qrels: &Qrels, level: i64) -> Vec<&str> {
    qrels
        .queries
        .iter()
        .filter(|(_, q)| !q.labels.values().any(|(l, _)| *l >= level))
        .map(|(qid, _)| qid.as_str())
        .collect()
}

pub fn first_n(items: &[&str]) -> Vec<String> {
    items.iter().take(20).map(|s| s.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{Format, parse_qrels, parse_run};
    use crate::metrics::Kind;

    fn settings(missing: Missing) -> Settings {
        Settings {
            relevance_level: 1,
            missing,
            max_docs: None,
            duplicates: Duplicates::Error,
            unjudged_k: 10,
            unjudged_threshold: 0.1,
        }
    }

    #[test]
    fn rejects_invalid_metrics_and_disjoint_runs() {
        let qrels = parse_qrels("q1 0 d1 1\n", "q", Format::Trec).unwrap();
        let run = parse_run("q1 Q0 d1 1 1 t\n", "r", Format::Trec).unwrap();
        for m in [
            Metric {
                kind: Kind::P,
                cutoff: Some(0),
            },
            Metric {
                kind: Kind::P,
                cutoff: None,
            },
            Metric {
                kind: Kind::NumRel,
                cutoff: Some(5),
            },
        ] {
            let e = evaluate(
                &qrels,
                &run,
                "r",
                &[m],
                &settings(Missing::Skip),
                &mut Vec::new(),
            );
            assert!(e.is_err(), "{m:?}");
        }
        let mut s = settings(Missing::Skip);
        s.max_docs = Some(0);
        let ap = Metric {
            kind: Kind::AP,
            cutoff: None,
        };
        assert!(evaluate(&qrels, &run, "r", &[ap], &s, &mut Vec::new()).is_err());
        let other = parse_run("zz Q0 d1 1 1 t\n", "o", Format::Trec).unwrap();
        for missing in [Missing::Skip, Missing::Zero] {
            let e = evaluate(
                &qrels,
                &other,
                "o",
                &[ap],
                &settings(missing),
                &mut Vec::new(),
            )
            .unwrap_err();
            assert!(
                e.0.contains("no query has both results and judgments"),
                "{e}"
            );
        }
    }

    #[test]
    fn duplicates_in_unjudged_queries_are_reported_not_refused() {
        let qrels = parse_qrels("q1 0 d1 1\n", "q", Format::Trec).unwrap();
        let run = parse_run(
            "q1 Q0 d1 1 1 t\nq9 Q0 d5 1 1 t\nq9 Q0 d5 2 0.5 t\n",
            "r",
            Format::Trec,
        )
        .unwrap();
        let mut findings = Vec::new();
        let ap = Metric {
            kind: Kind::AP,
            cutoff: None,
        };
        evaluate(
            &qrels,
            &run,
            "r",
            &[ap],
            &settings(Missing::Skip),
            &mut findings,
        )
        .unwrap();
        let f = findings
            .iter()
            .find(|f| f.rule == "run-duplicate-doc")
            .unwrap();
        assert_eq!(f.subject, "r:3");
    }

    fn reorder_finding(run: &str, format: Format, max_docs: Option<usize>) -> Option<Finding> {
        let qrels = parse_qrels(
            "q1 0 d1 1\nq1 0 d2 0\nq1 0 d3 0\nq2 0 d1 1\n",
            "q",
            Format::Trec,
        )
        .unwrap();
        let run = parse_run(run, "r", format).unwrap();
        let mut s = settings(Missing::Skip);
        s.max_docs = max_docs;
        let mut findings = Vec::new();
        let ap = Metric {
            kind: Kind::AP,
            cutoff: None,
        };
        evaluate(&qrels, &run, "r", &[ap], &s, &mut findings).unwrap();
        findings.into_iter().find(|f| f.rule == "tie-reorders-run")
    }

    #[test]
    fn tied_scores_that_change_the_written_order_are_reported() {
        // Every score ties; the ranks put d1 first, trec_eval puts d3 first.
        let run = "q1 Q0 d1 1 1 t\nq1 Q0 d2 2 1 t\nq1 Q0 d3 3 1 t\nq2 Q0 d1 1 1 t\n";
        let f = reorder_finding(run, Format::Trec, None).unwrap();
        assert_eq!(f.severity, Severity::Warn);
        assert_eq!(f.subject, "r:1");
        let data = f.data.unwrap();
        assert_eq!(data["queries_reordered"], 1);
        assert_eq!(data["groups_reordered"], 1);
        assert_eq!(data["qids"], json!(["q1"]));
        assert_eq!(data["qids_judgments_differ"], json!(["q1"]));
        assert!(
            f.message
                .contains(r#"written ["d1", "d2", "d3"], evaluated ["d3", "d2", "d1"]"#)
        );

        // Equal judgments in the reordered group: the values do not change.
        let run = "q1 Q0 d2 1 1 t\nq1 Q0 d3 2 1 t\n";
        let f = reorder_finding(run, Format::Trec, None).unwrap();
        assert_eq!(f.severity, Severity::Info);
    }

    #[test]
    fn written_order_follows_the_rank_column_then_the_lines() {
        // Ranks list d3 before d1, which is the trec_eval order: no finding.
        let run = "q1 Q0 d1 2 1 t\nq1 Q0 d3 1 1 t\n";
        assert!(reorder_finding(run, Format::Trec, None).is_none());
        // A rank that is not an integer falls back to the line order.
        let run = "q1 Q0 d1 x 1 t\nq1 Q0 d3 1 1 t\n";
        assert!(reorder_finding(run, Format::Trec, None).is_some());
        let run = "q1 Q0 d3 x 1 t\nq1 Q0 d1 1 1 t\n";
        assert!(reorder_finding(run, Format::Trec, None).is_none());
        // JSON Lines: the optional rank, else the line order.
        let run = "{\"qid\":\"q1\",\"docid\":\"d1\",\"score\":1,\"rank\":2}\n{\"qid\":\"q1\",\"docid\":\"d3\",\"score\":1,\"rank\":1}\n";
        assert!(reorder_finding(run, Format::Jsonl, None).is_none());
        let run = "{\"qid\":\"q1\",\"docid\":\"d1\",\"score\":1}\n{\"qid\":\"q1\",\"docid\":\"d3\",\"score\":1}\n";
        assert!(reorder_finding(run, Format::Jsonl, None).is_some());
    }

    #[test]
    fn distinct_scores_and_ties_below_the_depth_are_not_reported() {
        let run = "q1 Q0 d1 1 3 t\nq1 Q0 d2 2 2 t\nq1 Q0 d3 3 1 t\n";
        assert!(reorder_finding(run, Format::Trec, None).is_none());
        // d2 and d3 tie below -M 1.
        let run = "q1 Q0 d1 1 3 t\nq1 Q0 d2 2 1 t\nq1 Q0 d3 3 1 t\n";
        assert!(reorder_finding(run, Format::Trec, Some(1)).is_none());
        assert!(reorder_finding(run, Format::Trec, Some(2)).is_some());
    }
}
