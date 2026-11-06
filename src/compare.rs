//! Paired comparisons of runs over queries.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use serde_json::json;

use crate::eval::{RunResult, first_n};
use crate::findings::{Evidence, Finding, Severity};
use crate::metrics::Metric;
use crate::rng::{Xoshiro256pp, fnv1a, stream_seed};
use crate::stats::{self, Correction, PermMethod, TTest};

/// Which test's p-value is corrected and used for significance. In
/// `research.toml`: `"t"` or `"randomization"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Primary {
    T,
    Randomization,
}

#[derive(Clone, Debug)]
pub struct CompareSettings {
    pub primary: Primary,
    pub correction: Correction,
    pub seed: u64,
    pub trials: u64,
    pub bootstrap: u64,
    pub confidence: f64,
    pub alpha: f64,
    pub min_queries: usize,
    /// Compare every pair of runs instead of each run against the first.
    pub all_pairs: bool,
}

#[derive(Clone, Debug)]
pub struct Comparison {
    pub metric: Metric,
    /// Display names of the two runs.
    pub a: String,
    pub b: String,
    /// Paths of the two runs, as given.
    pub a_path: String,
    pub b_path: String,
    pub n: usize,
    pub mean_a: f64,
    pub mean_b: f64,
    /// Mean of `b - a` over the paired queries.
    pub diff: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    /// Cohen's d for paired samples: mean(d) / sd(d), sd with n - 1.
    pub cohens_d: f64,
    pub t: f64,
    pub df: f64,
    pub p_t: f64,
    pub p_perm: f64,
    pub perm_method: PermMethod,
    pub perm_samples: u64,
    /// Primary p-value after correction across the whole family.
    pub p_adjusted: f64,
    pub significant: bool,
}

/// True when the differences agree to within [`stats::PERM_TIE_TOLERANCE`]
/// relative to their largest magnitude. Binary rounding makes, for example,
/// P@10 differences of one document come out as 0.1, 0.09999999999999998 and
/// 0.10000000000000003; treating those as varying would give an enormous t
/// and Cohen's d.
fn effectively_constant(d: &[f64]) -> bool {
    let (lo, hi) = d
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
            (lo.min(x), hi.max(x))
        });
    let scale = d.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    hi - lo <= stats::PERM_TIE_TOLERANCE * scale
}

/// Identifies a pair's paired data for seeding: the metric, the query ids and
/// the exact values of both runs. Streams therefore depend only on the seed
/// and the data, not on file names or on which other runs are compared.
fn data_key(metric: &str, qids: &[&str], a: &[f64], b: &[f64]) -> String {
    let mut key = String::from(metric);
    for ((q, x), y) in qids.iter().zip(a).zip(b) {
        let _ = write!(
            key,
            "\u{1e}{q}\u{1f}{:016x}\u{1f}{:016x}",
            x.to_bits(),
            y.to_bits()
        );
    }
    format!("{:016x}", fnv1a(&[&key]))
}

pub fn compare(
    runs: &[RunResult],
    metrics: &[Metric],
    s: &CompareSettings,
    findings: &mut Vec<Finding>,
) -> Vec<Comparison> {
    let mut pairs = Vec::new();
    if s.all_pairs {
        for i in 0..runs.len() {
            for j in i + 1..runs.len() {
                pairs.push((i, j));
            }
        }
    } else {
        for j in 1..runs.len() {
            pairs.push((0, j));
        }
    }
    let mut out = Vec::new();
    let mut warned_pairs = BTreeSet::new();
    for (mi, &metric) in metrics.iter().enumerate() {
        for &(i, j) in &pairs {
            let (ra, rb) = (&runs[i], &runs[j]);
            let va = ra.by_qid(mi);
            let vb = rb.by_qid(mi);
            let common: Vec<&str> = va.keys().filter(|q| vb.contains_key(*q)).copied().collect();
            if (common.len() != va.len() || common.len() != vb.len()) && warned_pairs.insert((i, j))
            {
                let only: Vec<&str> = va
                    .keys()
                    .chain(vb.keys())
                    .filter(|q| !(va.contains_key(*q) && vb.contains_key(*q)))
                    .copied()
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                findings.push(Finding::new(
                    "comparison-queries-differ",
                    Severity::Warn,
                    format!("{} vs {}", ra.path, rb.path),
                    format!(
                        "{} queries are evaluated for only one of the two runs and were left out of the paired tests; use --missing zero to score them as 0",
                        only.len()
                    ),
                    vec![],
                    Some(json!({"a": ra.name, "b": rb.name, "dropped": only.len(), "qids": first_n(&only)})),
                ));
            }
            let a: Vec<f64> = common.iter().map(|q| va[q]).collect();
            let b: Vec<f64> = common.iter().map(|q| vb[q]).collect();
            let d: Vec<f64> = a.iter().zip(&b).map(|(x, y)| y - x).collect();
            let name = metric.to_string();
            let n = d.len();
            let diff = if n > 0 { stats::mean(&d) } else { f64::NAN };
            let constant = n > 1 && effectively_constant(&d);
            // Constant differences: t is ±inf with p = 0 (NaN when the
            // difference is 0), as scipy.stats.ttest_rel reports for exactly
            // constant input, and Cohen's d is undefined.
            let tt = if constant {
                let degenerate = diff == 0.0;
                TTest {
                    t: if degenerate {
                        f64::NAN
                    } else {
                        diff.signum() * f64::INFINITY
                    },
                    df: (n - 1) as f64,
                    p: if degenerate { f64::NAN } else { 0.0 },
                }
            } else {
                stats::paired_t(&d)
            };
            let key = data_key(&name, &common, &a, &b);
            let mut perm_rng = Xoshiro256pp::from_seed(stream_seed(s.seed, &[&key, "perm"]));
            let perm = stats::paired_randomization(&d, s.trials, &mut perm_rng);
            let mut boot_rng = Xoshiro256pp::from_seed(stream_seed(s.seed, &[&key, "bootstrap"]));
            let (ci_low, ci_high) =
                stats::bootstrap_mean_ci(&d, s.bootstrap, s.confidence, &mut boot_rng);
            let cohens_d = if constant || n < 2 {
                f64::NAN
            } else {
                diff / stats::sample_sd(&d)
            };
            if constant {
                findings.push(Finding::new(
                    "zero-variance",
                    Severity::Warn,
                    format!("{} vs {}", ra.path, rb.path),
                    format!(
                        "{name}: every paired difference is {diff} (to within relative {}); the t statistic is infinite or undefined and Cohen's d is undefined",
                        stats::PERM_TIE_TOLERANCE
                    ),
                    vec![],
                    Some(json!({"metric": name, "a": ra.name, "b": rb.name, "n": n})),
                ));
            }
            out.push(Comparison {
                metric,
                a: ra.name.clone(),
                b: rb.name.clone(),
                a_path: ra.path.clone(),
                b_path: rb.path.clone(),
                n,
                mean_a: if n > 0 { stats::mean(&a) } else { f64::NAN },
                mean_b: if n > 0 { stats::mean(&b) } else { f64::NAN },
                diff,
                ci_low,
                ci_high,
                cohens_d,
                t: tt.t,
                df: tt.df,
                p_t: tt.p,
                p_perm: perm.p,
                perm_method: perm.method,
                perm_samples: perm.samples,
                p_adjusted: f64::NAN,
                significant: false,
            });
        }
    }
    let primary: Vec<f64> = out
        .iter()
        .map(|c| match s.primary {
            Primary::T => c.p_t,
            Primary::Randomization => c.p_perm,
        })
        .collect();
    let adjusted = stats::adjust(&primary, s.correction);
    for (c, p) in out.iter_mut().zip(adjusted) {
        c.p_adjusted = p;
        c.significant = p <= s.alpha;
    }
    if let Some(min_n) = out.iter().map(|c| c.n).min()
        && min_n < s.min_queries
    {
        findings.push(Finding::new(
            "small-sample",
            Severity::Warn,
            runs.iter()
                .map(|r| r.path.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            format!(
                "paired tests use as few as {min_n} queries (fewer than {}); p-values and intervals are unreliable at this size",
                s.min_queries
            ),
            runs.iter()
                .map(|r| Evidence {
                    path: r.path.clone(),
                    locator: "whole file".into(),
                })
                .collect(),
            Some(json!({"min_n": min_n, "min_queries": s.min_queries})),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::QueryResult;
    use crate::metrics::Kind;

    fn run(name: &str, path: &str, values: &[(&str, f64)]) -> RunResult {
        RunResult {
            name: name.into(),
            path: path.into(),
            tag: None,
            queries: values
                .iter()
                .map(|&(q, v)| QueryResult {
                    qid: q.into(),
                    missing: false,
                    values: vec![v],
                })
                .collect(),
            aggregate: vec![0.0],
        }
    }

    fn settings(trials: u64) -> CompareSettings {
        CompareSettings {
            primary: Primary::Randomization,
            correction: Correction::Holm,
            seed: 3,
            trials,
            bootstrap: 200,
            confidence: 0.95,
            alpha: 0.05,
            min_queries: 1,
            all_pairs: false,
        }
    }

    const P10: Metric = Metric {
        kind: Kind::P,
        cutoff: Some(10),
    };

    #[test]
    fn rounding_noise_counts_as_constant_differences() {
        // One more relevant document in the top 10 for every query.
        let a = run("a", "a", &[("q1", 0.1), ("q2", 0.2), ("q3", 0.7)]);
        let b = run("b", "b", &[("q1", 0.2), ("q2", 0.3), ("q3", 0.8)]);
        let mut findings = Vec::new();
        let c = &compare(&[a, b], &[P10], &settings(1000), &mut findings)[0];
        assert!(c.cohens_d.is_nan());
        assert_eq!(c.t, f64::INFINITY);
        assert_eq!(c.p_t, 0.0);
        assert!(findings.iter().any(|f| f.rule == "zero-variance"));
    }

    #[test]
    fn a_pair_without_common_queries_stays_out_of_the_family() {
        let a = run("a", "a", &[("q1", 0.1), ("q2", 0.2), ("q3", 0.3)]);
        let b = run("b", "b", &[("q1", 0.5), ("q2", 0.6), ("q3", 0.9)]);
        let c = run("c", "c", &[("q9", 0.5)]);
        let mut findings = Vec::new();
        let two = compare(
            &[a.clone(), b.clone()],
            &[P10],
            &settings(1000),
            &mut findings,
        );
        let three = compare(&[a, b, c], &[P10], &settings(1000), &mut findings);
        assert_eq!(three[1].n, 0);
        assert!(three[1].p_perm.is_nan() && three[1].p_adjusted.is_nan());
        assert_eq!(two[0].p_adjusted, three[0].p_adjusted);
    }

    #[test]
    fn streams_depend_on_data_not_on_names() {
        let vals_a: Vec<(String, f64)> = (0..40)
            .map(|i| (format!("q{i}"), (i % 7) as f64 / 10.0))
            .collect();
        let vals_b: Vec<(String, f64)> = (0..40)
            .map(|i| (format!("q{i}"), (i % 5) as f64 / 10.0))
            .collect();
        let va: Vec<(&str, f64)> = vals_a.iter().map(|(q, v)| (q.as_str(), *v)).collect();
        let vb: Vec<(&str, f64)> = vals_b.iter().map(|(q, v)| (q.as_str(), *v)).collect();
        let mut findings = Vec::new();
        let x = &compare(
            &[run("a", "x/a.run", &va), run("b", "y/b.run", &vb)],
            &[P10],
            &settings(500),
            &mut findings,
        )[0];
        let y = &compare(
            &[
                run("x/a.run", "./x/a.run", &va),
                run("other", "z/b.run", &vb),
            ],
            &[P10],
            &settings(500),
            &mut findings,
        )[0];
        assert_eq!(x.perm_method, PermMethod::MonteCarlo);
        assert_eq!(
            (x.p_perm, x.ci_low, x.ci_high),
            (y.p_perm, y.ci_low, y.ci_high)
        );
    }
}
