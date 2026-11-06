//! Paired comparisons of runs over queries.

use std::collections::BTreeSet;

use serde_json::json;

use crate::eval::{RunResult, first_n};
use crate::findings::{Evidence, Finding, Severity};
use crate::metrics::Metric;
use crate::rng::{Xoshiro256pp, stream_seed};
use crate::stats::{self, Correction, PermMethod};

/// Which test's p-value is corrected and used for significance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    pub a: String,
    pub b: String,
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
            let tt = stats::paired_t(&d);
            let mut perm_rng =
                Xoshiro256pp::from_seed(stream_seed(s.seed, &[&name, &ra.name, &rb.name, "perm"]));
            let perm = stats::paired_randomization(&d, s.trials, &mut perm_rng);
            let mut boot_rng = Xoshiro256pp::from_seed(stream_seed(
                s.seed,
                &[&name, &ra.name, &rb.name, "bootstrap"],
            ));
            let (ci_low, ci_high) =
                stats::bootstrap_mean_ci(&d, s.bootstrap, s.confidence, &mut boot_rng);
            let sd = stats::sample_sd(&d);
            let diff = if n > 0 { stats::mean(&d) } else { f64::NAN };
            let cohens_d = if sd > 0.0 { diff / sd } else { f64::NAN };
            if n > 1 && sd == 0.0 {
                findings.push(Finding::new(
                    "zero-variance",
                    Severity::Warn,
                    format!("{} vs {}", ra.path, rb.path),
                    format!(
                        "{name}: every paired difference is {diff}; the t statistic and Cohen's d are undefined or infinite"
                    ),
                    vec![],
                    Some(json!({"metric": name, "a": ra.name, "b": rb.name, "n": n})),
                ));
            }
            out.push(Comparison {
                metric,
                a: ra.name.clone(),
                b: rb.name.clone(),
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
