//! Metric definitions with trec_eval semantics.
//!
//! Each function takes one query's ranking, already ordered as trec_eval
//! orders it (score descending, ties by docid descending) and truncated to the
//! maximum depth, mapped to judgments. See the README for each definition and
//! the trec_eval or ir_measures measure it reproduces.

use std::fmt;

/// Judgment of a retrieved document.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Judgment {
    /// Judged with a label `>= 0`.
    Label(i64),
    /// Not in the qrels, or in the qrels with a negative label (trec_eval's
    /// "in pool but unjudged"). Never relevant and never counted as judged.
    Unjudged,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    P,
    Recall,
    AP,
    RPrec,
    RR,
    Ndcg,
    NdcgExp,
    Success,
    Bpref,
    Judged,
    NumRet,
    NumRel,
    NumRelRet,
    NumQ,
}

impl Kind {
    pub fn base_name(self) -> &'static str {
        match self {
            Kind::P => "P",
            Kind::Recall => "R",
            Kind::AP => "AP",
            Kind::RPrec => "Rprec",
            Kind::RR => "RR",
            Kind::Ndcg => "nDCG",
            Kind::NdcgExp => "nDCG_exp",
            Kind::Success => "Success",
            Kind::Bpref => "Bpref",
            Kind::Judged => "Judged",
            Kind::NumRet => "NumRet",
            Kind::NumRel => "NumRel",
            Kind::NumRelRet => "NumRelRet",
            Kind::NumQ => "NumQ",
        }
    }

    /// Whether the metric is only defined at a cutoff.
    pub fn requires_cutoff(self) -> bool {
        matches!(self, Kind::P | Kind::Recall | Kind::Success | Kind::Judged)
    }

    pub fn allows_cutoff(self) -> bool {
        matches!(
            self,
            Kind::P
                | Kind::Recall
                | Kind::AP
                | Kind::RR
                | Kind::Ndcg
                | Kind::NdcgExp
                | Kind::Success
                | Kind::Judged
        )
    }

    /// Counts are summed over queries, as trec_eval reports them; every other
    /// metric is averaged.
    pub fn is_count(self) -> bool {
        matches!(
            self,
            Kind::NumRet | Kind::NumRel | Kind::NumRelRet | Kind::NumQ
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Metric {
    pub kind: Kind,
    pub cutoff: Option<usize>,
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.cutoff {
            Some(k) => write!(f, "{}@{k}", self.kind.base_name()),
            None => f.write_str(self.kind.base_name()),
        }
    }
}

/// A metric name as written by a user, before cutoff expansion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetricSpec {
    pub kind: Kind,
    pub cutoff: Option<usize>,
}

/// Parses a metric name. Accepts ireval names (`P@10`, `nDCG@10`, `AP`,
/// `nDCG_exp@10`), ir_measures names (`R@100`, `RR@10`, `Judged@10`,
/// `nDCG(dcg='exp-log2')@10`) and trec_eval names (`P_10`, `recall_100`,
/// `map`, `map_cut_10`, `ndcg_cut_10`, `recip_rank`, `success_10`, `bpref`,
/// `num_ret`, `num_rel`, `num_rel_ret`, `num_q`). Case-insensitive.
pub fn parse_spec(raw: &str) -> Result<MetricSpec, String> {
    let s = raw.trim();
    let lower = s.to_ascii_lowercase();
    let (name, cutoff) = split_cutoff(&lower)
        .ok_or_else(|| format!("metric {raw:?}: cutoff must be a positive integer"))?;
    let kind = match name {
        "p" | "precision" => Kind::P,
        "r" | "recall" => Kind::Recall,
        "ap" | "map" | "map_cut" => Kind::AP,
        "rprec" | "r-prec" | "r_prec" => Kind::RPrec,
        "rr" | "mrr" | "recip_rank" => Kind::RR,
        "ndcg" | "ndcg_cut" => Kind::Ndcg,
        "ndcg_exp" | "ndcg(dcg='exp-log2')" | "ndcg(dcg=\"exp-log2\")" | "ndcg_burges" => {
            Kind::NdcgExp
        }
        "success" => Kind::Success,
        "bpref" => Kind::Bpref,
        "judged" => Kind::Judged,
        "numret" | "num_ret" => Kind::NumRet,
        "numrel" | "num_rel" => Kind::NumRel,
        "numrelret" | "num_rel_ret" => Kind::NumRelRet,
        "numq" | "num_q" => Kind::NumQ,
        _ => return Err(format!("unknown metric {raw:?}")),
    };
    if cutoff.is_some() && !kind.allows_cutoff() {
        return Err(format!(
            "metric {raw:?}: {} takes no cutoff",
            kind.base_name()
        ));
    }
    Ok(MetricSpec { kind, cutoff })
}

/// Splits `name@k`, `name_k` or `name.k` (trec_eval) into name and cutoff.
fn split_cutoff(s: &str) -> Option<(&str, Option<usize>)> {
    let parse = |k: &str| k.parse::<usize>().ok().filter(|&k| k > 0);
    if let Some((name, k)) = s.rsplit_once('@') {
        return parse(k).map(|k| (name, Some(k)));
    }
    for sep in ['_', '.'] {
        if let Some((name, k)) = s.rsplit_once(sep)
            && !k.is_empty()
            && k.bytes().all(|b| b.is_ascii_digit())
        {
            return parse(k).map(|k| (name, Some(k)));
        }
    }
    Some((s, None))
}

/// Expands specs into metrics: a metric that requires a cutoff and was given
/// none gets one metric per entry of `cutoffs`. Duplicates are removed, first
/// occurrence wins.
pub fn expand(specs: &[MetricSpec], cutoffs: &[usize]) -> Result<Vec<Metric>, String> {
    let mut out: Vec<Metric> = Vec::new();
    let mut push = |m: Metric| {
        if !out.contains(&m) {
            out.push(m);
        }
    };
    for spec in specs {
        match spec.cutoff {
            Some(k) => push(Metric {
                kind: spec.kind,
                cutoff: Some(k),
            }),
            None if spec.kind.requires_cutoff() => {
                if cutoffs.is_empty() {
                    return Err(format!(
                        "{} needs a cutoff: write {}@10 or set `cutoffs` in [ireval]",
                        spec.kind.base_name(),
                        spec.kind.base_name()
                    ));
                }
                for &k in cutoffs {
                    push(Metric {
                        kind: spec.kind,
                        cutoff: Some(k),
                    });
                }
            }
            None => push(Metric {
                kind: spec.kind,
                cutoff: None,
            }),
        }
    }
    Ok(out)
}

/// Everything a metric needs about one query.
#[derive(Clone, Debug)]
pub struct QueryData<'a> {
    /// Retrieved documents in rank order (after any depth limit).
    pub ranked: &'a [Judgment],
    /// Relevance threshold: a label `>= level` is relevant.
    pub level: i64,
    /// Judged documents with label `>= level`.
    pub num_rel: usize,
    /// Judged documents with `0 <= label < level`.
    pub num_judged_nonrel: usize,
    /// Positive labels of all judged documents, descending (for ideal DCG;
    /// independent of `level`, as in trec_eval).
    pub ideal: &'a [i64],
}

impl QueryData<'_> {
    fn is_rel(&self, j: Judgment) -> bool {
        matches!(j, Judgment::Label(l) if l >= self.level)
    }

    fn top(&self, cutoff: Option<usize>) -> &[Judgment] {
        match cutoff {
            Some(k) if k < self.ranked.len() => &self.ranked[..k],
            _ => self.ranked,
        }
    }

    fn rel_in(&self, docs: &[Judgment]) -> usize {
        docs.iter().filter(|&&j| self.is_rel(j)).count()
    }
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

fn dcg(docs: &[Judgment], gain: impl Fn(i64) -> f64) -> f64 {
    let mut sum = 0.0;
    for (i, j) in docs.iter().enumerate() {
        if let Judgment::Label(l) = *j
            && l > 0
        {
            sum += gain(l) / ((i + 2) as f64).log2();
        }
    }
    sum
}

fn ndcg(q: &QueryData<'_>, cutoff: Option<usize>, gain: impl Fn(i64) -> f64) -> f64 {
    let ideal_docs: Vec<Judgment> = q.ideal.iter().map(|&l| Judgment::Label(l)).collect();
    let ideal_top = match cutoff {
        Some(k) if k < ideal_docs.len() => &ideal_docs[..k],
        _ => &ideal_docs[..],
    };
    let idcg = dcg(ideal_top, &gain);
    if idcg > 0.0 {
        dcg(q.top(cutoff), &gain) / idcg
    } else {
        0.0
    }
}

/// Value of `metric` for one query.
pub fn compute(metric: Metric, q: &QueryData<'_>) -> f64 {
    let k = metric.cutoff;
    match metric.kind {
        Kind::P => {
            let k = k.expect("P requires a cutoff");
            q.rel_in(q.top(Some(k))) as f64 / k as f64
        }
        Kind::Recall => ratio(q.rel_in(q.top(k)), q.num_rel),
        Kind::AP => {
            if q.num_rel == 0 {
                return 0.0;
            }
            let mut found = 0usize;
            let mut sum = 0.0;
            for (i, &j) in q.top(k).iter().enumerate() {
                if q.is_rel(j) {
                    found += 1;
                    sum += found as f64 / (i + 1) as f64;
                }
            }
            sum / q.num_rel as f64
        }
        Kind::RPrec => {
            let look = q.ranked.len().min(q.num_rel);
            ratio(q.rel_in(&q.ranked[..look]), q.num_rel)
        }
        Kind::RR => q
            .top(k)
            .iter()
            .position(|&j| q.is_rel(j))
            .map_or(0.0, |i| 1.0 / (i + 1) as f64),
        Kind::Ndcg => ndcg(q, k, |l| l as f64),
        Kind::NdcgExp => ndcg(q, k, |l| (l as f64).exp2() - 1.0),
        Kind::Success => {
            if q.rel_in(q.top(k)) > 0 {
                1.0
            } else {
                0.0
            }
        }
        Kind::Bpref => {
            let mut nonrel_so_far = 0usize;
            let mut sum = 0.0;
            for &j in q.ranked {
                match j {
                    Judgment::Unjudged => {}
                    Judgment::Label(l) if l < q.level => nonrel_so_far += 1,
                    Judgment::Label(_) => {
                        sum += if nonrel_so_far > 0 {
                            1.0 - nonrel_so_far.min(q.num_rel) as f64
                                / q.num_judged_nonrel.min(q.num_rel) as f64
                        } else {
                            1.0
                        };
                    }
                }
            }
            if q.num_rel > 0 {
                sum / q.num_rel as f64
            } else {
                sum
            }
        }
        Kind::Judged => {
            let top = q.top(k);
            let judged = top
                .iter()
                .filter(|j| matches!(j, Judgment::Label(_)))
                .count();
            ratio(judged, top.len())
        }
        Kind::NumRet => q.ranked.len() as f64,
        Kind::NumRel => q.num_rel as f64,
        Kind::NumRelRet => q.rel_in(q.ranked) as f64,
        Kind::NumQ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Judgment::{Label, Unjudged};

    #[test]
    fn parses_reference_names() {
        let cases = [
            ("P@10", "P@10"),
            ("P_10", "P@10"),
            ("p.5", "P@5"),
            ("recall_100", "R@100"),
            ("map", "AP"),
            ("map_cut_10", "AP@10"),
            ("ndcg_cut_10", "nDCG@10"),
            ("nDCG(dcg='exp-log2')@10", "nDCG_exp@10"),
            ("recip_rank", "RR"),
            ("MRR@10", "RR@10"),
            ("Rprec", "Rprec"),
            ("success_1", "Success@1"),
            ("num_rel_ret", "NumRelRet"),
            ("Judged@10", "Judged@10"),
        ];
        for (raw, want) in cases {
            let s = parse_spec(raw).unwrap();
            let m = Metric {
                kind: s.kind,
                cutoff: s.cutoff,
            };
            assert_eq!(m.to_string(), want, "{raw}");
        }
        assert!(parse_spec("bpref@10").is_err());
        assert!(parse_spec("P@0").is_err());
        assert!(parse_spec("foo").is_err());
    }

    #[test]
    fn expansion_requires_cutoffs() {
        let specs = [parse_spec("P").unwrap(), parse_spec("nDCG").unwrap()];
        assert!(expand(&specs, &[]).is_err());
        let m = expand(&specs, &[5, 10]).unwrap();
        let names: Vec<String> = m.iter().map(ToString::to_string).collect();
        assert_eq!(names, ["P@5", "P@10", "nDCG"]);
    }

    #[test]
    fn hand_computed_values() {
        // Ranking: rel 2, unjudged, rel 0, rel 1. Qrels also has a missing
        // relevant doc with label 1 and one non-relevant.
        let ranked = [Label(2), Unjudged, Label(0), Label(1)];
        let ideal = [2, 1, 1];
        let q = QueryData {
            ranked: &ranked,
            level: 1,
            num_rel: 3,
            num_judged_nonrel: 2,
            ideal: &ideal,
        };
        let m = |kind, cutoff| compute(Metric { kind, cutoff }, &q);
        assert_eq!(m(Kind::P, Some(2)), 0.5);
        assert_eq!(m(Kind::P, Some(10)), 0.2);
        assert!((m(Kind::AP, None) - (1.0 + 2.0 / 4.0) / 3.0).abs() < 1e-15);
        assert!((m(Kind::Recall, Some(4)) - 2.0 / 3.0).abs() < 1e-15);
        assert!((m(Kind::RPrec, None) - 1.0 / 3.0).abs() < 1e-15);
        assert_eq!(m(Kind::RR, None), 1.0);
        let dcg = 2.0 + 1.0 / 5f64.log2();
        let idcg = 2.0 + 1.0 / 3f64.log2() + 1.0 / 4f64.log2();
        assert!((m(Kind::Ndcg, None) - dcg / idcg).abs() < 1e-15);
        // bpref: first rel above no nonrel → 1; second rel has 1 nonrel above,
        // min(1, 3) / min(2, 3) = 0.5 → 0.5. Sum 1.5 / 3.
        assert!((m(Kind::Bpref, None) - 0.5).abs() < 1e-15);
        assert_eq!(m(Kind::Judged, Some(2)), 0.5);
        assert_eq!(m(Kind::Judged, Some(10)), 0.75);
    }
}
