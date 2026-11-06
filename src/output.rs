//! Table, CSV and JSON rendering.

use serde_json::{Map, Value, json};

use crate::compare::Comparison;
use crate::eval::RunResult;
use crate::metrics::Metric;
use crate::stats::PermMethod;

/// Number as JSON: NaN and infinities become null.
pub fn num(x: f64) -> Value {
    serde_json::Number::from_f64(x).map_or(Value::Null, Value::Number)
}

fn fmt_value(metric: Metric, x: f64, precision: Option<usize>) -> String {
    if metric.kind.is_count() {
        return format!("{x:.0}");
    }
    fmt_float(x, precision)
}

/// Full precision (shortest round-trip) when `precision` is None.
pub fn fmt_float(x: f64, precision: Option<usize>) -> String {
    if x.is_nan() {
        return "NA".into();
    }
    match precision {
        Some(p) => format!("{x:.p$}"),
        None => format!("{x}"),
    }
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn render_table(rows: &[Vec<String>], right_from: usize) -> String {
    let cols = rows.iter().map(Vec::len).max().unwrap_or(0);
    let widths: Vec<usize> = (0..cols)
        .map(|c| {
            rows.iter()
                .filter_map(|r| r.get(c))
                .map(|s| s.chars().count())
                .max()
                .unwrap_or(0)
        })
        .collect();
    let mut out = String::new();
    for r in rows {
        let cells: Vec<String> = r
            .iter()
            .enumerate()
            .map(|(c, s)| {
                if c >= right_from {
                    format!("{s:>w$}", w = widths[c])
                } else {
                    format!("{s:<w$}", w = widths[c])
                }
            })
            .collect();
        out.push_str(cells.join("  ").trim_end());
        out.push('\n');
    }
    out
}

pub fn eval_table(
    runs: &[RunResult],
    metrics: &[Metric],
    per_query: bool,
    precision: usize,
) -> String {
    let p = Some(precision);
    let mut out = String::new();
    if per_query {
        for (ri, run) in runs.iter().enumerate() {
            if ri > 0 {
                out.push('\n');
            }
            out.push_str(&format!("# {}\n", run.name));
            let mut rows = vec![
                std::iter::once("qid".to_string())
                    .chain(metrics.iter().map(ToString::to_string))
                    .collect::<Vec<_>>(),
            ];
            for q in &run.queries {
                rows.push(
                    std::iter::once(q.qid.clone())
                        .chain(
                            metrics
                                .iter()
                                .zip(&q.values)
                                .map(|(&m, &v)| fmt_value(m, v, p)),
                        )
                        .collect(),
                );
            }
            rows.push(
                std::iter::once("all".to_string())
                    .chain(
                        metrics
                            .iter()
                            .zip(&run.aggregate)
                            .map(|(&m, &v)| fmt_value(m, v, p)),
                    )
                    .collect(),
            );
            out.push_str(&render_table(&rows, 1));
        }
        return out;
    }
    let mut rows = vec![
        std::iter::once("metric".to_string())
            .chain(runs.iter().map(|r| r.name.clone()))
            .collect::<Vec<_>>(),
    ];
    for (mi, &m) in metrics.iter().enumerate() {
        rows.push(
            std::iter::once(m.to_string())
                .chain(runs.iter().map(|r| fmt_value(m, r.aggregate[mi], p)))
                .collect(),
        );
    }
    out.push_str(&render_table(&rows, 1));
    out
}

pub fn eval_csv(runs: &[RunResult], metrics: &[Metric], per_query: bool) -> String {
    let mut out = String::from("run,qid,metric,value\n");
    for run in runs {
        let name = csv_field(&run.name);
        if per_query {
            for q in &run.queries {
                for (&m, &v) in metrics.iter().zip(&q.values) {
                    out.push_str(&format!(
                        "{name},{},{m},{}\n",
                        csv_field(&q.qid),
                        fmt_value(m, v, None)
                    ));
                }
            }
        }
        for (&m, &v) in metrics.iter().zip(&run.aggregate) {
            out.push_str(&format!("{name},all,{m},{}\n", fmt_value(m, v, None)));
        }
    }
    out
}

fn metric_map(metrics: &[Metric], values: &[f64]) -> Value {
    let mut map = Map::new();
    for (m, &v) in metrics.iter().zip(values) {
        map.insert(m.to_string(), num(v));
    }
    Value::Object(map)
}

pub fn eval_json(
    runs: &[RunResult],
    metrics: &[Metric],
    per_query: bool,
    settings: Value,
    findings: Value,
) -> Value {
    let runs: Vec<Value> = runs
        .iter()
        .map(|r| {
            let mut obj = json!({
                "name": r.name,
                "path": r.path,
                "tag": r.tag,
                "aggregate": metric_map(metrics, &r.aggregate),
            });
            if per_query {
                obj["per_query"] = r
                    .queries
                    .iter()
                    .map(|q| {
                        json!({
                            "qid": q.qid,
                            "missing_from_run": q.missing,
                            "values": metric_map(metrics, &q.values),
                        })
                    })
                    .collect();
            }
            obj
        })
        .collect();
    json!({
        "schema": "ireval.eval/1",
        "settings": settings,
        "metrics": metrics.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "runs": runs,
        "findings": findings,
    })
}

pub fn comparison_json(c: &Comparison) -> Value {
    json!({
        "metric": c.metric.to_string(),
        "a": c.a,
        "b": c.b,
        "n": c.n,
        "mean_a": num(c.mean_a),
        "mean_b": num(c.mean_b),
        "diff": num(c.diff),
        "ci_low": num(c.ci_low),
        "ci_high": num(c.ci_high),
        "cohens_d": num(c.cohens_d),
        "t": num(c.t),
        "df": num(c.df),
        "p_t": num(c.p_t),
        "p_randomization": num(c.p_perm),
        "randomization_method": match c.perm_method {
            PermMethod::Exact => "exact",
            PermMethod::MonteCarlo => "monte-carlo",
        },
        "randomization_samples": c.perm_samples,
        "p_adjusted": num(c.p_adjusted),
        "significant": c.significant,
    })
}

const COMPARE_HEADER: [&str; 13] = [
    "metric", "a", "b", "n", "mean_a", "mean_b", "diff", "ci_low", "ci_high", "cohens_d", "p_t",
    "p_rand", "p_adj",
];

fn comparison_cells(c: &Comparison, p: Option<usize>) -> Vec<String> {
    vec![
        c.metric.to_string(),
        c.a.clone(),
        c.b.clone(),
        c.n.to_string(),
        fmt_float(c.mean_a, p),
        fmt_float(c.mean_b, p),
        fmt_float(c.diff, p),
        fmt_float(c.ci_low, p),
        fmt_float(c.ci_high, p),
        fmt_float(c.cohens_d, p),
        fmt_float(c.p_t, p),
        fmt_float(c.p_perm, p),
        fmt_float(c.p_adjusted, p),
    ]
}

pub fn compare_table(cs: &[Comparison], precision: usize) -> String {
    let mut rows = vec![
        COMPARE_HEADER
            .iter()
            .map(|s| s.to_string())
            .chain(std::iter::once("sig".to_string()))
            .collect::<Vec<_>>(),
    ];
    for c in cs {
        let mut r = comparison_cells(c, Some(precision));
        r.push(if c.significant {
            "*".into()
        } else {
            String::new()
        });
        rows.push(r);
    }
    render_table(&rows, 3)
}

pub fn compare_csv(cs: &[Comparison]) -> String {
    let mut out = COMPARE_HEADER.join(",");
    out.push_str(",significant\n");
    for c in cs {
        let cells: Vec<String> = comparison_cells(c, None)
            .iter()
            .map(|s| csv_field(s))
            .collect();
        out.push_str(&cells.join(","));
        out.push_str(&format!(",{}\n", c.significant));
    }
    out
}
