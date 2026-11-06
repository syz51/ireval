//! Metric parity against trec_eval 9.0.8 and 10.0-rc3, pytrec_eval,
//! ir_measures and ranx on randomized fixtures (see parity/README.md).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use ireval::eval::{self, Duplicates, Missing, Settings};
use ireval::input::{self, Format};
use ireval::metrics::{self, Metric};
use serde_json::Value;

const TOLERANCE: f64 = 1e-12;

fn metric(name: &str) -> Metric {
    let s = metrics::parse_spec(name).unwrap();
    Metric {
        kind: s.kind,
        cutoff: s.cutoff,
    }
}

#[test]
fn metrics_match_reference_implementations() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/parity");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with(".expected.json"))
        .collect();
    files.sort();
    assert!(files.len() >= 100, "expected the committed parity fixtures");

    // (metric family, reference) → (comparisons, max |diff|)
    let mut summary: BTreeMap<(String, String), (usize, f64)> = BTreeMap::new();
    let mut failures = Vec::new();
    let mut evaluations = 0;
    let mut max_per_query: f64 = 0.0;
    for file in &files {
        let spec: Value = serde_json::from_str(&std::fs::read_to_string(file).unwrap()).unwrap();
        let qpath = dir.join(spec["qrels"].as_str().unwrap());
        let rpath = dir.join(spec["run"].as_str().unwrap());
        let qrels = input::parse_qrels(
            &std::fs::read_to_string(&qpath).unwrap(),
            &qpath.display().to_string(),
            Format::Auto,
        )
        .unwrap();
        let run = input::parse_run(
            &std::fs::read_to_string(&rpath).unwrap(),
            &rpath.display().to_string(),
            Format::Auto,
        )
        .unwrap();
        for ev in spec["evaluations"].as_array().unwrap() {
            evaluations += 1;
            let settings = Settings {
                relevance_level: ev["relevance_level"].as_i64().unwrap(),
                missing: match ev["missing"].as_str().unwrap() {
                    "skip" => Missing::Skip,
                    _ => Missing::Zero,
                },
                max_docs: ev["max_docs"].as_u64().map(|d| d as usize),
                duplicates: Duplicates::Error,
                unjudged_k: 10,
                unjudged_threshold: 0.1,
            };
            let qids: Vec<&str> = ev["expected"]["qids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|q| q.as_str().unwrap())
                .collect();
            let values = ev["expected"]["values"].as_object().unwrap();
            let ms: Vec<Metric> = values.keys().map(|n| metric(n)).collect();
            let mut findings = Vec::new();
            let result =
                eval::evaluate(&qrels, &run, "run", &ms, &settings, &mut findings).unwrap();
            let mut got: BTreeMap<&str, &[f64]> = result
                .queries
                .iter()
                .map(|q| (q.qid.as_str(), q.values.as_slice()))
                .collect();
            let expected_qids: BTreeSet<&str> =
                qids.iter().copied().filter(|q| *q != "all").collect();
            let got_qids: BTreeSet<&str> = got.keys().copied().collect();
            if expected_qids != got_qids {
                failures.push(format!(
                    "{} level={} missing={}: query set {got_qids:?} != references {expected_qids:?}",
                    file.display(),
                    ev["relevance_level"],
                    ev["missing"]
                ));
            }
            got.insert("all", result.aggregate.as_slice());
            for (idx, (name, by_refs)) in values.iter().enumerate() {
                let family = name.split('@').next().unwrap().to_string();
                for (refs, column) in by_refs.as_object().unwrap() {
                    for (qid, want) in qids.iter().zip(column.as_array().unwrap()) {
                        let Some(want) = want.as_f64() else { continue };
                        let Some(row) = got.get(qid) else { continue };
                        let mine = row[idx];
                        let diff = (mine - want).abs();
                        if *qid != "all" {
                            max_per_query = max_per_query.max(diff);
                        }
                        for r in refs.split(' ') {
                            let s = summary.entry((family.clone(), r.to_string())).or_default();
                            s.0 += 1;
                            s.1 = s.1.max(diff);
                        }
                        if diff > TOLERANCE || diff.is_nan() {
                            failures.push(format!(
                                "{} level={} missing={} max_docs={}: {qid} {name}: ireval {mine} vs {want} ({refs})",
                                file.display(),
                                ev["relevance_level"],
                                ev["missing"],
                                ev["max_docs"]
                            ));
                        }
                    }
                }
            }
        }
    }
    println!("{} fixture files, {evaluations} evaluations", files.len());
    println!("max |diff| over per-query values: {max_per_query:e}");
    println!(
        "{:<12} {:<5} {:>8} {:>10}",
        "metric", "ref", "values", "max |diff|"
    );
    for ((family, r), (n, max)) in &summary {
        println!("{family:<12} {r:<5} {n:>8} {max:>10.3e}");
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures[..failures.len().min(30)].join("\n")
    );
}
