//! End-to-end tests of the `ireval` binary on small hand-made fixtures.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/cli")
        .join(name)
}

/// Runs the binary from the crate root with an explicit, empty config, so a
/// research.toml in a directory above the checkout cannot change results.
/// Finding paths are relative to that config's directory, tests/fixtures/cli.
fn command() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_ireval"));
    c.current_dir(env!("CARGO_MANIFEST_DIR"))
        .env("SOURCE_DATE_EPOCH", "1794020400")
        .args(["--config", "tests/fixtures/cli/research.toml"]);
    c
}

fn ireval(args: &[&str]) -> Output {
    command().args(args).output().unwrap()
}

fn stdout(o: &Output) -> String {
    String::from_utf8(o.stdout.clone()).unwrap()
}

fn stderr(o: &Output) -> String {
    String::from_utf8(o.stderr.clone()).unwrap()
}

const Q: &str = "tests/fixtures/cli/qrels.txt";
const A: &str = "tests/fixtures/cli/run_a.txt";
const B: &str = "tests/fixtures/cli/run_b.txt";

#[test]
fn eval_skip_and_complete_match_trec_eval() {
    // Values checked against trec_eval 9.0.8 (-q, with and without -c).
    let o = ireval(&[
        "eval",
        "--format",
        "json",
        "-q",
        "-m",
        "AP,nDCG@3,P@2",
        Q,
        A,
    ]);
    assert!(o.status.success(), "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    let agg = &v["runs"][0]["aggregate"];
    assert_eq!(agg["AP"], 0.2962962962962963);
    assert_eq!(agg["P@2"], 1.0 / 3.0);
    let keys: Vec<&str> = agg
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(keys, ["AP", "nDCG@3", "P@2"]);

    let o = ireval(&[
        "eval",
        "-c",
        "--format",
        "json",
        "-q",
        "-m",
        "AP,nDCG@3,P@2",
        Q,
        A,
    ]);
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    let agg = &v["runs"][0]["aggregate"];
    assert_eq!(agg["AP"], 0.2222222222222222);
    assert_eq!(agg["nDCG@3"], 0.2879597096779397);
    assert_eq!(agg["P@2"], 0.25);
    let q3 = &v["runs"][0]["per_query"][2];
    assert_eq!(q3["qid"], "q3");
    assert_eq!(q3["missing_from_run"], true);
}

#[test]
fn ties_break_by_docid_descending() {
    // run_a ranks q1 as d2 (0.9), then d1/d3 tie at 0.5: trec_eval puts d3
    // (larger docid) first, so P@2 sees d2, d3 → 1 relevant of 2.
    let o = ireval(&["eval", "--format", "csv", "-q", "-m", "P@2,RR", Q, A]);
    let out = stdout(&o);
    assert!(out.contains("run_a.txt,q1,P@2,0.5\n"), "{out}");
    assert!(out.contains("run_a.txt,q1,RR,0.5\n"), "{out}");
    assert!(stderr(&o).contains("warn[tie-at-cutoff]"), "{}", stderr(&o));
}

#[test]
fn duplicates_are_refused_like_trec_eval() {
    let dup = "tests/fixtures/cli/run_dup.txt";
    let o = ireval(&["eval", Q, dup]);
    assert_eq!(o.status.code(), Some(2));
    assert!(stderr(&o).contains("run_dup.txt:3: document \"d1\" appears twice"));
    let o = ireval(&["eval", "--duplicates", "first", "-m", "P@2", Q, dup]);
    assert!(o.status.success());
    assert!(stderr(&o).contains("warn[run-duplicate-doc]"));
    let o = ireval(&["check", "--format", "jsonl", Q, dup]);
    assert_eq!(o.status.code(), Some(1));
    let lines: Vec<Value> = stdout(&o)
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let dup = lines
        .iter()
        .find(|f| f["rule"] == "run-duplicate-doc")
        .unwrap();
    assert_eq!(dup["schema"], "rf.finding/1");
    assert_eq!(dup["tool"], "ireval");
    assert_eq!(dup["severity"], "error");
    assert_eq!(dup["at"], "2026-11-07T03:00:00Z");
    assert_eq!(dup["subject"], "run_dup.txt:3");
}

#[test]
fn check_reports_duplicates_in_unjudged_queries() {
    let dir = std::env::temp_dir().join(format!("ireval-dupx-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let run = dir.join("dupx.run");
    std::fs::write(
        &run,
        "q1 Q0 d1 1 1.0 t\nq9 Q0 d5 1 1.0 t\nq9 Q0 d5 2 0.5 t\n",
    )
    .unwrap();
    let run = run.to_str().unwrap();
    // eval accepts the run, as trec_eval does, but reports the duplicate.
    let o = ireval(&["eval", "-m", "P@1", Q, run]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert!(
        stderr(&o).contains("warn[run-duplicate-doc]"),
        "{}",
        stderr(&o)
    );
    let o = ireval(&["check", Q, run]);
    assert_eq!(o.status.code(), Some(1));
    assert!(
        stdout(&o).contains("error[run-duplicate-doc]"),
        "{}",
        stdout(&o)
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn compare_is_seeded_and_reports_tests() {
    let args = [
        "compare",
        "-c",
        "--format",
        "json",
        "-m",
        "AP,nDCG@10",
        "--seed",
        "7",
        "--trials",
        "4",
        "--bootstrap",
        "500",
        Q,
        A,
        B,
    ];
    let first = stdout(&ireval(&args));
    assert_eq!(first, stdout(&ireval(&args)));
    let v: Value = serde_json::from_str(&first).unwrap();
    let c = &v["comparisons"][0];
    assert_eq!(c["n"], 4);
    assert_eq!(c["metric"], "AP");
    // q4 scores 0 in both runs, so 3 non-zero differences: 2^3 > 4 trials →
    // Monte Carlo with (1 + hits) / (1 + 4).
    assert_eq!(c["randomization_method"], "monte-carlo");
    assert_eq!(v["settings"]["correction"], "holm");
    let rules: Vec<&str> = v["findings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["rule"].as_str().unwrap())
        .collect();
    assert!(rules.contains(&"small-sample"));
    // Exact enumeration once 2^n <= trials.
    let o = ireval(&["compare", "-c", "--format", "json", "-m", "AP", Q, A, B]);
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["comparisons"][0]["randomization_method"], "exact");
    assert_eq!(v["comparisons"][0]["randomization_samples"], 8);
}

#[test]
fn compare_without_complete_drops_unpaired_queries() {
    let o = ireval(&["compare", "--format", "jsonl", "-m", "AP", Q, A, B]);
    let lines: Vec<Value> = stdout(&o)
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(
        lines
            .iter()
            .any(|f| f["rule"] == "comparison-queries-differ")
    );
    let c = lines.iter().find(|f| f["rule"] == "comparison").unwrap();
    assert_eq!(c["data"]["n"], 3);
}

#[test]
fn config_supplies_defaults_and_flags_override() {
    let dir = std::env::temp_dir().join(format!("ireval-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("research.toml"),
        "[ireval]\nmetrics = ['P', 'AP']\ncutoffs = [1, 2]\nmissing = 'zero'\n",
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_ireval"))
        .args(["eval", "--format", "csv"])
        .arg(fixture("qrels.txt"))
        .arg(fixture("run_a.txt"))
        .current_dir(&dir)
        .output()
        .unwrap();
    let text = stdout(&out);
    assert!(text.contains("run_a.txt,all,P@1,"), "{text}");
    assert!(text.contains("run_a.txt,all,P@2,0.25\n"), "{text}");
    let out = Command::new(env!("CARGO_BIN_EXE_ireval"))
        .args(["eval", "--format", "csv", "--missing", "skip", "-m", "P@2"])
        .arg(fixture("qrels.txt"))
        .arg(fixture("run_a.txt"))
        .current_dir(&dir)
        .output()
        .unwrap();
    assert!(stdout(&out).contains("run_a.txt,all,P@2,0.3333333333333333\n"));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn input_errors_exit_2_with_location() {
    let o = ireval(&["eval", Q, "tests/fixtures/cli/does-not-exist.run"]);
    assert_eq!(o.status.code(), Some(2));
    let o = ireval(&["eval", "-m", "P", Q, A]);
    assert_eq!(o.status.code(), Some(2));
    assert!(stderr(&o).contains("P needs a cutoff"));
    // A run sharing no query with the qrels, as trec_eval, is an error.
    let dir = std::env::temp_dir().join(format!("ireval-empty-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let run = dir.join("other.run");
    std::fs::write(&run, "zz Q0 d1 1 1.0 t\n").unwrap();
    for missing in ["skip", "zero"] {
        let o = ireval(&["eval", "--missing", missing, Q, run.to_str().unwrap()]);
        assert_eq!(o.status.code(), Some(2), "{missing}");
        assert!(stderr(&o).contains("no query has both results and judgments"));
    }
    std::fs::remove_dir_all(&dir).unwrap();
    // A malformed SOURCE_DATE_EPOCH is an error, not a silent fallback.
    let o = command()
        .env("SOURCE_DATE_EPOCH", "1794020400.0")
        .args(["eval", Q, A])
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2));
    assert!(stderr(&o).contains("SOURCE_DATE_EPOCH"));
}

#[test]
fn closed_stdout_ends_output_without_a_panic() {
    // Enough output to fill a pipe buffer, read by nobody.
    let dir = std::env::temp_dir().join(format!("ireval-pipe-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (mut q, mut r) = (String::new(), String::new());
    for i in 0..5000 {
        q.push_str(&format!("q{i} 0 d1 1\n"));
        r.push_str(&format!("q{i} Q0 d1 1 1.0 t\nq{i} Q0 d2 2 0.5 t\n"));
    }
    std::fs::write(dir.join("q.txt"), q).unwrap();
    std::fs::write(dir.join("r.run"), r).unwrap();
    let mut child = command()
        .args(["eval", "-q", "--format", "csv"])
        .arg(dir.join("q.txt"))
        .arg(dir.join("r.run"))
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stdout.take());
    let o = child.wait_with_output().unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(o.status.success(), "{:?}: {}", o.status, stderr(&o));
    assert!(!stderr(&o).contains("panicked"), "{}", stderr(&o));
}
