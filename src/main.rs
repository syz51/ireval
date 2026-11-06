use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand, ValueEnum};
use serde_json::{Value, json};

use ireval::compare::{self, CompareSettings, Primary};
use ireval::config::{self, IrevalConfig};
use ireval::eval::{self, Duplicates, Missing, RunResult, Settings};
use ireval::findings::{Finding, Severity, now_rfc3339};
use ireval::input::{self, Error, Format, Result};
use ireval::metrics::{self, Metric};
use ireval::output;
use ireval::stats::Correction;

const DEFAULT_METRICS: &[&str] = &[
    "AP",
    "nDCG@10",
    "P@10",
    "R@100",
    "RR",
    "Bpref",
    "NumQ",
    "NumRet",
    "NumRel",
    "NumRelRet",
];

/// Information-retrieval evaluation with trec_eval semantics and paired
/// significance tests.
#[derive(Parser)]
#[command(name = "ireval", version)]
struct Cli {
    /// Path to research.toml (default: search upward from the current directory).
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Evaluate runs and print aggregate (and optionally per-query) metrics.
    Eval {
        #[command(flatten)]
        common: Common,
        /// Run files (TREC or JSON Lines).
        #[arg(required = true)]
        runs: Vec<PathBuf>,
        /// Print a value for every query as well as the aggregate (trec_eval -q).
        #[arg(short = 'q', long)]
        per_query: bool,
        #[arg(long, value_enum, default_value = "table")]
        format: EvalFormat,
        /// Decimal places in table output (CSV and JSON print full precision).
        #[arg(long, default_value_t = 4)]
        precision: usize,
    },
    /// Paired comparison of runs over queries: each run against the first, or all pairs.
    Compare {
        #[command(flatten)]
        common: Common,
        /// Run files; the first is the baseline.
        #[arg(required = true, num_args = 2..)]
        runs: Vec<PathBuf>,
        /// Compare every pair of runs instead of each run against the first.
        #[arg(long)]
        all_pairs: bool,
        /// Test whose p-value is corrected and decides significance.
        #[arg(long, value_enum)]
        test: Option<TestArg>,
        /// Multiple-comparison correction across all comparisons in this invocation.
        #[arg(long, value_enum)]
        correction: Option<CorrectionArg>,
        /// Seed for the randomization test and bootstrap.
        #[arg(long)]
        seed: Option<u64>,
        /// Randomization trials (exact enumeration when 2^n does not exceed this).
        #[arg(long)]
        trials: Option<u64>,
        /// Bootstrap resamples.
        #[arg(long)]
        bootstrap: Option<u64>,
        /// Bootstrap interval coverage.
        #[arg(long)]
        confidence: Option<f64>,
        /// Significance level for adjusted p-values.
        #[arg(long)]
        alpha: Option<f64>,
        /// Warn when a comparison has fewer paired queries than this.
        #[arg(long)]
        min_queries: Option<usize>,
        #[arg(long, value_enum, default_value = "table")]
        format: EvalFormat,
        #[arg(long, default_value_t = 4)]
        precision: usize,
    },
    /// Check qrels and runs for problems without printing metrics.
    Check {
        #[command(flatten)]
        common: Common,
        #[arg(required = true)]
        runs: Vec<PathBuf>,
        #[arg(long, value_enum, default_value = "text")]
        format: CheckFormat,
    },
}

#[derive(Args)]
struct Common {
    /// Relevance judgments (TREC qrels or JSON Lines).
    qrels: PathBuf,
    /// Metric(s), repeatable or comma-separated (e.g. -m nDCG@10,AP).
    #[arg(short = 'm', long = "metric", value_delimiter = ',')]
    metrics: Vec<String>,
    /// Minimum label counted as relevant (trec_eval -l; default 1).
    #[arg(short = 'l', long)]
    relevance_level: Option<i64>,
    /// Queries in the qrels but not in a run: skip them (trec_eval default) or score them 0 (trec_eval -c).
    #[arg(long, value_enum)]
    missing: Option<MissingArg>,
    /// Shorthand for --missing zero (trec_eval -c).
    #[arg(short = 'c', conflicts_with = "missing")]
    complete: bool,
    /// Only the top N documents of each ranking count (trec_eval -M).
    #[arg(short = 'M', long)]
    max_docs: Option<usize>,
    /// A document listed twice for one query: refuse the run (as trec_eval) or keep the higher-ranked copy.
    #[arg(long, value_enum, default_value = "error")]
    duplicates: DuplicatesArg,
    #[arg(long, default_value = "auto")]
    qrels_format: String,
    #[arg(long, default_value = "auto")]
    run_format: String,
    /// Depth for the unjudged-documents warning.
    #[arg(long)]
    unjudged_k: Option<usize>,
    /// Warn when a query's unjudged fraction in the top k exceeds this.
    #[arg(long)]
    unjudged_threshold: Option<f64>,
}

#[derive(Clone, Copy, ValueEnum)]
enum MissingArg {
    Skip,
    Zero,
}

#[derive(Clone, Copy, ValueEnum)]
enum DuplicatesArg {
    Error,
    First,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum EvalFormat {
    Table,
    Csv,
    Json,
    Jsonl,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum CheckFormat {
    Text,
    Jsonl,
}

#[derive(Clone, Copy, ValueEnum)]
enum TestArg {
    T,
    Randomization,
}

#[derive(Clone, Copy, ValueEnum)]
enum CorrectionArg {
    Holm,
    Bh,
    None,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("ireval: {e}");
            ExitCode::from(2)
        }
    }
}

fn usage(msg: impl Into<String>) -> Error {
    Error(msg.into())
}

struct Prepared {
    qrels: input::Qrels,
    runs: Vec<input::Run>,
    names: Vec<String>,
    metrics: Vec<Metric>,
    settings: Settings,
}

fn parse_format(s: &str, what: &str) -> Result<Format> {
    Format::parse(s).ok_or_else(|| usage(format!("--{what}-format must be auto, trec or jsonl")))
}

/// Display names: the file name, or the path as given when file names clash.
fn run_names(paths: &[PathBuf]) -> Vec<String> {
    let base: Vec<String> = paths
        .iter()
        .map(|p| {
            p.file_name().map_or_else(
                || p.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            )
        })
        .collect();
    base.iter()
        .enumerate()
        .map(|(i, b)| {
            if base.iter().filter(|o| *o == b).count() > 1 {
                paths[i].display().to_string()
            } else {
                b.clone()
            }
        })
        .collect()
}

fn prepare(
    common: &Common,
    runs: &[PathBuf],
    cfg: &IrevalConfig,
    require_metrics: bool,
) -> Result<Prepared> {
    let names: Vec<String> = if !common.metrics.is_empty() {
        common.metrics.clone()
    } else if let Some(m) = &cfg.metrics {
        m.clone()
    } else if require_metrics {
        return Err(usage(
            "compare needs at least one -m metric (or `metrics` in [ireval])",
        ));
    } else {
        DEFAULT_METRICS.iter().map(|s| s.to_string()).collect()
    };
    let specs = names
        .iter()
        .map(|n| metrics::parse_spec(n))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(usage)?;
    let cutoffs = cfg.cutoffs.clone().unwrap_or_default();
    if cutoffs.contains(&0) {
        return Err(usage("[ireval] cutoffs must be positive"));
    }
    let metrics = metrics::expand(&specs, &cutoffs).map_err(usage)?;
    let missing = if common.complete {
        Missing::Zero
    } else {
        match (common.missing, cfg.missing.as_deref()) {
            (Some(MissingArg::Skip), _) | (None, None | Some("skip")) => Missing::Skip,
            (Some(MissingArg::Zero), _) | (None, Some("zero")) => Missing::Zero,
            (None, Some(other)) => {
                return Err(usage(format!(
                    "[ireval] missing = {other:?}: expected \"skip\" or \"zero\""
                )));
            }
        }
    };
    let max_docs = common.max_docs.or(cfg.max_docs);
    if max_docs == Some(0) {
        return Err(usage("--max-docs must be positive"));
    }
    let unjudged_k = common.unjudged_k.or(cfg.unjudged_k).unwrap_or(10);
    let unjudged_threshold = common
        .unjudged_threshold
        .or(cfg.unjudged_threshold)
        .unwrap_or(0.1);
    if unjudged_k == 0 || !(0.0..=1.0).contains(&unjudged_threshold) {
        return Err(usage(
            "--unjudged-k must be positive and --unjudged-threshold within [0, 1]",
        ));
    }
    let settings = Settings {
        relevance_level: common.relevance_level.or(cfg.relevance_level).unwrap_or(1),
        missing,
        max_docs,
        duplicates: match common.duplicates {
            DuplicatesArg::Error => Duplicates::Error,
            DuplicatesArg::First => Duplicates::First,
        },
        unjudged_k,
        unjudged_threshold,
    };
    let qpath = common.qrels.display().to_string();
    let qrels = input::parse_qrels(
        &input::read_file(&common.qrels)?,
        &qpath,
        parse_format(&common.qrels_format, "qrels")?,
    )?;
    let run_format = parse_format(&common.run_format, "run")?;
    let parsed = runs
        .iter()
        .map(|p| input::parse_run(&input::read_file(p)?, &p.display().to_string(), run_format))
        .collect::<Result<Vec<_>>>()?;
    Ok(Prepared {
        qrels,
        runs: parsed,
        names: run_names(runs),
        metrics,
        settings,
    })
}

fn settings_json(p: &Prepared) -> Value {
    json!({
        "ireval_version": env!("CARGO_PKG_VERSION"),
        "qrels": p.qrels.path,
        "relevance_level": p.settings.relevance_level,
        "missing": match p.settings.missing { Missing::Skip => "skip", Missing::Zero => "zero" },
        "max_docs": p.settings.max_docs,
        "tie_break": "score descending, then docid descending (trec_eval)",
    })
}

fn evaluate_all(p: &Prepared, findings: &mut Vec<Finding>) -> Result<Vec<RunResult>> {
    let level = p.settings.relevance_level;
    let empty = eval::queries_without_relevant(&p.qrels, level);
    if !empty.is_empty() {
        findings.push(Finding::new(
            "query-without-relevant",
            Severity::Info,
            p.qrels.path.clone(),
            format!(
                "{} of {} qrels queries have no document labelled {level} or higher; they are averaged in and score 0 on most metrics",
                empty.len(),
                p.qrels.queries.len()
            ),
            vec![],
            Some(json!({"count": empty.len(), "relevance_level": level, "qids": eval::first_n(&empty)})),
        ));
    }
    p.runs
        .iter()
        .zip(&p.names)
        .map(|(run, name)| eval::evaluate(&p.qrels, run, name, &p.metrics, &p.settings, findings))
        .collect()
}

fn finish(findings: &mut [Finding], root: Option<&Path>) {
    let at = now_rfc3339();
    for f in findings.iter_mut() {
        f.finish(&at, root);
    }
}

fn print_findings_text(findings: &[Finding]) {
    for f in findings {
        eprintln!("{}", f.to_text());
    }
}

fn exit_for(findings: &[Finding]) -> ExitCode {
    if findings.iter().any(|f| f.severity == Severity::Error) {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

fn print_jsonl(findings: &[Finding]) -> Result<()> {
    for f in findings {
        let line = serde_json::to_string(f).map_err(|e| Error(e.to_string()))?;
        println!("{line}");
    }
    Ok(())
}

fn run(cli: Cli) -> Result<ExitCode> {
    let loaded = config::load(cli.config.as_deref())?;
    for k in &loaded.unknown_keys {
        eprintln!("ireval: warning: research.toml [ireval]: unknown key `{k}` ignored");
    }
    let cfg = &loaded.config;
    let root = loaded.root.as_deref();
    match cli.command {
        Command::Eval {
            common,
            runs,
            per_query,
            format,
            precision,
        } => {
            let p = prepare(&common, &runs, cfg, false)?;
            let mut findings = Vec::new();
            let results = evaluate_all(&p, &mut findings)?;
            finish(&mut findings, root);
            match format {
                EvalFormat::Table => {
                    print!(
                        "{}",
                        output::eval_table(&results, &p.metrics, per_query, precision)
                    );
                    print_findings_text(&findings);
                }
                EvalFormat::Csv => {
                    print!("{}", output::eval_csv(&results, &p.metrics, per_query));
                    print_findings_text(&findings);
                }
                EvalFormat::Json => {
                    let fj = serde_json::to_value(&findings).map_err(|e| Error(e.to_string()))?;
                    let v =
                        output::eval_json(&results, &p.metrics, per_query, settings_json(&p), fj);
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&v).map_err(|e| Error(e.to_string()))?
                    );
                }
                EvalFormat::Jsonl => {
                    let mut all = findings.clone();
                    let at = now_rfc3339();
                    for r in &results {
                        let mut rows: Vec<(&str, &[f64])> = Vec::new();
                        if per_query {
                            rows.extend(
                                r.queries
                                    .iter()
                                    .map(|q| (q.qid.as_str(), q.values.as_slice())),
                            );
                        }
                        rows.push(("all", r.aggregate.as_slice()));
                        for (qid, values) in rows {
                            for (m, &v) in p.metrics.iter().zip(values) {
                                let mut f = Finding::new(
                                    "measurement",
                                    Severity::Info,
                                    r.path.clone(),
                                    format!("{m} = {} ({qid})", output::fmt_float(v, None)),
                                    vec![],
                                    Some(
                                        json!({"run": r.name, "qid": qid, "metric": m.to_string(), "value": output::num(v)}),
                                    ),
                                );
                                f.finish(&at, root);
                                all.push(f);
                            }
                        }
                    }
                    print_jsonl(&all)?;
                }
            }
            Ok(exit_for(&findings))
        }
        Command::Compare {
            common,
            runs,
            all_pairs,
            test,
            correction,
            seed,
            trials,
            bootstrap,
            confidence,
            alpha,
            min_queries,
            format,
            precision,
        } => {
            let p = prepare(&common, &runs, cfg, true)?;
            let primary = match (test, cfg.test.as_deref()) {
                (Some(TestArg::T), _) | (None, Some("t")) => Primary::T,
                (Some(TestArg::Randomization), _) | (None, None | Some("randomization")) => {
                    Primary::Randomization
                }
                (None, Some(o)) => {
                    return Err(usage(format!(
                        "[ireval] test = {o:?}: expected \"t\" or \"randomization\""
                    )));
                }
            };
            let correction = match (correction, cfg.correction.as_deref()) {
                (Some(CorrectionArg::Holm), _) | (None, None | Some("holm")) => Correction::Holm,
                (Some(CorrectionArg::Bh), _) | (None, Some("bh")) => Correction::BenjaminiHochberg,
                (Some(CorrectionArg::None), _) | (None, Some("none")) => Correction::None,
                (None, Some(o)) => {
                    return Err(usage(format!(
                        "[ireval] correction = {o:?}: expected \"holm\", \"bh\" or \"none\""
                    )));
                }
            };
            let s = CompareSettings {
                primary,
                correction,
                seed: seed.or(cfg.seed).unwrap_or(0),
                trials: trials.or(cfg.trials).unwrap_or(10_000),
                bootstrap: bootstrap.or(cfg.bootstrap).unwrap_or(10_000),
                confidence: confidence.or(cfg.confidence).unwrap_or(0.95),
                alpha: alpha.or(cfg.alpha).unwrap_or(0.05),
                min_queries: min_queries.or(cfg.min_queries).unwrap_or(20),
                all_pairs,
            };
            if s.trials == 0 || s.bootstrap == 0 {
                return Err(usage("--trials and --bootstrap must be positive"));
            }
            if !(s.confidence > 0.0 && s.confidence < 1.0) || !(s.alpha > 0.0 && s.alpha <= 1.0) {
                return Err(usage(
                    "--confidence must be in (0, 1) and --alpha in (0, 1]",
                ));
            }
            let mut findings = Vec::new();
            let results = evaluate_all(&p, &mut findings)?;
            let comparisons = compare::compare(&results, &p.metrics, &s, &mut findings);
            finish(&mut findings, root);
            let mut settings = settings_json(&p);
            settings["test"] = json!(match s.primary {
                Primary::T => "t",
                Primary::Randomization => "randomization",
            });
            settings["correction"] = json!(match s.correction {
                Correction::Holm => "holm",
                Correction::BenjaminiHochberg => "bh",
                Correction::None => "none",
            });
            settings["seed"] = json!(s.seed);
            settings["trials"] = json!(s.trials);
            settings["bootstrap"] = json!(s.bootstrap);
            settings["confidence"] = json!(s.confidence);
            settings["alpha"] = json!(s.alpha);
            settings["rng"] = json!(
                "xoshiro256++ seeded by splitmix64; stream = splitmix64(seed ^ fnv1a(metric, a, b, test))"
            );
            match format {
                EvalFormat::Table => {
                    print!("{}", output::compare_table(&comparisons, precision));
                    print_findings_text(&findings);
                }
                EvalFormat::Csv => {
                    print!("{}", output::compare_csv(&comparisons));
                    print_findings_text(&findings);
                }
                EvalFormat::Json => {
                    let v = json!({
                        "schema": "ireval.compare/1",
                        "settings": settings,
                        "runs": results.iter().map(|r| json!({"name": r.name, "path": r.path, "tag": r.tag})).collect::<Vec<_>>(),
                        "comparisons": comparisons.iter().map(output::comparison_json).collect::<Vec<_>>(),
                        "findings": serde_json::to_value(&findings).map_err(|e| Error(e.to_string()))?,
                    });
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&v).map_err(|e| Error(e.to_string()))?
                    );
                }
                EvalFormat::Jsonl => {
                    let mut all = findings.clone();
                    let at = now_rfc3339();
                    for c in &comparisons {
                        let mut f = Finding::new(
                            "comparison",
                            Severity::Info,
                            format!("{} vs {}", c.a, c.b),
                            format!(
                                "{}: diff {} (p_adj {}, n {})",
                                c.metric,
                                output::fmt_float(c.diff, Some(4)),
                                output::fmt_float(c.p_adjusted, Some(4)),
                                c.n
                            ),
                            vec![],
                            Some(output::comparison_json(c)),
                        );
                        f.finish(&at, root);
                        all.push(f);
                    }
                    print_jsonl(&all)?;
                }
            }
            Ok(exit_for(&findings))
        }
        Command::Check {
            mut common,
            runs,
            format,
        } => {
            common.duplicates = DuplicatesArg::First;
            let p = prepare(&common, &runs, cfg, false)?;
            let mut findings = Vec::new();
            evaluate_all(&p, &mut findings)?;
            for f in &mut findings {
                if f.rule == "run-duplicate-doc" {
                    f.severity = Severity::Error;
                    f.message = f.message.replace(
                        "removed (kept the higher-ranked copy)",
                        "found; trec_eval rejects this run",
                    );
                }
            }
            finish(&mut findings, root);
            match format {
                CheckFormat::Text => {
                    for f in &findings {
                        println!("{}", f.to_text());
                    }
                }
                CheckFormat::Jsonl => print_jsonl(&findings)?,
            }
            Ok(exit_for(&findings))
        }
    }
}
