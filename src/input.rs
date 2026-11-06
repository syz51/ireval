//! Readers for relevance judgments (qrels) and runs, in TREC or JSON Lines form.
//!
//! TREC qrels lines are `qid iter docid rel` and TREC run lines are
//! `qid Q0 docid rank score tag`, separated by whitespace. As in trec_eval the
//! `iter`, `Q0` and `rank` columns are ignored: ranks are recomputed from the
//! scores. Blank lines and lines starting with `#` are skipped. JSON Lines
//! records are `{"qid","docid","rel"}` and `{"qid","docid","score"}` (plus an
//! optional `"tag"`); `qid` and `docid` may be strings or integers and unknown
//! fields are ignored.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::Path;

use serde_json::Value;

#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Auto,
    Trec,
    Jsonl,
}

impl Format {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(Self::Auto),
            "trec" => Some(Self::Trec),
            "jsonl" => Some(Self::Jsonl),
            _ => None,
        }
    }
}

/// Judgments for one query: docid → (label, source line).
#[derive(Clone, Debug, Default)]
pub struct QueryQrels {
    pub labels: HashMap<String, (i64, usize)>,
}

#[derive(Clone, Debug)]
pub struct Qrels {
    pub path: String,
    pub queries: BTreeMap<String, QueryQrels>,
}

#[derive(Clone, Debug)]
pub struct RunEntry {
    pub docid: String,
    pub score: f64,
    pub line: usize,
}

#[derive(Clone, Debug)]
pub struct Run {
    pub path: String,
    /// Run tag from the first record, when the format carries one.
    pub tag: Option<String>,
    pub queries: BTreeMap<String, Vec<RunEntry>>,
}

fn is_skipped(line: &str) -> bool {
    let t = line.trim_start();
    t.is_empty() || t.starts_with('#')
}

/// Drops a leading UTF-8 byte order mark, which editors on Windows often
/// write and which would otherwise become part of the first query id.
fn strip_bom(text: &str) -> &str {
    text.strip_prefix('\u{feff}').unwrap_or(text)
}

fn detect(text: &str, format: Format) -> Format {
    match format {
        Format::Auto => {
            let first = text.lines().find(|l| !is_skipped(l));
            if first.is_some_and(|l| l.trim_start().starts_with('{')) {
                Format::Jsonl
            } else {
                Format::Trec
            }
        }
        f => f,
    }
}

pub fn read_file(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|e| Error(format!("{}: {e}", path.display())))
}

pub fn parse_qrels(text: &str, path: &str, format: Format) -> Result<Qrels> {
    let text = strip_bom(text);
    let mut queries: BTreeMap<String, QueryQrels> = BTreeMap::new();
    let format = detect(text, format);
    for (i, line) in text.lines().enumerate() {
        let lineno = i + 1;
        if is_skipped(line) {
            continue;
        }
        let at = |msg: String| Error(format!("{path}:{lineno}: {msg}"));
        let (qid, docid, rel) = match format {
            Format::Jsonl => {
                let v: Value =
                    serde_json::from_str(line).map_err(|e| at(format!("invalid JSON: {e}")))?;
                let qid = json_id(&v, "qid").map_err(at)?;
                let docid = json_id(&v, "docid").map_err(at)?;
                let rel = match v.get("rel") {
                    Some(Value::Number(n)) if n.is_i64() => n.as_i64().unwrap_or_default(),
                    Some(other) => {
                        return Err(at(format!("\"rel\" must be an integer, got {other}")));
                    }
                    None => return Err(at("missing \"rel\"".into())),
                };
                (qid, docid, rel)
            }
            _ => {
                let fields: Vec<&str> = line.split_whitespace().collect();
                if fields.len() != 4 {
                    return Err(at(format!(
                        "expected 4 fields `qid iter docid rel`, found {}",
                        fields.len()
                    )));
                }
                let rel: i64 = fields[3]
                    .parse()
                    .map_err(|_| at(format!("relevance {:?} is not an integer", fields[3])))?;
                (fields[0].to_string(), fields[2].to_string(), rel)
            }
        };
        let q = queries.entry(qid.clone()).or_default();
        if let Some((_, first)) = q.labels.get(&docid) {
            return Err(at(format!(
                "duplicate judgment for document {docid:?} in query {qid:?} (first at line {first})"
            )));
        }
        q.labels.insert(docid, (rel, lineno));
    }
    Ok(Qrels {
        path: path.to_string(),
        queries,
    })
}

pub fn parse_run(text: &str, path: &str, format: Format) -> Result<Run> {
    let text = strip_bom(text);
    let mut queries: BTreeMap<String, Vec<RunEntry>> = BTreeMap::new();
    let mut tag = None;
    let format = detect(text, format);
    for (i, line) in text.lines().enumerate() {
        let lineno = i + 1;
        if is_skipped(line) {
            continue;
        }
        let at = |msg: String| Error(format!("{path}:{lineno}: {msg}"));
        let (qid, docid, score, line_tag) = match format {
            Format::Jsonl => {
                let v: Value =
                    serde_json::from_str(line).map_err(|e| at(format!("invalid JSON: {e}")))?;
                let qid = json_id(&v, "qid").map_err(at)?;
                let docid = json_id(&v, "docid").map_err(at)?;
                let score = match v.get("score") {
                    Some(Value::Number(n)) => n.as_f64().unwrap_or(f64::NAN),
                    Some(other) => {
                        return Err(at(format!("\"score\" must be a number, got {other}")));
                    }
                    None => return Err(at("missing \"score\"".into())),
                };
                let line_tag = v.get("tag").and_then(Value::as_str).map(str::to_string);
                (qid, docid, score, line_tag)
            }
            _ => {
                let fields: Vec<&str> = line.split_whitespace().collect();
                if fields.len() != 6 {
                    return Err(at(format!(
                        "expected 6 fields `qid Q0 docid rank score tag`, found {}",
                        fields.len()
                    )));
                }
                let score: f64 = fields[4]
                    .parse()
                    .map_err(|_| at(format!("score {:?} is not a number", fields[4])))?;
                (
                    fields[0].to_string(),
                    fields[2].to_string(),
                    score,
                    Some(fields[5].to_string()),
                )
            }
        };
        if score.is_nan() {
            return Err(at("score is NaN".into()));
        }
        if tag.is_none() {
            tag = line_tag;
        }
        queries.entry(qid).or_default().push(RunEntry {
            docid,
            score,
            line: lineno,
        });
    }
    Ok(Run {
        path: path.to_string(),
        tag,
        queries,
    })
}

fn json_id(v: &Value, key: &str) -> std::result::Result<String, String> {
    match v.get(key) {
        Some(Value::String(s)) if !s.is_empty() => Ok(s.clone()),
        Some(Value::Number(n)) if n.is_u64() || n.is_i64() => Ok(n.to_string()),
        Some(other) => Err(format!(
            "\"{key}\" must be a non-empty string or integer, got {other}"
        )),
        None => Err(format!("missing \"{key}\"")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trec_and_jsonl_agree() {
        let q1 = parse_qrels("q1 0 d1 2\n# c\n\nq1 0 d2 0\n", "a", Format::Auto).unwrap();
        let q2 = parse_qrels(
            "{\"qid\":\"q1\",\"docid\":\"d1\",\"rel\":2}\n{\"qid\":\"q1\",\"docid\":\"d2\",\"rel\":0,\"x\":1}\n",
            "b",
            Format::Auto,
        )
        .unwrap();
        assert_eq!(q1.queries["q1"].labels["d1"].0, 2);
        assert_eq!(q2.queries["q1"].labels["d1"].0, 2);
        assert_eq!(q2.queries["q1"].labels.len(), 2);
        let r = parse_run(
            "{\"qid\":1,\"docid\":\"d1\",\"score\":0.5,\"tag\":\"t\"}\n",
            "r",
            Format::Auto,
        )
        .unwrap();
        assert_eq!(r.tag.as_deref(), Some("t"));
        assert_eq!(r.queries["1"][0].score, 0.5);
    }

    #[test]
    fn leading_byte_order_mark_is_ignored() {
        let q = parse_qrels("\u{feff}q1 0 d1 1\n", "q", Format::Auto).unwrap();
        assert!(q.queries.contains_key("q1"));
        let r = parse_run(
            "\u{feff}{\"qid\":\"q1\",\"docid\":\"d1\",\"score\":1}\n",
            "r",
            Format::Auto,
        )
        .unwrap();
        assert!(r.queries.contains_key("q1"));
    }

    #[test]
    fn errors_name_file_and_line() {
        let e = parse_qrels("q1 0 d1 1\nq1 0 d1 0\n", "qrels.txt", Format::Trec).unwrap_err();
        assert!(e.0.starts_with("qrels.txt:2: duplicate judgment"), "{e}");
        let e = parse_qrels("q1 0 d1 1.5\n", "q", Format::Trec).unwrap_err();
        assert!(e.0.contains("q:1: relevance \"1.5\""), "{e}");
        let e = parse_run("q1 Q0 d1 1 nan t\n", "r", Format::Trec).unwrap_err();
        assert!(e.0.contains("r:1: score is NaN"), "{e}");
        let e = parse_run("q1 Q0 d1 1 0.3\n", "r", Format::Trec).unwrap_err();
        assert!(e.0.contains("expected 6 fields"), "{e}");
    }
}
