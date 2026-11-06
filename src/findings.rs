//! `rf.finding/1` records (see the research-factory interface contract).

use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Info,
    Warn,
    Error,
}

impl Severity {
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Info => "info",
            Severity::Warn => "warn",
            Severity::Error => "error",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Evidence {
    pub path: String,
    pub locator: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Finding {
    pub schema: &'static str,
    pub tool: &'static str,
    pub rule: &'static str,
    pub severity: Severity,
    pub subject: String,
    pub message: String,
    pub evidence: Vec<Evidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
    pub at: String,
}

impl Finding {
    pub fn new(
        rule: &'static str,
        severity: Severity,
        subject: String,
        message: String,
        evidence: Vec<Evidence>,
        data: Option<Value>,
    ) -> Self {
        Self {
            schema: "rf.finding/1",
            tool: "ireval",
            rule,
            severity,
            subject,
            message,
            evidence,
            data,
            at: String::new(),
        }
    }

    /// Stamps the time and rewrites paths inside `root` relative to it.
    /// Subjects naming several runs (`a vs b`, `a, b, c`) have each path
    /// rewritten.
    pub fn finish(&mut self, at: &str, root: Option<&Path>) {
        self.at = at.to_string();
        if let Some(root) = root {
            self.subject = relativize_subject(&self.subject, root);
            for e in &mut self.evidence {
                e.path = relativize(&e.path, root);
            }
        }
    }

    pub fn to_text(&self) -> String {
        format!(
            "{}[{}] {}: {}",
            self.severity.as_str(),
            self.rule,
            self.subject,
            self.message
        )
    }
}

fn relativize_subject(subject: &str, root: &Path) -> String {
    for sep in [" vs ", ", "] {
        if subject.contains(sep) {
            return subject
                .split(sep)
                .map(|part| relativize(part, root))
                .collect::<Vec<_>>()
                .join(sep);
        }
    }
    relativize(subject, root)
}

/// Absolute form of `path` with `.` and `..` resolved lexically, so that
/// `sub/../x` and `x` compare equal (`std::path::absolute` keeps `..`).
fn normalize(path: &Path) -> Option<PathBuf> {
    let abs = std::path::absolute(path).ok()?;
    let mut out = PathBuf::new();
    for c in abs.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other),
        }
    }
    Some(out)
}

/// Rewrites `path` (optionally followed by `:line`) relative to `root` when
/// it lies inside it.
fn relativize(subject: &str, root: &Path) -> String {
    let (path, suffix) = match subject.rsplit_once(':') {
        Some((p, line)) if !line.is_empty() && line.bytes().all(|b| b.is_ascii_digit()) => {
            (p, format!(":{line}"))
        }
        _ => (subject, String::new()),
    };
    let (Some(abs), Some(root)) = (normalize(Path::new(path)), normalize(root)) else {
        return subject.to_string();
    };
    match abs.strip_prefix(&root) {
        Ok(rel) => format!("{}{suffix}", rel.display()),
        Err(_) => subject.to_string(),
    }
}

/// Current UTC time in RFC 3339 (seconds precision). `SOURCE_DATE_EPOCH`
/// (integer seconds since the Unix epoch) overrides the clock for
/// reproducible output; a value that is set but not an integer is an error,
/// as the reproducible-builds specification asks.
pub fn now_rfc3339() -> Result<String, String> {
    let secs = match std::env::var("SOURCE_DATE_EPOCH") {
        Ok(s) => s.parse::<i64>().map_err(|_| {
            format!("SOURCE_DATE_EPOCH = {s:?}: expected integer seconds since the Unix epoch")
        })?,
        Err(std::env::VarError::NotPresent) => SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs() as i64),
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err("SOURCE_DATE_EPOCH is not valid Unicode".into());
        }
    };
    Ok(rfc3339(secs))
}

pub fn rfc3339(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// Howard Hinnant's days-to-civil algorithm (proleptic Gregorian calendar).
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relativizes_each_path_of_a_subject() {
        let root = std::env::current_dir().unwrap();
        let abs = root.join("x/a.run").display().to_string();
        assert_eq!(relativize_subject(&format!("{abs}:3"), &root), "x/a.run:3");
        assert_eq!(
            relativize_subject(&format!("{abs} vs ./y/b.run"), &root),
            "x/a.run vs y/b.run"
        );
        assert_eq!(
            relativize_subject("sub/../x/a.run, y/b.run", &root),
            "x/a.run, y/b.run"
        );
        assert_eq!(
            relativize_subject("x/a.run", &root.join("sub/..")),
            "x/a.run"
        );
        assert_eq!(
            relativize_subject("/elsewhere/a.run", &root),
            "/elsewhere/a.run"
        );
    }

    #[test]
    fn formats_utc() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(1_794_020_400), "2026-11-07T03:00:00Z");
        assert_eq!(rfc3339(951_782_400), "2000-02-29T00:00:00Z");
    }
}
