//! `rf.finding/1` records (see the research-factory interface contract).

use std::path::Path;
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
    pub fn finish(&mut self, at: &str, root: Option<&Path>) {
        self.at = at.to_string();
        if let Some(root) = root {
            self.subject = relativize(&self.subject, root);
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

/// Rewrites `path` (optionally followed by `:line`) relative to `root` when
/// it lies inside it.
fn relativize(subject: &str, root: &Path) -> String {
    let (path, suffix) = match subject.rsplit_once(':') {
        Some((p, line)) if !line.is_empty() && line.bytes().all(|b| b.is_ascii_digit()) => {
            (p, format!(":{line}"))
        }
        _ => (subject, String::new()),
    };
    let Ok(abs) = std::path::absolute(path) else {
        return subject.to_string();
    };
    let root = std::path::absolute(root).unwrap_or_else(|_| root.to_path_buf());
    match abs.strip_prefix(&root) {
        Ok(rel) => format!("{}{suffix}", rel.display()),
        Err(_) => subject.to_string(),
    }
}

/// Current UTC time in RFC 3339 (seconds precision). `SOURCE_DATE_EPOCH`
/// overrides the clock for reproducible output.
pub fn now_rfc3339() -> String {
    let secs = std::env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs() as i64)
        });
    rfc3339(secs)
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
    fn formats_utc() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(1_794_020_400), "2026-11-07T03:00:00Z");
        assert_eq!(rfc3339(951_782_400), "2000-02-29T00:00:00Z");
    }
}
