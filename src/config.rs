//! `[ireval]` defaults from `research.toml`.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::compare::Primary;
use crate::eval::Missing;
use crate::input::{Error, Result};
use crate::stats::Correction;

/// Values from the `[ireval]` table, validated when the file is loaded. Every
/// field is optional; command-line flags override them.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct IrevalConfig {
    pub metrics: Option<Vec<String>>,
    pub cutoffs: Option<Vec<usize>>,
    pub relevance_level: Option<i64>,
    pub missing: Option<Missing>,
    pub max_docs: Option<usize>,
    pub test: Option<Primary>,
    pub correction: Option<Correction>,
    pub seed: Option<u64>,
    pub trials: Option<u64>,
    pub bootstrap: Option<u64>,
    pub confidence: Option<f64>,
    pub alpha: Option<f64>,
    pub min_queries: Option<usize>,
    pub unjudged_k: Option<usize>,
    pub unjudged_threshold: Option<f64>,
}

const KNOWN_KEYS: &[&str] = &[
    "metrics",
    "cutoffs",
    "relevance_level",
    "missing",
    "max_docs",
    "test",
    "correction",
    "seed",
    "trials",
    "bootstrap",
    "confidence",
    "alpha",
    "min_queries",
    "unjudged_k",
    "unjudged_threshold",
];

#[derive(Clone, Debug, Default)]
pub struct Loaded {
    /// Directory of `research.toml`, used to relativize finding paths.
    pub root: Option<PathBuf>,
    pub config: IrevalConfig,
    /// Keys in `[ireval]` this version does not know.
    pub unknown_keys: Vec<String>,
}

/// Finds `research.toml` by walking up from `start`.
pub fn discover(start: &Path) -> Option<PathBuf> {
    let mut dir = Some(start);
    while let Some(d) = dir {
        let candidate = d.join("research.toml");
        if candidate.is_file() {
            return Some(candidate);
        }
        dir = d.parent();
    }
    None
}

pub fn load(explicit: Option<&Path>) -> Result<Loaded> {
    let path = match explicit {
        Some(p) => Some(p.to_path_buf()),
        None => {
            let cwd = std::env::current_dir()
                .map_err(|e| Error(format!("cannot read the current directory: {e}")))?;
            discover(&cwd)
        }
    };
    let Some(path) = path else {
        return Ok(Loaded::default());
    };
    let text =
        std::fs::read_to_string(&path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    let table: toml::Table =
        toml::from_str(&text).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    let root = path
        .parent()
        .map(|p| {
            if p.as_os_str().is_empty() {
                Path::new(".")
            } else {
                p
            }
        })
        .map(Path::to_path_buf);
    let Some(section) = table.get("ireval") else {
        return Ok(Loaded {
            root,
            ..Loaded::default()
        });
    };
    let unknown_keys = section
        .as_table()
        .map(|t| {
            t.keys()
                .filter(|k| !KNOWN_KEYS.contains(&k.as_str()))
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    let config: IrevalConfig = section
        .clone()
        .try_into()
        .map_err(|e| Error(format!("{}: [ireval]: {e}", path.display())))?;
    if config.metrics.as_ref().is_some_and(Vec::is_empty) {
        return Err(Error(format!(
            "{}: [ireval]: `metrics` is empty; list at least one metric or remove the key",
            path.display()
        )));
    }
    Ok(Loaded {
        root,
        config,
        unknown_keys,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_table_and_reports_unknown_keys() {
        let dir = std::env::temp_dir().join(format!("ireval-config-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("research.toml");
        std::fs::write(
            &path,
            "[project]\nname='x'\n[ireval]\nmetrics=['nDCG@10','AP']\ncutoffs=[5,10]\nseed=7\ncorrection='bh'\nfuture_key=1\n[papers]\nregistry='r'\n",
        )
        .unwrap();
        let loaded = load(Some(&path)).unwrap();
        assert_eq!(loaded.config.seed, Some(7));
        assert_eq!(
            loaded.config.correction,
            Some(Correction::BenjaminiHochberg)
        );
        assert_eq!(loaded.config.cutoffs, Some(vec![5, 10]));
        assert_eq!(loaded.unknown_keys, ["future_key"]);
        assert_eq!(discover(&dir).as_deref(), Some(path.as_path()));
        // Invalid values fail at load, whichever subcommand runs.
        for bad in [
            "correction='bonferroni'",
            "test='wilcoxon'",
            "missing='drop'",
            "metrics=[]",
        ] {
            std::fs::write(&path, format!("[ireval]\n{bad}\n")).unwrap();
            let e = load(Some(&path)).unwrap_err();
            assert!(e.0.contains("[ireval]"), "{bad}: {e}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
