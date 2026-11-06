//! `[ireval]` defaults from `research.toml`.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::input::{Error, Result};

/// Values from the `[ireval]` table. Every field is optional; command-line
/// flags override them.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct IrevalConfig {
    pub metrics: Option<Vec<String>>,
    pub cutoffs: Option<Vec<usize>>,
    pub relevance_level: Option<i64>,
    pub missing: Option<String>,
    pub max_docs: Option<usize>,
    pub test: Option<String>,
    pub correction: Option<String>,
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
        assert_eq!(loaded.config.cutoffs, Some(vec![5, 10]));
        assert_eq!(loaded.unknown_keys, ["future_key"]);
        assert_eq!(discover(&dir).as_deref(), Some(path.as_path()));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
