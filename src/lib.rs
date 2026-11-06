//! Information-retrieval evaluation with trec_eval semantics and paired
//! significance tests. The `ireval` binary is the main interface; this library
//! exposes the same pieces for tests and embedding.

pub mod compare;
pub mod config;
pub mod eval;
pub mod findings;
pub mod input;
pub mod metrics;
pub mod output;
pub mod rng;
pub mod stats;
