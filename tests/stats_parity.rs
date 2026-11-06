//! Statistics parity against SciPy, NumPy and statsmodels
//! (fixtures from parity/stats_fixtures.py).

use std::path::Path;

use ireval::rng::Xoshiro256pp;
use ireval::stats::{self, Correction, PermMethod};
use serde_json::Value;

fn fixtures() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/stats/stats.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn f(v: &Value) -> f64 {
    match v {
        Value::String(s) if s == "nan" => f64::NAN,
        Value::String(s) if s == "inf" => f64::INFINITY,
        Value::String(s) if s == "-inf" => f64::NEG_INFINITY,
        other => other.as_f64().unwrap(),
    }
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(f).collect()
}

/// Equal within a relative tolerance, treating NaN == NaN and equal infinities.
fn close(a: f64, b: f64, rel: f64, abs: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }
    (a - b).abs() <= abs.max(rel * b.abs())
}

#[test]
fn student_t_tail_matches_scipy() {
    let fx = fixtures();
    let mut max_rel: f64 = 0.0;
    for c in fx["t_dist"].as_array().unwrap() {
        let (t, df, want) = (f(&c["t"]), f(&c["df"]), f(&c["p"]));
        let got = stats::student_t_two_sided(t, df);
        if want > 0.0 {
            max_rel = max_rel.max((got - want).abs() / want);
        }
        assert!(
            close(got, want, 1e-11, 1e-300),
            "t={t} df={df}: {got} vs {want}"
        );
    }
    println!("t tail: max relative difference {max_rel:.3e}");
}

#[test]
fn ln_gamma_matches_scipy() {
    let fx = fixtures();
    for c in fx["ln_gamma"].as_array().unwrap() {
        let (x, want) = (f(&c["x"]), f(&c["value"]));
        let got = stats::ln_gamma(x);
        assert!(close(got, want, 1e-13, 1e-14), "x={x}: {got} vs {want}");
    }
}

#[test]
fn paired_statistics_match_scipy() {
    let fx = fixtures();
    let (mut exact, mut mc, mut boot, mut scipy_agree) = (0, 0, 0, 0);
    let mut max_t_rel: f64 = 0.0;
    for (i, c) in fx["paired"].as_array().unwrap().iter().enumerate() {
        let a = floats(&c["a"]);
        let b = floats(&c["b"]);
        let d: Vec<f64> = a.iter().zip(&b).map(|(x, y)| y - x).collect();
        let tt = stats::paired_t(&d);
        let (want_t, want_p) = (f(&c["t"]), f(&c["p_t"]));
        if want_p.is_finite() && want_p > 0.0 {
            max_t_rel = max_t_rel.max((tt.p - want_p).abs() / want_p);
        }
        // Near-constant differences make t ill-conditioned (SciPy warns about
        // catastrophic cancellation for case 1), hence 1e-9 relative.
        assert!(
            close(tt.t, want_t, 1e-9, 1e-15),
            "case {i}: t {} vs {want_t}",
            tt.t
        );
        assert!(
            close(tt.p, want_p, 1e-9, 1e-300),
            "case {i}: p {} vs {want_p}",
            tt.p
        );
        assert!(close(stats::mean(&d), f(&c["mean_diff"]), 1e-12, 1e-15));
        let sd = stats::sample_sd(&d);
        let d_z = if sd > 0.0 {
            stats::mean(&d) / sd
        } else {
            f64::NAN
        };
        assert!(
            close(d_z, f(&c["cohens_d"]), 1e-9, 1e-15),
            "case {i}: d {d_z}"
        );

        if let Some(p) = c.get("p_exact") {
            exact += 1;
            let mut rng = Xoshiro256pp::from_seed(1);
            let r = stats::paired_randomization(&d, 1 << 16, &mut rng);
            assert_eq!(r.method, PermMethod::Exact);
            assert!(
                close(r.p, f(p), 1e-15, 0.0),
                "case {i}: exact p {} vs {p}",
                r.p
            );
            // SciPy's permutation_test compares floats with a tolerance
            // relative to the observed statistic, so binary rounding residue
            // can turn intended ties into non-ties; count agreements only.
            if close(r.p, f(&c["p_exact_scipy"]), 1e-12, 0.0) {
                scipy_agree += 1;
            }
        }
        if let Some(p) = c.get("p_mc_scipy") {
            mc += 1;
            let want = f(p);
            let trials = 200_000;
            let mut rng = Xoshiro256pp::from_seed(i as u64);
            let r = stats::paired_randomization(&d, trials, &mut rng);
            // With 15-17 non-zero differences 2^m <= trials, so this is exact.
            // Two independent Monte Carlo estimates: allow 5 standard errors.
            let pbar = (r.p + want) / 2.0;
            let se = (2.0 * pbar * (1.0 - pbar) / trials as f64).sqrt();
            assert!(
                (r.p - want).abs() <= 5.0 * se + 2.0 / trials as f64,
                "case {i}: MC p {} vs SciPy {want} (se {se})",
                r.p
            );
        }
        if let Some(ci) = c.get("ci_scipy") {
            boot += 1;
            let want = floats(ci);
            let mut rng = Xoshiro256pp::from_seed(1000 + i as u64);
            let (lo, hi) = stats::bootstrap_mean_ci(&d, 200_000, 0.95, &mut rng);
            // Monte Carlo agreement: within 5% of the standard error of the mean.
            let tol = 0.05 * stats::sample_sd(&d) / (d.len() as f64).sqrt();
            assert!(
                (lo - want[0]).abs() <= tol && (hi - want[1]).abs() <= tol,
                "case {i}: CI ({lo}, {hi}) vs SciPy {want:?} (tol {tol})"
            );
        }
    }
    println!(
        "paired: t-test max relative p difference {max_t_rel:.3e}; {exact} exact ({scipy_agree} equal to SciPy), {mc} Monte Carlo, {boot} bootstrap comparisons"
    );
    assert!(exact >= 20 && mc >= 5 && boot >= 10);
}

#[test]
fn quantiles_match_numpy() {
    let fx = fixtures();
    for c in fx["quantiles"].as_array().unwrap() {
        let x = floats(&c["x"]);
        let got = stats::quantile_sorted(&x, f(&c["q"]));
        assert!(close(got, f(&c["value"]), 1e-14, 1e-15));
    }
}

#[test]
fn corrections_match_statsmodels() {
    let fx = fixtures();
    for c in fx["corrections"].as_array().unwrap() {
        let p = floats(&c["p"]);
        for (method, key) in [
            (Correction::Holm, "holm"),
            (Correction::BenjaminiHochberg, "bh"),
        ] {
            let got = stats::adjust(&p, method);
            let want = floats(&c[key]);
            for (g, w) in got.iter().zip(&want) {
                assert!(
                    close(*g, *w, 1e-14, 1e-16),
                    "{key} {p:?}: {got:?} vs {want:?}"
                );
            }
        }
    }
}

#[test]
fn seeded_results_are_reproducible() {
    let d = [
        0.1, -0.05, 0.2, 0.0, 0.15, -0.1, 0.05, 0.3, 0.12, -0.02, 0.07, 0.01, 0.2, 0.04, -0.03,
    ];
    let run = |seed| {
        let mut r1 = Xoshiro256pp::from_seed(seed);
        let mut r2 = Xoshiro256pp::from_seed(seed);
        (
            stats::paired_randomization(&d, 1000, &mut r1).p,
            stats::bootstrap_mean_ci(&d, 1000, 0.95, &mut r2),
        )
    };
    assert_eq!(run(42), run(42));
}
