//! Manager behaviour on a real GPU, with tiny tensors and a tiny budget.

use cutile::api;
use cutile::prelude::*;
use std::cell::Cell;
use std::rc::Rc;
use tiermem::{Config, Hint, Manager, Policy, TierError};

const N: usize = 1024;
const BYTES: usize = N * size_of::<f32>();

fn tensor(value: f32) -> Tensor<f32> {
    api::full(value, &[N]).sync().unwrap()
}

/// Room for exactly three tensors.
fn manager(policy: Policy) -> Manager<f32> {
    Manager::new(Config::new(3 * BYTES, policy))
}

#[test]
fn strict_fails_recoverably_when_full() {
    let mut m = manager(Policy::Strict);
    for v in 0..3 {
        m.register(tensor(v as f32), Hint::Evictable).unwrap();
    }
    let err = m.register(tensor(3.0), Hint::Evictable).unwrap_err();
    assert!(err.is_recoverable());
    match err {
        TierError::BudgetExceeded {
            could_evict,
            snapshot,
            ..
        } => {
            assert!(could_evict);
            assert_eq!(snapshot.entries.len(), 3);
            assert_eq!(snapshot.resident_bytes, 3 * BYTES);
        }
        other => panic!("expected BudgetExceeded, got {other}"),
    }
}

#[test]
fn evicts_least_recently_used() {
    let mut m = manager(Policy::EvictLru);
    let a = m.register(tensor(0.0), Hint::Evictable).unwrap();
    let b = m.register(tensor(1.0), Hint::Evictable).unwrap();
    let c = m.register(tensor(2.0), Hint::Evictable).unwrap();
    m.ensure_resident(&[a]).unwrap(); // b is now least recent
    let d = m.register(tensor(3.0), Hint::Evictable).unwrap();
    assert!(!m.is_resident(b));
    for id in [a, c, d] {
        assert!(m.is_resident(id));
    }
    assert_eq!(m.stats().resident_bytes, 3 * BYTES);
}

#[test]
fn evictable_goes_before_hot_and_pinned_never_goes() {
    let mut m = manager(Policy::EvictLru);
    let pinned = m.register(tensor(0.0), Hint::Pinned).unwrap();
    let hot = m.register(tensor(1.0), Hint::Hot).unwrap();
    let cold = m.register(tensor(2.0), Hint::Evictable).unwrap();
    m.register(tensor(3.0), Hint::Evictable).unwrap();
    assert!(
        !m.is_resident(cold),
        "evictable evicted first despite being newest"
    );
    assert!(m.is_resident(hot));

    // Evicts the last evictable, leaving pinned + two hot tensors resident.
    let newer_hot = m.register(tensor(4.0), Hint::Hot).unwrap();
    m.register(tensor(5.0), Hint::Evictable).unwrap();
    assert!(
        !m.is_resident(hot),
        "older hot tensor goes once no evictable is resident"
    );
    assert!(m.is_resident(newer_hot));
    assert!(m.is_resident(pinned));
}

#[test]
fn fails_when_only_pinned_tensors_remain() {
    let mut m = manager(Policy::EvictLru);
    for v in 0..3 {
        m.register(tensor(v as f32), Hint::Pinned).unwrap();
    }
    let err = m.register(tensor(9.0), Hint::Evictable).unwrap_err();
    assert!(matches!(
        err,
        TierError::BudgetExceeded {
            could_evict: false,
            ..
        }
    ));
}

#[test]
fn spill_round_trip_preserves_contents_and_shape() {
    let mut m = manager(Policy::EvictLru);
    let shaped = api::full(7.0f32, &[N])
        .sync()
        .unwrap()
        .reshape(&[32, 32])
        .unwrap();
    let a = m.register(shaped, Hint::Evictable).unwrap();
    for v in 0..3 {
        m.register(tensor(v as f32), Hint::Evictable).unwrap();
    }
    assert!(!m.is_resident(a));
    assert!(matches!(m.tensor(a), Err(TierError::NotResident(_))));

    m.ensure_resident(&[a]).unwrap();
    let t = m.tensor(a).unwrap();
    assert_eq!(t.shape(), &[32, 32]);
    let host = m
        .unregister(a)
        .unwrap()
        .unwrap()
        .to_host_vec()
        .sync()
        .unwrap();
    assert!(host.iter().all(|&v| v == 7.0));
    assert_eq!(m.stats().reloads, 1);
}

#[test]
fn reconstructible_tensors_are_rebuilt_not_copied() {
    let mut m = manager(Policy::EvictLru);
    let a = m
        .register_reconstructible(tensor(5.0), Hint::Evictable, || {
            Ok(api::full(5.0f32, &[N]).sync()?)
        })
        .unwrap();
    for v in 0..3 {
        m.register(tensor(v as f32), Hint::Evictable).unwrap();
    }
    m.ensure_resident(&[a]).unwrap();
    let s = m.stats();
    assert_eq!(s.reconstructions, 1);
    assert_eq!(s.reloads, 0);
    let host = m
        .unregister(a)
        .unwrap()
        .unwrap()
        .to_host_vec()
        .sync()
        .unwrap();
    assert!(host.iter().all(|&v| v == 5.0));
}

#[test]
fn ensure_resident_does_not_evict_its_own_arguments() {
    let mut m = manager(Policy::EvictLru);
    let ids: Vec<_> = (0..5)
        .map(|v| m.register(tensor(v as f32), Hint::Evictable).unwrap())
        .collect();
    m.ensure_resident(&ids[..3]).unwrap();
    for id in &ids[..3] {
        assert!(m.is_resident(*id));
    }
}

#[test]
fn pressure_fires_on_crossing_the_soft_limit() {
    let mut m = Manager::new(Config {
        budget_bytes: 4 * BYTES,
        soft_limit_bytes: 2 * BYTES,
        policy: Policy::Strict,
    });
    let fired = Rc::new(Cell::new(0));
    let f = fired.clone();
    m.on_pressure(move |p| {
        assert!(p.resident_bytes > p.soft_limit_bytes);
        f.set(f.get() + 1);
    });
    m.register(tensor(0.0), Hint::Evictable).unwrap();
    m.register(tensor(1.0), Hint::Evictable).unwrap();
    assert_eq!(fired.get(), 0, "at the soft limit, not above it");
    m.register(tensor(2.0), Hint::Evictable).unwrap();
    m.register(tensor(3.0), Hint::Evictable).unwrap();
    assert_eq!(
        fired.get(),
        1,
        "fires once per crossing, before the budget is hit"
    );
}

#[test]
fn lookahead_evicts_the_tensor_needed_farthest_away() {
    let mut m = manager(Policy::EvictLookahead);
    let a = m.register(tensor(0.0), Hint::Evictable).unwrap();
    let b = m.register(tensor(1.0), Hint::Evictable).unwrap();
    let c = m.register(tensor(2.0), Hint::Evictable).unwrap();
    // a is least recent, so LRU would evict it; lookahead keeps it.
    m.expect_accesses(&[a, c, b]).unwrap();
    let d = m.register(tensor(3.0), Hint::Evictable).unwrap();
    assert!(!m.is_resident(b), "b is needed last");
    for id in [a, c, d] {
        assert!(m.is_resident(id));
    }
}

#[test]
fn lookahead_evicts_undeclared_tensors_first() {
    let mut m = manager(Policy::EvictLookahead);
    let a = m.register(tensor(0.0), Hint::Evictable).unwrap();
    let b = m.register(tensor(1.0), Hint::Evictable).unwrap();
    let c = m.register(tensor(2.0), Hint::Evictable).unwrap();
    m.expect_accesses(&[a, b]).unwrap();
    m.register(tensor(3.0), Hint::Evictable).unwrap();
    assert!(!m.is_resident(c), "c has no declared use");
    assert!(m.is_resident(a) && m.is_resident(b));
}

#[test]
fn lookahead_consumes_declarations_as_tensors_are_used() {
    let mut m = manager(Policy::EvictLookahead);
    let a = m.register(tensor(0.0), Hint::Evictable).unwrap();
    let b = m.register(tensor(1.0), Hint::Evictable).unwrap();
    let c = m.register(tensor(2.0), Hint::Evictable).unwrap();
    m.expect_accesses(&[a, b, c, a]).unwrap();
    // Skipping a and b consumes their declarations; only the final a remains.
    m.ensure_resident(&[c]).unwrap();
    m.register(tensor(3.0), Hint::Evictable).unwrap();
    assert!(m.is_resident(a), "a is still expected");
    assert!(!m.is_resident(b), "b's declared use was skipped");
}

#[test]
fn lru_ignores_declared_accesses() {
    let mut m = manager(Policy::EvictLru);
    let a = m.register(tensor(0.0), Hint::Evictable).unwrap();
    let b = m.register(tensor(1.0), Hint::Evictable).unwrap();
    m.register(tensor(2.0), Hint::Evictable).unwrap();
    m.expect_accesses(&[a, b]).unwrap();
    m.register(tensor(3.0), Hint::Evictable).unwrap();
    assert!(!m.is_resident(a));
}

#[test]
fn expect_accesses_rejects_unknown_tensors() {
    let mut m = manager(Policy::EvictLookahead);
    let a = m.register(tensor(0.0), Hint::Evictable).unwrap();
    m.unregister(a).unwrap();
    assert!(matches!(
        m.expect_accesses(&[a]),
        Err(TierError::UnknownTensor(_))
    ));
}
