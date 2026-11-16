//! Manager behaviour on a real GPU, with tiny tensors and a tiny budget.

use cutile::api;
use cutile::prelude::*;
use std::cell::Cell;
use std::rc::Rc;
use std::time::Duration;
use tiermem::{Config, Hint, HostAlloc, Manager, Policy, TierError};

const N: usize = 1024;
const BYTES: usize = N * size_of::<f32>();

fn tensor(value: f32) -> Tensor<f32> {
    api::full(value, &[N]).sync().unwrap()
}

/// Room for exactly three tensors.
fn manager(policy: Policy) -> Manager {
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
    assert!(matches!(m.tensor::<f32>(a), Err(TierError::NotResident(_))));

    m.ensure_resident(&[a]).unwrap();
    let t = m.tensor::<f32>(a).unwrap();
    assert_eq!(t.shape(), &[32, 32]);
    let host = m
        .unregister::<f32>(a)
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
        .unregister::<f32>(a)
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
        soft_limit_bytes: 2 * BYTES,
        pressure_rearm_bytes: 2 * BYTES,
        ..Config::new(4 * BYTES, Policy::Strict)
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
    m.unregister::<f32>(a).unwrap();
    assert!(matches!(
        m.expect_accesses(&[a]),
        Err(TierError::UnknownTensor(_))
    ));
}

/// Registers `a` (all 7.0) and three fillers, then reloads `a`, which spills
/// `b`. Returns `[a, b, c, d]`; `a` is resident with a clean host copy.
fn reloaded_spill(m: &mut Manager) -> [tiermem::TensorId; 4] {
    let ids = [7.0, 1.0, 2.0, 3.0].map(|v| m.register(tensor(v), Hint::Evictable).unwrap());
    assert!(!m.is_resident(ids[0]));
    m.ensure_resident(&ids[..1]).unwrap();
    assert!(!m.is_resident(ids[1]));
    ids
}

#[test]
fn clean_reeviction_skips_the_copy() {
    let mut m = manager(Policy::EvictLru);
    let [a, b, c, d] = reloaded_spill(&mut m);
    m.ensure_resident(&[c, d]).unwrap(); // a is now least recent
    let before = m.stats().bytes_to_host;
    m.ensure_resident(&[b]).unwrap();
    assert!(!m.is_resident(a));
    let s = m.stats();
    assert_eq!(
        s.bytes_to_host, before,
        "unchanged weights are not copied back"
    );
    assert_eq!(s.clean_evictions, 1);

    m.ensure_resident(&[a]).unwrap();
    let host = m
        .unregister::<f32>(a)
        .unwrap()
        .unwrap()
        .to_host_vec()
        .sync()
        .unwrap();
    assert!(host.iter().all(|&v| v == 7.0));
}

#[test]
fn tensor_mut_forces_write_back() {
    let mut m = manager(Policy::EvictLru);
    let [a, b, c, d] = reloaded_spill(&mut m);
    let host_before = m.stats().host_bytes;
    *m.tensor_mut::<f32>(a).unwrap() = tensor(9.0);
    assert_eq!(
        m.stats().host_bytes,
        host_before - BYTES,
        "clean copy dropped"
    );

    m.ensure_resident(&[c, d]).unwrap();
    let before = m.stats().bytes_to_host;
    m.ensure_resident(&[b]).unwrap();
    assert!(!m.is_resident(a));
    let s = m.stats();
    assert_eq!(s.bytes_to_host, before + BYTES as u64);
    assert_eq!(s.clean_evictions, 0);

    m.ensure_resident(&[a]).unwrap();
    let host = m
        .unregister::<f32>(a)
        .unwrap()
        .unwrap()
        .to_host_vec()
        .sync()
        .unwrap();
    assert!(
        host.iter().all(|&v| v == 9.0),
        "reload sees the new contents"
    );
}

#[test]
fn tensor_mut_requires_residency() {
    let mut m = manager(Policy::EvictLru);
    let [_, b, _, _] = reloaded_spill(&mut m);
    assert!(matches!(
        m.tensor_mut::<f32>(b),
        Err(TierError::NotResident(_))
    ));
}

#[test]
fn host_bytes_tracks_spills_and_clean_copies() {
    let mut m = manager(Policy::EvictLru);
    let [a, b, ..] = reloaded_spill(&mut m);
    let s = m.stats();
    assert_eq!(s.host_bytes, 2 * BYTES, "a's clean copy plus spilled b");
    assert_eq!(s.spilled_bytes, BYTES, "only b is evicted");
    m.unregister::<f32>(a).unwrap();
    assert_eq!(m.stats().host_bytes, BYTES);
    m.unregister::<f32>(b).unwrap();
    assert_eq!(m.stats().host_bytes, 0);
}

#[test]
fn pressure_stays_quiet_until_residency_falls_to_the_rearm_level() {
    let mut m = Manager::new(Config {
        soft_limit_bytes: 3 * BYTES,
        pressure_rearm_bytes: BYTES,
        ..Config::new(4 * BYTES, Policy::EvictLru)
    });
    let fired = Rc::new(Cell::new(0));
    let f = fired.clone();
    m.on_pressure(move |_| f.set(f.get() + 1));
    let ids: Vec<_> = (0..4)
        .map(|v| m.register(tensor(v as f32), Hint::Evictable).unwrap())
        .collect();
    assert_eq!(fired.get(), 1);

    // Each admission evicts one tensor and re-crosses the soft limit.
    let mut extra: Vec<_> = (4..8)
        .map(|v| m.register(tensor(v as f32), Hint::Evictable).unwrap())
        .collect();
    assert_eq!(fired.get(), 1, "oscillating above the re-arm level");

    extra.extend(ids);
    for id in extra {
        m.unregister::<f32>(id).unwrap();
    }
    assert_eq!(m.stats().resident_bytes, 0);
    for v in 0..4 {
        m.register(tensor(v as f32), Hint::Evictable).unwrap();
    }
    assert_eq!(fired.get(), 2, "re-armed after falling to the re-arm level");
}

/// Large enough that a reload is still in flight when `prefetch` returns.
const BIG_N: usize = 8 * 1024 * 1024;
const BIG: usize = BIG_N * size_of::<f32>();

/// Room for three big tensors; registers four, so the first (all 7.0) is
/// spilled.
fn big_manager() -> (Manager, [tiermem::TensorId; 4]) {
    let mut m = Manager::new(Config::new(3 * BIG, Policy::EvictLru));
    let ids = [7.0f32, 1.0, 2.0, 3.0].map(|v| {
        m.register(api::full(v, &[BIG_N]).sync().unwrap(), Hint::Evictable)
            .unwrap()
    });
    assert!(!m.is_resident(ids[0]));
    (m, ids)
}

fn contents(m: &mut Manager, id: tiermem::TensorId) -> Vec<f32> {
    m.ensure_resident(&[id]).unwrap();
    m.unregister::<f32>(id)
        .unwrap()
        .unwrap()
        .to_host_vec()
        .sync()
        .unwrap()
}

#[test]
fn prefetch_reloads_without_blocking_and_preserves_contents() {
    let mut m = manager(Policy::EvictLru);
    let [a, b, c, d] =
        [7.0, 1.0, 2.0, 3.0].map(|v| m.register(tensor(v), Hint::Evictable).unwrap());
    assert!(!m.is_resident(a));
    m.ensure_resident(&[d]).unwrap();

    assert_eq!(m.prefetch(&[a]).unwrap(), 1);
    assert!(!m.is_resident(b), "b was least recent and made room");
    assert!(m.is_resident(c) && m.is_resident(d));
    let s = m.stats();
    assert_eq!(s.prefetches, 1);
    assert_eq!(s.resident_bytes, 3 * BYTES, "in-flight reloads hold VRAM");
    assert_eq!(m.prefetch(&[a]).unwrap(), 0, "already on its way");

    m.ensure_resident(&[a]).unwrap();
    assert_eq!(m.stats().reloads, 1);
    assert_eq!(m.tensor::<f32>(a).unwrap().shape(), &[N as i32]);
    assert!(contents(&mut m, a).iter().all(|&v| v == 7.0));
}

#[test]
fn prefetch_spares_tensors_in_use() {
    let mut m = manager(Policy::EvictLru);
    let [a, b, c, d] =
        [0.0, 1.0, 2.0, 3.0].map(|v| m.register(tensor(v), Hint::Evictable).unwrap());
    m.ensure_resident(&[b, c, d]).unwrap();
    assert_eq!(
        m.prefetch(&[a]).unwrap(),
        0,
        "everything resident is in use"
    );
    assert!(m.is_resident(b) && m.is_resident(c) && m.is_resident(d));
    assert_eq!(m.stats().evictions, 1, "only a's original eviction");
}

#[test]
fn evicting_a_prefetched_tensor_keeps_its_contents() {
    let (mut m, [a, b, c, d]) = big_manager();
    assert_eq!(m.prefetch(&[a]).unwrap(), 1);
    // b was evicted for a; bringing it back with c and d leaves no room for a.
    m.ensure_resident(&[b, c, d]).unwrap();
    assert!(!m.is_resident(a));
    let s = m.stats();
    assert_eq!(s.prefetches_cancelled, 1);
    assert_eq!(s.reloads, 1, "only b; a's reload was cancelled");
    assert_eq!(s.resident_bytes, 3 * BIG);
    assert_eq!(s.spilled_bytes, BIG);
    m.unregister::<f32>(b).unwrap();
    assert!(contents(&mut m, a).iter().all(|&v| v == 7.0));
}

#[test]
fn unregistering_a_prefetched_tensor_returns_it() {
    let (mut m, [a, ..]) = big_manager();
    assert_eq!(m.prefetch(&[a]).unwrap(), 1);
    let host = m
        .unregister::<f32>(a)
        .unwrap()
        .expect("on the device")
        .to_host_vec()
        .sync()
        .unwrap();
    assert!(host.iter().all(|&v| v == 7.0));
    let s = m.stats();
    assert_eq!(s.resident_bytes, 2 * BIG);
    assert_eq!(s.host_bytes, BIG, "only the tensor evicted for a");
}

#[test]
fn automatic_prefetch_follows_the_declared_order() {
    let mut m = Manager::new(Config {
        prefetch_depth: 1,
        ..Config::new(3 * BYTES, Policy::EvictLookahead)
    });
    let ids: Vec<_> = (0..6)
        .map(|v| m.register(tensor(v as f32), Hint::Evictable).unwrap())
        .collect();
    for _ in 0..3 {
        m.expect_accesses(&ids).unwrap();
    }
    for _ in 0..3 {
        for (v, &id) in ids.iter().enumerate() {
            m.ensure_resident(&[id]).unwrap();
            let host = m.tensor::<f32>(id).unwrap().dup().sync().unwrap();
            let host = host.to_host_vec().sync().unwrap();
            assert!(host.iter().all(|&x| x == v as f32));
        }
    }
    let s = m.stats();
    assert!(s.prefetches > 0);
    assert_eq!(s.prefetches_cancelled, 0, "the plan was followed");
    assert!(s.resident_bytes <= 3 * BYTES);
    assert!(s.peak_resident_bytes <= 3 * BYTES);
}

#[test]
fn prefetch_does_not_evict_a_tensor_needed_sooner() {
    let mut m = Manager::new(Config {
        prefetch_depth: 2,
        ..Config::new(2 * BYTES, Policy::EvictLookahead)
    });
    let [a, b, c] = [0.0, 1.0, 2.0].map(|v| m.register(tensor(v), Hint::Evictable).unwrap());
    assert!(!m.is_resident(a));
    m.expect_accesses(&[b, c, a]).unwrap();
    // After b, the plan is [c, a]: c is resident, and fetching a would have
    // to evict c, which is needed first.
    m.ensure_resident(&[b]).unwrap();
    assert!(m.is_resident(c));
    assert_eq!(m.stats().prefetches, 0);
}

#[test]
fn host_limit_frees_clean_copies_before_failing() {
    let mut m = Manager::new(Config {
        host_limit_bytes: 2 * BYTES,
        ..Config::new(3 * BYTES, Policy::EvictLru)
    });
    // a has a clean copy and b is spilled: the host limit is reached.
    let [a, b, c, d] = reloaded_spill(&mut m);
    assert_eq!(m.stats().host_bytes, 2 * BYTES);

    // Reloading b evicts c, which needs host room: a's clean copy goes.
    m.ensure_resident(&[b]).unwrap();
    assert!(!m.is_resident(c));
    let s = m.stats();
    assert_eq!(s.clean_copies_dropped, 1);
    assert_eq!(s.host_bytes, 2 * BYTES, "b's clean copy plus spilled c");

    // d is next; spilling it needs b's clean copy to go as well.
    m.ensure_resident(&[c]).unwrap();
    assert!(!m.is_resident(d));
    assert_eq!(
        m.stats().host_bytes,
        2 * BYTES,
        "spilled d plus c's clean copy"
    );

    m.ensure_resident(&[d]).unwrap();
    // The last clean copy (d's) makes room for one more spill.
    let e = m.register(tensor(9.0), Hint::Evictable).unwrap();
    let err = m.register(tensor(9.0), Hint::Evictable).unwrap_err();
    assert!(matches!(err, TierError::HostBudgetExceeded { .. }), "{err}");
    assert!(err.is_recoverable());
    assert_eq!(m.stats().resident_bytes, 3 * BYTES, "nothing was evicted");
    m.unregister::<f32>(e).unwrap();
    for (id, v) in [(a, 7.0), (b, 1.0), (c, 2.0), (d, 3.0)] {
        assert!(contents(&mut m, id).iter().all(|&x| x == v), "tensor {v}");
    }
}

#[test]
fn waiting_for_an_in_flight_prefetch_yields_the_contents() {
    let (mut m, [a, _, c, d]) = big_manager();
    m.ensure_resident(&[c, d]).unwrap();
    assert_eq!(m.prefetch(&[a]).unwrap(), 1);
    assert!(!m.is_resident(a), "not resident until waited for");
    assert!(matches!(m.tensor::<f32>(a), Err(TierError::NotResident(_))));
    assert!(
        m.is_resident(c) && m.is_resident(d),
        "tensors in use are spared"
    );
    assert!(contents(&mut m, a).iter().all(|&v| v == 7.0));
    assert_eq!(m.stats().reloads, 1);
}

#[test]
fn lazy_pins_on_first_spill_and_reuses_freed_blocks() {
    let mut m = manager(Policy::EvictLru);
    assert_eq!(
        m.stats().pinned_bytes,
        0,
        "nothing is pinned before a spill"
    );
    let [a, _, c, d] = reloaded_spill(&mut m);
    let s = m.stats();
    assert_eq!(s.pinned_bytes, 2 * BYTES, "a's clean copy plus spilled b");
    assert!(s.pin_time > Duration::ZERO);

    // Writing to a gives up its clean copy; the block stays page-locked.
    *m.tensor_mut::<f32>(a).unwrap() = tensor(9.0);
    assert_eq!(m.stats().host_bytes, BYTES);
    assert_eq!(m.stats().pinned_bytes, 2 * BYTES);

    // The next spill reuses that block instead of page-locking a new one.
    m.ensure_resident(&[a, d]).unwrap();
    m.register(tensor(4.0), Hint::Evictable).unwrap();
    assert!(!m.is_resident(c));
    let s = m.stats();
    assert_eq!(s.host_bytes, 2 * BYTES);
    assert_eq!(s.pinned_bytes, 2 * BYTES, "no new page-locked memory");
    assert!(contents(&mut m, c).iter().all(|&v| v == 2.0));
}

#[test]
fn eager_arena_is_pinned_once_at_the_first_registration() {
    let mut m = Manager::new(Config {
        host_alloc: HostAlloc::Eager,
        host_limit_bytes: 4 * BYTES,
        ..Config::new(3 * BYTES, Policy::EvictLru)
    });
    assert_eq!(
        m.stats().pinned_bytes,
        0,
        "the manager has not touched the device"
    );
    let first = m.register(tensor(7.0), Hint::Evictable).unwrap();
    let pinned = m.stats();
    assert_eq!(pinned.pinned_bytes, 4 * BYTES, "the whole arena, up front");
    assert!(pinned.pin_time > Duration::ZERO);

    let rest = [1.0, 2.0, 3.0].map(|v| m.register(tensor(v), Hint::Evictable).unwrap());
    assert!(!m.is_resident(first));
    for (id, v) in [(first, 7.0), (rest[0], 1.0), (rest[1], 2.0), (rest[2], 3.0)] {
        m.ensure_resident(&[id]).unwrap();
        let host = m.tensor::<f32>(id).unwrap().dup().sync().unwrap();
        assert!(host.to_host_vec().sync().unwrap().iter().all(|&x| x == v));
    }
    let s = m.stats();
    assert!(s.evictions >= 4);
    assert_eq!(
        s.pinned_bytes,
        4 * BYTES,
        "spills are served from the arena"
    );
    assert_eq!(s.pin_time, pinned.pin_time, "nothing was page-locked later");
}

#[test]
fn eager_arena_is_the_host_limit() {
    let mut m = Manager::new(Config {
        host_alloc: HostAlloc::Eager,
        host_limit_bytes: 2 * BYTES,
        ..Config::new(3 * BYTES, Policy::EvictLru)
    });
    let [a, b, c, d] = reloaded_spill(&mut m);
    // The arena holds a's clean copy and spilled b; reloading b spills c,
    // which takes over the block of a's clean copy.
    m.ensure_resident(&[b]).unwrap();
    assert!(!m.is_resident(c));
    assert_eq!(m.stats().clean_copies_dropped, 1);
    m.ensure_resident(&[c]).unwrap();
    m.ensure_resident(&[d]).unwrap();
    let e = m.register(tensor(9.0), Hint::Evictable).unwrap();
    let err = m.register(tensor(9.0), Hint::Evictable).unwrap_err();
    assert!(matches!(err, TierError::HostBudgetExceeded { .. }), "{err}");
    assert_eq!(m.stats().pinned_bytes, 2 * BYTES, "the arena never grows");
    m.unregister::<f32>(e).unwrap();
    for (id, v) in [(a, 7.0), (b, 1.0), (c, 2.0), (d, 3.0)] {
        assert!(contents(&mut m, id).iter().all(|&x| x == v), "tensor {v}");
    }
}

#[test]
#[should_panic(expected = "host limit")]
fn eager_allocation_needs_a_host_limit() {
    let _ = Manager::new(Config {
        host_alloc: HostAlloc::Eager,
        ..Config::new(3 * BYTES, Policy::EvictLru)
    });
}

#[test]
fn lazy_unpins_unused_blocks_when_the_host_limit_needs_the_room() {
    let mut m = Manager::new(Config {
        host_limit_bytes: 2 * BYTES,
        ..Config::new(3 * BYTES, Policy::EvictLru)
    });
    let [s1, s2, s3, s4] =
        [0.0, 1.0, 2.0, 3.0].map(|v| m.register(tensor(v), Hint::Evictable).unwrap());
    assert!(!m.is_resident(s1));
    // s1's block stays page-locked but unused; it is too small for `big`.
    for id in [s1, s2, s3] {
        m.unregister::<f32>(id).unwrap();
    }
    assert_eq!(m.stats().pinned_bytes, BYTES);
    let big = m
        .register(api::full(5.0f32, &[2 * N]).sync().unwrap(), Hint::Evictable)
        .unwrap();
    m.ensure_resident(&[s4]).unwrap(); // big is now least recent
    let last = m.register(tensor(6.0), Hint::Evictable).unwrap();
    assert!(!m.is_resident(big));
    let s = m.stats();
    assert_eq!(s.host_bytes, 2 * BYTES);
    assert_eq!(s.pinned_bytes, 2 * BYTES, "the unused block was unpinned");
    // Reloading big must not need to spill anything: the host is full.
    m.unregister::<f32>(last).unwrap();
    assert!(contents(&mut m, big).iter().all(|&v| v == 5.0));
}

#[test]
fn the_device_reports_its_async_engine_count() {
    assert!(tiermem::async_engine_count().unwrap() >= 1);
}

#[test]
fn write_back_makes_the_next_eviction_a_free() {
    let mut m = manager(Policy::EvictLru);
    let [a, b, c] = [7.0, 1.0, 2.0].map(|v| m.register(tensor(v), Hint::Evictable).unwrap());
    assert_eq!(m.write_back(&[a]).unwrap(), 1);
    let s = m.stats();
    assert_eq!(s.write_backs, 1);
    assert_eq!(s.bytes_to_host, BYTES as u64);
    assert_eq!(s.host_bytes, BYTES);
    assert_eq!(s.evictions, 0);
    assert!(m.is_resident(a), "writing back does not evict");
    assert_eq!(m.write_back(&[a]).unwrap(), 0, "already has a host copy");

    m.register(tensor(3.0), Hint::Evictable).unwrap();
    assert!(!m.is_resident(a));
    let s = m.stats();
    assert_eq!(s.bytes_to_host, BYTES as u64, "the eviction copied nothing");
    assert_eq!(s.clean_evictions, 1);
    assert!(m.is_resident(b) && m.is_resident(c));
    assert!(contents(&mut m, a).iter().all(|&v| v == 7.0));
}

#[test]
fn a_tensor_being_written_back_stays_resident_and_readable() {
    let mut m = Manager::new(Config::new(3 * BIG, Policy::EvictLru));
    let a = m
        .register(api::full(7.0f32, &[BIG_N]).sync().unwrap(), Hint::Evictable)
        .unwrap();
    assert_eq!(m.write_back(&[a]).unwrap(), 1);
    assert!(m.is_resident(a));
    assert_eq!(
        m.stats().resident_bytes,
        BIG,
        "the device memory is still held"
    );
    let copy = m.tensor::<f32>(a).unwrap().dup().sync().unwrap();
    assert!(copy.to_host_vec().sync().unwrap().iter().all(|&v| v == 7.0));
}

#[test]
fn evicting_during_a_write_back_waits_for_the_copy() {
    let mut m = Manager::new(Config::new(3 * BIG, Policy::EvictLru));
    let [a, _, _] = [7.0f32, 1.0, 2.0].map(|v| {
        m.register(api::full(v, &[BIG_N]).sync().unwrap(), Hint::Evictable)
            .unwrap()
    });
    assert_eq!(m.write_back(&[a]).unwrap(), 1);
    m.register(api::full(3.0f32, &[BIG_N]).sync().unwrap(), Hint::Evictable)
        .unwrap();
    assert!(!m.is_resident(a));
    let s = m.stats();
    assert_eq!(s.bytes_to_host, BIG as u64, "one copy, not two");
    assert_eq!(s.resident_bytes, 3 * BIG);
    assert!(contents(&mut m, a).iter().all(|&v| v == 7.0));
}

#[test]
fn writing_during_a_write_back_discards_the_stale_copy() {
    let mut m = Manager::new(Config::new(3 * BIG, Policy::EvictLru));
    let [a, _, _] = [7.0f32, 1.0, 2.0].map(|v| {
        m.register(api::full(v, &[BIG_N]).sync().unwrap(), Hint::Evictable)
            .unwrap()
    });
    assert_eq!(m.write_back(&[a]).unwrap(), 1);
    *m.tensor_mut::<f32>(a).unwrap() = api::full(9.0f32, &[BIG_N]).sync().unwrap();
    assert_eq!(
        m.stats().host_bytes,
        0,
        "the copy of the old contents is gone"
    );

    m.register(api::full(3.0f32, &[BIG_N]).sync().unwrap(), Hint::Evictable)
        .unwrap();
    assert!(!m.is_resident(a));
    assert_eq!(m.stats().bytes_to_host, 2 * BIG as u64, "copied again");
    assert!(contents(&mut m, a).iter().all(|&v| v == 9.0));
}

#[test]
fn unregistering_during_a_write_back_returns_the_tensor() {
    let mut m = Manager::new(Config::new(3 * BIG, Policy::EvictLru));
    let a = m
        .register(api::full(7.0f32, &[BIG_N]).sync().unwrap(), Hint::Evictable)
        .unwrap();
    assert_eq!(m.write_back(&[a]).unwrap(), 1);
    let host = m
        .unregister::<f32>(a)
        .unwrap()
        .expect("on the device")
        .to_host_vec()
        .sync()
        .unwrap();
    assert!(host.iter().all(|&v| v == 7.0));
    let s = m.stats();
    assert_eq!((s.resident_bytes, s.host_bytes), (0, 0));
}

#[test]
fn write_back_skips_tensors_with_nothing_to_copy() {
    let mut m = manager(Policy::EvictLru);
    let [a, b, _, _] = reloaded_spill(&mut m);
    let rebuilt = m
        .register_reconstructible(tensor(5.0), Hint::Evictable, || {
            Ok(api::full(5.0f32, &[N]).sync()?)
        })
        .unwrap();
    m.ensure_resident(&[a, rebuilt]).unwrap();
    let before = m.stats().bytes_to_host;
    assert_eq!(
        m.write_back(&[a, b, rebuilt]).unwrap(),
        0,
        "a is clean, b is on the host, and the third is rebuilt, not copied"
    );
    assert_eq!(m.stats().bytes_to_host, before);
    m.unregister::<f32>(a).unwrap();
    assert!(matches!(
        m.write_back(&[a]),
        Err(TierError::UnknownTensor(_))
    ));
}

#[test]
fn automatic_prefetch_starts_a_write_back_instead_of_waiting_for_one() {
    let mut m = Manager::new(Config {
        prefetch_depth: 1,
        ..Config::new(3 * BYTES, Policy::EvictLookahead)
    });
    let [a, b, c, d] =
        [0.0, 1.0, 2.0, 3.0].map(|v| m.register(tensor(v), Hint::Evictable).unwrap());
    assert!(!m.is_resident(a));
    let spilled = m.stats().bytes_to_host;
    m.expect_accesses(&[b, a]).unwrap();
    // Fetching a needs room, and neither c nor d has a host copy. Rather
    // than copy one back before returning, the manager starts the copy.
    m.ensure_resident(&[b]).unwrap();
    let s = m.stats();
    assert_eq!(s.prefetches, 0, "nothing could be evicted for free");
    assert_eq!(s.write_backs, 1);
    assert_eq!(s.evictions, 1, "only a's original eviction");
    assert!(m.is_resident(c) && m.is_resident(d));

    // The access itself evicts the tensor that was written back.
    m.ensure_resident(&[a]).unwrap();
    let s = m.stats();
    assert_eq!(s.bytes_to_host, spilled + BYTES as u64);
    assert_eq!(s.clean_evictions, 1);
}

/// Sweeps six tensors through room for four, overwriting each one every time
/// it is used, and checks that every tensor ends up with the last value
/// written to it.
fn dirty_sweep(copy_streams: usize) -> tiermem::Stats {
    let mut m = Manager::new(Config {
        prefetch_depth: 1,
        copy_streams,
        ..Config::new(4 * BIG, Policy::EvictLookahead)
    });
    let ids: Vec<_> = (0..6)
        .map(|v| {
            m.register(
                api::full(v as f32, &[BIG_N]).sync().unwrap(),
                Hint::Evictable,
            )
            .unwrap()
        })
        .collect();
    const EPOCHS: usize = 4;
    for _ in 0..EPOCHS {
        m.expect_accesses(&ids).unwrap();
    }
    for epoch in 0..EPOCHS {
        for (v, &id) in ids.iter().enumerate() {
            m.ensure_resident(&[id]).unwrap();
            let seen = m.tensor::<f32>(id).unwrap().dup().sync().unwrap();
            let seen = seen.to_host_vec().sync().unwrap();
            let expected = (100 * epoch + v) as f32;
            assert!(
                seen.iter().all(|&x| x == expected),
                "epoch {epoch} tensor {v}"
            );
            *m.tensor_mut::<f32>(id).unwrap() =
                api::full(expected + 100.0, &[BIG_N]).sync().unwrap();
        }
    }
    let s = m.stats();
    assert!(s.peak_resident_bytes <= 4 * BIG);
    for (v, &id) in ids.iter().enumerate() {
        let expected = (100 * EPOCHS + v) as f32;
        assert!(contents(&mut m, id).iter().all(|&x| x == expected));
    }
    s
}

#[test]
fn written_tensors_survive_write_behind_and_prefetch_on_two_streams() {
    let s = dirty_sweep(2);
    assert!(s.write_backs > 0, "copies to the host were started ahead");
    assert!(s.prefetches > 0, "and reloads too");
}

#[test]
fn written_tensors_survive_write_behind_and_prefetch_on_one_stream() {
    let s = dirty_sweep(1);
    assert!(s.write_backs > 0 && s.prefetches > 0);
}

#[test]
#[should_panic(expected = "copy streams")]
fn a_manager_has_one_or_two_copy_streams() {
    let _ = Manager::new(Config {
        copy_streams: 3,
        ..Config::new(3 * BYTES, Policy::EvictLru)
    });
}

// ---- Found by review: the host limit together with prefetch, write-back and
// ---- blocks of different sizes.

fn sized(value: f32, units: usize) -> Tensor<f32> {
    api::full(value, &[units * N]).sync().unwrap()
}

#[test]
fn automatic_prefetch_works_under_a_host_limit_that_works_without_it() {
    for depth in [0, 1, 2] {
        let mut m = Manager::new(Config {
            prefetch_depth: depth,
            host_limit_bytes: 2 * BYTES,
            ..Config::new(2 * BYTES, Policy::EvictLookahead)
        });
        let ids = [0.0, 1.0, 2.0].map(|v| m.register(tensor(v), Hint::Evictable).unwrap());
        for _ in 0..4 {
            m.expect_accesses(&ids).unwrap();
        }
        for round in 0..4 {
            for (v, &id) in ids.iter().enumerate() {
                m.ensure_resident(&[id])
                    .unwrap_or_else(|e| panic!("depth {depth} round {round} tensor {v}: {e}"));
                let seen = m.tensor::<f32>(id).unwrap().dup().sync().unwrap();
                assert!(
                    seen.to_host_vec()
                        .sync()
                        .unwrap()
                        .iter()
                        .all(|&x| x == v as f32)
                );
            }
        }
        assert!(m.stats().host_bytes <= 2 * BYTES);
    }
}

/// The host has room for one tensor and `b`'s write-back takes it. Evicting
/// `a` then needs that block, whether or not the copy has finished.
fn eviction_reclaims_a_write_back(settle: bool) {
    let mut m = Manager::new(Config {
        host_limit_bytes: BIG,
        ..Config::new(2 * BIG, Policy::EvictLru)
    });
    let [a, b] = [1.0f32, 2.0].map(|v| {
        m.register(api::full(v, &[BIG_N]).sync().unwrap(), Hint::Evictable)
            .unwrap()
    });
    assert_eq!(m.write_back(&[b]).unwrap(), 1);
    if settle {
        std::thread::sleep(Duration::from_millis(200));
    }
    let c = m
        .register(api::full(3.0f32, &[BIG_N]).sync().unwrap(), Hint::Evictable)
        .unwrap();
    assert!(!m.is_resident(a) && m.is_resident(b));
    assert_eq!(
        m.stats().host_bytes,
        BIG,
        "only a's contents are on the host"
    );
    m.unregister::<f32>(c).unwrap();
    assert!(contents(&mut m, a).iter().all(|&v| v == 1.0));
    assert!(contents(&mut m, b).iter().all(|&v| v == 2.0));
}

#[test]
fn eviction_reclaims_the_block_of_a_finished_write_back() {
    eviction_reclaims_a_write_back(true);
}

#[test]
fn eviction_reclaims_the_block_of_a_write_back_in_flight() {
    eviction_reclaims_a_write_back(false);
}

#[test]
fn write_back_does_not_take_other_tensors_clean_copies() {
    let mut m = Manager::new(Config {
        host_limit_bytes: 2 * BYTES,
        ..Config::new(3 * BYTES, Policy::EvictLru)
    });
    // a has a clean copy and b is spilled: the host is full.
    let [a, _b, c, _d] = reloaded_spill(&mut m);
    assert_eq!(m.stats().host_bytes, 2 * BYTES);
    assert_eq!(
        m.write_back(&[c]).unwrap(),
        0,
        "no free block; nothing is given up"
    );
    let s = m.stats();
    assert_eq!(s.clean_copies_dropped, 0);
    assert_eq!(s.write_backs, 0);
    // a's clean copy is intact: evicting it still copies nothing.
    let before = s.bytes_to_host;
    m.ensure_resident(&[c]).unwrap();
    let _ = a;
    assert_eq!(m.stats().bytes_to_host, before);
}

#[test]
fn the_eager_arena_serves_a_spill_that_fits_in_its_free_bytes() {
    let mut m = Manager::new(Config {
        host_alloc: HostAlloc::Eager,
        host_limit_bytes: 4 * BYTES,
        ..Config::new(2 * BYTES, Policy::EvictLru)
    });
    let ids: Vec<_> = (0..6)
        .map(|v| m.register(sized(v as f32, 1), Hint::Evictable).unwrap())
        .collect();
    // The first four are spilled, in arena order. Freeing the first and the
    // third leaves two holes of one unit each.
    for &id in &[ids[0], ids[2], ids[4], ids[5]] {
        m.unregister::<f32>(id).unwrap();
    }
    assert_eq!(m.stats().host_bytes, 2 * BYTES);
    let big = m.register(sized(9.0, 2), Hint::Evictable).unwrap();
    // Admitting one more unit evicts big, which needs two units in one piece.
    let last = m.register(sized(7.0, 1), Hint::Evictable).unwrap();
    assert!(!m.is_resident(big));
    let s = m.stats();
    assert_eq!(s.host_bytes, 4 * BYTES);
    assert_eq!(s.pinned_bytes, 4 * BYTES);
    m.unregister::<f32>(last).unwrap();
    assert!(contents(&mut m, big).iter().all(|&v| v == 9.0));
    assert!(
        contents(&mut m, ids[1]).iter().all(|&v| v == 1.0),
        "moved blocks keep their contents"
    );
    assert!(contents(&mut m, ids[3]).iter().all(|&v| v == 3.0));
}

#[test]
fn lazy_blocks_of_one_size_do_not_strand_room_for_another() {
    let mut m = Manager::new(Config {
        host_limit_bytes: 4 * BYTES,
        ..Config::new(2 * BYTES, Policy::EvictLru)
    });
    // Two blocks of two units are locked, then left unused.
    let twos = [1.0, 2.0, 3.0].map(|v| m.register(sized(v, 2), Hint::Evictable).unwrap());
    assert_eq!(m.stats().pinned_bytes, 4 * BYTES);
    for id in twos {
        m.unregister::<f32>(id).unwrap();
    }
    let ones: Vec<_> = (0..5)
        .map(|v| {
            m.register(sized(10.0 + v as f32, 1), Hint::Evictable)
                .unwrap()
        })
        .collect();
    assert!(!m.is_resident(ones[0]) && !m.is_resident(ones[1]) && !m.is_resident(ones[2]));
    for &id in &[ones[1], ones[3], ones[4]] {
        m.unregister::<f32>(id).unwrap();
    }
    assert_eq!(m.stats().host_bytes, 2 * BYTES);
    let big = m.register(sized(9.0, 2), Hint::Evictable).unwrap();
    // Two units are held and two requested: that fits a limit of four.
    let last = m.register(sized(7.0, 1), Hint::Evictable).unwrap();
    assert!(!m.is_resident(big));
    assert!(m.stats().pinned_bytes <= 4 * BYTES);
    m.unregister::<f32>(last).unwrap();
    assert!(contents(&mut m, big).iter().all(|&v| v == 9.0));
    assert!(contents(&mut m, ones[0]).iter().all(|&v| v == 10.0));
    assert!(contents(&mut m, ones[2]).iter().all(|&v| v == 12.0));
}

#[test]
fn the_host_limit_holds_the_same_tensors_in_both_modes_whatever_their_size() {
    // 25 floats are 100 bytes, which is not a multiple of anything useful.
    const ODD: usize = 25;
    for host_alloc in [HostAlloc::Lazy, HostAlloc::Eager] {
        let mut m = Manager::new(Config {
            host_alloc,
            host_limit_bytes: 2 * ODD * size_of::<f32>(),
            ..Config::new(ODD * size_of::<f32>(), Policy::EvictLru)
        });
        let ids = [1.0f32, 2.0, 3.0].map(|v| {
            m.register(api::full(v, &[ODD]).sync().unwrap(), Hint::Evictable)
                .unwrap_or_else(|e| panic!("{host_alloc:?}: {e}"))
        });
        assert_eq!(m.stats().host_bytes, 2 * ODD * size_of::<f32>());
        // The resident one first: with the host full, nothing else can come
        // back until it has left.
        for (id, v) in [(ids[2], 3.0), (ids[0], 1.0), (ids[1], 2.0)] {
            let seen = contents(&mut m, id);
            assert!(
                seen.len() == ODD && seen.iter().all(|&x| x == v),
                "{host_alloc:?} tensor {v}"
            );
        }
    }
}

#[test]
fn an_eager_arena_of_no_bytes_fails_spills_recoverably() {
    let mut m = Manager::new(Config {
        host_alloc: HostAlloc::Eager,
        host_limit_bytes: 0,
        ..Config::new(2 * BYTES, Policy::EvictLru)
    });
    m.register(tensor(1.0), Hint::Evictable).unwrap();
    m.register(tensor(2.0), Hint::Evictable).unwrap();
    let err = m.register(tensor(3.0), Hint::Evictable).unwrap_err();
    assert!(matches!(err, TierError::HostBudgetExceeded { .. }), "{err}");
    assert!(err.is_recoverable());
}

#[test]
fn a_prefetch_that_cannot_make_enough_room_evicts_nothing() {
    let mut m = Manager::new(Config {
        prefetch_depth: 1,
        ..Config::new(3 * BYTES, Policy::EvictLookahead)
    });
    let s1 = m.register(sized(1.0, 1), Hint::Evictable).unwrap();
    let big = m.register(sized(9.0, 2), Hint::Evictable).unwrap();
    let s2 = m.register(sized(2.0, 1), Hint::Evictable).unwrap(); // spills s1
    let s3 = m.register(sized(3.0, 1), Hint::Evictable).unwrap(); // spills big
    m.ensure_resident(&[s1]).unwrap(); // s1 is back, with a clean copy
    assert!(!m.is_resident(big) && m.is_resident(s2) && m.is_resident(s3));
    let evictions = m.stats().evictions;

    // Fetching big needs two units. Only s1 is free to evict, and that is
    // one unit, so evicting it would cost a reload and gain nothing.
    m.expect_accesses(&[s3, big]).unwrap();
    m.ensure_resident(&[s3]).unwrap();
    let s = m.stats();
    assert_eq!(s.prefetches, 0);
    assert_eq!(
        s.evictions, evictions,
        "nothing was evicted for a prefetch that did not start"
    );
    assert!(m.is_resident(s1));
    assert_eq!(
        s.write_backs, 1,
        "s2 is being saved so that the room exists next time"
    );
}

#[test]
fn automatic_prefetch_does_not_cancel_a_reload_in_flight() {
    let mut m = Manager::new(Config {
        prefetch_depth: 1,
        ..Config::new(3 * BIG, Policy::EvictLookahead)
    });
    let [a, b, _c, d] = [7.0f32, 1.0, 2.0, 3.0].map(|v| {
        m.register(api::full(v, &[BIG_N]).sync().unwrap(), Hint::Evictable)
            .unwrap()
    });
    assert!(!m.is_resident(a));
    assert_eq!(m.prefetch(&[a]).unwrap(), 1); // spills b; a is on its way
    m.expect_accesses(&[d, b]).unwrap();
    // Fetching b would need a's or c's memory. c has no host copy, and a's
    // reload is still running: waiting for it is not worth a prefetch.
    m.ensure_resident(&[d]).unwrap();
    let s = m.stats();
    assert_eq!(s.prefetches_cancelled, 0);
    assert_eq!(s.prefetches, 1, "only the explicit one");
    assert!(contents(&mut m, a).iter().all(|&v| v == 7.0));
}

#[test]
fn write_back_returns_before_the_copy_has_finished() {
    // An arena, so that page-locking is in neither measurement.
    let mut m = Manager::new(Config {
        host_alloc: HostAlloc::Eager,
        host_limit_bytes: 4 * BIG,
        ..Config::new(2 * BIG, Policy::EvictLru)
    });
    let big = |v: f32| api::full(v, &[BIG_N]).sync().unwrap();
    let a = m.register(big(1.0), Hint::Evictable).unwrap();
    let b = m.register(big(2.0), Hint::Evictable).unwrap();
    // A synchronous eviction of a, for scale.
    let start = std::time::Instant::now();
    let c = m.register(big(3.0), Hint::Evictable).unwrap();
    let waited = start.elapsed();
    assert!(!m.is_resident(a));
    let start = std::time::Instant::now();
    assert_eq!(m.write_back(&[b]).unwrap(), 1);
    let submitted = start.elapsed();
    assert!(
        submitted * 4 < waited,
        "write_back took {submitted:?}; an eviction that waits for the same copy took {waited:?}"
    );
    let _ = c;
}

// ---- element types: one manager, one byte budget

/// Element `i` of a test tensor, for each element type.
trait Sample: DType + PartialEq {
    fn sample(i: usize) -> Self;
}

impl Sample for cutile::half::bf16 {
    fn sample(i: usize) -> Self {
        cutile::half::bf16::from_f32(i as f32 * 0.37)
    }
}

impl Sample for f32 {
    fn sample(i: usize) -> Self {
        i as f32 * 0.37
    }
}

impl Sample for f64 {
    fn sample(i: usize) -> Self {
        i as f64 * 0.37
    }
}

impl Sample for i64 {
    fn sample(i: usize) -> Self {
        i as i64 * 1_000_003 - 7
    }
}

impl Sample for u8 {
    fn sample(i: usize) -> Self {
        (i * 31 % 251) as u8
    }
}

/// A device tensor of `shape` holding `values`. Built from a host vector,
/// which works for every element type (`api::full` does not for all).
fn upload<T: DType>(values: Vec<T>, shape: &[usize]) -> Tensor<T> {
    api::copy_host_vec_to_device(&Arc::new(values))
        .sync()
        .unwrap()
        .reshape(shape)
        .unwrap()
}

/// A tensor of `shape` filled with the samples of `T`, and its values.
fn sampled<T: Sample>(shape: &[usize]) -> (Tensor<T>, Vec<T>) {
    let values: Vec<T> = (0..shape.iter().product()).map(T::sample).collect();
    (upload(values.clone(), shape), values)
}

fn read_back<T: DType>(m: &Manager, id: tiermem::TensorId) -> Vec<T> {
    m.tensor::<T>(id)
        .unwrap()
        .dup()
        .to_host_vec()
        .sync()
        .unwrap()
}

#[test]
fn mixed_dtypes_share_one_budget() {
    use cutile::half::bf16;
    let mut m = Manager::new(Config::new(8192, Policy::EvictLru));
    let f = m
        .register(upload(vec![1.0f32; 1024], &[1024]), Hint::Evictable)
        .unwrap(); // 4096 bytes
    let b = m
        .register(
            upload(vec![bf16::from_f32(1.0); 2048], &[2048]),
            Hint::Evictable,
        )
        .unwrap(); // 4096 bytes
    let u = m
        .register(upload(vec![1u8; 1024], &[1024]), Hint::Evictable)
        .unwrap(); // 1024 bytes
    assert!(!m.is_resident(f), "the least recently used goes");
    assert!(m.is_resident(b));
    assert!(m.is_resident(u));
    let s = m.stats();
    assert_eq!(s.evictions, 1);
    assert_eq!(s.resident_bytes, 5120);
}

fn spill_round_trip<T: Sample>() {
    let shape = [6, 7];
    let (t, values) = sampled::<T>(&shape);
    let bytes = t.num_bytes();
    let mut m = Manager::new(Config::new(bytes, Policy::EvictLru));
    let a = m.register(t, Hint::Evictable).unwrap();
    m.register(sampled::<T>(&shape).0, Hint::Evictable).unwrap(); // evicts a
    assert!(!m.is_resident(a));
    m.ensure_resident(&[a]).unwrap();
    assert_eq!(m.tensor::<T>(a).unwrap().shape(), &[6, 7]);
    assert_eq!(read_back::<T>(&m, a), values, "{:?}", T::DTYPE);
    let s = m.stats();
    assert_eq!(s.reloads, 1);
    assert_eq!(s.bytes_to_device, bytes as u64);
}

#[test]
fn spill_round_trip_per_dtype() {
    spill_round_trip::<cutile::half::bf16>();
    spill_round_trip::<f32>();
    spill_round_trip::<f64>();
    spill_round_trip::<i64>();
    spill_round_trip::<u8>();
}

fn expect_wrong_dtype<T>(result: Result<T, TierError>, id: tiermem::TensorId, what: &str) {
    match result.map(|_| ()) {
        Err(TierError::WrongDType {
            id: got,
            stored,
            requested,
        }) => {
            assert_eq!(got, id, "{what}");
            assert_eq!(stored, cutile::DTypeId::F32, "{what}");
            assert_eq!(requested, cutile::DTypeId::F64, "{what}");
        }
        other => panic!("{what}: expected WrongDType, got {other:?}"),
    }
}

#[test]
fn wrong_dtype_is_reported() {
    let mut m = Manager::new(Config::new(2 * BYTES, Policy::EvictLru));
    let a = m.register(tensor(1.0), Hint::Evictable).unwrap();
    let b = m.register(tensor(2.0), Hint::Evictable).unwrap();
    let c = m.register(tensor(3.0), Hint::Evictable).unwrap(); // evicts a
    m.ensure_resident(&[a]).unwrap(); // reloads a, which keeps a clean host copy
    let _ = b;
    assert_eq!(m.stats().clean_evictions, 0);
    let host_bytes = m.stats().host_bytes;

    expect_wrong_dtype(m.tensor::<f64>(a), a, "tensor");
    expect_wrong_dtype(m.tensor_mut::<f64>(a), a, "tensor_mut");
    expect_wrong_dtype(m.unregister::<f64>(a), a, "unregister");
    assert!(m.is_resident(a));
    assert!(m.tensor::<f32>(a).is_ok());
    assert_eq!(m.stats().host_bytes, host_bytes, "the clean copy was kept");

    // Make `a` the least recently used and push it out: its clean copy must
    // have survived the failed `tensor_mut`, so the eviction copies nothing.
    m.ensure_resident(&[c]).unwrap();
    m.register(tensor(4.0), Hint::Evictable).unwrap();
    assert!(!m.is_resident(a));
    assert_eq!(m.stats().clean_evictions, 1);

    // Spilled: the type is still checked first, and the tensor stays.
    expect_wrong_dtype(m.tensor::<f64>(a), a, "tensor, spilled");
    expect_wrong_dtype(m.unregister::<f64>(a), a, "unregister, spilled");
    assert_eq!(m.dtype(a).unwrap(), cutile::DTypeId::F32);
    assert!(matches!(m.tensor::<f32>(a), Err(TierError::NotResident(_))));
}

#[test]
fn dtype_and_shape_are_reported() {
    use cutile::DTypeId;
    use cutile::half::bf16;
    let shape = [6, 7];
    let (t, _) = sampled::<bf16>(&shape);
    let mut m = Manager::new(Config::new(t.num_bytes(), Policy::EvictLru));
    let a = m.register(t, Hint::Evictable).unwrap();
    assert_eq!(m.dtype(a).unwrap(), DTypeId::BF16);
    assert_eq!(m.shape(a).unwrap(), &[6, 7]);
    m.register(sampled::<bf16>(&shape).0, Hint::Evictable)
        .unwrap();
    assert!(!m.is_resident(a));
    assert_eq!(m.dtype(a).unwrap(), DTypeId::BF16, "spilled");
    assert_eq!(m.shape(a).unwrap(), &[6, 7], "spilled");
    m.unregister::<bf16>(a).unwrap();
    assert!(matches!(m.dtype(a), Err(TierError::UnknownTensor(_))));
    assert!(matches!(m.shape(a), Err(TierError::UnknownTensor(_))));
}

#[test]
fn snapshot_reports_the_dtype() {
    use cutile::DTypeId;
    use cutile::half::bf16;
    let mut m = Manager::new(Config::new(64, Policy::Strict));
    m.register(
        upload(vec![bf16::from_f32(1.0); 16], &[16]),
        Hint::Evictable,
    )
    .unwrap(); // 32 bytes
    m.register(upload(vec![1.0f32; 8], &[8]), Hint::Evictable)
        .unwrap(); // 32 bytes
    let err = m
        .register(upload(vec![1u8; 1], &[1]), Hint::Evictable)
        .unwrap_err();
    let TierError::BudgetExceeded { snapshot, .. } = &err else {
        panic!("expected BudgetExceeded, got {err}");
    };
    let dtypes: Vec<DTypeId> = snapshot.entries.iter().map(|e| e.dtype).collect();
    assert_eq!(dtypes, [DTypeId::BF16, DTypeId::F32]);
    let text = err.to_string();
    assert!(text.contains("bf16") && text.contains("f32"), "{text}");
}

#[test]
fn reconstructible_bf16_is_rebuilt() {
    use cutile::half::bf16;
    let shape = [6usize, 7];
    let (t, values) = sampled::<bf16>(&shape);
    let mut m = Manager::new(Config::new(t.num_bytes(), Policy::EvictLru));
    let rebuild = |values: Vec<bf16>| move || Ok(upload(values.clone(), &[6, 7]));
    let a = m
        .register_reconstructible(t, Hint::Evictable, rebuild(values.clone()))
        .unwrap();
    let (other, other_values) = sampled::<bf16>(&shape);
    m.register_reconstructible(other, Hint::Evictable, rebuild(other_values))
        .unwrap(); // evicts a, which is dropped
    assert!(!m.is_resident(a));
    m.ensure_resident(&[a]).unwrap();
    let s = m.stats();
    assert_eq!(s.reconstructions, 1);
    assert_eq!(s.bytes_to_host, 0);
    assert_eq!(s.host_bytes, 0);
    assert_eq!(read_back::<bf16>(&m, a), values);
}

#[test]
fn element_size_drives_eviction_count() {
    use cutile::half::bf16;
    let n = 1024;
    let mut m = Manager::new(Config::new(8 * n, Policy::EvictLru));
    for _ in 0..4 {
        m.register(upload(vec![bf16::from_f32(1.0); n], &[n]), Hint::Evictable)
            .unwrap(); // 2 * n bytes each
    }
    assert_eq!(m.stats().evictions, 0);
    m.register(upload(vec![1.0f64; n], &[n]), Hint::Evictable)
        .unwrap(); // 8 * n bytes
    let s = m.stats();
    assert_eq!(s.evictions, 4);
    assert_eq!(s.resident_bytes, 8 * n);
}

#[test]
fn wrong_dtype_changes_nothing_while_a_prefetch_is_in_flight() {
    let (mut m, [a, ..]) = big_manager();
    assert_eq!(m.prefetch(&[a]).unwrap(), 1);
    let before = m.stats();
    assert_eq!(before.reloads, 0, "a's reload is still in flight");

    // The type is checked first, so these are WrongDType, not NotResident
    // (a loading tensor is not resident), and nothing waits for the copy.
    expect_wrong_dtype(m.tensor::<f64>(a), a, "tensor");
    expect_wrong_dtype(m.tensor_mut::<f64>(a), a, "tensor_mut");
    expect_wrong_dtype(m.unregister::<f64>(a), a, "unregister");

    let after = m.stats();
    assert_eq!(after.reloads, 0, "nothing waited for the reload");
    assert_eq!(after.prefetches, before.prefetches);
    assert_eq!(after.resident_bytes, before.resident_bytes);
    assert_eq!(after.spilled_bytes, before.spilled_bytes);
    assert_eq!(after.host_bytes, before.host_bytes);
    assert_eq!(
        m.dtype(a).unwrap(),
        cutile::DTypeId::F32,
        "still registered"
    );

    // The reload goes on and the tensor is intact.
    assert!(contents(&mut m, a).iter().all(|&v| v == 7.0));
}

#[test]
fn wrong_dtype_changes_nothing_while_a_write_back_is_in_flight() {
    let mut m = Manager::new(Config::new(2 * BIG, Policy::EvictLru));
    let a = m
        .register(api::full(7.0f32, &[BIG_N]).sync().unwrap(), Hint::Evictable)
        .unwrap();
    assert_eq!(m.write_back(&[a]).unwrap(), 1);
    let before = m.stats();
    assert_eq!(before.write_backs, 1);
    assert_eq!(before.host_bytes, BIG, "the copy's host block is counted");

    expect_wrong_dtype(m.tensor_mut::<f64>(a), a, "tensor_mut");
    expect_wrong_dtype(m.unregister::<f64>(a), a, "unregister");

    // A late check would wait for the copy and drop its host block, or
    // remove the tensor.
    let after = m.stats();
    assert_eq!(
        after.host_bytes, before.host_bytes,
        "the host block is kept"
    );
    assert_eq!(after.write_backs, before.write_backs);
    assert!(m.is_resident(a));
    assert_eq!(
        m.dtype(a).unwrap(),
        cutile::DTypeId::F32,
        "still registered"
    );

    // Asking for the right type is what drops the clean copy.
    m.tensor_mut::<f32>(a).unwrap();
    assert_eq!(m.stats().host_bytes, 0);
}

#[test]
fn wrong_dtype_has_a_readable_message_and_is_not_recoverable() {
    let mut m = Manager::new(Config::new(2 * BYTES, Policy::EvictLru));
    let a = m.register(tensor(1.0), Hint::Evictable).unwrap();
    let err = m.tensor::<f64>(a).map(|_| ()).expect_err("wrong type");
    assert_eq!(err.to_string(), "tensor TensorId(0) holds f32, not f64");
    assert!(!err.is_recoverable());
    let err = m
        .register(upload(vec![1u8; 4], &[4]), Hint::Evictable)
        .and_then(|u8_id| m.tensor::<cutile::half::bf16>(u8_id).map(|_| ()))
        .unwrap_err();
    assert_eq!(err.to_string(), "tensor TensorId(1) holds u8, not bf16");
}
