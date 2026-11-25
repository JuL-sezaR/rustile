//! Wall-clock assertions about the manager, kept apart from `manager.rs`.
//!
//! Cargo runs each test binary in turn, so these run with no other test
//! using the GPU's copy engine. Inside `manager.rs` the tests run in
//! parallel and the timing of this one was not reliable.

use cutile::api;
use cutile::prelude::*;
use tiermem::{Config, Hint, HostAlloc, Manager, Policy};

const BIG_N: usize = 8 * 1024 * 1024;
const BIG: usize = BIG_N * size_of::<f32>();

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
