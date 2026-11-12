//! Streams a stack of "layer weights" whose total size exceeds a VRAM budget
//! through a kernel, the way inference walks a model's layers.
//!
//!   cargo run --release -p tiermem --example overbudget -- [MODE] [--lookahead]
//!       [--prefetch N] [--host-limit-mib N] [--host-alloc lazy|eager]
//!       [--work N] [--dirty] [--budget-mib N] [--layers N] [--layer-mib N]
//!       [--epochs N] [--trace PATH]
//!
//! MODE is one of:
//!   strict       no eviction: behaves like a GPU with only --budget-mib of memory
//!   spill        evict LRU, spilling contents to host RAM and copying them back
//!   rebuild      evict LRU, dropping contents and regenerating them on the device
//!
//! --lookahead declares the layer order up front and evicts the layer needed
//! farthest in the future instead of the least recently used one.
//! --prefetch N also starts reloading the next N declared layers while the
//! current one is being computed on (needs --lookahead).
//! --host-limit-mib N caps the host RAM held for spilled layers.
//! --host-alloc eager page-locks that much host RAM once, before the layers
//! are registered, instead of a block per layer as each is first spilled
//! (lazy, the default). Without --host-limit-mib the arena is as large as the
//! working set.
//! --work N runs the kernel N times per layer, standing in for a heavier
//! layer; this is the compute a prefetch can overlap with.
//!
//! --dirty marks each layer as written after it is used, the way a training
//! step updates its weights, so every eviction has to copy it back to the
//! host. The contents are not actually changed, which keeps the final check
//! the same.
//!
//! The two accumulator buffers are not managed and sit outside the budget.
//!
//! The reported time is split into setup (creating and registering the
//! layers, which already spills the ones that do not fit), run (the epochs)
//! and the final check of the result.

use cutile::api;
use cutile::prelude::*;
use cutile::tensor::PartitionMut;
use std::cell::Cell;
use std::rc::Rc;
use std::time::{Duration, Instant};
use tiermem::{Config, Hint, HostAlloc, Manager, Policy, TensorId};

#[cutile::module]
mod kernel {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like(x, z);
        let ty = load_tile_like(y, z);
        z.store(tx + ty);
    }
}

const TILE: usize = 1024;
const MIB: usize = 1024 * 1024;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Strict,
    Spill,
    Rebuild,
}

struct Args {
    mode: Mode,
    lookahead: bool,
    prefetch: usize,
    host_limit_mib: Option<usize>,
    host_alloc: HostAlloc,
    work: usize,
    dirty: bool,
    budget_mib: usize,
    layers: usize,
    layer_mib: usize,
    epochs: usize,
    trace: Option<String>,
}

fn parse_args() -> Args {
    let mut args = Args {
        mode: Mode::Spill,
        lookahead: false,
        prefetch: 0,
        host_limit_mib: None,
        host_alloc: HostAlloc::Lazy,
        work: 1,
        dirty: false,
        budget_mib: 256,
        layers: 12,
        layer_mib: 64,
        epochs: 3,
        trace: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut num = |name: &str| -> usize {
            it.next()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("{name} needs a number"))
        };
        match arg.as_str() {
            "strict" => args.mode = Mode::Strict,
            "spill" => args.mode = Mode::Spill,
            "rebuild" => args.mode = Mode::Rebuild,
            "--lookahead" => args.lookahead = true,
            "--prefetch" => args.prefetch = num("--prefetch"),
            "--host-limit-mib" => args.host_limit_mib = Some(num("--host-limit-mib")),
            "--host-alloc" => {
                args.host_alloc = match it.next().as_deref() {
                    Some("lazy") => HostAlloc::Lazy,
                    Some("eager") => HostAlloc::Eager,
                    other => panic!("--host-alloc needs lazy or eager, got {other:?}"),
                }
            }
            "--work" => args.work = num("--work"),
            "--dirty" => args.dirty = true,
            "--budget-mib" => args.budget_mib = num("--budget-mib"),
            "--layers" => args.layers = num("--layers"),
            "--layer-mib" => args.layer_mib = num("--layer-mib"),
            "--epochs" => args.epochs = num("--epochs"),
            "--trace" => args.trace = Some(it.next().expect("--trace needs a path")),
            other => panic!("unknown argument {other:?}"),
        }
    }
    assert!(
        args.prefetch == 0 || args.lookahead,
        "--prefetch needs --lookahead"
    );
    if args.host_alloc == HostAlloc::Eager && args.host_limit_mib.is_none() {
        args.host_limit_mib = Some(args.layers * args.layer_mib);
    }
    args
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let len = args.layer_mib * MIB / size_of::<f32>();
    assert!(
        len.is_multiple_of(TILE),
        "layer size must be a multiple of {TILE} floats"
    );

    let policy = match args.mode {
        Mode::Strict => Policy::Strict,
        Mode::Spill | Mode::Rebuild if args.lookahead => Policy::EvictLookahead,
        Mode::Spill | Mode::Rebuild => Policy::EvictLru,
    };
    let mut mgr = Manager::<f32>::new(Config {
        prefetch_depth: args.prefetch,
        host_limit_bytes: args.host_limit_mib.map_or(usize::MAX, |mib| mib * MIB),
        host_alloc: args.host_alloc,
        ..Config::new(args.budget_mib * MIB, policy)
    });
    let pressure_events = Rc::new(Cell::new(0u32));
    let counter = pressure_events.clone();
    mgr.on_pressure(move |_| counter.set(counter.get() + 1));

    println!(
        "mode={}{}{}{}{} budget={} MiB working set={} MiB ({} x {} MiB) epochs={}",
        match args.mode {
            Mode::Strict => "strict",
            Mode::Spill => "spill",
            Mode::Rebuild => "rebuild",
        },
        if args.lookahead { " lookahead" } else { "" },
        if args.prefetch > 0 {
            format!(" prefetch={}", args.prefetch)
        } else {
            String::new()
        },
        match args.host_alloc {
            HostAlloc::Lazy => "",
            HostAlloc::Eager => " host-alloc=eager",
        },
        if args.dirty { " dirty" } else { "" },
        args.budget_mib,
        args.layers * args.layer_mib,
        args.layers,
        args.layer_mib,
        args.epochs,
    );

    // Warm the JIT so compile time does not land in the measurement.
    {
        let mut out = api::zeros::<f32>(&[TILE]).sync()?;
        let x = api::zeros::<f32>(&[TILE]).sync()?;
        kernel::add((&mut out).partition([TILE]), &x, &x).sync()?;
    }

    let start = Instant::now();
    let mut layers: Vec<TensorId> = Vec::with_capacity(args.layers);
    for i in 0..args.layers {
        let value = (i + 1) as f32;
        let w = api::full(value, &[len]).sync()?;
        let admitted = match args.mode {
            Mode::Rebuild => mgr.register_reconstructible(w, Hint::Evictable, move || {
                Ok(api::full(value, &[len]).sync()?)
            }),
            _ => mgr.register(w, Hint::Evictable),
        };
        match admitted {
            Ok(id) => layers.push(id),
            Err(e) => {
                println!(
                    "FAILED admitting layer {i} (recoverable: {})",
                    e.is_recoverable()
                );
                println!("{e}");
                let times = Times {
                    total: start.elapsed(),
                    ..Times::default()
                };
                return finish(&mgr, &args, times, pressure_events.get(), false);
            }
        }
    }

    let mut acc = api::zeros::<f32>(&[len]).sync()?;
    let mut out = api::zeros::<f32>(&[len]).sync()?;
    if args.lookahead {
        for _ in 0..args.epochs {
            mgr.expect_accesses(&layers)?;
        }
    }
    let setup = start.elapsed();
    for _ in 0..args.epochs {
        for &id in &layers {
            mgr.ensure_resident(&[id])?;
            for _ in 0..args.work {
                kernel::add((&mut out).partition([TILE]), &acc, mgr.tensor(id)?).sync()?;
                std::mem::swap(&mut acc, &mut out);
            }
            if args.dirty {
                mgr.tensor_mut(id)?;
            }
        }
    }

    // Every kernel was waited for, so the epochs are finished here.
    let run = start.elapsed() - setup;

    let n = args.layers as f32;
    let expected = (args.epochs * args.work) as f32 * n * (n + 1.0) / 2.0;
    let host = acc.to_host_vec().sync()?;
    let errors = host
        .iter()
        .filter(|&&v| (v - expected).abs() > 1e-3)
        .count();
    if errors == 0 {
        println!("PASSED: all {} elements == {expected}", host.len());
    } else {
        println!("WRONG: {errors} of {} elements != {expected}", host.len());
    }
    let times = Times {
        total: start.elapsed(),
        setup,
        run,
    };
    finish(&mgr, &args, times, pressure_events.get(), errors == 0)
}

#[derive(Default)]
struct Times {
    total: Duration,
    /// Creating and registering the layers.
    setup: Duration,
    /// The epochs, without the final check.
    run: Duration,
}

fn finish(
    mgr: &Manager<f32>,
    args: &Args,
    times: Times,
    pressure_events: u32,
    ok: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let s = mgr.stats();
    let moved_mib = (s.bytes_to_host + s.bytes_to_device) as f64 / MIB as f64;
    println!("time            {:.3} s", times.total.as_secs_f64());
    println!("  setup         {:.3} s", times.setup.as_secs_f64());
    println!("  run           {:.3} s", times.run.as_secs_f64());
    println!(
        "page-locking    {:.3} s for {} MiB ({})",
        s.pin_time.as_secs_f64(),
        s.pinned_bytes / MIB,
        match args.host_alloc {
            HostAlloc::Lazy => "as first spilled",
            HostAlloc::Eager => "one arena, up front",
        }
    );
    println!("peak resident   {} MiB", s.peak_resident_bytes / MIB);
    println!("evictions       {}", s.evictions);
    println!(
        "clean evictions {} (no copy back to host)",
        s.clean_evictions
    );
    println!("reloads         {}", s.reloads);
    println!(
        "prefetches      {} ({} cancelled)",
        s.prefetches, s.prefetches_cancelled
    );
    println!("reconstructions {}", s.reconstructions);
    println!(
        "moved           {moved_mib:.0} MiB (to host {} + to device {})",
        s.bytes_to_host / MIB as u64,
        s.bytes_to_device / MIB as u64
    );
    println!(
        "host RAM held   {} MiB ({} clean copies dropped)",
        s.host_bytes / MIB,
        s.clean_copies_dropped
    );
    println!("pressure events {pressure_events}");
    if let Some(path) = &args.trace {
        mgr.trace().write_chrome_trace(path)?;
        println!("trace           {path} (open in ui.perfetto.dev)");
    }
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
