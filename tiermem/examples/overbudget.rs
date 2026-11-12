//! Streams a stack of "layer weights" whose total size exceeds a VRAM budget
//! through a kernel, the way inference walks a model's layers.
//!
//!   cargo run --release -p tiermem --example overbudget -- [MODE] [--lookahead]
//!       [--prefetch N] [--host-limit-mib N] [--host-alloc lazy|eager]
//!       [--copy-streams N] [--work N] [--dirty] [--budget-mib N]
//!       [--layers N] [--layer-mib N] [--epochs N] [--runs N] [--trace PATH]
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
//! --copy-streams 1 runs copies in both directions on one stream instead of
//! one each (2, the default).
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
//! --runs N (default 5) repeats the whole workload N times after one warmup
//! run, each with a fresh manager, and reports the median, minimum and
//! maximum of each time. The counters and the trace are those of the last
//! run.
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
use tiermem::{Config, Hint, HostAlloc, Manager, Policy, Stats, TensorId};

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
    copy_streams: usize,
    work: usize,
    dirty: bool,
    budget_mib: usize,
    layers: usize,
    layer_mib: usize,
    epochs: usize,
    runs: usize,
    trace: Option<String>,
}

fn parse_args(mut it: impl Iterator<Item = String>) -> Args {
    let mut args = Args {
        mode: Mode::Spill,
        lookahead: false,
        prefetch: 0,
        host_limit_mib: None,
        host_alloc: HostAlloc::Lazy,
        copy_streams: 2,
        work: 1,
        dirty: false,
        budget_mib: 256,
        layers: 12,
        layer_mib: 64,
        epochs: 3,
        runs: 5,
        trace: None,
    };
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
            "--copy-streams" => args.copy_streams = num("--copy-streams"),
            "--work" => args.work = num("--work"),
            "--dirty" => args.dirty = true,
            "--budget-mib" => args.budget_mib = num("--budget-mib"),
            "--layers" => args.layers = num("--layers"),
            "--layer-mib" => args.layer_mib = num("--layer-mib"),
            "--epochs" => args.epochs = num("--epochs"),
            "--runs" => args.runs = num("--runs"),
            "--trace" => args.trace = Some(it.next().expect("--trace needs a path")),
            other => panic!("unknown argument {other:?}"),
        }
    }
    assert!(
        args.prefetch == 0 || args.lookahead,
        "--prefetch needs --lookahead"
    );
    assert!(args.runs >= 1, "--runs needs at least 1");
    if args.host_alloc == HostAlloc::Eager && args.host_limit_mib.is_none() {
        args.host_limit_mib = Some(args.layers * args.layer_mib);
    }
    args
}

/// The median, minimum and maximum of a set of timings.
#[derive(Debug, PartialEq)]
struct Summary {
    median: Duration,
    min: Duration,
    max: Duration,
}

impl Summary {
    fn of(times: &[Duration]) -> Self {
        let mut sorted = times.to_vec();
        sorted.sort();
        let mid = sorted.len() / 2;
        let median = if sorted.len().is_multiple_of(2) {
            (sorted[mid - 1] + sorted[mid]) / 2
        } else {
            sorted[mid]
        };
        Self {
            median,
            min: sorted[0],
            max: sorted[sorted.len() - 1],
        }
    }
}

impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.3} s  (min {:.3}, max {:.3})",
            self.median.as_secs_f64(),
            self.min.as_secs_f64(),
            self.max.as_secs_f64()
        )
    }
}

#[derive(Default)]
struct Times {
    total: Duration,
    /// Creating and registering the layers.
    setup: Duration,
    /// The epochs, without the final check.
    run: Duration,
}

/// What one pass over the workload produced.
struct Outcome {
    times: Times,
    stats: Stats,
    pressure_events: u32,
    /// Set when a layer could not be admitted; the workload did not run.
    failure: Option<String>,
    /// Elements of the result that differ from `expected`.
    errors: usize,
    elements: usize,
    expected: f32,
}

/// Runs the whole workload once with a fresh manager: registers the layers,
/// sweeps them for the configured epochs and checks the result.
fn run_once(args: &Args, trace: Option<&str>) -> Result<Outcome, Box<dyn std::error::Error>> {
    let len = args.layer_mib * MIB / size_of::<f32>();
    let policy = match args.mode {
        Mode::Strict => Policy::Strict,
        Mode::Spill | Mode::Rebuild if args.lookahead => Policy::EvictLookahead,
        Mode::Spill | Mode::Rebuild => Policy::EvictLru,
    };
    let mut mgr = Manager::<f32>::new(Config {
        prefetch_depth: args.prefetch,
        host_limit_bytes: args.host_limit_mib.map_or(usize::MAX, |mib| mib * MIB),
        host_alloc: args.host_alloc,
        copy_streams: args.copy_streams,
        ..Config::new(args.budget_mib * MIB, policy)
    });
    let pressure_events = Rc::new(Cell::new(0u32));
    let counter = pressure_events.clone();
    mgr.on_pressure(move |_| counter.set(counter.get() + 1));
    let n = args.layers as f32;
    let expected = (args.epochs * args.work) as f32 * n * (n + 1.0) / 2.0;

    let start = Instant::now();
    let mut failure = None;
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
                failure = Some(format!(
                    "FAILED admitting layer {i} (recoverable: {})\n{e}",
                    e.is_recoverable()
                ));
                break;
            }
        }
    }

    let mut times = Times::default();
    let (mut errors, mut elements) = (0, 0);
    if failure.is_none() {
        let mut acc = api::zeros::<f32>(&[len]).sync()?;
        let mut out = api::zeros::<f32>(&[len]).sync()?;
        if args.lookahead {
            for _ in 0..args.epochs {
                mgr.expect_accesses(&layers)?;
            }
        }
        times.setup = start.elapsed();
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
        times.run = start.elapsed() - times.setup;

        let host = acc.to_host_vec().sync()?;
        elements = host.len();
        errors = host
            .iter()
            .filter(|&&v| (v - expected).abs() > 1e-3)
            .count();
    }
    times.total = start.elapsed();
    if let Some(path) = trace {
        mgr.trace().write_chrome_trace(path)?;
    }
    Ok(Outcome {
        times,
        stats: mgr.stats(),
        pressure_events: pressure_events.get(),
        failure,
        errors,
        elements,
        expected,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args(std::env::args().skip(1));
    assert!(
        (args.layer_mib * MIB / size_of::<f32>()).is_multiple_of(TILE),
        "layer size must be a multiple of {TILE} floats"
    );

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
    println!(
        "async engines   {} (copies the device can run at once); {} copy stream(s)",
        tiermem::async_engine_count()?,
        args.copy_streams
    );

    // Warm the JIT so compile time does not land in the measurement.
    {
        let mut out = api::zeros::<f32>(&[TILE]).sync()?;
        let x = api::zeros::<f32>(&[TILE]).sync()?;
        kernel::add((&mut out).partition([TILE]), &x, &x).sync()?;
    }

    // The warmup run is not reported, unless it shows that the workload
    // cannot run at all, in which case repeating it would tell nothing more.
    let warmup = run_once(&args, None)?;
    if let Some(failure) = &warmup.failure {
        println!("{failure}");
        println!("time            {:.3} s", warmup.times.total.as_secs_f64());
        print_counters(&args, &warmup);
        std::process::exit(1);
    }

    let mut outcomes = Vec::with_capacity(args.runs);
    for run in 1..=args.runs {
        let trace = args.trace.as_deref().filter(|_| run == args.runs);
        outcomes.push(run_once(&args, trace)?);
    }
    let last = outcomes.last().unwrap();
    let wrong = outcomes.iter().filter(|o| o.errors > 0).count();
    if wrong == 0 {
        println!(
            "PASSED: all {} elements == {} in each of {} run(s), after 1 warmup run",
            last.elements, last.expected, args.runs
        );
    } else {
        println!(
            "WRONG: {wrong} of {} run(s) had elements != {}",
            args.runs, last.expected
        );
    }

    let summary = |time: fn(&Outcome) -> Duration| {
        Summary::of(&outcomes.iter().map(time).collect::<Vec<_>>())
    };
    println!("                median");
    println!("time            {}", summary(|o| o.times.total));
    println!("  setup         {}", summary(|o| o.times.setup));
    println!("  run           {}", summary(|o| o.times.run));
    println!(
        "page-locking    {}  for {} MiB ({})",
        summary(|o| o.stats.pin_time),
        last.stats.pinned_bytes / MIB,
        match args.host_alloc {
            HostAlloc::Lazy => "as first spilled",
            HostAlloc::Eager => "one arena, up front",
        }
    );
    println!("counters of the last run:");
    print_counters(&args, last);
    if wrong > 0 {
        std::process::exit(1);
    }
    Ok(())
}

fn print_counters(args: &Args, outcome: &Outcome) {
    let s = &outcome.stats;
    let moved_mib = (s.bytes_to_host + s.bytes_to_device) as f64 / MIB as f64;
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
    println!(
        "write-backs     {} (started ahead of eviction)",
        s.write_backs
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
    println!("pressure events {}", outcome.pressure_events);
    if let Some(path) = &args.trace {
        println!("trace           {path} (open in ui.perfetto.dev)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Args {
        parse_args(list.iter().map(|s| s.to_string()))
    }

    fn ms(values: &[u64]) -> Vec<Duration> {
        values.iter().map(|&v| Duration::from_millis(v)).collect()
    }

    #[test]
    fn five_runs_by_default() {
        assert_eq!(args(&[]).runs, 5);
    }

    #[test]
    fn runs_is_read_from_the_command_line() {
        assert_eq!(args(&["spill", "--runs", "3"]).runs, 3);
    }

    #[test]
    #[should_panic(expected = "--runs needs at least 1")]
    fn zero_runs_is_rejected() {
        args(&["--runs", "0"]);
    }

    #[test]
    fn the_flags_from_before_runs_still_parse() {
        let a = args(&[
            "spill",
            "--lookahead",
            "--prefetch",
            "2",
            "--host-limit-mib",
            "576",
            "--host-alloc",
            "eager",
            "--copy-streams",
            "1",
            "--work",
            "8",
            "--dirty",
            "--budget-mib",
            "128",
            "--layers",
            "6",
            "--layer-mib",
            "32",
            "--epochs",
            "30",
            "--trace",
            "out.json",
        ]);
        assert!(a.mode == Mode::Spill && a.lookahead && a.dirty);
        assert_eq!(
            (a.prefetch, a.host_limit_mib, a.copy_streams),
            (2, Some(576), 1)
        );
        assert_eq!(
            (a.work, a.budget_mib, a.layers, a.layer_mib),
            (8, 128, 6, 32)
        );
        assert_eq!((a.epochs, a.trace.as_deref()), (30, Some("out.json")));
        assert!(a.host_alloc == HostAlloc::Eager);
    }

    #[test]
    fn the_summary_of_an_odd_number_of_runs_takes_the_middle_one() {
        let summary = Summary::of(&ms(&[40, 10, 30, 20, 500]));
        assert_eq!(summary.median, Duration::from_millis(30));
        assert_eq!(summary.min, Duration::from_millis(10));
        assert_eq!(summary.max, Duration::from_millis(500));
    }

    #[test]
    fn the_summary_of_an_even_number_of_runs_averages_the_middle_two() {
        let summary = Summary::of(&ms(&[40, 10, 30, 20]));
        assert_eq!(summary.median, Duration::from_millis(25));
    }

    #[test]
    fn the_summary_of_one_run_is_that_run() {
        let one = Duration::from_millis(7);
        let expected = Summary {
            median: one,
            min: one,
            max: one,
        };
        assert_eq!(Summary::of(&[one]), expected);
    }
}
