//! Benchmark of the tile kernels against a single-threaded CPU loop.
//!
//! ```sh
//! cargo run --release -p tilekernels --example kernelbench -- softmax [--runs N] [--reps R]
//! ```
//!
//! For each case: the first call is timed on its own (it includes the JIT
//! compile) and is the warmup. Then `--runs` measurements are taken, each
//! the wall time of `--reps` consecutive calls divided by `--reps`. Every
//! call waits for its kernel. The CPU reference gets one unreported warmup
//! call and `--runs` timed calls. Medians are reported with min and max;
//! nothing else is excluded.

use cutile::DTypeId;
use cutile::api;
use cutile::half::bf16;
use cutile::prelude::*;
use std::hint::black_box;
use std::time::Instant;
use tilekernels::{Float, KernelError, reference, softmax_into, tiles};

const DEFAULT_RUNS: usize = 5;
const DEFAULT_REPS: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kernel {
    Softmax,
}

#[derive(Debug, PartialEq)]
struct Args {
    kernel: Kernel,
    runs: usize,
    reps: usize,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut kernel = None;
    let mut runs = DEFAULT_RUNS;
    let mut reps = DEFAULT_REPS;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut number = |flag: &str| -> Result<usize, String> {
            let value = it.next().ok_or(format!("{flag} needs a number"))?;
            match value.parse::<usize>() {
                Ok(n) if n >= 1 => Ok(n),
                _ => Err(format!("{flag} needs a number of at least 1, got {value}")),
            }
        };
        match arg.as_str() {
            "--runs" => runs = number("--runs")?,
            "--reps" => reps = number("--reps")?,
            "softmax" => kernel = Some(Kernel::Softmax),
            other if other.starts_with("--") => return Err(format!("unknown option {other}")),
            other => return Err(format!("unknown kernel {other}; available: softmax")),
        }
    }
    let kernel =
        kernel.ok_or("usage: kernelbench <kernel> [--runs N] [--reps R]; kernels: softmax")?;
    Ok(Args { kernel, runs, reps })
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Stats {
    median: f64,
    min: f64,
    max: f64,
}

impl Stats {
    fn of(samples: &[f64]) -> Stats {
        assert!(!samples.is_empty());
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        let median = if n % 2 == 1 {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        };
        Stats {
            median,
            min: sorted[0],
            max: sorted[n - 1],
        }
    }
}

/// `runs` measurements, each the time of `reps` calls divided by `reps`.
fn time_gpu(
    runs: usize,
    reps: usize,
    mut call: impl FnMut() -> Result<(), KernelError>,
) -> Result<Stats, KernelError> {
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        for _ in 0..reps {
            call()?;
        }
        samples.push(start.elapsed().as_secs_f64() / reps as f64);
    }
    Ok(Stats::of(&samples))
}

/// One unreported warmup call, then `runs` timed calls.
fn time_cpu<R>(runs: usize, mut call: impl FnMut() -> R) -> Stats {
    black_box(call());
    let samples: Vec<f64> = (0..runs)
        .map(|_| {
            let start = Instant::now();
            black_box(call());
            start.elapsed().as_secs_f64()
        })
        .collect();
    Stats::of(&samples)
}

fn format_line(label: &str, tiles: &str, first: f64, gpu: Stats, cpu: Stats) -> String {
    let ms = |s: f64| s * 1e3;
    format!(
        "{label} tiles {tiles}: first {first:.2} s | gpu {:.3} ms (min {:.3}, max {:.3}) | cpu {:.1} ms (min {:.1}, max {:.1})",
        ms(gpu.median),
        ms(gpu.min),
        ms(gpu.max),
        ms(cpu.median),
        ms(cpu.min),
        ms(cpu.max),
    )
}

/// Deterministic values in [-1, 1): xorshift64, the same generator as the tests.
fn rnd(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
        })
        .collect()
}

trait Num: Copy {
    fn to_f64(self) -> f64;
    fn from_f32(v: f32) -> Self;
}

impl Num for bf16 {
    fn to_f64(self) -> f64 {
        bf16::to_f64(self)
    }
    fn from_f32(v: f32) -> Self {
        bf16::from_f32(v)
    }
}

impl Num for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f32(v: f32) -> Self {
        v
    }
}

impl Num for f64 {
    fn to_f64(self) -> f64 {
        self
    }
    fn from_f32(v: f32) -> Self {
        v as f64
    }
}

/// The softmax test tolerance for this output type: (per element, per row sum).
fn softmax_tolerance<E: Float>() -> (f64, f64) {
    if <E::Out as DType>::DTYPE == DTypeId::F64 {
        (1e-14, 1e-13)
    } else {
        (1e-6, 1e-5)
    }
}

/// One softmax case. Returns whether the last GPU result matched the reference.
fn softmax_case<E>(name: &str, rows: usize, cols: usize, args: &Args) -> Result<bool, KernelError>
where
    E: Float + Num,
    E::Out: Num,
{
    let (bm, bn) = tiles::row_tiles(rows, cols)?;
    let xs: Vec<E> = rnd(rows * cols, 7)
        .iter()
        .map(|v| E::from_f32(v * 8.0))
        .collect();
    let x = api::copy_host_vec_to_device(&Arc::new(xs.clone()))
        .sync()?
        .reshape(&[rows, cols])?;
    let mut y = api::zeros::<E::Out>(&[rows, cols]).sync()?;

    let start = Instant::now();
    softmax_into(&mut y, &x)?;
    let first = start.elapsed().as_secs_f64();
    let gpu = time_gpu(args.runs, args.reps, || softmax_into(&mut y, &x))?;

    let xs64: Vec<f64> = xs.iter().map(|&v| v.to_f64()).collect();
    let cpu = time_cpu(args.runs, || reference::softmax(&xs64, rows, cols));

    let got: Vec<f64> = y
        .dup()
        .to_host_vec()
        .sync()?
        .into_iter()
        .map(Num::to_f64)
        .collect();
    let want = reference::softmax(&xs64, rows, cols);
    let (tol, sum_tol) = softmax_tolerance::<E>();
    let err = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f64::max);
    let sum_err = got
        .chunks(cols)
        .map(|row| (row.iter().sum::<f64>() - 1.0).abs())
        .fold(0.0, f64::max);

    let label = format!("softmax {name} {rows}x{cols}");
    println!(
        "{}",
        format_line(&label, &format!("{bm}x{bn}"), first, gpu, cpu)
    );
    let ok = err <= tol && sum_err <= sum_tol;
    if !ok {
        eprintln!(
            "MISMATCH {label}: max error {err:e} (limit {tol:e}), row sum error {sum_err:e} (limit {sum_tol:e})"
        );
    }
    Ok(ok)
}

fn run(args: &Args) -> Result<bool, KernelError> {
    println!(
        "# {:?}: median of {} runs after a warmup call, {} calls per measurement",
        args.kernel, args.runs, args.reps
    );
    match args.kernel {
        Kernel::Softmax => {
            let mut ok = softmax_case::<f32>("f32", 4096, 1024, args)?;
            ok &= softmax_case::<f32>("f32", 16, 32768, args)?;
            ok &= softmax_case::<bf16>("bf16", 4096, 1024, args)?;
            Ok(ok)
        }
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&argv) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    match run(&args) {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Result<Args, String> {
        parse_args(&list.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn defaults_are_5_runs_and_20_reps() {
        assert_eq!(
            args(&["softmax"]).unwrap(),
            Args {
                kernel: Kernel::Softmax,
                runs: 5,
                reps: 20
            }
        );
    }

    #[test]
    fn runs_and_reps_can_come_before_or_after_the_kernel() {
        let want = Args {
            kernel: Kernel::Softmax,
            runs: 3,
            reps: 7,
        };
        assert_eq!(
            args(&["softmax", "--runs", "3", "--reps", "7"]).unwrap(),
            want
        );
        assert_eq!(
            args(&["--reps", "7", "--runs", "3", "softmax"]).unwrap(),
            want
        );
    }

    #[test]
    fn bad_arguments_are_rejected() {
        assert!(args(&[]).unwrap_err().contains("usage"));
        assert!(args(&["rmsnorm"]).unwrap_err().contains("unknown kernel"));
        assert!(
            args(&["softmax", "--bogus"])
                .unwrap_err()
                .contains("unknown option")
        );
        assert!(
            args(&["softmax", "--runs"])
                .unwrap_err()
                .contains("needs a number")
        );
        assert!(args(&["softmax", "--runs", "0"]).is_err());
        assert!(args(&["softmax", "--reps", "x"]).is_err());
    }

    #[test]
    fn median_of_an_odd_count_is_the_middle_value() {
        let s = Stats::of(&[5.0, 1.0, 3.0, 9.0, 2.0]);
        assert_eq!(
            s,
            Stats {
                median: 3.0,
                min: 1.0,
                max: 9.0
            }
        );
    }

    #[test]
    fn median_of_an_even_count_is_the_mean_of_the_middle_two() {
        assert_eq!(Stats::of(&[4.0, 1.0, 2.0, 10.0]).median, 3.0);
    }

    #[test]
    fn median_of_one_sample_is_that_sample() {
        let s = Stats::of(&[2.5]);
        assert_eq!((s.median, s.min, s.max), (2.5, 2.5, 2.5));
    }

    #[test]
    fn a_result_line_has_the_documented_format() {
        let gpu = Stats {
            median: 0.000123,
            min: 0.00012,
            max: 0.00014,
        };
        let cpu = Stats {
            median: 0.0456,
            min: 0.0451,
            max: 0.047,
        };
        assert_eq!(
            format_line("softmax f32 4096x1024", "4x1024", 0.31, gpu, cpu),
            "softmax f32 4096x1024 tiles 4x1024: first 0.31 s | gpu 0.123 ms (min 0.120, max 0.140) | cpu 45.6 ms (min 45.1, max 47.0)"
        );
    }

    #[test]
    fn time_cpu_calls_once_more_than_it_times() {
        let mut calls = 0;
        let stats = time_cpu(3, || calls += 1);
        assert_eq!(calls, 4);
        assert!(stats.min <= stats.median && stats.median <= stats.max);
    }

    #[test]
    fn time_gpu_makes_runs_times_reps_calls() {
        let mut calls = 0;
        time_gpu(3, 4, || {
            calls += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(calls, 12);
    }
}
