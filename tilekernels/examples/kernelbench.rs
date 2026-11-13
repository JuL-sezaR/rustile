//! Benchmark of the tile kernels against a single-threaded CPU loop.
//!
//! ```sh
//! # latency per call against the CPU loop
//! cargo run --release -p tilekernels --example kernelbench -- softmax [--runs N] [--reps R]
//! # the first call of each case (the JIT compile), over separate processes
//! cargo run --release -p tilekernels --example kernelbench -- softmax --first-call [--processes N]
//! ```
//!
//! Before anything is timed, a throwaway kernel is compiled and launched, so
//! that cutile's one-time check of the toolchain is not part of any case.
//!
//! **Latency.** For each case the first call is the warmup and is not
//! reported. Then `--runs` measurements are taken, each the wall time of
//! `--reps` consecutive calls divided by `--reps`. A call is the whole of
//! `softmax_into`: argument checks, kernel cache lookup, launch, and waiting
//! for the stream. It is not the kernel's own time. The CPU reference gets one
//! unreported warmup call and `--runs` timed calls. Medians are reported with
//! min and max; nothing else is excluded.
//!
//! **First call.** A compile happens once per process, so it cannot be
//! repeated inside one. `--first-call` starts `--processes` copies of this
//! program, each of which times only the first call of every case, and
//! reports the median, min and max over the processes.

use cutile::DTypeId;
use cutile::api;
use cutile::half::bf16;
use cutile::prelude::*;
use std::hint::black_box;
use std::process::{Command, Stdio};
use std::time::Instant;
use tilekernels::{Float, KernelError, reference, softmax_into, tiles};

const DEFAULT_RUNS: usize = 5;
const DEFAULT_REPS: usize = 20;
const DEFAULT_PROCESSES: usize = 5;

type BoxError = Box<dyn std::error::Error>;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kernel {
    Softmax,
}

impl Kernel {
    fn name(self) -> &'static str {
        match self {
            Kernel::Softmax => "softmax",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    /// Latency per call against the CPU loop.
    Bench,
    /// Start `processes` children and report their first calls.
    FirstCall { processes: usize },
    /// Time only the first call of each case and print it for the parent.
    FirstCallChild,
}

#[derive(Debug, PartialEq)]
struct Args {
    kernel: Kernel,
    runs: usize,
    reps: usize,
    mode: Mode,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut kernel = None;
    let mut runs = DEFAULT_RUNS;
    let mut reps = DEFAULT_REPS;
    let mut processes = None;
    let mut first_call = false;
    let mut child = false;
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
            "--processes" => processes = Some(number("--processes")?),
            "--first-call" => first_call = true,
            "--first-call-child" => child = true,
            "softmax" => kernel = Some(Kernel::Softmax),
            other if other.starts_with("--") => return Err(format!("unknown option {other}")),
            other => return Err(format!("unknown kernel {other}; available: softmax")),
        }
    }
    let kernel = kernel.ok_or(
        "usage: kernelbench <kernel> [--runs N] [--reps R] [--first-call [--processes N]]; kernels: softmax",
    )?;
    let mode = match (first_call, child, processes) {
        (true, true, _) => {
            return Err("--first-call and --first-call-child exclude each other".into());
        }
        (false, true, _) => Mode::FirstCallChild,
        (true, false, n) => Mode::FirstCall {
            processes: n.unwrap_or(DEFAULT_PROCESSES),
        },
        (false, false, None) => Mode::Bench,
        (false, false, Some(_)) => return Err("--processes needs --first-call".into()),
    };
    Ok(Args {
        kernel,
        runs,
        reps,
        mode,
    })
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

fn format_line(label: &str, tiles: &str, gpu: Stats, cpu: Stats) -> String {
    let ms = |s: f64| s * 1e3;
    format!(
        "{label} tiles {tiles}: latency {:.3} ms/call (min {:.3}, max {:.3}) | cpu {:.1} ms (min {:.1}, max {:.1})",
        ms(gpu.median),
        ms(gpu.min),
        ms(gpu.max),
        ms(cpu.median),
        ms(cpu.min),
        ms(cpu.max),
    )
}

/// What a child prints for the parent: one line per case.
fn child_line(label: &str, tiles: &str, first: f64) -> String {
    format!("FIRST|{label}|{tiles}|{first}")
}

/// The cases of one child's output, in order: (label, tiles, seconds).
fn parse_child_output(output: &str) -> Result<Vec<(String, String, f64)>, String> {
    let mut cases = Vec::new();
    for line in output.lines().filter(|l| l.starts_with("FIRST|")) {
        let parts: Vec<&str> = line.split('|').collect();
        let [_, label, tiles, seconds] = parts[..] else {
            return Err(format!("cannot read the child's line {line:?}"));
        };
        let seconds = seconds
            .parse::<f64>()
            .map_err(|_| format!("cannot read the time in the child's line {line:?}"))?;
        cases.push((label.to_string(), tiles.to_string(), seconds));
    }
    Ok(cases)
}

/// The first-call line of one case, from its times in each process.
fn format_first_line(label: &str, tiles: &str, seconds: &[f64]) -> String {
    let s = Stats::of(seconds);
    format!(
        "{label} tiles {tiles}: first call {:.2} s (min {:.2}, max {:.2}) over {} processes",
        s.median,
        s.min,
        s.max,
        seconds.len()
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

/// The largest value, or NaN if any is NaN (`f64::max` would skip it).
fn worst(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0, |m, v| {
        if m.is_nan() || v.is_nan() {
            f64::NAN
        } else {
            m.max(v)
        }
    })
}

/// Compares a softmax result with the reference. The error is the text that
/// follows `MISMATCH <case>:`.
fn check_softmax(
    got: &[f64],
    want: &[f64],
    cols: usize,
    tol: f64,
    sum_tol: f64,
) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!("got {} values, want {}", got.len(), want.len()));
    }
    let err = worst(got.iter().zip(want).map(|(g, w)| (g - w).abs()));
    let sum_err = worst(
        got.chunks(cols)
            .map(|row| (row.iter().sum::<f64>() - 1.0).abs()),
    );
    // Written so that NaN fails.
    if err <= tol && sum_err <= sum_tol {
        Ok(())
    } else {
        Err(format!(
            "max error {err:e} (limit {tol:e}), row sum error {sum_err:e} (limit {sum_tol:e})"
        ))
    }
}

/// cutile checks the toolchain once per process, when it compiles its first
/// kernel. Compile and launch one that no case uses (F32, a 1 x 1 tile), so
/// that check is not part of any case's first call.
fn warm_toolchain() -> Result<(), KernelError> {
    let x = api::ones::<f32>(&[1, 1]).sync()?;
    let mut y = api::zeros::<f32>(&[1, 1]).sync()?;
    softmax_into(&mut y, &x)
}

/// One softmax case. In a first-call child it prints the first call's time
/// and stops. Otherwise it returns whether the last GPU result matched the
/// reference.
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
    let label = format!("softmax {name} {rows}x{cols}");
    let tile = format!("{bm}x{bn}");
    if args.mode == Mode::FirstCallChild {
        println!("{}", child_line(&label, &tile, first));
        return Ok(true);
    }
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
    let verdict = check_softmax(&got, &want, cols, tol, sum_tol);

    println!("{}", format_line(&label, &tile, gpu, cpu));
    if let Err(message) = &verdict {
        eprintln!("MISMATCH {label}: {message}");
    }
    Ok(verdict.is_ok())
}

/// Starts `processes` children that each time the first call of every case.
fn first_call(args: &Args, processes: usize) -> Result<bool, BoxError> {
    println!(
        "# {}: first call of each case (compile included), after a throwaway kernel has paid the toolchain check; median of {processes} processes",
        args.kernel.name()
    );
    let exe = std::env::current_exe()?;
    let mut cases: Vec<(String, String, Vec<f64>)> = Vec::new();
    for i in 0..processes {
        let output = Command::new(&exe)
            .args([args.kernel.name(), "--first-call-child"])
            .stderr(Stdio::inherit())
            .output()?;
        if !output.status.success() {
            return Err(
                format!("process {} of {processes} failed: {}", i + 1, output.status).into(),
            );
        }
        let parsed = parse_child_output(&String::from_utf8_lossy(&output.stdout))?;
        if i == 0 {
            cases = parsed
                .into_iter()
                .map(|(l, t, s)| (l, t, vec![s]))
                .collect();
        } else {
            if parsed.len() != cases.len() {
                return Err(format!(
                    "process {} reported {} cases, the first {}",
                    i + 1,
                    parsed.len(),
                    cases.len()
                )
                .into());
            }
            for (case, (label, _, seconds)) in cases.iter_mut().zip(parsed) {
                if case.0 != label {
                    return Err(
                        format!("process {} reported {label}, expected {}", i + 1, case.0).into(),
                    );
                }
                case.2.push(seconds);
            }
        }
    }
    for (label, tiles, seconds) in &cases {
        println!("{}", format_first_line(label, tiles, seconds));
    }
    Ok(true)
}

fn run(args: &Args) -> Result<bool, BoxError> {
    if let Mode::FirstCall { processes } = args.mode {
        return first_call(args, processes);
    }
    warm_toolchain()?;
    if args.mode == Mode::Bench {
        println!(
            "# {}: latency is end to end per call (argument checks, kernel cache lookup, launch, wait); median of {} runs after a warmup call, {} calls per measurement",
            args.kernel.name(),
            args.runs,
            args.reps
        );
    }
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
                reps: 20,
                mode: Mode::Bench
            }
        );
    }

    #[test]
    fn runs_and_reps_can_come_before_or_after_the_kernel() {
        let want = Args {
            kernel: Kernel::Softmax,
            runs: 3,
            reps: 7,
            mode: Mode::Bench,
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
    fn first_call_mode_defaults_to_5_processes() {
        assert_eq!(
            args(&["softmax", "--first-call"]).unwrap().mode,
            Mode::FirstCall { processes: 5 }
        );
        assert_eq!(
            args(&["softmax", "--first-call", "--processes", "3"])
                .unwrap()
                .mode,
            Mode::FirstCall { processes: 3 }
        );
        assert_eq!(
            args(&["softmax", "--first-call-child"]).unwrap().mode,
            Mode::FirstCallChild
        );
    }

    #[test]
    fn first_call_flags_that_do_not_fit_together_are_rejected() {
        assert!(
            args(&["softmax", "--processes", "3"])
                .unwrap_err()
                .contains("needs --first-call")
        );
        assert!(args(&["softmax", "--first-call", "--first-call-child"]).is_err());
        assert!(args(&["softmax", "--first-call", "--processes", "0"]).is_err());
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
            format_line("softmax f32 4096x1024", "4x1024", gpu, cpu),
            "softmax f32 4096x1024 tiles 4x1024: latency 0.123 ms/call (min 0.120, max 0.140) | cpu 45.6 ms (min 45.1, max 47.0)"
        );
    }

    #[test]
    fn a_child_line_survives_the_trip_to_the_parent() {
        let out = format!(
            "noise\n{}\n{}\n",
            child_line("softmax f32 4096x1024", "4x1024", 0.2125),
            child_line("softmax bf16 4096x1024", "4x1024", 0.18)
        );
        assert_eq!(
            parse_child_output(&out).unwrap(),
            vec![
                (
                    "softmax f32 4096x1024".to_string(),
                    "4x1024".to_string(),
                    0.2125
                ),
                (
                    "softmax bf16 4096x1024".to_string(),
                    "4x1024".to_string(),
                    0.18
                ),
            ]
        );
        assert!(parse_child_output("FIRST|a|b").is_err());
        assert!(parse_child_output("FIRST|a|b|slow").is_err());
        assert!(parse_child_output("nothing here").unwrap().is_empty());
    }

    #[test]
    fn the_first_call_line_reports_median_and_range_over_processes() {
        assert_eq!(
            format_first_line(
                "softmax f32 4096x1024",
                "4x1024",
                &[0.25, 0.20, 0.22, 0.30, 0.21]
            ),
            "softmax f32 4096x1024 tiles 4x1024: first call 0.22 s (min 0.20, max 0.30) over 5 processes"
        );
    }

    fn softmax_row() -> Vec<f64> {
        vec![0.1, 0.2, 0.3, 0.4]
    }

    #[test]
    fn a_correct_result_passes() {
        let want = softmax_row();
        assert!(check_softmax(&want, &want, 4, 1e-6, 1e-5).is_ok());
    }

    #[test]
    fn a_wrong_element_is_a_mismatch() {
        let want = softmax_row();
        let mut got = want.clone();
        got[1] += 1e-3;
        got[2] -= 1e-3; // the row still sums to 1: only the element check can catch it
        let message = check_softmax(&got, &want, 4, 1e-6, 1e-5).unwrap_err();
        assert!(message.starts_with("max error "), "{message}");
        // with a looser limit it passes, so it was the element check that failed
        assert!(check_softmax(&got, &want, 4, 1e-2, 1e-5).is_ok());
    }

    #[test]
    fn a_row_that_does_not_sum_to_one_is_a_mismatch() {
        let want = softmax_row();
        let got: Vec<f64> = want.iter().map(|v| v * 1.001).collect();
        // each element is within 4e-4 of the reference, so allow that
        let message = check_softmax(&got, &want, 4, 1e-3, 1e-5).unwrap_err();
        assert!(message.contains("row sum error"), "{message}");
    }

    #[test]
    fn nan_and_a_wrong_length_are_mismatches() {
        let want = softmax_row();
        let mut got = want.clone();
        got[0] = f64::NAN;
        assert!(check_softmax(&got, &want, 4, 1e-6, 1e-5).is_err());
        assert!(check_softmax(&[f64::NAN; 4], &want, 4, 1e-6, 1e-5).is_err());
        assert!(check_softmax(&want[..3], &want, 4, 1e-6, 1e-5).is_err());
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
