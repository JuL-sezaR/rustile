//! RMSNorm on the GPU against the CPU reference. Needs a GPU.

mod common;

use common::{Num, dev, from_f32s, host, rnd, to_f64s, worst};
use cutile::api;
use cutile::half::bf16;
use cutile::prelude::*;
use tilekernels::{Float, KernelError, reference, rmsnorm, rmsnorm_into};

const EPS: f64 = 1e-6;

/// Tolerance for an output of this type: `abs + rel * |want|`.
fn tolerance<E: Float>() -> (f64, f64) {
    if <E::Out as DType>::DTYPE == cutile::DTypeId::F64 {
        (1e-12, 1e-12)
    } else {
        (1e-6, 1e-5)
    }
}

/// Runs RMSNorm and checks it against the reference, returning the device
/// result.
fn check<E>(label: &str, xs: &[E], gs: &[E], rows: usize, d: usize, eps: f64) -> Vec<f64>
where
    E: Float + Num,
    E::Out: Num,
{
    let x = dev(xs.to_vec(), &[rows, d]);
    let g = dev(gs.to_vec(), &[d]);
    let y = rmsnorm(&x, &g, eps).expect("rmsnorm");
    assert_eq!(y.shape(), [rows as i32, d as i32]);
    let got = to_f64s(&host(&y));
    let want = reference::rmsnorm(&to_f64s(xs), &to_f64s(gs), rows, d, eps);
    let (abs, rel) = tolerance::<E>();
    let err = worst(got.iter().zip(&want).map(|(g, w)| (g - w).abs()));
    // How close the worst element came to its limit; above 1 fails.
    let ratio = worst(
        got.iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs() / (abs + rel * w.abs())),
    );
    println!("{label} {rows}x{d}: max error {err:e}, worst error / limit {ratio:.3}");
    assert!(
        ratio <= 1.0,
        "{label} {rows}x{d}: error {ratio:.3} times the limit"
    );
    got
}

fn random<E: Num>(n: usize, seed: u64) -> Vec<E> {
    from_f32s(&rnd(n, seed))
}

#[test]
fn f32_matches_the_reference_on_sizes_around_the_tile() {
    for (rows, d) in [(7, 1000), (4, 1024), (3, 1025), (3, 1023), (1, 1), (33, 16)] {
        check::<f32>("f32", &random(rows * d, 11), &random(d, 12), rows, d, EPS);
    }
}

#[test]
fn bf16_input_matches_the_reference() {
    check::<bf16>("bf16", &random(7000, 11), &random(1000, 12), 7, 1000, EPS);
}

#[test]
fn f64_matches_the_reference() {
    check::<f64>("f64", &random(7000, 11), &random(1000, 12), 7, 1000, EPS);
}

#[test]
fn a_row_of_zeros_stays_exactly_zero() {
    let d = 100;
    let mut xs = random::<f32>(3 * d, 11);
    xs[d..2 * d].fill(0.0);
    let got = check::<f32>("f32 zero row", &xs, &random(d, 12), 3, d, EPS);
    assert!(got[d..2 * d].iter().all(|&v| v == 0.0));
}

#[test]
fn a_constant_row_with_unit_gain_is_c_over_its_root_mean_square() {
    let (d, c) = (100, 0.5f32);
    let got = check::<f32>("f32 constant", &vec![c; d], &vec![1.0; d], 1, d, EPS);
    let c = c as f64;
    let want = c / (c * c + EPS).sqrt();
    for &v in &got {
        assert!((v - want).abs() <= 1e-6, "got {v}, want {want}");
    }
}

fn expect_shape_error<T>(result: Result<T, KernelError>, what: &str) {
    match result {
        Err(KernelError::Shape(message)) => println!("{what}: {message}"),
        Err(other) => panic!("{what}: expected a shape error, got {other}"),
        Ok(_) => panic!("{what}: expected a shape error, got Ok"),
    }
}

#[test]
fn wrong_shapes_are_reported_before_launching() {
    let x = dev(vec![1.0f32; 8], &[2, 4]);
    let short_gain = dev(vec![1.0f32; 3], &[3]);
    expect_shape_error(rmsnorm(&x, &short_gain, EPS), "gain of the wrong length");
    let flat_gain = dev(vec![1.0f32; 4], &[1, 4]);
    expect_shape_error(rmsnorm(&x, &flat_gain, EPS), "2-D gain");
    let one_d = dev(vec![1.0f32; 4], &[4]);
    let gain = dev(vec![1.0f32; 4], &[4]);
    expect_shape_error(rmsnorm(&one_d, &gain, EPS), "1-D input");
    let mut y = api::zeros::<f32>(&[2, 3]).sync().unwrap();
    expect_shape_error(
        rmsnorm_into(&mut y, &x, &gain, EPS),
        "output of the wrong shape",
    );
}

/// cutile 0.4 cannot allocate a tensor with no elements, so these are
/// `api::meta` tensors: a shape and no device memory.
#[test]
fn a_tensor_with_no_rows_or_no_columns_is_a_shape_error() {
    for (shape, d) in [([0usize, 4], 4), ([4, 0], 0), ([0, 0], 0)] {
        let x = api::meta::<f32>(&shape).sync().unwrap();
        let gain = api::meta::<f32>(&[d]).sync().unwrap();
        expect_shape_error(rmsnorm(&x, &gain, EPS), &format!("rmsnorm of {shape:?}"));
        let mut y = api::meta::<f32>(&shape).sync().unwrap();
        expect_shape_error(
            rmsnorm_into(&mut y, &x, &gain, EPS),
            &format!("rmsnorm_into of {shape:?}"),
        );
    }
}

#[test]
fn rmsnorm_into_overwrites_the_output() {
    let (rows, d) = (3, 100);
    let xs = random::<f32>(rows * d, 3);
    let gs = random::<f32>(d, 4);
    let mut y = api::full(9.0f32, &[rows, d]).sync().unwrap();
    rmsnorm_into(
        &mut y,
        &dev(xs.clone(), &[rows, d]),
        &dev(gs.clone(), &[d]),
        EPS,
    )
    .unwrap();
    let got = to_f64s(&host(&y));
    let want = reference::rmsnorm(&to_f64s(&xs), &to_f64s(&gs), rows, d, EPS);
    let ratio = worst(
        got.iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs() / (1e-6 + 1e-5 * w.abs())),
    );
    println!("rmsnorm_into 3x100 over a filled output: worst error / limit {ratio:.3}");
    assert!(ratio <= 1.0);
}
