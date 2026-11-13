//! Softmax on the GPU against the CPU reference. Needs a GPU.

mod common;

use common::{Num, dev, from_f32s, host, max_abs_err, rnd, to_f64s, worst};
use cutile::api;
use cutile::half::bf16;
use cutile::prelude::*;
use tilekernels::{Float, KernelError, reference, softmax, softmax_into};

/// Tolerance for an output of this type: (per element, per row sum).
fn tolerance<E: Float>() -> (f64, f64) {
    if <E::Out as DType>::DTYPE == cutile::DTypeId::F64 {
        (1e-14, 1e-13)
    } else {
        (1e-6, 1e-5)
    }
}

/// Runs softmax on `xs` (`rows × cols`) and checks it against the reference,
/// returning the device result.
fn check<E>(label: &str, xs: &[E], rows: usize, cols: usize) -> Vec<f64>
where
    E: Float + Num,
    E::Out: Num,
{
    let x = dev(xs.to_vec(), &[rows, cols]);
    let y = softmax(&x).expect("softmax");
    assert_eq!(y.shape(), [rows as i32, cols as i32]);
    let got = to_f64s(&host(&y));
    let want = reference::softmax(&to_f64s(xs), rows, cols);
    let (tol, sum_tol) = tolerance::<E>();
    let err = max_abs_err(&got, &want);
    let sum_err = worst(
        got.chunks(cols)
            .map(|row| (row.iter().sum::<f64>() - 1.0).abs()),
    );
    println!("{label} {rows}x{cols}: max error {err:e}, worst row sum error {sum_err:e}");
    assert!(err <= tol, "{label} {rows}x{cols}: error {err:e} > {tol:e}");
    assert!(
        sum_err <= sum_tol,
        "{label} {rows}x{cols}: row sum error {sum_err:e} > {sum_tol:e}"
    );
    got
}

#[test]
fn the_error_helpers_do_not_hide_a_nan() {
    assert!(max_abs_err(&[1.0, f64::NAN], &[1.0, 1.0]).is_nan());
    assert!(max_abs_err(&[f64::NAN, 1.0], &[1.0, 1.0]).is_nan());
    assert!(worst([0.5, f64::NAN, 0.1].into_iter()).is_nan());
    assert_eq!(max_abs_err(&[1.0, 3.0], &[1.5, 1.0]), 2.0);
}

fn random<E: Num>(rows: usize, cols: usize, seed: u64) -> Vec<E> {
    let values: Vec<f32> = rnd(rows * cols, seed).iter().map(|v| v * 8.0).collect();
    from_f32s(&values)
}

#[test]
fn f32_matches_the_reference_on_sizes_around_the_tile() {
    for (rows, cols) in [
        (5, 1000),
        (4, 64),
        (3, 65),
        (3, 63),
        (1, 1),
        (17, 7),
        (3, 4096),
    ] {
        check::<f32>("f32", &random(rows, cols, 7), rows, cols);
    }
}

#[test]
fn bf16_input_matches_the_reference() {
    check::<bf16>("bf16", &random(5, 1000, 7), 5, 1000);
}

#[test]
fn f64_matches_the_reference() {
    check::<f64>("f64", &random(5, 1000, 7), 5, 1000);
}

#[test]
fn a_single_element_row_is_exactly_one() {
    let got = check::<f32>("f32", &[3.5], 1, 1);
    assert_eq!(got, vec![1.0]);
}

#[test]
fn large_values_do_not_overflow() {
    let got = check::<f32>("f32 large", &[1e4, 1e4 - 1.0, -1e4], 1, 3);
    assert!(got.iter().all(|v| v.is_finite()));
}

#[test]
fn minus_infinity_gets_exactly_zero_and_the_row_still_sums_to_one() {
    let ninf = f32::NEG_INFINITY;
    let xs = [ninf, 0.0, 0.5, ninf, 1.0, -2.0];
    let got = check::<f32>("f32 -inf", &xs, 1, 6);
    assert_eq!(got[0], 0.0);
    assert_eq!(got[3], 0.0);
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
    let one_d = dev(vec![1.0f32; 8], &[8]);
    expect_shape_error(softmax(&one_d), "1-D input");
    let three_d = dev(vec![1.0f32; 8], &[2, 2, 2]);
    expect_shape_error(softmax(&three_d), "3-D input");

    let x = dev(vec![1.0f32; 8], &[2, 4]);
    let mut y = api::zeros::<f32>(&[2, 3]).sync().unwrap();
    expect_shape_error(softmax_into(&mut y, &x), "output of the wrong shape");
    let mut y = api::zeros::<f32>(&[4, 2]).sync().unwrap();
    expect_shape_error(softmax_into(&mut y, &x), "transposed output");
}

#[test]
fn softmax_into_overwrites_the_output() {
    let xs = random::<f32>(3, 100, 3);
    let x = dev(xs.clone(), &[3, 100]);
    let mut y = api::full(9.0f32, &[3, 100]).sync().unwrap();
    softmax_into(&mut y, &x).unwrap();
    let got = to_f64s(&host(&y));
    let want = reference::softmax(&to_f64s(&xs), 3, 100);
    let err = max_abs_err(&got, &want);
    println!("softmax_into 3x100 over a filled output: max error {err:e}");
    assert!(err <= 1e-6, "error {err:e} > 1e-6");
}

/// cutile 0.4 cannot allocate a tensor with no elements (`Tensor::uninitialized`
/// asserts `len > 0`), so these are `api::meta` tensors: a shape and no device
/// memory. The shape is rejected before anything touches the device.
#[test]
fn a_tensor_with_no_rows_or_no_columns_is_a_shape_error() {
    for shape in [[0usize, 4], [4, 0], [0, 0]] {
        let x = api::meta::<f32>(&shape).sync().unwrap();
        expect_shape_error(softmax(&x), &format!("softmax of {shape:?}"));
        let mut y = api::meta::<f32>(&shape).sync().unwrap();
        expect_shape_error(
            softmax_into(&mut y, &x),
            &format!("softmax_into of {shape:?}"),
        );
    }
}

/// The first call took about 20 s in the design probe, so this is run by hand:
/// `cargo test -p tilekernels --test softmax -- --ignored --nocapture`
#[test]
#[ignore]
fn wide_row() {
    let start = std::time::Instant::now();
    check::<f32>("f32 wide", &random(3, 248_320, 7), 3, 248_320);
    println!(
        "wide_row took {:?} including the first compile",
        start.elapsed()
    );
}
