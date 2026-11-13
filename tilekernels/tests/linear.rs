//! The linear layer on the GPU against the CPU reference. Needs a GPU.

mod common;

use common::{Num, dev, from_f32s, host, rnd, to_f64s, worst};
use cutile::api;
use cutile::half::bf16;
use cutile::prelude::*;
use tilekernels::tiles::LinearTiles;
use tilekernels::{Float, KernelError, linear, linear_into, reference};

/// The tolerance factor for an output of this type: `16 × unit roundoff`,
/// times the sum of the magnitudes of the terms.
fn factor<E: Float>() -> (f64, f64) {
    if <E::Out as DType>::DTYPE == cutile::DTypeId::F64 {
        (16.0 * 2f64.powi(-53), 1e-300)
    } else {
        (16.0 * 2f64.powi(-24), 1e-30)
    }
}

/// Inputs for `x: [m, k]`, `w: [n, k]` and `bias: [n]`.
fn inputs<E: Num + Float>(m: usize, k: usize, n: usize) -> (Vec<E>, Vec<E>, Vec<E::Out>)
where
    E::Out: Num,
{
    (
        from_f32s(&rnd(m * k, 1)),
        from_f32s(&rnd(n * k, 2)),
        from_f32s(&rnd(n, 3)),
    )
}

/// Compares `got` with the reference for these inputs.
#[allow(clippy::too_many_arguments)]
fn compare<E>(
    label: &str,
    got: &[f64],
    xs: &[E],
    ws: &[E],
    bs: Option<&[E::Out]>,
    m: usize,
    k: usize,
    n: usize,
) where
    E: Float + Num,
    E::Out: Num,
{
    let bias = bs.map(to_f64s);
    let (want, scale) = reference::linear(&to_f64s(xs), &to_f64s(ws), bias.as_deref(), m, k, n);
    let (f, tiny) = factor::<E>();
    let err = worst(got.iter().zip(&want).map(|(g, w)| (g - w).abs()));
    let ratio = worst(
        got.iter()
            .zip(&want)
            .zip(&scale)
            .map(|((g, w), s)| (g - w).abs() / (f * s + tiny)),
    );
    println!("{label} {m}x{k}x{n}: max error {err:e}, worst error / limit {ratio:.3}");
    assert!(
        ratio <= 1.0,
        "{label} {m}x{k}x{n}: error {ratio:.3} times the limit"
    );
}

/// `linear` with the default tile and a bias.
fn check<E>(label: &str, m: usize, k: usize, n: usize)
where
    E: Float + Num,
    E::Out: Num,
{
    let (xs, ws, bs) = inputs::<E>(m, k, n);
    let x = dev(xs.clone(), &[m, k]);
    let w = dev(ws.clone(), &[n, k]);
    let b = dev(bs.clone(), &[n]);
    let y = linear(&x, &w, Some(&b)).expect("linear");
    assert_eq!(y.shape(), [m as i32, n as i32]);
    compare::<E>(label, &to_f64s(&host(&y)), &xs, &ws, Some(&bs), m, k, n);
}

#[test]
fn f32_matches_the_reference_on_sizes_around_the_tile() {
    for (m, k, n) in [
        (37, 100, 70),
        (64, 32, 64),
        (65, 33, 65),
        (63, 31, 63),
        (1, 1, 1),
        (1, 1024, 16),
        (200, 8, 3),
    ] {
        check::<f32>("f32", m, k, n);
    }
}

#[test]
fn explicit_tiles_write_every_element_of_the_output() {
    let (m, k, n) = (37, 100, 70);
    let (xs, ws, bs) = inputs::<f32>(m, k, n);
    let (x, w, b) = (
        dev(xs.clone(), &[m, k]),
        dev(ws.clone(), &[n, k]),
        dev(bs.clone(), &[n]),
    );
    for (bm, bn, bk) in [(16, 16, 16), (32, 64, 32)] {
        let mut y = api::full(9.0f32, &[m, n]).sync().unwrap();
        linear_into(&mut y, &x, &w, Some(&b), LinearTiles { bm, bn, bk }).unwrap();
        let label = format!("f32 tiles {bm}x{bn}x{bk}");
        compare::<f32>(&label, &to_f64s(&host(&y)), &xs, &ws, Some(&bs), m, k, n);
    }
}

#[test]
fn bf16_input_matches_the_reference() {
    check::<bf16>("bf16", 37, 100, 70);
}

#[test]
fn f64_matches_the_reference() {
    check::<f64>("f64", 37, 100, 70);
}

#[test]
fn no_bias_is_a_zero_bias_and_a_bias_is_added() {
    let (m, k, n) = (37, 100, 70);
    let (xs, ws, bs) = inputs::<f32>(m, k, n);
    let (x, w) = (dev(xs.clone(), &[m, k]), dev(ws.clone(), &[n, k]));
    let none = host(&linear(&x, &w, None).unwrap());
    let zero = host(&linear(&x, &w, Some(&dev(vec![0.0f32; n], &[n]))).unwrap());
    assert_eq!(none, zero);
    compare::<f32>("f32 no bias", &to_f64s(&none), &xs, &ws, None, m, k, n);
    let with = host(&linear(&x, &w, Some(&dev(bs.clone(), &[n]))).unwrap());
    // the bias moves every output by its column's bias
    let moved = (0..m * n).all(|p| ((with[p] - none[p]) - bs[p % n]).abs() <= 1e-6);
    assert!(moved);
}

#[test]
fn an_identity_weight_returns_x_exactly() {
    let (m, k) = (37, 64);
    let xs = from_f32s::<f32>(&rnd(m * k, 1));
    let mut identity = vec![0.0f32; k * k];
    for i in 0..k {
        identity[i * k + i] = 1.0;
    }
    let y = linear(&dev(xs.clone(), &[m, k]), &dev(identity, &[k, k]), None).unwrap();
    assert_eq!(host(&y), xs);
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
    let x = dev(vec![1.0f32; 2 * 4], &[2, 4]);
    let w = dev(vec![1.0f32; 3 * 4], &[3, 4]);
    let w5 = dev(vec![1.0f32; 3 * 5], &[3, 5]);
    expect_shape_error(linear(&x, &w5, None), "w columns differ from x columns");
    let short_bias = dev(vec![1.0f32; 2], &[2]);
    expect_shape_error(
        linear(&x, &w, Some(&short_bias)),
        "bias of the wrong length",
    );
    let flat_bias = dev(vec![1.0f32; 3], &[1, 3]);
    expect_shape_error(linear(&x, &w, Some(&flat_bias)), "2-D bias");
    let tiles = LinearTiles {
        bm: 16,
        bn: 16,
        bk: 16,
    };
    let mut y = api::zeros::<f32>(&[3, 2]).sync().unwrap();
    expect_shape_error(
        linear_into(&mut y, &x, &w, None, tiles),
        "output of the wrong shape",
    );
    let mut y = api::zeros::<f32>(&[2, 3]).sync().unwrap();
    let odd = LinearTiles {
        bm: 16,
        bn: 24,
        bk: 16,
    };
    expect_shape_error(
        linear_into(&mut y, &x, &w, None, odd),
        "a tile that is not a power of two",
    );
}

/// cutile 0.4 cannot allocate a tensor with no elements, so these are
/// `api::meta` tensors: a shape and no device memory.
#[test]
fn an_empty_dimension_is_a_shape_error() {
    for (xs, ws) in [
        ([0usize, 4], [3usize, 4]),
        ([2, 0], [3, 0]),
        ([2, 4], [0, 4]),
    ] {
        let x = api::meta::<f32>(&xs).sync().unwrap();
        let w = api::meta::<f32>(&ws).sync().unwrap();
        expect_shape_error(linear(&x, &w, None), &format!("x {xs:?} and w {ws:?}"));
    }
}
