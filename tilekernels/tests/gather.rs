//! Embedding gather on the GPU against the CPU reference, bit-exact. Needs a
//! GPU.

mod common;

use common::{dev, host};
use cutile::api;
use cutile::half::bf16;
use cutile::prelude::*;
use std::fmt::Debug;
use tilekernels::{KernelError, gather, gather_into_with_tiles, reference};

/// The table `[v, d]` whose element `i` (row-major) is `value(i)`.
fn table<T>(v: usize, d: usize, value: impl Fn(usize) -> T) -> Vec<T> {
    (0..v * d).map(value).collect()
}

/// The ids of the spec: `(i * 17 + 3) % v`.
fn ids(n: usize, v: usize) -> Vec<i32> {
    (0..n).map(|i| ((i * 17 + 3) % v) as i32).collect()
}

/// Gathers `ids` from `table` (with the default tile, or `tiles`) and
/// compares every element with the reference.
fn check<T: DType + PartialEq + Debug>(
    label: &str,
    table: &[T],
    v: usize,
    d: usize,
    ids: &[i32],
    zero: T,
    tiles: Option<(usize, usize)>,
) -> Vec<T> {
    let t = dev(table.to_vec(), &[v, d]);
    let i = dev(ids.to_vec(), &[ids.len()]);
    let n = ids.len();
    let y = match tiles {
        None => gather(&t, &i).expect("gather"),
        Some((bm, bd)) => {
            let mut y = api::zeros::<T>(&[n, d]).sync().unwrap();
            gather_into_with_tiles(&mut y, &t, &i, bm, bd).expect("gather_into_with_tiles");
            y
        }
    };
    assert_eq!(y.shape(), [n as i32, d as i32]);
    let got = host(&y);
    let want = reference::gather(table, ids, d, zero);
    let wrong: Vec<usize> = (0..got.len()).filter(|&p| got[p] != want[p]).collect();
    println!(
        "{label}: {n} ids x {d} columns, tiles {tiles:?}: {} of {} elements wrong",
        wrong.len(),
        got.len()
    );
    if let Some(&p) = wrong.first() {
        panic!(
            "{label}: first wrong element is row {}, column {}: got {:?}, want {:?}",
            p / d,
            p % d,
            got[p],
            want[p]
        );
    }
    got
}

#[test]
fn f32_with_the_default_and_explicit_tiles() {
    let (v, d) = (50, 100);
    let t = table(v, d, |i| i as f32);
    for tiles in [None, Some((4, 32)), Some((1, 128)), Some((16, 16))] {
        check("f32", &t, v, d, &ids(13, v), 0.0, tiles);
    }
}

#[test]
fn every_element_type_is_copied_bit_for_bit() {
    let (v, d) = (50, 100);
    let i = ids(13, v);
    check(
        "bf16",
        &table(v, d, |i| bf16::from_f32(i as f32 * 0.37)),
        v,
        d,
        &i,
        bf16::ZERO,
        None,
    );
    check(
        "f64",
        &table(v, d, |i| i as f64 + 0.25),
        v,
        d,
        &i,
        0.0,
        None,
    );
    check(
        "i64",
        &table(v, d, |i| i as i64 * 1_000_003 - 7),
        v,
        d,
        &i,
        0,
        None,
    );
    check("u8", &table(v, d, |i| (i % 251) as u8), v, d, &i, 0, None);
}

#[test]
fn column_counts_around_the_256_wide_tile() {
    let v = 50;
    for d in [256, 257, 255, 1] {
        check(
            "f32",
            &table(v, d, |i| i as f32),
            v,
            d,
            &ids(13, v),
            0.0,
            None,
        );
    }
}

#[test]
fn id_counts_around_the_16_tall_tile() {
    let (v, d) = (50, 100);
    let t = table(v, d, |i| i as f32);
    for n in [1, 16, 17] {
        check("f32", &t, v, d, &ids(n, v), 0.0, None);
    }
}

#[test]
fn repeated_ids_and_the_first_and_last_rows() {
    let (v, d) = (50, 100);
    let t = table(v, d, |i| i as f32);
    let last = (v - 1) as i32;
    check("f32", &t, v, d, &[0, last, 0, 0, last, 3, 3], 0.0, None);
}

#[test]
fn ids_just_outside_the_table_give_zero_rows() {
    let (v, d) = (50, 100);
    // + 1 keeps every real element nonzero, so a zero row cannot be a real one
    let t = table(v, d, |i| i as f32 + 1.0);
    let got = check("f32", &t, v, d, &[7, -1, 8, v as i32, 9], 0.0, None);
    assert!(got[d..2 * d].iter().all(|&x| x == 0.0));
    assert!(got[3 * d..4 * d].iter().all(|&x| x == 0.0));
    let t = table(v, d, |i| i as i64 + 1);
    check("i64", &t, v, d, &[7, -1, 8, v as i32, 9], 0, None);
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
    let t = dev(vec![1.0f32; 12], &[3, 4]);
    let flat_ids = dev(vec![0i32; 4], &[2, 2]);
    expect_shape_error(gather(&t, &flat_ids), "2-D ids");
    let i = dev(vec![0i32, 1], &[2]);
    let flat_table = dev(vec![1.0f32; 12], &[12]);
    expect_shape_error(gather(&flat_table, &i), "1-D table");
    let mut y = api::zeros::<f32>(&[2, 3]).sync().unwrap();
    expect_shape_error(
        tilekernels::gather_into(&mut y, &t, &i),
        "output of the wrong shape",
    );
    let mut y = api::zeros::<f32>(&[2, 4]).sync().unwrap();
    expect_shape_error(
        gather_into_with_tiles(&mut y, &t, &i, 3, 4),
        "a tile that is not a power of two",
    );
}

/// cutile 0.4 cannot allocate a tensor with no elements, so these are
/// `api::meta` tensors: a shape and no device memory.
#[test]
fn an_empty_table_or_no_ids_is_a_shape_error() {
    let ids2 = api::meta::<i32>(&[2]).sync().unwrap();
    for shape in [[0usize, 4], [4, 0]] {
        let t = api::meta::<f32>(&shape).sync().unwrap();
        expect_shape_error(gather(&t, &ids2), &format!("table of {shape:?}"));
    }
    let t = api::meta::<f32>(&[3, 4]).sync().unwrap();
    let no_ids = api::meta::<i32>(&[0]).sync().unwrap();
    expect_shape_error(gather(&t, &no_ids), "no ids");
}

#[test]
fn ids_far_outside_the_table_give_zero_rows() {
    let (v, d) = (50, 100);
    let t = table(v, d, |i| i as f32 + 1.0);
    let far = [i32::MAX, i32::MIN, 1 << 20, -(1 << 20), 21_474_837, 5];
    let got = check("f32 far", &t, v, d, &far, 0.0, None);
    assert!(got[..5 * d].iter().all(|&x| x == 0.0));
}
