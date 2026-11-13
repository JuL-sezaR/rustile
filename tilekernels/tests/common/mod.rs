//! Helpers shared by the kernel tests.

#![allow(dead_code)]

use cutile::api;
use cutile::half::bf16;
use cutile::prelude::*;

/// Deterministic values in [-1, 1): xorshift64.
pub fn rnd(n: usize, seed: u64) -> Vec<f32> {
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

/// Uploads a host vector as a tensor of `shape`.
pub fn dev<T: DType>(v: Vec<T>, shape: &[usize]) -> Tensor<T> {
    api::copy_host_vec_to_device(&Arc::new(v))
        .sync()
        .expect("upload")
        .reshape(shape)
        .expect("reshape")
}

/// Reads a tensor back.
pub fn host<T: DType>(t: &Tensor<T>) -> Vec<T> {
    t.dup().to_host_vec().sync().expect("read back")
}

/// Exact conversion of a stored value to `f64`, and back from `f32` for
/// building inputs.
pub trait Num: Copy {
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

pub fn to_f64s<T: Num>(v: &[T]) -> Vec<f64> {
    v.iter().map(|&x| x.to_f64()).collect()
}

pub fn from_f32s<T: Num>(v: &[f32]) -> Vec<T> {
    v.iter().map(|&x| T::from_f32(x)).collect()
}

/// The largest value, or NaN if any is NaN. `f64::max` alone would skip a
/// NaN, and a kernel that returned NaN would pass every tolerance check.
pub fn worst(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0, |m, v| {
        if m.is_nan() || v.is_nan() {
            f64::NAN
        } else {
            m.max(v)
        }
    })
}

/// The largest absolute difference; NaN if either side holds a NaN.
pub fn max_abs_err(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    worst(got.iter().zip(want).map(|(g, w)| (g - w).abs()))
}
