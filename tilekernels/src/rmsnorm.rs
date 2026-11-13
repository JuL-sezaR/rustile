//! RMSNorm along the columns of a `[rows, d]` tensor.

use crate::kernels::entries;
use crate::{Float, KernelError, dims_2d, tiles};
use cutile::prelude::*;

/// `y[i][j] = x[i][j] / sqrt(mean_j(x[i][j]^2) + eps) * gain[j]`, with `x` of
/// shape `[rows, d]` and `gain` of shape `[d]`. BF16 or F32 input gives F32
/// output; F64 gives F64. For F32 output `eps` is rounded to `f32`. The
/// result is a new tensor of the same shape as `x`.
pub fn rmsnorm<E: Float>(
    x: &Tensor<E>,
    gain: &Tensor<E>,
    eps: f64,
) -> Result<Tensor<E::Out>, KernelError> {
    let (rows, d) = check(x, gain)?;
    tiles::row_tiles(rows, d)?;
    let mut y = api::zeros::<E::Out>(&[rows, d]).sync()?;
    rmsnorm_into(&mut y, x, gain, eps)?;
    Ok(y)
}

/// Like [`rmsnorm`], writing into `y`, which must have the shape of `x`.
/// Every element of `y` is overwritten and none is read.
pub fn rmsnorm_into<E: Float>(
    y: &mut Tensor<E::Out>,
    x: &Tensor<E>,
    gain: &Tensor<E>,
    eps: f64,
) -> Result<(), KernelError> {
    let (rows, d) = check(x, gain)?;
    let (bm, bd) = tiles::row_tiles(rows, d)?;
    if y.shape() != x.shape() {
        return Err(KernelError::shape(format!(
            "y has shape {:?}, expected {:?} like x",
            y.shape(),
            x.shape()
        )));
    }
    E::launch_rmsnorm(y, x, gain, eps, bm, bd)
}

/// `x` is 2-D and `gain` has one element per column.
fn check<E: Float>(x: &Tensor<E>, gain: &Tensor<E>) -> Result<(usize, usize), KernelError> {
    let (rows, d) = dims_2d("x", x.shape())?;
    match gain.shape() {
        [n] if *n as usize == d => Ok((rows, d)),
        [n] => Err(KernelError::shape(format!(
            "gain has {n} elements, expected {d}, one per column of x"
        ))),
        other => Err(KernelError::shape(format!(
            "gain must be 1-D, got shape {other:?}"
        ))),
    }
}

pub(crate) fn launch_f32<E: DType>(
    y: &mut Tensor<f32>,
    x: &Tensor<E>,
    gain: &Tensor<E>,
    eps: f64,
    bm: usize,
    bd: usize,
) -> Result<(), KernelError> {
    entries::rmsnorm_f32((&mut *y).partition([bm, bd]), x, gain, eps as f32)
        .generics(vec![
            E::DTYPE.as_str().to_string(),
            bm.to_string(),
            bd.to_string(),
        ])
        .sync()?;
    Ok(())
}

pub(crate) fn launch_f64(
    y: &mut Tensor<f64>,
    x: &Tensor<f64>,
    gain: &Tensor<f64>,
    eps: f64,
    bm: usize,
    bd: usize,
) -> Result<(), KernelError> {
    entries::rmsnorm_f64((&mut *y).partition([bm, bd]), x, gain, eps)
        .generics(vec![bm.to_string(), bd.to_string()])
        .sync()?;
    Ok(())
}
