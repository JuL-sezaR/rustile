//! Softmax along the columns of a `[rows, cols]` tensor.

use crate::kernels::entries;
use crate::{Float, KernelError, dims_2d, tiles};
use cutile::prelude::*;

/// `y[i][j] = exp(x[i][j] - max_j) / sum_j exp(x[i][j] - max_j)`, for BF16 or
/// F32 input (F32 output) or F64 input (F64 output). The result is a new
/// tensor of the same shape as `x`.
///
/// An entry of `-inf` gives exactly 0. A row that is all `-inf`, or that
/// holds NaN or `+inf`, is not defined.
pub fn softmax<E: Float>(x: &Tensor<E>) -> Result<Tensor<E::Out>, KernelError> {
    let (rows, cols) = dims_2d("x", x.shape())?;
    tiles::row_tiles(rows, cols)?;
    let mut y = api::zeros::<E::Out>(&[rows, cols]).sync()?;
    softmax_into(&mut y, x)?;
    Ok(y)
}

/// Like [`softmax`], writing into `y`, which must have the shape of `x`.
/// Every element of `y` is overwritten and none is read.
pub fn softmax_into<E: Float>(y: &mut Tensor<E::Out>, x: &Tensor<E>) -> Result<(), KernelError> {
    let (rows, cols) = dims_2d("x", x.shape())?;
    let (bm, bn) = tiles::row_tiles(rows, cols)?;
    if y.shape() != x.shape() {
        return Err(KernelError::shape(format!(
            "y has shape {:?}, expected {:?} like x",
            y.shape(),
            x.shape()
        )));
    }
    E::launch_softmax(y, x, bm, bn)
}

pub(crate) fn launch_f32<E: DType>(
    y: &mut Tensor<f32>,
    x: &Tensor<E>,
    bm: usize,
    bn: usize,
) -> Result<(), KernelError> {
    entries::softmax_f32((&mut *y).partition([bm, bn]), x)
        .generics(vec![
            E::DTYPE.as_str().to_string(),
            bm.to_string(),
            bn.to_string(),
        ])
        .sync()?;
    Ok(())
}

pub(crate) fn launch_f64(
    y: &mut Tensor<f64>,
    x: &Tensor<f64>,
    bm: usize,
    bn: usize,
) -> Result<(), KernelError> {
    entries::softmax_f64((&mut *y).partition([bm, bn]), x)
        .generics(vec![bm.to_string(), bn.to_string()])
        .sync()?;
    Ok(())
}
