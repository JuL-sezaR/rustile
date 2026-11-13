//! A linear layer: `x @ w^T + bias`, with `w` stored one row per output.

use crate::kernels::entries;
use crate::tiles::{LinearTiles, MAX_TILE_ELEMENTS};
use crate::{Float, KernelError, dims_2d};
use cutile::prelude::*;

/// `y[i][j] = sum_p x[i][p] * w[j][p] + bias[j]`, with `x` of shape `[m, k]`,
/// `w` of shape `[n, k]` (one row per output, as weight files store it),
/// `bias` of shape `[n]` and the result `[m, n]`. BF16 or F32 input gives F32
/// output and takes an F32 bias; F64 gives F64. The tile is
/// [`LinearTiles::for_shape`].
///
/// With `bias: None` a zero bias is allocated for the call; a caller who
/// minds can allocate one and pass it.
pub fn linear<E: Float>(
    x: &Tensor<E>,
    w: &Tensor<E>,
    bias: Option<&Tensor<E::Out>>,
) -> Result<Tensor<E::Out>, KernelError> {
    let (m, k, n) = check(x, w, bias)?;
    let tiles = LinearTiles::for_shape(m, k, n)?;
    let mut y = api::zeros::<E::Out>(&[m, n]).sync()?;
    linear_into(&mut y, x, w, bias, tiles)?;
    Ok(y)
}

/// Like [`linear`], writing into `y`, which must have shape `[m, n]`, with
/// the tile chosen by the caller. Every element of `y` is overwritten and
/// none is read.
pub fn linear_into<E: Float>(
    y: &mut Tensor<E::Out>,
    x: &Tensor<E>,
    w: &Tensor<E>,
    bias: Option<&Tensor<E::Out>>,
    tiles: LinearTiles,
) -> Result<(), KernelError> {
    let (m, _, n) = check(x, w, bias)?;
    check_tiles(tiles)?;
    if y.shape() != [m as i32, n as i32] {
        return Err(KernelError::shape(format!(
            "y has shape {:?}, expected [{m}, {n}]",
            y.shape()
        )));
    }
    match bias {
        Some(b) => E::launch_linear(y, x, w, b, tiles),
        None => {
            let zero = api::zeros::<E::Out>(&[n]).sync()?;
            E::launch_linear(y, x, w, &zero, tiles)
        }
    }
}

/// `x` is `[m, k]`, `w` is `[n, k]` and `bias` is `[n]`, all non-empty;
/// returns `(m, k, n)`.
fn check<E: Float>(
    x: &Tensor<E>,
    w: &Tensor<E>,
    bias: Option<&Tensor<E::Out>>,
) -> Result<(usize, usize, usize), KernelError> {
    let (m, k) = dims_2d("x", x.shape())?;
    let (n, kw) = dims_2d("w", w.shape())?;
    if kw != k {
        return Err(KernelError::shape(format!(
            "w has {kw} columns and x has {k}: w must be [n, k] for x of [m, k]"
        )));
    }
    if let Some(b) = bias {
        match b.shape() {
            [len] if *len as usize == n => {}
            [len] => {
                return Err(KernelError::shape(format!(
                    "bias has {len} elements, expected {n}, one per row of w"
                )));
            }
            other => {
                return Err(KernelError::shape(format!(
                    "bias must be 1-D, got shape {other:?}"
                )));
            }
        }
    }
    // Rejects empty and oversized dimensions.
    LinearTiles::for_shape(m, k, n)?;
    Ok((m, k, n))
}

fn check_tiles(t: LinearTiles) -> Result<(), KernelError> {
    for (name, v) in [("bm", t.bm), ("bn", t.bn), ("bk", t.bk)] {
        if !v.is_power_of_two() {
            return Err(KernelError::shape(format!(
                "tile {name} = {v} is not a power of two"
            )));
        }
    }
    for (name, a, b) in [
        ("output", t.bm, t.bn),
        ("x operand", t.bm, t.bk),
        ("w operand", t.bn, t.bk),
    ] {
        if a.saturating_mul(b) > MAX_TILE_ELEMENTS {
            return Err(KernelError::shape(format!(
                "the linear {name} tile [{a}, {b}] would exceed {MAX_TILE_ELEMENTS} elements"
            )));
        }
    }
    Ok(())
}

pub(crate) fn launch_f32<E: DType>(
    y: &mut Tensor<f32>,
    x: &Tensor<E>,
    w: &Tensor<E>,
    b: &Tensor<f32>,
    t: LinearTiles,
) -> Result<(), KernelError> {
    entries::linear_f32((&mut *y).partition([t.bm, t.bn]), x, w, b)
        .generics(vec![
            E::DTYPE.as_str().to_string(),
            t.bm.to_string(),
            t.bn.to_string(),
            t.bk.to_string(),
        ])
        .sync()?;
    Ok(())
}

pub(crate) fn launch_f64(
    y: &mut Tensor<f64>,
    x: &Tensor<f64>,
    w: &Tensor<f64>,
    b: &Tensor<f64>,
    t: LinearTiles,
) -> Result<(), KernelError> {
    entries::linear_f64((&mut *y).partition([t.bm, t.bn]), x, w, b)
        .generics(vec![t.bm.to_string(), t.bn.to_string(), t.bk.to_string()])
        .sync()?;
    Ok(())
}
