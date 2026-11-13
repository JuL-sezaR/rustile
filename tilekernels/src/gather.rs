//! Embedding gather: rows of a `[v, d]` table picked by a vector of ids.

use crate::kernels::entries;
use crate::tiles::{self, MAX_TILE_ELEMENTS};
use crate::{KernelError, dims_2d};
use cutile::prelude::*;

/// `out[i] = table[ids[i]]`, with `table` of shape `[v, d]`, `ids` of shape
/// `[n]` and the result `[n, d]`. Any element type; ids are `i32`.
///
/// **An id outside `0..v` gives a row of zeros**, not an error: the ids are
/// on the device, and checking them would cost a copy back.
pub fn gather<E: DType>(table: &Tensor<E>, ids: &Tensor<i32>) -> Result<Tensor<E>, KernelError> {
    let (n, d) = check(table, ids)?;
    tiles::gather_tiles(n, d)?;
    let mut y = api::zeros::<E>(&[n, d]).sync()?;
    gather_into(&mut y, table, ids)?;
    Ok(y)
}

/// Like [`gather`], writing into `y`, which must have shape `[n, d]`. Every
/// element of `y` is overwritten and none is read.
pub fn gather_into<E: DType>(
    y: &mut Tensor<E>,
    table: &Tensor<E>,
    ids: &Tensor<i32>,
) -> Result<(), KernelError> {
    let (n, d) = check(table, ids)?;
    let (bm, bd) = tiles::gather_tiles(n, d)?;
    gather_into_with_tiles(y, table, ids, bm, bd)
}

/// Like [`gather_into`], with the tile chosen by the caller: `bm` ids by `bd`
/// columns per block, both powers of two. For tests and tile experiments;
/// [`gather_into`] picks the tile from [`tiles::gather_tiles`].
pub fn gather_into_with_tiles<E: DType>(
    y: &mut Tensor<E>,
    table: &Tensor<E>,
    ids: &Tensor<i32>,
    bm: usize,
    bd: usize,
) -> Result<(), KernelError> {
    let (n, d) = check(table, ids)?;
    for (name, t) in [("bm", bm), ("bd", bd)] {
        if !t.is_power_of_two() {
            return Err(KernelError::shape(format!(
                "tile {name} = {t} is not a power of two"
            )));
        }
    }
    if bm.saturating_mul(bd) > MAX_TILE_ELEMENTS {
        return Err(KernelError::shape(format!(
            "the gather tile [{bm}, {bd}] would exceed {MAX_TILE_ELEMENTS} elements"
        )));
    }
    if y.shape() != [n as i32, d as i32] {
        return Err(KernelError::shape(format!(
            "y has shape {:?}, expected [{n}, {d}]: one row of the table per id",
            y.shape()
        )));
    }
    // SAFETY: the entry reads `table` through a gather view with zero
    // padding, so an id outside the table reads zeros instead of memory past
    // it, and ids past the end of `ids` read as 0, a row that exists. All
    // three tensors are borrowed for the whole call, and `sync` returns only
    // after the kernel has finished, so none is freed or written while the
    // kernel reads it.
    unsafe { entries::gather((&mut *y).partition([bm, bd]), table, ids) }
        .generics(vec![
            E::DTYPE.as_str().to_string(),
            bm.to_string(),
            bd.to_string(),
        ])
        .sync()?;
    Ok(())
}

/// `table` is a non-empty `[v, d]` and `ids` is 1-D; returns `(n, d)`.
fn check<E: DType>(table: &Tensor<E>, ids: &Tensor<i32>) -> Result<(usize, usize), KernelError> {
    let (v, d) = dims_2d("table", table.shape())?;
    if v == 0 {
        return Err(KernelError::shape("table has no rows"));
    }
    match ids.shape() {
        [n] => {
            tiles::gather_tiles(*n as usize, d)?;
            Ok((*n as usize, d))
        }
        other => Err(KernelError::shape(format!(
            "ids must be 1-D, got shape {other:?}"
        ))),
    }
}
