//! Tile-shape rules. Pure functions, no GPU.
//!
//! cutile requires every tile dimension to be a positive power of two and
//! caps a tile at [`MAX_TILE_ELEMENTS`]. Sizes that are not tile multiples
//! need no special handling here: reads past the end of a tensor are zero and
//! stores past the end are dropped.

use crate::KernelError;

/// The largest tile cutile accepts, in elements.
pub const MAX_TILE_ELEMENTS: usize = 16_777_216;

/// Row kernels put `bm` rows in a block while the tile stays within this.
const ROW_TILE_TARGET: usize = 4096;

/// The columns one gather block copies, at most.
const GATHER_MAX_COLUMNS: usize = 256;

fn pow2(name: &str, n: usize) -> Result<usize, KernelError> {
    n.checked_next_power_of_two()
        .ok_or_else(|| KernelError::shape(format!("{name} = {n} is too large")))
}

/// A dimension of the data: nonzero and addressable by the kernels' `i32`.
fn check_dim(name: &str, n: usize) -> Result<(), KernelError> {
    if n == 0 {
        return Err(KernelError::shape(format!("{name} is 0")));
    }
    if i32::try_from(n).is_err() {
        return Err(KernelError::shape(format!(
            "{name} = {n} does not fit in i32"
        )));
    }
    Ok(())
}

fn check_tile(name: &str, dims: &[usize]) -> Result<(), KernelError> {
    let elements = dims
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .filter(|&n| n <= MAX_TILE_ELEMENTS);
    match elements {
        Some(_) => Ok(()),
        None => Err(KernelError::shape(format!(
            "the {name} tile {dims:?} would exceed {MAX_TILE_ELEMENTS} elements"
        ))),
    }
}

/// Tile `(bm, bn)` for a kernel that reduces over whole rows (softmax,
/// RMSNorm) of a `rows × cols` tensor.
///
/// The tile spans the row, `bn = pow2(cols)`, because a block can only store
/// its own tile. `bm` rows share a block as long as the tile stays at 4096
/// elements or fewer, up to 16 rows.
pub fn row_tiles(rows: usize, cols: usize) -> Result<(usize, usize), KernelError> {
    check_dim("rows", rows)?;
    check_dim("cols", cols)?;
    let bn = pow2("cols", cols)?;
    let bm = pow2("rows", rows)?.min((ROW_TILE_TARGET / bn).clamp(1, 16));
    check_tile("row", &[bm, bn])?;
    Ok((bm, bn))
}

/// Tile `(bm, bd)` for gathering `n_ids` rows of `d` columns: up to 16 ids
/// by up to 256 columns per block.
pub fn gather_tiles(n_ids: usize, d: usize) -> Result<(usize, usize), KernelError> {
    check_dim("n_ids", n_ids)?;
    check_dim("d", d)?;
    let bd = pow2("d", d)?.min(GATHER_MAX_COLUMNS);
    let bm = pow2("n_ids", n_ids)?.min(16);
    check_tile("gather", &[bm, bd])?;
    Ok((bm, bd))
}

/// The tile of a linear layer: `bm × bn` outputs per block, stepping `bk`
/// along the reduced axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearTiles {
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
}

impl LinearTiles {
    /// The default for `x: m × k` times `w: n × k`. Every axis is at least 16;
    /// the output tile is at most 64 × 64 and `bk` at most 32. These are a
    /// starting point, to be settled by the tile sweep in the benchmark.
    pub fn for_shape(m: usize, k: usize, n: usize) -> Result<Self, KernelError> {
        check_dim("m", m)?;
        check_dim("k", k)?;
        check_dim("n", n)?;
        let tiles = LinearTiles {
            bm: pow2("m", m)?.clamp(16, 64),
            bn: pow2("n", n)?.clamp(16, 64),
            bk: pow2("k", k)?.clamp(16, 32),
        };
        check_tile("linear output", &[tiles.bm, tiles.bn])?;
        check_tile("linear x operand", &[tiles.bm, tiles.bk])?;
        check_tile("linear w operand", &[tiles.bn, tiles.bk])?;
        Ok(tiles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(rows: usize, cols: usize) -> (usize, usize) {
        row_tiles(rows, cols).unwrap()
    }

    #[test]
    fn row_tile_spans_the_row_with_a_power_of_two() {
        // one under, exact, one over a power of two
        assert_eq!(rows(1000, 63).1, 64);
        assert_eq!(rows(1000, 64).1, 64);
        assert_eq!(rows(1000, 65).1, 128);
        assert_eq!(rows(1, 1).1, 1);
        assert_eq!(rows(1, 1000).1, 1024);
    }

    #[test]
    fn row_tile_holds_about_4096_elements() {
        assert_eq!(rows(4096, 1024), (4, 1024));
        assert_eq!(rows(4096, 64), (16, 64)); // 4096 / 64 = 64, capped at 16 rows
        assert_eq!(rows(4096, 256), (16, 256));
        assert_eq!(rows(4096, 512), (8, 512));
        assert_eq!(rows(4096, 4096), (1, 4096));
        assert_eq!(rows(4096, 4097), (1, 8192)); // wider than the target: one row
        assert_eq!(rows(3, 248_320), (1, 262_144));
    }

    #[test]
    fn row_tile_does_not_use_more_rows_than_there_are() {
        assert_eq!(rows(1, 64), (1, 64));
        assert_eq!(rows(3, 64), (4, 64));
        assert_eq!(rows(5, 64), (8, 64));
        assert_eq!(rows(17, 64), (16, 64));
    }

    #[test]
    fn row_tile_limit_is_16777216_elements() {
        assert_eq!(rows(1, MAX_TILE_ELEMENTS), (1, MAX_TILE_ELEMENTS));
        assert_eq!(rows(1, MAX_TILE_ELEMENTS / 2 + 1), (1, MAX_TILE_ELEMENTS));
        assert!(matches!(
            row_tiles(1, MAX_TILE_ELEMENTS + 1),
            Err(KernelError::Shape(_))
        ));
        assert!(matches!(
            row_tiles(1, usize::MAX),
            Err(KernelError::Shape(_))
        ));
    }

    #[test]
    fn row_tile_rejects_empty_and_oversized_dimensions() {
        assert!(row_tiles(0, 8).is_err());
        assert!(row_tiles(8, 0).is_err());
        assert!(row_tiles(i32::MAX as usize + 1, 8).is_err());
        assert!(row_tiles(i32::MAX as usize, 8).is_ok());
    }

    #[test]
    fn gather_tile_splits_columns_at_256() {
        assert_eq!(gather_tiles(13, 100).unwrap(), (16, 128));
        assert_eq!(gather_tiles(13, 255).unwrap(), (16, 256));
        assert_eq!(gather_tiles(13, 256).unwrap(), (16, 256));
        assert_eq!(gather_tiles(13, 257).unwrap(), (16, 256));
        assert_eq!(gather_tiles(13, 1).unwrap(), (16, 1));
        assert_eq!(gather_tiles(13, 1 << 20).unwrap(), (16, 256));
    }

    #[test]
    fn gather_tile_takes_up_to_16_ids() {
        assert_eq!(gather_tiles(1, 64).unwrap(), (1, 64));
        assert_eq!(gather_tiles(2, 64).unwrap(), (2, 64));
        assert_eq!(gather_tiles(16, 64).unwrap(), (16, 64));
        assert_eq!(gather_tiles(17, 64).unwrap(), (16, 64));
        assert_eq!(gather_tiles(1 << 20, 64).unwrap(), (16, 64));
    }

    #[test]
    fn gather_tile_rejects_empty_and_oversized_dimensions() {
        assert!(gather_tiles(0, 8).is_err());
        assert!(gather_tiles(8, 0).is_err());
        assert!(gather_tiles(i32::MAX as usize + 1, 8).is_err());
        assert!(gather_tiles(8, i32::MAX as usize + 1).is_err());
    }

    fn linear(m: usize, k: usize, n: usize) -> (usize, usize, usize) {
        let t = LinearTiles::for_shape(m, k, n).unwrap();
        (t.bm, t.bn, t.bk)
    }

    #[test]
    fn linear_tiles_are_at_least_16_on_every_axis() {
        assert_eq!(linear(1, 1, 1), (16, 16, 16));
        assert_eq!(linear(15, 15, 15), (16, 16, 16));
        assert_eq!(linear(16, 16, 16), (16, 16, 16));
        assert_eq!(linear(1, 1024, 16), (16, 16, 32));
    }

    #[test]
    fn linear_tiles_follow_the_size_up_to_their_caps() {
        assert_eq!(linear(17, 17, 17), (32, 32, 32)); // one over 16
        assert_eq!(linear(32, 32, 32), (32, 32, 32));
        assert_eq!(linear(33, 33, 33), (64, 64, 32)); // bk is capped at 32
        assert_eq!(linear(63, 31, 63), (64, 64, 32));
        assert_eq!(linear(64, 32, 64), (64, 64, 32));
        assert_eq!(linear(65, 33, 65), (64, 64, 32));
        assert_eq!(linear(37, 100, 70), (64, 64, 32));
        assert_eq!(linear(256, 1024, 4096), (64, 64, 32));
        assert_eq!(linear(200, 8, 3), (64, 16, 16)); // k below bk
    }

    #[test]
    fn linear_tiles_reject_empty_and_oversized_dimensions() {
        assert!(LinearTiles::for_shape(0, 8, 8).is_err());
        assert!(LinearTiles::for_shape(8, 0, 8).is_err());
        assert!(LinearTiles::for_shape(8, 8, 0).is_err());
        assert!(LinearTiles::for_shape(8, i32::MAX as usize + 1, 8).is_err());
        assert!(LinearTiles::for_shape(i32::MAX as usize, i32::MAX as usize, 8).is_ok());
    }

    #[test]
    fn every_tile_dimension_is_a_power_of_two() {
        for rows_n in [1, 2, 3, 5, 17, 100, 4096] {
            for cols in [1, 3, 7, 64, 65, 1000, 5000] {
                let (bm, bn) = rows(rows_n, cols);
                assert!(bm.is_power_of_two() && bn.is_power_of_two());
                let (gm, gd) = gather_tiles(rows_n, cols).unwrap();
                assert!(gm.is_power_of_two() && gd.is_power_of_two());
                let t = LinearTiles::for_shape(rows_n, cols, rows_n + cols).unwrap();
                assert!(t.bm.is_power_of_two());
                assert!(t.bn.is_power_of_two());
                assert!(t.bk.is_power_of_two());
            }
        }
    }
}
