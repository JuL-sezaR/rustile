//! Tile kernels on cuTile Rust, and the CPU code they are tested against.
//!
//! The design is in `docs/design/kernels.md`. The kernels are [`softmax`],
//! [`rmsnorm`], [`gather`] and [`linear`]. The tile-shape rules are in
//! [`tiles`] and the CPU references the kernels are tested against in
//! [`reference`](mod@reference).

mod gather;
mod kernels;
mod linear;
pub mod reference;
mod rmsnorm;
mod softmax;
pub mod tiles;

pub use gather::{gather, gather_into, gather_into_with_tiles};
pub use linear::{linear, linear_into};
pub use rmsnorm::{rmsnorm, rmsnorm_into};
pub use softmax::{softmax, softmax_into};

use cutile::half::bf16;
use cutile::prelude::{DType, DeviceError, Tensor};
use std::fmt;
use tiles::LinearTiles;

#[derive(Debug)]
pub enum KernelError {
    /// The arguments do not fit together. Nothing was launched.
    Shape(String),
    /// A cutile or CUDA failure, including a kernel that does not compile.
    Device(cutile::error::Error),
}

impl KernelError {
    pub(crate) fn shape(message: impl Into<String>) -> Self {
        KernelError::Shape(message.into())
    }
}

/// The `[rows, cols]` of a 2-D tensor.
pub(crate) fn dims_2d(name: &str, shape: &[i32]) -> Result<(usize, usize), KernelError> {
    match shape {
        [rows, cols] => Ok((*rows as usize, *cols as usize)),
        _ => Err(KernelError::shape(format!(
            "{name} must be 2-D, got shape {shape:?}"
        ))),
    }
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::Shape(message) => write!(f, "shape error: {message}"),
            KernelError::Device(e) => write!(f, "device error: {e}"),
        }
    }
}

impl std::error::Error for KernelError {}

impl From<cutile::error::Error> for KernelError {
    fn from(e: cutile::error::Error) -> Self {
        KernelError::Device(e)
    }
}

impl From<DeviceError> for KernelError {
    fn from(e: DeviceError) -> Self {
        KernelError::Device(e.into())
    }
}

mod sealed {
    use super::KernelError;
    use cutile::prelude::{DType, Tensor};

    /// What the kernels need from an element type and callers do not: the
    /// launch of each entry, for this input type and the output type `O`
    /// that [`Float::Out`] names. It lives in a module no other crate can
    /// name, which seals [`Float`] (a type needs this trait to implement it)
    /// and keeps the launch methods crate-private.
    ///
    /// [`Float`]: super::Float
    /// [`Float::Out`]: super::Float::Out
    pub trait Launch<O: DType>: DType {
        /// Launches the softmax entry for this element type, with no
        /// checks: `y` and `x` must have the same 2-D shape and `bm`, `bn`
        /// must be the tile from `tiles::row_tiles` for it.
        fn launch_softmax(
            y: &mut Tensor<O>,
            x: &Tensor<Self>,
            bm: usize,
            bn: usize,
        ) -> Result<(), KernelError>;

        /// Launches the RMSNorm entry, with no checks: `y` and `x` must have
        /// the same 2-D shape, `gain` one element per column, and `bm`, `bd`
        /// must be the tile from `tiles::row_tiles` for it.
        fn launch_rmsnorm(
            y: &mut Tensor<O>,
            x: &Tensor<Self>,
            gain: &Tensor<Self>,
            eps: f64,
            bm: usize,
            bd: usize,
        ) -> Result<(), KernelError>;

        /// Launches the linear entry, with no checks: `x` is `[m, k]`, `w` is
        /// `[n, k]`, `bias` is `[n]`, `y` is `[m, n]`, and the tile sizes are
        /// powers of two.
        fn launch_linear(
            y: &mut Tensor<O>,
            x: &Tensor<Self>,
            w: &Tensor<Self>,
            bias: &Tensor<O>,
            tiles: crate::tiles::LinearTiles,
        ) -> Result<(), KernelError>;
    }
}

/// Element types the computing kernels accept: `bf16` and `f32`, which both
/// write `f32`, and `f64`, which writes `f64`. The type written is
/// `<E as Float>::Out`, or `E::Out` in generic code. Sealed: no other type
/// can implement it.
pub trait Float: sealed::Launch<<Self as Float>::Out> {
    /// The type the kernels write for this input type.
    type Out: DType;
}

impl Float for bf16 {
    type Out = f32;
}

impl sealed::Launch<f32> for bf16 {
    fn launch_softmax(
        y: &mut Tensor<f32>,
        x: &Tensor<Self>,
        bm: usize,
        bn: usize,
    ) -> Result<(), KernelError> {
        softmax::launch_f32(y, x, bm, bn)
    }

    fn launch_rmsnorm(
        y: &mut Tensor<f32>,
        x: &Tensor<Self>,
        gain: &Tensor<Self>,
        eps: f64,
        bm: usize,
        bd: usize,
    ) -> Result<(), KernelError> {
        rmsnorm::launch_f32(y, x, gain, eps, bm, bd)
    }

    fn launch_linear(
        y: &mut Tensor<f32>,
        x: &Tensor<Self>,
        w: &Tensor<Self>,
        bias: &Tensor<f32>,
        tiles: LinearTiles,
    ) -> Result<(), KernelError> {
        linear::launch_f32(y, x, w, bias, tiles)
    }
}

impl Float for f32 {
    type Out = f32;
}

impl sealed::Launch<f32> for f32 {
    fn launch_softmax(
        y: &mut Tensor<f32>,
        x: &Tensor<Self>,
        bm: usize,
        bn: usize,
    ) -> Result<(), KernelError> {
        softmax::launch_f32(y, x, bm, bn)
    }

    fn launch_rmsnorm(
        y: &mut Tensor<f32>,
        x: &Tensor<Self>,
        gain: &Tensor<Self>,
        eps: f64,
        bm: usize,
        bd: usize,
    ) -> Result<(), KernelError> {
        rmsnorm::launch_f32(y, x, gain, eps, bm, bd)
    }

    fn launch_linear(
        y: &mut Tensor<f32>,
        x: &Tensor<Self>,
        w: &Tensor<Self>,
        bias: &Tensor<f32>,
        tiles: LinearTiles,
    ) -> Result<(), KernelError> {
        linear::launch_f32(y, x, w, bias, tiles)
    }
}

impl Float for f64 {
    type Out = f64;
}

impl sealed::Launch<f64> for f64 {
    fn launch_softmax(
        y: &mut Tensor<f64>,
        x: &Tensor<Self>,
        bm: usize,
        bn: usize,
    ) -> Result<(), KernelError> {
        softmax::launch_f64(y, x, bm, bn)
    }

    fn launch_rmsnorm(
        y: &mut Tensor<f64>,
        x: &Tensor<Self>,
        gain: &Tensor<Self>,
        eps: f64,
        bm: usize,
        bd: usize,
    ) -> Result<(), KernelError> {
        rmsnorm::launch_f64(y, x, gain, eps, bm, bd)
    }

    fn launch_linear(
        y: &mut Tensor<f64>,
        x: &Tensor<Self>,
        w: &Tensor<Self>,
        bias: &Tensor<f64>,
        tiles: LinearTiles,
    ) -> Result<(), KernelError> {
        linear::launch_f64(y, x, w, bias, tiles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutile::DTypeId;

    fn out_dtype<E: Float>() -> DTypeId {
        <E::Out as DType>::DTYPE
    }

    #[test]
    fn bf16_and_f32_write_f32_and_f64_writes_f64() {
        assert_eq!(out_dtype::<bf16>(), DTypeId::F32);
        assert_eq!(out_dtype::<f32>(), DTypeId::F32);
        assert_eq!(out_dtype::<f64>(), DTypeId::F64);
    }

    #[test]
    fn shape_errors_name_the_problem() {
        let e = KernelError::shape("gain has 3 elements, expected 4");
        assert_eq!(
            e.to_string(),
            "shape error: gain has 3 elements, expected 4"
        );
    }
}
