//! Tile kernels on cuTile Rust, and the CPU code they are tested against.
//!
//! The design is in `docs/design/kernels.md`. The kernels are added one at a
//! time; so far there is [`softmax`]. The tile-shape rules are in [`tiles`]
//! and the CPU references the kernels are tested against in [`reference`](mod@reference).

mod kernels;
pub mod reference;
mod softmax;
pub mod tiles;

pub use softmax::{softmax, softmax_into};

use cutile::half::bf16;
use cutile::prelude::{DeviceError, Tensor};
use std::fmt;

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

    /// What the kernels need from an element type and callers do not. It
    /// lives in a module no other crate can name, which seals [`Float`] and
    /// keeps the launch methods crate-private.
    ///
    /// [`Float`]: super::Float
    pub trait Sealed: DType {
        /// The type the kernels write for this input type.
        type Out: DType;

        /// Launches the softmax entry for this element type, with no
        /// checks: `y` and `x` must have the same 2-D shape and `bm`, `bn`
        /// must be the tile from `tiles::row_tiles` for it.
        fn launch_softmax(
            y: &mut Tensor<Self::Out>,
            x: &Tensor<Self>,
            bm: usize,
            bn: usize,
        ) -> Result<(), KernelError>;
    }
}

/// Element types the computing kernels accept: `bf16` and `f32`, which both
/// write `f32`, and `f64`, which writes `f64`. The type written is
/// `<E as Float>::Out`. Sealed: no other type can implement it.
pub trait Float: sealed::Sealed {}

impl<T: sealed::Sealed> Float for T {}

impl sealed::Sealed for bf16 {
    type Out = f32;

    fn launch_softmax(
        y: &mut Tensor<f32>,
        x: &Tensor<Self>,
        bm: usize,
        bn: usize,
    ) -> Result<(), KernelError> {
        softmax::launch_f32(y, x, bm, bn)
    }
}

impl sealed::Sealed for f32 {
    type Out = f32;

    fn launch_softmax(
        y: &mut Tensor<f32>,
        x: &Tensor<Self>,
        bm: usize,
        bn: usize,
    ) -> Result<(), KernelError> {
        softmax::launch_f32(y, x, bm, bn)
    }
}

impl sealed::Sealed for f64 {
    type Out = f64;

    fn launch_softmax(
        y: &mut Tensor<f64>,
        x: &Tensor<Self>,
        bm: usize,
        bn: usize,
    ) -> Result<(), KernelError> {
        softmax::launch_f64(y, x, bm, bn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutile::DTypeId;
    use cutile::prelude::DType;

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
