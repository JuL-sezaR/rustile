//! Tile kernels on cuTile Rust, and the CPU code they are tested against.
//!
//! The design is in `docs/design/kernels.md`. So far this crate holds the
//! parts that need no GPU: the tile-shape rules ([`tiles`]) and the CPU
//! references ([`reference`]). The kernels are added one at a time.

pub mod reference;
pub mod tiles;

use cutile::half::bf16;
use cutile::prelude::{DType, DeviceError};
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
    pub trait Sealed {}
    impl Sealed for cutile::half::bf16 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Element types the computing kernels accept, with the type they write.
/// Sealed: implemented for `bf16` and `f32` (both write `f32`) and `f64`.
pub trait Float: DType + sealed::Sealed {
    type Out: DType;
}

impl Float for bf16 {
    type Out = f32;
}

impl Float for f32 {
    type Out = f32;
}

impl Float for f64 {
    type Out = f64;
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
