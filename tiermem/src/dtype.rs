//! What the manager needs to know about an element type once the type is
//! erased: the manager holds tensors of any type in one map, as
//! `Arc<Erased>` that is always an `Arc<Tensor<T>>`, plus the [`DTypeOps`]
//! table made when the tensor was registered.

use cutile::DTypeId;
use cutile::cuda_core::sys;
use cutile::prelude::*;
use std::any::{Any, TypeId};

/// A device tensor of some element type.
pub(crate) type Erased = dyn Any + Send + Sync;

type ReshapeFn = fn(Box<Erased>, &[usize]) -> Result<Arc<Erased>, Error>;

/// The operations on an erased tensor that depend on its element type.
#[derive(Clone, Copy)]
pub(crate) struct DTypeOps {
    /// The Rust type, which `tensor::<T>` compares: two types that shared a
    /// `DTypeId` could not be mixed up.
    pub type_id: TypeId,
    pub id: DTypeId,
    /// `size_of::<T>()`.
    pub elem_bytes: usize,
    /// The device address of the tensor's first element.
    pub device_ptr: fn(&Erased) -> sys::CUdeviceptr,
    /// A 1-D tensor of this many elements, not yet initialized.
    pub alloc: fn(&ExecutionContext, usize) -> Result<Box<Erased>, DeviceError>,
    /// Gives a 1-D tensor of this type the shape it had before it was spilled.
    pub reshape: ReshapeFn,
}

pub(crate) fn ops<T: DType>() -> DTypeOps {
    DTypeOps {
        type_id: TypeId::of::<T>(),
        id: T::DTYPE,
        elem_bytes: size_of::<T>(),
        device_ptr: |t| {
            t.downcast_ref::<Tensor<T>>()
                .expect("the entry holds a tensor of its own type")
                .device_pointer()
                .cu_deviceptr()
        },
        alloc: |ctx, elements| {
            // SAFETY: the copy that follows initializes every element before
            // the submission completes, which is the earliest anyone can read it.
            let tensor = unsafe {
                Tensor::<T>::uninitialized(elements)
                    .execute(ctx)?
                    .assume_init()
            };
            Ok(Box::new(tensor))
        },
        reshape: |boxed, shape| {
            let tensor = *boxed
                .downcast::<Tensor<T>>()
                .unwrap_or_else(|_| unreachable!("the entry holds a tensor of its own type"));
            Ok(Arc::new(tensor.reshape(shape)?))
        },
    }
}
