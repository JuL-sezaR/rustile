//! Host-device migration: page-locked host buffers and the copy operations
//! that run on the manager's dedicated copy stream.
//!
//! Page-locked ("pinned") memory lets the driver DMA straight to and from the
//! buffer, and a host-to-device copy from it really is asynchronous. A copy
//! from pageable memory is staged by the driver before the call returns, so it
//! can never overlap with compute.

use cutile::cuda_async::device_context::with_default_device_policy;
use cutile::cuda_async::device_future::DeviceFuture;
use cutile::cuda_core::{DriverError, Stream, memcpy_dtoh_async, memcpy_htod_async, sys};
use cutile::prelude::*;
use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::ptr::NonNull;
use std::task::{Context, Poll, Waker};

/// Page-locked host memory holding the contents of one tensor.
///
/// It is allocated uninitialized and only ever handed out by [`SpillToHost`],
/// whose result is unreachable until the device has filled it.
pub(crate) struct PinnedBuf<T: DType> {
    ptr: NonNull<T>,
    len: usize,
    device: Arc<Device>,
}

// SAFETY: plain host memory holding `T: Send + Sync` values; after it has been
// filled it is only read.
unsafe impl<T: DType> Send for PinnedBuf<T> {}
unsafe impl<T: DType> Sync for PinnedBuf<T> {}

impl<T: DType> PinnedBuf<T> {
    pub(crate) fn uninit(device: &Arc<Device>, len: usize) -> Result<Self, DriverError> {
        let num_bytes = len
            .checked_mul(size_of::<T>())
            .ok_or(DriverError(sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE))?;
        let ptr = if num_bytes == 0 {
            NonNull::dangling()
        } else {
            device.bind_to_thread()?;
            let mut raw = std::ptr::null_mut();
            // SAFETY: the device's context is current and `raw` is a valid
            // out-pointer.
            let code = unsafe { sys::cuMemAllocHost_v2(&mut raw, num_bytes) };
            if code != sys::cudaError_enum_CUDA_SUCCESS {
                return Err(DriverError(code));
            }
            NonNull::new(raw.cast::<T>())
                .ok_or(DriverError(sys::cudaError_enum_CUDA_ERROR_OUT_OF_MEMORY))?
        };
        Ok(Self {
            ptr,
            len,
            device: device.clone(),
        })
    }
}

impl<T: DType> PinnedBuf<T> {
    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

impl<T: DType> Drop for PinnedBuf<T> {
    fn drop(&mut self) {
        if self.len * size_of::<T>() != 0 {
            let _ = self.device.bind_to_thread();
            // SAFETY: allocated by `cuMemAllocHost_v2` and freed exactly once.
            // Every copy using the buffer holds a reference until it completes.
            unsafe { sys::cuMemFreeHost(self.ptr.as_ptr().cast()) };
        }
    }
}

/// Copies a tensor's contents into `host`, consuming the tensor.
pub(crate) struct SpillToHost<T: DType> {
    pub tensor: Tensor<T>,
    pub host: PinnedBuf<T>,
}

impl<T: DType> DeviceOp for SpillToHost<T> {
    type Output = PinnedBuf<T>;

    unsafe fn execute(self, ctx: &ExecutionContext) -> Result<PinnedBuf<T>, DeviceError> {
        let len = self.tensor.size();
        if len != self.host.len {
            return Err(DeviceError::Internal(
                "host buffer and tensor differ in size".into(),
            ));
        }
        let src = self.tensor.device_pointer().cu_deviceptr();
        // The device memory must outlive the copy; the host buffer is the
        // output, which the future holds until the copy has completed.
        ctx.retain(self.tensor)?;
        if len > 0 {
            // SAFETY: both sides hold `len` elements and stay alive until the
            // submission completes.
            unsafe { memcpy_dtoh_async(self.host.ptr.as_ptr(), src, len, ctx.get_cuda_stream())? };
        }
        Ok(self.host)
    }
}

/// Allocates a 1-D device tensor and fills it from `host`.
pub(crate) struct LoadFromHost<T: DType> {
    pub host: Arc<PinnedBuf<T>>,
}

impl<T: DType> DeviceOp for LoadFromHost<T> {
    type Output = Tensor<T>;

    unsafe fn execute(self, ctx: &ExecutionContext) -> Result<Tensor<T>, DeviceError> {
        let len = self.host.len;
        if len == 0 {
            return Err(DeviceError::Internal(
                "cannot reload an empty tensor".into(),
            ));
        }
        // SAFETY: the copy enqueued below initializes every element before the
        // submission completes, which is the earliest anyone can read it.
        let tensor = unsafe { Tensor::<T>::uninitialized(len).execute(ctx)?.assume_init() };
        let dst = tensor.device_pointer().cu_deviceptr();
        let src = self.host.ptr.as_ptr().cast_const();
        ctx.retain(self.host)?;
        // SAFETY: both sides hold `len` elements; the host buffer is retained
        // by the submission and the tensor by the future's result.
        unsafe { memcpy_htod_async(dst, src, len, ctx.get_cuda_stream())? };
        Ok(tensor)
    }
}

macro_rules! impl_into_future {
    ($op:ident, $out:ty) => {
        impl<T: DType> IntoFuture for $op<T> {
            type Output = Result<$out, DeviceError>;
            type IntoFuture = DeviceFuture<$out, $op<T>>;
            fn into_future(self) -> Self::IntoFuture {
                match with_default_device_policy(|policy| {
                    let stream = policy.next_stream()?;
                    Ok(DeviceFuture::scheduled(self, ExecutionContext::new(stream)))
                }) {
                    Ok(Ok(future)) => future,
                    Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
                }
            }
        }
    };
}
impl_into_future!(SpillToHost, PinnedBuf<T>);
impl_into_future!(LoadFromHost, Tensor<T>);

/// A reload that has been submitted to the copy stream but not waited for.
///
/// Dropping it waits for the copy and frees the device memory.
pub(crate) struct PendingLoad<T: DType>(DeviceFuture<Tensor<T>, LoadFromHost<T>>);

pub(crate) enum Started<T: DType> {
    /// The copy finished before the submission returned.
    Done(Tensor<T>),
    Pending(PendingLoad<T>),
}

/// Submits a reload of `host` on `stream` and returns without waiting.
pub(crate) fn start_load<T: DType>(
    host: &Arc<PinnedBuf<T>>,
    stream: &Arc<Stream>,
) -> Result<Started<T>, DeviceError> {
    let op = LoadFromHost { host: host.clone() };
    let mut pending = PendingLoad(DeviceFuture::scheduled(
        op,
        ExecutionContext::new(stream.clone()),
    ));
    match pending.poll() {
        Poll::Ready(result) => result.map(Started::Done),
        Poll::Pending => Ok(Started::Pending(pending)),
    }
}

impl<T: DType> PendingLoad<T> {
    /// The first poll submits the work. Later polls check the stream itself,
    /// so no waker is needed.
    fn poll(&mut self) -> Poll<Result<Tensor<T>, DeviceError>> {
        Pin::new(&mut self.0).poll(&mut Context::from_waker(Waker::noop()))
    }

    /// Blocks until the copy has finished.
    pub(crate) fn wait(mut self, stream: &Arc<Stream>) -> Result<Tensor<T>, DeviceError> {
        loop {
            // SAFETY: the stream is alive and its context is current on this
            // thread, which submitted the copy.
            unsafe { stream.synchronize()? };
            if let Poll::Ready(result) = self.poll() {
                return result;
            }
            std::thread::yield_now();
        }
    }
}
