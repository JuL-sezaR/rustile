//! Host-device migration: the copy operations that run on the manager's
//! dedicated copy stream, between device tensors and page-locked host blocks
//! (see [`crate::host`]).
//!
//! Page-locked ("pinned") memory lets the driver DMA straight to and from the
//! buffer, and a host-to-device copy from it really is asynchronous. A copy
//! from pageable memory is staged by the driver before the call returns, so it
//! can never overlap with compute.

use crate::host::HostBlock;
use cutile::cuda_async::device_context::with_default_device_policy;
use cutile::cuda_async::device_future::DeviceFuture;
use cutile::cuda_core::{Stream, memcpy_dtoh_async, memcpy_htod_async};
use cutile::prelude::*;
use std::future::{Future, IntoFuture};
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

/// Copies a tensor's contents into `host`, consuming the tensor.
pub(crate) struct SpillToHost<T: DType> {
    pub tensor: Tensor<T>,
    pub host: HostBlock,
}

impl<T: DType> DeviceOp for SpillToHost<T> {
    type Output = HostBlock;

    unsafe fn execute(self, ctx: &ExecutionContext) -> Result<HostBlock, DeviceError> {
        let len = self.tensor.size();
        if len * size_of::<T>() != self.host.bytes() {
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
            unsafe { memcpy_dtoh_async(self.host.as_ptr::<T>(), src, len, ctx.get_cuda_stream())? };
        }
        Ok(self.host)
    }
}

/// Allocates a 1-D device tensor and fills it from `host`.
pub(crate) struct LoadFromHost<T: DType> {
    pub host: Arc<HostBlock>,
    pub dtype: PhantomData<T>,
}

impl<T: DType> DeviceOp for LoadFromHost<T> {
    type Output = Tensor<T>;

    unsafe fn execute(self, ctx: &ExecutionContext) -> Result<Tensor<T>, DeviceError> {
        let len = self.host.bytes() / size_of::<T>();
        if len == 0 {
            return Err(DeviceError::Internal(
                "cannot reload an empty tensor".into(),
            ));
        }
        // SAFETY: the copy enqueued below initializes every element before the
        // submission completes, which is the earliest anyone can read it.
        let tensor = unsafe { Tensor::<T>::uninitialized(len).execute(ctx)?.assume_init() };
        let dst = tensor.device_pointer().cu_deviceptr();
        let src = self.host.as_ptr::<T>().cast_const();
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
impl_into_future!(SpillToHost, HostBlock);
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
    host: &Arc<HostBlock>,
    stream: &Arc<Stream>,
) -> Result<Started<T>, DeviceError> {
    let op = LoadFromHost {
        host: host.clone(),
        dtype: PhantomData,
    };
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
