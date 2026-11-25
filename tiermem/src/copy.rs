//! Host-device migration: the copy operations that run on the manager's
//! copy streams, between device tensors and page-locked host blocks (see
//! [`crate::host`]).
//!
//! Page-locked ("pinned") memory lets the driver DMA straight to and from the
//! buffer, and a copy to or from it really is asynchronous. A copy involving
//! pageable memory is staged by the driver before the call returns, so it can
//! never overlap with compute.

use crate::dtype::{DTypeOps, Erased};
use crate::host::HostBlock;
use cutile::cuda_async::device_context::with_default_device_policy;
use cutile::cuda_async::device_future::DeviceFuture;
use cutile::cuda_core::{Stream, memcpy_dtoh_async, memcpy_htod_async, sys};
use cutile::prelude::*;
use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

/// Copies a tensor's bytes into `host`. The tensor is only read, and is kept
/// alive until the copy has completed. A copy needs no element type, only
/// the bytes.
pub(crate) struct SaveToHost {
    pub keep: Arc<Erased>,
    pub src: sys::CUdeviceptr,
    pub bytes: usize,
    pub host: HostBlock,
}

impl DeviceOp for SaveToHost {
    type Output = HostBlock;

    unsafe fn execute(self, ctx: &ExecutionContext) -> Result<HostBlock, DeviceError> {
        if self.bytes != self.host.bytes() {
            return Err(DeviceError::Internal(
                "host buffer and tensor differ in size".into(),
            ));
        }
        // The device memory must outlive the copy; the host block is the
        // output, which the future holds until the copy has completed.
        ctx.retain(self.keep)?;
        if self.bytes > 0 {
            // SAFETY: both sides hold `bytes` bytes and stay alive until the
            // submission completes.
            unsafe {
                memcpy_dtoh_async(
                    self.host.as_ptr::<u8>(),
                    self.src,
                    self.bytes,
                    ctx.get_cuda_stream(),
                )?
            };
        }
        Ok(self.host)
    }
}

/// Allocates a 1-D device tensor and fills it from `host`. The output is
/// erased; the manager gives it its shape (`DTypeOps::reshape`).
pub(crate) struct LoadFromHost {
    pub host: Arc<HostBlock>,
    pub ops: DTypeOps,
}

impl DeviceOp for LoadFromHost {
    type Output = Box<Erased>;

    unsafe fn execute(self, ctx: &ExecutionContext) -> Result<Box<Erased>, DeviceError> {
        let bytes = self.host.bytes();
        let elements = element_count(bytes, self.ops.elem_bytes)?;
        // SAFETY: the copy enqueued below writes every element of the tensor,
        // on the same stream, before the submission completes, which is the
        // earliest anyone can read it.
        let tensor = unsafe { (self.ops.alloc)(ctx, elements)? };
        let dst = (self.ops.device_ptr)(&*tensor);
        let src = self.host.as_ptr::<u8>().cast_const();
        ctx.retain(self.host)?;
        // SAFETY: the tensor holds exactly `elements * elem_bytes == bytes`
        // bytes (checked by `element_count`), and the host block holds
        // `bytes` bytes. The host block is retained by the submission and the
        // tensor by the future's result.
        unsafe { memcpy_htod_async(dst, src, bytes, ctx.get_cuda_stream())? };
        Ok(tensor)
    }
}

/// The number of elements in a host block of `bytes` bytes. The copy moves
/// exactly `bytes` bytes, so the block must be a whole, non-zero number of
/// elements; otherwise the copy would write past the end of the tensor.
fn element_count(bytes: usize, elem_bytes: usize) -> Result<usize, DeviceError> {
    if bytes == 0 {
        return Err(DeviceError::Internal(
            "cannot reload an empty tensor".into(),
        ));
    }
    if !bytes.is_multiple_of(elem_bytes) {
        return Err(DeviceError::Internal(
            "host block is not a whole number of elements".into(),
        ));
    }
    Ok(bytes / elem_bytes)
}

macro_rules! impl_into_future {
    ($op:ident, $out:ty) => {
        impl IntoFuture for $op {
            type Output = Result<$out, DeviceError>;
            type IntoFuture = DeviceFuture<$out, $op>;
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
impl_into_future!(SaveToHost, HostBlock);
impl_into_future!(LoadFromHost, Box<Erased>);

/// A copy that has been submitted to a stream but not waited for.
///
/// Dropping it waits for the copy and then drops the result.
pub(crate) struct Pending<Op: DeviceOp>(DeviceFuture<<Op as DeviceOp>::Output, Op>);

pub(crate) type PendingLoad = Pending<LoadFromHost>;
pub(crate) type PendingSave = Pending<SaveToHost>;

pub(crate) enum Started<Op: DeviceOp> {
    /// The copy finished before the submission returned.
    Done(<Op as DeviceOp>::Output),
    Pending(Pending<Op>),
}

/// Submits `op` on `stream` and returns without waiting.
pub(crate) fn start<Op: DeviceOp>(
    op: Op,
    stream: &Arc<Stream>,
) -> Result<Started<Op>, DeviceError> {
    let mut pending = Pending(DeviceFuture::scheduled(
        op,
        ExecutionContext::new(stream.clone()),
    ));
    match pending.poll() {
        Poll::Ready(result) => result.map(Started::Done),
        Poll::Pending => Ok(Started::Pending(pending)),
    }
}

impl<Op: DeviceOp> Pending<Op> {
    /// The first poll submits the work. Later polls check the stream itself,
    /// so no waker is needed. Must not be called again once it has returned
    /// `Ready`.
    pub(crate) fn poll(&mut self) -> Poll<Result<<Op as DeviceOp>::Output, DeviceError>> {
        Pin::new(&mut self.0).poll(&mut Context::from_waker(Waker::noop()))
    }

    /// Blocks until the copy has finished. `stream` is the one it runs on.
    pub(crate) fn wait(
        mut self,
        stream: &Arc<Stream>,
    ) -> Result<<Op as DeviceOp>::Output, DeviceError> {
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

#[cfg(test)]
mod tests {
    use super::element_count;

    #[test]
    fn whole_elements_are_counted() {
        assert_eq!(element_count(8, 4).unwrap(), 2);
        assert_eq!(element_count(2, 2).unwrap(), 1);
    }

    #[test]
    fn empty_and_partial_blocks_are_refused() {
        assert!(element_count(0, 4).is_err());
        assert!(element_count(6, 4).is_err());
        assert!(element_count(1, 2).is_err());
    }
}
