//! Page-locked host memory for spilled contents.
//!
//! Page-locking is the slow part of a first spill (about 0.3 ms per MiB on
//! the development machine), so memory that has been locked is kept: a block
//! that is given back goes on a free list and serves a later spill. The pool
//! either grows one chunk per spill ([`HostAlloc::Lazy`](crate::HostAlloc)),
//! or is locked once as a single arena
//! ([`HostAlloc::Eager`](crate::HostAlloc)).
//!
//! The pool hands out bytes, not elements, so tensors of every element type
//! share it.

use cutile::cuda_core::{Device, DriverError, sys};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// Blocks start on this boundary, which suits every element type and keeps
/// neighbours off each other's cache lines.
const ALIGN: usize = 64;

/// The free byte ranges of one chunk as `(offset, len)`, sorted by offset.
/// Neighbouring ranges are merged, so a chunk with nothing in use is one
/// range.
#[derive(Debug, PartialEq)]
struct FreeList(Vec<(usize, usize)>);

impl FreeList {
    fn whole(len: usize) -> Self {
        Self(vec![(0, len)])
    }

    /// The smallest free range that holds `len` bytes: its index and length.
    fn best_fit(&self, len: usize) -> Option<(usize, usize)> {
        self.0
            .iter()
            .enumerate()
            .filter(|&(_, &(_, free))| free >= len)
            .min_by_key(|&(_, &(_, free))| free)
            .map(|(index, &(_, free))| (index, free))
    }

    /// Takes `len` bytes, rounded up to [`ALIGN`], from the front of range
    /// `index`. Returns the offset and the number of bytes actually taken.
    fn take(&mut self, index: usize, len: usize) -> (usize, usize) {
        let (offset, free) = self.0[index];
        let taken = len.next_multiple_of(ALIGN).min(free);
        if taken == free {
            self.0.remove(index);
        } else {
            self.0[index] = (offset + taken, free - taken);
        }
        (offset, taken)
    }

    fn give(&mut self, offset: usize, len: usize) {
        let at = self.0.partition_point(|&(o, _)| o < offset);
        self.0.insert(at, (offset, len));
        if at + 1 < self.0.len() && offset + len == self.0[at + 1].0 {
            self.0[at].1 += self.0[at + 1].1;
            self.0.remove(at + 1);
        }
        if at > 0 && self.0[at - 1].0 + self.0[at - 1].1 == offset {
            self.0[at - 1].1 += self.0[at].1;
            self.0.remove(at);
        }
    }

    fn is_whole(&self, len: usize) -> bool {
        self.0 == [(0, len)]
    }
}

/// One page-locked allocation.
struct Chunk {
    id: u64,
    ptr: NonNull<u8>,
    bytes: usize,
    free: FreeList,
}

struct Inner {
    limit: usize,
    chunks: Vec<Chunk>,
    next_id: u64,
    pinned_bytes: usize,
    pin_time: Duration,
    /// The device whose context the chunks were allocated in.
    device: Option<Arc<Device>>,
}

// SAFETY: the chunks are plain host memory, and the pointers into them are
// only used while the lock is held or through a `HostBlock`, which owns its
// range exclusively.
unsafe impl Send for Inner {}

impl Inner {
    fn pin(&mut self, device: &Arc<Device>, bytes: usize) -> Result<&mut Chunk, DriverError> {
        let start = Instant::now();
        device.bind_to_thread()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: the device's context is current and `raw` is a valid
        // out-pointer.
        let code = unsafe { sys::cuMemAllocHost_v2(&mut raw, bytes) };
        if code != sys::cudaError_enum_CUDA_SUCCESS {
            return Err(DriverError(code));
        }
        let ptr = NonNull::new(raw.cast::<u8>())
            .ok_or(DriverError(sys::cudaError_enum_CUDA_ERROR_OUT_OF_MEMORY))?;
        self.pin_time += start.elapsed();
        self.pinned_bytes += bytes;
        self.device.get_or_insert_with(|| device.clone());
        let id = self.next_id;
        self.next_id += 1;
        self.chunks.push(Chunk {
            id,
            ptr,
            bytes,
            free: FreeList::whole(bytes),
        });
        Ok(self.chunks.last_mut().unwrap())
    }

    fn unpin(&mut self, index: usize) {
        let chunk = self.chunks.swap_remove(index);
        self.pinned_bytes -= chunk.bytes;
        if let Some(device) = &self.device {
            let _ = device.bind_to_thread();
        }
        // SAFETY: allocated by `cuMemAllocHost_v2` and freed exactly once.
        // The chunk has no block in use, so no copy can still be reading or
        // writing it.
        unsafe { sys::cuMemFreeHost(chunk.ptr.as_ptr().cast()) };
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        // Every block holds a reference to the pool, so none is left.
        while !self.chunks.is_empty() {
            self.unpin(0);
        }
    }
}

/// A pool of page-locked host memory, limited to `limit` bytes.
pub(crate) struct HostPool(Arc<Mutex<Inner>>);

fn lock(inner: &Mutex<Inner>) -> MutexGuard<'_, Inner> {
    inner
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl HostPool {
    pub(crate) fn new(limit: usize) -> Self {
        Self(Arc::new(Mutex::new(Inner {
            limit,
            chunks: Vec::new(),
            next_id: 0,
            pinned_bytes: 0,
            pin_time: Duration::ZERO,
            device: None,
        })))
    }

    /// Page-locks one chunk of `bytes` now.
    pub(crate) fn reserve(&self, device: &Arc<Device>, bytes: usize) -> Result<(), DriverError> {
        lock(&self.0).pin(device, bytes).map(|_| ())
    }

    /// A block of `bytes` bytes, or `None` if no free range holds it and the
    /// pool may not, or cannot within its limit, lock more memory.
    ///
    /// With `grow`, a new chunk of exactly `bytes` is locked when nothing
    /// free fits. Unused chunks are unlocked first if the limit requires it.
    pub(crate) fn alloc(
        &self,
        device: &Arc<Device>,
        bytes: usize,
        grow: bool,
    ) -> Result<Option<HostBlock>, DriverError> {
        let block = |ptr, home| HostBlock {
            ptr,
            bytes,
            home,
            pool: self.0.clone(),
        };
        if bytes == 0 {
            return Ok(Some(block(NonNull::dangling(), None)));
        }
        let mut inner = lock(&self.0);
        let fit = inner
            .chunks
            .iter()
            .enumerate()
            .filter_map(|(c, chunk)| chunk.free.best_fit(bytes).map(|(r, free)| (free, c, r)))
            .min();
        let (chunk, range) = match fit {
            Some((_, c, r)) => (&mut inner.chunks[c], r),
            None if !grow => return Ok(None),
            None => {
                while inner.pinned_bytes.saturating_add(bytes) > inner.limit {
                    let unused = inner
                        .chunks
                        .iter()
                        .position(|chunk| chunk.free.is_whole(chunk.bytes));
                    match unused {
                        Some(index) => inner.unpin(index),
                        None => return Ok(None),
                    }
                }
                (inner.pin(device, bytes)?, 0)
            }
        };
        let (offset, taken) = chunk.free.take(range, bytes);
        // SAFETY: `offset + taken` lies within the chunk.
        let ptr = unsafe { chunk.ptr.add(offset) };
        Ok(Some(block(ptr, Some((chunk.id, offset, taken)))))
    }

    /// Bytes page-locked, in use or not.
    pub(crate) fn pinned_bytes(&self) -> usize {
        lock(&self.0).pinned_bytes
    }

    /// Total time spent page-locking.
    pub(crate) fn pin_time(&self) -> Duration {
        lock(&self.0).pin_time
    }
}

/// An exclusively owned range of page-locked memory holding the contents of
/// one tensor. Dropping it gives the range back to the pool.
///
/// The memory starts out uninitialized; a block is only made reachable as a
/// tensor's host copy once the device has filled it.
pub(crate) struct HostBlock {
    ptr: NonNull<u8>,
    bytes: usize,
    /// Chunk id, offset and length of the range; `None` for an empty block.
    home: Option<(u64, usize, usize)>,
    pool: Arc<Mutex<Inner>>,
}

// SAFETY: plain host memory owned by this block; after it has been filled it
// is only read.
unsafe impl Send for HostBlock {}
unsafe impl Sync for HostBlock {}

impl HostBlock {
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    /// The start of the block, aligned for any element type.
    pub(crate) fn as_ptr<T>(&self) -> *mut T {
        self.ptr.as_ptr().cast()
    }
}

impl Drop for HostBlock {
    fn drop(&mut self) {
        let Some((id, offset, len)) = self.home else {
            return;
        };
        let mut inner = lock(&self.pool);
        if let Some(chunk) = inner.chunks.iter_mut().find(|chunk| chunk.id == id) {
            chunk.free.give(offset, len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn takes_are_rounded_up_to_the_alignment() {
        let mut free = FreeList::whole(1024);
        assert_eq!(free.take(0, 100), (0, 128));
        assert_eq!(free.take(0, 64), (128, 64));
        assert_eq!(free, FreeList(vec![(192, 832)]));
    }

    #[test]
    fn a_take_never_exceeds_its_range() {
        let mut free = FreeList::whole(100);
        assert_eq!(free.take(0, 100), (0, 100));
        assert_eq!(free, FreeList(vec![]));
    }

    #[test]
    fn best_fit_prefers_the_smallest_range_that_holds_the_request() {
        let free = FreeList(vec![(0, 512), (1024, 128), (2048, 256)]);
        assert_eq!(free.best_fit(128), Some((1, 128)));
        assert_eq!(free.best_fit(200), Some((2, 256)));
        assert_eq!(free.best_fit(1024), None);
    }

    #[test]
    fn giving_back_merges_with_both_neighbours() {
        let mut free = FreeList::whole(384);
        let a = free.take(0, 128);
        let b = free.take(0, 128);
        let c = free.take(0, 128);
        free.give(a.0, a.1);
        free.give(c.0, c.1);
        assert_eq!(free, FreeList(vec![(0, 128), (256, 128)]));
        assert!(!free.is_whole(384));
        free.give(b.0, b.1);
        assert!(free.is_whole(384));
    }
}
