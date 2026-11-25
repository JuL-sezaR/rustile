//! Workload-aware GPU memory tiering.
//!
//! A [`Manager`] owns a set of registered device tensors, of any element type,
//! and keeps their total size in bytes under a VRAM budget. Each tensor carries
//! a [`Hint`] saying how costly it is to evict. When an admission or reload would exceed the budget, the
//! manager evicts lower-priority tensors first (spilling them to host RAM, or
//! dropping them if they can be rebuilt) instead of failing.
//!
//! The budget is *logical*: it is the manager's own accounting of the bytes it
//! has admitted, not a reading of the driver's free-memory counter. That makes
//! behaviour reproducible on any GPU and lets a 16 GB card stand in for a much
//! smaller one.
//!
//! Spilled contents live in page-locked host memory, which is locked either
//! as it is first needed or up front (see [`HostAlloc`]). They move on CUDA
//! streams of the manager's own, one for each direction, so a reload started
//! with [`Manager::prefetch`] and a copy to the host started with
//! [`Manager::write_back`] (or both automatically, see
//! [`Config::prefetch_depth`]) overlap with kernels running on other streams
//! and with each other.

mod copy;
mod dtype;
mod error;
mod host;
mod trace;

pub use error::{EntrySnapshot, Snapshot, TierError};
pub use trace::{Event, EventKind, Trace};

use copy::{LoadFromHost, PendingLoad, PendingSave, SaveToHost, Started};
use cutile::DTypeId;
use cutile::cuda_async::device_context::{get_default_device, with_device};
use cutile::cuda_core::{DriverError, Stream, sys};
use cutile::prelude::*;
use dtype::{DTypeOps, Erased, ops};
use host::{HostBlock, HostPool};
use std::any::TypeId;
use std::cmp::Reverse;
use std::collections::{HashMap, VecDeque};
use std::task::Poll;
use std::time::{Duration, Instant};

/// How expensive a tensor is to move off the device, in eviction order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Hint {
    /// Never evicted. Admission fails if pinned tensors alone exceed the budget.
    Pinned,
    /// Latency-critical; evicted only after every `Evictable` tensor is gone.
    Hot,
    /// Safe to evict whenever room is needed.
    Evictable,
}

impl Hint {
    /// Lower ranks are evicted first; `None` means never.
    fn eviction_rank(self) -> Option<u8> {
        match self {
            Hint::Evictable => Some(0),
            Hint::Hot => Some(1),
            Hint::Pinned => None,
        }
    }
}

/// What happens to a tensor's contents when it is evicted.
pub enum Backing<T: DType> {
    /// Copy the contents to host RAM and copy them back on reload.
    Spill,
    /// Drop the contents and call this to rebuild them on reload.
    Reconstruct(Box<dyn Fn() -> Result<Tensor<T>, Error>>),
}

/// [`Backing`] with the element type erased.
enum ErasedBacking {
    Spill,
    Reconstruct(Box<dyn Fn() -> Result<Arc<Erased>, Error>>),
}

/// What the manager does when room is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Never evict; return [`TierError::BudgetExceeded`]. This is the baseline
    /// that behaves like a GPU with `budget_bytes` of memory.
    Strict,
    /// Evict by hint rank, then least-recently-used.
    EvictLru,
    /// Evict by hint rank, then by the access order declared with
    /// [`Manager::expect_accesses`]: tensors with no declared use go first
    /// (least-recently-used among them), then the one needed farthest in the
    /// future. With nothing declared this is the same as `EvictLru`.
    EvictLookahead,
}

/// When the host memory that spilled contents live in is page-locked.
/// Locking is slow (about 0.3 ms per MiB on the development machine), and
/// memory that has been locked is kept for reuse either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostAlloc {
    /// Lock a block the first time a spill finds no free one. The cost lands
    /// on the eviction that needs it.
    Lazy,
    /// Lock one arena of [`Config::host_limit_bytes`] when the first tensor
    /// backed by a spill is registered, and serve every spill from it. The
    /// cost is paid once, during setup, and the arena never grows.
    Eager,
}

#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Hard limit on bytes resident on the device.
    pub budget_bytes: usize,
    /// Rising above this fires the pressure callback, before anything fails.
    pub soft_limit_bytes: usize,
    /// After firing, the pressure callback stays quiet until residency has
    /// fallen to this level or below. Setting it equal to the soft limit
    /// fires on every crossing.
    pub pressure_rearm_bytes: usize,
    /// Limit on the page-locked host RAM held for spilled contents and clean
    /// copies. When a spill would exceed it, clean copies of resident tensors
    /// are freed first; if that is not enough the eviction fails with
    /// [`TierError::HostBudgetExceeded`].
    pub host_limit_bytes: usize,
    /// When that memory is page-locked. [`HostAlloc::Eager`] needs a host
    /// limit, which is the size of its arena.
    pub host_alloc: HostAlloc,
    /// After each [`Manager::ensure_resident`], start reloading up to this
    /// many of the next tensors declared with [`Manager::expect_accesses`].
    /// A prefetch never evicts a tensor that is declared to be used sooner
    /// than the one being fetched, and never waits for a copy to the host:
    /// it evicts only tensors that already have a host copy or can be
    /// rebuilt. For the others it starts the copy (see
    /// [`Manager::write_back`]), so that they can make room for the prefetch
    /// after it. Zero disables all of this.
    pub prefetch_depth: usize,
    /// How many streams copies run on: 2 gives each direction its own, so a
    /// reload and a copy to the host can be in flight together; 1 runs every
    /// copy one after another on a single stream.
    pub copy_streams: usize,
    pub policy: Policy,
}

impl Config {
    /// A config with the soft limit at 80% of the budget, pressure re-arming
    /// at 60%, no host limit, host memory locked as it is needed, and no
    /// automatic prefetch, and a copy stream for each direction.
    pub fn new(budget_bytes: usize, policy: Policy) -> Self {
        Self {
            budget_bytes,
            soft_limit_bytes: budget_bytes / 5 * 4,
            pressure_rearm_bytes: budget_bytes / 5 * 3,
            host_limit_bytes: usize::MAX,
            host_alloc: HostAlloc::Lazy,
            prefetch_depth: 0,
            copy_streams: 2,
            policy,
        }
    }
}

/// Passed to the pressure callback when residency crosses the soft limit.
#[derive(Debug, Clone, Copy)]
pub struct Pressure {
    pub resident_bytes: usize,
    pub soft_limit_bytes: usize,
    pub budget_bytes: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    /// Bytes of device memory held, including reloads still in flight.
    pub resident_bytes: usize,
    /// Bytes of evicted tensors whose contents live in host RAM.
    pub spilled_bytes: usize,
    /// Host RAM holding contents: spilled tensors plus clean copies of
    /// resident ones.
    pub host_bytes: usize,
    /// Host RAM page-locked, whether or not it holds contents right now.
    pub pinned_bytes: usize,
    /// Total time spent page-locking host memory.
    pub pin_time: Duration,
    pub peak_resident_bytes: usize,
    pub evictions: u64,
    /// Evictions that reused a clean host copy instead of copying to the host.
    pub clean_evictions: u64,
    /// Clean copies freed to stay under the host limit.
    pub clean_copies_dropped: u64,
    pub reloads: u64,
    /// Reloads started ahead of use on the copy stream.
    pub prefetches: u64,
    /// Prefetched tensors evicted again before they were used.
    pub prefetches_cancelled: u64,
    pub reconstructions: u64,
    /// Copies to the host started ahead of eviction, by
    /// [`Manager::write_back`] or alongside an automatic prefetch.
    pub write_backs: u64,
    pub bytes_to_host: u64,
    pub bytes_to_device: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(u64);

enum Location {
    /// Always an `Arc<Tensor<T>>` of the entry's element type.
    Device(Arc<Erased>),
    /// A prefetch is copying the contents back; the device memory is already
    /// allocated and counted, and `clean_copy` holds the source.
    Loading(Box<PendingLoad>),
    Host(Arc<HostBlock>),
    /// Evicted with [`Backing::Reconstruct`]; nothing is kept.
    Dropped,
}

struct Entry {
    hint: Hint,
    backing: ErasedBacking,
    /// What depends on the element type, made when the tensor was registered.
    ops: DTypeOps,
    bytes: usize,
    shape: Vec<usize>,
    location: Location,
    /// For a resident tensor reloaded from the host: the contents it was
    /// reloaded from, still valid because nothing has written to it since.
    /// Evicting it again reuses this instead of copying back to the host.
    clean_copy: Option<Arc<HostBlock>>,
    /// For a resident tensor without a clean copy: a copy to the host that
    /// is in flight. Once it has finished, its block becomes the clean copy.
    /// Until then the tensor can be read but not written or freed.
    save: Option<PendingSave>,
    last_use: u64,
}

impl Entry {
    fn is_resident(&self) -> bool {
        matches!(self.location, Location::Device(_))
    }

    /// Holds device memory: resident, or being reloaded.
    fn on_device(&self) -> bool {
        matches!(self.location, Location::Device(_) | Location::Loading(_))
    }
}

/// Whether room could be made for an admission or reload.
enum Room {
    Made,
    Blocked { could_evict: bool },
}

enum Fetch {
    Started,
    /// Already on the device, or not backed by a host copy.
    Skipped,
    NoRoom,
}

#[derive(PartialEq)]
enum Evicted {
    Copied,
    Clean,
    Cancelled,
    Dropped,
}

fn dev(e: impl Into<Error>) -> TierError {
    TierError::Device(e.into())
}

/// How many copies the default device can run at once, as the driver reports
/// it (`CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT`). With 1, a copy to the host
/// and a copy to the device take turns even on separate streams.
pub fn async_engine_count() -> Result<i32, TierError> {
    with_device(get_default_device(), |device| {
        let mut count = 0;
        // SAFETY: `count` is a valid out-pointer and the device handle
        // belongs to a live `Device`.
        let code = unsafe {
            sys::cuDeviceGetAttribute(
                &mut count,
                sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,
                device.cu_device(),
            )
        };
        if code == sys::cudaError_enum_CUDA_SUCCESS {
            Ok(count)
        } else {
            Err(DriverError(code))
        }
    })
    .map_err(dev)?
    .map_err(dev)
}

pub struct Manager {
    config: Config,
    entries: HashMap<TensorId, Entry>,
    next_id: u64,
    clock: u64,
    /// Declared future accesses, next first; consumed by `ensure_resident`.
    expected: VecDeque<TensorId>,
    /// The arguments of the last `ensure_resident`; prefetching spares them.
    in_use: Vec<TensorId>,
    /// The streams for copies to the device and to the host. Created on
    /// first use, so a manager that never migrates needs no stream.
    stream_to_device: Option<Arc<Stream>>,
    stream_to_host: Option<Arc<Stream>>,
    host_pool: HostPool,
    /// The eager arena has been page-locked.
    host_reserved: bool,
    stats: Stats,
    pressure_armed: bool,
    on_pressure: Option<Box<dyn FnMut(Pressure)>>,
    trace: Trace,
}

impl Manager {
    pub fn new(config: Config) -> Self {
        assert!(
            config.soft_limit_bytes <= config.budget_bytes,
            "soft limit must not exceed the budget"
        );
        assert!(
            config.pressure_rearm_bytes <= config.soft_limit_bytes,
            "pressure re-arm level must not exceed the soft limit"
        );
        assert!(
            config.host_alloc == HostAlloc::Lazy || config.host_limit_bytes != usize::MAX,
            "eager host allocation needs a host limit to size the arena"
        );
        assert!(
            (1..=2).contains(&config.copy_streams),
            "a manager has one or two copy streams"
        );
        Self {
            config,
            entries: HashMap::new(),
            next_id: 0,
            clock: 0,
            expected: VecDeque::new(),
            in_use: Vec::new(),
            stream_to_device: None,
            stream_to_host: None,
            host_pool: HostPool::new(config.host_limit_bytes),
            host_reserved: false,
            stats: Stats::default(),
            pressure_armed: true,
            on_pressure: None,
            trace: Trace::new(Instant::now()),
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Called when residency rises above the soft limit, and not again until
    /// it has fallen to [`Config::pressure_rearm_bytes`] and risen once more.
    pub fn on_pressure(&mut self, callback: impl FnMut(Pressure) + 'static) {
        self.on_pressure = Some(Box::new(callback));
    }

    /// Admits a device tensor whose contents are spilled to host RAM on eviction.
    ///
    /// On failure the tensor is dropped and its device memory freed.
    pub fn register<T: DType>(
        &mut self,
        tensor: Tensor<T>,
        hint: Hint,
    ) -> Result<TensorId, TierError> {
        self.register_with(tensor, hint, Backing::Spill)
    }

    /// Admits a device tensor that `rebuild` can recreate, so eviction costs
    /// no host memory or device-to-host copy.
    pub fn register_reconstructible<T: DType>(
        &mut self,
        tensor: Tensor<T>,
        hint: Hint,
        rebuild: impl Fn() -> Result<Tensor<T>, Error> + 'static,
    ) -> Result<TensorId, TierError> {
        self.register_with(tensor, hint, Backing::Reconstruct(Box::new(rebuild)))
    }

    pub fn register_with<T: DType>(
        &mut self,
        tensor: Tensor<T>,
        hint: Hint,
        backing: Backing<T>,
    ) -> Result<TensorId, TierError> {
        let bytes = tensor.num_bytes();
        let start = self.trace.now();
        if matches!(backing, Backing::Spill) {
            self.reserve_host()?;
        }
        self.make_room(bytes, &[])?;
        let id = TensorId(self.next_id);
        self.next_id += 1;
        let shape = tensor.shape().iter().map(|&d| d as usize).collect();
        let backing = match backing {
            Backing::Spill => ErasedBacking::Spill,
            Backing::Reconstruct(rebuild) => ErasedBacking::Reconstruct(Box::new(move || {
                let rebuilt: Arc<Erased> = Arc::new(rebuild()?);
                Ok(rebuilt)
            })),
        };
        self.clock += 1;
        self.entries.insert(
            id,
            Entry {
                hint,
                backing,
                ops: ops::<T>(),
                bytes,
                shape,
                location: Location::Device(Arc::new(tensor)),
                clean_copy: None,
                save: None,
                last_use: self.clock,
            },
        );
        self.add_resident(bytes);
        self.trace.span(EventKind::Admit, Some(id), bytes, start);
        Ok(id)
    }

    /// Removes a tensor from management, returning it if it is on the device.
    /// Waits for a copy of it that is in flight.
    pub fn unregister<T: DType>(&mut self, id: TensorId) -> Result<Option<Tensor<T>>, TierError> {
        let entry = self.typed_entry::<T>(id)?;
        if matches!(entry.location, Location::Loading(_)) {
            self.finish_load(id)?;
        }
        self.finish_save(id)?;
        let entry = self.entries.remove(&id).unwrap();
        self.expected.retain(|&e| e != id);
        if entry.clean_copy.is_some() {
            self.stats.host_bytes -= entry.bytes;
        }
        match entry.location {
            Location::Device(t) => {
                self.sub_resident(entry.bytes);
                let t = Arc::downcast::<Tensor<T>>(t)
                    .unwrap_or_else(|_| unreachable!("the type was checked above"));
                Ok(Some(Arc::into_inner(t).expect("no copy holds the tensor")))
            }
            Location::Host(_) => {
                self.stats.spilled_bytes -= entry.bytes;
                self.stats.host_bytes -= entry.bytes;
                Ok(None)
            }
            Location::Dropped => Ok(None),
            Location::Loading(_) => unreachable!("the load was finished above"),
        }
    }

    /// Makes every tensor in `ids` resident, evicting others if needed.
    /// Afterwards, [`Manager::tensor`] returns each of them until the next
    /// `&mut self` call.
    pub fn ensure_resident(&mut self, ids: &[TensorId]) -> Result<(), TierError> {
        for &id in ids {
            let entry = self.entries.get(&id).ok_or(TierError::UnknownTensor(id))?;
            match entry.location {
                Location::Device(_) => {}
                Location::Loading(_) => self.finish_load(id)?,
                Location::Host(_) | Location::Dropped => {
                    let bytes = entry.bytes;
                    self.make_room(bytes, ids)?;
                    self.reload(id)?;
                }
            }
            self.clock += 1;
            self.entries.get_mut(&id).unwrap().last_use = self.clock;
            // Accesses the workload skipped are dropped along with this one.
            if let Some(pos) = self.expected.iter().position(|&e| e == id) {
                self.expected.drain(..=pos);
            }
        }
        self.in_use.clear();
        self.in_use.extend_from_slice(ids);
        if self.config.prefetch_depth > 0 {
            self.prefetch_expected(ids)?;
        }
        Ok(())
    }

    /// Starts copying spilled tensors back to the device without waiting, so
    /// the copies overlap with whatever runs next. A later
    /// [`Manager::ensure_resident`] waits only for what is left of the copy.
    ///
    /// Room is made as for `ensure_resident`, sparing `ids` and the arguments
    /// of the last `ensure_resident`. Tensors that are already on the device
    /// or are rebuilt rather than copied are skipped, and prefetching stops at
    /// the first tensor there is no room for. Returns how many reloads were
    /// started.
    pub fn prefetch(&mut self, ids: &[TensorId]) -> Result<usize, TierError> {
        let mut protect = self.in_use.clone();
        protect.extend_from_slice(ids);
        let mut started = 0;
        for &id in ids {
            match self.start_prefetch(id, &protect, None, true)? {
                Fetch::Started => started += 1,
                Fetch::Skipped => {}
                Fetch::NoRoom => break,
            }
        }
        Ok(started)
    }

    /// Starts copying modified resident tensors to the host without waiting,
    /// so that evicting them later only frees the device memory. The tensors
    /// stay resident and can be read while the copy runs.
    ///
    /// A tensor whose copy is in flight cannot be reused until the copy has
    /// finished: evicting it, unregistering it or asking for
    /// [`Manager::tensor_mut`] waits for what is left. Writing to it then
    /// discards the copy.
    ///
    /// Tensors that are not resident, already have a host copy, or are
    /// rebuilt rather than copied are skipped. Only host memory that nobody
    /// is using is taken: under a host limit, writing back stops at the first
    /// tensor there is no free block for, rather than give up another
    /// tensor's clean copy. Returns how many copies were started.
    pub fn write_back(&mut self, ids: &[TensorId]) -> Result<usize, TierError> {
        let mut started = 0;
        for &id in ids {
            if !self.entries.contains_key(&id) {
                return Err(TierError::UnknownTensor(id));
            }
            if self.needs_save(id) {
                match self.start_save(id, false) {
                    Ok(()) => {
                        self.stats.write_backs += 1;
                        started += 1;
                    }
                    Err(TierError::HostBudgetExceeded { .. }) => break,
                    Err(e) => return Err(e),
                }
            }
        }
        Ok(started)
    }

    /// Declares that `ids` will be passed to [`Manager::ensure_resident`] in
    /// this order, after any accesses declared earlier.
    /// [`Policy::EvictLookahead`] and [`Config::prefetch_depth`] use this.
    ///
    /// Each `ensure_resident` of a declared tensor consumes the declaration
    /// up to and including its first occurrence, so a stale plan corrects
    /// itself; [`Manager::clear_expected`] discards it outright.
    pub fn expect_accesses(&mut self, ids: &[TensorId]) -> Result<(), TierError> {
        if let Some(&id) = ids.iter().find(|id| !self.entries.contains_key(id)) {
            return Err(TierError::UnknownTensor(id));
        }
        self.expected.extend(ids);
        Ok(())
    }

    pub fn clear_expected(&mut self) {
        self.expected.clear();
    }

    /// The device tensor for `id`, which must be resident (see
    /// [`Manager::ensure_resident`]).
    pub fn tensor<T: DType>(&self, id: TensorId) -> Result<&Tensor<T>, TierError> {
        match &self.typed_entry::<T>(id)?.location {
            Location::Device(t) => Ok(t
                .downcast_ref::<Tensor<T>>()
                .expect("the type was checked above")),
            _ => Err(TierError::NotResident(id)),
        }
    }

    /// The element type `id` was registered with, wherever its contents are.
    pub fn dtype(&self, id: TensorId) -> Result<DTypeId, TierError> {
        Ok(self
            .entries
            .get(&id)
            .ok_or(TierError::UnknownTensor(id))?
            .ops
            .id)
    }

    /// The shape of `id`, wherever its contents are.
    pub fn shape(&self, id: TensorId) -> Result<&[usize], TierError> {
        Ok(&self
            .entries
            .get(&id)
            .ok_or(TierError::UnknownTensor(id))?
            .shape)
    }

    /// The entry for `id`, if it exists and holds a `T`. Nothing is changed.
    fn typed_entry<T: DType>(&self, id: TensorId) -> Result<&Entry, TierError> {
        let entry = self.entries.get(&id).ok_or(TierError::UnknownTensor(id))?;
        if entry.ops.type_id != TypeId::of::<T>() {
            return Err(TierError::WrongDType {
                id,
                stored: entry.ops.id,
                requested: T::DTYPE,
            });
        }
        Ok(entry)
    }

    /// Mutable access to the device tensor for `id`, which must be resident.
    ///
    /// Marks the tensor dirty: its clean host copy, if any, is freed and the
    /// next eviction copies the contents back to the host. If a copy to the
    /// host is in flight this waits for it first. Replacing the tensor with
    /// one of a different shape is a bug.
    ///
    /// Writes to a reconstructible tensor do not survive eviction: it is
    /// rebuilt from its original recipe.
    pub fn tensor_mut<T: DType>(&mut self, id: TensorId) -> Result<&mut Tensor<T>, TierError> {
        let entry = self.typed_entry::<T>(id)?;
        if !entry.is_resident() {
            return Err(TierError::NotResident(id));
        }
        self.finish_save(id)?;
        let entry = self.entries.get_mut(&id).unwrap();
        if entry.clean_copy.take().is_some() {
            self.stats.host_bytes -= entry.bytes;
        }
        let Location::Device(t) = &mut entry.location else {
            unreachable!("checked above");
        };
        Ok(Arc::get_mut(t)
            .expect("no copy holds the tensor")
            .downcast_mut::<Tensor<T>>()
            .expect("the type was checked above"))
    }

    pub fn is_resident(&self, id: TensorId) -> bool {
        self.entries.get(&id).is_some_and(Entry::is_resident)
    }

    pub fn stats(&self) -> Stats {
        Stats {
            pinned_bytes: self.host_pool.pinned_bytes(),
            pin_time: self.host_pool.pin_time(),
            ..self.stats
        }
    }

    pub fn trace(&self) -> &Trace {
        &self.trace
    }

    /// What is resident and where, for explaining a failure.
    pub fn snapshot(&self) -> Snapshot {
        let mut entries: Vec<EntrySnapshot> = self
            .entries
            .iter()
            .map(|(&id, e)| EntrySnapshot {
                id,
                hint: e.hint,
                dtype: e.ops.id,
                bytes: e.bytes,
                resident: e.on_device(),
                last_use: e.last_use,
            })
            .collect();
        entries.sort_by_key(|e| e.id);
        Snapshot {
            budget_bytes: self.config.budget_bytes,
            resident_bytes: self.stats.resident_bytes,
            entries,
        }
    }

    /// Evicts until `bytes` more fit in the budget, never touching `protect`.
    fn make_room(&mut self, bytes: usize, protect: &[TensorId]) -> Result<(), TierError> {
        match self.try_make_room(bytes, protect, None)? {
            Room::Made => Ok(()),
            Room::Blocked { could_evict } => {
                self.trace.instant(EventKind::AdmitFailed, None, bytes);
                Err(TierError::BudgetExceeded {
                    requested_bytes: bytes,
                    could_evict,
                    snapshot: self.snapshot(),
                })
            }
        }
    }

    /// With `keep_before`, a tensor whose next declared use is at or before
    /// that position in the plan is not evicted.
    fn try_make_room(
        &mut self,
        bytes: usize,
        protect: &[TensorId],
        keep_before: Option<usize>,
    ) -> Result<Room, TierError> {
        if bytes > self.config.budget_bytes {
            return Ok(Room::Blocked { could_evict: false });
        }
        while self.stats.resident_bytes + bytes > self.config.budget_bytes {
            let Some(&(victim, next_use)) = self.victims(protect).first() else {
                return Ok(Room::Blocked { could_evict: false });
            };
            if self.config.policy == Policy::Strict {
                return Ok(Room::Blocked { could_evict: true });
            }
            if keep_before.is_some_and(|pos| next_use <= pos) {
                return Ok(Room::Blocked { could_evict: true });
            }
            self.evict(victim)?;
        }
        Ok(Room::Made)
    }

    /// Like [`Self::try_make_room`], but never waits for a copy to the host:
    /// only tensors that can be evicted for free are, in the policy's order.
    fn try_make_room_now(
        &mut self,
        bytes: usize,
        protect: &[TensorId],
        keep_before: Option<usize>,
    ) -> Result<Room, TierError> {
        let budget = self.config.budget_bytes;
        if bytes > budget {
            return Ok(Room::Blocked { could_evict: false });
        }
        if self.stats.resident_bytes + bytes <= budget {
            return Ok(Room::Made);
        }
        if self.config.policy == Policy::Strict {
            return Ok(Room::Blocked { could_evict: true });
        }
        // Choose before evicting: a victim given up for a prefetch that
        // then cannot start would cost a reload and gain nothing.
        let needed = self.stats.resident_bytes + bytes - budget;
        let mut chosen = Vec::new();
        let mut freed = 0;
        for (victim, next_use) in self.victims(protect) {
            if freed >= needed || keep_before.is_some_and(|pos| next_use <= pos) {
                break;
            }
            if self.evicts_for_free(victim)? {
                freed += self.entries[&victim].bytes;
                chosen.push(victim);
            }
        }
        if freed < needed {
            return Ok(Room::Blocked { could_evict: true });
        }
        for victim in chosen {
            self.evict(victim)?;
        }
        Ok(Room::Made)
    }

    /// Starts copies to the host for the tensors the policy would evict next
    /// to make `bytes` of room, unless they can already be evicted for free.
    fn write_behind(
        &mut self,
        bytes: usize,
        protect: &[TensorId],
        keep_before: Option<usize>,
    ) -> Result<(), TierError> {
        let mut covered = 0;
        for (victim, next_use) in self.victims(protect) {
            if covered >= bytes || keep_before.is_some_and(|pos| next_use <= pos) {
                break;
            }
            if !self.entries[&victim].is_resident() {
                continue;
            }
            if self.needs_save(victim) {
                self.start_save(victim, false)?;
                self.stats.write_backs += 1;
            }
            covered += self.entries[&victim].bytes;
        }
        Ok(())
    }

    /// The tensors that may be evicted, first victim first, each with the
    /// position of its next declared use (`usize::MAX` if it has none).
    fn victims(&self, protect: &[TensorId]) -> Vec<(TensorId, usize)> {
        let mut next_use = HashMap::new();
        for (pos, &id) in self.expected.iter().enumerate() {
            next_use.entry(id).or_insert(pos);
        }
        let lookahead = self.config.policy == Policy::EvictLookahead;
        let mut victims: Vec<_> = self
            .entries
            .iter()
            .filter(|(id, e)| e.on_device() && !protect.contains(id))
            .filter_map(|(&id, e)| {
                let next = next_use.get(&id).copied().unwrap_or(usize::MAX);
                let order = if lookahead { next } else { usize::MAX };
                Some((
                    (e.hint.eviction_rank()?, Reverse(order), e.last_use),
                    (id, next),
                ))
            })
            .collect();
        victims.sort_by_key(|&(key, _)| key);
        victims.into_iter().map(|(_, victim)| victim).collect()
    }

    /// Whether a resident tensor has contents that exist nowhere else and no
    /// copy to the host under way.
    fn needs_save(&self, id: TensorId) -> bool {
        let e = &self.entries[&id];
        e.is_resident()
            && matches!(e.backing, ErasedBacking::Spill)
            && e.clean_copy.is_none()
            && e.save.is_none()
    }

    /// Whether evicting this tensor now would copy nothing and wait for
    /// nothing. Notices a copy to the host that has finished. A tensor whose
    /// reload is in flight is not free: evicting it waits for the reload.
    fn evicts_for_free(&mut self, id: TensorId) -> Result<bool, TierError> {
        if self.poll_save(id)? || matches!(self.entries[&id].location, Location::Loading(_)) {
            return Ok(false);
        }
        Ok(!self.needs_save(id))
    }

    fn new_stream() -> Result<Arc<Stream>, TierError> {
        with_device(get_default_device(), |device| device.new_stream())
            .map_err(dev)?
            .map_err(dev)
    }

    fn stream_to_device(&mut self) -> Result<Arc<Stream>, TierError> {
        if let Some(stream) = &self.stream_to_device {
            return Ok(stream.clone());
        }
        let stream = Self::new_stream()?;
        self.stream_to_device = Some(stream.clone());
        Ok(stream)
    }

    fn stream_to_host(&mut self) -> Result<Arc<Stream>, TierError> {
        if self.config.copy_streams == 1 {
            return self.stream_to_device();
        }
        if let Some(stream) = &self.stream_to_host {
            return Ok(stream.clone());
        }
        let stream = Self::new_stream()?;
        self.stream_to_host = Some(stream.clone());
        Ok(stream)
    }

    /// Page-locks the eager arena, once.
    fn reserve_host(&mut self) -> Result<(), TierError> {
        if self.config.host_alloc == HostAlloc::Eager && !self.host_reserved {
            let stream = self.stream_to_host()?;
            self.host_pool
                .reserve(stream.device(), self.config.host_limit_bytes)
                .map_err(dev)?;
            self.host_reserved = true;
        }
        Ok(())
    }

    /// A host block for `bytes` of contents. With `reclaim`, room is made if
    /// the pool has none; without it only memory nobody is using is taken,
    /// which is what a copy made ahead of need is allowed.
    fn alloc_host(
        &mut self,
        bytes: usize,
        stream: &Arc<Stream>,
        reclaim: bool,
    ) -> Result<HostBlock, TierError> {
        let grow = self.config.host_alloc == HostAlloc::Lazy;
        loop {
            let block = self.host_pool.alloc(stream.device(), bytes, grow);
            if let Some(block) = block.map_err(dev)? {
                return Ok(block);
            }
            if !(reclaim && self.reclaim_host(bytes)?) {
                return Err(TierError::HostBudgetExceeded {
                    requested_bytes: bytes,
                    host_bytes: self.stats.host_bytes,
                    host_limit_bytes: self.config.host_limit_bytes,
                });
            }
        }
    }

    /// Frees host memory for a spill of `bytes` that found no block. Returns
    /// false when there is nothing left to free.
    fn reclaim_host(&mut self, bytes: usize) -> Result<bool, TierError> {
        // A write-back that has finished holds a clean copy like any other.
        let saving: Vec<TensorId> = self
            .entries
            .iter()
            .filter(|(_, e)| e.save.is_some())
            .map(|(&id, _)| id)
            .collect();
        for &id in &saving {
            self.poll_save(id)?;
        }
        // The arena may have the room, only not in one piece.
        if self.host_pool.fragmented_free_bytes() >= bytes && self.compact_host()? {
            return Ok(true);
        }
        // Give up the least recently used clean copy of a resident tensor.
        let victim = self
            .entries
            .values_mut()
            .filter(|e| e.is_resident() && e.clean_copy.is_some())
            .min_by_key(|e| e.last_use);
        if let Some(entry) = victim {
            // Only an in-flight load shares a clean copy, and loading tensors
            // are not candidates, so this returns the block to the pool.
            entry.clean_copy = None;
            self.stats.host_bytes -= entry.bytes;
            self.stats.clean_copies_dropped += 1;
            return Ok(true);
        }
        // Wait for a write-back that is still running; the next round gives
        // up the clean copy it leaves.
        match saving.iter().find(|id| self.entries[id].save.is_some()) {
            Some(&id) => self.finish_save(id).map(|()| true),
            None => Ok(false),
        }
    }

    /// Moves the arena's blocks together. Blocks can only move while no copy
    /// reads or writes them, so every copy in flight is waited for first.
    fn compact_host(&mut self) -> Result<bool, TierError> {
        let busy: Vec<TensorId> = self
            .entries
            .iter()
            .filter(|(_, e)| e.save.is_some() || matches!(e.location, Location::Loading(_)))
            .map(|(&id, _)| id)
            .collect();
        for id in busy {
            self.finish_save(id)?;
            if matches!(self.entries[&id].location, Location::Loading(_)) {
                self.finish_load(id)?;
            }
        }
        let mut blocks = Vec::new();
        for entry in self.entries.values_mut() {
            let spilled = match &mut entry.location {
                Location::Host(host) => Some(host),
                _ => None,
            };
            for host in spilled.into_iter().chain(entry.clean_copy.as_mut()) {
                match Arc::get_mut(host) {
                    Some(block) => blocks.push(block),
                    None => return Ok(false),
                }
            }
        }
        Ok(self.host_pool.compact(&mut blocks))
    }

    /// Notices a copy to the host that has finished and makes its block the
    /// tensor's clean copy. Returns whether a copy is still running.
    fn poll_save(&mut self, id: TensorId) -> Result<bool, TierError> {
        let entry = self.entries.get_mut(&id).unwrap();
        let Some(save) = &mut entry.save else {
            return Ok(false);
        };
        match save.poll() {
            Poll::Pending => Ok(true),
            Poll::Ready(result) => {
                entry.save = None;
                match result {
                    Ok(host) => {
                        entry.clean_copy = Some(Arc::new(host));
                        Ok(false)
                    }
                    Err(e) => {
                        self.stats.host_bytes -= entry.bytes;
                        Err(dev(e))
                    }
                }
            }
        }
    }

    /// Submits a copy of a resident tensor to the host and returns without
    /// waiting. The tensor must need one (see [`Self::needs_save`]).
    /// `reclaim` is passed to [`Self::alloc_host`].
    fn start_save(&mut self, id: TensorId, reclaim: bool) -> Result<(), TierError> {
        let start = self.trace.now();
        let stream = self.stream_to_host()?;
        let bytes = self.entries[&id].bytes;
        let host = self.alloc_host(bytes, &stream, reclaim)?;
        let entry = self.entries.get_mut(&id).unwrap();
        let Location::Device(tensor) = &entry.location else {
            unreachable!("only a resident tensor is copied to the host");
        };
        let keep = tensor.clone();
        let src = (entry.ops.device_ptr)(&*keep);
        let save = SaveToHost {
            keep,
            src,
            bytes,
            host,
        };
        match copy::start(save, &stream).map_err(dev)? {
            Started::Done(host) => entry.clean_copy = Some(Arc::new(host)),
            Started::Pending(pending) => entry.save = Some(pending),
        }
        self.stats.host_bytes += bytes;
        self.stats.bytes_to_host += bytes as u64;
        self.trace
            .span(EventKind::WriteBack, Some(id), bytes, start);
        Ok(())
    }

    /// Waits for a copy to the host that is in flight, if there is one, and
    /// makes its block the tensor's clean copy.
    fn finish_save(&mut self, id: TensorId) -> Result<(), TierError> {
        if self.entries[&id].save.is_none() {
            return Ok(());
        }
        let stream = self.stream_to_host()?;
        let entry = self.entries.get_mut(&id).unwrap();
        let pending = entry.save.take().unwrap();
        match pending.wait(&stream) {
            Ok(host) => {
                entry.clean_copy = Some(Arc::new(host));
                Ok(())
            }
            // The tensor is still on the device, without a host copy.
            Err(e) => {
                self.stats.host_bytes -= entry.bytes;
                Err(dev(e))
            }
        }
    }

    fn evict(&mut self, id: TensorId) -> Result<(), TierError> {
        let start = self.trace.now();
        // Everything that can fail happens before the tensor is taken off
        // the device, so a failed eviction loses nothing.
        let copied = self.needs_save(id);
        if copied {
            self.start_save(id, true)?;
        }
        self.finish_save(id)?;
        let entry = self.entries.get_mut(&id).unwrap();
        let bytes = entry.bytes;
        let outcome = match std::mem::replace(&mut entry.location, Location::Dropped) {
            Location::Loading(pending) => {
                // Waits for the copy, then frees the device memory.
                drop(pending);
                let host = entry.clean_copy.take();
                entry.location = Location::Host(host.expect("a loading tensor keeps its source"));
                Evicted::Cancelled
            }
            // Dropping the tensor frees the device memory.
            Location::Device(_) => match (&entry.backing, entry.clean_copy.take()) {
                (ErasedBacking::Reconstruct(_), _) => Evicted::Dropped,
                (ErasedBacking::Spill, Some(host)) => {
                    entry.location = Location::Host(host);
                    if copied {
                        Evicted::Copied
                    } else {
                        Evicted::Clean
                    }
                }
                (ErasedBacking::Spill, None) => unreachable!("a host copy was made above"),
            },
            Location::Host(_) | Location::Dropped => {
                unreachable!("only tensors on the device are eviction victims")
            }
        };
        self.sub_resident(bytes);
        self.stats.evictions += 1;
        match outcome {
            Evicted::Copied => self.stats.spilled_bytes += bytes,
            Evicted::Clean | Evicted::Cancelled => {
                self.stats.spilled_bytes += bytes;
                self.stats.clean_evictions += 1;
            }
            Evicted::Dropped => {}
        }
        if outcome == Evicted::Cancelled {
            self.stats.prefetches_cancelled += 1;
        }
        let kind = EventKind::Evict {
            spilled: outcome != Evicted::Dropped,
            copied: outcome == Evicted::Copied,
        };
        self.trace.span(kind, Some(id), bytes, start);
        Ok(())
    }

    /// Brings an evicted tensor back and waits for it. Room must have been made.
    fn reload(&mut self, id: TensorId) -> Result<(), TierError> {
        let start = self.trace.now();
        let was_spilled = matches!(self.entries[&id].location, Location::Host(_));
        let stream = if was_spilled {
            Some(self.stream_to_device()?)
        } else {
            None
        };
        let entry = self.entries.get_mut(&id).unwrap();
        let bytes = entry.bytes;
        let tensor = match (&entry.location, &entry.backing) {
            (Location::Device(_) | Location::Loading(_), _) => return Ok(()),
            (Location::Host(host), _) => {
                let loaded = (LoadFromHost {
                    host: host.clone(),
                    ops: entry.ops,
                })
                .sync_on(stream.as_ref().unwrap())
                .map_err(dev)?;
                (entry.ops.reshape)(loaded, &entry.shape)?
            }
            (Location::Dropped, ErasedBacking::Reconstruct(rebuild)) => rebuild()?,
            (Location::Dropped, ErasedBacking::Spill) => {
                unreachable!("spilled tensors are kept on the host")
            }
        };
        let old = std::mem::replace(&mut entry.location, Location::Device(tensor));
        if let Location::Host(host) = old {
            // Kept as a clean copy, so host_bytes is unchanged.
            entry.clean_copy = Some(host);
        }
        let kind = if was_spilled {
            self.stats.spilled_bytes -= bytes;
            self.stats.reloads += 1;
            self.stats.bytes_to_device += bytes as u64;
            EventKind::Reload
        } else {
            self.stats.reconstructions += 1;
            EventKind::Reconstruct
        };
        self.add_resident(bytes);
        self.trace.span(kind, Some(id), bytes, start);
        Ok(())
    }

    /// Starts reloading the next `prefetch_depth` distinct declared tensors,
    /// without waiting for anything, and starts the copies to the host that
    /// will let the prefetches after these make room for free.
    fn prefetch_expected(&mut self, in_use: &[TensorId]) -> Result<(), TierError> {
        let mut targets: Vec<(usize, TensorId)> = Vec::new();
        for (pos, &id) in self.expected.iter().enumerate() {
            if targets.len() == self.config.prefetch_depth {
                break;
            }
            if !targets.iter().any(|&(_, t)| t == id) {
                targets.push((pos, id));
            }
        }
        let mut protect = in_use.to_vec();
        protect.extend(targets.iter().map(|&(_, id)| id));
        // How much room this round of prefetches asked for, and the plan
        // position of the last one that asked.
        let mut wanted = 0;
        let mut last = None;
        for (pos, id) in targets {
            let fetch = self.start_prefetch(id, &protect, Some(pos), false);
            if !matches!(fetch, Ok(Fetch::Skipped)) {
                wanted += self.entries[&id].bytes;
                last = Some(pos);
            }
            match fetch {
                Ok(Fetch::Started | Fetch::Skipped) => {}
                Ok(Fetch::NoRoom) => break,
                // Prefetching is best effort; the access itself will report
                // a budget problem if there still is one.
                Err(e) if e.is_recoverable() => break,
                Err(e) => return Err(e),
            }
        }
        if wanted > 0 {
            match self.write_behind(wanted, &protect, last) {
                Err(e) if !e.is_recoverable() => return Err(e),
                _ => {}
            }
        }
        Ok(())
    }

    /// With `wait`, room is made as for an access, which may copy a victim
    /// to the host first; without it only free evictions are used.
    fn start_prefetch(
        &mut self,
        id: TensorId,
        protect: &[TensorId],
        keep_before: Option<usize>,
        wait: bool,
    ) -> Result<Fetch, TierError> {
        let entry = self.entries.get(&id).ok_or(TierError::UnknownTensor(id))?;
        if !matches!(entry.location, Location::Host(_)) {
            return Ok(Fetch::Skipped);
        }
        let bytes = entry.bytes;
        let room = if wait {
            self.try_make_room(bytes, protect, keep_before)?
        } else {
            self.try_make_room_now(bytes, protect, keep_before)?
        };
        if let Room::Blocked { .. } = room {
            return Ok(Fetch::NoRoom);
        }
        let stream = self.stream_to_device()?;
        let start = self.trace.now();
        self.clock += 1;
        let entry = self.entries.get_mut(&id).unwrap();
        // Making room only evicts, so the tensor is still on the host.
        let Location::Host(host) = &entry.location else {
            unreachable!("making room does not reload anything");
        };
        let host = host.clone();
        let load = LoadFromHost {
            host: host.clone(),
            ops: entry.ops,
        };
        let (location, done) = match copy::start(load, &stream).map_err(dev)? {
            Started::Done(loaded) => (
                Location::Device((entry.ops.reshape)(loaded, &entry.shape)?),
                true,
            ),
            Started::Pending(pending) => (Location::Loading(Box::new(pending)), false),
        };
        entry.location = location;
        entry.clean_copy = Some(host);
        // Otherwise an LRU policy would evict it before it is ever used.
        entry.last_use = self.clock;
        self.stats.spilled_bytes -= bytes;
        self.stats.bytes_to_device += bytes as u64;
        self.stats.prefetches += 1;
        if done {
            self.stats.reloads += 1;
        }
        self.add_resident(bytes);
        self.trace.span(EventKind::Prefetch, Some(id), bytes, start);
        Ok(Fetch::Started)
    }

    /// Waits for a prefetched tensor's copy and makes it resident.
    fn finish_load(&mut self, id: TensorId) -> Result<(), TierError> {
        let start = self.trace.now();
        let stream = self.stream_to_device()?;
        let entry = self.entries.get_mut(&id).unwrap();
        let bytes = entry.bytes;
        let Location::Loading(pending) = std::mem::replace(&mut entry.location, Location::Dropped)
        else {
            unreachable!("finish_load is only called on a loading tensor");
        };
        let loaded = pending
            .wait(&stream)
            .map_err(dev)
            .and_then(|loaded| Ok((entry.ops.reshape)(loaded, &entry.shape)?));
        match loaded {
            Ok(tensor) => {
                entry.location = Location::Device(tensor);
                self.stats.reloads += 1;
                self.trace.span(EventKind::Reload, Some(id), bytes, start);
                Ok(())
            }
            // The device memory is gone but the host copy is intact.
            Err(e) => {
                let host = entry.clean_copy.take();
                entry.location = Location::Host(host.expect("a loading tensor keeps its source"));
                self.stats.spilled_bytes += bytes;
                self.sub_resident(bytes);
                Err(e)
            }
        }
    }

    fn add_resident(&mut self, bytes: usize) {
        self.stats.resident_bytes += bytes;
        self.stats.peak_resident_bytes = self
            .stats
            .peak_resident_bytes
            .max(self.stats.resident_bytes);
        self.trace.counter(self.stats.resident_bytes);
        if self.pressure_armed && self.stats.resident_bytes > self.config.soft_limit_bytes {
            self.pressure_armed = false;
            self.trace
                .instant(EventKind::Pressure, None, self.stats.resident_bytes);
            if let Some(cb) = self.on_pressure.as_mut() {
                cb(Pressure {
                    resident_bytes: self.stats.resident_bytes,
                    soft_limit_bytes: self.config.soft_limit_bytes,
                    budget_bytes: self.config.budget_bytes,
                });
            }
        }
    }

    fn sub_resident(&mut self, bytes: usize) {
        self.stats.resident_bytes -= bytes;
        self.trace.counter(self.stats.resident_bytes);
        if self.stats.resident_bytes <= self.config.pressure_rearm_bytes {
            self.pressure_armed = true;
        }
    }
}
