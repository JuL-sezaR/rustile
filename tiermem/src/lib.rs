//! Workload-aware GPU memory tiering.
//!
//! A [`Manager`] owns a set of registered device tensors and keeps their total
//! size under a VRAM budget. Each tensor carries a [`Hint`] saying how costly
//! it is to evict. When an admission or reload would exceed the budget, the
//! manager evicts lower-priority tensors first (spilling them to host RAM, or
//! dropping them if they can be rebuilt) instead of failing.
//!
//! The budget is *logical*: it is the manager's own accounting of the bytes it
//! has admitted, not a reading of the driver's free-memory counter. That makes
//! behaviour reproducible on any GPU and lets a 16 GB card stand in for a much
//! smaller one.
//!
//! Spilled contents live in page-locked host memory, which is locked either
//! as it is first needed or up front (see [`HostAlloc`]), and move on a CUDA
//! stream of the manager's own, so a reload started with [`Manager::prefetch`] (or
//! automatically, see [`Config::prefetch_depth`]) overlaps with kernels
//! running on other streams.

mod copy;
mod error;
mod host;
mod trace;

pub use error::{EntrySnapshot, Snapshot, TierError};
pub use trace::{Event, EventKind, Trace};

use copy::{LoadFromHost, PendingLoad, SpillToHost, Started};
use cutile::cuda_async::device_context::{get_default_device, with_device};
use cutile::cuda_core::Stream;
use cutile::prelude::*;
use host::{HostBlock, HostPool};
use std::cmp::Reverse;
use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
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
    /// than the one being fetched. Zero disables it.
    pub prefetch_depth: usize,
    pub policy: Policy,
}

impl Config {
    /// A config with the soft limit at 80% of the budget, pressure re-arming
    /// at 60%, no host limit, host memory locked as it is needed, and no
    /// automatic prefetch.
    pub fn new(budget_bytes: usize, policy: Policy) -> Self {
        Self {
            budget_bytes,
            soft_limit_bytes: budget_bytes / 5 * 4,
            pressure_rearm_bytes: budget_bytes / 5 * 3,
            host_limit_bytes: usize::MAX,
            host_alloc: HostAlloc::Lazy,
            prefetch_depth: 0,
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
    pub bytes_to_host: u64,
    pub bytes_to_device: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(u64);

enum Location<T: DType> {
    Device(Tensor<T>),
    /// A prefetch is copying the contents back; the device memory is already
    /// allocated and counted, and `clean_copy` holds the source.
    Loading(PendingLoad<T>),
    Host(Arc<HostBlock>),
    /// Evicted with [`Backing::Reconstruct`]; nothing is kept.
    Dropped,
}

struct Entry<T: DType> {
    hint: Hint,
    backing: Backing<T>,
    bytes: usize,
    shape: Vec<usize>,
    location: Location<T>,
    /// For a resident tensor reloaded from the host: the contents it was
    /// reloaded from, still valid because nothing has written to it since.
    /// Evicting it again reuses this instead of copying back to the host.
    clean_copy: Option<Arc<HostBlock>>,
    last_use: u64,
}

impl<T: DType> Entry<T> {
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

pub struct Manager<T: DType> {
    config: Config,
    entries: HashMap<TensorId, Entry<T>>,
    next_id: u64,
    clock: u64,
    /// Declared future accesses, next first; consumed by `ensure_resident`.
    expected: VecDeque<TensorId>,
    /// The arguments of the last `ensure_resident`; prefetching spares them.
    in_use: Vec<TensorId>,
    /// Created on first use, so a manager that never migrates needs no stream.
    copy_stream: Option<Arc<Stream>>,
    host_pool: HostPool,
    /// The eager arena has been page-locked.
    host_reserved: bool,
    stats: Stats,
    pressure_armed: bool,
    on_pressure: Option<Box<dyn FnMut(Pressure)>>,
    trace: Trace,
}

impl<T: DType> Manager<T> {
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
        Self {
            config,
            entries: HashMap::new(),
            next_id: 0,
            clock: 0,
            expected: VecDeque::new(),
            in_use: Vec::new(),
            copy_stream: None,
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
    pub fn register(&mut self, tensor: Tensor<T>, hint: Hint) -> Result<TensorId, TierError> {
        self.register_with(tensor, hint, Backing::Spill)
    }

    /// Admits a device tensor that `rebuild` can recreate, so eviction costs
    /// no host memory or device-to-host copy.
    pub fn register_reconstructible(
        &mut self,
        tensor: Tensor<T>,
        hint: Hint,
        rebuild: impl Fn() -> Result<Tensor<T>, Error> + 'static,
    ) -> Result<TensorId, TierError> {
        self.register_with(tensor, hint, Backing::Reconstruct(Box::new(rebuild)))
    }

    pub fn register_with(
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
        self.clock += 1;
        self.entries.insert(
            id,
            Entry {
                hint,
                backing,
                bytes,
                shape,
                location: Location::Device(tensor),
                clean_copy: None,
                last_use: self.clock,
            },
        );
        self.add_resident(bytes);
        self.trace.span(EventKind::Admit, Some(id), bytes, start);
        Ok(id)
    }

    /// Removes a tensor from management, returning it if it is on the device.
    pub fn unregister(&mut self, id: TensorId) -> Result<Option<Tensor<T>>, TierError> {
        let entry = self.entries.get(&id).ok_or(TierError::UnknownTensor(id))?;
        if matches!(entry.location, Location::Loading(_)) {
            self.finish_load(id)?;
        }
        let entry = self.entries.remove(&id).unwrap();
        self.expected.retain(|&e| e != id);
        if entry.clean_copy.is_some() {
            self.stats.host_bytes -= entry.bytes;
        }
        match entry.location {
            Location::Device(t) => {
                self.sub_resident(entry.bytes);
                Ok(Some(t))
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
            match self.start_prefetch(id, &protect, None)? {
                Fetch::Started => started += 1,
                Fetch::Skipped => {}
                Fetch::NoRoom => break,
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
    pub fn tensor(&self, id: TensorId) -> Result<&Tensor<T>, TierError> {
        match &self
            .entries
            .get(&id)
            .ok_or(TierError::UnknownTensor(id))?
            .location
        {
            Location::Device(t) => Ok(t),
            _ => Err(TierError::NotResident(id)),
        }
    }

    /// Mutable access to the device tensor for `id`, which must be resident.
    ///
    /// Marks the tensor dirty: its clean host copy, if any, is freed and the
    /// next eviction copies the contents back to the host. Replacing the
    /// tensor with one of a different shape is a bug.
    ///
    /// Writes to a reconstructible tensor do not survive eviction: it is
    /// rebuilt from its original recipe.
    pub fn tensor_mut(&mut self, id: TensorId) -> Result<&mut Tensor<T>, TierError> {
        let entry = self
            .entries
            .get_mut(&id)
            .ok_or(TierError::UnknownTensor(id))?;
        let Location::Device(t) = &mut entry.location else {
            return Err(TierError::NotResident(id));
        };
        if entry.clean_copy.take().is_some() {
            self.stats.host_bytes -= entry.bytes;
        }
        Ok(t)
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
            let Some((victim, next_use)) = self.pick_victim(protect) else {
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

    /// The tensor to evict next and the position of its next declared use
    /// (`usize::MAX` if it has none).
    fn pick_victim(&self, protect: &[TensorId]) -> Option<(TensorId, usize)> {
        let mut next_use = HashMap::new();
        for (pos, &id) in self.expected.iter().enumerate() {
            next_use.entry(id).or_insert(pos);
        }
        let lookahead = self.config.policy == Policy::EvictLookahead;
        self.entries
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
            .min_by_key(|&(key, _)| key)
            .map(|(_, victim)| victim)
    }

    fn copy_stream(&mut self) -> Result<Arc<Stream>, TierError> {
        if let Some(stream) = &self.copy_stream {
            return Ok(stream.clone());
        }
        let stream = with_device(get_default_device(), |device| device.new_stream())
            .map_err(dev)?
            .map_err(dev)?;
        self.copy_stream = Some(stream.clone());
        Ok(stream)
    }

    /// Page-locks the eager arena, once.
    fn reserve_host(&mut self) -> Result<(), TierError> {
        if self.config.host_alloc == HostAlloc::Eager && !self.host_reserved {
            let stream = self.copy_stream()?;
            self.host_pool
                .reserve(stream.device(), self.config.host_limit_bytes)
                .map_err(dev)?;
            self.host_reserved = true;
        }
        Ok(())
    }

    /// A host block for spilling `bytes`. If the pool has no room, clean
    /// copies of resident tensors are given up, least recently used first,
    /// until it does.
    fn alloc_host(&mut self, bytes: usize, stream: &Arc<Stream>) -> Result<HostBlock, TierError> {
        let grow = self.config.host_alloc == HostAlloc::Lazy;
        loop {
            let block = self.host_pool.alloc(stream.device(), bytes, grow);
            if let Some(block) = block.map_err(dev)? {
                return Ok(block);
            }
            let victim = self
                .entries
                .values_mut()
                .filter(|e| e.is_resident() && e.clean_copy.is_some())
                .min_by_key(|e| e.last_use);
            let Some(entry) = victim else {
                return Err(TierError::HostBudgetExceeded {
                    requested_bytes: bytes,
                    host_bytes: self.stats.host_bytes,
                    host_limit_bytes: self.config.host_limit_bytes,
                });
            };
            // Only an in-flight load shares a clean copy, and loading tensors
            // are not candidates, so this returns the block to the pool.
            entry.clean_copy = None;
            self.stats.host_bytes -= entry.bytes;
            self.stats.clean_copies_dropped += 1;
        }
    }

    fn evict(&mut self, id: TensorId) -> Result<(), TierError> {
        let start = self.trace.now();
        let entry = &self.entries[&id];
        let bytes = entry.bytes;
        // Everything that can fail without losing the contents happens before
        // the tensor is taken off the device.
        let fresh_host = match (&entry.location, &entry.backing, &entry.clean_copy) {
            (Location::Device(_), Backing::Spill, None) => {
                let stream = self.copy_stream()?;
                let host = self.alloc_host(bytes, &stream)?;
                Some((stream, host))
            }
            _ => None,
        };
        let entry = self.entries.get_mut(&id).unwrap();
        let outcome = match std::mem::replace(&mut entry.location, Location::Dropped) {
            Location::Loading(pending) => {
                // Waits for the copy, then frees the device memory.
                drop(pending);
                let host = entry.clean_copy.take();
                entry.location = Location::Host(host.expect("a loading tensor keeps its source"));
                Evicted::Cancelled
            }
            Location::Device(tensor) => {
                match (&entry.backing, entry.clean_copy.take(), fresh_host) {
                    (Backing::Reconstruct(_), _, _) => Evicted::Dropped,
                    (Backing::Spill, Some(host), _) => {
                        entry.location = Location::Host(host);
                        Evicted::Clean
                    }
                    (Backing::Spill, None, Some((stream, host))) => {
                        match (SpillToHost { tensor, host }).sync_on(&stream) {
                            Ok(host) => {
                                entry.location = Location::Host(Arc::new(host));
                                Evicted::Copied
                            }
                            // The copy consumed the tensor, so its contents are lost.
                            Err(e) => {
                                self.entries.remove(&id);
                                self.expected.retain(|&e| e != id);
                                self.sub_resident(bytes);
                                return Err(dev(e));
                            }
                        }
                    }
                    (Backing::Spill, None, None) => {
                        unreachable!("a host buffer was prepared above")
                    }
                }
            }
            Location::Host(_) | Location::Dropped => {
                unreachable!("only tensors on the device are eviction victims")
            }
        };
        self.sub_resident(bytes);
        self.stats.evictions += 1;
        match outcome {
            Evicted::Copied => {
                self.stats.spilled_bytes += bytes;
                self.stats.host_bytes += bytes;
                self.stats.bytes_to_host += bytes as u64;
            }
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
            Some(self.copy_stream()?)
        } else {
            None
        };
        let entry = self.entries.get_mut(&id).unwrap();
        let bytes = entry.bytes;
        let tensor = match (&entry.location, &entry.backing) {
            (Location::Device(_) | Location::Loading(_), _) => return Ok(()),
            (Location::Host(host), _) => (LoadFromHost {
                host: host.clone(),
                dtype: PhantomData,
            })
            .sync_on(stream.as_ref().unwrap())
            .map_err(dev)?
            .reshape(&entry.shape)?,
            (Location::Dropped, Backing::Reconstruct(rebuild)) => rebuild()?,
            (Location::Dropped, Backing::Spill) => {
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

    /// Starts reloading the next `prefetch_depth` distinct declared tensors.
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
        for (pos, id) in targets {
            match self.start_prefetch(id, &protect, Some(pos)) {
                Ok(Fetch::Started | Fetch::Skipped) => {}
                Ok(Fetch::NoRoom) => break,
                // Prefetching is best effort; the access itself will report
                // a budget problem if there still is one.
                Err(e) if e.is_recoverable() => break,
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    fn start_prefetch(
        &mut self,
        id: TensorId,
        protect: &[TensorId],
        keep_before: Option<usize>,
    ) -> Result<Fetch, TierError> {
        let entry = self.entries.get(&id).ok_or(TierError::UnknownTensor(id))?;
        if !matches!(entry.location, Location::Host(_)) {
            return Ok(Fetch::Skipped);
        }
        let bytes = entry.bytes;
        if let Room::Blocked { .. } = self.try_make_room(bytes, protect, keep_before)? {
            return Ok(Fetch::NoRoom);
        }
        let stream = self.copy_stream()?;
        let start = self.trace.now();
        self.clock += 1;
        let entry = self.entries.get_mut(&id).unwrap();
        // Making room only evicts, so the tensor is still on the host.
        let Location::Host(host) = &entry.location else {
            unreachable!("making room does not reload anything");
        };
        let host = host.clone();
        let (location, done) = match copy::start_load(&host, &stream).map_err(dev)? {
            Started::Done(tensor) => (Location::Device(tensor.reshape(&entry.shape)?), true),
            Started::Pending(pending) => (Location::Loading(pending), false),
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
        let stream = self.copy_stream()?;
        let entry = self.entries.get_mut(&id).unwrap();
        let bytes = entry.bytes;
        let Location::Loading(pending) = std::mem::replace(&mut entry.location, Location::Dropped)
        else {
            unreachable!("finish_load is only called on a loading tensor");
        };
        let loaded = pending
            .wait(&stream)
            .map_err(dev)
            .and_then(|tensor| Ok(tensor.reshape(&entry.shape)?));
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
