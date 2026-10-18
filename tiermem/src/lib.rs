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

mod error;
mod trace;

pub use error::{EntrySnapshot, Snapshot, TierError};
pub use trace::{Event, EventKind, Trace};

use cutile::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

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
}

#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Hard limit on bytes resident on the device.
    pub budget_bytes: usize,
    /// Crossing this fires the pressure callback, before anything fails.
    pub soft_limit_bytes: usize,
    pub policy: Policy,
}

impl Config {
    /// A config with the soft limit at 80% of the budget.
    pub fn new(budget_bytes: usize, policy: Policy) -> Self {
        Self {
            budget_bytes,
            soft_limit_bytes: budget_bytes / 5 * 4,
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
    pub resident_bytes: usize,
    pub spilled_bytes: usize,
    pub peak_resident_bytes: usize,
    pub evictions: u64,
    pub reloads: u64,
    pub reconstructions: u64,
    pub bytes_to_host: u64,
    pub bytes_to_device: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(u64);

enum Location<T: DType> {
    Device(Tensor<T>),
    Host(Arc<Vec<T>>),
    /// Evicted with [`Backing::Reconstruct`]; nothing is kept.
    Dropped,
}

struct Entry<T: DType> {
    hint: Hint,
    backing: Backing<T>,
    bytes: usize,
    shape: Vec<usize>,
    location: Location<T>,
    last_use: u64,
}

impl<T: DType> Entry<T> {
    fn is_resident(&self) -> bool {
        matches!(self.location, Location::Device(_))
    }
}

pub struct Manager<T: DType> {
    config: Config,
    entries: HashMap<TensorId, Entry<T>>,
    next_id: u64,
    clock: u64,
    stats: Stats,
    above_soft_limit: bool,
    on_pressure: Option<Box<dyn FnMut(Pressure)>>,
    trace: Trace,
}

impl<T: DType> Manager<T> {
    pub fn new(config: Config) -> Self {
        assert!(
            config.soft_limit_bytes <= config.budget_bytes,
            "soft limit must not exceed the budget"
        );
        Self {
            config,
            entries: HashMap::new(),
            next_id: 0,
            clock: 0,
            stats: Stats::default(),
            above_soft_limit: false,
            on_pressure: None,
            trace: Trace::new(Instant::now()),
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Called each time residency rises above the soft limit.
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
                last_use: self.clock,
            },
        );
        self.add_resident(bytes);
        self.trace.span(EventKind::Admit, Some(id), bytes, start);
        Ok(id)
    }

    /// Removes a tensor from management, returning it if it is on the device.
    pub fn unregister(&mut self, id: TensorId) -> Result<Option<Tensor<T>>, TierError> {
        let entry = self
            .entries
            .remove(&id)
            .ok_or(TierError::UnknownTensor(id))?;
        match entry.location {
            Location::Device(t) => {
                self.sub_resident(entry.bytes);
                Ok(Some(t))
            }
            Location::Host(_) => {
                self.stats.spilled_bytes -= entry.bytes;
                Ok(None)
            }
            Location::Dropped => Ok(None),
        }
    }

    /// Makes every tensor in `ids` resident, evicting others if needed.
    /// Afterwards, [`Manager::tensor`] returns each of them until the next
    /// `&mut self` call.
    pub fn ensure_resident(&mut self, ids: &[TensorId]) -> Result<(), TierError> {
        for &id in ids {
            let entry = self.entries.get(&id).ok_or(TierError::UnknownTensor(id))?;
            if !entry.is_resident() {
                let bytes = entry.bytes;
                self.make_room(bytes, ids)?;
                self.reload(id)?;
            }
            self.clock += 1;
            self.entries.get_mut(&id).unwrap().last_use = self.clock;
        }
        Ok(())
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

    pub fn is_resident(&self, id: TensorId) -> bool {
        self.entries.get(&id).is_some_and(Entry::is_resident)
    }

    pub fn stats(&self) -> Stats {
        self.stats
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
                resident: e.is_resident(),
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
        if bytes > self.config.budget_bytes {
            return Err(self.budget_exceeded(bytes, false));
        }
        while self.stats.resident_bytes + bytes > self.config.budget_bytes {
            let victim = self.pick_victim(protect);
            match (self.config.policy, victim) {
                (Policy::EvictLru, Some(id)) => self.evict(id)?,
                (_, victim) => return Err(self.budget_exceeded(bytes, victim.is_some())),
            }
        }
        Ok(())
    }

    fn budget_exceeded(&mut self, bytes: usize, could_evict: bool) -> TierError {
        self.trace.instant(EventKind::AdmitFailed, None, bytes);
        TierError::BudgetExceeded {
            requested_bytes: bytes,
            could_evict,
            snapshot: self.snapshot(),
        }
    }

    fn pick_victim(&self, protect: &[TensorId]) -> Option<TensorId> {
        self.entries
            .iter()
            .filter(|(id, e)| e.is_resident() && !protect.contains(id))
            .filter_map(|(&id, e)| Some(((e.hint.eviction_rank()?, e.last_use), id)))
            .min()
            .map(|(_, id)| id)
    }

    fn evict(&mut self, id: TensorId) -> Result<(), TierError> {
        let start = self.trace.now();
        let entry = self.entries.get_mut(&id).unwrap();
        let bytes = entry.bytes;
        let Location::Device(tensor) = std::mem::replace(&mut entry.location, Location::Dropped)
        else {
            unreachable!("only resident tensors are eviction victims");
        };
        let spilled = match entry.backing {
            Backing::Spill => match tensor.to_host_vec().sync() {
                Ok(host) => {
                    entry.location = Location::Host(Arc::new(host));
                    true
                }
                // The copy consumed the tensor, so its contents are lost.
                Err(e) => {
                    self.entries.remove(&id);
                    self.sub_resident(bytes);
                    return Err(TierError::Device(e.into()));
                }
            },
            Backing::Reconstruct(_) => false,
        };
        self.sub_resident(bytes);
        self.stats.evictions += 1;
        if spilled {
            self.stats.spilled_bytes += bytes;
            self.stats.bytes_to_host += bytes as u64;
        }
        self.trace
            .span(EventKind::Evict { spilled }, Some(id), bytes, start);
        Ok(())
    }

    fn reload(&mut self, id: TensorId) -> Result<(), TierError> {
        let start = self.trace.now();
        let entry = self.entries.get_mut(&id).unwrap();
        let bytes = entry.bytes;
        let was_spilled = matches!(entry.location, Location::Host(_));
        let tensor = match (&entry.location, &entry.backing) {
            (Location::Device(_), _) => return Ok(()),
            (Location::Host(host), _) => cutile::api::copy_host_vec_to_device(host)
                .sync()
                .map_err(|e| TierError::Device(e.into()))?
                .reshape(&entry.shape)?,
            (Location::Dropped, Backing::Reconstruct(rebuild)) => rebuild()?,
            (Location::Dropped, Backing::Spill) => {
                unreachable!("spilled tensors are kept on the host")
            }
        };
        entry.location = Location::Device(tensor);
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

    fn add_resident(&mut self, bytes: usize) {
        self.stats.resident_bytes += bytes;
        self.stats.peak_resident_bytes = self
            .stats
            .peak_resident_bytes
            .max(self.stats.resident_bytes);
        self.trace.counter(self.stats.resident_bytes);
        let above = self.stats.resident_bytes > self.config.soft_limit_bytes;
        if above && !self.above_soft_limit {
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
        self.above_soft_limit = above;
    }

    fn sub_resident(&mut self, bytes: usize) {
        self.stats.resident_bytes -= bytes;
        self.trace.counter(self.stats.resident_bytes);
        self.above_soft_limit = self.stats.resident_bytes > self.config.soft_limit_bytes;
    }
}
