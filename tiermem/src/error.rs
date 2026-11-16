use crate::{Hint, TensorId};
use cutile::DTypeId;
use std::fmt;

#[derive(Debug)]
pub enum TierError {
    /// Recoverable: the request does not fit in the budget. The device is fine.
    BudgetExceeded {
        requested_bytes: usize,
        /// True when evictable tensors were resident but the policy forbade
        /// evicting them, i.e. a different policy would have succeeded.
        could_evict: bool,
        snapshot: Snapshot,
    },
    /// Recoverable: spilling a tensor would exceed the host RAM limit and no
    /// clean copy is left to free. The tensor is still on the device.
    HostBudgetExceeded {
        requested_bytes: usize,
        host_bytes: usize,
        host_limit_bytes: usize,
    },
    UnknownTensor(TensorId),
    NotResident(TensorId),
    /// The tensor was registered with another element type than the one
    /// asked for. Nothing was changed.
    WrongDType {
        id: TensorId,
        stored: DTypeId,
        requested: DTypeId,
    },
    /// Not recoverable by the manager: a CUDA or cutile operation failed.
    Device(cutile::error::Error),
}

impl TierError {
    /// Whether the caller can retry after freeing memory or changing policy,
    /// as opposed to a device failure.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            TierError::BudgetExceeded { .. } | TierError::HostBudgetExceeded { .. }
        )
    }
}

impl From<cutile::error::Error> for TierError {
    fn from(e: cutile::error::Error) -> Self {
        TierError::Device(e)
    }
}

impl fmt::Display for TierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TierError::BudgetExceeded {
                requested_bytes,
                could_evict,
                snapshot,
            } => {
                write!(
                    f,
                    "budget exceeded: requested {} with {} of {} resident",
                    mib(*requested_bytes),
                    mib(snapshot.resident_bytes),
                    mib(snapshot.budget_bytes),
                )?;
                if *could_evict {
                    write!(f, " (evictable tensors present; policy is Strict)")?;
                }
                write!(f, "\n{snapshot}")
            }
            TierError::HostBudgetExceeded {
                requested_bytes,
                host_bytes,
                host_limit_bytes,
            } => write!(
                f,
                "host budget exceeded: spilling {} with {} of {} held",
                mib(*requested_bytes),
                mib(*host_bytes),
                mib(*host_limit_bytes),
            ),
            TierError::UnknownTensor(id) => write!(f, "unknown tensor {id:?}"),
            TierError::NotResident(id) => {
                write!(
                    f,
                    "tensor {id:?} is not resident; call ensure_resident first"
                )
            }
            TierError::WrongDType {
                id,
                stored,
                requested,
            } => write!(f, "tensor {id:?} holds {stored}, not {requested}"),
            TierError::Device(e) => write!(f, "device error: {e}"),
        }
    }
}

impl std::error::Error for TierError {}

/// The manager's view of its tensors at one moment.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub budget_bytes: usize,
    pub resident_bytes: usize,
    pub entries: Vec<EntrySnapshot>,
}

#[derive(Debug, Clone)]
pub struct EntrySnapshot {
    pub id: TensorId,
    pub hint: Hint,
    pub dtype: DTypeId,
    pub bytes: usize,
    pub resident: bool,
    pub last_use: u64,
}

impl fmt::Display for Snapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for e in &self.entries {
            writeln!(
                f,
                "  {:?} {:?} {} {} {} last_use={}",
                e.id,
                e.hint,
                e.dtype,
                mib(e.bytes),
                if e.resident { "device" } else { "off-device" },
                e.last_use,
            )?;
        }
        Ok(())
    }
}

fn mib(bytes: usize) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}
