//! A timeline of manager activity, exportable as Chrome trace JSON for
//! Perfetto (ui.perfetto.dev) or chrome://tracing.

use crate::TensorId;
use std::fmt::Write as _;
use std::io;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Admit,
    AdmitFailed,
    Evict {
        /// The contents were kept in host RAM rather than dropped.
        spilled: bool,
        /// They were copied off the device; false when a clean host copy
        /// was reused.
        copied: bool,
    },
    /// A reload awaited by the caller. For a prefetched tensor this covers
    /// only the wait for what was left of the copy.
    Reload,
    /// A reload submitted to the copy stream without waiting.
    Prefetch,
    Reconstruct,
    Pressure,
    /// Resident bytes after a change; `bytes` holds the new total.
    Resident,
}

impl EventKind {
    fn name(self) -> &'static str {
        match self {
            EventKind::Admit => "admit",
            EventKind::AdmitFailed => "admit_failed",
            EventKind::Evict {
                spilled: true,
                copied: true,
            } => "evict_spill",
            EventKind::Evict {
                spilled: true,
                copied: false,
            } => "evict_clean",
            EventKind::Evict { spilled: false, .. } => "evict_drop",
            EventKind::Reload => "reload",
            EventKind::Prefetch => "prefetch",
            EventKind::Reconstruct => "reconstruct",
            EventKind::Pressure => "pressure",
            EventKind::Resident => "resident_bytes",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Event {
    pub kind: EventKind,
    pub id: Option<TensorId>,
    pub bytes: usize,
    /// Microseconds since the manager was created.
    pub start_us: u64,
    /// Zero for instant and counter events.
    pub duration_us: u64,
}

#[derive(Debug)]
pub struct Trace {
    origin: Instant,
    events: Vec<Event>,
}

impl Trace {
    pub(crate) fn new(origin: Instant) -> Self {
        Self {
            origin,
            events: Vec::new(),
        }
    }

    pub fn events(&self) -> &[Event] {
        &self.events
    }

    pub(crate) fn now(&self) -> u64 {
        self.origin.elapsed().as_micros() as u64
    }

    pub(crate) fn span(
        &mut self,
        kind: EventKind,
        id: Option<TensorId>,
        bytes: usize,
        start_us: u64,
    ) {
        let duration_us = self.now().saturating_sub(start_us);
        self.events.push(Event {
            kind,
            id,
            bytes,
            start_us,
            duration_us,
        });
    }

    pub(crate) fn instant(&mut self, kind: EventKind, id: Option<TensorId>, bytes: usize) {
        let start_us = self.now();
        self.events.push(Event {
            kind,
            id,
            bytes,
            start_us,
            duration_us: 0,
        });
    }

    pub(crate) fn counter(&mut self, resident_bytes: usize) {
        self.instant(EventKind::Resident, None, resident_bytes);
    }

    pub fn to_chrome_json(&self) -> String {
        let mut out = String::from("{\"traceEvents\":[\n");
        for (i, e) in self.events.iter().enumerate() {
            if i > 0 {
                out.push_str(",\n");
            }
            let name = e.kind.name();
            let ts = e.start_us;
            match e.kind {
                EventKind::Resident => write!(
                    out,
                    r#"{{"name":"{name}","ph":"C","ts":{ts},"pid":0,"args":{{"MiB":{:.1}}}}}"#,
                    e.bytes as f64 / (1024.0 * 1024.0)
                ),
                EventKind::Pressure | EventKind::AdmitFailed => write!(
                    out,
                    r#"{{"name":"{name}","ph":"i","s":"p","ts":{ts},"pid":0,"tid":0,"args":{{"bytes":{}}}}}"#,
                    e.bytes
                ),
                _ => {
                    let id = e.id.map_or(-1, |TensorId(n)| n as i64);
                    write!(
                        out,
                        r#"{{"name":"{name}","ph":"X","ts":{ts},"dur":{},"pid":0,"tid":0,"args":{{"id":{id},"bytes":{}}}}}"#,
                        e.duration_us, e.bytes
                    )
                }
            }
            .unwrap();
        }
        out.push_str("\n]}\n");
        out
    }

    pub fn write_chrome_trace(&self, path: impl AsRef<Path>) -> io::Result<()> {
        std::fs::write(path, self.to_chrome_json())
    }
}
