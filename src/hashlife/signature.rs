use crate::bitgrid::{Cell, Coord};

/// An exact interned-root identity within one HashLife session and GC epoch.
///
/// A compaction changes the epoch before a remapped root can be observed. The
/// root level prevents an accidentally reused numeric id from matching a root
/// with different geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HashLifeStateIdentity {
    pub(crate) session: u64,
    pub(crate) epoch: u64,
    pub(crate) root: u64,
    pub(crate) level: u32,
}

impl HashLifeStateIdentity {
    pub(crate) fn same_epoch(self, other: Self) -> bool {
        self.session == other.session && self.epoch == other.epoch
    }
}

/// A constant-time checkpoint over the current interned root.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HashLifeStateCheckpoint {
    pub generation: u64,
    pub origin: Cell,
    pub identity: HashLifeStateIdentity,
    pub population: u128,
    pub root_span: Coord,
}
