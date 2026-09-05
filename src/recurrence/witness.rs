use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bitgrid::{BitGrid, CHUNK_SIZE};

use super::{MAX_RECURRENCE_BYTES, MAX_WITNESS_CELLS, MAX_WITNESS_CHUNKS};

static NEXT_LINEAGE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Lineage {
    pub session: u64,
    pub epoch: u64,
}

impl Lineage {
    pub const fn new(session: u64, epoch: u64) -> Self {
        Self { session, epoch }
    }

    pub fn fresh() -> Self {
        Self {
            session: NEXT_LINEAGE.fetch_add(1, Ordering::Relaxed),
            epoch: 0,
        }
    }

    pub const fn next_epoch(self) -> Option<Self> {
        match self.epoch.checked_add(1) {
            Some(epoch) => Some(Self {
                session: self.session,
                epoch,
            }),
            None => None,
        }
    }
}

impl Default for Lineage {
    fn default() -> Self {
        Self::fresh()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DagWitness {
    session: u64,
    epoch: u64,
    root: u64,
    level: u32,
}

impl DagWitness {
    pub const fn new(session: u64, epoch: u64, root: u64, level: u32) -> Self {
        Self {
            session,
            epoch,
            root,
            level,
        }
    }

    pub const fn session(self) -> u64 {
        self.session
    }

    pub const fn epoch(self) -> u64 {
        self.epoch
    }

    pub const fn root(self) -> u64 {
        self.root
    }

    pub const fn level(self) -> u32 {
        self.level
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecurrenceUnavailable {
    Allocation,
    ByteLimit,
    EntryLimit,
    LineageMismatch,
    NonMonotonicGeneration,
    CoordinateOverflow,
    WitnessLimit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PackedChunk {
    x: i128,
    y: i128,
    bits: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PackedChunkWitness {
    width: i128,
    height: i128,
    chunks: Vec<PackedChunk>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExactWitness {
    /// Translation-normalized packed chunks; equality is exact cell equality.
    Cells(PackedChunkWitness),
    /// Weak arena identity. Numeric IDs are observations, not liveness roots.
    Dag(DagWitness),
}

impl ExactWitness {
    pub fn from_grid(grid: &BitGrid) -> Result<(Self, (i128, i128)), RecurrenceUnavailable> {
        if grid.chunk_count() > MAX_WITNESS_CHUNKS {
            return Err(RecurrenceUnavailable::WitnessLimit);
        }
        if grid.population() > MAX_WITNESS_CELLS {
            return Err(RecurrenceUnavailable::WitnessLimit);
        }
        let Some((min_x, min_y, max_x, max_y)) = grid.bounds() else {
            return Ok((
                Self::Cells(PackedChunkWitness {
                    width: 0,
                    height: 0,
                    chunks: Vec::new(),
                }),
                (0, 0),
            ));
        };
        let anchor = (i128::from(min_x), i128::from(min_y));
        let width = i128::from(max_x)
            .checked_sub(anchor.0)
            .and_then(|span| span.checked_add(1))
            .ok_or(RecurrenceUnavailable::CoordinateOverflow)?;
        let height = i128::from(max_y)
            .checked_sub(anchor.1)
            .and_then(|span| span.checked_add(1))
            .ok_or(RecurrenceUnavailable::CoordinateOverflow)?;
        let capacity = grid
            .chunk_count()
            .checked_mul(4)
            .ok_or(RecurrenceUnavailable::ByteLimit)?;
        let capacity_bytes = capacity
            .checked_mul(size_of::<PackedChunk>())
            .ok_or(RecurrenceUnavailable::ByteLimit)?;
        if capacity_bytes > MAX_RECURRENCE_BYTES {
            return Err(RecurrenceUnavailable::ByteLimit);
        }
        let mut chunks = Vec::new();
        chunks
            .try_reserve(capacity)
            .map_err(|_| RecurrenceUnavailable::Allocation)?;
        for ((chunk_x, chunk_y), bits) in grid.occupied_chunks() {
            append_normalized_chunk(&mut chunks, chunk_x, chunk_y, bits, anchor)?;
        }
        chunks.retain(|chunk| chunk.bits != 0);
        chunks.sort_unstable();
        Ok((
            Self::Cells(PackedChunkWitness {
                width,
                height,
                chunks,
            }),
            anchor,
        ))
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        match self {
            Self::Cells(witness) => witness.chunks.capacity() * size_of::<PackedChunk>(),
            Self::Dag(_) => 0,
        }
    }

    pub(crate) fn fingerprint(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    pub(crate) const fn dag_arena(&self) -> Option<(u64, u64)> {
        match self {
            Self::Cells(_) => None,
            Self::Dag(witness) => Some((witness.session, witness.epoch)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Observation {
    pub(crate) lineage: Lineage,
    pub(crate) generation: u64,
    pub(crate) anchor: (i128, i128),
    pub(crate) witness: ExactWitness,
}

impl Observation {
    pub fn from_grid(
        lineage: Lineage,
        generation: u64,
        grid: &BitGrid,
    ) -> Result<Self, RecurrenceUnavailable> {
        let (witness, anchor) = ExactWitness::from_grid(grid)?;
        Ok(Self {
            lineage,
            generation,
            anchor,
            witness,
        })
    }

    pub const fn from_dag(
        lineage: Lineage,
        generation: u64,
        anchor: (i128, i128),
        witness: DagWitness,
    ) -> Self {
        Self {
            lineage,
            generation,
            anchor,
            witness: ExactWitness::Dag(witness),
        }
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn anchor(&self) -> (i128, i128) {
        self.anchor
    }

    pub const fn lineage(&self) -> Lineage {
        self.lineage
    }

    pub const fn witness(&self) -> &ExactWitness {
        &self.witness
    }
}

fn append_normalized_chunk(
    output: &mut Vec<PackedChunk>,
    chunk_x: i64,
    chunk_y: i64,
    bits: u64,
    anchor: (i128, i128),
) -> Result<(), RecurrenceUnavailable> {
    let chunk_size = i128::from(CHUNK_SIZE);
    let source_x = i128::from(chunk_x)
        .checked_mul(chunk_size)
        .and_then(|value| value.checked_sub(anchor.0))
        .ok_or(RecurrenceUnavailable::CoordinateOverflow)?;
    let source_y = i128::from(chunk_y)
        .checked_mul(chunk_size)
        .and_then(|value| value.checked_sub(anchor.1))
        .ok_or(RecurrenceUnavailable::CoordinateOverflow)?;
    let target_x = source_x.div_euclid(chunk_size);
    let local_x = u32::try_from(source_x.rem_euclid(chunk_size))
        .map_err(|_| RecurrenceUnavailable::CoordinateOverflow)?;
    for row in 0_u32..8 {
        let row_bits = (bits >> (row * 8)) & 0xff;
        if row_bits == 0 {
            continue;
        }
        let y = source_y
            .checked_add(i128::from(row))
            .ok_or(RecurrenceUnavailable::CoordinateOverflow)?;
        let target_y = y.div_euclid(chunk_size);
        let local_y = u32::try_from(y.rem_euclid(chunk_size))
            .map_err(|_| RecurrenceUnavailable::CoordinateOverflow)?;
        let shifted = u16::try_from(row_bits)
            .map_err(|_| RecurrenceUnavailable::CoordinateOverflow)?
            << local_x;
        accumulate(
            output,
            target_x,
            target_y,
            u64::from(shifted & 0xff) << (local_y * 8),
        )?;
        let carry = shifted >> 8;
        if carry != 0 {
            let carry_x = target_x
                .checked_add(1)
                .ok_or(RecurrenceUnavailable::CoordinateOverflow)?;
            accumulate(output, carry_x, target_y, u64::from(carry) << (local_y * 8))?;
        }
    }
    Ok(())
}

fn accumulate(
    output: &mut Vec<PackedChunk>,
    x: i128,
    y: i128,
    bits: u64,
) -> Result<(), RecurrenceUnavailable> {
    if bits == 0 {
        return Ok(());
    }
    if let Some(chunk) = output.iter_mut().find(|chunk| chunk.x == x && chunk.y == y) {
        chunk.bits |= bits;
        return Ok(());
    }
    output
        .try_reserve(1)
        .map_err(|_| RecurrenceUnavailable::Allocation)?;
    output.push(PackedChunk { x, y, bits });
    Ok(())
}
