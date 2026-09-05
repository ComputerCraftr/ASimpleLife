use super::geometry::{RootGeometry, ValidatedLevel};
use super::*;
use crate::RequiredExt;
use crate::bitgrid::{BitGrid, Coord};
use crate::persistence::HASHLIFE_SNAPSHOT_MAGIC;
use std::error::Error;
use std::fmt;
use std::io::{BufRead, BufReader, Cursor, Read, Write};

const MAX_SNAPSHOT_LINE_BYTES: usize = 4_096;
// Buffered input, scratch line, and the bounded header/record strings coexist.
const SNAPSHOT_PARSE_SCRATCH_BYTES: u128 = 32 * 1_024;

#[cfg(test)]
mod resource_tests;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedHashLifeSnapshot {
    pub(super) bytes: Vec<u8>,
}

impl OwnedHashLifeSnapshot {
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HashLifeSnapshotError {
    message: String,
    requested_bytes: Option<u128>,
}

impl HashLifeSnapshotError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            requested_bytes: None,
        }
    }

    pub(super) fn allocation(requested_bytes: u128) -> Self {
        Self {
            message: format!("snapshot allocation failed for {requested_bytes} bytes"),
            requested_bytes: Some(requested_bytes),
        }
    }

    pub(super) const fn allocation_bytes(&self) -> Option<u128> {
        self.requested_bytes
    }

    fn io(error: std::io::Error) -> Self {
        Self::new(format!("snapshot I/O failed: {error}"))
    }
}

impl From<std::io::Error> for HashLifeSnapshotError {
    fn from(error: std::io::Error) -> Self {
        Self::io(error)
    }
}

impl fmt::Display for HashLifeSnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for HashLifeSnapshotError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SnapshotNodeRef {
    DeadLeaf,
    LiveLeaf,
    Node(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SnapshotChildRef {
    node: SnapshotNodeRef,
    symmetry: Symmetry,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SnapshotNode {
    level: ValidatedLevel,
    children: [SnapshotChildRef; 4],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ParsedHashLifeSnapshot {
    generation: u64,
    origin_x: Coord,
    origin_y: Coord,
    root: SnapshotChildRef,
    nodes: Vec<SnapshotNode>,
}

impl ParsedHashLifeSnapshot {
    fn allocated_bytes(&self) -> u128 {
        self.nodes.capacity() as u128 * std::mem::size_of::<SnapshotNode>() as u128
    }

    pub(super) fn estimated_import_bytes(&self) -> u128 {
        u128::try_from(self.nodes.len())
            .unwrap_or(u128::MAX)
            .saturating_mul(256)
            .saturating_add(1_024)
    }
}

impl SnapshotNodeRef {
    fn decode(token: &str) -> Result<Self, HashLifeSnapshotError> {
        match token {
            "D" => Ok(Self::DeadLeaf),
            "L" => Ok(Self::LiveLeaf),
            _ => {
                let index = token
                    .strip_prefix('N')
                    .ok_or_else(|| {
                        HashLifeSnapshotError::new(format!(
                            "invalid snapshot node reference {token:?}"
                        ))
                    })?
                    .parse::<u32>()
                    .map_err(|_| {
                        HashLifeSnapshotError::new(format!("invalid snapshot node index {token:?}"))
                    })?;
                Ok(Self::Node(index))
            }
        }
    }
}

impl SnapshotChildRef {
    fn decode(token: &str) -> Result<Self, HashLifeSnapshotError> {
        let (node_token, symmetry_token) = token.split_once('@').ok_or_else(|| {
            HashLifeSnapshotError::new(format!("invalid snapshot child reference {token:?}"))
        })?;
        let symmetry_index = symmetry_token.parse::<usize>().map_err(|_| {
            HashLifeSnapshotError::new(format!("invalid symmetry index {symmetry_token:?}"))
        })?;
        let symmetry = Symmetry::ALL.get(symmetry_index).copied().ok_or_else(|| {
            HashLifeSnapshotError::new(format!("unknown symmetry index {symmetry_index}"))
        })?;
        Ok(Self {
            node: SnapshotNodeRef::decode(node_token)?,
            symmetry,
        })
    }
}

fn write_child_ref(out: &mut impl Write, child: SnapshotChildRef) -> std::io::Result<()> {
    match child.node {
        SnapshotNodeRef::DeadLeaf => out.write_all(b"D")?,
        SnapshotNodeRef::LiveLeaf => out.write_all(b"L")?,
        SnapshotNodeRef::Node(index) => write!(out, "N{index}")?,
    }
    write!(out, "@{}", child.symmetry as u8)
}

fn write_snapshot(
    snapshot: &ParsedHashLifeSnapshot,
    out: &mut impl Write,
) -> Result<(), HashLifeSnapshotError> {
    writeln!(out, "{HASHLIFE_SNAPSHOT_MAGIC}").map_err(HashLifeSnapshotError::io)?;
    writeln!(out, "generation {}", snapshot.generation)?;
    writeln!(out, "origin {} {}", snapshot.origin_x, snapshot.origin_y)?;
    out.write_all(b"root ").map_err(HashLifeSnapshotError::io)?;
    write_child_ref(out, snapshot.root)?;
    out.write_all(b"\n").map_err(HashLifeSnapshotError::io)?;
    writeln!(out, "nodes {}", snapshot.nodes.len())?;
    for node in &snapshot.nodes {
        write!(out, "node {} ", node.level.get())?;
        for (index, child) in node.children.iter().copied().enumerate() {
            if index != 0 {
                out.write_all(b" ").map_err(HashLifeSnapshotError::io)?;
            }
            write_child_ref(out, child)?;
        }
        out.write_all(b"\n").map_err(HashLifeSnapshotError::io)?;
    }
    Ok(())
}

fn read_bounded_line(
    input: &mut impl BufRead,
    line: &mut Vec<u8>,
) -> Result<Option<String>, HashLifeSnapshotError> {
    line.clear();
    loop {
        let available = input.fill_buf().map_err(HashLifeSnapshotError::io)?;
        if available.is_empty() {
            break;
        }
        let consumed = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |index| index + 1);
        if line.len().saturating_add(consumed) > MAX_SNAPSHOT_LINE_BYTES {
            return Err(HashLifeSnapshotError::new(format!(
                "snapshot line exceeds {MAX_SNAPSHOT_LINE_BYTES} bytes"
            )));
        }
        line.extend_from_slice(&available[..consumed]);
        input.consume(consumed);
        if line.last() == Some(&b'\n') {
            break;
        }
    }
    if line.is_empty() {
        return Ok(None);
    }
    if line.last() == Some(&b'\n') {
        line.pop();
    }
    if line.last() == Some(&b'\r') {
        line.pop();
    }
    String::from_utf8(line.clone())
        .map(Some)
        .map_err(|_| HashLifeSnapshotError::new("snapshot is not valid UTF-8"))
}

pub(super) fn read_snapshot_with_limit(
    reader: impl Read,
    max_import_bytes: u128,
) -> Result<ParsedHashLifeSnapshot, HashLifeSnapshotError> {
    if max_import_bytes < SNAPSHOT_PARSE_SCRATCH_BYTES {
        return Err(HashLifeSnapshotError::allocation(
            SNAPSHOT_PARSE_SCRATCH_BYTES,
        ));
    }
    let mut input = BufReader::new(reader);
    let mut buffer = Vec::with_capacity(128);
    let mut next_line = || read_bounded_line(&mut input, &mut buffer);
    match next_line()? {
        Some(line) if line == HASHLIFE_SNAPSHOT_MAGIC => {}
        Some(other) => {
            return Err(HashLifeSnapshotError::new(format!(
                "unrecognized hashlife snapshot header: {other:?}"
            )));
        }
        None => return Err(HashLifeSnapshotError::new("empty hashlife snapshot")),
    }

    let generation_line =
        next_line()?.ok_or_else(|| HashLifeSnapshotError::new("missing generation line"))?;
    let origin_line =
        next_line()?.ok_or_else(|| HashLifeSnapshotError::new("missing origin line"))?;
    let root_line = next_line()?.ok_or_else(|| HashLifeSnapshotError::new("missing root line"))?;
    let nodes_line =
        next_line()?.ok_or_else(|| HashLifeSnapshotError::new("missing nodes line"))?;

    let generation = generation_line
        .strip_prefix("generation ")
        .ok_or_else(|| HashLifeSnapshotError::new("invalid generation line"))?
        .parse::<u64>()
        .map_err(|_| HashLifeSnapshotError::new("invalid generation value"))?;

    let mut origin_tokens = origin_line.split_whitespace();
    let (Some(kind), Some(origin_x), Some(origin_y)) = (
        origin_tokens.next(),
        origin_tokens.next(),
        origin_tokens.next(),
    ) else {
        return Err(HashLifeSnapshotError::new("invalid origin line"));
    };
    if kind != "origin" || origin_tokens.next().is_some() {
        return Err(HashLifeSnapshotError::new("invalid origin line"));
    }
    let origin_x = origin_x
        .parse::<Coord>()
        .map_err(|_| HashLifeSnapshotError::new("invalid snapshot origin x"))?;
    let origin_y = origin_y
        .parse::<Coord>()
        .map_err(|_| HashLifeSnapshotError::new("invalid snapshot origin y"))?;

    let root = SnapshotChildRef::decode(
        root_line
            .strip_prefix("root ")
            .ok_or_else(|| HashLifeSnapshotError::new("invalid root line"))?,
    )?;

    let node_count = nodes_line
        .strip_prefix("nodes ")
        .ok_or_else(|| HashLifeSnapshotError::new("invalid nodes line"))?
        .parse::<usize>()
        .map_err(|_| HashLifeSnapshotError::new("invalid node count"))?;

    let estimated_import_bytes = u128::try_from(node_count)
        .unwrap_or(u128::MAX)
        .saturating_mul(256)
        .saturating_add(1_024);
    let parse_bytes = estimated_import_bytes.saturating_add(SNAPSHOT_PARSE_SCRATCH_BYTES);
    if parse_bytes > max_import_bytes {
        return Err(HashLifeSnapshotError::allocation(parse_bytes));
    }

    let requested_bytes =
        (node_count as u128).saturating_mul(std::mem::size_of::<SnapshotNode>() as u128);
    let mut nodes = Vec::new();
    nodes
        .try_reserve_exact(node_count)
        .map_err(|_| HashLifeSnapshotError::allocation(requested_bytes))?;
    let mut record_index = 0_usize;
    while let Some(line) = next_line()? {
        if line.trim().is_empty() {
            continue;
        }
        if record_index == node_count {
            return Err(HashLifeSnapshotError::new(format!(
                "snapshot node count exceeds declared count {node_count}"
            )));
        }
        let mut tokens = line.split_whitespace();
        let Some(kind) = tokens.next() else {
            continue;
        };
        let parts = [
            tokens.next(),
            tokens.next(),
            tokens.next(),
            tokens.next(),
            tokens.next(),
        ];
        if kind != "node" || parts.iter().any(Option::is_none) || tokens.next().is_some() {
            return Err(HashLifeSnapshotError::new(format!(
                "invalid node record on line {}",
                record_index + 6
            )));
        }
        let [Some(level), Some(nw), Some(ne), Some(sw), Some(se)] = parts else {
            return Err(HashLifeSnapshotError::new("invalid node record"));
        };
        let level = level
            .parse::<u32>()
            .map_err(|_| HashLifeSnapshotError::new("invalid node level"))?;
        let level = ValidatedLevel::new(level).map_err(|_| {
            HashLifeSnapshotError::new(format!(
                "snapshot node level {level} exceeds representable maximum {}",
                super::MAX_COORD_ROOT_LEVEL
            ))
        })?;
        let node = SnapshotNode {
            level,
            children: [
                SnapshotChildRef::decode(nw)?,
                SnapshotChildRef::decode(ne)?,
                SnapshotChildRef::decode(sw)?,
                SnapshotChildRef::decode(se)?,
            ],
        };
        nodes.push(node);
        record_index += 1;
    }

    if nodes.len() != node_count {
        return Err(HashLifeSnapshotError::new(format!(
            "snapshot node count mismatch: expected {node_count}, found {}",
            nodes.len()
        )));
    }

    for (index, node) in nodes.iter().enumerate() {
        for child in node.children {
            if let SnapshotNodeRef::Node(child_index) = child.node
                && child_index as usize >= index
            {
                return Err(HashLifeSnapshotError::new(format!(
                    "snapshot child reference N{child_index} is not topologically earlier than node N{index}"
                )));
            }
            let child_level = match child.node {
                SnapshotNodeRef::DeadLeaf | SnapshotNodeRef::LiveLeaf => 0,
                SnapshotNodeRef::Node(child_index) => nodes[child_index as usize].level.get(),
            };
            if child_level + 1 != node.level.get() {
                return Err(HashLifeSnapshotError::new(format!(
                    "snapshot node N{index} level {} requires children one level lower, found {child_level}",
                    node.level.get()
                )));
            }
        }
    }
    if let SnapshotNodeRef::Node(root_index) = root.node
        && root_index as usize >= nodes.len()
    {
        return Err(HashLifeSnapshotError::new(format!(
            "snapshot root reference N{root_index} exceeds node table"
        )));
    }

    Ok(ParsedHashLifeSnapshot {
        generation,
        origin_x,
        origin_y,
        root,
        nodes,
    })
}

#[cfg(test)]
pub(super) fn read_snapshot(
    reader: impl Read,
) -> Result<ParsedHashLifeSnapshot, HashLifeSnapshotError> {
    read_snapshot_with_limit(reader, super::session_types::DEFAULT_HARD_MEMORY_BYTES)
}

#[cfg(test)]
fn deserialize_snapshot(s: &str) -> Result<ParsedHashLifeSnapshot, HashLifeSnapshotError> {
    read_snapshot(Cursor::new(s.as_bytes()))
}

impl HashLifeEngine {
    fn snapshot_child_ref_for_node(
        &mut self,
        node: NodeId,
        canonical_indices: &mut ProbeTable<PackedNodeKey, u32>,
        nodes: &mut Vec<SnapshotNode>,
    ) -> Result<SnapshotChildRef, HashLifeSnapshotError> {
        enum Work {
            Visit(NodeId),
            Finish {
                packed: PackedNodeKey,
                required_symmetry: Symmetry,
            },
        }

        let capacity = self.node_count().max(2);
        let Some(mut work) = self.try_transient_vec(capacity.saturating_mul(5)) else {
            return Err(HashLifeSnapshotError::allocation(capacity as u128));
        };
        let Some(mut completed) = self.try_transient_vec(capacity.saturating_mul(4)) else {
            return Err(HashLifeSnapshotError::allocation(capacity as u128));
        };
        if !self.try_push_transient(&mut work, Work::Visit(node)) {
            return Err(HashLifeSnapshotError::allocation(capacity as u128));
        }

        while let Some(next) = work.pop() {
            match next {
                Work::Visit(node) => {
                    if self.node_columns.level(node) == 0 {
                        if !self.try_push_transient(
                            &mut completed,
                            SnapshotChildRef {
                                node: if self.node_columns.population(node) == 0 {
                                    SnapshotNodeRef::DeadLeaf
                                } else {
                                    SnapshotNodeRef::LiveLeaf
                                },
                                symmetry: Symmetry::Identity,
                            },
                        ) {
                            return Err(HashLifeSnapshotError::allocation(capacity as u128));
                        }
                        continue;
                    }

                    let canonical = self
                        .canonicalize_packed_key_for_snapshot(self.node_columns.packed_key(node));
                    let required_symmetry = canonical.node.symmetry.inverse();
                    if let Some(existing) = canonical_indices.get(&canonical.node.packed) {
                        if !self.try_push_transient(
                            &mut completed,
                            SnapshotChildRef {
                                node: SnapshotNodeRef::Node(existing),
                                symmetry: required_symmetry,
                            },
                        ) {
                            return Err(HashLifeSnapshotError::allocation(capacity as u128));
                        }
                        continue;
                    }

                    if !self.try_push_transient(
                        &mut work,
                        Work::Finish {
                            packed: canonical.node.packed,
                            required_symmetry,
                        },
                    ) {
                        return Err(HashLifeSnapshotError::allocation(capacity as u128));
                    }
                    for child in canonical.node.packed.children.into_iter().rev() {
                        if !self.try_push_transient(&mut work, Work::Visit(child)) {
                            return Err(HashLifeSnapshotError::allocation(capacity as u128));
                        }
                    }
                }
                Work::Finish {
                    packed,
                    required_symmetry,
                } => {
                    let se = completed
                        .pop()
                        .or_invariant("snapshot southeast child missing");
                    let sw = completed
                        .pop()
                        .or_invariant("snapshot southwest child missing");
                    let ne = completed
                        .pop()
                        .or_invariant("snapshot northeast child missing");
                    let nw = completed
                        .pop()
                        .or_invariant("snapshot northwest child missing");
                    let index = u32::try_from(nodes.len()).map_err(|_| {
                        HashLifeSnapshotError::new("snapshot node table exceeded u32 capacity")
                    })?;
                    if !self.try_push_transient(
                        nodes,
                        SnapshotNode {
                            level: ValidatedLevel::new(packed.level).or_invariant(
                                "engine snapshot node level must satisfy coordinate geometry",
                            ),
                            children: [nw, ne, sw, se],
                        },
                    ) || !self.try_insert_transient_table(canonical_indices, packed, index)
                        || !self.try_push_transient(
                            &mut completed,
                            SnapshotChildRef {
                                node: SnapshotNodeRef::Node(index),
                                symmetry: required_symmetry,
                            },
                        )
                    {
                        return Err(HashLifeSnapshotError::allocation(capacity as u128));
                    }
                }
            }
        }

        debug_assert_eq!(completed.len(), 1);
        Ok(completed
            .pop()
            .or_invariant("snapshot traversal produced no root"))
    }

    pub(super) fn write_snapshot(
        &mut self,
        root: NodeId,
        origin_x: Coord,
        origin_y: Coord,
        generation: u64,
        writer: &mut impl Write,
    ) -> Result<(), HashLifeSnapshotError> {
        let capacity = self.node_count().max(2);
        let Some(mut canonical_indices) = self.try_transient_probe_table(capacity) else {
            return Err(HashLifeSnapshotError::allocation(capacity as u128));
        };
        let Some(mut nodes) = self.try_transient_vec(capacity) else {
            return Err(HashLifeSnapshotError::allocation(capacity as u128));
        };
        let root_ref =
            self.snapshot_child_ref_for_node(root, &mut canonical_indices, &mut nodes)?;
        write_snapshot(
            &ParsedHashLifeSnapshot {
                generation,
                origin_x,
                origin_y,
                root: root_ref,
                nodes,
            },
            writer,
        )
    }

    fn import_snapshot_child_ref(
        &mut self,
        child: SnapshotChildRef,
        canonical_nodes: &[NodeId],
    ) -> Result<NodeId, HashLifeSnapshotError> {
        let node = match child.node {
            SnapshotNodeRef::DeadLeaf => self.dead_leaf,
            SnapshotNodeRef::LiveLeaf => self.live_leaf,
            SnapshotNodeRef::Node(index) => canonical_nodes
                .get(index as usize)
                .copied()
                .ok_or_else(|| {
                    HashLifeSnapshotError::new(format!(
                        "snapshot node reference N{index} is out of range"
                    ))
                })?,
        };
        if child.symmetry == Symmetry::Identity {
            Ok(node)
        } else {
            let packed = self.node_columns.packed_key(node);
            Ok(self.materialize_oriented_packed_result(packed, Symmetry::Identity, child.symmetry))
        }
    }

    pub(super) fn import_snapshot(
        &mut self,
        snapshot: ParsedHashLifeSnapshot,
    ) -> Result<(NodeId, Coord, Coord, u64), HashLifeSnapshotError> {
        self.with_transient_allocation_scope(|engine| engine.import_snapshot_reserved(snapshot))
    }

    fn import_snapshot_reserved(
        &mut self,
        snapshot: ParsedHashLifeSnapshot,
    ) -> Result<(NodeId, Coord, Coord, u64), HashLifeSnapshotError> {
        let snapshot_bytes = snapshot.allocated_bytes();
        if !self.reserve_transient_bytes(snapshot_bytes) {
            return Err(HashLifeSnapshotError::allocation(snapshot_bytes));
        }
        let root_level = match snapshot.root.node {
            SnapshotNodeRef::DeadLeaf | SnapshotNodeRef::LiveLeaf => 0,
            SnapshotNodeRef::Node(index) => {
                let index = usize::try_from(index).map_err(|_| {
                    HashLifeSnapshotError::new("snapshot root index exceeds platform capacity")
                })?;
                snapshot
                    .nodes
                    .get(index)
                    .map(|node| node.level.get())
                    .ok_or_else(|| HashLifeSnapshotError::new("snapshot root node is missing"))?
            }
        };
        RootGeometry::new(root_level, snapshot.origin_x, snapshot.origin_y).map_err(|error| {
            HashLifeSnapshotError::new(format!("snapshot root geometry is invalid: {error:?}"))
        })?;
        let requested_bytes =
            (snapshot.nodes.len() as u128).saturating_mul(std::mem::size_of::<NodeId>() as u128);
        let mut canonical_nodes = self
            .try_transient_vec::<NodeId>(snapshot.nodes.len())
            .ok_or_else(|| HashLifeSnapshotError::allocation(requested_bytes))?;

        for node in snapshot.nodes {
            let children = [
                self.import_snapshot_child_ref(node.children[0], &canonical_nodes)?,
                self.import_snapshot_child_ref(node.children[1], &canonical_nodes)?,
                self.import_snapshot_child_ref(node.children[2], &canonical_nodes)?,
                self.import_snapshot_child_ref(node.children[3], &canonical_nodes)?,
            ];
            let imported = self.join(children[0], children[1], children[2], children[3]);
            if self.node_columns.level(imported) != node.level.get() {
                return Err(HashLifeSnapshotError::new(format!(
                    "snapshot node level mismatch: expected {}, reconstructed {}",
                    node.level.get(),
                    self.node_columns.level(imported)
                )));
            }
            canonical_nodes.push(imported);
        }

        let root = self.import_snapshot_child_ref(snapshot.root, &canonical_nodes)?;
        Ok((
            root,
            snapshot.origin_x,
            snapshot.origin_y,
            snapshot.generation,
        ))
    }
}

pub fn serialize_grid(grid: &BitGrid) -> Result<String, HashLifeSnapshotError> {
    let mut session = HashLifeSession::new();
    session.try_load_grid(grid).map_err(|error| {
        HashLifeSnapshotError::new(format!("snapshot grid conversion failed: {error:?}"))
    })?;
    session.export_snapshot_string().and_then(|snapshot| {
        snapshot.ok_or_else(|| HashLifeSnapshotError::new("snapshot grid was not loaded"))
    })
}

pub fn serialize_grid_to_writer(
    grid: &BitGrid,
    writer: &mut impl Write,
) -> Result<(), HashLifeSnapshotError> {
    let mut session = HashLifeSession::new();
    session.try_load_grid(grid).map_err(|error| {
        HashLifeSnapshotError::new(format!("snapshot grid conversion failed: {error:?}"))
    })?;
    if session.write_snapshot(writer)? {
        Ok(())
    } else {
        Err(HashLifeSnapshotError::new("snapshot grid was not loaded"))
    }
}

pub fn deserialize_from_reader(reader: impl Read) -> Result<BitGrid, HashLifeSnapshotError> {
    let mut session = HashLifeSession::new();
    session
        .load_snapshot_reader(reader)
        .map_err(snapshot_conversion_error)?;
    session.sample_grid().map_err(|error| {
        HashLifeSnapshotError::new(format!("snapshot grid extraction failed: {error:?}"))
    })
}

pub fn deserialize_to_grid(s: &str) -> Result<BitGrid, HashLifeSnapshotError> {
    deserialize_from_reader(Cursor::new(s.as_bytes()))
}

fn snapshot_conversion_error(error: HashLifeConversionError) -> HashLifeSnapshotError {
    match error {
        HashLifeConversionError::Cancelled => {
            HashLifeSnapshotError::new("snapshot conversion cancelled")
        }
        HashLifeConversionError::Snapshot(error) => error,
        HashLifeConversionError::MemoryBudgetExceeded { .. }
        | HashLifeConversionError::AllocationFailed { .. }
        | HashLifeConversionError::NodeIdExhausted
        | HashLifeConversionError::CanonicalReferenceExhausted
        | HashLifeConversionError::CoordinateRangeExceeded { .. } => {
            HashLifeSnapshotError::new(format!("snapshot resource failure: {error:?}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequiredErrorExt;

    #[test]
    fn deep_sparse_snapshot_is_topological_and_roundtrips() {
        let distant = 1_i64 << 40;
        let grid = BitGrid::from_cells(&[(0, 0), (distant, distant)]);

        let serialized = serialize_grid(&grid).or_invariant("snapshot should serialize");
        let snapshot = deserialize_snapshot(&serialized).or_invariant("snapshot should parse");
        assert!(
            snapshot
                .nodes
                .last()
                .is_some_and(|node| node.level.get() >= 40)
        );
        for (index, node) in snapshot.nodes.iter().enumerate() {
            let index = u32::try_from(index).or_invariant("test snapshot index exceeds u32");
            for child in node.children {
                if let SnapshotNodeRef::Node(child_index) = child.node {
                    assert!(child_index < index);
                }
            }
        }

        let restored = deserialize_to_grid(&serialized).or_invariant("snapshot should roundtrip");
        assert_eq!(restored, grid);
    }

    #[test]
    fn snapshot_rejects_unrepresentable_level_before_reconstruction() {
        let serialized = serialize_grid(&BitGrid::from_cells(&[(0, 0), (1, 1)]))
            .or_invariant("snapshot should serialize");
        let mut replaced = false;
        let malformed = serialized
            .lines()
            .map(|line| {
                if !replaced && line.starts_with("node ") {
                    replaced = true;
                    let children = line
                        .split_whitespace()
                        .skip(2)
                        .collect::<Vec<_>>()
                        .join(" ");
                    format!("node 63 {children}")
                } else {
                    line.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(replaced, "fixture must contain a node record");

        let error = deserialize_snapshot(&malformed).error_or_invariant("expected level error");
        assert!(
            error
                .to_string()
                .contains("exceeds representable maximum 62"),
            "unexpected snapshot error: {error}"
        );
    }

    #[test]
    fn streaming_snapshot_reader_rejects_oversized_lines() {
        let oversized = format!(
            "{HASHLIFE_SNAPSHOT_MAGIC}\ngeneration 0\norigin 0 0\nroot D@0\nnodes 0\n{}\n",
            "x".repeat(MAX_SNAPSHOT_LINE_BYTES + 1)
        );

        let error = read_snapshot(Cursor::new(oversized.as_bytes()))
            .error_or_invariant("oversized snapshot line should fail");

        assert!(
            error.to_string().contains("snapshot line exceeds"),
            "oversized line returned the wrong error: {error}"
        );
    }

    #[test]
    fn snapshot_rejects_root_endpoint_overflow_before_engine_mutation() {
        let serialized = serialize_grid(&BitGrid::from_cells(&[(0, 0), (1, 1)]))
            .or_invariant("snapshot should serialize");
        let malformed = serialized
            .lines()
            .map(|line| {
                if line.starts_with("origin ") {
                    format!("origin {} {}", Coord::MAX, Coord::MAX)
                } else {
                    line.to_owned()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        let mut session = HashLifeSession::new();
        session
            .try_load_grid(&BitGrid::from_cells(&[
                (10, 10),
                (11, 10),
                (10, 11),
                (11, 11),
            ]))
            .or_invariant("authoritative fixture should load");
        let snapshot_before = session
            .export_snapshot_string()
            .or_invariant("authoritative fixture should serialize");

        let error = session
            .load_snapshot_string(&malformed)
            .error_or_invariant("overflowing root interval should fail");

        assert!(
            matches!(
                &error,
                HashLifeConversionError::Snapshot(snapshot)
                    if snapshot.to_string().contains("root geometry is invalid")
            ),
            "unexpected snapshot geometry error: {error:?}"
        );
        assert_eq!(
            session
                .export_snapshot_string()
                .or_invariant("authoritative fixture should remain serializable"),
            snapshot_before,
            "invalid external geometry changed the authoritative session"
        );
    }

    #[test]
    fn snapshot_export_allocation_failure_preserves_authoritative_state() {
        let mut session = HashLifeSession::new();
        session
            .try_load_grid(&BitGrid::from_cells(&[(0, 0), (1, 0), (2, 0)]))
            .or_invariant("snapshot fixture should load");
        session
            .advance_root(1)
            .or_invariant("snapshot fixture should advance");
        let generation = session.generation();
        let checkpoint = *session
            .signature_checkpoint()
            .or_invariant("snapshot fixture should have a checkpoint");
        session.configure_allocation_failure(Some(HashLifeAllocationFailure {
            class: HashLifeAllocationClass::SnapshotExport,
            ordinal: 1,
        }));

        let error = session
            .export_snapshot_string()
            .error_or_invariant("injected snapshot allocation should fail");

        assert!(
            error.allocation_bytes().is_some(),
            "snapshot failure lost allocation context: {error}"
        );
        assert_eq!(
            session.generation(),
            generation,
            "export changed generation"
        );
        assert_eq!(
            session
                .signature_checkpoint()
                .or_invariant("failed export should retain checkpoint"),
            &checkpoint,
            "failed export changed authoritative checkpoint identity"
        );
    }
}
