use super::*;
use crate::bitgrid::{BitGrid, Coord};
use crate::persistence::HASHLIFE_SNAPSHOT_MAGIC;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HashLifeSnapshotError {
    message: String,
}

impl HashLifeSnapshotError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
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
    level: u32,
    children: [SnapshotChildRef; 4],
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct HashLifeSnapshot {
    generation: u64,
    origin_x: Coord,
    origin_y: Coord,
    root: SnapshotChildRef,
    nodes: Vec<SnapshotNode>,
}

impl SnapshotNodeRef {
    fn encode(self) -> String {
        match self {
            Self::DeadLeaf => "D".to_string(),
            Self::LiveLeaf => "L".to_string(),
            Self::Node(index) => format!("N{index}"),
        }
    }

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
                        HashLifeSnapshotError::new(format!(
                            "invalid snapshot node index {token:?}"
                        ))
                    })?;
                Ok(Self::Node(index))
            }
        }
    }
}

impl SnapshotChildRef {
    fn encode(self) -> String {
        format!("{}@{}", self.node.encode(), self.symmetry as u8)
    }

    fn decode(token: &str) -> Result<Self, HashLifeSnapshotError> {
        let (node_token, symmetry_token) = token.split_once('@').ok_or_else(|| {
            HashLifeSnapshotError::new(format!(
                "invalid snapshot child reference {token:?}"
            ))
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

fn serialize_snapshot(snapshot: &HashLifeSnapshot) -> String {
    let mut out = String::new();
    out.push_str(HASHLIFE_SNAPSHOT_MAGIC);
    out.push('\n');
    out.push_str(&format!("generation {}\n", snapshot.generation));
    out.push_str(&format!(
        "origin {} {}\n",
        snapshot.origin_x, snapshot.origin_y
    ));
    out.push_str(&format!("root {}\n", snapshot.root.encode()));
    out.push_str(&format!("nodes {}\n", snapshot.nodes.len()));
    for node in &snapshot.nodes {
        out.push_str(&format!(
            "node {} {} {} {} {}\n",
            node.level,
            node.children[0].encode(),
            node.children[1].encode(),
            node.children[2].encode(),
            node.children[3].encode()
        ));
    }
    out
}

fn deserialize_snapshot(s: &str) -> Result<HashLifeSnapshot, HashLifeSnapshotError> {
    let mut lines = s.lines();
    match lines.next() {
        Some(line) if line == HASHLIFE_SNAPSHOT_MAGIC => {}
        Some(other) => {
            return Err(HashLifeSnapshotError::new(format!(
                "unrecognized hashlife snapshot header: {other:?}"
            )));
        }
        None => return Err(HashLifeSnapshotError::new("empty hashlife snapshot")),
    }

    let generation_line = lines
        .next()
        .ok_or_else(|| HashLifeSnapshotError::new("missing generation line"))?;
    let origin_line = lines
        .next()
        .ok_or_else(|| HashLifeSnapshotError::new("missing origin line"))?;
    let root_line = lines
        .next()
        .ok_or_else(|| HashLifeSnapshotError::new("missing root line"))?;
    let nodes_line = lines
        .next()
        .ok_or_else(|| HashLifeSnapshotError::new("missing nodes line"))?;

    let generation = generation_line
        .strip_prefix("generation ")
        .ok_or_else(|| HashLifeSnapshotError::new("invalid generation line"))?
        .parse::<u64>()
        .map_err(|_| HashLifeSnapshotError::new("invalid generation value"))?;

    let origin_tokens: Vec<_> = origin_line.split_whitespace().collect();
    if origin_tokens.len() != 3 || origin_tokens[0] != "origin" {
        return Err(HashLifeSnapshotError::new("invalid origin line"));
    }
    let origin_x = origin_tokens[1]
        .parse::<Coord>()
        .map_err(|_| HashLifeSnapshotError::new("invalid snapshot origin x"))?;
    let origin_y = origin_tokens[2]
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

    let mut nodes = Vec::with_capacity(node_count);
    for (index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<_> = line.split_whitespace().collect();
        if parts.len() != 6 || parts[0] != "node" {
            return Err(HashLifeSnapshotError::new(format!(
                "invalid node record on line {}",
                index + 5
            )));
        }
        let level = parts[1]
            .parse::<u32>()
            .map_err(|_| HashLifeSnapshotError::new("invalid node level"))?;
        let node = SnapshotNode {
            level,
            children: [
                SnapshotChildRef::decode(parts[2])?,
                SnapshotChildRef::decode(parts[3])?,
                SnapshotChildRef::decode(parts[4])?,
                SnapshotChildRef::decode(parts[5])?,
            ],
        };
        nodes.push(node);
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
        }
    }
    if let SnapshotNodeRef::Node(root_index) = root.node
        && root_index as usize >= nodes.len()
    {
        return Err(HashLifeSnapshotError::new(format!(
            "snapshot root reference N{root_index} exceeds node table"
        )));
    }

    Ok(HashLifeSnapshot {
        generation,
        origin_x,
        origin_y,
        root,
        nodes,
    })
}

impl HashLifeEngine {
    fn snapshot_child_ref_for_node(
        &mut self,
        node: NodeId,
        canonical_indices: &mut BTreeMap<PackedNodeKey, u32>,
        nodes: &mut Vec<SnapshotNode>,
    ) -> SnapshotChildRef {
        if self.node_columns.level(node) == 0 {
            return SnapshotChildRef {
                node: if self.node_columns.population(node) == 0 {
                    SnapshotNodeRef::DeadLeaf
                } else {
                    SnapshotNodeRef::LiveLeaf
                },
                symmetry: Symmetry::Identity,
            };
        }

        let canonical = self.canonicalize_packed_key_for_snapshot(self.node_columns.packed_key(node));
        let required_symmetry = canonical.node.symmetry.inverse();
        let index = if let Some(existing) = canonical_indices.get(&canonical.node.packed) {
            *existing
        } else {
            let children = canonical.node.packed.children.map(|child| {
                self.snapshot_child_ref_for_node(child, canonical_indices, nodes)
            });
            let index = nodes.len() as u32;
            nodes.push(SnapshotNode {
                level: canonical.node.packed.level,
                children,
            });
            canonical_indices.insert(canonical.node.packed, index);
            index
        };

        SnapshotChildRef {
            node: SnapshotNodeRef::Node(index),
            symmetry: required_symmetry,
        }
    }

    pub(super) fn export_snapshot_string(
        &mut self,
        root: NodeId,
        origin_x: Coord,
        origin_y: Coord,
        generation: u64,
    ) -> String {
        let mut canonical_indices = BTreeMap::new();
        let mut nodes = Vec::new();
        let root_ref = self.snapshot_child_ref_for_node(root, &mut canonical_indices, &mut nodes);
        serialize_snapshot(&HashLifeSnapshot {
            generation,
            origin_x,
            origin_y,
            root: root_ref,
            nodes,
        })
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
            Ok(self.materialize_oriented_packed_result(
                packed,
                Symmetry::Identity,
                child.symmetry,
            ))
        }
    }

    pub(super) fn import_snapshot_string(
        &mut self,
        s: &str,
    ) -> Result<(NodeId, Coord, Coord, u64), HashLifeSnapshotError> {
        let snapshot = deserialize_snapshot(s)?;
        let mut canonical_nodes = Vec::with_capacity(snapshot.nodes.len());

        for node in snapshot.nodes {
            let children = [
                self.import_snapshot_child_ref(node.children[0], &canonical_nodes)?,
                self.import_snapshot_child_ref(node.children[1], &canonical_nodes)?,
                self.import_snapshot_child_ref(node.children[2], &canonical_nodes)?,
                self.import_snapshot_child_ref(node.children[3], &canonical_nodes)?,
            ];
            let imported = self.join(children[0], children[1], children[2], children[3]);
            if self.node_columns.level(imported) != node.level {
                return Err(HashLifeSnapshotError::new(format!(
                    "snapshot node level mismatch: expected {}, reconstructed {}",
                    node.level,
                    self.node_columns.level(imported)
                )));
            }
            canonical_nodes.push(imported);
        }

        let root = self.import_snapshot_child_ref(snapshot.root, &canonical_nodes)?;
        Ok((root, snapshot.origin_x, snapshot.origin_y, snapshot.generation))
    }
}

pub fn serialize_grid(grid: &BitGrid) -> String {
    let mut session = HashLifeSession::new();
    session.load_grid(grid);
    session
        .export_snapshot_string()
        .expect("loaded hashlife session should be exportable")
}

pub fn deserialize_to_grid(s: &str) -> Result<BitGrid, HashLifeSnapshotError> {
    let mut session = HashLifeSession::new();
    session.load_snapshot_string(s)?;
    session
        .sample_grid()
        .ok_or_else(|| HashLifeSnapshotError::new("snapshot deserialized to an empty session"))
}
