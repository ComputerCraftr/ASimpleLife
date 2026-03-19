use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::ir::BfIr;

mod analysis;
mod emit;
mod planner;

pub use emit::emit_c_super;

pub(super) const SUPER_MEMO_WINDOW_MAX: isize = 8;
pub(super) const SUPER_MEMO_CAPACITY: usize = 4096;
pub(super) const SUPER_C_TAPE_LEN: usize = 30_000;
pub(super) const SUPER_SEQ_TILE_WIDTH: usize = 8;
pub(super) const SUPER_LOOP_POWER_MAX: u8 = 15;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct NodeId(pub(super) u32);

/// Interned DAG form of optimized `BfIr`.
///
/// This preserves BF opcode semantics while adding explicit shared `Seq` nodes
/// so the super backend can do bottom-up memoized planning over a canonical DAG.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) enum NodeKind {
    Add(i32),
    Move(isize),
    Input,
    Output,
    Clear,
    Distribute(Vec<(isize, i32)>),
    Diverge,
    Seq(Vec<NodeId>),
    Loop(NodeId),
}

#[derive(Default)]
pub(super) struct Interner {
    pub(super) nodes: Vec<NodeKind>,
    pub(super) map: HashMap<NodeKind, NodeId>,
}

impl Interner {
    pub(super) fn intern(&mut self, kind: NodeKind) -> NodeId {
        if let Some(&id) = self.map.get(&kind) {
            return id;
        }
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(kind.clone());
        self.map.insert(kind, id);
        id
    }

    pub(super) fn get(&self, id: NodeId) -> &NodeKind {
        &self.nodes[id.0 as usize]
    }

    pub(super) fn len(&self) -> usize {
        self.nodes.len()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum SymbolicCellEffect {
    AddConst(i64),
    Clear,
    AddScaledSource { source_offset: isize, coeff: i32 },
    Opaque,
}

#[derive(Clone, Debug)]
pub(super) struct SymbolicTransfer {
    pub(super) ptr_delta: isize,
    pub(super) effects: BTreeMap<isize, SymbolicCellEffect>,
    pub(super) reads: BTreeSet<isize>,
    pub(super) may_input: bool,
    pub(super) may_output: bool,
    pub(super) may_diverge: bool,
    pub(super) unknown: bool,
}

impl SymbolicTransfer {
    pub(super) fn identity() -> Self {
        Self {
            ptr_delta: 0,
            effects: BTreeMap::new(),
            reads: BTreeSet::new(),
            may_input: false,
            may_output: false,
            may_diverge: false,
            unknown: false,
        }
    }

    pub(super) fn unknown() -> Self {
        Self {
            ptr_delta: 0,
            effects: BTreeMap::new(),
            reads: BTreeSet::new(),
            may_input: false,
            may_output: false,
            may_diverge: false,
            unknown: true,
        }
    }

    pub(super) fn accessed_offsets(&self) -> BTreeSet<isize> {
        let mut accessed = self.reads.clone();
        accessed.extend(self.effects.keys().copied());
        accessed
    }

    pub(super) fn is_pure_windowed(&self) -> bool {
        !self.unknown && !self.may_input && !self.may_output && !self.may_diverge
    }

    pub(super) fn memo_window(&self) -> Option<MemoWindow> {
        if !self.is_pure_windowed() || self.ptr_delta != 0 {
            return None;
        }
        let accessed = self.accessed_offsets();
        let min = *accessed.iter().next()?;
        let max = *accessed.iter().next_back()?;
        let span = max - min + 1;
        if span <= 0 || span > SUPER_MEMO_WINDOW_MAX {
            return None;
        }
        Some(MemoWindow {
            start: min,
            len: span as u8,
        })
    }

    pub(super) fn has_opaque_effects(&self) -> bool {
        self.effects
            .values()
            .any(|effect| matches!(effect, SymbolicCellEffect::Opaque))
    }

    pub(super) fn is_direct_kernel_loop_shape(&self) -> bool {
        self.is_pure_windowed() && self.ptr_delta == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MemoWindow {
    pub(super) start: isize,
    pub(super) len: u8,
}

#[derive(Clone, Debug)]
pub(super) struct PoweredLoopAnalysis {
    pub(super) body: NodeId,
    pub(super) window: MemoWindow,
    pub(super) guard_offset: isize,
    pub(super) guard_delta: i64,
    pub(super) powers: Vec<SymbolicTransfer>,
}

#[derive(Clone, Debug)]
pub(super) enum LoopAnalysis {
    DirectKernel {
        body: NodeId,
        accessed_offsets: BTreeSet<isize>,
    },
    Powered(PoweredLoopAnalysis),
    Symbolic {
        body: NodeId,
        window: MemoWindow,
    },
    Residual {
        body: NodeId,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum ExecPlan {
    Primitive,
    DirectKernel(&'static str),
    SymbolicMemo(MemoWindow),
    SymbolicLoop { body: NodeId, window: MemoWindow },
    PoweredSymbolicLoop {
        body: NodeId,
        window: MemoWindow,
        max_power: u8,
    },
    Residual,
}

#[derive(Clone, Debug)]
pub(super) struct PlanDecision {
    pub(super) plan: ExecPlan,
    pub(super) estimated_cost: usize,
}

pub(super) struct EmitterEngine {
    pub(super) interner: Interner,
    pub(super) transfers: HashMap<NodeId, SymbolicTransfer>,
    pub(super) loop_analyses: HashMap<NodeId, Option<LoopAnalysis>>,
    pub(super) plan_decisions: HashMap<NodeId, PlanDecision>,
}

impl EmitterEngine {
    pub(super) fn new() -> Self {
        Self {
            interner: Interner::default(),
            transfers: HashMap::new(),
            loop_analyses: HashMap::new(),
            plan_decisions: HashMap::new(),
        }
    }

    pub(super) fn build_program(&mut self, program: &[BfIr]) -> NodeId {
        let ids = program
            .iter()
            .map(|node| self.build_node(node))
            .collect::<Vec<_>>();
        self.build_tiled_seq(&ids)
    }

    pub(super) fn build_node(&mut self, node: &BfIr) -> NodeId {
        match node {
            BfIr::Add(v) => self.interner.intern(NodeKind::Add(*v)),
            BfIr::MovePtr(v) => self.interner.intern(NodeKind::Move(*v)),
            BfIr::Input => self.interner.intern(NodeKind::Input),
            BfIr::Output => self.interner.intern(NodeKind::Output),
            BfIr::Clear => self.interner.intern(NodeKind::Clear),
            BfIr::Distribute { targets } => self.interner.intern(NodeKind::Distribute(targets.clone())),
            BfIr::Diverge => self.interner.intern(NodeKind::Diverge),
            BfIr::Loop(body) => {
                let body_id = self.build_program(body);
                self.interner.intern(NodeKind::Loop(body_id))
            }
        }
    }

    pub(super) fn build_tiled_seq(&mut self, ids: &[NodeId]) -> NodeId {
        match ids.len() {
            0 => self.interner.intern(NodeKind::Seq(Vec::new())),
            1 => ids[0],
            _ => {
                let mut current = ids.to_vec();
                while current.len() > SUPER_SEQ_TILE_WIDTH {
                    let mut next =
                        Vec::with_capacity(current.len().div_ceil(SUPER_SEQ_TILE_WIDTH));
                    for chunk in current.chunks(SUPER_SEQ_TILE_WIDTH) {
                        if chunk.len() == 1 {
                            next.push(chunk[0]);
                        } else {
                            next.push(self.interner.intern(NodeKind::Seq(chunk.to_vec())));
                        }
                    }
                    current = next;
                }
                self.interner.intern(NodeKind::Seq(current))
            }
        }
    }
}
