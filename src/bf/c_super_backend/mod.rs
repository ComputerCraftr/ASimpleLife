use crate::RequiredExt;
use std::collections::HashMap;

use super::c_support::normalized_c_offset;
use super::ir::{BfIr, ShiftDir, validate_canonical_ir};
use super::summary::SymbolicTransfer;
use super::symbolic::{SYMBOLIC_TERM_MAX, SymbolicMonomial, SymbolicPolynomial};

mod analysis;
mod emit;
mod planner;

pub use emit::emit_c_super;

pub(super) const SUPER_MEMO_WINDOW_MAX: crate::bf::BfOffset = 8;
pub(super) const SUPER_MEMO_CAPACITY: usize = 4096;
pub(super) const SUPER_C_TAPE_LEN: usize = super::BF_C_TAPE_LEN;
pub(super) const SUPER_SEQ_TILE_WIDTH: usize = 8;
pub(super) const SUPER_LOOP_POWER_MAX: u8 = 62;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct NodeId(pub(super) u32);

/// Interned DAG form of optimized `BfIr`.
///
/// This preserves BF opcode semantics while adding explicit shared `Seq` nodes
/// so the super backend can do bottom-up memoized planning over a canonical DAG.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) enum NodeKind {
    Add(i32),
    Move(crate::bf::BfOffset),
    Input,
    Output,
    Clear,
    ClearAt(crate::bf::BfOffset),
    Scan {
        stride: crate::bf::BfOffset,
    },
    Distribute {
        targets: Vec<(crate::bf::BfOffset, i32)>,
        preserve_src: bool,
    },
    Affine {
        src: crate::bf::BfOffset,
        dst: crate::bf::BfOffset,
        coeff: i32,
        preserve_src: bool,
        set_dst: bool,
    },
    Shift {
        src: crate::bf::BfOffset,
        dst: crate::bf::BfOffset,
        amount: u32,
        dir: ShiftDir,
        preserve_src: bool,
        set_dst: bool,
    },
    Square {
        src: crate::bf::BfOffset,
        dst: crate::bf::BfOffset,
        preserve_src: bool,
        set_dst: bool,
    },
    MulAdd {
        lhs: crate::bf::BfOffset,
        rhs: crate::bf::BfOffset,
        dst: crate::bf::BfOffset,
        preserve_lhs: bool,
        preserve_rhs: bool,
        set_dst: bool,
    },
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
        let id =
            NodeId(u32::try_from(self.nodes.len()).or_invariant("super-C node count exceeded u32"));
        self.nodes.push(kind.clone());
        self.map.insert(kind, id);
        id
    }

    pub(super) fn get(&self, id: NodeId) -> &NodeKind {
        &self.nodes[usize::try_from(id.0).or_invariant("u32 node id exceeded usize")]
    }

    pub(super) fn len(&self) -> usize {
        self.nodes.len()
    }
}

impl SymbolicTransfer {
    pub(super) fn memo_window(&self) -> Option<MemoWindow> {
        if !self.is_pure_windowed() || self.ptr_delta != 0 {
            return None;
        }
        let accessed = self.accessed_offsets();
        let min = *accessed.iter().next()?;
        let max = *accessed.iter().next_back()?;
        i32::try_from(min).ok()?;
        i32::try_from(max).ok()?;
        let span = max.checked_sub(min)?.checked_add(1)?;
        if span <= 0 || span > SUPER_MEMO_WINDOW_MAX {
            return None;
        }
        Some(MemoWindow {
            start: min,
            len: u8::try_from(span).or_invariant("validated memo span exceeded u8"),
        })
    }

    pub(super) fn is_direct_kernel_loop_shape(&self) -> bool {
        self.is_pure_windowed() && self.ptr_delta == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MemoWindow {
    pub(super) start: crate::bf::BfOffset,
    pub(super) len: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ExactMemoSpec {
    pub(super) window: MemoWindow,
    pub(super) ptr_delta: crate::bf::BfOffset,
}

#[derive(Clone, Debug)]
pub(super) struct PoweredLoopAnalysis {
    pub(super) body: NodeId,
    pub(super) guard_offset: crate::bf::BfOffset,
    pub(super) guard_delta: i64,
    pub(super) powers: Vec<SymbolicTransfer>,
}

#[derive(Clone, Debug)]
pub(super) enum LoopAnalysis {
    ExactMemoPlusDirectKernel {
        body: NodeId,
        exact: ExactMemoSpec,
    },
    ExactMemoPlusSymbolicPower {
        exact: ExactMemoSpec,
        powered: PoweredLoopAnalysis,
    },
    ExactMemoOnly {
        body: NodeId,
        exact: ExactMemoSpec,
    },
    Residual {
        body: NodeId,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExecPlan {
    Primitive,
    ExactMemo(ExactMemoSpec),
    ExactLoopMemo {
        body: NodeId,
        exact: ExactMemoSpec,
    },
    ExactPoweredLoopMemo {
        body: NodeId,
        exact: ExactMemoSpec,
        max_power: u8,
    },
    Residual,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PlanDecision {
    pub(super) plan: ExecPlan,
    pub(super) estimated_cost: usize,
}

pub(super) struct EmitterEngine {
    pub(super) interner: Interner,
    pub(super) transfers: HashMap<NodeId, SymbolicTransfer>,
    pub(super) exact_specs: HashMap<NodeId, Option<ExactMemoSpec>>,
    pub(super) loop_analyses: HashMap<NodeId, Option<LoopAnalysis>>,
    pub(super) plan_decisions: HashMap<NodeId, PlanDecision>,
}

impl EmitterEngine {
    pub(super) fn new() -> Self {
        Self {
            interner: Interner::default(),
            transfers: HashMap::new(),
            exact_specs: HashMap::new(),
            loop_analyses: HashMap::new(),
            plan_decisions: HashMap::new(),
        }
    }

    pub(super) fn build_program(&mut self, program: &[BfIr]) -> NodeId {
        validate_canonical_ir(program).or_invariant("super C backend requires canonical richer IR");
        let ids = program
            .iter()
            .map(|node| self.build_node(node))
            .collect::<Vec<_>>();
        self.build_tiled_seq(&ids)
    }

    pub(super) fn build_node(&mut self, node: &BfIr) -> NodeId {
        match node {
            BfIr::Add(v) => self.interner.intern(NodeKind::Add(*v)),
            // Preserve represented source movement for semantic-fuel accounting.
            // Physical addressing is normalized only when emitted through bf_wrap_ptr.
            BfIr::MovePtr(v) => self.interner.intern(NodeKind::Move(*v)),
            BfIr::Input => self.interner.intern(NodeKind::Input),
            BfIr::Output => self.interner.intern(NodeKind::Output),
            BfIr::Clear => self.interner.intern(NodeKind::Clear),
            BfIr::ClearAt { offset } => self
                .interner
                .intern(NodeKind::ClearAt(normalized_c_offset(*offset))),
            BfIr::Scan { stride } => self.interner.intern(NodeKind::Scan {
                stride: normalized_c_offset(*stride),
            }),
            BfIr::Distribute {
                targets,
                preserve_src,
            } => self.interner.intern(NodeKind::Distribute {
                targets: targets
                    .iter()
                    .map(|&(offset, coeff)| (normalized_c_offset(offset), coeff))
                    .collect(),
                preserve_src: *preserve_src,
            }),
            BfIr::Affine {
                src,
                dst,
                coeff,
                preserve_src,
                set_dst,
            } => self.interner.intern(NodeKind::Affine {
                src: normalized_c_offset(*src),
                dst: normalized_c_offset(*dst),
                coeff: *coeff,
                preserve_src: *preserve_src,
                set_dst: *set_dst,
            }),
            BfIr::Shift {
                src,
                dst,
                amount,
                dir,
                preserve_src,
                set_dst,
            } => self.interner.intern(NodeKind::Shift {
                src: normalized_c_offset(*src),
                dst: normalized_c_offset(*dst),
                amount: *amount,
                dir: *dir,
                preserve_src: *preserve_src,
                set_dst: *set_dst,
            }),
            BfIr::Square {
                src,
                dst,
                preserve_src,
                set_dst,
            } => self.interner.intern(NodeKind::Square {
                src: normalized_c_offset(*src),
                dst: normalized_c_offset(*dst),
                preserve_src: *preserve_src,
                set_dst: *set_dst,
            }),
            BfIr::MulAdd {
                lhs,
                rhs,
                dst,
                preserve_lhs,
                preserve_rhs,
                set_dst,
            } => self.interner.intern(NodeKind::MulAdd {
                lhs: normalized_c_offset(*lhs),
                rhs: normalized_c_offset(*rhs),
                dst: normalized_c_offset(*dst),
                preserve_lhs: *preserve_lhs,
                preserve_rhs: *preserve_rhs,
                set_dst: *set_dst,
            }),
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
                    let mut next = Vec::with_capacity(current.len().div_ceil(SUPER_SEQ_TILE_WIDTH));
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
