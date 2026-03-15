use std::collections::{BTreeMap, BTreeSet};

use super::ir::{BfIr, ShiftDir};
use super::symbolic::SymbolicPolynomial;

pub(crate) const RUNTIME_SUMMARY_WINDOW_MAX: crate::bf::BfOffset = 8;
pub(crate) const RUNTIME_SUMMARY_EFFECT_MAX: usize = 16;
pub(crate) const DEFAULT_DYNAMIC_LOOP_HOT_THRESHOLD: u32 = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LoopId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SummaryProvenance {
    Static,
    Runtime,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SummaryGuard {
    Unconditional,
    Structural {
        body_hash: u64,
        cell_bits: u8,
        signed_cells: bool,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SummaryEffect {
    Clear {
        offset: crate::bf::BfOffset,
    },
    AddConst {
        offset: crate::bf::BfOffset,
        delta: i64,
    },
    AddScaled {
        src: crate::bf::BfOffset,
        dst: crate::bf::BfOffset,
        coeff: i32,
        set_dst: bool,
    },
    Shift {
        src: crate::bf::BfOffset,
        dst: crate::bf::BfOffset,
        amount: u32,
        dir: ShiftDir,
        set_dst: bool,
    },
    Square {
        src: crate::bf::BfOffset,
        dst: crate::bf::BfOffset,
        set_dst: bool,
    },
    AddProduct {
        lhs: crate::bf::BfOffset,
        rhs: crate::bf::BfOffset,
        dst: crate::bf::BfOffset,
        coeff: i32,
        set_dst: bool,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LoopSummary {
    pub id: LoopId,
    pub provenance: SummaryProvenance,
    pub touched_offsets: Vec<crate::bf::BfOffset>,
    pub read_offsets: Vec<crate::bf::BfOffset>,
    pub write_offsets: Vec<crate::bf::BfOffset>,
    pub exit_pointer_delta: crate::bf::BfOffset,
    pub effects: Vec<SummaryEffect>,
    pub guard: SummaryGuard,
}

impl LoopSummary {
    pub(crate) fn from_ir_nodes(
        id: LoopId,
        provenance: SummaryProvenance,
        nodes: &[BfIr],
    ) -> Result<Self, String> {
        let mut effects = Vec::new();
        let mut touched = BTreeSet::new();
        let mut reads = BTreeSet::new();
        let mut writes = BTreeSet::new();
        for node in nodes {
            match node {
                BfIr::Add(delta) => {
                    touched.insert(0);
                    reads.insert(0);
                    writes.insert(0);
                    effects.push(SummaryEffect::AddConst {
                        offset: 0,
                        delta: i64::from(*delta),
                    });
                }
                BfIr::Clear => {
                    touched.insert(0);
                    writes.insert(0);
                    effects.push(SummaryEffect::Clear { offset: 0 });
                }
                BfIr::ClearAt { offset } => {
                    touched.insert(*offset);
                    writes.insert(*offset);
                    effects.push(SummaryEffect::Clear { offset: *offset });
                }
                BfIr::Distribute {
                    targets,
                    preserve_src,
                } => {
                    touched.insert(0);
                    reads.insert(0);
                    for &(dst, coeff) in targets {
                        if dst == 0 {
                            return Err("distribute summary targets must not alias their source"
                                .to_string());
                        }
                        touched.insert(dst);
                        reads.insert(dst);
                        writes.insert(dst);
                        effects.push(SummaryEffect::AddScaled {
                            src: 0,
                            dst,
                            coeff,
                            set_dst: false,
                        });
                    }
                    if !preserve_src {
                        writes.insert(0);
                        effects.push(SummaryEffect::Clear { offset: 0 });
                    }
                }
                BfIr::Affine {
                    src,
                    dst,
                    coeff,
                    preserve_src,
                    set_dst,
                } => {
                    touched.extend([*src, *dst]);
                    reads.insert(*src);
                    if !set_dst {
                        reads.insert(*dst);
                    }
                    writes.insert(*dst);
                    effects.push(SummaryEffect::AddScaled {
                        src: *src,
                        dst: *dst,
                        coeff: *coeff,
                        set_dst: *set_dst,
                    });
                    if !preserve_src && src != dst {
                        writes.insert(*src);
                        effects.push(SummaryEffect::Clear { offset: *src });
                    }
                }
                BfIr::Shift {
                    src,
                    dst,
                    amount,
                    dir,
                    preserve_src,
                    set_dst,
                } => {
                    touched.extend([*src, *dst]);
                    reads.insert(*src);
                    if !set_dst {
                        reads.insert(*dst);
                    }
                    writes.insert(*dst);
                    effects.push(SummaryEffect::Shift {
                        src: *src,
                        dst: *dst,
                        amount: *amount,
                        dir: *dir,
                        set_dst: *set_dst,
                    });
                    if !preserve_src && src != dst {
                        writes.insert(*src);
                        effects.push(SummaryEffect::Clear { offset: *src });
                    }
                }
                BfIr::Square {
                    src,
                    dst,
                    preserve_src,
                    set_dst,
                } => {
                    touched.extend([*src, *dst]);
                    reads.insert(*src);
                    if !set_dst {
                        reads.insert(*dst);
                    }
                    writes.insert(*dst);
                    effects.push(SummaryEffect::Square {
                        src: *src,
                        dst: *dst,
                        set_dst: *set_dst,
                    });
                    if !preserve_src && src != dst {
                        writes.insert(*src);
                        effects.push(SummaryEffect::Clear { offset: *src });
                    }
                }
                BfIr::MulAdd {
                    lhs,
                    rhs,
                    dst,
                    preserve_lhs,
                    preserve_rhs,
                    set_dst,
                } => {
                    touched.extend([*lhs, *rhs, *dst]);
                    reads.extend([*lhs, *rhs]);
                    if !set_dst {
                        reads.insert(*dst);
                    }
                    writes.insert(*dst);
                    effects.push(SummaryEffect::AddProduct {
                        lhs: *lhs,
                        rhs: *rhs,
                        dst: *dst,
                        coeff: 1,
                        set_dst: *set_dst,
                    });
                    if !preserve_lhs && lhs != dst {
                        writes.insert(*lhs);
                        effects.push(SummaryEffect::Clear { offset: *lhs });
                    }
                    if !preserve_rhs && rhs != dst && rhs != lhs {
                        writes.insert(*rhs);
                        effects.push(SummaryEffect::Clear { offset: *rhs });
                    }
                }
                BfIr::MovePtr(0) => {}
                _ => {
                    return Err(format!(
                        "node is not a closed-form summary effect: {node:?}"
                    ));
                }
            }
        }
        let summary = Self {
            id,
            provenance,
            touched_offsets: touched.into_iter().collect(),
            read_offsets: reads.into_iter().collect(),
            write_offsets: writes.into_iter().collect(),
            exit_pointer_delta: 0,
            effects,
            guard: SummaryGuard::Unconditional,
        };
        summary.validate()?;
        Ok(summary)
    }

    pub(crate) fn rebase(&mut self, delta: crate::bf::BfOffset) -> Result<(), String> {
        fn shifted(
            offset: &mut crate::bf::BfOffset,
            delta: crate::bf::BfOffset,
        ) -> Result<(), String> {
            *offset = offset
                .checked_add(delta)
                .ok_or_else(|| "summary offset overflow while rebasing".to_string())?;
            Ok(())
        }

        for effect in &mut self.effects {
            match effect {
                SummaryEffect::Clear { offset } | SummaryEffect::AddConst { offset, .. } => {
                    shifted(offset, delta)?;
                }
                SummaryEffect::AddScaled { src, dst, .. }
                | SummaryEffect::Shift { src, dst, .. }
                | SummaryEffect::Square { src, dst, .. } => {
                    shifted(src, delta)?;
                    shifted(dst, delta)?;
                }
                SummaryEffect::AddProduct { lhs, rhs, dst, .. } => {
                    shifted(lhs, delta)?;
                    shifted(rhs, delta)?;
                    shifted(dst, delta)?;
                }
            }
        }
        self.rebuild_metadata();
        self.validate()
    }

    fn rebuild_metadata(&mut self) {
        let (reads, writes) = effect_metadata(&self.effects);
        self.touched_offsets = reads.union(&writes).copied().collect();
        self.read_offsets = reads.into_iter().collect();
        self.write_offsets = writes.into_iter().collect();
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.exit_pointer_delta != 0 {
            return Err("loop summary must return to its entry pointer".to_string());
        }
        fn sorted_unique(values: &[crate::bf::BfOffset]) -> bool {
            values.windows(2).all(|pair| pair[0] < pair[1])
        }

        if !sorted_unique(&self.touched_offsets)
            || !sorted_unique(&self.read_offsets)
            || !sorted_unique(&self.write_offsets)
        {
            return Err("summary offset sets must be sorted and unique".to_string());
        }

        let (reads, writes) = effect_metadata(&self.effects);
        let touched = reads.union(&writes).copied().collect::<Vec<_>>();
        if self.read_offsets != reads.into_iter().collect::<Vec<_>>()
            || self.write_offsets != writes.into_iter().collect::<Vec<_>>()
            || self.touched_offsets != touched
        {
            return Err("summary offset metadata does not match its effects".to_string());
        }
        Ok(())
    }

    pub fn validate_runtime(&self) -> Result<(), String> {
        self.validate()?;
        if self.effects.len() > RUNTIME_SUMMARY_EFFECT_MAX {
            return Err(format!(
                "summary has {} effects, runtime maximum is {RUNTIME_SUMMARY_EFFECT_MAX}",
                self.effects.len()
            ));
        }
        if let (Some(min), Some(max)) = (self.touched_offsets.first(), self.touched_offsets.last())
            && max
                .checked_sub(*min)
                .and_then(|span| span.checked_add(1))
                .is_none_or(|span| span > RUNTIME_SUMMARY_WINDOW_MAX)
        {
            return Err(format!(
                "summary window {}..={} exceeds runtime maximum of {RUNTIME_SUMMARY_WINDOW_MAX} cells",
                min, max
            ));
        }
        Ok(())
    }

    pub fn lower_to_ir(&self) -> Result<Vec<BfIr>, String> {
        self.validate()?;
        let mut nodes = Vec::with_capacity(self.effects.len());
        for effect in &self.effects {
            nodes.push(match *effect {
                SummaryEffect::Clear { offset: 0 } => BfIr::Clear,
                SummaryEffect::Clear { offset } => BfIr::ClearAt { offset },
                SummaryEffect::AddConst { offset: 0, delta } => BfIr::Add(
                    i32::try_from(delta).map_err(|_| "summary constant does not fit i32")?,
                ),
                SummaryEffect::AddConst { .. } => {
                    return Err("nonzero-offset constants require offset IR lowering".to_string());
                }
                SummaryEffect::AddScaled {
                    src,
                    dst,
                    coeff,
                    set_dst,
                } => BfIr::Affine {
                    src,
                    dst,
                    coeff,
                    preserve_src: true,
                    set_dst,
                },
                SummaryEffect::Shift {
                    src,
                    dst,
                    amount,
                    dir,
                    set_dst,
                } => BfIr::Shift {
                    src,
                    dst,
                    amount,
                    dir,
                    preserve_src: true,
                    set_dst,
                },
                SummaryEffect::Square { src, dst, set_dst } => BfIr::Square {
                    src,
                    dst,
                    preserve_src: true,
                    set_dst,
                },
                SummaryEffect::AddProduct {
                    lhs,
                    rhs,
                    dst,
                    coeff: 1,
                    set_dst,
                } if lhs != rhs => BfIr::MulAdd {
                    lhs,
                    rhs,
                    dst,
                    preserve_lhs: true,
                    preserve_rhs: true,
                    set_dst,
                },
                SummaryEffect::AddProduct {
                    lhs,
                    rhs,
                    dst,
                    coeff: 1,
                    set_dst,
                } if lhs == rhs => BfIr::Square {
                    src: lhs,
                    dst,
                    preserve_src: true,
                    set_dst,
                },
                SummaryEffect::AddProduct { .. } => {
                    return Err("scaled product requires runtime summary application".to_string());
                }
            });
        }
        Ok(nodes)
    }
}

fn effect_metadata(
    effects: &[SummaryEffect],
) -> (BTreeSet<crate::bf::BfOffset>, BTreeSet<crate::bf::BfOffset>) {
    let mut reads = BTreeSet::new();
    let mut writes = BTreeSet::new();
    for effect in effects {
        match *effect {
            SummaryEffect::Clear { offset } => {
                writes.insert(offset);
            }
            SummaryEffect::AddConst { offset, .. } => {
                reads.insert(offset);
                writes.insert(offset);
            }
            SummaryEffect::AddScaled {
                src, dst, set_dst, ..
            }
            | SummaryEffect::Shift {
                src, dst, set_dst, ..
            }
            | SummaryEffect::Square { src, dst, set_dst } => {
                reads.insert(src);
                if !set_dst {
                    reads.insert(dst);
                }
                writes.insert(dst);
            }
            SummaryEffect::AddProduct {
                lhs,
                rhs,
                dst,
                set_dst,
                ..
            } => {
                reads.extend([lhs, rhs]);
                if !set_dst {
                    reads.insert(dst);
                }
                writes.insert(dst);
            }
        }
    }
    (reads, writes)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DynamicLoopMetadata {
    pub body_hash: u64,
    pub has_io: bool,
    pub touched_offsets: Option<Vec<crate::bf::BfOffset>>,
    pub read_offsets: Option<Vec<crate::bf::BfOffset>>,
    pub write_offsets: Option<Vec<crate::bf::BfOffset>>,
    pub min_pointer_offset: Option<crate::bf::BfOffset>,
    pub max_pointer_offset: Option<crate::bf::BfOffset>,
    pub nesting_depth: u16,
    pub hot_threshold: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OffsetOp {
    AddAt {
        offset: crate::bf::BfOffset,
        delta: i32,
    },
    InputAt {
        offset: crate::bf::BfOffset,
    },
    OutputAt {
        offset: crate::bf::BfOffset,
    },
    Summary(LoopSummary),
    OpaqueLoop,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormalizedBody {
    pub ops: Vec<OffsetOp>,
    pub net_pointer_delta: crate::bf::BfOffset,
    pub min_pointer_offset: crate::bf::BfOffset,
    pub max_pointer_offset: crate::bf::BfOffset,
    pub touched_offsets: Vec<crate::bf::BfOffset>,
    pub read_offsets: Vec<crate::bf::BfOffset>,
    pub write_offsets: Vec<crate::bf::BfOffset>,
    pub has_io: bool,
}

pub(crate) fn normalize_offset_body(body: &[BfIr]) -> NormalizedBody {
    fn opaque() -> NormalizedBody {
        NormalizedBody {
            ops: vec![OffsetOp::OpaqueLoop],
            net_pointer_delta: 0,
            min_pointer_offset: 0,
            max_pointer_offset: 0,
            touched_offsets: Vec::new(),
            read_offsets: Vec::new(),
            write_offsets: Vec::new(),
            has_io: false,
        }
    }

    let mut pointer: crate::bf::BfOffset = 0;
    let mut min_pointer: crate::bf::BfOffset = 0;
    let mut max_pointer: crate::bf::BfOffset = 0;
    let mut adds = BTreeMap::<crate::bf::BfOffset, i32>::new();
    let mut ordered = Vec::new();
    let mut touched = BTreeSet::new();
    let mut reads = BTreeSet::new();
    let mut writes = BTreeSet::new();
    let mut has_io = false;

    let flush_adds = |ordered: &mut Vec<OffsetOp>,
                      adds: &mut BTreeMap<crate::bf::BfOffset, i32>| {
        for (offset, delta) in std::mem::take(adds) {
            if delta != 0 {
                ordered.push(OffsetOp::AddAt { offset, delta });
            }
        }
    };

    for node in body {
        match node {
            BfIr::MovePtr(delta) => {
                let Some(next_pointer) = pointer.checked_add(*delta) else {
                    return opaque();
                };
                pointer = next_pointer;
                min_pointer = min_pointer.min(pointer);
                max_pointer = max_pointer.max(pointer);
            }
            BfIr::Add(delta) => {
                touched.insert(pointer);
                reads.insert(pointer);
                writes.insert(pointer);
                let entry = adds.entry(pointer).or_insert(0);
                let Some(combined) = entry.checked_add(*delta) else {
                    return opaque();
                };
                *entry = combined;
            }
            BfIr::Input => {
                flush_adds(&mut ordered, &mut adds);
                has_io = true;
                touched.insert(pointer);
                writes.insert(pointer);
                ordered.push(OffsetOp::InputAt { offset: pointer });
            }
            BfIr::Output => {
                flush_adds(&mut ordered, &mut adds);
                has_io = true;
                touched.insert(pointer);
                reads.insert(pointer);
                ordered.push(OffsetOp::OutputAt { offset: pointer });
            }
            BfIr::Clear
            | BfIr::ClearAt { .. }
            | BfIr::Distribute { .. }
            | BfIr::Affine { .. }
            | BfIr::Shift { .. }
            | BfIr::Square { .. }
            | BfIr::MulAdd { .. } => {
                flush_adds(&mut ordered, &mut adds);
                match LoopSummary::from_ir_nodes(
                    LoopId::default(),
                    SummaryProvenance::Static,
                    std::slice::from_ref(node),
                ) {
                    Ok(mut summary) => {
                        if summary.rebase(pointer).is_err() {
                            return opaque();
                        }
                        touched.extend(summary.touched_offsets.iter().copied());
                        reads.extend(summary.read_offsets.iter().copied());
                        writes.extend(summary.write_offsets.iter().copied());
                        ordered.push(OffsetOp::Summary(summary));
                    }
                    Err(_) => ordered.push(OffsetOp::OpaqueLoop),
                }
            }
            BfIr::Loop(_) | BfIr::Scan { .. } | BfIr::Diverge => {
                flush_adds(&mut ordered, &mut adds);
                ordered.push(OffsetOp::OpaqueLoop);
            }
        }
    }
    flush_adds(&mut ordered, &mut adds);
    NormalizedBody {
        ops: ordered,
        net_pointer_delta: pointer,
        min_pointer_offset: min_pointer,
        max_pointer_offset: max_pointer,
        touched_offsets: touched.into_iter().collect(),
        read_offsets: reads.into_iter().collect(),
        write_offsets: writes.into_iter().collect(),
        has_io,
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SymbolicTransfer {
    pub ptr_delta: crate::bf::BfOffset,
    pub effects: BTreeMap<crate::bf::BfOffset, SymbolicPolynomial>,
    pub reads: BTreeSet<crate::bf::BfOffset>,
    pub may_input: bool,
    pub may_output: bool,
    pub may_diverge: bool,
    pub unknown: bool,
}

impl SymbolicTransfer {
    pub(crate) fn identity() -> Self {
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

    pub(crate) fn unknown() -> Self {
        Self {
            unknown: true,
            ..Self::identity()
        }
    }

    pub(crate) fn accessed_offsets(&self) -> BTreeSet<crate::bf::BfOffset> {
        let mut accessed = self.reads.clone();
        accessed.extend(self.effects.keys().copied());
        accessed
    }

    pub(crate) fn is_pure_windowed(&self) -> bool {
        !self.unknown && !self.may_input && !self.may_output && !self.may_diverge
    }
}

pub(crate) fn stable_summary_hash(text: &str) -> u64 {
    crate::hashing::hash_words(0x4246_5355_4D4D_4152, text.bytes().map(u64::from))
}
