//! Bounded, allocation-accounted normalization of exact DAG occurrences.

use crate::RequiredExt;
use crate::bitgrid::Coord;
use crate::hashing::hash_words;
use crate::hashlife::{HashLifeEngine, NodeId, RelativeBounds};
use crate::probe_table::ProbeKey;
use crate::recurrence::RecurrenceUnavailable;

pub(super) const RECURRENCE_REBLOCK_WORK_LIMIT: usize = 4_096;
const RECURRENCE_REBLOCK_STACK_LEN: usize = 256;
const RECURRENCE_BOUNDS_STACK_LEN: usize = 64;
const RECURRENCE_BOUNDS_MEMO_LEN: usize = 1_024;
const RECURRENCE_BOUNDS_MEMO_MASK: u64 = 1_023;
const RECURRENCE_REBLOCK_MEMO_LEN: usize = 1_024;
const RECURRENCE_REBLOCK_MEMO_MASK: u64 = 1_023;

#[derive(Clone, Copy)]
pub(super) struct ReblockFrame {
    pub(super) source: NodeId,
    pub(super) source_x: i128,
    pub(super) source_y: i128,
    pub(super) target_x: i128,
    pub(super) target_y: i128,
    pub(super) target_level: u32,
}

#[derive(Clone, Copy)]
enum ReblockOp {
    Enter(ReblockFrame),
    Publish(ReblockKey),
    Combine(ReblockKey),
}

const EMPTY_FRAME: ReblockFrame = ReblockFrame {
    source: NodeId::ZERO,
    source_x: 0,
    source_y: 0,
    target_x: 0,
    target_y: 0,
    target_level: 0,
};

#[derive(Clone, Copy, PartialEq, Eq)]
struct ReblockKey {
    source: NodeId,
    relative_x: i128,
    relative_y: i128,
    target_level: u32,
}

const EMPTY_REBLOCK_KEY: ReblockKey = ReblockKey {
    source: NodeId::MAX,
    relative_x: 0,
    relative_y: 0,
    target_level: 0,
};

pub(super) struct ReblockMemo {
    keys: [ReblockKey; RECURRENCE_REBLOCK_MEMO_LEN],
    values: [NodeId; RECURRENCE_REBLOCK_MEMO_LEN],
}

impl ReblockMemo {
    pub(super) fn new() -> Self {
        Self {
            keys: [EMPTY_REBLOCK_KEY; RECURRENCE_REBLOCK_MEMO_LEN],
            values: [NodeId::ZERO; RECURRENCE_REBLOCK_MEMO_LEN],
        }
    }

    fn get(&self, key: ReblockKey) -> Option<NodeId> {
        let slot = reblock_memo_slot(key);
        (self.keys[slot] == key).then_some(self.values[slot])
    }

    fn insert(&mut self, key: ReblockKey, value: NodeId) {
        let slot = reblock_memo_slot(key);
        self.keys[slot] = key;
        self.values[slot] = value;
    }
}

#[derive(Clone, Copy)]
struct BoundsFrame {
    node: NodeId,
    next_child: usize,
    bounds: RelativeBounds,
}

const EMPTY_RELATIVE_BOUNDS: RelativeBounds = RelativeBounds {
    min_x: Coord::MAX,
    min_y: Coord::MAX,
    max_x: Coord::MIN,
    max_y: Coord::MIN,
};

const EMPTY_BOUNDS_FRAME: BoundsFrame = BoundsFrame {
    node: NodeId::ZERO,
    next_child: 0,
    bounds: EMPTY_RELATIVE_BOUNDS,
};

pub(super) struct BoundsMemo {
    keys: [NodeId; RECURRENCE_BOUNDS_MEMO_LEN],
    values: [RelativeBounds; RECURRENCE_BOUNDS_MEMO_LEN],
}

impl BoundsMemo {
    pub(super) fn new() -> Self {
        Self {
            keys: [NodeId::MAX; RECURRENCE_BOUNDS_MEMO_LEN],
            values: [EMPTY_RELATIVE_BOUNDS; RECURRENCE_BOUNDS_MEMO_LEN],
        }
    }

    fn get(&self, node: NodeId) -> Option<RelativeBounds> {
        let slot = bounds_memo_slot(node);
        (self.keys[slot] == node).then_some(self.values[slot])
    }

    fn insert(&mut self, node: NodeId, bounds: RelativeBounds) {
        let slot = bounds_memo_slot(node);
        self.keys[slot] = node;
        self.values[slot] = bounds;
    }
}

fn bounds_memo_slot(node: NodeId) -> usize {
    usize::try_from(ProbeKey::fingerprint(&node) & RECURRENCE_BOUNDS_MEMO_MASK)
        .or_invariant("recurrence bounds memo slot must fit usize")
}

fn reblock_key(frame: ReblockFrame) -> Result<ReblockKey, RecurrenceUnavailable> {
    Ok(ReblockKey {
        source: frame.source,
        relative_x: frame
            .target_x
            .checked_sub(frame.source_x)
            .ok_or(RecurrenceUnavailable::CoordinateOverflow)?,
        relative_y: frame
            .target_y
            .checked_sub(frame.source_y)
            .ok_or(RecurrenceUnavailable::CoordinateOverflow)?,
        target_level: frame.target_level,
    })
}

fn reblock_memo_slot(key: ReblockKey) -> usize {
    let x = key.relative_x.to_le_bytes();
    let y = key.relative_y.to_le_bytes();
    let fingerprint = hash_words(
        0x5245_424C_4F43_4B31,
        [
            u64::from(key.source),
            u64::from_le_bytes([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]]),
            u64::from_le_bytes([x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15]]),
            u64::from_le_bytes([y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]]),
            u64::from_le_bytes([y[8], y[9], y[10], y[11], y[12], y[13], y[14], y[15]]),
            u64::from(key.target_level),
        ],
    );
    usize::try_from(fingerprint & RECURRENCE_REBLOCK_MEMO_MASK)
        .or_invariant("recurrence reblock memo slot must fit usize")
}

pub(super) fn reblock_square(
    engine: &mut HashLifeEngine,
    first: ReblockFrame,
    global_bounds: (i128, i128, i128, i128),
    bounds_memo: &BoundsMemo,
    reblock_memo: &mut ReblockMemo,
    work: &mut usize,
) -> Result<NodeId, RecurrenceUnavailable> {
    let mut ops = [ReblockOp::Enter(EMPTY_FRAME); RECURRENCE_REBLOCK_STACK_LEN];
    let mut op_len = 1;
    ops[0] = ReblockOp::Enter(first);
    let mut results = [NodeId::ZERO; RECURRENCE_REBLOCK_STACK_LEN];
    let mut result_len = 0;
    while op_len != 0 {
        op_len -= 1;
        match ops[op_len] {
            ReblockOp::Publish(key) => {
                if result_len == 0 {
                    crate::invariant_failure!("recurrence reblock publish stack underflow");
                }
                reblock_memo.insert(key, results[result_len - 1]);
            }
            ReblockOp::Combine(key) => {
                if result_len < 4 {
                    crate::invariant_failure!("recurrence reblock result stack underflow");
                }
                let joined = engine.join(
                    results[result_len - 4],
                    results[result_len - 3],
                    results[result_len - 2],
                    results[result_len - 1],
                );
                result_len -= 4;
                if engine.allocation_failed() {
                    return Err(RecurrenceUnavailable::Allocation);
                }
                results[result_len] = joined;
                result_len += 1;
                reblock_memo.insert(key, joined);
            }
            ReblockOp::Enter(frame) => {
                let key = reblock_key(frame)?;
                if let Some(result) = reblock_memo.get(key) {
                    if result_len == results.len() {
                        return Err(RecurrenceUnavailable::EntryLimit);
                    }
                    results[result_len] = result;
                    result_len += 1;
                    continue;
                }
                charge_work(work)?;
                let source_level = engine.node_columns.level(frame.source);
                let source_size = 1_i128
                    .checked_shl(source_level)
                    .ok_or(RecurrenceUnavailable::CoordinateOverflow)?;
                let target_size = 1_i128
                    .checked_shl(frame.target_level)
                    .ok_or(RecurrenceUnavailable::CoordinateOverflow)?;
                let target_max_x = frame
                    .target_x
                    .checked_add(target_size - 1)
                    .ok_or(RecurrenceUnavailable::CoordinateOverflow)?;
                let target_max_y = frame
                    .target_y
                    .checked_add(target_size - 1)
                    .ok_or(RecurrenceUnavailable::CoordinateOverflow)?;
                let source_live_bounds = known_relative_bounds(engine, bounds_memo, frame.source)
                    .map(|relative| {
                        (
                            frame.source_x + i128::from(relative.min_x),
                            frame.source_y + i128::from(relative.min_y),
                            frame.source_x + i128::from(relative.max_x),
                            frame.source_y + i128::from(relative.max_y),
                        )
                    });
                let outside = |(live_min_x, live_min_y, live_max_x, live_max_y)| {
                    target_max_x < live_min_x
                        || target_max_y < live_min_y
                        || frame.target_x > live_max_x
                        || frame.target_y > live_max_y
                };

                let result = if outside(global_bounds)
                    || engine.node_columns.population(frame.source) == 0
                    || source_live_bounds.is_some_and(outside)
                {
                    let empty = engine.empty(frame.target_level);
                    if engine.allocation_failed() {
                        return Err(RecurrenceUnavailable::Allocation);
                    }
                    Some(empty)
                } else if source_level == frame.target_level
                    && frame.source_x == frame.target_x
                    && frame.source_y == frame.target_y
                {
                    Some(frame.source)
                } else {
                    None
                };
                if let Some(result) = result {
                    if result_len == results.len() {
                        return Err(RecurrenceUnavailable::EntryLimit);
                    }
                    results[result_len] = result;
                    result_len += 1;
                    reblock_memo.insert(key, result);
                    continue;
                }

                if source_level != 0 {
                    let half = source_size / 2;
                    let children = engine.node_columns.quadrants(frame.source);
                    let mut intersections = 0;
                    let mut sole_child = (NodeId::ZERO, 0, 0);
                    for (child_index, child) in children.into_iter().enumerate() {
                        if engine.node_columns.population(child) == 0 {
                            continue;
                        }
                        let child_x = frame.source_x
                            + if child_index.is_multiple_of(2) {
                                0
                            } else {
                                half
                            };
                        let child_y = frame.source_y + if child_index < 2 { 0 } else { half };
                        let child_bounds = known_relative_bounds(engine, bounds_memo, child)
                            .map_or(
                                (child_x, child_y, child_x + half - 1, child_y + half - 1),
                                |relative| {
                                    (
                                        child_x + i128::from(relative.min_x),
                                        child_y + i128::from(relative.min_y),
                                        child_x + i128::from(relative.max_x),
                                        child_y + i128::from(relative.max_y),
                                    )
                                },
                            );
                        if !outside(child_bounds) {
                            intersections += 1;
                            sole_child = (child, child_x, child_y);
                        }
                    }
                    if intersections == 0 {
                        let empty = engine.empty(frame.target_level);
                        if engine.allocation_failed() {
                            return Err(RecurrenceUnavailable::Allocation);
                        }
                        if result_len == results.len() {
                            return Err(RecurrenceUnavailable::EntryLimit);
                        }
                        results[result_len] = empty;
                        result_len += 1;
                        reblock_memo.insert(key, empty);
                        continue;
                    }
                    if intersections == 1 {
                        if op_len + 2 > ops.len() {
                            return Err(RecurrenceUnavailable::EntryLimit);
                        }
                        ops[op_len] = ReblockOp::Publish(key);
                        ops[op_len + 1] = ReblockOp::Enter(ReblockFrame {
                            source: sole_child.0,
                            source_x: sole_child.1,
                            source_y: sole_child.2,
                            ..frame
                        });
                        op_len += 2;
                        continue;
                    }
                }
                if frame.target_level == 0 || op_len + 5 > ops.len() {
                    return Err(RecurrenceUnavailable::EntryLimit);
                }
                let half = target_size / 2;
                ops[op_len] = ReblockOp::Combine(key);
                ops[op_len + 1] = ReblockOp::Enter(ReblockFrame {
                    target_x: frame.target_x + half,
                    target_y: frame.target_y + half,
                    target_level: frame.target_level - 1,
                    ..frame
                });
                ops[op_len + 2] = ReblockOp::Enter(ReblockFrame {
                    target_y: frame.target_y + half,
                    target_level: frame.target_level - 1,
                    ..frame
                });
                ops[op_len + 3] = ReblockOp::Enter(ReblockFrame {
                    target_x: frame.target_x + half,
                    target_level: frame.target_level - 1,
                    ..frame
                });
                ops[op_len + 4] = ReblockOp::Enter(ReblockFrame {
                    target_level: frame.target_level - 1,
                    ..frame
                });
                op_len += 5;
            }
        }
    }
    if result_len != 1 {
        crate::invariant_failure!("recurrence reblock did not produce one root");
    }
    Ok(results[0])
}

pub(super) fn bounded_relative_bounds(
    engine: &HashLifeEngine,
    root: NodeId,
    memo: &mut BoundsMemo,
    work: &mut usize,
) -> Result<Option<RelativeBounds>, RecurrenceUnavailable> {
    if engine.node_columns.population(root) == 0 {
        return Ok(None);
    }
    if let Some(bounds) = engine.result_caches.bounds.get(&root) {
        return Ok(Some(bounds));
    }
    charge_work(work)?;
    let mut stack = [EMPTY_BOUNDS_FRAME; RECURRENCE_BOUNDS_STACK_LEN];
    stack[0] = BoundsFrame {
        node: root,
        next_child: 0,
        bounds: EMPTY_RELATIVE_BOUNDS,
    };
    let mut stack_len = 1;
    let mut completed = None;
    while stack_len != 0 {
        if let Some(child_bounds) = completed.take() {
            let parent = &mut stack[stack_len - 1];
            merge_relative_bounds(
                &mut parent.bounds,
                child_bounds,
                parent.next_child - 1,
                engine.node_columns.level(parent.node),
            );
        }
        let frame = &mut stack[stack_len - 1];
        let level = engine.node_columns.level(frame.node);
        if level == 0 || frame.next_child == 4 {
            let bounds = if level == 0 {
                RelativeBounds {
                    min_x: 0,
                    min_y: 0,
                    max_x: 0,
                    max_y: 0,
                }
            } else {
                frame.bounds
            };
            memo.insert(frame.node, bounds);
            stack_len -= 1;
            if stack_len == 0 {
                return Ok(Some(bounds));
            }
            completed = Some(bounds);
            continue;
        }

        let child_index = frame.next_child;
        frame.next_child += 1;
        let child = engine.node_columns.quadrants(frame.node)[child_index];
        if engine.node_columns.population(child) == 0 {
            continue;
        }
        if let Some(bounds) = known_relative_bounds(engine, memo, child) {
            merge_relative_bounds(&mut frame.bounds, bounds, child_index, level);
            continue;
        }
        charge_work(work)?;
        if stack_len == stack.len() {
            return Err(RecurrenceUnavailable::WitnessLimit);
        }
        stack[stack_len] = BoundsFrame {
            node: child,
            next_child: 0,
            bounds: EMPTY_RELATIVE_BOUNDS,
        };
        stack_len += 1;
    }
    crate::invariant_failure!("recurrence bounds traversal produced no result")
}

fn known_relative_bounds(
    engine: &HashLifeEngine,
    memo: &BoundsMemo,
    node: NodeId,
) -> Option<RelativeBounds> {
    if engine.node_columns.population(node) == 0 {
        None
    } else {
        engine
            .result_caches
            .bounds
            .get(&node)
            .or_else(|| memo.get(node))
    }
}

fn merge_relative_bounds(
    bounds: &mut RelativeBounds,
    child: RelativeBounds,
    child_index: usize,
    parent_level: u32,
) {
    let half = 1_i64 << (parent_level - 1);
    let offset_x = if child_index.is_multiple_of(2) {
        0
    } else {
        half
    };
    let offset_y = if child_index < 2 { 0 } else { half };
    bounds.min_x = bounds.min_x.min(child.min_x + offset_x);
    bounds.min_y = bounds.min_y.min(child.min_y + offset_y);
    bounds.max_x = bounds.max_x.max(child.max_x + offset_x);
    bounds.max_y = bounds.max_y.max(child.max_y + offset_y);
}

pub(super) fn charge_work(work: &mut usize) -> Result<(), RecurrenceUnavailable> {
    *work = work
        .checked_add(1)
        .ok_or(RecurrenceUnavailable::WitnessLimit)?;
    if *work > RECURRENCE_REBLOCK_WORK_LIMIT {
        return Err(RecurrenceUnavailable::WitnessLimit);
    }
    Ok(())
}
