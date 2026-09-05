//! Exact positive-time interfaces for reusing completed HashLife jumps.
//!
//! A level-3 (8x8) interface retains the current two-cell border and the
//! complete next generation of the strict 6x6 interior. Every radius-one
//! neighborhood needed for those 36 next cells lies inside the 8x8 source,
//! so this signature is exact rather than probabilistic. Smaller nodes retain
//! their complete current shape. Larger interfaces are congruence tuples of
//! their four child interfaces, level, and rule. Induction over that tuple is
//! the composition proof used by the normal HashLife recurrence.
//!
//! The result table is queried only by jump operations. `step_exp == 0` means
//! one generation in HashLife, never a zero-time identity query. Structural
//! and D4 caches are always probed first. A cached result remains in the
//! canonical input frame; `PackedSymmetryKey::symmetry` and the caller's
//! inverse input symmetry restore the original orientation. Its packed level
//! records the normal one-level centered crop explicitly.

use super::*;

const FUTURE_PROOF_MAX_VISITS: usize = 4_096;
const FUTURE_OPTIONAL_MAX_BYTES: usize = 32 * 1024 * 1024;
const FUTURE_MISS_SAMPLE: usize = 32;
const FUTURE_MISS_COOLDOWN: usize = 512;
const FUTURE_HASH_DOMAIN: u64 = 0x4655_5455_5245_434C;
const FUTURE_RESULT_HASH_DOMAIN: u64 = 0x4655_5455_5245_4A50;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FutureClass(u32);

impl FutureClass {
    fn from_index(index: usize) -> Option<Self> {
        u32::try_from(index).ok().map(Self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FutureRule {
    Conway,
}

impl FutureRule {
    const fn code(self) -> u64 {
        match self {
            Self::Conway => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FutureKey {
    Exact {
        level: u32,
        rule: FutureRule,
        shape: CanonicalShapeId,
    },
    Exact8x8 {
        rule: FutureRule,
        current_border: u64,
        next_interior: u64,
    },
    Parent {
        level: u32,
        rule: FutureRule,
        children: [FutureClass; 4],
    },
}

impl ProbeKey for FutureKey {
    fn fingerprint(&self) -> u64 {
        match *self {
            Self::Exact { level, rule, shape } => hash_words(
                FUTURE_HASH_DOMAIN,
                [0, u64::from(level), rule.code(), u64::from(shape)],
            ),
            Self::Exact8x8 {
                rule,
                current_border,
                next_interior,
            } => hash_words(
                FUTURE_HASH_DOMAIN,
                [1, rule.code(), current_border, next_interior],
            ),
            Self::Parent {
                level,
                rule,
                children,
            } => hash_words(
                FUTURE_HASH_DOMAIN,
                [
                    2,
                    u64::from(level),
                    rule.code(),
                    u64::from(children[0].0),
                    u64::from(children[1].0),
                    u64::from(children[2].0),
                    u64::from(children[3].0),
                ],
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FutureResultKey {
    class: FutureClass,
    jump: PositiveJump,
    source_level: u32,
    symmetry_admitted: bool,
}

impl ProbeKey for FutureResultKey {
    fn fingerprint(&self) -> u64 {
        hash_words(
            FUTURE_RESULT_HASH_DOMAIN,
            [
                u64::from(self.class.0),
                u64::from(self.jump.step_exp),
                u64::from(self.source_level),
                u64::from(self.symmetry_admitted),
            ],
        )
    }
}

/// A strictly positive HashLife jump. Exponent zero denotes one generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PositiveJump {
    step_exp: u32,
}

impl PositiveJump {
    fn power_of_two(step_exp: u32, source_level: u32) -> Option<Self> {
        let max_exp = source_level.checked_sub(2)?;
        (step_exp <= max_exp).then_some(Self { step_exp })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FutureJumpResult {
    entry: PackedSymmetryKey,
    result_level: u32,
    registry_epoch: u64,
}

#[derive(Debug)]
pub(super) struct FutureState {
    registry_epoch: u64,
    shape_epoch: u64,
    class_intern: ProbeTable<FutureKey, FutureClass>,
    class_count: usize,
    shape_classes: ProbeTable<CanonicalShapeId, FutureClass>,
    results: ProbeTable<FutureResultKey, FutureJumpResult>,
    analyzed_misses: usize,
    analysis_cooldown: usize,
}

impl FutureState {
    pub(super) fn new(shape_epoch: u64) -> Self {
        Self {
            registry_epoch: 0,
            shape_epoch,
            class_intern: ProbeTable::new(ProbeMode::AppendOnly),
            class_count: 0,
            shape_classes: ProbeTable::new(ProbeMode::RebuildOnGc),
            results: ProbeTable::new(ProbeMode::RebuildOnGc),
            analyzed_misses: 0,
            analysis_cooldown: 0,
        }
    }

    pub(super) fn allocated_bytes(&self) -> usize {
        self.class_intern.allocated_bytes()
            + self.shape_classes.allocated_bytes()
            + self.results.allocated_bytes()
    }

    pub(super) fn result_len(&self) -> usize {
        self.results.len()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FutureProofBudget {
    visits: usize,
    exhausted: bool,
}

#[derive(Clone, Copy, Debug)]
struct FutureProofFrame {
    shape: CanonicalShapeId,
    key: CanonicalStructKey,
    children: [FutureClass; 4],
    next_child: usize,
    entered: bool,
}

impl FutureProofFrame {
    fn new(shape: CanonicalShapeId) -> Self {
        Self {
            shape,
            key: CanonicalStructKey::leaf(false),
            children: [FutureClass(0); 4],
            next_child: 0,
            entered: false,
        }
    }
}

impl FutureProofBudget {
    fn visit_child_slots(&mut self, count: usize) -> bool {
        let Some(total) = self.visits.checked_add(count) else {
            self.exhausted = true;
            return false;
        };
        if total > FUTURE_PROOF_MAX_VISITS {
            self.exhausted = true;
            return false;
        }
        self.visits = total;
        true
    }
}

impl HashLifeEngine {
    fn future_shape_epoch_is_current(&self) -> bool {
        self.future_state.shape_epoch == self.canonical_caches.shape_epoch
    }

    fn reset_future_registry(&mut self, shape_epoch: u64, release_storage: bool) {
        debug_assert!(
            self.at_gc_safepoint(),
            "FutureClass registry reset requires a quiescent engine"
        );
        self.future_state.registry_epoch = self
            .future_state
            .registry_epoch
            .checked_add(1)
            .or_invariant("FutureClass registry epoch overflow");
        self.future_state.shape_epoch = shape_epoch;
        self.stats.future.registry_resets += 1;
        self.future_state.analyzed_misses = 0;
        self.future_state.analysis_cooldown = 0;
        if release_storage {
            self.future_state.class_intern.release_storage();
            self.future_state.class_count = 0;
            self.future_state.shape_classes.release_storage();
            self.future_state.results.release_storage();
        } else {
            self.future_state.class_intern.clear();
            self.future_state.class_count = 0;
            self.future_state.shape_classes.reset();
            self.future_state.results.reset();
        }
    }

    pub(super) fn prepare_future_for_shape_rebuild(&mut self) {
        let next_shape_epoch = self
            .canonical_caches
            .shape_epoch
            .checked_add(1)
            .or_invariant("canonical shape registry epoch overflow");
        if self.future_state.shape_epoch != next_shape_epoch {
            self.reset_future_registry(next_shape_epoch, false);
        }
    }

    pub(super) fn release_future_state(&mut self) {
        self.reset_future_registry(self.canonical_caches.shape_epoch, true);
    }

    pub(super) fn clear_future_results(&mut self) {
        self.future_state.results.reset();
    }

    pub(super) fn filter_future_results_to_live_nodes(&mut self) {
        let before = self.future_state.results.len();
        let columns = &self.node_columns;
        let epoch = self.future_state.registry_epoch;
        self.future_state.results.retain(|_, result| {
            result.registry_epoch == epoch
                && super::gc::packed_node_is_live_for_cache(result.entry.packed, columns)
        });
        self.stats.future.weak_results_dropped += before - self.future_state.results.len();
    }

    fn future_optional_growth_permitted(&self, required_bytes: u128) -> bool {
        let future_bytes = self.future_state.allocated_bytes() as u128;
        if future_bytes
            .checked_add(required_bytes)
            .is_none_or(|bytes| bytes > FUTURE_OPTIONAL_MAX_BYTES as u128)
        {
            return false;
        }
        crate::hashlife::memory::wide_allocated_bytes(self.allocated_bytes())
            .checked_add(self.allocation_transient_reserved)
            .and_then(|used| used.checked_add(required_bytes))
            .is_some_and(|projected| projected <= self.allocation_hard_limit)
    }

    fn future_analysis_admitted(&mut self) -> bool {
        if self.future_state.analysis_cooldown == 0 {
            return true;
        }
        self.future_state.analysis_cooldown -= 1;
        self.stats.future.analysis_sampling_bypasses += 1;
        false
    }

    fn record_future_analysis_miss(&mut self) {
        self.future_state.analyzed_misses += 1;
        if self.future_state.analyzed_misses == FUTURE_MISS_SAMPLE {
            self.future_state.analyzed_misses = 0;
            self.future_state.analysis_cooldown = FUTURE_MISS_COOLDOWN;
            self.stats.future.analysis_cooldowns += 1;
        }
    }

    fn intern_future_key(&mut self, key: FutureKey) -> Option<FutureClass> {
        let fingerprint = key.fingerprint();
        if let Some(class) = self
            .future_state
            .class_intern
            .get_with_fingerprint(&key, fingerprint)
        {
            return Some(class);
        }
        let class = FutureClass::from_index(self.future_state.class_count)?;
        let next_class_count = self.future_state.class_count.checked_add(1)?;
        let table_bytes = self
            .future_state
            .class_intern
            .reservation_bytes_for_insert_with_fingerprint(&key, fingerprint)
            .ok()?;
        if !self.future_optional_growth_permitted(table_bytes) {
            return None;
        }
        if self
            .future_state
            .class_intern
            .try_insert_with_fingerprint(key, fingerprint, class)
            .is_err()
        {
            return None;
        }
        self.future_state.class_count = next_class_count;
        self.stats.future.classes_interned += 1;
        Some(class)
    }

    fn cache_future_shape_class(&mut self, shape: CanonicalShapeId, class: FutureClass) {
        let fingerprint = shape.fingerprint();
        let Ok(bytes) = self
            .future_state
            .shape_classes
            .reservation_bytes_for_insert_with_fingerprint(&shape, fingerprint)
        else {
            return;
        };
        if !self.future_optional_growth_permitted(bytes) {
            return;
        }
        let _ =
            self.future_state
                .shape_classes
                .try_insert_with_fingerprint(shape, fingerprint, class);
    }

    fn future_class_for_shape(
        &mut self,
        shape: CanonicalShapeId,
        budget: &mut FutureProofBudget,
    ) -> Option<FutureClass> {
        const MAX_PROOF_DEPTH: usize = 128;
        let mut stack: [FutureProofFrame; MAX_PROOF_DEPTH] =
            std::array::from_fn(|_| FutureProofFrame::new(CanonicalShapeId::DEAD));
        stack[0] = FutureProofFrame::new(shape);
        let mut stack_len = 1;
        loop {
            let frame_index = stack_len - 1;
            if !stack[frame_index].entered {
                let frame_shape = stack[frame_index].shape;
                self.stats.future.class_lookups += 1;
                if let Some(class) = self.future_state.shape_classes.get(&frame_shape) {
                    self.stats.future.class_hits += 1;
                    stack_len -= 1;
                    if stack_len == 0 {
                        return Some(class);
                    }
                    let parent = &mut stack[stack_len - 1];
                    parent.children[parent.next_child] = class;
                    parent.next_child += 1;
                    continue;
                }
                self.stats.future.class_misses += 1;
                let metadata = *self.canonical_caches.shapes.get(frame_shape.index())?;
                stack[frame_index].key = metadata.key;
                stack[frame_index].entered = true;
                let terminal_key = match metadata.key.level {
                    0..=2 => Some(FutureKey::Exact {
                        level: metadata.key.level,
                        rule: FutureRule::Conway,
                        shape: frame_shape,
                    }),
                    3 => {
                        let current = self.canonical_shape_bits_8x8(frame_shape, budget)?;
                        let (current_border, next_interior) = exact_8x8_signature(current);
                        Some(FutureKey::Exact8x8 {
                            rule: FutureRule::Conway,
                            current_border,
                            next_interior,
                        })
                    }
                    _ => None,
                };
                if let Some(key) = terminal_key {
                    let class = self.intern_future_key(key)?;
                    self.cache_future_shape_class(frame_shape, class);
                    stack_len -= 1;
                    if stack_len == 0 {
                        return Some(class);
                    }
                    let parent = &mut stack[stack_len - 1];
                    parent.children[parent.next_child] = class;
                    parent.next_child += 1;
                }
                continue;
            }

            if stack[frame_index].next_child < 4 {
                if !budget.visit_child_slots(1) {
                    return None;
                }
                if stack_len == stack.len() {
                    self.stats.future.proof_budget_bypasses += 1;
                    return None;
                }
                let child = stack[frame_index].key.children[stack[frame_index].next_child];
                stack[stack_len] = FutureProofFrame::new(child);
                stack_len += 1;
                continue;
            }

            let frame = stack[frame_index];
            let class = self.intern_future_key(FutureKey::Parent {
                level: frame.key.level,
                rule: FutureRule::Conway,
                children: frame.children,
            })?;
            self.cache_future_shape_class(frame.shape, class);
            stack_len -= 1;
            if stack_len == 0 {
                return Some(class);
            }
            let parent = &mut stack[stack_len - 1];
            parent.children[parent.next_child] = class;
            parent.next_child += 1;
        }
    }

    fn future_class_for_structural(
        &mut self,
        structural: CanonicalStructKey,
    ) -> Option<FutureClass> {
        if !self.future_shape_epoch_is_current() {
            self.stats.future.proof_bypasses += 1;
            return None;
        }
        let Some(shape) = self.canonical_caches.shape_intern.get(&structural) else {
            self.stats.future.proof_bypasses += 1;
            return None;
        };
        let mut budget = FutureProofBudget::default();
        let class = self.future_class_for_shape(shape, &mut budget);
        self.stats.future.proof_visits += budget.visits;
        if budget.exhausted {
            self.stats.future.proof_budget_bypasses += 1;
        }
        if class.is_none() {
            self.stats.future.proof_bypasses += 1;
        }
        class
    }

    fn canonical_shape_bits_8x8(
        &self,
        shape: CanonicalShapeId,
        budget: &mut FutureProofBudget,
    ) -> Option<u64> {
        let root = self.canonical_caches.shapes.get(shape.index())?.key;
        if root.level != 3 || !budget.visit_child_slots(4) {
            return None;
        }
        let mut bits = 0_u64;
        for (quadrant_4x4, shape_4x4) in root.children.into_iter().enumerate() {
            let node_4x4 = self.canonical_caches.shapes.get(shape_4x4.index())?.key;
            if node_4x4.level != 2 || !budget.visit_child_slots(4) {
                return None;
            }
            for (quadrant_2x2, shape_2x2) in node_4x4.children.into_iter().enumerate() {
                let node_2x2 = self.canonical_caches.shapes.get(shape_2x2.index())?.key;
                if node_2x2.level != 1 || !budget.visit_child_slots(4) {
                    return None;
                }
                for (leaf_index, leaf) in node_2x2.children.into_iter().enumerate() {
                    if leaf == CanonicalShapeId::LIVE {
                        let x = (quadrant_4x4 % 2) * 4 + (quadrant_2x2 % 2) * 2 + leaf_index % 2;
                        let y = (quadrant_4x4 / 2) * 4 + (quadrant_2x2 / 2) * 2 + leaf_index / 2;
                        bits |= 1_u64 << (y * 8 + x);
                    }
                }
            }
        }
        Some(bits)
    }

    pub(in crate::hashlife) fn lookup_future_jump_result(
        &mut self,
        key: CanonicalJumpKey,
    ) -> Option<PackedSymmetryKey> {
        self.stats.future.result_lookups += 1;
        if self.allocation_failed() {
            self.stats.future.result_bypasses += 1;
            return None;
        }
        if key.structural.level < 3 {
            // Exact classes below 8x8 cannot add reuse beyond the structural
            // cache that was already probed.
            self.stats.future.ineligible_lookups += 1;
            return None;
        }
        let Some(jump) = PositiveJump::power_of_two(key.step_exp, key.structural.level) else {
            self.stats.future.result_bypasses += 1;
            return None;
        };
        if !self.future_analysis_admitted() {
            return None;
        }
        if self.future_state.results.len() == 0 {
            self.stats.future.result_misses += 1;
            self.record_future_analysis_miss();
            return None;
        }
        let Some(class) = self.future_class_for_structural(key.structural) else {
            self.stats.future.result_bypasses += 1;
            return None;
        };
        let result_key = FutureResultKey {
            class,
            jump,
            source_level: key.structural.level,
            symmetry_admitted: key.symmetry_admitted,
        };
        let Some(result) = self.future_state.results.get(&result_key) else {
            self.stats.future.result_misses += 1;
            self.record_future_analysis_miss();
            return None;
        };
        if result.registry_epoch != self.future_state.registry_epoch
            || result.result_level.checked_add(1) != Some(key.structural.level)
        {
            self.stats.future.result_bypasses += 1;
            return None;
        }
        self.future_state.analyzed_misses = 0;
        self.stats.future.result_hits += 1;
        Some(result.entry)
    }

    pub(in crate::hashlife) fn publish_future_jump_entry(
        &mut self,
        key: CanonicalJumpKey,
        entry: PackedSymmetryKey,
    ) {
        if self.allocation_failed() {
            self.stats.future.publication_bypasses += 1;
            return;
        }
        if key.structural.level < 3 {
            self.stats.future.ineligible_publications += 1;
            return;
        }
        let Some(result_level) = key.structural.level.checked_sub(1) else {
            return;
        };
        let Some(jump) = PositiveJump::power_of_two(key.step_exp, key.structural.level) else {
            self.stats.future.publication_bypasses += 1;
            return;
        };
        if self.future_state.analysis_cooldown != 0 {
            self.stats.future.sampling_publication_bypasses += 1;
            return;
        }
        if entry.packed.level != result_level {
            self.stats.future.publication_bypasses += 1;
            return;
        }
        let Some(class) = self.future_class_for_structural(key.structural) else {
            self.stats.future.publication_bypasses += 1;
            return;
        };
        let result_key = FutureResultKey {
            class,
            jump,
            source_level: key.structural.level,
            symmetry_admitted: key.symmetry_admitted,
        };
        let value = FutureJumpResult {
            entry,
            result_level,
            registry_epoch: self.future_state.registry_epoch,
        };
        let fingerprint = result_key.fingerprint();
        let Ok(bytes) = self
            .future_state
            .results
            .reservation_bytes_for_insert_with_fingerprint(&result_key, fingerprint)
        else {
            self.stats.future.publication_bypasses += 1;
            return;
        };
        if !self.future_optional_growth_permitted(bytes)
            || self
                .future_state
                .results
                .try_insert_with_fingerprint(result_key, fingerprint, value)
                .is_err()
        {
            self.stats.future.publication_bypasses += 1;
            return;
        }
        self.stats.future.result_publications += 1;
    }
}

fn exact_8x8_signature(current: u64) -> (u64, u64) {
    const BORDER_MASK: u64 = 0xFFFF_C3C3_C3C3_FFFF;
    let next_8x8 = crate::life::evolve_center_chunk_bitwise(&crate::memo::ChunkNeighborhood([
        0, 0, 0, 0, current, 0, 0, 0, 0,
    ]));
    let mut next_interior = 0_u64;
    for row in 0..6 {
        let packed_row = (next_8x8 >> ((row + 1) * 8 + 1)) & 0x3F;
        next_interior |= packed_row << (row * 6);
    }
    (current & BORDER_MASK, next_interior)
}

#[cfg(test)]
mod tests;
