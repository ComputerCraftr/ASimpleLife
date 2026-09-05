//! Bottom-up bisimulation of oriented quadtrees: equal level/leaf observations
//! and four equal labelled child classes prove exact equality by induction.
//! This quotients construction work, never Life's future behavior or ordering.

use super::*;

/// Exact oriented identities; representative masks are derived from equality,
/// not stored as a second, potentially inconsistent partition.
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::hashlife) struct OrientationRecord {
    pub(super) registry_epoch: u64,
    pub(super) resolved: u8,
    references: [CanonicalNodeRef; 8],
}

impl OrientationRecord {
    pub(super) fn reference(self, symmetry: Symmetry) -> CanonicalNodeRef {
        debug_assert_ne!(self.resolved & bit(symmetry), 0);
        self.references[symmetry as usize]
    }

    fn insert(&mut self, symmetry: Symmetry, reference: CanonicalNodeRef) {
        debug_assert!(
            self.resolved & bit(symmetry) == 0 || self.reference(symmetry) == reference,
            "exact orientation proofs must agree"
        );
        self.resolved |= bit(symmetry);
        self.references[symmetry as usize] = reference;
    }

    fn extend_stabilizer(&mut self, stabilizer: u8) {
        if stabilizer == bit(Symmetry::Identity) || self.resolved == u8::MAX {
            return;
        }
        for orientation in Symmetry::ALL {
            if self.resolved & bit(orientation) == 0 {
                continue;
            }
            let reference = self.reference(orientation);
            for automorphism in Symmetry::ALL {
                if stabilizer & bit(automorphism) != 0 {
                    // h(input) == input, so applying h then t equals applying t.
                    self.insert(automorphism.then(orientation), reference);
                }
            }
        }
    }

    fn stabilizer(self, shape: CanonicalNodeRef) -> u8 {
        Symmetry::ALL.into_iter().fold(0, |mask, orientation| {
            if self.resolved & bit(orientation) != 0 && self.reference(orientation) == shape {
                mask | bit(orientation)
            } else {
                mask
            }
        })
    }

    fn merge(&mut self, other: Self) {
        debug_assert_eq!(self.registry_epoch, other.registry_epoch);
        for orientation in Symmetry::ALL {
            if other.resolved & bit(orientation) != 0 {
                self.insert(orientation, other.reference(orientation));
            }
        }
    }
}

fn bit(symmetry: Symmetry) -> u8 {
    1 << (symmetry as u8)
}

pub(super) fn quotient_representative(orientation: Symmetry, stabilizer: u8) -> Symmetry {
    if stabilizer == 1 {
        return orientation;
    }
    Symmetry::ALL
        .into_iter()
        .filter(|h| stabilizer & bit(*h) != 0)
        .map(|h| h.then(orientation))
        .min()
        .unwrap_or(orientation)
}

fn close_stabilizer(mut stabilizer: u8) -> u8 {
    stabilizer |= bit(Symmetry::Identity);
    if stabilizer == bit(Symmetry::Identity) || stabilizer == u8::MAX {
        return stabilizer;
    }
    loop {
        let mut closed = stabilizer;
        for left in Symmetry::ALL {
            if stabilizer & bit(left) == 0 {
                continue;
            }
            for right in Symmetry::ALL {
                if stabilizer & bit(right) != 0 {
                    closed |= bit(left.then(right));
                }
            }
        }
        if closed == stabilizer {
            return closed;
        }
        stabilizer = closed;
    }
}

#[derive(Clone, Copy)]
struct OrientationFrame {
    shape: CanonicalNodeRef,
    requested: u8,
    needed: u8,
    representatives: [Symmetry; 8],
    record: OrientationRecord,
    children: [[CanonicalNodeRef; 8]; 4],
    next_child: usize,
}

impl OrientationFrame {
    const EMPTY: Self = Self {
        shape: CanonicalShapeId::DEAD,
        requested: 0,
        needed: 0,
        representatives: Symmetry::ALL,
        record: OrientationRecord {
            registry_epoch: 0,
            resolved: 0,
            references: [CanonicalShapeId::DEAD; 8],
        },
        children: [[CanonicalShapeId::DEAD; 8]; 4],
        next_child: 0,
    };
}

#[derive(Clone, Copy)]
struct ScalarOrientationFrame {
    shape: CanonicalNodeRef,
    requested: Symmetry,
    proof: Symmetry,
    record: OrientationRecord,
    children: [CanonicalNodeRef; 4],
    next_child: usize,
}

impl ScalarOrientationFrame {
    const EMPTY: Self = Self {
        shape: CanonicalShapeId::DEAD,
        requested: Symmetry::Identity,
        proof: Symmetry::Identity,
        record: OrientationRecord {
            registry_epoch: 0,
            resolved: 0,
            references: [CanonicalShapeId::DEAD; 8],
        },
        children: [CanonicalShapeId::DEAD; 4],
        next_child: 0,
    };
}

enum ScalarFramePreparation {
    Complete(OrientationRecord),
    Pending(ScalarOrientationFrame),
}

// One frame per level, not one frame per node. Child results live in the parent
// frame, so failed optional publication never loses a dependency's exact result.
const FRAME_COUNT: usize = 64;
pub(super) const RESOLVER_BYTES: u128 =
    std::mem::size_of::<[OrientationFrame; FRAME_COUNT]>() as u128;
#[cfg(test)]
pub(super) const SCALAR_RESOLVER_STACK_BYTES: usize =
    std::mem::size_of::<[ScalarOrientationFrame; FRAME_COUNT]>();
// Charge every child-slot visit, including hits and repeated children. Each
// frame has at most eight orientations, so cold analysis work is bounded as
// well as depth. Completed proofs survive a switch to exact scalar traversal.
const QUOTIENT_WORK_LIMIT: usize = FRAME_COUNT * 8;

enum QuotientResolution {
    Complete(OrientationRecord),
    OptimizationUnavailable,
    MandatoryFailure,
}

impl HashLifeEngine {
    fn orientation_frame(&mut self, shape: CanonicalNodeRef, requested: u8) -> OrientationFrame {
        let request_count = requested.count_ones() as usize;
        self.stats.transform.orientation_requests += request_count;
        let meta = self.canonical_caches.shapes[shape.index()];
        let record = OrientationRecord {
            registry_epoch: self.canonical_caches.shape_epoch,
            resolved: meta.stabilizer,
            references: [shape; 8],
        };
        let known_stabilizer = requested & meta.stabilizer;
        let mut frame = OrientationFrame {
            shape,
            requested,
            record,
            ..OrientationFrame::EMPTY
        };
        let mut eliminations = known_stabilizer.count_ones() as usize;
        let mut cache_hits = 0;
        let mut uncached = 0;
        if requested & !meta.stabilizer == 0 {
            self.stats.transform.orientation_quotient_eliminations += eliminations;
            debug_assert_eq!(request_count, eliminations + cache_hits + uncached);
            return frame;
        }

        if let Some(cached) = self
            .canonical_caches
            .symmetry_refs
            .get(&shape)
            .filter(|cached| cached.registry_epoch == self.canonical_caches.shape_epoch)
        {
            let cached_requests = requested & cached.resolved & !meta.stabilizer;
            cache_hits = cached_requests.count_ones() as usize;
            frame.record.merge(cached);
            if requested & !frame.record.resolved == 0 {
                self.stats.transform.orientation_quotient_eliminations += eliminations;
                self.stats.transform.orientation_cache_hits += cache_hits;
                debug_assert_eq!(request_count, eliminations + cache_hits + uncached);
                return frame;
            }
            let resolved_before_extension = frame.record.resolved;
            frame.record.extend_stabilizer(meta.stabilizer);
            eliminations += (requested & !resolved_before_extension & frame.record.resolved)
                .count_ones() as usize;
        }
        for orientation in Symmetry::ALL {
            if requested & bit(orientation) == 0 || frame.record.resolved & bit(orientation) != 0 {
                continue;
            }
            let representative = quotient_representative(orientation, meta.stabilizer);
            frame.representatives[orientation as usize] = representative;
            if frame.record.resolved & bit(representative) != 0 {
                frame
                    .record
                    .insert(orientation, frame.record.reference(representative));
                eliminations += 1;
            } else if frame.needed & bit(representative) != 0 {
                eliminations += 1;
            } else {
                frame.needed |= bit(representative);
                uncached += 1;
            }
        }
        self.stats.transform.orientation_quotient_eliminations += eliminations;
        self.stats.transform.orientation_cache_hits += cache_hits;
        self.stats.transform.orientation_uncached_requests += uncached;
        debug_assert_eq!(request_count, eliminations + cache_hits + uncached);
        frame
    }

    pub(super) fn resolve_shape_orientations(
        &mut self,
        shape: CanonicalNodeRef,
        requested: u8,
    ) -> Option<OrientationRecord> {
        self.resolve_shape_orientations_with_quotient_limit(shape, requested, QUOTIENT_WORK_LIMIT)
    }

    pub(super) fn resolve_shape_orientations_with_quotient_limit(
        &mut self,
        shape: CanonicalNodeRef,
        requested: u8,
        quotient_work_limit: usize,
    ) -> Option<OrientationRecord> {
        if self.allocation_failed() {
            return None;
        }
        let frame = self.orientation_frame(shape, requested);
        if frame.needed == 0 {
            return Some(frame.record);
        }
        let quotient = self.with_transient_allocation_scope(|engine| {
            if !engine.reserve_optional_transient_bytes(RESOLVER_BYTES) {
                engine.stats.transform.orientation_scratch_bypasses += 1;
                return QuotientResolution::OptimizationUnavailable;
            }
            engine.resolve_orientation_frames(frame, quotient_work_limit)
        });
        match quotient {
            QuotientResolution::Complete(record) => Some(record),
            QuotientResolution::MandatoryFailure => None,
            QuotientResolution::OptimizationUnavailable => {
                debug_assert!(!self.allocation_failed());
                self.resolve_shape_orientations_scalar(shape, requested)
            }
        }
    }

    fn resolve_orientation_frames(
        &mut self,
        first: OrientationFrame,
        work_limit: usize,
    ) -> QuotientResolution {
        if work_limit == 0 {
            self.stats.transform.orientation_work_bypasses += 1;
            return QuotientResolution::OptimizationUnavailable;
        }
        let mut frames = [OrientationFrame::EMPTY; FRAME_COUNT];
        frames[0] = first;
        let mut depth = 1;
        let mut work = 1;
        loop {
            let frame = &mut frames[depth - 1];
            let key = self.canonical_caches.shapes[frame.shape.index()].key;
            if frame.next_child < 4 {
                if work == work_limit {
                    self.stats.transform.orientation_work_bypasses += 1;
                    return QuotientResolution::OptimizationUnavailable;
                }
                work += 1;
                let slot = frame.next_child;
                let child = key.children[slot];
                if let Some(earlier) = key.children[..slot]
                    .iter()
                    .position(|&previous| previous == child)
                {
                    frame.children[slot] = frame.children[earlier];
                    frame.next_child += 1;
                    self.stats.transform.orientation_requests += frame.needed.count_ones() as usize;
                    self.stats.transform.orientation_quotient_eliminations +=
                        frame.needed.count_ones() as usize;
                    continue;
                }
                let child_frame = self.orientation_frame(child, frame.needed);
                if child_frame.needed == 0 {
                    frame.children[slot] = child_frame.record.references;
                    frame.next_child += 1;
                    continue;
                }
                assert!(
                    depth < FRAME_COUNT,
                    "validated shape depth exceeds orientation workspace"
                );
                frames[depth] = child_frame;
                depth += 1;
                continue;
            }
            let Some(record) = self.complete_orientation_frame(*frame) else {
                return QuotientResolution::MandatoryFailure;
            };
            depth -= 1;
            if depth == 0 {
                return QuotientResolution::Complete(record);
            }
            let parent = &mut frames[depth - 1];
            parent.children[parent.next_child] = record.references;
            parent.next_child += 1;
        }
    }

    fn resolve_shape_orientations_scalar(
        &mut self,
        shape: CanonicalNodeRef,
        requested: u8,
    ) -> Option<OrientationRecord> {
        let mut aggregate = self.orientation_frame(shape, 0).record;
        for orientation in Symmetry::ALL {
            if requested & bit(orientation) == 0 || aggregate.resolved & bit(orientation) != 0 {
                continue;
            }
            let resolved = self.resolve_one_shape_orientation_scalar(shape, orientation)?;
            aggregate.merge(resolved);
        }
        self.publish_orientation_record(shape, aggregate);
        Some(aggregate)
    }

    fn prepare_scalar_orientation_frame(
        &mut self,
        shape: CanonicalNodeRef,
        requested: Symmetry,
    ) -> ScalarFramePreparation {
        let quotient = self.orientation_frame(shape, bit(requested));
        if quotient.needed == 0 {
            return ScalarFramePreparation::Complete(quotient.record);
        }
        debug_assert_eq!(quotient.needed.count_ones(), 1);
        let proof_index = usize::try_from(quotient.needed.trailing_zeros())
            .or_invariant("orientation proof index exceeds usize");
        let proof = *Symmetry::ALL
            .get(proof_index)
            .or_invariant("orientation proof index exceeds D4");
        ScalarFramePreparation::Pending(ScalarOrientationFrame {
            shape,
            requested,
            proof,
            record: quotient.record,
            ..ScalarOrientationFrame::EMPTY
        })
    }

    fn resolve_one_shape_orientation_scalar(
        &mut self,
        shape: CanonicalNodeRef,
        requested: Symmetry,
    ) -> Option<OrientationRecord> {
        let first = match self.prepare_scalar_orientation_frame(shape, requested) {
            ScalarFramePreparation::Complete(record) => return Some(record),
            ScalarFramePreparation::Pending(frame) => frame,
        };
        let mut frames = [ScalarOrientationFrame::EMPTY; FRAME_COUNT];
        frames[0] = first;
        let mut depth = 1;
        loop {
            let frame = &mut frames[depth - 1];
            let key = self.canonical_caches.shapes[frame.shape.index()].key;
            if frame.next_child < 4 {
                let output_slot = frame.next_child;
                let permutation = frame.proof.quadrant_perm();
                let child = key.children[permutation[output_slot]];
                if let Some(earlier) =
                    (0..output_slot).find(|&slot| key.children[permutation[slot]] == child)
                {
                    frame.children[output_slot] = frame.children[earlier];
                    frame.next_child += 1;
                    self.stats.transform.orientation_requests += 1;
                    self.stats.transform.orientation_quotient_eliminations += 1;
                    continue;
                }
                match self.prepare_scalar_orientation_frame(child, frame.proof) {
                    ScalarFramePreparation::Complete(record) => {
                        frame.children[output_slot] = record.reference(frame.proof);
                        frame.next_child += 1;
                    }
                    ScalarFramePreparation::Pending(child_frame) => {
                        if depth == FRAME_COUNT {
                            crate::invariant_failure!(
                                "validated shape depth exceeds scalar orientation workspace"
                            );
                        }
                        frames[depth] = child_frame;
                        depth += 1;
                    }
                }
                continue;
            }
            let record = self.complete_scalar_orientation_frame(*frame)?;
            let reference = record.reference(frame.requested);
            depth -= 1;
            if depth == 0 {
                return Some(record);
            }
            let parent = &mut frames[depth - 1];
            parent.children[parent.next_child] = reference;
            parent.next_child += 1;
        }
    }

    fn complete_scalar_orientation_frame(
        &mut self,
        mut frame: ScalarOrientationFrame,
    ) -> Option<OrientationRecord> {
        let original = self.canonical_caches.shapes[frame.shape.index()].key;
        let signature = self.canonical_parent_key(original.level, frame.children);
        let reference = if signature == original {
            self.stats.transform.orientation_signature_reuses += 1;
            frame.shape
        } else {
            self.stats.transform.orientation_resolved_signatures += 1;
            let reference = self.intern_canonical_shape(signature);
            if self.allocation_failed() {
                return None;
            }
            reference
        };
        frame.record.insert(frame.proof, reference);
        frame
            .record
            .insert(frame.requested, frame.record.reference(frame.proof));
        self.commit_proven_stabilizer(frame.shape, &mut frame.record);
        self.publish_orientation_record(frame.shape, frame.record);
        Some(frame.record)
    }

    fn complete_orientation_frame(
        &mut self,
        mut frame: OrientationFrame,
    ) -> Option<OrientationRecord> {
        let original = self.canonical_caches.shapes[frame.shape.index()].key;
        let mut signatures = [original; 8];
        let mut signature_refs = [frame.shape; 8];
        let mut unique = 1;
        for orientation in Symmetry::ALL {
            if frame.needed & bit(orientation) == 0 {
                continue;
            }
            let children = orientation
                .quadrant_perm()
                .map(|slot| frame.children[slot][orientation as usize]);
            let signature = self.canonical_parent_key(original.level, children);
            let reference = if let Some(previous) = signatures[..unique]
                .iter()
                .position(|&key| key == signature)
            {
                self.stats.transform.orientation_signature_reuses += 1;
                signature_refs[previous]
            } else {
                self.stats.transform.orientation_resolved_signatures += 1;
                let reference = self.intern_canonical_shape(signature);
                if self.allocation_failed() {
                    return None;
                }
                signatures[unique] = signature;
                signature_refs[unique] = reference;
                unique += 1;
                reference
            };
            frame.record.insert(orientation, reference);
        }
        for orientation in Symmetry::ALL {
            if frame.requested & bit(orientation) != 0 {
                let representative = frame.representatives[orientation as usize];
                frame
                    .record
                    .insert(orientation, frame.record.reference(representative));
            }
        }
        self.commit_proven_stabilizer(frame.shape, &mut frame.record);
        self.publish_orientation_record(frame.shape, frame.record);
        Some(frame.record)
    }

    fn commit_proven_stabilizer(
        &mut self,
        shape: CanonicalNodeRef,
        record: &mut OrientationRecord,
    ) {
        let existing = self.canonical_caches.shapes[shape.index()].stabilizer;
        let proven = existing | record.stabilizer(shape);
        let stabilizer = if proven == existing {
            existing
        } else {
            close_stabilizer(proven)
        };
        self.canonical_caches.shapes[shape.index()].stabilizer = stabilizer;
        record.extend_stabilizer(stabilizer);
    }

    fn publish_orientation_record(&mut self, shape: CanonicalNodeRef, record: OrientationRecord) {
        debug_assert_eq!(record.registry_epoch, self.canonical_caches.shape_epoch);
        self.publish_optional_cache(
            |engine| &engine.canonical_caches.symmetry_refs,
            |engine| &mut engine.canonical_caches.symmetry_refs,
            shape,
            ProbeKey::fingerprint(&shape),
            record,
        );
    }
}
