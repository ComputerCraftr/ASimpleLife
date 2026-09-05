use super::*;
use crate::RequiredExt;

#[derive(Clone, Copy)]
struct TransformFrame {
    packed: PackedNodeKey,
    next_child: usize,
    child_ids: [PackedTransformId; 4],
}

impl HashLifeEngine {
    pub(in crate::hashlife) fn ensure_packed_transform_state(&mut self) {
        if self.transform_state.nodes.is_empty() {
            self.reset_packed_transform_state();
        }
    }

    pub(in crate::hashlife) fn release_packed_transform_state(&mut self) {
        self.transform_state.canonical_cache.release_storage();
        self.transform_state.intern.release_storage();
        self.transform_state.nodes = Vec::new();
        self.transform_state.materialized = Vec::new();
        self.transform_state.packed_roots = Vec::new();
        #[cfg(test)]
        {
            self.transform_state.cache.release_storage();
        }
    }

    pub(in crate::hashlife) fn clear_packed_transform_state(&mut self) {
        #[cfg(test)]
        {
            self.transform_state.cache.clear();
        }
        self.transform_state.canonical_cache.reset();
        self.transform_state.intern.reset();
        self.transform_state.nodes.clear();
        self.transform_state.materialized.clear();
        self.transform_state.packed_roots.clear();
    }

    pub(in crate::hashlife) fn reset_packed_transform_state(&mut self) {
        self.clear_packed_transform_state();
        if !self.prepare_mandatory_transform_growth(2) {
            return;
        }

        for alive in [false, true] {
            let Ok(leaf_id) = PackedTransformId::try_from(self.transform_state.nodes.len()) else {
                self.reject_allocation(u128::MAX);
                return;
            };
            let population = u64::from(alive);
            self.transform_state.nodes.push(PackedTransformNode {
                level: 0,
                leaf_population: population,
                children: [PackedTransformId::ZERO; 4],
                canonical_ref: if alive {
                    CanonicalShapeId::LIVE
                } else {
                    CanonicalShapeId::DEAD
                },
            });
            self.transform_state.materialized.push(Some(if alive {
                self.live_leaf
            } else {
                self.dead_leaf
            }));
            self.transform_state
                .packed_roots
                .push(Some(PackedNodeKey::new(
                    0,
                    [
                        NodeId::from(alive),
                        NodeId::ZERO,
                        NodeId::ZERO,
                        NodeId::ZERO,
                    ],
                )));
            if self
                .transform_state
                .intern
                .try_insert(
                    PackedTransformShapeKey {
                        level: 0,
                        children: [
                            PackedTransformId::from(alive),
                            PackedTransformId::ZERO,
                            PackedTransformId::ZERO,
                            PackedTransformId::ZERO,
                        ],
                    },
                    leaf_id,
                )
                .is_err()
            {
                self.reject_allocation(1);
                return;
            }
        }
    }

    #[inline]
    fn leaf_transform_id(&self, alive: bool) -> PackedTransformId {
        PackedTransformId::from(alive)
    }

    fn intern_packed_transform_node(
        &mut self,
        level: u32,
        children: [PackedTransformId; 4],
    ) -> PackedTransformId {
        let shape = PackedTransformShapeKey { level, children };
        if let Some(existing) = self.transform_state.intern.get(&shape) {
            return existing;
        }
        if !self.prepare_mandatory_transform_growth(1) {
            return PackedTransformId::ZERO;
        }
        let Ok(id) = PackedTransformId::try_from(self.transform_state.nodes.len()) else {
            self.reject_allocation(u128::MAX);
            return PackedTransformId::ZERO;
        };
        let order_children =
            children.map(|child| self.transform_state.nodes[child.index()].canonical_ref);
        let structural = self.canonical_parent_key(level, order_children);
        let canonical_ref = self.intern_canonical_shape(structural);
        if self.allocation_failed() {
            return PackedTransformId::ZERO;
        }
        self.transform_state.nodes.push(PackedTransformNode {
            level,
            leaf_population: 0,
            children,
            canonical_ref,
        });
        self.transform_state.materialized.push(None);
        self.transform_state.packed_roots.push(None);
        if self.transform_state.intern.try_insert(shape, id).is_err() {
            self.transform_state.nodes.pop();
            self.transform_state.materialized.pop();
            self.transform_state.packed_roots.pop();
            self.reject_allocation(1);
            return PackedTransformId::ZERO;
        }
        id
    }

    pub(in crate::hashlife) fn transform_packed_node_key(
        &mut self,
        packed: PackedNodeKey,
        symmetry: Symmetry,
    ) -> PackedTransformId {
        self.ensure_packed_transform_state();
        if self.allocation_failed() {
            return PackedTransformId::ZERO;
        }
        if packed.level == 0 {
            return self.leaf_transform_id(packed.children[0] != NodeId::ZERO);
        }
        let root_key = PackedSymmetryKey { packed, symmetry };
        if let Some(transformed) = self.transform_state.canonical_cache.get(&root_key) {
            self.stats.transform.packed_recursive_transform_hits += 1;
            return transformed;
        }
        self.stats.transform.packed_recursive_transform_misses += 1;
        let empty_frame = TransformFrame {
            packed,
            next_child: 0,
            child_ids: [PackedTransformId::ZERO; 4],
        };
        let mut stack = [empty_frame; 64];
        let mut stack_len = 1;
        let mut completed = None;
        while stack_len != 0 {
            if let Some(child_id) = completed.take() {
                let parent = &mut stack[stack_len - 1];
                parent.child_ids[parent.next_child - 1] = child_id;
            }
            let current = stack[stack_len - 1].packed;
            let current_key = PackedSymmetryKey {
                packed: current,
                symmetry,
            };
            let mut descended = false;
            while stack[stack_len - 1].next_child < 4 {
                let child_index = stack[stack_len - 1].next_child;
                stack[stack_len - 1].next_child += 1;
                let child = self.node_columns.packed_key(current.children[child_index]);
                let child_id = if child.level == 0 {
                    Some(self.leaf_transform_id(child.children[0] != NodeId::ZERO))
                } else {
                    self.transform_state
                        .canonical_cache
                        .get(&PackedSymmetryKey {
                            packed: child,
                            symmetry,
                        })
                };
                if let Some(child_id) = child_id {
                    stack[stack_len - 1].child_ids[child_index] = child_id;
                    continue;
                }
                if stack_len == stack.len() {
                    crate::invariant_failure!("validated transform depth exceeded fixed workspace");
                }
                stack[stack_len] = TransformFrame {
                    packed: child,
                    next_child: 0,
                    child_ids: [PackedTransformId::ZERO; 4],
                };
                stack_len += 1;
                descended = true;
                break;
            }
            if descended {
                continue;
            }
            let child_ids = stack[stack_len - 1].child_ids;
            let perm = symmetry.quadrant_perm();
            let transformed = self.intern_packed_transform_node(
                current.level,
                [
                    child_ids[perm[0]],
                    child_ids[perm[1]],
                    child_ids[perm[2]],
                    child_ids[perm[3]],
                ],
            );
            self.publish_optional_cache(
                |engine| &engine.transform_state.canonical_cache,
                |engine| &mut engine.transform_state.canonical_cache,
                current_key,
                current_key.fingerprint(),
                transformed,
            );
            stack_len -= 1;
            if stack_len == 0 {
                return transformed;
            }
            completed = Some(transformed);
        }
        crate::invariant_failure!("transform traversal produced no root")
    }

    #[cfg(test)]
    pub(in crate::hashlife) fn compare_packed_transform_ids(
        &self,
        left: PackedTransformId,
        right: PackedTransformId,
    ) -> std::cmp::Ordering {
        if left == right {
            return std::cmp::Ordering::Equal;
        }
        let mut ordering = std::cmp::Ordering::Equal;
        let mut stack = [(left, right, 0_usize); 128];
        let mut stack_len = 1;
        while stack_len != 0 {
            stack_len -= 1;
            let (current_left, current_right, child_index) = stack[stack_len];
            if current_left == current_right {
                continue;
            }
            let left_node = self.transform_state.nodes[current_left.index()];
            let right_node = self.transform_state.nodes[current_right.index()];
            let node_ordering = left_node.level.cmp(&right_node.level).then_with(|| {
                if left_node.level == 0 {
                    left_node.leaf_population.cmp(&right_node.leaf_population)
                } else {
                    std::cmp::Ordering::Equal
                }
            });
            if node_ordering != std::cmp::Ordering::Equal {
                ordering = node_ordering;
                break;
            }
            if left_node.level == 0 {
                continue;
            }
            if child_index < 4 {
                if stack_len + 2 > stack.len() {
                    crate::invariant_failure!("validated compare depth exceeded fixed workspace");
                }
                stack[stack_len] = (current_left, current_right, child_index + 1);
                stack_len += 1;
                let next_left = left_node.children[child_index];
                let next_right = right_node.children[child_index];
                if next_left != next_right {
                    stack[stack_len] = (next_left, next_right, 0);
                    stack_len += 1;
                }
            }
        }
        ordering
    }
}

#[cfg(test)]
impl HashLifeEngine {
    fn materialize_packed_transform_node_internal(&mut self, id: PackedTransformId) -> NodeId {
        if let Some(node) = self.transform_state.materialized[id.index()] {
            return node;
        }
        let mut stack = [(id, false); 256];
        let mut stack_len = 1;
        while stack_len != 0 {
            stack_len -= 1;
            let (current, ready) = stack[stack_len];
            if self.transform_state.materialized[current.index()].is_some() {
                continue;
            }
            let transform_node = self.transform_state.nodes[current.index()];
            debug_assert!(transform_node.level != 0);
            if !ready {
                if stack_len + 5 > stack.len() {
                    crate::invariant_failure!(
                        "validated packed-root depth exceeded fixed workspace"
                    );
                }
                stack[stack_len] = (current, true);
                stack_len += 1;
                for child in transform_node.children {
                    if self.transform_state.materialized[child.index()].is_none() {
                        if self.transform_state.nodes[child.index()].level == 0 {
                            let alive =
                                self.transform_state.nodes[child.index()].leaf_population != 0;
                            self.transform_state.materialized[child.index()] = Some(if alive {
                                self.live_leaf
                            } else {
                                self.dead_leaf
                            });
                        } else {
                            stack[stack_len] = (child, false);
                            stack_len += 1;
                        }
                    }
                }
                continue;
            }
            let children = transform_node.children.map(|child| {
                self.transform_state.materialized[child.index()].or_invariant(
                    "iterative transform materialization must resolve child before parent",
                )
            });
            let node = self.join(children[0], children[1], children[2], children[3]);
            self.transform_state.materialized[current.index()] = Some(node);
        }
        self.transform_state.materialized[id.index()]
            .or_invariant("iterative transform materialization must resolve target")
    }
}

impl HashLifeEngine {
    pub(super) fn packed_root_from_transform_id(&mut self, id: PackedTransformId) -> PackedNodeKey {
        if let Some(packed) = self.transform_state.packed_roots[id.index()] {
            return packed;
        }
        let mut stack = [(id, false); 256];
        let mut stack_len = 1;
        while stack_len != 0 {
            stack_len -= 1;
            let (current, ready) = stack[stack_len];
            if self.transform_state.packed_roots[current.index()].is_some() {
                continue;
            }
            let transform_node = self.transform_state.nodes[current.index()];
            debug_assert!(transform_node.level != 0);
            if !ready {
                if stack_len + 5 > stack.len() {
                    crate::invariant_failure!(
                        "validated packed-root depth exceeded fixed workspace"
                    );
                }
                stack[stack_len] = (current, true);
                stack_len += 1;
                for child in transform_node.children {
                    if self.transform_state.packed_roots[child.index()].is_none() {
                        if self.transform_state.nodes[child.index()].level == 0 {
                            let leaf_population =
                                self.transform_state.nodes[child.index()].leaf_population;
                            self.transform_state.packed_roots[child.index()] =
                                Some(PackedNodeKey::new(
                                    0,
                                    [
                                        NodeId::from(leaf_population != 0),
                                        NodeId::ZERO,
                                        NodeId::ZERO,
                                        NodeId::ZERO,
                                    ],
                                ));
                        } else {
                            stack[stack_len] = (child, false);
                            stack_len += 1;
                        }
                    }
                }
                continue;
            }
            let child_roots = transform_node.children.map(|child| {
                self.transform_state.packed_roots[child.index()]
                    .or_invariant("iterative packed roots must resolve children before parent")
            });
            let packed = PackedNodeKey::new(
                transform_node.level,
                child_roots.map(|child| self.materialize_packed_node_key_internal(child)),
            );
            self.transform_state.packed_roots[current.index()] = Some(packed);
        }
        self.transform_state.packed_roots[id.index()]
            .or_invariant("iterative packed root must resolve target")
    }
}

#[cfg(test)]
impl HashLifeEngine {
    pub(in crate::hashlife) fn materialize_winning_packed_transform_root(
        &mut self,
        id: PackedTransformId,
    ) -> PackedNodeKey {
        self.packed_root_from_transform_id(id)
    }

    pub(in crate::hashlife) fn materialize_packed_transform_root(
        &mut self,
        id: PackedTransformId,
    ) -> NodeId {
        self.stats.transform.packed_cache_result_materializations += 1;
        self.materialize_packed_transform_node_internal(id)
    }
}

impl HashLifeEngine {
    #[inline]
    pub(in crate::hashlife) fn canonical_jump_probe(
        &mut self,
        key: (NodeId, u32),
    ) -> CanonicalJumpProbe {
        let symmetry_admitted = self.record_symmetry_gate_decision(key.0);
        let (node, _packed_fingerprint, used_cached_fingerprint) = if symmetry_admitted {
            let canonical = self.canonicalize_packed_node(key.0);
            (
                canonical.node,
                canonical.fingerprint,
                canonical.used_cached_fingerprint,
            )
        } else {
            let canonical = self.canonicalize_blocked_jump_node(key.0);
            (
                canonical.node,
                canonical.fingerprint,
                canonical.used_cached_fingerprint,
            )
        };
        CanonicalJumpProbe {
            key: CanonicalJumpKey {
                structural: node.structural,
                step_exp: key.1,
                symmetry_admitted,
            },
            node,
            fingerprint: CanonicalJumpKey {
                structural: node.structural,
                step_exp: key.1,
                symmetry_admitted,
            }
            .fingerprint(),
            used_cached_fingerprint,
        }
    }

    #[inline]
    pub(in crate::hashlife) fn record_symmetry_gate_decision(&mut self, node: NodeId) -> bool {
        let allowed = self.should_symmetry_canonicalize_jump_node(node);
        if allowed {
            self.stats.canonical_cache.symmetry_gate_allowed += 1;
        } else {
            self.stats.canonical_cache.symmetry_gate_blocked += 1;
        }
        allowed
    }
}

#[cfg(test)]
impl HashLifeEngine {
    pub(in crate::hashlife) fn transform_node(
        &mut self,
        node: NodeId,
        symmetry: Symmetry,
    ) -> NodeId {
        if symmetry == Symmetry::Identity || self.node_columns.level(node) == 0 {
            return node;
        }
        if let Some(transformed) = self
            .transform_state
            .cache
            .get(&TransformCacheKey { node, symmetry })
        {
            return transformed;
        }
        let mut stack = Vec::with_capacity(
            (self.node_columns.level(node) as usize)
                .saturating_mul(4)
                .max(8),
        );
        stack.push((node, false));
        while let Some((current, ready)) = stack.pop() {
            if self.node_columns.level(current) == 0 {
                continue;
            }
            let key = TransformCacheKey {
                node: current,
                symmetry,
            };
            if self.transform_state.cache.get(&key).is_some() {
                continue;
            }
            if !ready {
                stack.push((current, true));
                for child in self.node_columns.quadrants(current) {
                    if self.node_columns.level(child) != 0
                        && self
                            .transform_state
                            .cache
                            .get(&TransformCacheKey {
                                node: child,
                                symmetry,
                            })
                            .is_none()
                    {
                        stack.push((child, false));
                    }
                }
                continue;
            }
            let transformed_children = self.node_columns.quadrants(current).map(|child| {
                if self.node_columns.level(child) == 0 {
                    child
                } else {
                    self.transform_state
                        .cache
                        .get(&TransformCacheKey {
                            node: child,
                            symmetry,
                        })
                        .or_invariant("iterative transform_node must resolve child before parent")
                }
            });
            let [next_nw, next_ne, next_sw, next_se] =
                symmetry.transform_quadrants(transformed_children);
            let transformed = self.join(next_nw, next_ne, next_sw, next_se);
            self.stats.materialization.transformed_node_materializations += 1;
            self.transform_state
                .cache
                .try_insert(key, transformed)
                .or_invariant("test transform cache allocation failed");
        }
        self.transform_state
            .cache
            .get(&TransformCacheKey { node, symmetry })
            .or_invariant("iterative transform_node must resolve target")
    }

    pub(in crate::hashlife) fn transform_node_batch<const N: usize>(
        &mut self,
        nodes: [NodeId; N],
        symmetry: Symmetry,
    ) -> [NodeId; N] {
        if symmetry == Symmetry::Identity {
            return nodes;
        }

        let mut transformed = [NodeId::ZERO; N];
        for lane in 0..N {
            let node = nodes[lane];
            if self.node_columns.level(node) == 0 {
                transformed[lane] = node;
                continue;
            }
            let mut reused = None;
            for prev in 0..lane {
                if nodes[prev] == node {
                    reused = Some(transformed[prev]);
                    break;
                }
            }
            transformed[lane] = reused.unwrap_or_else(|| self.transform_node(node, symmetry));
        }
        transformed
    }
}

#[cfg(test)]
mod allocation_tests {
    use super::*;

    #[test]
    fn packed_transform_reset_preflights_all_mandatory_columns() {
        let mut engine = HashLifeEngine::default();
        engine.release_packed_transform_state();
        let retained = super::super::super::memory::wide_allocated_bytes(engine.allocated_bytes());
        engine.begin_allocation_transaction(retained);

        engine.ensure_packed_transform_state();

        assert!(engine.transform_state.nodes.is_empty());
        assert!(engine.transform_state.materialized.is_empty());
        assert!(engine.transform_state.packed_roots.is_empty());
        assert!(
            matches!(
                engine.take_allocation_failure(),
                Some(EngineAllocationFailure::Allocation { .. })
            ),
            "transform growth did not report allocation failure"
        );

        engine.begin_allocation_transaction(u128::MAX);
        engine.ensure_packed_transform_state();
        assert_eq!(engine.transform_state.nodes.len(), 2);
        assert_eq!(engine.transform_state.materialized.len(), 2);
        assert_eq!(engine.transform_state.packed_roots.len(), 2);
        assert!(engine.take_allocation_failure().is_none());
    }

    #[test]
    fn pressured_transform_caches_do_not_control_exact_transform_results() {
        let mut engine = HashLifeEngine::default();
        engine.begin_allocation_transaction(u128::MAX);
        let source = engine.join(
            engine.live_leaf,
            engine.dead_leaf,
            engine.dead_leaf,
            engine.dead_leaf,
        );
        let packed = engine.node_columns.packed_key(source);
        let expected = engine.transform_packed_node_key(packed, Symmetry::Rotate90);
        assert_eq!(engine.take_allocation_failure(), None);

        engine.transform_state.canonical_cache.release_storage();
        let retained = super::super::super::memory::wide_allocated_bytes(engine.allocated_bytes());
        engine.begin_allocation_transaction(retained);
        let transformed = engine.transform_packed_node_key(packed, Symmetry::Rotate90);

        assert_eq!(engine.take_allocation_failure(), None);
        assert_eq!(transformed, expected);
        assert_eq!(engine.transform_state.canonical_cache.len(), 0);
    }
}
