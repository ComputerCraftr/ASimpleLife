use super::{
    CanonicalCaches, CanonicalShapeMeta, NodeId, PackedNodeKey, PackedTransformNode, ResultCaches,
    TransformState,
};
use crate::probe_table::ProbeTable;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CacheStorageLifecycle {
    RetainCapacity,
    ReleaseStorage,
}

impl CacheStorageLifecycle {
    fn apply<K: Copy + Eq, V: Copy>(self, table: &mut ProbeTable<K, V>) {
        match self {
            Self::RetainCapacity => table.reset(),
            Self::ReleaseStorage => table.release_storage(),
        }
    }
}

impl ResultCaches {
    pub(super) fn allocated_bytes(&self) -> usize {
        let Self {
            jump,
            root,
            overlap,
            oriented,
            materialized_packed,
            structural_fast_path,
            packed_structural_fast_path,
            shells,
            bounds,
        } = self;

        jump.allocated_bytes()
            + root.allocated_bytes()
            + overlap.allocated_bytes()
            + oriented.allocated_bytes()
            + materialized_packed.allocated_bytes()
            + structural_fast_path.allocated_bytes()
            + packed_structural_fast_path.allocated_bytes()
            + shells.allocated_bytes()
            + bounds.allocated_bytes()
    }

    pub(super) fn apply_lifecycle(&mut self, lifecycle: CacheStorageLifecycle) {
        let Self {
            jump,
            root,
            overlap,
            oriented,
            materialized_packed,
            structural_fast_path,
            packed_structural_fast_path,
            shells,
            bounds,
        } = self;

        lifecycle.apply(jump);
        lifecycle.apply(root);
        lifecycle.apply(overlap);
        lifecycle.apply(oriented);
        lifecycle.apply(materialized_packed);
        lifecycle.apply(structural_fast_path);
        lifecycle.apply(packed_structural_fast_path);
        lifecycle.apply(shells);
        lifecycle.apply(bounds);
    }
}

impl CanonicalCaches {
    pub(super) fn allocated_bytes(&self) -> usize {
        let Self {
            shape_epoch: _,
            shape_intern,
            shapes,
            node,
            packed,
            hot_packed,
            hot_packed_budget: _,
            oriented,
            hot_oriented,
            hot_oriented_budget: _,
            direct_parent,
            hot_direct_parent,
            hot_direct_parent_budget: _,
            symmetry_refs,
        } = self;

        shape_intern.allocated_bytes()
            + shapes.capacity() * std::mem::size_of::<CanonicalShapeMeta>()
            + node.allocated_bytes()
            + packed.allocated_bytes()
            + hot_packed.allocated_bytes()
            + oriented.allocated_bytes()
            + hot_oriented.allocated_bytes()
            + direct_parent.allocated_bytes()
            + hot_direct_parent.allocated_bytes()
            + symmetry_refs.allocated_bytes()
    }

    pub(super) fn apply_lifecycle(&mut self, lifecycle: CacheStorageLifecycle, preserve_hot: bool) {
        // Mandatory shape registry state and budgets survive optional cache lifecycle changes.
        let Self {
            shape_epoch: _,
            shape_intern: _,
            shapes: _,
            node,
            packed,
            hot_packed,
            hot_packed_budget: _,
            oriented,
            hot_oriented,
            hot_oriented_budget: _,
            direct_parent,
            hot_direct_parent,
            hot_direct_parent_budget: _,
            symmetry_refs,
        } = self;

        lifecycle.apply(node);
        lifecycle.apply(packed);
        lifecycle.apply(oriented);
        lifecycle.apply(direct_parent);
        lifecycle.apply(symmetry_refs);
        if !preserve_hot {
            lifecycle.apply(hot_packed);
            lifecycle.apply(hot_oriented);
            lifecycle.apply(hot_direct_parent);
        }
    }
}

impl TransformState {
    pub(super) fn allocated_bytes(&self) -> usize {
        let Self {
            #[cfg(test)]
                cache: _,
            canonical_cache,
            intern,
            nodes,
            materialized,
            packed_roots,
        } = self;
        // Test-only transform cache storage is intentionally outside the engine memory model.

        canonical_cache.allocated_bytes()
            + intern.allocated_bytes()
            + nodes.capacity() * std::mem::size_of::<PackedTransformNode>()
            + materialized.capacity() * std::mem::size_of::<Option<NodeId>>()
            + packed_roots.capacity() * std::mem::size_of::<Option<PackedNodeKey>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hashlife::{
        CanonicalNodeIdentity, CanonicalStructKey, HashLifeEngine, NodeId, PackedNodeKey,
    };
    use crate::probe_table::{ProbeMode, ProbeTable};
    use crate::symmetry::D4Symmetry as Symmetry;

    #[test]
    fn lifecycle_clears_while_retaining_capacity_and_releases_storage() {
        let mut table = ProbeTable::with_capacity(ProbeMode::Mutable, 128);
        table.insert(NodeId::ZERO, NodeId::ZERO);
        let capacity = table.capacity();
        let allocated_bytes = table.allocated_bytes();

        CacheStorageLifecycle::RetainCapacity.apply(&mut table);

        assert_eq!(table.len(), 0);
        assert_eq!(table.capacity(), capacity);
        assert_eq!(table.allocated_bytes(), allocated_bytes);

        CacheStorageLifecycle::ReleaseStorage.apply(&mut table);

        assert_eq!(table.len(), 0);
        assert_eq!(table.capacity(), 0);
        assert_eq!(table.allocated_bytes(), 0);
    }

    #[test]
    fn canonical_lifecycle_preserves_requested_hot_entries_and_mandatory_registry() {
        let mut engine = HashLifeEngine::default();
        let packed = PackedNodeKey::new(0, [NodeId::ZERO; 4]);
        let identity = CanonicalNodeIdentity {
            packed,
            structural: CanonicalStructKey::leaf(false),
            symmetry: Symmetry::Identity,
        };
        engine.canonical_caches.packed.insert(packed, identity);
        engine.canonical_caches.hot_packed.insert(packed, identity);
        engine.canonical_caches.hot_packed_budget = 11;
        engine.canonical_caches.hot_oriented_budget = 13;
        engine.canonical_caches.hot_direct_parent_budget = 17;

        let shape_epoch = engine.canonical_caches.shape_epoch;
        let shape_count = engine.canonical_caches.shapes.len();
        let dead_shape = engine
            .canonical_caches
            .shape_intern
            .get(&identity.structural);
        let shape_intern_capacity = engine.canonical_caches.shape_intern.capacity();

        engine
            .canonical_caches
            .apply_lifecycle(CacheStorageLifecycle::RetainCapacity, true);

        assert_eq!(engine.canonical_caches.packed.len(), 0);
        assert_eq!(engine.canonical_caches.hot_packed.len(), 1);
        assert_eq!(engine.canonical_caches.shape_epoch, shape_epoch);
        assert_eq!(engine.canonical_caches.shapes.len(), shape_count);
        assert_eq!(
            engine
                .canonical_caches
                .shape_intern
                .get(&identity.structural),
            dead_shape
        );
        assert_eq!(
            engine.canonical_caches.shape_intern.capacity(),
            shape_intern_capacity
        );
        assert_eq!(engine.canonical_caches.hot_packed_budget, 11);
        assert_eq!(engine.canonical_caches.hot_oriented_budget, 13);
        assert_eq!(engine.canonical_caches.hot_direct_parent_budget, 17);
    }

    #[test]
    fn owning_lifecycles_release_optional_table_allocations_only() {
        let mut engine = HashLifeEngine::default();
        let mandatory_shape_bytes = engine.canonical_caches.shape_intern.allocated_bytes();
        let mandatory_shape_capacity = engine.canonical_caches.shapes.capacity();

        engine
            .result_caches
            .apply_lifecycle(CacheStorageLifecycle::ReleaseStorage);
        engine
            .canonical_caches
            .apply_lifecycle(CacheStorageLifecycle::ReleaseStorage, false);

        assert_eq!(engine.result_caches.jump.allocated_bytes(), 0);
        assert_eq!(engine.result_caches.bounds.allocated_bytes(), 0);
        assert_eq!(engine.canonical_caches.node.allocated_bytes(), 0);
        assert_eq!(
            engine.canonical_caches.hot_direct_parent.allocated_bytes(),
            0
        );
        assert_eq!(
            engine.canonical_caches.shape_intern.allocated_bytes(),
            mandatory_shape_bytes
        );
        assert_eq!(
            engine.canonical_caches.shapes.capacity(),
            mandatory_shape_capacity
        );
    }

    #[test]
    fn owner_accounting_matches_engine_deltas_across_clear_and_release() {
        let mut engine = HashLifeEngine::default();
        let transform_bytes_without_test_cache = engine.transform_state.allocated_bytes();
        let engine_bytes_without_test_cache = engine.allocated_bytes();
        let test_cache_bytes = engine.transform_state.cache.allocated_bytes();
        assert_eq!(
            engine.transform_state.cache.try_reserve(1_024),
            Ok(()),
            "test transform cache growth should succeed"
        );
        assert!(engine.transform_state.cache.allocated_bytes() > test_cache_bytes);
        assert_eq!(
            engine.transform_state.allocated_bytes(),
            transform_bytes_without_test_cache
        );
        assert_eq!(engine.allocated_bytes(), engine_bytes_without_test_cache);

        engine.transform_state.nodes.reserve(128);
        engine.transform_state.materialized.reserve(128);
        engine.transform_state.packed_roots.reserve(128);

        let owner_bytes_before = engine.result_caches.allocated_bytes()
            + engine.canonical_caches.allocated_bytes()
            + engine.transform_state.allocated_bytes();
        let engine_bytes_before = engine.allocated_bytes();

        engine
            .result_caches
            .apply_lifecycle(CacheStorageLifecycle::RetainCapacity);
        engine
            .canonical_caches
            .apply_lifecycle(CacheStorageLifecycle::RetainCapacity, false);
        engine.clear_packed_transform_state();

        assert_eq!(
            engine.result_caches.allocated_bytes()
                + engine.canonical_caches.allocated_bytes()
                + engine.transform_state.allocated_bytes(),
            owner_bytes_before
        );
        assert_eq!(engine.allocated_bytes(), engine_bytes_before);

        engine
            .result_caches
            .apply_lifecycle(CacheStorageLifecycle::ReleaseStorage);
        engine
            .canonical_caches
            .apply_lifecycle(CacheStorageLifecycle::ReleaseStorage, false);
        engine.release_packed_transform_state();

        let owner_bytes_after = engine.result_caches.allocated_bytes()
            + engine.canonical_caches.allocated_bytes()
            + engine.transform_state.allocated_bytes();
        let engine_bytes_after = engine.allocated_bytes();
        assert_eq!(
            engine_bytes_before - engine_bytes_after,
            owner_bytes_before - owner_bytes_after
        );
        assert_eq!(engine.result_caches.allocated_bytes(), 0);
        assert_eq!(engine.transform_state.allocated_bytes(), 0);
        assert_eq!(
            engine.canonical_caches.allocated_bytes(),
            engine.canonical_caches.shape_intern.allocated_bytes()
                + engine.canonical_caches.shapes.capacity()
                    * std::mem::size_of::<CanonicalShapeMeta>()
        );
    }
}
