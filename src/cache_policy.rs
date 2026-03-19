pub(crate) const HASHLIFE_GC_MIN_NODES: usize = 4_096;
pub(crate) const HASHLIFE_GC_GROWTH_TRIGGER: usize = 1_024;
pub(crate) const HASHLIFE_GC_MIN_RECLAIM: usize = 256;
pub(crate) const HASHLIFE_TRANSIENT_CACHE_GROWTH_TRIGGER: usize = 131_072;

pub(crate) const SIMD_TRANSITION_CACHE_LIMIT: usize = 262_144;
pub(crate) const SIMD_CANONICALIZATION_CACHE_LIMIT: usize = 262_144;
pub(crate) const SIMD_RETAINED_CACHE_CAPACITY: usize = 4_096;

pub(crate) fn should_collect_simd_transition_caches(
    transition_cache_len: usize,
    canonicalization_cache_len: usize,
) -> bool {
    transition_cache_len > SIMD_TRANSITION_CACHE_LIMIT
        || canonicalization_cache_len > SIMD_CANONICALIZATION_CACHE_LIMIT
}

pub(crate) fn hashlife_gc_reason(
    previous_root_changed: bool,
    node_count: usize,
    last_gc_nodes: usize,
) -> &'static str {
    let grew = node_count.saturating_sub(last_gc_nodes) >= HASHLIFE_GC_GROWTH_TRIGGER;
    if previous_root_changed {
        "root_changed"
    } else if node_count >= HASHLIFE_GC_MIN_NODES {
        "node_threshold"
    } else if grew {
        "growth_threshold"
    } else {
        "skip"
    }
}

pub(crate) fn should_run_active_hashlife_gc(node_count: usize, last_gc_nodes: usize) -> bool {
    node_count >= HASHLIFE_GC_MIN_NODES
        || node_count.saturating_sub(last_gc_nodes) >= HASHLIFE_GC_GROWTH_TRIGGER
}
