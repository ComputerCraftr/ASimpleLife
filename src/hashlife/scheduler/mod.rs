use super::*;

mod deps;
mod discovery;
mod flush;

pub(super) struct RecursiveSchedulerState<'a> {
    pub(super) discover: &'a mut Vec<DiscoveredJumpTask>,
    pub(super) task_index: &'a mut ProbeTable<CanonicalJumpKey, usize>,
    pub(super) tasks: &'a mut Vec<Option<TaskRecord>>,
    pub(super) task_keys: &'a mut Vec<Option<CanonicalJumpKey>>,
    pub(super) dependents: &'a mut ProbeTable<CanonicalJumpKey, usize>,
    pub(super) dependent_edges: &'a mut Vec<DependentEdge>,
    pub(super) ready: &'a mut Vec<usize>,
}

pub(super) struct Step0SchedulerState<'a> {
    pub(super) discover: &'a mut Vec<DiscoveredJumpTask>,
    pub(super) task_index: &'a mut ProbeTable<CanonicalJumpKey, usize>,
    pub(super) tasks: &'a mut Vec<Option<Step0TaskRecord>>,
    pub(super) task_keys: &'a mut Vec<Option<CanonicalJumpKey>>,
    pub(super) dependents: &'a mut ProbeTable<CanonicalJumpKey, usize>,
    pub(super) dependent_edges: &'a mut Vec<DependentEdge>,
    pub(super) ready: &'a mut Vec<usize>,
}

impl HashLifeEngine {
    pub(super) fn advance_power_of_two_recursive(
        &mut self,
        root_node: NodeId,
        root_step_exp: u32,
    ) -> NodeId {
        self.advance_power_of_two_recursive_impl(root_node, root_step_exp)
    }

    pub(super) fn advance_one_generation_centered(&mut self, root_node: NodeId) -> NodeId {
        self.advance_one_generation_centered_impl(root_node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequiredExt;

    #[test]
    fn transient_scheduler_vector_growth_fails_before_mutation() {
        let mut engine = HashLifeEngine::default();
        let retained = super::super::memory::wide_allocated_bytes(engine.allocated_bytes());
        engine.begin_allocation_transaction(retained + std::mem::size_of::<u64>() as u128);
        let mut values = engine
            .try_transient_vec::<u64>(1)
            .or_invariant("one scheduler lane should fit");
        values.push(7);

        assert!(
            !engine.try_push_transient(&mut values, 9),
            "growth beyond the transaction budget unexpectedly succeeded"
        );
        assert_eq!(values, [7], "failed growth mutated scheduler state");
        assert!(
            matches!(
                engine.take_allocation_failure(),
                Some(EngineAllocationFailure::Allocation { .. })
            ),
            "failed scheduler growth did not report typed allocation failure"
        );
    }

    #[test]
    fn transient_scheduler_table_rehash_fails_before_insertion() {
        let mut engine = HashLifeEngine::default();
        engine.begin_allocation_transaction(u128::MAX);
        let mut table = engine
            .try_transient_probe_table::<CanonicalJumpKey, usize>(8)
            .or_invariant("scheduler table should allocate");
        let table_capacity = table.capacity();
        for index in 0..table_capacity {
            let index = u32::try_from(index).or_invariant("test table capacity should fit u32");
            let key = CanonicalJumpKey {
                structural: CanonicalStructKey::synthetic(
                    index + 2,
                    [CanonicalShapeId::from_raw(index); 4],
                ),
                step_exp: 0,
                symmetry_admitted: false,
            };
            assert!(engine.try_insert_transient_table(
                &mut table,
                key,
                usize::try_from(index).or_invariant("test index should fit usize")
            ));
        }
        engine.take_allocation_failure();
        let retained = super::super::memory::wide_allocated_bytes(engine.allocated_bytes());
        engine.begin_allocation_transaction(retained);
        let before = table.len();
        assert_eq!(before, table_capacity, "fixture must fill the table");
        let key = CanonicalJumpKey {
            structural: CanonicalStructKey::synthetic(99, [CanonicalShapeId::from_raw(99); 4]),
            step_exp: 0,
            symmetry_admitted: false,
        };

        assert!(!engine.try_insert_transient_table(&mut table, key, 99));
        assert_eq!(table.len(), before, "failed rehash inserted a task");
    }
}
