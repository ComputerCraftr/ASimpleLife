use super::*;

pub(super) fn notify_dependents(
    engine: &mut HashLifeEngine,
    key: &CanonicalJumpKey,
    tasks: &mut [Option<TaskRecord>],
    dependents: &mut ProbeTable<CanonicalJumpKey, usize>,
    dependent_edges: &[DependentEdge],
    ready: &mut Vec<usize>,
) {
    if let Some(mut head) = dependents.remove(key) {
        while head != NO_DEPENDENT {
            let waiter_id = dependent_edges[head].task_id;
            let Some(task) = tasks[waiter_id].as_mut() else {
                crate::invariant_failure!(
                    "dependent edge referenced missing recursive task waiter_id={waiter_id}"
                );
            };
            task.remaining -= 1;
            if task.remaining == 0 && !engine.try_push_transient(ready, waiter_id) {
                return;
            }
            head = dependent_edges[head].next;
        }
    }
}

pub(super) fn notify_step0_dependents(
    engine: &mut HashLifeEngine,
    key: CanonicalJumpKey,
    tasks: &mut [Option<Step0TaskRecord>],
    dependents: &mut ProbeTable<CanonicalJumpKey, usize>,
    dependent_edges: &[DependentEdge],
    ready: &mut Vec<usize>,
) {
    if let Some(mut head) = dependents.remove(&key) {
        while head != NO_DEPENDENT {
            let waiter_id = dependent_edges[head].task_id;
            let Some(task) = tasks[waiter_id].as_mut() else {
                crate::invariant_failure!(
                    "dependent edge referenced missing step0 task waiter_id={waiter_id}"
                );
            };
            task.remaining -= 1;
            if task.remaining == 0 && !engine.try_push_transient(ready, waiter_id) {
                return;
            }
            head = dependent_edges[head].next;
        }
    }
}

pub(super) fn push_dependent(
    engine: &mut HashLifeEngine,
    dependents: &mut ProbeTable<CanonicalJumpKey, usize>,
    dependent_edges: &mut Vec<DependentEdge>,
    key: CanonicalJumpKey,
    task_id: usize,
) -> bool {
    let next = dependents.get(&key).unwrap_or(NO_DEPENDENT);
    let head = dependent_edges.len();
    if !engine.try_push_transient(dependent_edges, DependentEdge { task_id, next }) {
        return false;
    }
    engine.try_insert_transient_table(dependents, key, head)
}
