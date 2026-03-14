use super::*;

pub(super) fn notify_dependents(
    key: &CanonicalJumpKey,
    tasks: &mut [Option<TaskRecord>],
    dependents: &mut FlatTable<CanonicalJumpKey, usize>,
    dependent_edges: &[DependentEdge],
    ready: &mut Vec<usize>,
) {
    if let Some(mut head) = dependents.remove(key) {
        while head != NO_DEPENDENT {
            let waiter_id = dependent_edges[head].task_id;
            let Some(task) = tasks[waiter_id].as_mut() else {
                panic!("dependent edge referenced missing recursive task waiter_id={waiter_id}");
            };
            task.remaining -= 1;
            if task.remaining == 0 {
                ready.push(waiter_id);
            }
            head = dependent_edges[head].next;
        }
    }
}

pub(super) fn notify_step0_dependents(
    key: CanonicalJumpKey,
    tasks: &mut [Option<Step0TaskRecord>],
    dependents: &mut FlatTable<CanonicalJumpKey, usize>,
    dependent_edges: &[DependentEdge],
    ready: &mut Vec<usize>,
) {
    if let Some(mut head) = dependents.remove(&key) {
        while head != NO_DEPENDENT {
            let waiter_id = dependent_edges[head].task_id;
            let Some(task) = tasks[waiter_id].as_mut() else {
                panic!("dependent edge referenced missing step0 task waiter_id={waiter_id}");
            };
            task.remaining -= 1;
            if task.remaining == 0 {
                ready.push(waiter_id);
            }
            head = dependent_edges[head].next;
        }
    }
}

pub(super) fn push_dependent(
    dependents: &mut FlatTable<CanonicalJumpKey, usize>,
    dependent_edges: &mut Vec<DependentEdge>,
    key: CanonicalJumpKey,
    task_id: usize,
) {
    let next = dependents.get(&key).unwrap_or(NO_DEPENDENT);
    let head = dependent_edges.len();
    dependent_edges.push(DependentEdge { task_id, next });
    dependents.insert(key, head);
}
