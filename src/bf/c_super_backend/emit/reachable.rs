use super::batches::{seq_add_batch_end, seq_clear_batch_end};
use super::*;

pub(super) fn executable_nodes(engine: &mut EmitterEngine, root: NodeId) -> Vec<bool> {
    let mut required = vec![false; engine.interner.len()];
    let mut pending = vec![root];
    while let Some(id) = pending.pop() {
        let index = usize::try_from(id.0).or_invariant("super-C node id exceeded usize");
        if required[index] {
            continue;
        }
        required[index] = true;
        match engine.plan_node(id) {
            ExecPlan::ExactLoopMemo { body, .. } | ExecPlan::ExactPoweredLoopMemo { body, .. } => {
                pending.push(body)
            }
            _ => push_raw_dependencies(engine, id, &mut pending),
        }
    }
    required
}

fn push_raw_dependencies(engine: &EmitterEngine, id: NodeId, pending: &mut Vec<NodeId>) {
    match engine.interner.get(id) {
        NodeKind::Seq(children) => {
            let mut index = 0;
            while index < children.len() {
                if let Some(next) = seq_add_batch_end(engine, children, index)
                    .or_else(|| seq_clear_batch_end(engine, children, index))
                {
                    index = next;
                } else {
                    pending.push(children[index]);
                    index += 1;
                }
            }
        }
        NodeKind::Loop(body) => pending.push(*body),
        _ => {}
    }
}
