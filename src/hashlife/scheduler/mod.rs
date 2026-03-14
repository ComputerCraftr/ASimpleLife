use super::*;

mod deps;
mod discovery;
mod flush;

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
