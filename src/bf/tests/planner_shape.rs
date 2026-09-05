use super::*;
use crate::bf::c_super_backend::{EmitterEngine, ExecPlan, NodeId};

pub(super) fn program_plans(program: &[BfIr], opts: CodegenOpts) -> Vec<ExecPlan> {
    let mut engine = EmitterEngine::with_opts(opts);
    engine.build_program(program);
    (0..engine.interner.len())
        .map(|index| engine.plan_node(NodeId(u32::try_from(index).or_invariant("test node index"))))
        .collect()
}

#[test]
fn symbolic_regions_choose_typed_memo_and_residual_plans() {
    let plans = program_plans(&parse_and_opt("+++[->+<]>+."), default_c_opts());
    assert!(
        plans
            .iter()
            .any(|plan| matches!(plan, ExecPlan::ExactMemo(_))),
        "{plans:?}"
    );
    assert_eq!(
        plans.last(),
        Some(&ExecPlan::Residual),
        "output region must remain residual: {plans:?}"
    );
    let plans = program_plans(&parse_and_opt("[->+<+]"), default_c_opts());
    assert!(
        plans
            .iter()
            .any(|plan| matches!(plan, ExecPlan::ExactLoopMemo { .. })),
        "{plans:?}"
    );
    let plans = program_plans(&parse_and_opt("++>+<"), default_c_opts());
    assert!(
        matches!(plans.last(), Some(ExecPlan::ExactMemo(_))),
        "{plans:?}"
    );
}

#[test]
fn powered_plans_require_an_eligible_loop_not_io_or_an_already_summarized_transfer() {
    for (source, expected) in [("[--]", true), ("[->+<]", false), ("[, .]", false)] {
        let plans = program_plans(&parse_and_opt(source), default_c_opts());
        assert_eq!(
            plans
                .iter()
                .any(|plan| matches!(plan, ExecPlan::ExactPoweredLoopMemo { .. })),
            expected,
            "source={source} plans={plans:?}"
        );
    }
}
