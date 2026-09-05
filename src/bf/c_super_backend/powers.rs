use super::*;

const POWER_EMISSION_BUDGET: usize = 512;

pub(super) fn transfer_cost(transfer: &SymbolicTransfer) -> usize {
    crate::bf::polynomial_emit::evaluation_cost(transfer)
}

pub(super) fn maximum_proven_iterations(opts: super::super::CodegenOpts, delta: i64) -> u64 {
    let bits = opts.cell_bits.min(63);
    if bits == 0 || delta == 0 || delta == i64::MIN {
        return 0;
    }
    // Match the emitted monotone-drain proof, not all modularly terminating inputs.
    let magnitude = match (opts.cell_sign, delta < 0) {
        (super::super::CellSign::Unsigned, true) => (1_u64 << bits) - 1,
        (super::super::CellSign::Unsigned, false) => return 0,
        (super::super::CellSign::Signed, true) => (1_u64 << (bits - 1)) - 1,
        (super::super::CellSign::Signed, false) => 1_u64 << (bits - 1),
    };
    magnitude / delta.unsigned_abs()
}

impl EmitterEngine {
    pub(super) fn prepare_powers(&mut self, id: NodeId, body: NodeId, body_cost: usize) {
        let Some(Some(LoopAnalysis::ExactMemoPlusDirectKernel { exact, .. })) =
            self.loop_analyses.get(&id)
        else {
            return;
        };
        let exact = *exact;
        let transfer = &self.transfers[&body];
        if transfer.memo_window().is_none() {
            return;
        }
        let Some(guard_delta) = transfer
            .effects
            .get(&0)
            .and_then(|p| p.additive_delta_for(0))
        else {
            return;
        };
        if guard_delta == 0 || guard_delta == i64::MIN || self.opts.cell_bits == 0 {
            return;
        }
        // The emitted guard proves a monotone drain before using any of these
        // powers. Unsupported entry values still execute the original loop.
        let maximum_iterations = maximum_proven_iterations(self.opts, guard_delta);
        if maximum_iterations < 2 {
            return;
        }
        let guard_only = transfer.effects.len() == 1 && transfer.effects.contains_key(&0);
        let max_power = if guard_only {
            0
        } else {
            usize::try_from(maximum_iterations.ilog2())
                .or_invariant("cell exponent fits usize")
                .min(usize::from(SUPER_LOOP_POWER_MAX))
        };
        if !crate::bf::polynomial_emit::can_emit_transfer(transfer)
            || !self.compile_work.admit_evaluation(transfer)
        {
            return;
        }
        let first_cost = transfer_cost(transfer);
        let per_power_budget = (body_cost * 4).max(8);
        if first_cost > per_power_budget || first_cost + 2 > POWER_EMISSION_BUDGET {
            return;
        }
        if let Some(powers) = self.power_cache.get(transfer)
            && powers.iter().all(|power| {
                self.compile_work.admit_evaluation(power)
                    && transfer_cost(power) <= per_power_budget
            })
        {
            // Reuse shares storage, but still consumes this compilation's
            // symbolic admission allowance; it cannot bypass stricter budgets.
            if !powers
                .iter()
                .all(|power| self.compile_work.budget.admit(power))
            {
                return;
            }
            self.compile_work.power_cache_hits += 1;
            self.loop_analyses.insert(
                id,
                Some(LoopAnalysis::ExactMemoPlusSymbolicPower {
                    exact,
                    powered: PoweredLoopAnalysis {
                        body,
                        guard_offset: 0,
                        guard_delta,
                        powers: Arc::clone(powers),
                    },
                }),
            );
            return;
        }
        self.compile_work.power_attempts += 1;
        if !self.compile_work.budget.admit(transfer) {
            return;
        }
        let mut powers = Vec::with_capacity(max_power + 1);
        powers.push(transfer.clone());
        self.compile_work.powers_built += 1;
        let mut emitted_cost = first_cost + 2;
        // A real aggregate code-size bound, independent of the cell exponent.
        for exponent in 1..=max_power {
            let previous = &powers[exponent - 1];
            let power = super::analysis::compose_transfer_owned(
                previous.clone(),
                previous,
                self.polynomial_semantics,
                &mut self.compile_work,
            );
            if power.unknown {
                break;
            }
            if !self.compile_work.admit_evaluation(&power) {
                break;
            }
            let cost = transfer_cost(&power);
            if cost > per_power_budget || emitted_cost + cost + 2 > POWER_EMISSION_BUDGET {
                break;
            }
            // A fixed power is reusable by the largest-power loop in the emitter.
            let repeated = power == *previous;
            powers.push(power);
            emitted_cost += cost + 2;
            self.compile_work.powers_built += 1;
            if repeated {
                break;
            }
        }
        if powers.len() < 2 && !guard_only {
            return;
        }
        let powers: Arc<[SymbolicTransfer]> = powers.into();
        // Keys compare normalized content, never insertion-order identities.
        // Both the number of keys and retained terms are bounded.
        let terms = |transfer: &SymbolicTransfer| {
            transfer
                .effects
                .values()
                .map(|p| p.terms.len())
                .sum::<usize>()
        };
        let retained_terms = self
            .power_cache
            .iter()
            .map(|(key, powers)| terms(key) + powers.iter().map(terms).sum::<usize>())
            .sum::<usize>();
        if self.power_cache.len() < 16
            && retained_terms + terms(transfer) + powers.iter().map(terms).sum::<usize>() <= 4096
            && self.compile_work.budget.admit(transfer)
        {
            self.power_cache
                .insert(transfer.clone(), Arc::clone(&powers));
        }
        self.loop_analyses.insert(
            id,
            Some(LoopAnalysis::ExactMemoPlusSymbolicPower {
                exact,
                powered: PoweredLoopAnalysis {
                    body,
                    guard_offset: 0,
                    guard_delta,
                    powers,
                },
            }),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine(bits: u32) -> EmitterEngine {
        EmitterEngine::with_opts(crate::bf::CodegenOpts {
            io_mode: crate::bf::IoMode::Number,
            cell_bits: bits,
            input_bits: None,
            output_bits: None,
            cell_sign: crate::bf::CellSign::Unsigned,
        })
    }

    #[test]
    fn eligibility_analysis_does_not_build_unused_powers() {
        let mut engine = engine(8);
        let root = engine.build_program(&[BfIr::Loop(vec![BfIr::Add(-2)])]);
        assert!(matches!(
            engine.loop_analysis(root),
            Some(LoopAnalysis::ExactMemoPlusDirectKernel { .. })
        ));
        assert_eq!(
            engine.compile_work.powers_built, 0,
            "eligibility must not expand powers"
        );
        assert_eq!(engine.compile_work.power_attempts, 0);
    }

    #[test]
    fn signed_power_range_matches_guard_direction_and_excludes_unreachable_exponents() {
        let mut engine = engine(8);
        engine.opts.cell_sign = crate::bf::CellSign::Signed;
        engine.polynomial_semantics =
            PolynomialSemantics::new(8, engine.opts.cell_sign, SUPER_C_TAPE_LEN);
        assert_eq!(maximum_proven_iterations(engine.opts, -1), 127);
        assert_eq!(maximum_proven_iterations(engine.opts, 1), 128);
        let root = engine.build_program(&[BfIr::Loop(vec![
            BfIr::Add(-1),
            BfIr::MovePtr(1),
            BfIr::Add(1),
            BfIr::MovePtr(-1),
        ])]);
        assert!(
            matches!(
                engine.plan_node(root),
                ExecPlan::ExactPoweredLoopMemo { max_power: 6, .. }
            ),
            "signed descending guards never require a 128-iteration power"
        );
        let Some(Some(LoopAnalysis::ExactMemoPlusSymbolicPower { powered, .. })) =
            engine.loop_analyses.get(&root)
        else {
            crate::invariant_failure!("fixture must retain its proved power analysis");
        };
        assert!(
            powered
                .powers
                .iter()
                .map(|power| transfer_cost(power) + 2)
                .sum::<usize>()
                <= POWER_EMISSION_BUDGET,
            "every emitted entry, including the final branch, counts toward the size cap"
        );
        engine.opts.cell_sign = crate::bf::CellSign::Unsigned;
        assert_eq!(
            maximum_proven_iterations(engine.opts, 1),
            0,
            "unsigned positive guards have no emitted monotone proof"
        );
    }

    #[test]
    fn wide_guard_only_loop_uses_one_proven_clear_instead_of_a_power_table() {
        let mut engine = engine(63);
        let root = engine.build_program(&[BfIr::Loop(vec![BfIr::Add(-2)])]);
        assert!(matches!(
            engine.plan_node(root),
            ExecPlan::ExactPoweredLoopMemo { max_power: 0, .. }
        ));
        assert_eq!(engine.compile_work.powers_built, 1);
        assert_eq!(engine.compile_work.compositions, 0);
        let output = emit_c_super(
            &[BfIr::Add(2), BfIr::Loop(vec![BfIr::Add(-2)]), BfIr::Output],
            engine.opts,
        );
        assert!(
            !output.contains("remaining_iters >= UINT64_C("),
            "a guard-only drain must not emit a branch for every exponent"
        );
        assert!(
            output.contains("powered_ok"),
            "compact clear still requires its exact divisibility/zero proof"
        );
    }

    #[test]
    fn eight_bit_power_planning_builds_only_reachable_exponents_once() {
        let mut engine = engine(8);
        let root = engine.build_program(&[BfIr::Loop(vec![
            BfIr::Add(-2),
            BfIr::MovePtr(1),
            BfIr::Add(1),
            BfIr::MovePtr(-1),
        ])]);
        assert!(matches!(
            engine.plan_node(root),
            ExecPlan::ExactPoweredLoopMemo { max_power: 6, .. }
        ));
        assert_eq!(
            engine.compile_work.powers_built, 7,
            "8-bit monotone -2 drain has at most 127 iterations, not 63 powers"
        );
        let compositions = engine.compile_work.compositions;
        let attempts = engine.compile_work.power_attempts;
        engine.plan_node(root);
        assert_eq!(
            engine.compile_work.compositions, compositions,
            "cached plan must not recompose powers"
        );
        assert_eq!(engine.compile_work.power_attempts, attempts);
    }

    #[test]
    fn zero_guard_delta_and_io_never_enter_power_construction() {
        for body in [
            vec![BfIr::Add(-1), BfIr::Add(1)],
            vec![BfIr::Add(-1), BfIr::Output],
        ] {
            let mut engine = engine(8);
            let root = engine.build_program(&[BfIr::Loop(body)]);
            assert!(!matches!(
                engine.plan_node(root),
                ExecPlan::ExactPoweredLoopMemo { .. }
            ));
            assert_eq!(
                engine.compile_work.power_attempts, 0,
                "unsupported guard or I/O cannot trigger speculative powers"
            );
        }
    }

    #[test]
    fn structurally_different_bodies_share_exact_normalized_powers() {
        let mut engine = engine(8);
        let first = engine.build_program(&[BfIr::Loop(vec![BfIr::Add(-2)])]);
        let second = engine.build_program(&[BfIr::Loop(vec![BfIr::Add(-1), BfIr::Add(-1)])]);
        assert_ne!(first, second, "fixture needs distinct structural DAG nodes");
        assert!(matches!(
            engine.plan_node(first),
            ExecPlan::ExactPoweredLoopMemo { .. }
        ));
        let built = engine.compile_work.powers_built;
        assert!(matches!(
            engine.plan_node(second),
            ExecPlan::ExactPoweredLoopMemo { .. }
        ));
        assert_eq!(engine.compile_work.power_cache_hits, 1);
        assert_eq!(
            engine.compile_work.powers_built, built,
            "equal symbolic content must reuse powers without rebuilding"
        );
    }

    #[test]
    fn composition_rebases_wrapped_sources_before_substitution() {
        let mut engine = engine(8);
        let root = engine.build_program(&[
            BfIr::Add(5),
            BfIr::MovePtr(30_000),
            BfIr::Affine {
                src: 0,
                dst: 1,
                coeff: 3,
                preserve_src: true,
                set_dst: true,
            },
        ]);
        let transfer = engine.transfer(root);
        assert!(!transfer.unknown);
        assert_eq!(transfer.ptr_delta, 0);
        assert_eq!(transfer.effects[&1].constant, 15);
        assert_eq!(
            transfer.effects[&1].terms.get(&SymbolicMonomial::Linear(0)),
            Some(&3)
        );
        assert!(
            !transfer.reads.contains(&30_000),
            "wrapped source must share the entry cell snapshot"
        );
    }

    #[test]
    fn compile_time_polynomial_composition_executes_bounded_coefficient_batches() {
        let mut engine = engine(63);
        let mut body = vec![BfIr::ClearAt { offset: 5 }];
        for src in 1..=4 {
            body.push(BfIr::Affine {
                src,
                dst: 5,
                coeff: 3,
                preserve_src: true,
                set_dst: false,
            });
        }
        body.push(BfIr::Square {
            src: 5,
            dst: 6,
            preserve_src: true,
            set_dst: true,
        });
        let root = engine.build_program(&body);
        assert_eq!(engine.transfer(root).effects[&6].degree(), 2);
        let accounting = engine.compile_work.coefficient_kernels;
        assert!(
            accounting.native_avx2_lanes + accounting.native_neon_lanes + accounting.scalar_lanes
                > 0,
            "composition must run coefficient arithmetic, not just admit candidate lanes: {accounting:?}"
        );
        assert_eq!(engine.compile_work.peak_scratch_lanes, 64);
        #[cfg(target_arch = "aarch64")]
        if !cfg!(miri) {
            assert!(
                accounting.native_neon_lanes >= 16,
                "full coefficient wave must execute native multiplication: {accounting:?}"
            );
        }
        #[cfg(target_arch = "x86_64")]
        if !cfg!(miri) && std::arch::is_x86_feature_detected!("avx2") {
            assert!(
                accounting.native_avx2_lanes >= 16,
                "full coefficient wave must execute native multiplication: {accounting:?}"
            );
        }
    }
}
