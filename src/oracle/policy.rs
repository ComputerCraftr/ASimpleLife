use super::*;

pub(super) fn confirmation_full_grid_policy() -> GridExtractionPolicy {
    GridExtractionPolicy::FullGridIfUnder {
        max_population: HASHLIFE_FULL_GRID_MAX_POPULATION,
        max_chunks: HASHLIFE_FULL_GRID_MAX_CHUNKS,
        max_bounds_span: Coord::MAX,
    }
}

pub(super) fn continuation_step_span(
    current: OracleStateMetrics,
    generation: u64,
    generation_limit: u64,
    nominal_generation_limit: u64,
    simulation: &mut SimulationSession,
    hashlife_active: bool,
) -> u64 {
    if generation < nominal_generation_limit {
        return 1;
    }

    let remaining = generation_limit.saturating_sub(generation);
    if remaining <= 1 {
        return 1;
    }

    if hashlife_active && current.population > 0 {
        let safe_hashlife_jump = max_hashlife_safe_jump_from_span(current.bounds_span);
        if safe_hashlife_jump >= 2 {
            return largest_power_of_two_leq(remaining.min(safe_hashlife_jump));
        }
    }

    let exact_tail = oracle_exact_tail_window(current.population, current.bounds_span);
    if remaining <= exact_tail {
        return 1;
    }
    let jump_budget = remaining - exact_tail;
    let safe_hashlife_jump = max_hashlife_safe_jump_from_span(current.bounds_span);

    if jump_budget >= ORACLE_HASHLIFE_MIN_JUMP_BUDGET && safe_hashlife_jump >= 2 {
        return largest_power_of_two_leq(jump_budget.min(safe_hashlife_jump));
    }

    match simulation.planned_backend_from_session_metrics(
        current.population,
        current.bounds_span,
        remaining,
    ) {
        SimulationBackend::SimdChunk => 1,
        SimulationBackend::HybridSegmented => jump_budget.min(ORACLE_HYBRID_SEGMENT_MAX_STEP),
        SimulationBackend::HashLife => {
            if safe_hashlife_jump >= 2 {
                largest_power_of_two_leq(jump_budget.min(safe_hashlife_jump))
            } else {
                1
            }
        }
    }
}

pub(super) fn oracle_exact_tail_window(population: usize, span: Coord) -> u64 {
    if population <= ORACLE_SMALL_EXACT_POPULATION && span <= ORACLE_SMALL_EXACT_SPAN {
        ORACLE_SMALL_EXACT_WINDOW
    } else if population <= ORACLE_MEDIUM_EXACT_POPULATION && span <= ORACLE_MEDIUM_EXACT_SPAN {
        ORACLE_MEDIUM_EXACT_WINDOW
    } else {
        ORACLE_HYBRID_SEGMENT_MAX_STEP
    }
}

pub(super) fn target_exact_suffix_window(population: usize, span: Coord) -> u64 {
    if population > ORACLE_TARGET_SUFFIX_MAX_POPULATION || span > ORACLE_TARGET_SUFFIX_MAX_SPAN {
        0
    } else if population <= ORACLE_SMALL_EXACT_POPULATION && span <= ORACLE_SMALL_EXACT_SPAN {
        ORACLE_SMALL_EXACT_WINDOW
    } else if population <= ORACLE_MEDIUM_EXACT_POPULATION && span <= ORACLE_MEDIUM_EXACT_SPAN {
        ORACLE_MEDIUM_EXACT_WINDOW
    } else if population <= ORACLE_LARGE_EXACT_POPULATION && span <= ORACLE_LARGE_EXACT_SPAN {
        ORACLE_LARGE_EXACT_WINDOW
    } else {
        ORACLE_MIN_EXACT_WINDOW
    }
}

pub(super) fn cycle_probe_prefix_window(population: usize, span: Coord) -> u64 {
    if population <= 64 && span <= 32 {
        64
    } else if population <= 256 && span <= 64 {
        16
    } else {
        0
    }
}

pub(super) fn largest_power_of_two_leq(value: u64) -> u64 {
    debug_assert!(value > 0);
    1_u64 << (63 - value.leading_zeros())
}

pub(super) fn max_hashlife_safe_jump_from_span(span: Coord) -> u64 {
    if span <= 0 {
        return 1;
    }
    let raw_max_jump = (((Coord::MAX as i128) - (2 * span as i128) - 8) / 4).max(1) as u64;
    let mut jump = 1_u64 << (63 - raw_max_jump.leading_zeros());
    while jump > 1 && required_root_size_for_jump(span as u64, jump) > Coord::MAX as u64 {
        jump >>= 1;
    }
    jump
}

pub(super) fn hybrid_target_prefix_generations(population: usize, generations: u64) -> u64 {
    let prefix = (population as u64)
        .saturating_div(32)
        .clamp(16, 64)
        .min(generations);
    prefix.max(1).min(generations)
}

pub(super) fn required_root_size_for_jump(span: u64, jump: u64) -> u64 {
    (2 * span + 4 * (jump + 2))
        .max((4 * jump) + 4)
        .max(4)
        .next_power_of_two()
}
