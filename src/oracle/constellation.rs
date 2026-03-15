use super::*;
#[cfg(test)]
use crate::RequiredExt;
use std::collections::{HashSet, VecDeque};

const COMPONENT_PERIOD_LIMIT: u64 = 64;
const CONSTELLATION_LCM_LIMIT: u64 = 256;

struct PeriodicComponent {
    phases: Vec<Vec<Cell>>,
    period: u64,
    delta: Cell,
}

pub(super) fn project_periodic_constellation(grid: &BitGrid, remaining: u64) -> Option<BitGrid> {
    if remaining == 0 {
        return Some(grid.clone());
    }
    let components = connected_components(grid);
    if components.len() < 2 {
        return None;
    }
    let models = components
        .iter()
        .map(|cells| solve_periodic_component(cells))
        .collect::<Option<Vec<_>>>()?;
    if !components_stay_independent(&models, remaining) {
        return None;
    }

    let mut projected = Vec::with_capacity(grid.population());
    for model in &models {
        append_projected_component(model, remaining, &mut projected)?;
    }
    Some(BitGrid::from_cells(&projected))
}

fn solve_periodic_component(cells: &[Cell]) -> Option<PeriodicComponent> {
    let initial = BitGrid::from_cells(cells);
    let (initial_signature, initial_origin) = normalize(&initial);
    let mut phases = Vec::new();
    let mut current = initial;
    let mut memo = Memo::default();

    for period in 1..=COMPONENT_PERIOD_LIMIT {
        phases.push(current.live_cells());
        current = step_grid_with_changes_and_memo(&current, &mut memo).0;
        if current.is_empty() {
            return None;
        }
        let (signature, origin) = normalize(&current);
        if signature == initial_signature {
            return Some(PeriodicComponent {
                phases,
                period,
                delta: (origin.0 - initial_origin.0, origin.1 - initial_origin.1),
            });
        }
    }
    None
}

fn components_stay_independent(models: &[PeriodicComponent], remaining: u64) -> bool {
    let Some(super_period) = models.iter().try_fold(1_u64, |period, model| {
        let combined = lcm(period, model.period)?;
        (combined <= CONSTELLATION_LCM_LIMIT).then_some(combined)
    }) else {
        return false;
    };
    if !equal_velocity_groups_are_independent(models, super_period) {
        return false;
    }

    for left_index in 0..models.len() {
        for right_index in left_index + 1..models.len() {
            if !pair_stays_independent(
                &models[left_index],
                &models[right_index],
                super_period,
                remaining,
            ) {
                return false;
            }
        }
    }
    true
}

fn pair_stays_independent(
    left: &PeriodicComponent,
    right: &PeriodicComponent,
    super_period: u64,
    remaining: u64,
) -> bool {
    let Some(left_velocity) = scaled_delta(left.delta, super_period / left.period) else {
        return false;
    };
    let Some(right_velocity) = scaled_delta(right.delta, super_period / right.period) else {
        return false;
    };
    if left_velocity == right_velocity {
        return true;
    }
    let phase_limit = super_period.min(remaining.saturating_add(1));

    for phase in 0..phase_limit {
        let Some(left_bounds) = component_bounds_at(left, phase) else {
            return false;
        };
        let Some(right_bounds) = component_bounds_at(right, phase) else {
            return false;
        };
        let macro_steps = (remaining - phase) / super_period;
        let x_separated = axis_stays_separated(
            (left_bounds.0, left_bounds.2),
            (right_bounds.0, right_bounds.2),
            left_velocity.0,
            right_velocity.0,
            macro_steps,
        );
        let y_separated = axis_stays_separated(
            (left_bounds.1, left_bounds.3),
            (right_bounds.1, right_bounds.3),
            left_velocity.1,
            right_velocity.1,
            macro_steps,
        );
        if !x_separated && !y_separated {
            return false;
        }
    }
    true
}

fn equal_velocity_groups_are_independent(models: &[PeriodicComponent], super_period: u64) -> bool {
    let mut groups: Vec<(Cell, Vec<usize>)> = Vec::new();
    for (index, model) in models.iter().enumerate() {
        let Some(velocity) = scaled_delta(model.delta, super_period / model.period) else {
            return false;
        };
        if let Some((_, indices)) = groups.iter_mut().find(|(key, _)| *key == velocity) {
            indices.push(index);
        } else {
            groups.push((velocity, vec![index]));
        }
    }

    groups.into_iter().all(|(_, indices)| {
        indices.len() == 1 || periodic_group_is_independent(models, &indices, super_period)
    })
}

fn periodic_group_is_independent(
    models: &[PeriodicComponent],
    indices: &[usize],
    super_period: u64,
) -> bool {
    let mut memo = Memo::default();
    for generation in 0..super_period {
        let mut current_cells = Vec::new();
        let mut expected_cells = Vec::new();
        for &index in indices {
            if append_projected_component(&models[index], generation, &mut current_cells).is_none()
                || append_projected_component(&models[index], generation + 1, &mut expected_cells)
                    .is_none()
            {
                return false;
            }
        }
        let actual =
            step_grid_with_changes_and_memo(&BitGrid::from_cells(&current_cells), &mut memo).0;
        if actual != BitGrid::from_cells(&expected_cells) {
            return false;
        }
    }
    true
}

fn component_bounds_at(
    model: &PeriodicComponent,
    generation: u64,
) -> Option<(Coord, Coord, Coord, Coord)> {
    let phase = usize::try_from(generation % model.period).ok()?;
    let cycles = generation / model.period;
    let shift = scaled_delta(model.delta, cycles)?;
    let mut bounds = component_bounds(&model.phases[phase])?;
    bounds.0 = bounds.0.checked_add(shift.0)?;
    bounds.1 = bounds.1.checked_add(shift.1)?;
    bounds.2 = bounds.2.checked_add(shift.0)?;
    bounds.3 = bounds.3.checked_add(shift.1)?;
    Some(bounds)
}

fn axis_stays_separated(
    left: (Coord, Coord),
    right: (Coord, Coord),
    left_velocity: Coord,
    right_velocity: Coord,
    macro_steps: u64,
) -> bool {
    let separated = |step: u64, left_before_right: bool| {
        let left_shift = i128::from(left_velocity) * i128::from(step);
        let right_shift = i128::from(right_velocity) * i128::from(step);
        if left_before_right {
            i128::from(left.1) + left_shift + 2 < i128::from(right.0) + right_shift
        } else {
            i128::from(right.1) + right_shift + 2 < i128::from(left.0) + left_shift
        }
    };
    (separated(0, true) && separated(macro_steps, true))
        || (separated(0, false) && separated(macro_steps, false))
}

fn append_projected_component(
    model: &PeriodicComponent,
    generation: u64,
    output: &mut Vec<Cell>,
) -> Option<()> {
    let phase = usize::try_from(generation % model.period).ok()?;
    let shift = scaled_delta(model.delta, generation / model.period)?;
    for &(x, y) in &model.phases[phase] {
        output.push((x.checked_add(shift.0)?, y.checked_add(shift.1)?));
    }
    Some(())
}

fn scaled_delta(delta: Cell, scale: u64) -> Option<Cell> {
    let scale = Coord::try_from(scale).ok()?;
    Some((delta.0.checked_mul(scale)?, delta.1.checked_mul(scale)?))
}

fn lcm(left: u64, right: u64) -> Option<u64> {
    left.checked_div(gcd(left, right))?.checked_mul(right)
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn connected_components(grid: &BitGrid) -> Vec<Vec<Cell>> {
    let mut remaining: HashSet<Cell> = grid.live_cells().into_iter().collect();
    let mut components = Vec::new();
    while let Some(&start) = remaining.iter().next() {
        remaining.remove(&start);
        let mut queue = VecDeque::from([start]);
        let mut component = Vec::new();
        while let Some((x, y)) = queue.pop_front() {
            component.push((x, y));
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let Some(neighbor) = x.checked_add(dx).zip(y.checked_add(dy)) else {
                        continue;
                    };
                    if remaining.remove(&neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        components.push(component);
    }
    components
}

fn component_bounds(cells: &[Cell]) -> Option<(Coord, Coord, Coord, Coord)> {
    let &(first_x, first_y) = cells.first()?;
    Some(cells.iter().skip(1).fold(
        (first_x, first_y, first_x, first_y),
        |(min_x, min_y, max_x, max_y), &(x, y)| {
            (min_x.min(x), min_y.min(y), max_x.max(x), max_y.max(y))
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators::{pattern_by_name, random_soup};

    #[test]
    fn periodic_constellation_projection_matches_exact_evolution() {
        let mut cells = pattern_by_name("block")
            .or_invariant("required value")
            .live_cells();
        cells.extend(
            pattern_by_name("glider")
                .or_invariant("required value")
                .translated(100, 100)
                .live_cells(),
        );
        let grid = BitGrid::from_cells(&cells);
        let projected = project_periodic_constellation(&grid, 256)
            .or_invariant("separating periodic components should be projectable");
        let mut expected = grid;
        let mut memo = Memo::default();
        for _ in 0..256 {
            expected = step_grid_with_changes_and_memo(&expected, &mut memo).0;
        }

        assert_eq!(normalize(&projected).0, normalize(&expected).0);
        assert_eq!(normalize(&projected).1, normalize(&expected).1);
    }

    #[test]
    fn periodic_constellation_rejects_components_with_overlapping_frontiers() {
        let grid = BitGrid::from_cells(&[(0, 0), (1, 0), (2, 0), (0, 2), (1, 2), (2, 2)]);

        assert!(
            project_periodic_constellation(&grid, 256).is_none(),
            "nearby components must fall back when their Life frontiers can interact"
        );
    }

    #[test]
    fn seeded_runtime_soup_reaches_a_projectable_periodic_constellation() {
        let grid = random_soup(53, 24, 37, 420);
        let mut simulation = SimulationSession::new();
        simulation
            .try_load_hashlife_state(&grid)
            .or_invariant("test HashLife state should load");
        let mut generation = 0_u64;
        let projected_at = [64, 128, 256, 512, 1_024, 2_048, 4_096]
            .into_iter()
            .find(|&probe| {
                let Ok(stats) = simulation.advance_hashlife_root(probe - generation) else {
                    return false;
                };
                generation = stats.reached_generation;
                simulation
                    .sample_hashlife_state_grid(GridExtractionPolicy::FullGridIfUnder {
                        max_population: u128::from(ORACLE_CONSTELLATION_MAX_POPULATION),
                        max_chunks: ORACLE_CONSTELLATION_MAX_CHUNKS,
                        max_bounds_span: ORACLE_CONSTELLATION_MAX_SPAN,
                    })
                    .ok()
                    .and_then(|sample| {
                        project_periodic_constellation(&sample, 1_000_000_000 - probe)
                    })
                    .is_some()
            });

        assert!(
            projected_at.is_some(),
            "seed 420 should settle into independently projectable periodic components by generation 4096"
        );
    }

    #[test]
    fn r_pentomino_ash_becomes_a_projectable_periodic_constellation() {
        let mut grid = pattern_by_name("r_pentomino").or_invariant("required value");
        let mut memo = Memo::default();
        for _ in 0..2_048 {
            grid = step_grid_with_changes_and_memo(&grid, &mut memo).0;
        }
        let components = connected_components(&grid);
        let solved = components
            .iter()
            .filter_map(|cells| solve_periodic_component(cells))
            .collect::<Vec<_>>();

        assert_eq!(
            solved.len(),
            components.len(),
            "every separated R-pentomino ash component should have a bounded periodic model; component_sizes={:?}",
            components.iter().map(Vec::len).collect::<Vec<_>>()
        );
        assert!(
            components_stay_independent(&solved, 100_000_000 - 2_048),
            "solved R-pentomino ash components should remain independent"
        );
        assert!(
            project_periodic_constellation(&grid, 100_000_000 - 2_048).is_some(),
            "R-pentomino ash should project to a large target without building a huge HashLife root"
        );
    }
}
