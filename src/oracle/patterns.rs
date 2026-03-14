use super::*;

pub(super) fn build_emitter_macro_model(
    seed: Option<&BitGrid>,
    generation: u64,
) -> Option<EmitterMacroModel> {
    if generation != 0 {
        return None;
    }
    let seed = seed?;
    let gosper = pattern_by_name("gosper_glider_gun")?;
    if normalize(seed).0 != normalize(&gosper).0 {
        return None;
    }

    const PERIOD: u64 = 30;
    const BASELINE_GENERATION: u64 = 300;
    let baseline_grid = advance_exact_steps(seed, BASELINE_GENERATION);
    let next_grid = advance_exact_steps(seed, BASELINE_GENERATION + PERIOD);
    let baseline_state = extract_gosper_state(&baseline_grid)?;
    let next_state = extract_gosper_state(&next_grid)?;
    if normalize(&baseline_state.core).0 != normalize(&next_state.core).0 {
        return None;
    }
    if next_state.gliders.len() != baseline_state.gliders.len() + 1 {
        return None;
    }

    let mut core_population_by_phase = [0_usize; 30];
    let mut core_bounds_by_phase = [(0_i64, 0_i64, 0_i64, 0_i64); 30];
    let mut core_phase = baseline_state.core.clone();
    for residual in 0..30 {
        core_population_by_phase[residual] = core_phase.population();
        core_bounds_by_phase[residual] = core_phase.bounds().unwrap_or((0, 0, 0, 0));
        core_phase = step_grid_with_changes_and_memo(&core_phase, &mut Memo::default()).0;
    }

    let oldest_glider = baseline_state
        .gliders
        .iter()
        .max_by_key(|glider| (glider.origin.0, glider.origin.1))
        .copied()?;

    Some(EmitterMacroModel {
        baseline_generation: BASELINE_GENERATION,
        baseline_glider_count: u64::try_from(baseline_state.gliders.len())
            .expect("gosper glider count exceeded u64"),
        core_population_by_phase,
        core_bounds_by_phase,
        oldest_glider_origin: oldest_glider.origin,
        oldest_glider_phase: oldest_glider.phase,
    })
}

pub(super) fn emitter_runtime_population(
    model: &EmitterMacroModel,
    target_generation: u64,
) -> usize {
    let delta = target_generation.saturating_sub(model.baseline_generation);
    let emitted = delta / 30;
    let residual = usize::try_from(delta % 30).expect("gosper residual exceeded usize");
    model.core_population_by_phase[residual]
        + usize::try_from((model.baseline_glider_count + emitted) * 5)
            .expect("gosper runtime population exceeded usize")
}

pub(super) fn emitter_runtime_bounds_span(
    model: &EmitterMacroModel,
    target_generation: u64,
) -> Coord {
    let delta = target_generation.saturating_sub(model.baseline_generation);
    let residual = delta % 120;
    let cycles = delta / 120;
    let phase = usize::from(model.oldest_glider_phase);
    let table = glider_runtime_table();
    let glider_bounds =
        table[phase][usize::try_from(residual).expect("glider residual exceeded usize")];
    let cycle_shift = Coord::try_from(cycles)
        .expect("gosper cycle count exceeded Coord")
        .checked_mul(30)
        .expect("gosper cycle shift overflow");
    let glider_max_x = model
        .oldest_glider_origin
        .0
        .checked_add(cycle_shift)
        .and_then(|origin_x| origin_x.checked_add(glider_bounds.2))
        .expect("gosper glider max x overflow");
    let glider_max_y = model
        .oldest_glider_origin
        .1
        .checked_add(cycle_shift)
        .and_then(|origin_y| origin_y.checked_add(glider_bounds.3))
        .expect("gosper glider max y overflow");
    let residual_index = usize::try_from(delta % 30).expect("gosper residual exceeded usize");
    let core_bounds = model.core_bounds_by_phase[residual_index];
    let width = glider_max_x
        .max(core_bounds.2)
        .checked_sub(core_bounds.0)
        .and_then(|span| span.checked_add(1))
        .expect("gosper runtime width overflow");
    let height = glider_max_y
        .max(core_bounds.3)
        .checked_sub(core_bounds.1)
        .and_then(|span| span.checked_add(1))
        .expect("gosper runtime height overflow");
    width.max(height)
}

#[derive(Clone, Copy, Debug)]
struct GosperGliderInstance {
    origin: Cell,
    phase: u8,
}

#[derive(Clone, Debug)]
struct GosperExactState {
    core: BitGrid,
    gliders: Vec<GosperGliderInstance>,
}

fn extract_gosper_state(grid: &BitGrid) -> Option<GosperExactState> {
    let mut core_cells = crop_grid_region(grid, 0, 0, 36, 9).live_cells();
    let field = exclude_rect(grid, 0, 0, 36, 9);
    let variants = canonical_glider_variants();
    let mut gliders = Vec::new();
    for component in connected_components(&field) {
        let component_grid = BitGrid::from_cells(&component);
        let (signature, origin) = normalize(&component_grid);
        let Some(phase) = variants
            .iter()
            .position(|variant| *variant == signature)
            .and_then(|index| u8::try_from(index).ok())
        else {
            core_cells.extend(component);
            continue;
        };
        gliders.push(GosperGliderInstance { origin, phase });
    }
    if gliders.is_empty() {
        return None;
    }
    Some(GosperExactState {
        core: BitGrid::from_cells(&core_cells),
        gliders,
    })
}

fn advance_exact_steps(seed: &BitGrid, generations: u64) -> BitGrid {
    let mut grid = seed.clone();
    let mut memo = Memo::default();
    for _ in 0..generations {
        grid = step_grid_with_changes_and_memo(&grid, &mut memo).0;
    }
    grid
}

fn crop_grid_region(
    grid: &BitGrid,
    min_x: Coord,
    min_y: Coord,
    width: Coord,
    height: Coord,
) -> BitGrid {
    let max_x = min_x + width - 1;
    let max_y = min_y + height - 1;
    let cells = grid
        .live_cells()
        .into_iter()
        .filter(|(x, y)| *x >= min_x && *x <= max_x && *y >= min_y && *y <= max_y)
        .collect::<Vec<_>>();
    BitGrid::from_cells(&cells)
}

fn exclude_rect(
    grid: &BitGrid,
    min_x: Coord,
    min_y: Coord,
    width: Coord,
    height: Coord,
) -> BitGrid {
    let max_x = min_x + width - 1;
    let max_y = min_y + height - 1;
    let cells = grid
        .live_cells()
        .into_iter()
        .filter(|(x, y)| !(*x >= min_x && *x <= max_x && *y >= min_y && *y <= max_y))
        .collect::<Vec<_>>();
    BitGrid::from_cells(&cells)
}

fn connected_components(grid: &BitGrid) -> Vec<Vec<Cell>> {
    let live = grid.live_cells();
    let mut remaining = live.iter().copied().collect::<std::collections::HashSet<_>>();
    let mut components = Vec::new();
    while let Some(&start) = remaining.iter().next() {
        let mut queue = std::collections::VecDeque::from([start]);
        let mut component = Vec::new();
        remaining.remove(&start);
        while let Some((x, y)) = queue.pop_front() {
            component.push((x, y));
            for ny in (y - 1)..=(y + 1) {
                for nx in (x - 1)..=(x + 1) {
                    if nx == x && ny == y {
                        continue;
                    }
                    if remaining.remove(&(nx, ny)) {
                        queue.push_back((nx, ny));
                    }
                }
            }
        }
        components.push(component);
    }
    components
}

fn canonical_glider_variants() -> &'static [NormalizedGridSignature; 4] {
    static VARIANTS: OnceLock<[NormalizedGridSignature; 4]> = OnceLock::new();
    VARIANTS.get_or_init(|| {
        let mut variants = Vec::with_capacity(4);
        let mut grid = pattern_by_name("glider").expect("glider pattern should exist");
        let mut memo = Memo::default();
        for _ in 0..4 {
            variants.push(normalize(&grid).0);
            grid = step_grid_with_changes_and_memo(&grid, &mut memo).0;
        }
        [
            variants[0].clone(),
            variants[1].clone(),
            variants[2].clone(),
            variants[3].clone(),
        ]
    })
}

type GliderBoundsTable = [[(Coord, Coord, Coord, Coord); 120]; 4];

fn glider_runtime_table() -> &'static GliderBoundsTable {
    static TABLE: OnceLock<GliderBoundsTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [[(0, 0, 0, 0); 120]; 4];
        let mut phase_grid = pattern_by_name("glider").expect("glider pattern should exist");
        let variants = canonical_glider_variants();
        for phase in 0..4 {
            let phase_signature = &variants[phase];
            let mut memo = Memo::default();
            for residual in 0..120 {
                if residual == 0 {
                    let bounds = phase_grid.bounds().expect("glider should be non-empty");
                    table[phase][residual] = bounds;
                } else {
                    phase_grid = step_grid_with_changes_and_memo(&phase_grid, &mut memo).0;
                    let bounds = phase_grid.bounds().expect("glider should be non-empty");
                    table[phase][residual] = bounds;
                }
            }
            phase_grid = BitGrid::from_cells(&phase_signature.cells);
        }
        table
    })
}
