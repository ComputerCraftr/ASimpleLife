use super::*;

pub(super) fn benchmark_family_filter(
    options: &BenchmarkOptions,
) -> Option<HashSet<BenchmarkFamily>> {
    let tokens = options.families.as_ref()?;
    let mut families = HashSet::new();
    for token in tokens {
        if let Some(family) = parse_family(token) {
            families.insert(family);
        }
    }
    Some(families)
}

pub(super) fn benchmark_run_mode_and_seed(options: &BenchmarkOptions) -> (BenchmarkRunMode, u64) {
    match (options.randomized, options.seed) {
        (false, None) => (BenchmarkRunMode::DefaultSeeded, DEFAULT_BENCHMARK_RUN_SEED),
        (_, Some(seed)) => (BenchmarkRunMode::Seeded, seed),
        (true, None) => (BenchmarkRunMode::TimeSeeded, time_seed()),
    }
}

fn parse_family(token: &str) -> Option<BenchmarkFamily> {
    match token {
        "iid" => Some(BenchmarkFamily::IidRandom),
        "structured" => Some(BenchmarkFamily::StructuredRandom),
        "clustered" => Some(BenchmarkFamily::ClusteredNoise),
        "smallbox" => Some(BenchmarkFamily::ExhaustiveSmallBox),
        "smallbox_5x5" => Some(BenchmarkFamily::ExhaustiveFiveByFive),
        "methuselah" => Some(BenchmarkFamily::LongLivedMethuselah),
        "mover" => Some(BenchmarkFamily::TranslatedPeriodicMover),
        "gun" => Some(BenchmarkFamily::GunPufferBreeder),
        "delayed" => Some(BenchmarkFamily::DelayedInteraction),
        "ash" => Some(BenchmarkFamily::DeceptiveAsh),
        "gadget" => Some(BenchmarkFamily::ComputationalGadget),
        "emitter" => Some(BenchmarkFamily::EmitterInteraction),
        _ => None,
    }
}

pub(super) fn seeded_benchmark_suite(run_seed: u64, cases_per_family: usize) -> Vec<BenchmarkCase> {
    let mut suite = Vec::new();
    suite.extend(seeded_iid_cases(run_seed, cases_per_family));
    suite.extend(seeded_structured_cases(run_seed, cases_per_family));
    suite.extend(seeded_clustered_cases(run_seed, cases_per_family));
    suite.extend(seeded_methuselah_cases(run_seed, cases_per_family));
    suite.extend(seeded_mover_cases(run_seed, cases_per_family));
    suite.extend(seeded_gun_cases(run_seed, cases_per_family));
    suite.extend(seeded_delayed_cases(run_seed, cases_per_family));
    suite.extend(seeded_ash_cases(run_seed, cases_per_family));
    suite.extend(seeded_gadget_cases(run_seed, cases_per_family));
    suite.extend(seeded_emitter_cases(run_seed, cases_per_family));
    suite
}

pub(super) fn seeded_case(
    name: String,
    family: BenchmarkFamily,
    size: i32,
    density_percent: u32,
    grid: BitGrid,
    replay: Option<String>,
) -> BenchmarkCase {
    BenchmarkCase {
        name,
        family,
        size,
        density_percent,
        grid,
        replay,
    }
}

fn seeded_iid_cases(run_seed: u64, cases_per_family: usize) -> Vec<BenchmarkCase> { let mut cases=Vec::new(); for idx in 0..cases_per_family { let case_seed = mix_seed(run_seed ^ 0x1000_0000_0000_0000 ^ idx as u64); let size = pick_from(&[16_i64, 32, 64, 128, 256], case_seed, 0); let density = pick_from(&[5_u32, 10, 20, 30, 50], case_seed, 1); let grid_seed = mix_seed(case_seed ^ 0xA11D_D00D); let grid = random_soup(size, size, density, grid_seed); cases.push(seeded_case(format!("iid_s{size}_d{density}_seed{grid_seed}"), BenchmarkFamily::IidRandom, i32::try_from(size).expect("benchmark size exceeded i32"), density, grid, Some(format!("family=iid,size={size},density={density},seed={grid_seed}")))); } cases }
fn seeded_structured_cases(run_seed: u64, cases_per_family: usize) -> Vec<BenchmarkCase> { let mut cases=Vec::new(); for idx in 0..cases_per_family { let case_seed = mix_seed(run_seed ^ 0x2000_0000_0000_0000 ^ idx as u64); let size = pick_from(&[16_i64, 32, 64, 128, 192], case_seed, 0); let grid_seed = mix_seed(case_seed ^ 0x51DE_BAAD); let grid = structured_random_soup(size, size, grid_seed); cases.push(seeded_case(format!("structured_s{size}_seed{grid_seed}"), BenchmarkFamily::StructuredRandom, i32::try_from(size).expect("benchmark size exceeded i32"), estimate_density_percent(&grid), grid, Some(format!("family=structured,size={size},seed={grid_seed}")))); } cases }
fn seeded_clustered_cases(run_seed: u64, cases_per_family: usize) -> Vec<BenchmarkCase> { let mut cases=Vec::new(); for idx in 0..cases_per_family { let case_seed = mix_seed(run_seed ^ 0x3000_0000_0000_0000 ^ idx as u64); let size = pick_from(&[24_i64, 48, 96, 144], case_seed, 0); let density = pick_from(&[8_u32, 12, 18, 24, 30], case_seed, 1); let grid_seed = mix_seed(case_seed ^ 0xC1A5_7EED); let grid = clustered_noise_soup(size, size, density, grid_seed); cases.push(seeded_case(format!("clustered_s{size}_d{density}_seed{grid_seed}"), BenchmarkFamily::ClusteredNoise, i32::try_from(size).expect("benchmark size exceeded i32"), density, grid, Some(format!("family=clustered,size={size},density={density},seed={grid_seed}")))); } cases }
fn seeded_delayed_cases(run_seed: u64, cases_per_family: usize) -> Vec<BenchmarkCase> { let mut cases=Vec::new(); for idx in 0..cases_per_family { let case_seed = mix_seed(run_seed ^ 0x4000_0000_0000_0000 ^ idx as u64); let distance = pick_from(&[24_i64, 48, 96, 192, 384, 768], case_seed, 0); let variant = pick_from(&[0_u32, 1, 2], case_seed, 1); let grid = match variant { 0 => head_on_glider_collision(distance), 1 => distant_glider_trigger(distance, pattern_by_name("block").unwrap(), (0, 0)), _ => distant_glider_trigger(distance, pattern_by_name("blinker").unwrap(), (0, 0)), }; let variant_name = match variant { 0 => "head_on_gliders", 1 => "block_trigger", _ => "blinker_trigger", }; cases.push(seeded_case(format!("delayed_{variant_name}_d{distance}_seed{case_seed}"), BenchmarkFamily::DelayedInteraction, i32::try_from(grid.bounds().map(|(min_x, _, max_x, _)| max_x - min_x + 1).unwrap_or(0)).expect("benchmark size exceeded i32"), estimate_density_percent(&grid), grid, Some(format!("family=delayed,variant={variant_name},distance={distance},seed={case_seed}")))); } cases }
fn seeded_gadget_cases(run_seed: u64, cases_per_family: usize) -> Vec<BenchmarkCase> { let mut cases=Vec::new(); let glider = pattern_by_name("glider").unwrap(); let block = pattern_by_name("block").unwrap(); let blinker = pattern_by_name("blinker").unwrap(); for idx in 0..cases_per_family { let case_seed = mix_seed(run_seed ^ 0x5000_0000_0000_0000 ^ idx as u64); let dx = (case_seed % 64) as Coord; let dy = ((case_seed >> 6) % 48) as Coord; let block_dx = 20 + ((case_seed >> 12) % 32) as Coord; let block_dy = 8 + ((case_seed >> 17) % 32) as Coord; let blink_dx = 40 + ((case_seed >> 22) % 32) as Coord; let blink_dy = 16 + ((case_seed >> 27) % 32) as Coord; let mut cells = glider.live_cells().into_iter().map(|(x, y)| (x + dx, y + dy)).collect::<Vec<_>>(); cells.extend(block.live_cells().into_iter().map(|(x, y)| (x + block_dx, y + block_dy))); cells.extend(blinker.live_cells().into_iter().map(|(x, y)| (x + blink_dx, y + blink_dy))); let grid = BitGrid::from_cells(&cells); cases.push(seeded_case(format!("gadget_seed{case_seed}_g{dx}_{dy}_b{block_dx}_{block_dy}_l{blink_dx}_{blink_dy}"), BenchmarkFamily::ComputationalGadget, i32::try_from(grid.bounds().map(|(min_x, _, max_x, _)| max_x - min_x + 1).unwrap_or(0)).expect("benchmark size exceeded i32"), estimate_density_percent(&grid), grid, Some(format!("family=gadget,glider=({dx},{dy}),block=({block_dx},{block_dy}),blinker=({blink_dx},{blink_dy}),seed={case_seed}")))); } cases }
fn seeded_emitter_cases(run_seed: u64, cases_per_family: usize) -> Vec<BenchmarkCase> { let mut cases=Vec::new(); let gun = pattern_by_name("gosper_glider_gun").unwrap(); let blocker = pattern_by_name("block").unwrap(); for idx in 0..cases_per_family { let case_seed = mix_seed(run_seed ^ 0x6000_0000_0000_0000 ^ idx as u64); let block_dx = 90 + ((case_seed >> 8) % 160) as Coord; let block_dy = 10 + ((case_seed >> 20) % 80) as Coord; let mut cells = gun.live_cells(); cells.extend(blocker.live_cells().into_iter().map(|(x, y)| (x + block_dx, y + block_dy))); let grid = BitGrid::from_cells(&cells); cases.push(seeded_case(format!("emitter_seed{case_seed}_b{block_dx}_{block_dy}"), BenchmarkFamily::EmitterInteraction, i32::try_from(grid.bounds().map(|(min_x, _, max_x, _)| max_x - min_x + 1).unwrap_or(0)).expect("benchmark size exceeded i32"), estimate_density_percent(&grid), grid, Some(format!("family=emitter,blocker=({block_dx},{block_dy}),seed={case_seed}")))); } cases }

fn time_seed() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => mix_seed(duration.as_secs() ^ u64::from(duration.subsec_nanos())),
        Err(_) => mix_seed(0xD1CE_F00D_1234_5678),
    }
}

pub(super) fn exhaustive_small_box_cases() -> Vec<BenchmarkCase> { let mut cases=Vec::new(); for mask in 1_u32..(1_u32 << 12) { let grid = bitmask_pattern(mask, 4, 3); cases.push(seeded_case(format!("smallbox_{mask}"), BenchmarkFamily::ExhaustiveSmallBox, 4, estimate_density_percent(&grid), grid, None)); } cases }
fn seeded_methuselah_cases(run_seed: u64, cases_per_family: usize) -> Vec<BenchmarkCase> { seeded_translated_pattern_cases(run_seed, cases_per_family, BenchmarkFamily::LongLivedMethuselah, &["acorn", "diehard", "r_pentomino"], 0x7000_0000_0000_0000) }
fn seeded_mover_cases(run_seed: u64, cases_per_family: usize) -> Vec<BenchmarkCase> { seeded_translated_pattern_cases(run_seed, cases_per_family, BenchmarkFamily::TranslatedPeriodicMover, &["glider", "blinker"], 0x7100_0000_0000_0000) }
fn seeded_gun_cases(run_seed: u64, cases_per_family: usize) -> Vec<BenchmarkCase> { seeded_translated_pattern_cases(run_seed, cases_per_family, BenchmarkFamily::GunPufferBreeder, &["gosper_glider_gun", "glider_producing_switch_engine", "blinker_puffer_1"], 0x7200_0000_0000_0000) }
fn seeded_ash_cases(run_seed: u64, cases_per_family: usize) -> Vec<BenchmarkCase> { let traffic_jam = BitGrid::from_cells(&[(0,1),(1,1),(2,1),(1,0),(1,2),(6,1),(7,1),(8,1),(7,0),(7,2)]); seeded_translated_grid_cases(run_seed, cases_per_family, BenchmarkFamily::DeceptiveAsh, &[("traffic_jam", traffic_jam), ("block", pattern_by_name("block").unwrap())], 0x7300_0000_0000_0000) }
fn seeded_translated_pattern_cases(run_seed: u64, cases_per_family: usize, family: BenchmarkFamily, names: &[&str], salt: u64) -> Vec<BenchmarkCase> { let mut cases=Vec::new(); for idx in 0..cases_per_family { let case_seed = mix_seed(run_seed ^ salt ^ idx as u64); let name = pick_from(names, case_seed, 0); let base = pattern_by_name(name).unwrap(); let (dx, dy) = seeded_translation(case_seed); let grid = base.translated(dx, dy); cases.push(seeded_case(format!("{family}_{name}_dx{dx}_dy{dy}_seed{case_seed}"), family, i32::try_from(grid.bounds().map(|(min_x, _, max_x, _)| max_x - min_x + 1).unwrap_or(0)).expect("benchmark size exceeded i32"), estimate_density_percent(&grid), grid, Some(format!("family={family},pattern={name},dx={dx},dy={dy},seed={case_seed}")))); } cases }
fn seeded_translated_grid_cases(run_seed: u64, cases_per_family: usize, family: BenchmarkFamily, bases: &[(&str, BitGrid)], salt: u64) -> Vec<BenchmarkCase> { let mut cases=Vec::new(); for idx in 0..cases_per_family { let case_seed = mix_seed(run_seed ^ salt ^ idx as u64); let (name, base) = &bases[(mix_seed(case_seed) as usize) % bases.len()]; let (dx, dy) = seeded_translation(case_seed); let grid = base.translated(dx, dy); cases.push(seeded_case(format!("{family}_{name}_dx{dx}_dy{dy}_seed{case_seed}"), family, i32::try_from(grid.bounds().map(|(min_x, _, max_x, _)| max_x - min_x + 1).unwrap_or(0)).expect("benchmark size exceeded i32"), estimate_density_percent(&grid), grid, Some(format!("family={family},pattern={name},dx={dx},dy={dy},seed={case_seed}")))); } cases }
fn head_on_glider_collision(distance: Coord) -> BitGrid { let southeast_glider = [(1,0),(2,1),(0,2),(1,2),(2,2)]; let northwest_glider = [(0,0),(1,0),(2,0),(0,1),(1,2)]; let mut cells = offset_cells(&southeast_glider, 0, 0); cells.extend(offset_cells(&northwest_glider, distance, distance)); BitGrid::from_cells(&cells) }
pub(crate) fn distant_glider_trigger(distance: Coord, target: BitGrid, target_origin: Cell) -> BitGrid { let glider = [(1,0),(2,1),(0,2),(1,2),(2,2)]; let mut cells = target.live_cells().into_iter().map(|(x,y)| (x + target_origin.0, y + target_origin.1)).collect::<Vec<_>>(); cells.extend(offset_cells(&glider, target_origin.0 - distance, target_origin.1 - distance)); BitGrid::from_cells(&cells) }
fn offset_cells(cells: &[Cell], dx: Coord, dy: Coord) -> Vec<Cell> { cells.iter().map(|&(x, y)| (x + dx, y + dy)).collect() }
fn seeded_translation(seed: u64) -> (Coord, Coord) { (((seed >> 8) % 48) as Coord, ((seed >> 20) % 48) as Coord) }
pub(super) fn estimate_density_percent(grid: &BitGrid) -> u32 { let Some((min_x, min_y, max_x, max_y)) = grid.bounds() else { return 0; }; let area = ((max_x - min_x + 1) * (max_y - min_y + 1)).max(1) as usize; ((grid.population() * 100) / area) as u32 }
fn clustered_noise_soup(width: Coord, height: Coord, fill_percent: u32, seed: u64) -> BitGrid { let base = random_soup(width, height, fill_percent, seed); let mut cells = Vec::new(); for (x, y) in base.live_cells() { cells.push((x, y)); if ((x + y).unsigned_abs() + seed).is_multiple_of(3) && x + 1 < width { cells.push((x + 1, y)); } if ((x * 3 + y * 5).unsigned_abs() + seed).is_multiple_of(5) && y + 1 < height { cells.push((x, y + 1)); } } BitGrid::from_cells(&cells) }
fn structured_random_soup(width: Coord, height: Coord, seed: u64) -> BitGrid { let left = random_soup(width / 2, height, 18, seed); let right = random_soup(width / 2, height, 12, seed ^ SPLITMIX64_GAMMA); let mut cells = left.live_cells(); cells.extend(right.live_cells().into_iter().map(|(x,y)| (x + (width / 2), y))); cells.extend(pattern_by_name("blinker").unwrap().live_cells().into_iter().map(|(x,y)| (x + width / 3, y + height / 3))); cells.extend(pattern_by_name("block").unwrap().live_cells().into_iter().map(|(x,y)| (x + width / 2, y + height / 2))); BitGrid::from_cells(&cells) }
fn pick_from<T: Copy>(values: &[T], seed: u64, salt: u64) -> T { values[(mix_seed(seed ^ salt) as usize) % values.len()] }
pub(super) fn reference_is_decisive_runtime(classification: &Classification) -> bool { !matches!(classification, Classification::Unknown { .. }) }
pub(crate) fn effective_generation_limit(limits: &ClassificationLimits, population: usize, bounds: Option<(Coord, Coord, Coord, Coord)>) -> u64 { const SMALL_PATTERN_POPULATION: usize = 64; const SMALL_PATTERN_SPAN: Coord = 24; const MIN_EXTENDED_LIMIT: u64 = 1024; const MAX_EXTENDED_LIMIT: u64 = 2048; let Some((min_x, min_y, max_x, max_y)) = bounds else { return limits.max_generations; }; let width = max_x - min_x + 1; let height = max_y - min_y + 1; if population <= SMALL_PATTERN_POPULATION && width <= SMALL_PATTERN_SPAN && height <= SMALL_PATTERN_SPAN { return limits.max_generations.clamp(MIN_EXTENDED_LIMIT, MAX_EXTENDED_LIMIT); } limits.max_generations }
pub(crate) fn canonical_small_box_mask(mask: u32, width: usize, height: usize) -> u32 { let mut best = transform_mask(mask, width, height, 0); for transform in 1..8 { best = best.min(transform_mask(mask, width, height, transform)); } best }
fn transform_mask(mask: u32, width: usize, height: usize, transform: usize) -> u32 { let mut transformed = 0_u32; for y in 0..height { for x in 0..width { let bit = y * width + x; if (mask & (1_u32 << bit)) == 0 { continue; } let (tx, ty) = transform_small_box_coord(x, y, width, height, transform); transformed |= 1_u32 << (ty * width + tx); } } transformed }
fn transform_small_box_coord(x: usize, y: usize, width: usize, height: usize, transform: usize) -> (usize, usize) { let max_x = width - 1; let max_y = height - 1; match transform { 0 => (x, y), 1 => (max_x - x, y), 2 => (x, max_y - y), 3 => (max_x - x, max_y - y), 4 => (y, x), 5 => (max_y - y, x), 6 => (y, max_x - x), 7 => (max_y - y, max_x - x), _ => unreachable!(), } }
