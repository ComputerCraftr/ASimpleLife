use crate::bitgrid::{BitGrid, Cell, Coord};
use crate::engine::SimulationSession;
use std::cmp::Reverse;
use std::error::Error;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewportMode {
    Auto,
    Manual,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ViewportError {
    InvalidDimensions { width: usize, height: usize },
    CoordinateRangeExceeded,
    RegionUnavailable,
    RevisionExhausted,
}

impl fmt::Display for ViewportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions { width, height } => {
                write!(f, "invalid viewport dimensions {width}x{height}")
            }
            Self::CoordinateRangeExceeded => f.write_str("viewport exceeded coordinate range"),
            Self::RegionUnavailable => f.write_str("viewport region could not be sampled"),
            Self::RevisionExhausted => f.write_str("viewport identity revisions exhausted"),
        }
    }
}

impl Error for ViewportError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewportSample {
    pub origin: Cell,
    pub grid: BitGrid,
}

/// A source that can expose bounds and a clipped region without materializing
/// its complete state.
pub trait ViewportSource {
    fn viewport_bounds(&mut self) -> Option<(Coord, Coord, Coord, Coord)>;

    fn sample_viewport_region(
        &mut self,
        min_x: Coord,
        min_y: Coord,
        max_x: Coord,
        max_y: Coord,
    ) -> Option<BitGrid>;
}

impl ViewportSource for SimulationSession {
    fn viewport_bounds(&mut self) -> Option<(Coord, Coord, Coord, Coord)> {
        self.hashlife_bounds()
    }

    fn sample_viewport_region(
        &mut self,
        min_x: Coord,
        min_y: Coord,
        max_x: Coord,
        max_y: Coord,
    ) -> Option<BitGrid> {
        self.sample_hashlife_state_region(min_x, min_y, max_x, max_y)
    }
}

#[derive(Clone, Debug)]
pub struct ViewportController {
    origin: Option<Cell>,
    width: usize,
    height: usize,
    mode: ViewportMode,
    preserve_origin_once: bool,
}

impl ViewportController {
    pub fn new(width: usize, height: usize) -> Result<Self, ViewportError> {
        checked_world_dimensions(width, height)?;
        Ok(Self {
            origin: None,
            width,
            height,
            mode: ViewportMode::Auto,
            preserve_origin_once: false,
        })
    }

    pub fn origin(&self) -> Option<Cell> {
        self.origin
    }

    pub fn mode(&self) -> ViewportMode {
        self.mode
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub fn set_mode(&mut self, mode: ViewportMode) {
        self.mode = mode;
    }

    pub fn set_origin(&mut self, origin: Cell) {
        self.origin = Some(origin);
        self.mode = ViewportMode::Manual;
        self.preserve_origin_once = false;
    }

    pub fn toggle_auto(&mut self) {
        self.mode = match self.mode {
            ViewportMode::Auto => ViewportMode::Manual,
            ViewportMode::Manual => ViewportMode::Auto,
        };
    }

    /// Follow a discovered group without changing auto/manual mode or chasing
    /// every phase's centroid. Resize gets one sample at its preserved center.
    pub(crate) fn focus_group(
        &mut self,
        bounds: crate::bitgrid::Bounds,
        force: bool,
    ) -> Result<(), ViewportError> {
        if self.preserve_origin_once && !force {
            return Ok(());
        }
        let (width, height) = checked_world_dimensions(self.width, self.height)?;
        if !force && let Some((x, y)) = self.origin {
            let inset_x = (width / 8).min(4);
            let inset_y = (height / 8).min(4);
            let fits = i128::from(bounds.2) - i128::from(bounds.0)
                < i128::from(width - 2 * inset_x)
                && i128::from(bounds.3) - i128::from(bounds.1) < i128::from(height - 2 * inset_y);
            let visible = if fits {
                i128::from(bounds.0) >= i128::from(x) + i128::from(inset_x)
                    && i128::from(bounds.1) >= i128::from(y) + i128::from(inset_y)
                    && i128::from(bounds.2) < i128::from(x) + i128::from(width - inset_x)
                    && i128::from(bounds.3) < i128::from(y) + i128::from(height - inset_y)
            } else {
                let cx = (i128::from(bounds.0) + i128::from(bounds.2)) / 2;
                let cy = (i128::from(bounds.1) + i128::from(bounds.3)) / 2;
                cx >= i128::from(x) + i128::from(width / 4)
                    && cx < i128::from(x) + i128::from(width - width / 4)
                    && cy >= i128::from(y) + i128::from(height / 4)
                    && cy < i128::from(y) + i128::from(height - height / 4)
            };
            if visible {
                return Ok(());
            }
        }
        self.origin = Some(checked_centered_origin(self.width, self.height, bounds)?);
        self.preserve_origin_once = false;
        Ok(())
    }

    pub fn pan(&mut self, dx: Coord, dy: Coord) -> Result<Cell, ViewportError> {
        let (x, y) = self.origin.unwrap_or((0, 0));
        let next = (
            x.checked_add(dx)
                .ok_or(ViewportError::CoordinateRangeExceeded)?,
            y.checked_add(dy)
                .ok_or(ViewportError::CoordinateRangeExceeded)?,
        );
        self.origin = Some(next);
        self.mode = ViewportMode::Manual;
        self.preserve_origin_once = false;
        Ok(next)
    }

    pub fn resize(&mut self, width: usize, height: usize) -> Result<(), ViewportError> {
        checked_world_dimensions(width, height)?;
        if self.width == width && self.height == height {
            return Ok(());
        }

        self.origin = checked_resized_origin(self.origin, self.width, self.height, width, height)?;
        self.width = width;
        self.height = height;
        self.preserve_origin_once = self.origin.is_some();
        Ok(())
    }

    /// Recenter once while preserving the current auto/manual mode.
    pub fn recenter<S: ViewportSource>(
        &mut self,
        source: &mut S,
    ) -> Result<Option<Cell>, ViewportError> {
        let Some(bounds) = source.viewport_bounds() else {
            self.origin = None;
            self.preserve_origin_once = false;
            return Ok(None);
        };
        let origin = checked_centered_origin(self.width, self.height, bounds)?;
        self.origin = Some(origin);
        self.preserve_origin_once = false;
        Ok(Some(origin))
    }

    pub fn sample<S: ViewportSource>(
        &mut self,
        source: &mut S,
    ) -> Result<ViewportSample, ViewportError> {
        let Some(bounds) = source.viewport_bounds() else {
            if self.mode == ViewportMode::Auto {
                self.origin = None;
            }
            self.preserve_origin_once = false;
            return Ok(ViewportSample {
                origin: self.origin.unwrap_or((0, 0)),
                grid: BitGrid::empty(),
            });
        };

        if self.mode == ViewportMode::Manual {
            let origin = match self.origin {
                Some(origin) => origin,
                None => checked_centered_origin(self.width, self.height, bounds)?,
            };
            self.origin = Some(origin);
            self.preserve_origin_once = false;
            return sample_at(source, origin, self.width, self.height);
        }

        let (world_width, world_height) = checked_world_dimensions(self.width, self.height)?;
        if let Some(origin) = self.origin {
            let sample = sample_at(source, origin, self.width, self.height)?;
            // Follow the occupied region, not whichever distant oscillator has
            // the largest population in this phase. Reacquire only after loss.
            if !sample.grid.is_empty() || self.preserve_origin_once {
                self.preserve_origin_once = false;
                return Ok(sample);
            }
        }
        let centered = checked_centered_origin(self.width, self.height, bounds)?;
        let (min_x, min_y, max_x, max_y) = bounds;
        let right = aligned_end_origin(max_x, world_width)?;
        let bottom = aligned_end_origin(max_y, world_height)?;
        let mut origins = vec![
            centered,
            (min_x, min_y),
            (right, min_y),
            (min_x, bottom),
            (right, bottom),
        ];
        if let Some(origin) = self.origin {
            origins.push(origin);
        }
        origins.sort_unstable();
        origins.dedup();

        let coarse_count = origins.len();
        let mut candidates = Vec::with_capacity(coarse_count.saturating_mul(2));
        for origin in origins {
            match sample_at(source, origin, self.width, self.height) {
                Ok(sample) => candidates.push(sample),
                Err(ViewportError::CoordinateRangeExceeded) => {}
                Err(error) => return Err(error),
            }
        }
        if candidates.is_empty() {
            return Err(ViewportError::CoordinateRangeExceeded);
        }

        let refinement_limit = coarse_count.saturating_mul(4);
        let mut index = 0;
        while index < candidates.len() && candidates.len() < refinement_limit {
            let Some(visible_bounds) = candidates[index].grid.bounds() else {
                index += 1;
                continue;
            };
            let origin = checked_centered_origin(self.width, self.height, visible_bounds)?;
            if !candidates.iter().any(|sample| sample.origin == origin) {
                match sample_at(source, origin, self.width, self.height) {
                    Ok(sample) => candidates.push(sample),
                    Err(ViewportError::CoordinateRangeExceeded) => {}
                    Err(error) => return Err(error),
                }
            }
            index += 1;
        }

        let proposed_index = candidates
            .iter()
            .enumerate()
            .max_by_key(|(_, sample)| candidate_score(sample, world_width, world_height))
            .map_or(0, |(index, _)| index);
        let selected = candidates.swap_remove(proposed_index);
        self.preserve_origin_once = false;
        self.origin = Some(selected.origin);
        Ok(selected)
    }

    pub(crate) fn sample_focus<S: ViewportSource>(
        &mut self,
        source: &mut S,
    ) -> Result<ViewportSample, ViewportError> {
        if self.origin.is_none() {
            self.recenter(source)?;
        }
        self.preserve_origin_once = false;
        sample_at(
            source,
            self.origin.unwrap_or((0, 0)),
            self.width,
            self.height,
        )
    }
}

fn checked_world_dimensions(width: usize, height: usize) -> Result<(Coord, Coord), ViewportError> {
    if width == 0 || height == 0 {
        return Err(ViewportError::InvalidDimensions { width, height });
    }
    let width = Coord::try_from(width).map_err(|_| ViewportError::CoordinateRangeExceeded)?;
    let height = Coord::try_from(height).map_err(|_| ViewportError::CoordinateRangeExceeded)?;
    let height = height
        .checked_mul(2)
        .ok_or(ViewportError::CoordinateRangeExceeded)?;
    Ok((width, height))
}

fn checked_centered_origin(
    width: usize,
    height: usize,
    bounds: (Coord, Coord, Coord, Coord),
) -> Result<Cell, ViewportError> {
    checked_world_dimensions(width, height)?;
    let (min_x, min_y, max_x, max_y) = bounds;
    let viewport_width =
        i128::try_from(width).map_err(|_| ViewportError::CoordinateRangeExceeded)?;
    let viewport_height = i128::try_from(height)
        .map_err(|_| ViewportError::CoordinateRangeExceeded)?
        .checked_mul(2)
        .ok_or(ViewportError::CoordinateRangeExceeded)?;
    let x = (i128::from(min_x) + i128::from(max_x) - viewport_width + 1).div_euclid(2);
    let y = (i128::from(min_y) + i128::from(max_y) - viewport_height + 1).div_euclid(2);
    Ok((
        Coord::try_from(x).map_err(|_| ViewportError::CoordinateRangeExceeded)?,
        Coord::try_from(y).map_err(|_| ViewportError::CoordinateRangeExceeded)?,
    ))
}

fn checked_resized_origin(
    origin: Option<Cell>,
    old_width: usize,
    old_height: usize,
    new_width: usize,
    new_height: usize,
) -> Result<Option<Cell>, ViewportError> {
    let Some((x, y)) = origin else {
        return Ok(None);
    };
    let old_width =
        i128::try_from(old_width).map_err(|_| ViewportError::CoordinateRangeExceeded)?;
    let old_height =
        i128::try_from(old_height).map_err(|_| ViewportError::CoordinateRangeExceeded)?;
    let new_width =
        i128::try_from(new_width).map_err(|_| ViewportError::CoordinateRangeExceeded)?;
    let new_height =
        i128::try_from(new_height).map_err(|_| ViewportError::CoordinateRangeExceeded)?;
    let next_x = i128::from(x) + old_width / 2 - new_width / 2;
    let next_y = i128::from(y) + old_height - new_height;
    Ok(Some((
        Coord::try_from(next_x).map_err(|_| ViewportError::CoordinateRangeExceeded)?,
        Coord::try_from(next_y).map_err(|_| ViewportError::CoordinateRangeExceeded)?,
    )))
}

fn aligned_end_origin(end: Coord, span: Coord) -> Result<Coord, ViewportError> {
    let origin = i128::from(end) - i128::from(span) + 1;
    Coord::try_from(origin).map_err(|_| ViewportError::CoordinateRangeExceeded)
}

fn sample_at<S: ViewportSource>(
    source: &mut S,
    origin: Cell,
    width: usize,
    height: usize,
) -> Result<ViewportSample, ViewportError> {
    let (world_width, world_height) = checked_world_dimensions(width, height)?;
    let max_x = aligned_region_end(origin.0, world_width)?;
    let max_y = aligned_region_end(origin.1, world_height)?;
    let grid = source
        .sample_viewport_region(origin.0, origin.1, max_x, max_y)
        .ok_or(ViewportError::RegionUnavailable)?;
    Ok(ViewportSample { origin, grid })
}

fn aligned_region_end(origin: Coord, span: Coord) -> Result<Coord, ViewportError> {
    let end = i128::from(origin) + i128::from(span) - 1;
    Coord::try_from(end).map_err(|_| ViewportError::CoordinateRangeExceeded)
}

fn candidate_score(
    sample: &ViewportSample,
    width: Coord,
    height: Coord,
) -> (usize, i128, Reverse<u128>, Reverse<Cell>) {
    let Some((min_x, min_y, max_x, max_y)) = sample.grid.bounds() else {
        return (0, i128::MIN, Reverse(u128::MAX), Reverse(sample.origin));
    };
    let right = i128::from(sample.origin.0) + i128::from(width) - 1;
    let bottom = i128::from(sample.origin.1) + i128::from(height) - 1;
    let left_margin = i128::from(min_x) - i128::from(sample.origin.0);
    let right_margin = right - i128::from(max_x);
    let top_margin = i128::from(min_y) - i128::from(sample.origin.1);
    let bottom_margin = bottom - i128::from(max_y);
    let minimum_margin = left_margin
        .min(right_margin)
        .min(top_margin)
        .min(bottom_margin);
    let imbalance = left_margin
        .abs_diff(right_margin)
        .saturating_add(top_margin.abs_diff(bottom_margin));
    (
        sample.grid.population(),
        minimum_margin,
        Reverse(imbalance),
        Reverse(sample.origin),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequiredExt;
    use crate::life::GameOfLife;

    #[test]
    fn distant_out_of_phase_pulsars_never_compete_for_camera_focus() {
        let seed = crate::generators::pattern_by_name("pulsar").or_invariant("pulsar exists");
        let mut left = GameOfLife::new(seed.clone());
        let mut right = GameOfLife::new(seed.translated(1_000, 1_000));
        right.step();
        let mut source = GridSource::new(BitGrid::empty());
        let mut viewport = ViewportController::new(30, 15).or_invariant("valid viewport");
        let mut first = None;
        for generation in 0..36 {
            let mut cells = left.grid().live_cells();
            cells.extend(right.grid().live_cells());
            source.replace(BitGrid::from_cells(&cells));
            source.sampled_regions.clear();
            let sample = viewport
                .sample(&mut source)
                .or_invariant("pulsars can be sampled");
            if let Some(origin) = first {
                assert_eq!(
                    sample.origin, origin,
                    "camera switched organisms at generation {generation}"
                );
                assert_eq!(
                    source.sampled_regions.len(),
                    1,
                    "occupied focus triggered a distant search"
                );
            } else {
                first = Some(sample.origin);
            }
            assert!(
                sample.grid.population() >= 48,
                "focused pulsar was lost at {generation}"
            );
            left.step();
            right.step();
        }
    }

    struct GridSource {
        grid: BitGrid,
        sampled_regions: Vec<(Coord, Coord, Coord, Coord)>,
    }

    impl GridSource {
        fn new(grid: BitGrid) -> Self {
            Self {
                grid,
                sampled_regions: Vec::new(),
            }
        }

        fn replace(&mut self, grid: BitGrid) {
            self.grid = grid;
        }
    }

    impl ViewportSource for GridSource {
        fn viewport_bounds(&mut self) -> Option<(Coord, Coord, Coord, Coord)> {
            self.grid.bounds()
        }

        fn sample_viewport_region(
            &mut self,
            min_x: Coord,
            min_y: Coord,
            max_x: Coord,
            max_y: Coord,
        ) -> Option<BitGrid> {
            self.sampled_regions.push((min_x, min_y, max_x, max_y));
            Some(BitGrid::from_cells(
                &self
                    .grid
                    .live_cells()
                    .into_iter()
                    .filter(|&(x, y)| x >= min_x && x <= max_x && y >= min_y && y <= max_y)
                    .collect::<Vec<_>>(),
            ))
        }
    }

    #[test]
    fn auto_viewport_keeps_one_origin_across_blinker_phases() {
        let mut game = GameOfLife::new(BitGrid::from_cells(&[(0, 0), (1, 0), (2, 0)]));
        let mut source = GridSource::new(game.grid().clone());
        let mut viewport =
            ViewportController::new(20, 10).or_invariant("test viewport should be valid");
        let mut origins = Vec::new();

        for _ in 0..6 {
            source.replace(game.grid().clone());
            let sample = viewport
                .sample(&mut source)
                .or_invariant("blinker viewport should be sampleable");
            origins.push(sample.origin);
            game.step();
        }

        assert!(
            origins.windows(2).all(|pair| pair[0] == pair[1]),
            "bounded oscillator moved the automatic viewport: {origins:?}"
        );
    }

    #[test]
    fn manual_viewport_never_recenters_as_pattern_moves() {
        let mut source = GridSource::new(BitGrid::from_cells(&[(0, 0), (1, 0), (2, 0)]));
        let mut viewport =
            ViewportController::new(8, 4).or_invariant("test viewport should be valid");
        viewport
            .recenter(&mut source)
            .or_invariant("initial recenter should fit coordinates");
        let manual_origin = viewport
            .pan(7, -3)
            .or_invariant("test pan should fit coordinates");

        source.replace(BitGrid::from_cells(&[(1_000, 1_000), (1_001, 1_000)]));
        let sample = viewport
            .sample(&mut source)
            .or_invariant("manual viewport should be sampleable");

        assert_eq!(viewport.mode(), ViewportMode::Manual);
        assert_eq!(sample.origin, manual_origin);
    }

    #[test]
    fn manual_sampling_does_not_center_an_offscreen_extreme_coordinate() {
        let mut source = GridSource::new(BitGrid::from_cells(&[(Coord::MIN, Coord::MIN)]));
        let mut viewport = ViewportController::new(8, 4).or_invariant("valid viewport");
        viewport.set_origin((0, 0));
        let sample = viewport
            .sample(&mut source)
            .or_invariant("valid manual region");
        assert_eq!(sample.origin, (0, 0));
        assert!(sample.grid.is_empty());
        assert_eq!(source.sampled_regions, vec![(0, 0, 7, 7)]);
    }

    #[test]
    fn fully_visible_auto_pattern_samples_only_the_current_region() {
        let mut source = GridSource::new(BitGrid::from_cells(&[(0, 0), (1, 0), (2, 0)]));
        let mut viewport = ViewportController::new(20, 10).or_invariant("valid viewport");
        let first = viewport.sample(&mut source).or_invariant("initial sample");
        source.sampled_regions.clear();
        let second = viewport.sample(&mut source).or_invariant("repeated sample");
        assert_eq!(second, first);
        assert_eq!(
            source.sampled_regions.len(),
            1,
            "stable view redundantly sampled candidates"
        );
    }

    #[test]
    fn auto_viewport_recenters_after_a_mover_leaves_the_window() {
        let mut game = GameOfLife::new(BitGrid::from_cells(&[
            (1, 0),
            (2, 1),
            (0, 2),
            (1, 2),
            (2, 2),
        ]));
        let mut source = GridSource::new(game.grid().clone());
        let mut viewport =
            ViewportController::new(4, 2).or_invariant("test viewport should be valid");
        let first = viewport
            .sample(&mut source)
            .or_invariant("initial viewport should be sampleable")
            .origin;

        let mut moved = None;
        for _ in 0..32 {
            game.step();
            source.replace(game.grid().clone());
            let sample = viewport
                .sample(&mut source)
                .or_invariant("moving glider viewport should be sampleable");
            if sample.origin != first {
                moved = Some(sample);
                break;
            }
        }
        let moved = moved.or_invariant("glider should eventually leave the initial viewport");

        assert_ne!(moved.origin, first);
        assert_eq!(moved.grid.population(), 5);
    }

    #[test]
    fn resize_preserves_world_space_center_and_next_sample_origin() {
        let mut source = GridSource::new(BitGrid::from_cells(&[(100, 200), (101, 200)]));
        let mut viewport =
            ViewportController::new(20, 10).or_invariant("test viewport should be valid");
        viewport
            .pan(100, 200)
            .or_invariant("test pan should fit coordinates");
        viewport
            .resize(30, 15)
            .or_invariant("test resize should fit coordinates");

        assert_eq!(viewport.origin(), Some((95, 195)));
        assert_eq!(
            viewport
                .sample(&mut source)
                .or_invariant("resized viewport should be sampleable")
                .origin,
            (95, 195)
        );
    }

    #[test]
    fn sampling_requests_only_clipped_viewport_regions() {
        let mut source = GridSource::new(BitGrid::from_cells(&[(0, 0)]));
        let mut viewport =
            ViewportController::new(7, 4).or_invariant("test viewport should be valid");
        let sample = viewport
            .sample(&mut source)
            .or_invariant("clipped viewport should be sampleable");

        assert!(!source.sampled_regions.is_empty());
        assert!(
            source
                .sampled_regions
                .iter()
                .all(|&(min_x, min_y, max_x, max_y)| {
                    max_x - min_x + 1 == 7 && max_y - min_y + 1 == 8
                })
        );
        assert_eq!(sample.grid.population(), 1);
    }

    #[test]
    fn pan_and_resize_reject_coordinate_overflow_without_mutation() {
        let mut viewport =
            ViewportController::new(8, 4).or_invariant("test viewport should be valid");
        viewport
            .pan(Coord::MAX, 0)
            .or_invariant("initial pan should fit coordinates");
        assert_eq!(
            viewport.pan(1, 0),
            Err(ViewportError::CoordinateRangeExceeded)
        );
        assert_eq!(viewport.origin(), Some((Coord::MAX, 0)));

        assert_eq!(
            viewport.resize(0, 4),
            Err(ViewportError::InvalidDimensions {
                width: 0,
                height: 4
            })
        );
        assert_eq!(viewport.dimensions(), (8, 4));
    }
}
