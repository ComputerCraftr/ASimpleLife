use super::*;
use crate::bitgrid::BitGrid;
use crate::hashlife::GridExtractionPolicy;

fn limits() -> CaptureLimits {
    CaptureLimits {
        residency: Duration::from_secs(5),
        ..CaptureLimits::default()
    }
}
fn extract(session: &mut HashLifeSession) -> BitGrid {
    session
        .extract_grid(GridExtractionPolicy::FullGridIfUnder {
            max_population: 1000,
            max_chunks: 1000,
            max_bounds_span: 10000,
        })
        .or_invariant("bounded captured state")
}

#[test]
fn owned_capture_preserves_exact_occurrence_and_survives_source_mutation() {
    let grid = BitGrid::from_cells(&[
        (-17, 12),
        (-16, 12),
        (-15, 12),
        (105, -27),
        (105, -26),
        (106, -27),
        (106, -26),
    ]);
    let mut source = HashLifeSession::new();
    source.try_load_grid(&grid).or_invariant("source");
    let cancelled = AtomicBool::new(false);
    let capture = source
        .capture_analysis(None, limits(), &cancelled)
        .or_invariant("capture");
    source
        .advance_root(5)
        .or_invariant("advance source independently");
    source.unload();
    let mut independent = capture
        .into_analysis_session(128 * 1024 * 1024, &cancelled)
        .or_invariant("independent session");
    assert_eq!(
        extract(&mut independent),
        grid,
        "capture changed world orientation, coordinates, or cells"
    );
}

#[test]
fn exact_crop_has_dead_infinite_exterior_not_wrapped_or_clipped_evolution() {
    let mut source = HashLifeSession::new();
    source
        .try_load_grid(&BitGrid::from_cells(&[(0, 0), (1, 0), (2, 0), (3, 0)]))
        .or_invariant("source");
    let cancelled = AtomicBool::new(false);
    let capture = source
        .capture_analysis(Some((0, 0, 2, 0)), limits(), &cancelled)
        .or_invariant("exact crop");
    let mut independent = capture
        .into_analysis_session(128 * 1024 * 1024, &cancelled)
        .or_invariant("independent crop");
    assert_eq!(
        extract(&mut independent),
        BitGrid::from_cells(&[(0, 0), (1, 0), (2, 0)])
    );
    independent.advance_root(1).or_invariant("crop evolution");
    assert_eq!(
        extract(&mut independent),
        BitGrid::from_cells(&[(1, -1), (1, 0), (1, 1)]),
        "births outside capture rectangle must survive"
    );
}

#[test]
fn capture_limits_and_cancellation_preserve_authoritative_state() {
    let grid = BitGrid::from_cells(&[(0, 0), (1, 0), (2, 0)]);
    let mut source = HashLifeSession::new();
    source.try_load_grid(&grid).or_invariant("source");
    for (limit, cancel, expected) in [
        (
            CaptureLimits {
                bytes: 1,
                ..limits()
            },
            false,
            CaptureError::TooLarge,
        ),
        (
            CaptureLimits {
                visits: 0,
                ..limits()
            },
            false,
            CaptureError::TooLarge,
        ),
        (
            CaptureLimits {
                residency: Duration::ZERO,
                ..limits()
            },
            false,
            CaptureError::Deadline,
        ),
        (limits(), true, CaptureError::Cancelled),
    ] {
        let result = source.capture_analysis(None, limit, &AtomicBool::new(cancel));
        assert_eq!(result.err(), Some(expected));
        assert_eq!(source.generation(), 0);
        assert_eq!(
            extract(&mut source),
            grid,
            "failed capture changed authoritative cells"
        );
    }
    assert!(
        source
            .capture_analysis(None, limits(), &AtomicBool::new(false))
            .is_ok(),
        "denied capture must remain retryable"
    );
}
