use crate::bitgrid::Coord;

pub const MAX_COORD_ROOT_LEVEL: u32 = Coord::BITS - 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HashLifeGeometryError {
    LevelOutOfRange { level: u32, maximum: u32 },
    CoordinateRangeExceeded { axis: &'static str },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ValidatedLevel(u32);

impl ValidatedLevel {
    pub(crate) fn new(level: u32) -> Result<Self, HashLifeGeometryError> {
        if level > MAX_COORD_ROOT_LEVEL {
            return Err(HashLifeGeometryError::LevelOutOfRange {
                level,
                maximum: MAX_COORD_ROOT_LEVEL,
            });
        }
        Ok(Self(level))
    }

    pub(crate) const fn get(self) -> u32 {
        self.0
    }

    pub(crate) fn span(self) -> i128 {
        1_i128 << self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RootGeometry {
    pub(crate) level: ValidatedLevel,
    pub(crate) origin_x: Coord,
    pub(crate) origin_y: Coord,
}

impl RootGeometry {
    pub(crate) fn new(
        level: u32,
        origin_x: Coord,
        origin_y: Coord,
    ) -> Result<Self, HashLifeGeometryError> {
        let level = ValidatedLevel::new(level)?;
        validate_axis(level, origin_x, "x")?;
        validate_axis(level, origin_y, "y")?;
        Ok(Self {
            level,
            origin_x,
            origin_y,
        })
    }

    pub(crate) fn containing_bounds(
        level: u32,
        min_x: Coord,
        min_y: Coord,
        max_x: Coord,
        max_y: Coord,
    ) -> Result<Self, HashLifeGeometryError> {
        let level = ValidatedLevel::new(level)?;
        let span = level.span();
        let origin_x = centered_origin(min_x, max_x, span, "x")?;
        let origin_y = centered_origin(min_y, max_y, span, "y")?;
        Self::new(level.get(), origin_x, origin_y)
    }
}

fn validate_axis(
    level: ValidatedLevel,
    origin: Coord,
    axis: &'static str,
) -> Result<(), HashLifeGeometryError> {
    let end = i128::from(origin) + level.span() - 1;
    Coord::try_from(end)
        .map(|_| ())
        .map_err(|_| HashLifeGeometryError::CoordinateRangeExceeded { axis })
}

fn centered_origin(
    minimum: Coord,
    maximum: Coord,
    span: i128,
    axis: &'static str,
) -> Result<Coord, HashLifeGeometryError> {
    let minimum = i128::from(minimum);
    let maximum = i128::from(maximum);
    let width = maximum - minimum + 1;
    if width <= 0 || width > span {
        return Err(HashLifeGeometryError::CoordinateRangeExceeded { axis });
    }
    let desired = minimum - (span - width) / 2;
    let lowest = i128::from(Coord::MIN);
    let highest = i128::from(Coord::MAX) - span + 1;
    Coord::try_from(desired.clamp(lowest, highest))
        .map_err(|_| HashLifeGeometryError::CoordinateRangeExceeded { axis })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequiredExt as _;

    #[test]
    fn containing_geometry_clamps_at_both_coordinate_edges() {
        let low = RootGeometry::containing_bounds(2, Coord::MIN, 0, Coord::MIN, 0)
            .or_invariant("minimum coordinate should fit a level-2 root");
        assert_eq!(low.origin_x, Coord::MIN);
        assert_eq!(
            i128::from(low.origin_x) + low.level.span() - 1,
            i128::from(Coord::MIN) + 3
        );

        let high = RootGeometry::containing_bounds(2, Coord::MAX, 0, Coord::MAX, 0)
            .or_invariant("maximum coordinate should fit a level-2 root");
        assert_eq!(high.origin_x, Coord::MAX - 3);
        assert_eq!(
            i128::from(high.origin_x) + high.level.span() - 1,
            i128::from(Coord::MAX)
        );
    }

    #[test]
    fn geometry_rejects_levels_and_intervals_outside_coordinate_contract() {
        assert!(matches!(
            RootGeometry::new(MAX_COORD_ROOT_LEVEL + 1, 0, 0),
            Err(HashLifeGeometryError::LevelOutOfRange { .. })
        ));
        assert!(matches!(
            RootGeometry::new(MAX_COORD_ROOT_LEVEL, Coord::MAX, 0),
            Err(HashLifeGeometryError::CoordinateRangeExceeded { axis: "x" })
        ));
    }
}
