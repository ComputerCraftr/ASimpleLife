#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum D4Symmetry {
    Identity,
    Rotate90,
    Rotate180,
    Rotate270,
    MirrorX,
    MirrorXRotate90,
    MirrorXRotate180,
    MirrorXRotate270,
}

impl D4Symmetry {
    pub(crate) const ALL: [Self; 8] = [
        Self::Identity,
        Self::Rotate90,
        Self::Rotate180,
        Self::Rotate270,
        Self::MirrorX,
        Self::MirrorXRotate90,
        Self::MirrorXRotate180,
        Self::MirrorXRotate270,
    ];

    pub(crate) const fn inverse(self) -> Self {
        match self {
            Self::Identity => Self::Identity,
            Self::Rotate90 => Self::Rotate270,
            Self::Rotate180 => Self::Rotate180,
            Self::Rotate270 => Self::Rotate90,
            Self::MirrorX => Self::MirrorX,
            Self::MirrorXRotate90 => Self::MirrorXRotate90,
            Self::MirrorXRotate180 => Self::MirrorXRotate180,
            Self::MirrorXRotate270 => Self::MirrorXRotate270,
        }
    }

    pub(crate) const fn then(self, next: Self) -> Self {
        const COMPOSE: [[D4Symmetry; 8]; 8] = [
            [
                Identity,
                Rotate90,
                Rotate180,
                Rotate270,
                MirrorX,
                MirrorXRotate90,
                MirrorXRotate180,
                MirrorXRotate270,
            ],
            [
                Rotate90,
                Rotate180,
                Rotate270,
                Identity,
                MirrorXRotate90,
                MirrorXRotate180,
                MirrorXRotate270,
                MirrorX,
            ],
            [
                Rotate180,
                Rotate270,
                Identity,
                Rotate90,
                MirrorXRotate180,
                MirrorXRotate270,
                MirrorX,
                MirrorXRotate90,
            ],
            [
                Rotate270,
                Identity,
                Rotate90,
                Rotate180,
                MirrorXRotate270,
                MirrorX,
                MirrorXRotate90,
                MirrorXRotate180,
            ],
            [
                MirrorX,
                MirrorXRotate270,
                MirrorXRotate180,
                MirrorXRotate90,
                Identity,
                Rotate270,
                Rotate180,
                Rotate90,
            ],
            [
                MirrorXRotate90,
                MirrorX,
                MirrorXRotate270,
                MirrorXRotate180,
                Rotate90,
                Identity,
                Rotate270,
                Rotate180,
            ],
            [
                MirrorXRotate180,
                MirrorXRotate90,
                MirrorX,
                MirrorXRotate270,
                Rotate180,
                Rotate90,
                Identity,
                Rotate270,
            ],
            [
                MirrorXRotate270,
                MirrorXRotate180,
                MirrorXRotate90,
                MirrorX,
                Rotate270,
                Rotate180,
                Rotate90,
                Identity,
            ],
        ];
        use D4Symmetry::*;
        COMPOSE[self as usize][next as usize]
    }

    pub(crate) const fn quadrant_perm(self) -> [usize; 4] {
        match self {
            Self::Identity => [0, 1, 2, 3],
            Self::Rotate90 => [1, 3, 0, 2],
            Self::Rotate180 => [3, 2, 1, 0],
            Self::Rotate270 => [2, 0, 3, 1],
            Self::MirrorX => [1, 0, 3, 2],
            Self::MirrorXRotate90 => [3, 1, 2, 0],
            Self::MirrorXRotate180 => [2, 3, 0, 1],
            Self::MirrorXRotate270 => [0, 2, 1, 3],
        }
    }

    #[cfg(test)]
    pub(crate) const fn grid3_perm(self) -> [usize; 9] {
        match self {
            Self::Identity => [0, 1, 2, 3, 4, 5, 6, 7, 8],
            Self::Rotate90 => [2, 5, 8, 1, 4, 7, 0, 3, 6],
            Self::Rotate180 => [8, 7, 6, 5, 4, 3, 2, 1, 0],
            Self::Rotate270 => [6, 3, 0, 7, 4, 1, 8, 5, 2],
            Self::MirrorX => [2, 1, 0, 5, 4, 3, 8, 7, 6],
            Self::MirrorXRotate90 => [8, 5, 2, 7, 4, 1, 6, 3, 0],
            Self::MirrorXRotate180 => [6, 7, 8, 3, 4, 5, 0, 1, 2],
            Self::MirrorXRotate270 => [0, 3, 6, 1, 4, 7, 2, 5, 8],
        }
    }

    pub(crate) fn transform_coords(self, x: usize, y: usize, max: usize) -> (usize, usize) {
        match self {
            Self::Identity => (x, y),
            Self::Rotate90 => (max - y, x),
            Self::Rotate180 => (max - x, max - y),
            Self::Rotate270 => (y, max - x),
            Self::MirrorX => (max - x, y),
            Self::MirrorXRotate90 => (max - y, max - x),
            Self::MirrorXRotate180 => (x, max - y),
            Self::MirrorXRotate270 => (y, x),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::D4Symmetry;

    #[test]
    fn d4_composition_table_matches_quadrant_geometry() {
        for left in D4Symmetry::ALL {
            for right in D4Symmetry::ALL {
                let left_perm = left.quadrant_perm();
                let right_perm = right.quadrant_perm();
                let expected = right_perm.map(|index| left_perm[index]);
                assert_eq!(
                    left.then(right).quadrant_perm(),
                    expected,
                    "composition mismatch for left={left:?} right={right:?}"
                );
            }
        }
    }
}
