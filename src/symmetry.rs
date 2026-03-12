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
