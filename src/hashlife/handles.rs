use crate::RequiredExt;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(super) struct NodeId(u32);

impl NodeId {
    pub(super) const ZERO: Self = Self(0);
    pub(super) const MAX: Self = Self(u32::MAX);
    pub(super) const MAX_COUNT: usize = u32::MAX as usize;

    pub(super) fn index(self) -> usize {
        usize::try_from(self.0).or_invariant("node id exceeds usize")
    }

    #[cfg(test)]
    pub(super) const fn raw(self) -> u32 {
        self.0
    }

    pub(super) const fn precedes(self, parent: Self) -> bool {
        self.0 < parent.0
    }
}

impl TryFrom<usize> for NodeId {
    type Error = std::num::TryFromIntError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        u32::try_from(value).map(Self)
    }
}

impl From<NodeId> for u64 {
    fn from(value: NodeId) -> Self {
        value.0.into()
    }
}

impl From<bool> for NodeId {
    fn from(value: bool) -> Self {
        Self(u32::from(value))
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(super) struct CanonicalShapeId(u32);

impl CanonicalShapeId {
    pub(super) const DEAD: Self = Self(0);
    pub(super) const LIVE: Self = Self(1);
    pub(super) const MAX_COUNT: usize = u32::MAX as usize;

    pub(super) const fn from_raw(value: u32) -> Self {
        Self(value)
    }

    pub(super) fn index(self) -> usize {
        usize::try_from(self.0).or_invariant("canonical shape id exceeds usize")
    }

    pub(super) const fn raw(self) -> u32 {
        self.0
    }
}

impl TryFrom<usize> for CanonicalShapeId {
    type Error = std::num::TryFromIntError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        u32::try_from(value).map(Self)
    }
}

impl From<CanonicalShapeId> for u64 {
    fn from(value: CanonicalShapeId) -> Self {
        value.0.into()
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(super) struct PackedTransformId(u32);

impl PackedTransformId {
    pub(super) const ZERO: Self = Self(0);
    pub(super) const MAX_COUNT: usize = u32::MAX as usize;

    pub(super) fn index(self) -> usize {
        usize::try_from(self.0).or_invariant("packed transform id exceeds usize")
    }
}

impl TryFrom<usize> for PackedTransformId {
    type Error = std::num::TryFromIntError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        u32::try_from(value).map(Self)
    }
}

impl From<bool> for PackedTransformId {
    fn from(value: bool) -> Self {
        Self(u32::from(value))
    }
}

impl From<PackedTransformId> for u64 {
    fn from(value: PackedTransformId) -> Self {
        value.0.into()
    }
}
