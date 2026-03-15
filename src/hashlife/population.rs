#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PopulationCount {
    Exact(u128),
    AtLeast(u128),
}

impl PopulationCount {
    pub fn lower_bound(self) -> u128 {
        match self {
            Self::Exact(value) | Self::AtLeast(value) => value,
        }
    }

    pub fn is_zero(self) -> bool {
        matches!(self, Self::Exact(0))
    }

    pub fn is_exact(self) -> bool {
        matches!(self, Self::Exact(_))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PopulationStat {
    pub(super) lo: u64,
    pub(super) hi: u64,
    pub(super) saturated: bool,
}

impl PopulationStat {
    pub(super) fn exact(value: u128) -> Self {
        let (lo, hi) = super::memory::split_u128(value);
        Self {
            lo,
            hi,
            saturated: false,
        }
    }

    pub(super) const fn from_limbs(lo: u64, hi: u64, saturated: bool) -> Self {
        if saturated {
            Self {
                lo: u64::MAX,
                hi: u64::MAX,
                saturated: true,
            }
        } else {
            Self {
                lo,
                hi,
                saturated: false,
            }
        }
    }

    pub(super) fn value(self) -> u128 {
        (u128::from(self.hi) << 64) | u128::from(self.lo)
    }

    pub(super) fn sum(children: [Self; 4]) -> Self {
        let mut lo = 0_u64;
        let mut hi = 0_u64;
        for child in children {
            if child.saturated {
                return Self::from_limbs(u64::MAX, u64::MAX, true);
            }
            let (next_lo, carry) = crate::wide_math::add_u64_carry(lo, child.lo, false);
            let (next_hi, overflow) = crate::wide_math::add_u64_carry(hi, child.hi, carry);
            if overflow {
                return Self::from_limbs(u64::MAX, u64::MAX, true);
            }
            lo = next_lo;
            hi = next_hi;
        }
        Self::from_limbs(lo, hi, false)
    }

    pub(super) fn count(self) -> PopulationCount {
        if self.saturated {
            PopulationCount::AtLeast(u128::MAX)
        } else {
            PopulationCount::Exact(self.value())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct U192 {
        limbs: [u64; 3],
    }

    impl U192 {
        fn add_u128(&mut self, value: u128) {
            let (lo, hi) = super::super::memory::split_u128(value);
            let (next_lo, carry_lo) = self.limbs[0].overflowing_add(lo);
            let (next_hi, carry_hi) = self.limbs[1].overflowing_add(hi);
            let (next_hi, carry_from_lo) = next_hi.overflowing_add(u64::from(carry_lo));
            self.limbs = [
                next_lo,
                next_hi,
                self.limbs[2]
                    .saturating_add(u64::from(carry_hi))
                    .saturating_add(u64::from(carry_from_lo)),
            ];
        }

        fn expected_population(self) -> PopulationCount {
            if self.limbs[2] != 0 {
                PopulationCount::AtLeast(u128::MAX)
            } else {
                PopulationCount::Exact(
                    (u128::from(self.limbs[1]) << 64) | u128::from(self.limbs[0]),
                )
            }
        }
    }

    #[test]
    fn population_sum_reports_exact_values_without_overflow() {
        let sum = PopulationStat::sum([
            PopulationStat::exact(3),
            PopulationStat::exact(5),
            PopulationStat::exact(7),
            PopulationStat::exact(11),
        ]);
        assert_eq!(sum.count(), PopulationCount::Exact(26));
    }

    #[test]
    fn population_sum_saturates_without_claiming_exactness() {
        let sum = PopulationStat::sum([
            PopulationStat::exact(u128::MAX),
            PopulationStat::exact(1),
            PopulationStat::exact(0),
            PopulationStat::exact(0),
        ]);
        assert_eq!(sum.count(), PopulationCount::AtLeast(u128::MAX));
    }

    #[test]
    fn exact_u128_max_is_distinct_from_saturated_population() {
        let exact = PopulationStat::sum([
            PopulationStat::exact(u128::MAX),
            PopulationStat::exact(0),
            PopulationStat::exact(0),
            PopulationStat::exact(0),
        ]);
        assert_eq!(exact.count(), PopulationCount::Exact(u128::MAX));
    }

    #[test]
    fn population_sum_propagates_cross_limb_carry() {
        let sum = PopulationStat::sum([
            PopulationStat::exact(u128::from(u64::MAX)),
            PopulationStat::exact(1),
            PopulationStat::exact(0),
            PopulationStat::exact(0),
        ]);
        assert_eq!(sum.count(), PopulationCount::Exact(1_u128 << 64));
    }

    #[test]
    fn population_saturation_matches_three_limb_oracle() {
        let cases = [
            [u128::MAX, 0, 0, 0],
            [u128::MAX, 1, 0, 0],
            [u128::MAX, u128::MAX, 0, 0],
            [u128::MAX; 4],
        ];
        for children in cases {
            let mut oracle = U192::default();
            for child in children {
                oracle.add_u128(child);
            }
            let actual = PopulationStat::sum(children.map(PopulationStat::exact)).count();
            assert_eq!(
                actual,
                oracle.expected_population(),
                "population saturation mismatch children={children:?} oracle={oracle:?}"
            );
        }
    }
}
