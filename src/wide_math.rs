#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct U129 {
    high: bool,
    low: u128,
}

impl U129 {
    pub(crate) const fn from_add(low: u128, high: bool) -> Self {
        Self { high, low }
    }

    #[cfg(test)]
    pub(crate) const fn parts(self) -> (bool, u128) {
        (self.high, self.low)
    }
}

pub(crate) fn add_u64_carry(lhs: u64, rhs: u64, carry: bool) -> (u64, bool) {
    let (sum, first_carry) = lhs.overflowing_add(rhs);
    let (sum, second_carry) = sum.overflowing_add(u64::from(carry));
    (sum, first_carry || second_carry)
}

pub(crate) fn add_u128_carry(lhs: u128, rhs: u128) -> U129 {
    let (low, high) = lhs.overflowing_add(rhs);
    U129::from_add(low, high)
}

pub(crate) fn squared_i64_distance(lhs: (i64, i64), rhs: (i64, i64)) -> U129 {
    let dx = u128::from(lhs.0.abs_diff(rhs.0));
    let dy = u128::from(lhs.1.abs_diff(rhs.1));
    add_u128_carry(dx * dx, dy * dy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_u129_orders_values_across_u128_overflow() {
        let below = add_u128_carry(u128::MAX - 1, 1);
        let above = add_u128_carry(u128::MAX, 1);
        assert_eq!(below.parts(), (false, u128::MAX));
        assert_eq!(above.parts(), (true, 0));
        assert!(above > below);
    }

    #[test]
    fn squared_distance_preserves_the_129th_bit() {
        let distance = squared_i64_distance((i64::MIN, i64::MIN), (i64::MAX, i64::MAX));
        assert!(distance.parts().0);
    }
}
