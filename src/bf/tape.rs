use crate::RequiredExt;

use super::BfOffset;

pub(crate) fn normalize_offset(offset: BfOffset, tape_len: usize) -> BfOffset {
    let modulus = BfOffset::try_from(tape_len).or_invariant("BF tape length exceeded BfOffset");
    offset.rem_euclid(modulus)
}

pub(crate) fn centered_offset(offset: BfOffset, tape_len: usize) -> BfOffset {
    let modulus = BfOffset::try_from(tape_len).or_invariant("BF tape length exceeded BfOffset");
    let normalized = normalize_offset(offset, tape_len);
    if normalized > modulus / 2 {
        normalized - modulus
    } else {
        normalized
    }
}

#[cfg(test)]
pub(crate) fn wrapped_index(base: usize, offset: BfOffset, tape_len: usize) -> usize {
    let amount = usize::try_from(normalize_offset(offset, tape_len))
        .or_invariant("wrapped BF tape offset exceeded usize");
    (base + amount) % tape_len
}

pub(crate) fn offsets_alias(lhs: BfOffset, rhs: BfOffset, tape_len: usize) -> bool {
    normalize_offset(lhs, tape_len) == normalize_offset(rhs, tape_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circular_addressing_handles_full_wraps_and_boundary_neighbors() {
        const LEN: usize = 30_000;
        assert_eq!(wrapped_index(0, 30_000, LEN), 0);
        assert_eq!(wrapped_index(0, -30_000, LEN), 0);
        assert_eq!(wrapped_index(LEN - 1, 1, LEN), 0);
        assert_eq!(wrapped_index(0, -1, LEN), LEN - 1);
        assert!(offsets_alias(-1, 29_999, LEN));
        assert_eq!(centered_offset(29_999, LEN), -1);
    }
}
