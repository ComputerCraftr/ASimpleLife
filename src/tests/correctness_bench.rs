use crate::benchmark::{
    assert_same_outcome, bitmask_pattern, canonical_small_box_mask, reference_classify,
    reference_is_decisive,
};
use crate::classify::{ClassificationLimits, classify_seed};
use crate::memo::Memo;

#[test]
#[ignore = "exhaustive 5x5 sweep is expensive"]
fn exhaustive_all_5x5_patterns_reference_check() {
    let limits = ClassificationLimits {
        max_generations: 128,
        max_population: 10_000,
        max_bounding_box: 256,
    };
    for mask in 0_u32..(1_u32 << 25) {
        if canonical_small_box_mask(mask, 5, 5) != mask {
            continue;
        }
        let grid = bitmask_pattern(mask, 5, 5);
        let expected = reference_classify(&grid, &limits);
        if !reference_is_decisive(&expected) {
            continue;
        }
        let actual = classify_seed(&grid, &limits, &mut Memo::default());
        assert_same_outcome(&format!("mask_{mask}"), &expected, &actual);
    }
}
