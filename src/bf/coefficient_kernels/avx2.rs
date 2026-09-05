#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use super::CoefficientKernelAccounting;

#[cfg(target_arch = "x86_64")]
pub(super) const LANES: usize = 4;

#[cfg(target_arch = "x86_64")]
pub(super) fn multiply(
    output: &mut [u64],
    lhs: &[u64],
    rhs: &[u64],
    mask: u64,
) -> CoefficientKernelAccounting {
    assert_eq!(output.len(), lhs.len());
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(output.len() % LANES, 0);
    assert!(std::arch::is_x86_feature_detected!("avx2"));
    // SAFETY: the assertions establish AVX2 support and equal, complete-vector slices.
    unsafe { multiply_kernel(output, lhs, rhs, mask) }
}

#[cfg(target_arch = "x86_64")]
pub(super) fn scale(
    output: &mut [u64],
    values: &[u64],
    factor: u64,
    mask: u64,
) -> CoefficientKernelAccounting {
    assert_eq!(output.len(), values.len());
    assert_eq!(output.len() % LANES, 0);
    assert!(std::arch::is_x86_feature_detected!("avx2"));
    // SAFETY: the assertions establish AVX2 support and equal, complete-vector slices.
    unsafe { scale_kernel(output, values, factor, mask) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn multiply_kernel(
    output: &mut [u64],
    lhs: &[u64],
    rhs: &[u64],
    mask: u64,
) -> CoefficientKernelAccounting {
    let mask = _mm256_set1_epi64x(i64::from_ne_bytes(mask.to_ne_bytes()));
    for offset in (0..output.len()).step_by(LANES) {
        // SAFETY: every offset starts a complete four-lane chunk in each validated slice.
        let left = unsafe { _mm256_loadu_si256(lhs[offset..].as_ptr().cast()) };
        // SAFETY: `rhs` has the same validated length and chunk layout.
        let right = unsafe { _mm256_loadu_si256(rhs[offset..].as_ptr().cast()) };
        let product = _mm256_and_si256(mullo_u64(left, right), mask);
        // SAFETY: every output offset has four writable lanes.
        unsafe { _mm256_storeu_si256(output[offset..].as_mut_ptr().cast(), product) };
    }
    CoefficientKernelAccounting {
        native_avx2_lanes: output.len(),
        ..CoefficientKernelAccounting::default()
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_kernel(
    output: &mut [u64],
    values: &[u64],
    factor: u64,
    mask: u64,
) -> CoefficientKernelAccounting {
    let factor = _mm256_set1_epi64x(i64::from_ne_bytes(factor.to_ne_bytes()));
    let mask = _mm256_set1_epi64x(i64::from_ne_bytes(mask.to_ne_bytes()));
    for offset in (0..output.len()).step_by(LANES) {
        // SAFETY: every offset starts a complete four-lane chunk in each validated slice.
        let value = unsafe { _mm256_loadu_si256(values[offset..].as_ptr().cast()) };
        let product = _mm256_and_si256(mullo_u64(value, factor), mask);
        // SAFETY: every output offset has four writable lanes.
        unsafe { _mm256_storeu_si256(output[offset..].as_mut_ptr().cast(), product) };
    }
    CoefficientKernelAccounting {
        native_avx2_lanes: output.len(),
        ..CoefficientKernelAccounting::default()
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn mullo_u64(left: __m256i, right: __m256i) -> __m256i {
    let low = _mm256_mul_epu32(left, right);
    let cross_left = _mm256_mul_epu32(_mm256_srli_epi64::<32>(left), right);
    let cross_right = _mm256_mul_epu32(left, _mm256_srli_epi64::<32>(right));
    _mm256_add_epi64(
        low,
        _mm256_slli_epi64::<32>(_mm256_add_epi64(cross_left, cross_right)),
    )
}
