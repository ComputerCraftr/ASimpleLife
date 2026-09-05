#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use super::CoefficientKernelAccounting;

#[cfg(target_arch = "aarch64")]
pub(super) const LANES: usize = 2;

#[cfg(target_arch = "aarch64")]
pub(super) fn multiply(
    output: &mut [u64],
    lhs: &[u64],
    rhs: &[u64],
    mask: u64,
) -> CoefficientKernelAccounting {
    assert_eq!(output.len(), lhs.len());
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(output.len() % LANES, 0);
    assert!(std::arch::is_aarch64_feature_detected!("neon"));
    // SAFETY: the assertions establish NEON support and equal, complete-vector slices.
    unsafe { multiply_kernel(output, lhs, rhs, mask) }
}

#[cfg(target_arch = "aarch64")]
pub(super) fn scale(
    output: &mut [u64],
    values: &[u64],
    factor: u64,
    mask: u64,
) -> CoefficientKernelAccounting {
    assert_eq!(output.len(), values.len());
    assert_eq!(output.len() % LANES, 0);
    assert!(std::arch::is_aarch64_feature_detected!("neon"));
    // SAFETY: the assertions establish NEON support and equal, complete-vector slices.
    unsafe { scale_kernel(output, values, factor, mask) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn multiply_kernel(
    output: &mut [u64],
    lhs: &[u64],
    rhs: &[u64],
    mask: u64,
) -> CoefficientKernelAccounting {
    let mask = vdupq_n_u64(mask);
    for offset in (0..output.len()).step_by(LANES) {
        // SAFETY: every offset starts a complete two-lane chunk in each validated slice.
        let left = unsafe { vld1q_u64(lhs[offset..].as_ptr()) };
        // SAFETY: `rhs` has the same validated length and chunk layout.
        let right = unsafe { vld1q_u64(rhs[offset..].as_ptr()) };
        let product = vandq_u64(mullo_u64(left, right), mask);
        // SAFETY: every output offset has two writable lanes.
        unsafe { vst1q_u64(output[offset..].as_mut_ptr(), product) };
    }
    CoefficientKernelAccounting {
        native_neon_lanes: output.len(),
        ..CoefficientKernelAccounting::default()
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scale_kernel(
    output: &mut [u64],
    values: &[u64],
    factor: u64,
    mask: u64,
) -> CoefficientKernelAccounting {
    let factor = vdupq_n_u64(factor);
    let mask = vdupq_n_u64(mask);
    for offset in (0..output.len()).step_by(LANES) {
        // SAFETY: every offset starts a complete two-lane chunk in each validated slice.
        let value = unsafe { vld1q_u64(values[offset..].as_ptr()) };
        let product = vandq_u64(mullo_u64(value, factor), mask);
        // SAFETY: every output offset has two writable lanes.
        unsafe { vst1q_u64(output[offset..].as_mut_ptr(), product) };
    }
    CoefficientKernelAccounting {
        native_neon_lanes: output.len(),
        ..CoefficientKernelAccounting::default()
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
fn mullo_u64(left: uint64x2_t, right: uint64x2_t) -> uint64x2_t {
    let left_low = vmovn_u64(left);
    let right_low = vmovn_u64(right);
    let low = vmull_u32(left_low, right_low);
    let cross_left = vmull_u32(vshrn_n_u64::<32>(left), right_low);
    let cross_right = vmull_u32(left_low, vshrn_n_u64::<32>(right));
    vaddq_u64(low, vshlq_n_u64::<32>(vaddq_u64(cross_left, cross_right)))
}
