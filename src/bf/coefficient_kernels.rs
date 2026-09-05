use std::sync::OnceLock;

#[allow(unsafe_code)]
#[path = "coefficient_kernels/avx2.rs"]
mod avx2;
#[allow(unsafe_code)]
#[path = "coefficient_kernels/neon.rs"]
mod neon;

const NATIVE_BREAK_EVEN_LEN: usize = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CoefficientKernelAccounting {
    pub(crate) native_avx2_lanes: usize,
    pub(crate) native_neon_lanes: usize,
    pub(crate) scalar_lanes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CoefficientKernelError {
    LengthMismatch,
    InvalidBitWidth,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KernelFlavor {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

/// Multiplies corresponding input lanes modulo `2^bits` into `output`.
///
/// All slices must have the same length and `bits` must be at most 63. Invalid
/// calls return before mutating `output`.
pub(crate) fn multiply_low64_mod(
    output: &mut [u64],
    lhs: &[u64],
    rhs: &[u64],
    bits: u32,
) -> Result<CoefficientKernelAccounting, CoefficientKernelError> {
    if output.len() != lhs.len() || lhs.len() != rhs.len() {
        return Err(CoefficientKernelError::LengthMismatch);
    }
    let mask = modulo_mask(bits)?;
    Ok(dispatch(output, lhs, rhs, mask))
}

/// Scales every input lane by `factor` modulo `2^bits` into `output`.
///
/// Both slices must have the same length and `bits` must be at most 63. Invalid
/// calls return before mutating `output`.
pub(crate) fn scale_low64_mod(
    output: &mut [u64],
    values: &[u64],
    factor: u64,
    bits: u32,
) -> Result<CoefficientKernelAccounting, CoefficientKernelError> {
    if output.len() != values.len() {
        return Err(CoefficientKernelError::LengthMismatch);
    }
    let mask = modulo_mask(bits)?;
    Ok(dispatch_scale(output, values, factor, mask))
}

fn modulo_mask(bits: u32) -> Result<u64, CoefficientKernelError> {
    if bits > 63 {
        return Err(CoefficientKernelError::InvalidBitWidth);
    }
    Ok(if bits == 0 { 0 } else { (1_u64 << bits) - 1 })
}

fn dispatch(
    output: &mut [u64],
    lhs: &[u64],
    rhs: &[u64],
    mask: u64,
) -> CoefficientKernelAccounting {
    if output.len() < NATIVE_BREAK_EVEN_LEN {
        scalar_multiply(output, lhs, rhs, mask);
        return CoefficientKernelAccounting {
            scalar_lanes: output.len(),
            ..CoefficientKernelAccounting::default()
        };
    }

    match selected_flavor() {
        #[cfg(target_arch = "x86_64")]
        KernelFlavor::Avx2 => {
            let native_lanes = output.len() / avx2::LANES * avx2::LANES;
            let mut accounting = avx2::multiply(
                &mut output[..native_lanes],
                &lhs[..native_lanes],
                &rhs[..native_lanes],
                mask,
            );
            scalar_multiply(
                &mut output[native_lanes..],
                &lhs[native_lanes..],
                &rhs[native_lanes..],
                mask,
            );
            accounting.scalar_lanes += output.len() - native_lanes;
            accounting
        }
        #[cfg(target_arch = "aarch64")]
        KernelFlavor::Neon => {
            let native_lanes = output.len() / neon::LANES * neon::LANES;
            let mut accounting = neon::multiply(
                &mut output[..native_lanes],
                &lhs[..native_lanes],
                &rhs[..native_lanes],
                mask,
            );
            scalar_multiply(
                &mut output[native_lanes..],
                &lhs[native_lanes..],
                &rhs[native_lanes..],
                mask,
            );
            accounting.scalar_lanes += output.len() - native_lanes;
            accounting
        }
        KernelFlavor::Scalar => {
            scalar_multiply(output, lhs, rhs, mask);
            CoefficientKernelAccounting {
                scalar_lanes: output.len(),
                ..CoefficientKernelAccounting::default()
            }
        }
    }
}

fn dispatch_scale(
    output: &mut [u64],
    values: &[u64],
    factor: u64,
    mask: u64,
) -> CoefficientKernelAccounting {
    if output.len() < NATIVE_BREAK_EVEN_LEN {
        scalar_scale(output, values, factor, mask);
        return CoefficientKernelAccounting {
            scalar_lanes: output.len(),
            ..CoefficientKernelAccounting::default()
        };
    }

    match selected_flavor() {
        #[cfg(target_arch = "x86_64")]
        KernelFlavor::Avx2 => {
            let native_lanes = output.len() / avx2::LANES * avx2::LANES;
            let mut accounting = avx2::scale(
                &mut output[..native_lanes],
                &values[..native_lanes],
                factor,
                mask,
            );
            scalar_scale(
                &mut output[native_lanes..],
                &values[native_lanes..],
                factor,
                mask,
            );
            accounting.scalar_lanes += output.len() - native_lanes;
            accounting
        }
        #[cfg(target_arch = "aarch64")]
        KernelFlavor::Neon => {
            let native_lanes = output.len() / neon::LANES * neon::LANES;
            let mut accounting = neon::scale(
                &mut output[..native_lanes],
                &values[..native_lanes],
                factor,
                mask,
            );
            scalar_scale(
                &mut output[native_lanes..],
                &values[native_lanes..],
                factor,
                mask,
            );
            accounting.scalar_lanes += output.len() - native_lanes;
            accounting
        }
        KernelFlavor::Scalar => {
            scalar_scale(output, values, factor, mask);
            CoefficientKernelAccounting {
                scalar_lanes: output.len(),
                ..CoefficientKernelAccounting::default()
            }
        }
    }
}

fn scalar_multiply(output: &mut [u64], lhs: &[u64], rhs: &[u64], mask: u64) {
    for ((result, &left), &right) in output.iter_mut().zip(lhs).zip(rhs) {
        *result = left.wrapping_mul(right) & mask;
    }
}

fn scalar_scale(output: &mut [u64], values: &[u64], factor: u64, mask: u64) {
    for (result, &value) in output.iter_mut().zip(values) {
        *result = value.wrapping_mul(factor) & mask;
    }
}

fn selected_flavor() -> KernelFlavor {
    static SELECTED: OnceLock<KernelFlavor> = OnceLock::new();
    *SELECTED.get_or_init(detect_flavor)
}

#[cfg(target_arch = "x86_64")]
fn detect_flavor() -> KernelFlavor {
    if !cfg!(miri) && std::arch::is_x86_feature_detected!("avx2") {
        KernelFlavor::Avx2
    } else {
        KernelFlavor::Scalar
    }
}

#[cfg(target_arch = "aarch64")]
fn detect_flavor() -> KernelFlavor {
    if cfg!(miri) {
        KernelFlavor::Scalar
    } else {
        KernelFlavor::Neon
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn detect_flavor() -> KernelFlavor {
    KernelFlavor::Scalar
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequiredExt as _;

    const LENGTHS: [usize; 5] = [1, 4, 8, 9, 17];
    const BITS: [u32; 6] = [1, 8, 16, 32, 33, 63];
    const POISON: u64 = 0xd3d3_d3d3_d3d3_d3d3;

    fn oracle(left: u64, right: u64, bits: u32) -> u64 {
        let product = u128::from(left) * u128::from(right);
        let bytes = product.to_le_bytes();
        let low = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        low & modulo_mask(bits).or_invariant("oracle bit width is valid")
    }

    fn fixtures(len: usize) -> (Vec<u64>, Vec<u64>) {
        let lhs = (0..len)
            .map(|lane| {
                u64::MAX.wrapping_sub(
                    u64::try_from(lane)
                        .or_invariant("fixture lane fits u64")
                        .wrapping_mul(0x1_0000_0001),
                )
            })
            .collect();
        let rhs = (0..len)
            .map(|lane| {
                0xffff_ffff_0000_0001_u64.wrapping_add(
                    u64::try_from(lane)
                        .or_invariant("fixture lane fits u64")
                        .wrapping_mul(0x8000_0001),
                )
            })
            .collect();
        (lhs, rhs)
    }

    #[test]
    fn multiply_matches_u128_oracle_for_full_partial_and_carry_heavy_batches() {
        for len in LENGTHS {
            let (lhs, rhs) = fixtures(len);
            for bits in BITS {
                let mut output = vec![POISON; len + 3];
                let accounting = multiply_low64_mod(&mut output[..len], &lhs, &rhs, bits)
                    .or_invariant("valid multiply fixture");
                let expected = lhs
                    .iter()
                    .zip(&rhs)
                    .map(|(&left, &right)| oracle(left, right, bits))
                    .collect::<Vec<_>>();
                assert_eq!(&output[..len], expected);
                assert_eq!(&output[len..], &[POISON; 3], "poisoned tail was mutated");
                assert_eq!(
                    accounting.native_avx2_lanes
                        + accounting.native_neon_lanes
                        + accounting.scalar_lanes,
                    len
                );
                if len < NATIVE_BREAK_EVEN_LEN {
                    assert_eq!(accounting.scalar_lanes, len);
                }
            }
        }
    }

    #[test]
    fn scale_matches_u128_oracle_for_full_partial_and_carry_heavy_batches() {
        const FACTOR: u64 = 0xffff_fffe_ffff_ffff;
        for len in LENGTHS {
            let (values, _) = fixtures(len);
            for bits in BITS {
                let mut output = vec![POISON; len + 3];
                let accounting = scale_low64_mod(&mut output[..len], &values, FACTOR, bits)
                    .or_invariant("valid scale fixture");
                let expected = values
                    .iter()
                    .map(|&value| oracle(value, FACTOR, bits))
                    .collect::<Vec<_>>();
                assert_eq!(&output[..len], expected);
                assert_eq!(&output[len..], &[POISON; 3], "poisoned tail was mutated");
                assert_eq!(
                    accounting.native_avx2_lanes
                        + accounting.native_neon_lanes
                        + accounting.scalar_lanes,
                    len
                );
            }
        }
    }

    #[test]
    fn zero_bits_zeroes_active_lanes() {
        let values = [u64::MAX; 5];
        let mut output = [POISON; 5];
        let result = scale_low64_mod(&mut output, &values, u64::MAX, 0);
        assert!(result.is_ok());
        assert_eq!(output, [0; 5]);
    }

    #[test]
    fn public_api_accounts_non_full_native_wave_as_scalar_tail() {
        let lhs = [u64::MAX; 5];
        let rhs = [0xffff_ffff_0000_0001; 5];
        let mut output = [POISON; 5];
        let accounting = multiply_low64_mod(&mut output, &lhs, &rhs, 63)
            .or_invariant("valid non-full-wave multiply");
        assert_eq!(
            accounting.native_avx2_lanes + accounting.native_neon_lanes + accounting.scalar_lanes,
            output.len()
        );
        if selected_flavor() == KernelFlavor::Scalar {
            assert_eq!(accounting.scalar_lanes, output.len());
        } else {
            assert_eq!(accounting.scalar_lanes, 1);
        }
    }

    #[test]
    fn invalid_calls_leave_output_untouched() {
        let lhs = [2, 3, 5, 7];
        let rhs = [11, 13, 17];
        let mut output = [POISON; 4];
        assert_eq!(
            multiply_low64_mod(&mut output, &lhs, &rhs, 8),
            Err(CoefficientKernelError::LengthMismatch)
        );
        assert_eq!(output, [POISON; 4]);
        assert_eq!(
            scale_low64_mod(&mut output, &lhs, 19, 64),
            Err(CoefficientKernelError::InvalidBitWidth)
        );
        assert_eq!(output, [POISON; 4]);
    }
}
