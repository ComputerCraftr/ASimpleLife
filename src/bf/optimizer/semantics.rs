use super::super::ir::ShiftDir;
use super::super::{BF_C_TAPE_LEN, BF_LIFE_TAPE_LEN};

/// Cells occupy at most 63 raw bits; zero-bit cells are always zero.
pub const MAX_CELL_BITS: u32 = 63;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodegenOptsError {
    CellWidth(u32),
    InputWidth(u32),
    OutputWidth(u32),
}

impl std::fmt::Display for CodegenOptsError {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (field, width) = match self {
            Self::CellWidth(width) => ("cell", width),
            Self::InputWidth(width) => ("input", width),
            Self::OutputWidth(width) => ("output", width),
        };
        write!(
            out,
            "unsupported BF {field} width {width}; expected 0..={MAX_CELL_BITS}"
        )
    }
}

impl std::error::Error for CodegenOptsError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoMode {
    Char,
    Number,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellSign {
    Signed,
    Unsigned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodegenOpts {
    pub io_mode: IoMode,
    pub cell_bits: u32,
    pub input_bits: Option<u32>,
    pub output_bits: Option<u32>,
    pub cell_sign: CellSign,
}

impl CodegenOpts {
    pub fn validate(self) -> Result<(), CodegenOptsError> {
        if self.cell_bits > MAX_CELL_BITS {
            return Err(CodegenOptsError::CellWidth(self.cell_bits));
        }
        if let Some(bits) = self.input_bits
            && bits > MAX_CELL_BITS
        {
            return Err(CodegenOptsError::InputWidth(bits));
        }
        if let Some(bits) = self.output_bits
            && bits > MAX_CELL_BITS
        {
            return Err(CodegenOptsError::OutputWidth(bits));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct OptimizerSemantics {
    cell_bits: u32,
    cell_sign: CellSign,
}

impl OptimizerSemantics {
    pub(super) fn from_opts(opts: CodegenOpts) -> Self {
        Self {
            cell_bits: opts.cell_bits,
            cell_sign: opts.cell_sign,
        }
    }

    #[cfg(test)]
    pub(super) fn default_unsigned() -> Self {
        Self {
            cell_bits: 8,
            cell_sign: CellSign::Unsigned,
        }
    }

    fn has_cells(self) -> bool {
        self.cell_bits > 0
    }

    pub(super) fn supports_shift(self, dir: ShiftDir, amount: u32, coeff: i32) -> bool {
        self.has_cells()
            && matches!(dir, ShiftDir::Left | ShiftDir::Right)
            && amount < 64
            && coeff > 0
            && u32::try_from(coeff)
                .ok()
                .is_some_and(|raw| raw.is_power_of_two())
    }

    pub(super) fn supports_muladd(self) -> bool {
        self.has_cells() && matches!(self.cell_sign, CellSign::Signed | CellSign::Unsigned)
    }

    pub(super) fn offsets_alias_on_supported_tape(
        self,
        lhs: crate::bf::BfOffset,
        rhs: crate::bf::BfOffset,
    ) -> bool {
        [BF_LIFE_TAPE_LEN, BF_C_TAPE_LEN]
            .into_iter()
            .any(|len| super::super::tape::offsets_alias(lhs, rhs, len))
    }

    pub(super) fn has_no_wrapped_offset_aliases(
        self,
        offsets: impl IntoIterator<Item = crate::bf::BfOffset>,
    ) -> bool {
        let offsets = offsets.into_iter().collect::<Vec<_>>();
        offsets.iter().enumerate().all(|(index, lhs)| {
            offsets[index + 1..]
                .iter()
                .all(|rhs| lhs == rhs || !self.offsets_alias_on_supported_tape(*lhs, *rhs))
        })
    }

    pub(super) fn shift_amount_for_coeff(self, coeff: i32) -> Option<u32> {
        if !self.supports_shift(ShiftDir::Left, 0, coeff) {
            return None;
        }
        u32::try_from(coeff).ok().map(u32::trailing_zeros)
    }

    pub(super) fn multiply_coefficients(self, lhs: i32, rhs: i32) -> Option<i32> {
        self.lower_coefficient(i128::from(lhs) * i128::from(rhs))
    }

    pub(super) fn in_place_scale_coeff(self, coeff: i32) -> Option<i32> {
        self.lower_coefficient(i128::from(coeff) + 1)
    }

    fn lower_coefficient(self, value: i128) -> Option<i32> {
        let semantics = super::super::symbolic::PolynomialSemantics::new(
            self.cell_bits,
            self.cell_sign,
            BF_C_TAPE_LEN,
        )?;
        // Preserve small positive coefficients for canonical shift recognition.
        // Overflowing IR coefficients can still have an exact modular lowering.
        i32::try_from(value)
            .ok()
            .or_else(|| i32::try_from(semantics.normalize_wide(value)?).ok())
    }

    pub(super) fn wrapping_modulus(self) -> Option<i128> {
        self.has_cells()
            .then(|| 1_i128.checked_shl(self.cell_bits))
            .flatten()
    }

    pub(super) fn wrap_coeff_to_i32(self, value: i128) -> Option<i32> {
        let semantics = super::super::symbolic::PolynomialSemantics::new(
            self.cell_bits,
            self.cell_sign,
            BF_C_TAPE_LEN,
        )?;
        i32::try_from(semantics.normalize_wide(value)?).ok()
    }

    pub(super) fn multiplicative_inverse(self, value: i32) -> Option<i128> {
        let modulus = self.wrapping_modulus()?;
        let value = i128::from(value).rem_euclid(modulus);
        if value == 0 {
            return None;
        }

        let (gcd, x, _) = extended_gcd(value, modulus);
        (gcd == 1).then_some(x.rem_euclid(modulus))
    }
}

fn extended_gcd(a: i128, b: i128) -> (i128, i128, i128) {
    let (mut old_r, mut r) = (a, b);
    let (mut old_s, mut s) = (1_i128, 0_i128);
    let (mut old_t, mut t) = (0_i128, 1_i128);

    while r != 0 {
        let quotient = old_r / r;
        (old_r, r) = (r, old_r - quotient * r);
        (old_s, s) = (s, old_s - quotient * s);
        (old_t, t) = (t, old_t - quotient * t);
    }

    (old_r, old_s, old_t)
}
