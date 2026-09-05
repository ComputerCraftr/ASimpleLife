use super::*;

pub(crate) fn interpret_for_tests(
    program: &[BfIr],
    opts: CodegenOpts,
) -> Result<(Vec<i64>, usize), BfEvalError> {
    opts.validate().map_err(|_| BfEvalError::InvalidOptions)?;
    #[doc = "source-policy: checked-narrowing-boundary"]
    fn wrap(v: i128, bits: u32, sign: CellSign) -> i64 {
        if bits == 0 {
            return 0;
        }
        let modulus = 1_i128 << bits;
        let raw = v.rem_euclid(modulus);
        let value = if sign == CellSign::Signed && raw >= modulus / 2 {
            raw - modulus
        } else {
            raw
        };
        i64::try_from(value).or_invariant("validated cell width fits the oracle storage")
    }

    let cell_bits = opts.cell_bits;
    let mut tape = vec![0_i64; C_TAPE_LEN];
    let mut ptr = 0_usize;
    let mut stack: Vec<(&[BfIr], usize)> = vec![(program, 0)];
    let mut steps = 0_u64;

    while let Some((nodes, index)) = stack.last_mut() {
        if *index >= nodes.len() {
            stack.pop();
            continue;
        }
        if steps >= BF_TEST_STEP_BUDGET {
            return Err(BfEvalError::StepBudgetExceeded);
        }
        steps += 1;

        let node = &nodes[*index];
        *index += 1;

        match node {
            BfIr::MovePtr(n) => {
                ptr = crate::bf::tape::wrapped_index(ptr, *n, tape.len());
            }
            BfIr::Add(n) => {
                tape[ptr] = wrap(
                    i128::from(tape[ptr]) + i128::from(*n),
                    cell_bits,
                    opts.cell_sign,
                );
            }
            BfIr::Input | BfIr::Output => {}
            BfIr::Clear => tape[ptr] = 0,
            BfIr::ClearAt { offset } => {
                let target = crate::bf::tape::wrapped_index(ptr, *offset, tape.len());
                tape[target] = 0;
            }
            BfIr::Scan { stride } => {
                while tape[ptr] != 0 {
                    if steps >= BF_TEST_STEP_BUDGET {
                        return Err(BfEvalError::StepBudgetExceeded);
                    }
                    steps += 1;
                    ptr = crate::bf::tape::wrapped_index(ptr, *stride, tape.len());
                }
            }
            BfIr::Distribute {
                targets,
                preserve_src,
            } => {
                let v = tape[ptr];
                for &(offset, coeff) in targets {
                    let t = crate::bf::tape::wrapped_index(ptr, offset, tape.len());
                    tape[t] = wrap(
                        i128::from(tape[t]) + i128::from(v) * i128::from(coeff),
                        cell_bits,
                        opts.cell_sign,
                    );
                }
                if !preserve_src {
                    tape[ptr] = 0;
                }
            }
            BfIr::Affine {
                src,
                dst,
                coeff,
                preserve_src,
                set_dst,
            } => {
                let s = crate::bf::tape::wrapped_index(ptr, *src, tape.len());
                let d = crate::bf::tape::wrapped_index(ptr, *dst, tape.len());
                let src_v = tape[s];
                let base = if *set_dst { 0 } else { tape[d] };
                let dst_next = wrap(
                    i128::from(base) + i128::from(src_v) * i128::from(*coeff),
                    cell_bits,
                    opts.cell_sign,
                );
                if !preserve_src && s != d {
                    tape[s] = 0;
                }
                tape[d] = dst_next;
            }
            BfIr::Shift {
                src,
                dst,
                amount,
                dir,
                preserve_src,
                set_dst,
            } => {
                let s = crate::bf::tape::wrapped_index(ptr, *src, tape.len());
                let d = crate::bf::tape::wrapped_index(ptr, *dst, tape.len());
                let src_raw = tape[s].cast_unsigned() & ((1_u64 << cell_bits) - 1);
                let shifted_raw = match dir {
                    ShiftDir::Left => src_raw.checked_shl(*amount).unwrap_or(0),
                    ShiftDir::Right => src_raw.checked_shr(*amount).unwrap_or(0),
                };
                let shifted = wrap(i128::from(shifted_raw), cell_bits, opts.cell_sign);
                let base = if *set_dst { 0 } else { tape[d] };
                let dst_next = wrap(
                    i128::from(base) + i128::from(shifted),
                    cell_bits,
                    opts.cell_sign,
                );
                if !preserve_src && s != d {
                    tape[s] = 0;
                }
                tape[d] = dst_next;
            }
            BfIr::Square {
                src,
                dst,
                preserve_src,
                set_dst,
            } => {
                let s = crate::bf::tape::wrapped_index(ptr, *src, tape.len());
                let d = crate::bf::tape::wrapped_index(ptr, *dst, tape.len());
                let src_v = tape[s];
                let base = if *set_dst { 0 } else { tape[d] };
                let dst_next = wrap(
                    i128::from(base) + i128::from(src_v) * i128::from(src_v),
                    cell_bits,
                    opts.cell_sign,
                );
                if !preserve_src && s != d {
                    tape[s] = 0;
                }
                tape[d] = dst_next;
            }
            BfIr::MulAdd {
                lhs,
                rhs,
                dst,
                preserve_lhs,
                preserve_rhs,
                set_dst,
            } => {
                let l = crate::bf::tape::wrapped_index(ptr, *lhs, tape.len());
                let r = crate::bf::tape::wrapped_index(ptr, *rhs, tape.len());
                let d = crate::bf::tape::wrapped_index(ptr, *dst, tape.len());
                let lhs_v = tape[l];
                let rhs_v = tape[r];
                let base = if *set_dst { 0 } else { tape[d] };
                let dst_next = wrap(
                    i128::from(base) + i128::from(lhs_v) * i128::from(rhs_v),
                    cell_bits,
                    opts.cell_sign,
                );
                if !preserve_lhs && l != d {
                    tape[l] = 0;
                }
                if !preserve_rhs && r != d && (r != l || *preserve_lhs) {
                    tape[r] = 0;
                }
                tape[d] = dst_next;
            }
            BfIr::Diverge => return Err(BfEvalError::DivergenceDetected),
            BfIr::Loop(body) => {
                if tape[ptr] != 0 {
                    stack.last_mut().or_invariant("required value").1 -= 1;
                    stack.push((body, 0));
                }
            }
        }
    }
    Ok((tape, ptr))
}
