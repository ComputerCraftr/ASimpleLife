use super::ir::{BfIr, ShiftDir, validate_canonical_ir};
use crate::RequiredExt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhysicalBfInstr {
    Add(i32),
    MovePtr(crate::bf::BfOffset),
    Clear,
    ClearAt(crate::bf::BfOffset),
    Distribute {
        targets: Vec<(crate::bf::BfOffset, i32)>,
        preserve_src: bool,
    },
    Affine {
        src: crate::bf::BfOffset,
        dst: crate::bf::BfOffset,
        coeff: i32,
        preserve_src: bool,
        set_dst: bool,
    },
    Shift {
        src: crate::bf::BfOffset,
        dst: crate::bf::BfOffset,
        amount: u32,
        dir: ShiftDir,
        preserve_src: bool,
        set_dst: bool,
    },
    Square {
        src: crate::bf::BfOffset,
        dst: crate::bf::BfOffset,
        preserve_src: bool,
        set_dst: bool,
    },
    MulAdd {
        lhs: crate::bf::BfOffset,
        rhs: crate::bf::BfOffset,
        dst: crate::bf::BfOffset,
        preserve_lhs: bool,
        preserve_rhs: bool,
        set_dst: bool,
    },
    Output,
    JumpIfZero(usize),
    JumpIfNonZero(usize),
    Diverge,
    Halt,
}

pub fn lower_bf_control_flow(program: &[BfIr]) -> Vec<PhysicalBfInstr> {
    validate_canonical_ir(program)
        .or_invariant("physical BF lowering requires canonical richer IR");

    let mut lowered = Vec::new();
    let mut stack = vec![(program, 0usize, None::<usize>)];
    while let Some((nodes, index, loop_start)) = stack.pop() {
        if index >= nodes.len() {
            if let Some(jz_pos) = loop_start {
                let body_start = jz_pos + 1;
                lowered.push(PhysicalBfInstr::JumpIfNonZero(body_start));
                let after_loop = lowered.len();
                lowered[jz_pos] = PhysicalBfInstr::JumpIfZero(after_loop);
            }
            continue;
        }

        stack.push((nodes, index + 1, loop_start));
        match &nodes[index] {
            BfIr::Add(n) => {
                if *n != 0 {
                    lowered.push(PhysicalBfInstr::Add(*n));
                }
            }
            BfIr::MovePtr(n) => {
                if *n != 0 {
                    lowered.push(PhysicalBfInstr::MovePtr(*n));
                }
            }
            BfIr::Input => crate::invariant_failure!("physical BF lowering does not support input"),
            BfIr::Output => lowered.push(PhysicalBfInstr::Output),
            BfIr::Clear => lowered.push(PhysicalBfInstr::Clear),
            BfIr::ClearAt { offset } => lowered.push(PhysicalBfInstr::ClearAt(*offset)),
            BfIr::Scan { stride } => {
                let guard = lowered.len();
                lowered.push(PhysicalBfInstr::JumpIfZero(usize::MAX));
                lowered.push(PhysicalBfInstr::MovePtr(*stride));
                lowered.push(PhysicalBfInstr::JumpIfNonZero(guard + 1));
                let after_scan = lowered.len();
                lowered[guard] = PhysicalBfInstr::JumpIfZero(after_scan);
            }
            BfIr::Diverge => lowered.push(PhysicalBfInstr::Diverge),
            BfIr::Distribute {
                targets,
                preserve_src,
            } => lowered.push(PhysicalBfInstr::Distribute {
                targets: targets.clone(),
                preserve_src: *preserve_src,
            }),
            BfIr::Affine {
                src,
                dst,
                coeff,
                preserve_src,
                set_dst,
            } => lowered.push(PhysicalBfInstr::Affine {
                src: *src,
                dst: *dst,
                coeff: *coeff,
                preserve_src: *preserve_src,
                set_dst: *set_dst,
            }),
            BfIr::Shift {
                src,
                dst,
                amount,
                dir,
                preserve_src,
                set_dst,
            } => lowered.push(PhysicalBfInstr::Shift {
                src: *src,
                dst: *dst,
                amount: *amount,
                dir: *dir,
                preserve_src: *preserve_src,
                set_dst: *set_dst,
            }),
            BfIr::Square {
                src,
                dst,
                preserve_src,
                set_dst,
            } => lowered.push(PhysicalBfInstr::Square {
                src: *src,
                dst: *dst,
                preserve_src: *preserve_src,
                set_dst: *set_dst,
            }),
            BfIr::MulAdd {
                lhs,
                rhs,
                dst,
                preserve_lhs,
                preserve_rhs,
                set_dst,
            } => lowered.push(PhysicalBfInstr::MulAdd {
                lhs: *lhs,
                rhs: *rhs,
                dst: *dst,
                preserve_lhs: *preserve_lhs,
                preserve_rhs: *preserve_rhs,
                set_dst: *set_dst,
            }),
            BfIr::Loop(body) => {
                let jz_pos = lowered.len();
                lowered.push(PhysicalBfInstr::JumpIfZero(usize::MAX));
                stack.push((body, 0, Some(jz_pos)));
            }
        }
    }
    lowered.push(PhysicalBfInstr::Halt);
    lowered
}

pub fn expand_distribute_to_primitive(program: &[BfIr]) -> Vec<BfIr> {
    let mut out = Vec::new();
    let mut stack = vec![(program, 0usize, None::<Vec<BfIr>>)];
    while let Some((nodes, index, loop_body)) = stack.pop() {
        if index >= nodes.len() {
            if let Some(body) = loop_body {
                if let Some((_, _, parent_loop_body)) = stack.last_mut() {
                    parent_loop_body
                        .as_mut()
                        .or_invariant("parent loop body must exist")
                        .push(BfIr::Loop(body));
                } else {
                    out.extend(body);
                }
            }
            continue;
        }

        stack.push((nodes, index + 1, loop_body));
        let target = if let Some((_, _, parent_loop_body)) = stack.last_mut() {
            parent_loop_body.as_mut()
        } else {
            None
        };

        match &nodes[index] {
            BfIr::Loop(body) => stack.push((body, 0, Some(Vec::new()))),
            BfIr::Distribute {
                targets,
                preserve_src,
            } => {
                let push_target = |dst: &mut Vec<BfIr>| {
                    if *preserve_src {
                        dst.push(nodes[index].clone());
                        return;
                    }
                    if targets.is_empty() {
                        dst.push(BfIr::Clear);
                        return;
                    }
                    let mut body = Vec::new();
                    body.push(BfIr::Add(-1));
                    for &(offset, coeff) in targets {
                        if coeff == 0 {
                            continue;
                        }
                        if offset != 0 {
                            body.push(BfIr::MovePtr(offset));
                        }
                        body.push(BfIr::Add(coeff));
                        if offset != 0 {
                            body.push(BfIr::MovePtr(-offset));
                        }
                    }
                    dst.push(BfIr::Loop(body));
                };
                if let Some(dst) = target {
                    push_target(dst);
                } else {
                    push_target(&mut out);
                }
            }
            other => {
                if let Some(dst) = target {
                    dst.push(other.clone());
                } else {
                    out.push(other.clone());
                }
            }
        }
    }
    out
}
