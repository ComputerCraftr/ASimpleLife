use super::ir::BfIr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhysicalBfInstr {
    Add(i32),
    MovePtr(isize),
    Clear,
    Output,
    JumpIfZero(usize),
    JumpIfNonZero(usize),
    Diverge,
    Halt,
}

pub fn lower_bf_control_flow(program: &[BfIr]) -> Vec<PhysicalBfInstr> {
    fn lower(nodes: &[BfIr], out: &mut Vec<PhysicalBfInstr>) {
        for node in nodes {
            match node {
                BfIr::Add(n) => {
                    if *n != 0 {
                        out.push(PhysicalBfInstr::Add(*n));
                    }
                }
                BfIr::MovePtr(n) => {
                    if *n != 0 {
                        out.push(PhysicalBfInstr::MovePtr(*n));
                    }
                }
                BfIr::Input => panic!("physical BF lowering does not support input"),
                BfIr::Output => out.push(PhysicalBfInstr::Output),
                BfIr::Clear => out.push(PhysicalBfInstr::Clear),
                BfIr::Diverge => out.push(PhysicalBfInstr::Diverge),
                BfIr::Distribute { targets: _ } => {
                    panic!("distribute must be expanded before physical BF lowering")
                }
                BfIr::Loop(body) => {
                    let jz_pos = out.len();
                    out.push(PhysicalBfInstr::JumpIfZero(usize::MAX));
                    lower(body, out);
                    let body_start = jz_pos + 1;
                    out.push(PhysicalBfInstr::JumpIfNonZero(body_start));
                    let after_loop = out.len();
                    out[jz_pos] = PhysicalBfInstr::JumpIfZero(after_loop);
                }
            }
        }
    }

    let mut lowered = Vec::new();
    lower(&expand_distribute_to_primitive(program), &mut lowered);
    lowered.push(PhysicalBfInstr::Halt);
    lowered
}

pub fn expand_distribute_to_primitive(program: &[BfIr]) -> Vec<BfIr> {
    fn expand(nodes: &[BfIr], out: &mut Vec<BfIr>) {
        for node in nodes {
            match node {
                BfIr::Loop(body) => {
                    let mut lowered_body = Vec::new();
                    expand(body, &mut lowered_body);
                    out.push(BfIr::Loop(lowered_body));
                }
                BfIr::Distribute { targets } => {
                    if targets.is_empty() {
                        out.push(BfIr::Clear);
                        continue;
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
                    out.push(BfIr::Loop(body));
                }
                other => out.push(other.clone()),
            }
        }
    }

    let mut out = Vec::new();
    expand(program, &mut out);
    out
}
