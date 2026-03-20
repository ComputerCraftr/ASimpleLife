use super::ir::BfIr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoweredBfInstr {
    Add(i32),
    MovePtr(isize),
    Input,
    Output,
    Clear,
    Distribute(Vec<(isize, i32)>),
    JumpIfZero(usize),
    JumpIfNonZero(usize),
    Diverge,
    Halt,
}

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

pub fn lower_bf_control_flow(program: &[BfIr]) -> Vec<LoweredBfInstr> {
    fn lower(nodes: &[BfIr], out: &mut Vec<LoweredBfInstr>) {
        for node in nodes {
            match node {
                BfIr::Add(n) => {
                    if *n != 0 {
                        out.push(LoweredBfInstr::Add(*n));
                    }
                }
                BfIr::MovePtr(n) => {
                    if *n != 0 {
                        out.push(LoweredBfInstr::MovePtr(*n));
                    }
                }
                BfIr::Input => out.push(LoweredBfInstr::Input),
                BfIr::Output => out.push(LoweredBfInstr::Output),
                BfIr::Clear => out.push(LoweredBfInstr::Clear),
                BfIr::Diverge => out.push(LoweredBfInstr::Diverge),
                BfIr::Distribute { targets } => {
                    out.push(LoweredBfInstr::Distribute(targets.clone()))
                }
                BfIr::Loop(body) => {
                    let jz_pos = out.len();
                    out.push(LoweredBfInstr::JumpIfZero(usize::MAX));
                    lower(body, out);
                    let body_start = jz_pos + 1;
                    out.push(LoweredBfInstr::JumpIfNonZero(body_start));
                    let after_loop = out.len();
                    out[jz_pos] = LoweredBfInstr::JumpIfZero(after_loop);
                }
            }
        }
    }

    let mut lowered = Vec::new();
    lower(program, &mut lowered);
    lowered.push(LoweredBfInstr::Halt);
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

pub fn lower_bf_physical_control_flow(program: &[BfIr]) -> Vec<PhysicalBfInstr> {
    lower_bf_control_flow(&expand_distribute_to_primitive(program))
        .into_iter()
        .map(|instr| match instr {
            LoweredBfInstr::Add(n) => PhysicalBfInstr::Add(n),
            LoweredBfInstr::MovePtr(n) => PhysicalBfInstr::MovePtr(n),
            LoweredBfInstr::Clear => PhysicalBfInstr::Clear,
            LoweredBfInstr::Output => PhysicalBfInstr::Output,
            LoweredBfInstr::JumpIfZero(target) => PhysicalBfInstr::JumpIfZero(target),
            LoweredBfInstr::JumpIfNonZero(target) => PhysicalBfInstr::JumpIfNonZero(target),
            LoweredBfInstr::Diverge => PhysicalBfInstr::Diverge,
            LoweredBfInstr::Halt => PhysicalBfInstr::Halt,
            LoweredBfInstr::Input => panic!("physical BF lowering does not support input"),
            LoweredBfInstr::Distribute(_) => {
                panic!("distribute must be expanded before physical BF lowering")
            }
        })
        .collect()
}
