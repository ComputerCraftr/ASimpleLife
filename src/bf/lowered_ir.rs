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
