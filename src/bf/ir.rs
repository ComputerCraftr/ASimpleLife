use crate::RequiredExt;

/// A pointer-relative BF cell offset with target-independent serialized width.
pub type BfOffset = i64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShiftDir {
    Left,
    Right,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BfIr {
    MovePtr(BfOffset),
    Add(i32),
    Input,
    Output,
    Loop(Vec<BfIr>),

    Clear,
    ClearAt {
        offset: BfOffset,
    },

    /// Move by `stride` until the current cell is zero.
    Scan {
        stride: BfOffset,
    },

    /// Consume the current cell and add signed multiples of its original value
    /// into target cells relative to the current pointer, then clear source.
    ///
    /// Example:
    ///   Distribute { targets: vec![(1, 1), (2, 2)] }
    ///
    /// means:
    ///   cell[ptr + 1] += x
    ///   cell[ptr + 2] += 2*x
    ///   cell[ptr] = 0
    ///
    /// where x is the original value of cell[ptr].
    Distribute {
        targets: Vec<(BfOffset, i32)>,
        preserve_src: bool,
    },

    Shift {
        src: BfOffset,
        dst: BfOffset,
        amount: u32,
        dir: ShiftDir,
        preserve_src: bool,
        set_dst: bool,
    },

    Affine {
        src: BfOffset,
        dst: BfOffset,
        coeff: i32,
        preserve_src: bool,
        set_dst: bool,
    },

    Square {
        src: BfOffset,
        dst: BfOffset,
        preserve_src: bool,
        set_dst: bool,
    },

    /// Multiply the original values of `lhs` and `rhs`, then add or set the
    /// result at `dst`.
    ///
    /// Canonical richer IR forbids `lhs == rhs`; that form must be normalized
    /// to `Square` before backend lowering.
    MulAdd {
        lhs: BfOffset,
        rhs: BfOffset,
        dst: BfOffset,
        preserve_lhs: bool,
        preserve_rhs: bool,
        set_dst: bool,
    },

    Diverge,
}

pub(crate) fn validate_canonical_ir(program: &[BfIr]) -> Result<(), String> {
    let mut stack = vec![program];
    while let Some(nodes) = stack.pop() {
        for node in nodes {
            match node {
                BfIr::Loop(body) => stack.push(body),
                BfIr::ClearAt { offset: 0 } => {
                    return Err(
                        "non-canonical richer IR: ClearAt at offset 0 must be normalized to Clear"
                            .to_string(),
                    );
                }
                BfIr::MulAdd { lhs, rhs, .. } if lhs == rhs => {
                    return Err(format!(
                        "non-canonical richer IR: MulAdd with aliased sources (lhs == rhs == {lhs}) must be normalized to Square"
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

pub struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    pub fn new(src: &str) -> Self {
        let chars = src
            .chars()
            .filter(|c| matches!(c, '>' | '<' | '+' | '-' | '.' | ',' | '[' | ']'))
            .collect();
        Self { chars, pos: 0 }
    }

    pub fn parse(mut self) -> Result<Vec<BfIr>, String> {
        let mut stack: Vec<Vec<BfIr>> = vec![Vec::new()];

        while let Some(&ch) = self.chars.get(self.pos) {
            match ch {
                '+' => {
                    self.pos += 1;
                    stack
                        .last_mut()
                        .or_invariant("required value")
                        .push(BfIr::Add(1));
                }
                '-' => {
                    self.pos += 1;
                    stack
                        .last_mut()
                        .or_invariant("required value")
                        .push(BfIr::Add(-1));
                }
                '>' => {
                    self.pos += 1;
                    stack
                        .last_mut()
                        .or_invariant("required value")
                        .push(BfIr::MovePtr(1));
                }
                '<' => {
                    self.pos += 1;
                    stack
                        .last_mut()
                        .or_invariant("required value")
                        .push(BfIr::MovePtr(-1));
                }
                '.' => {
                    self.pos += 1;
                    stack
                        .last_mut()
                        .or_invariant("required value")
                        .push(BfIr::Output);
                }
                ',' => {
                    self.pos += 1;
                    stack
                        .last_mut()
                        .or_invariant("required value")
                        .push(BfIr::Input);
                }
                '[' => {
                    self.pos += 1;
                    stack.push(Vec::new());
                }
                ']' => {
                    if stack.len() == 1 {
                        return Err(format!(
                            "unmatched ']' at filtered token index {}",
                            self.pos
                        ));
                    }
                    self.pos += 1;
                    let body = stack.pop().or_invariant("required value");
                    stack
                        .last_mut()
                        .or_invariant("required value")
                        .push(BfIr::Loop(body));
                }
                _ => crate::invariant_failure!(),
            }
        }

        if stack.len() != 1 {
            Err("unmatched '['".to_string())
        } else {
            Ok(stack.pop().or_invariant("required value"))
        }
    }
}
