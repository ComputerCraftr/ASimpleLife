#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BfIr {
    MovePtr(isize),
    Add(i32),
    Input,
    Output,
    Loop(Vec<BfIr>),

    Clear,

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
        targets: Vec<(isize, i32)>,
    },

    Diverge,
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
                    stack.last_mut().unwrap().push(BfIr::Add(1));
                }
                '-' => {
                    self.pos += 1;
                    stack.last_mut().unwrap().push(BfIr::Add(-1));
                }
                '>' => {
                    self.pos += 1;
                    stack.last_mut().unwrap().push(BfIr::MovePtr(1));
                }
                '<' => {
                    self.pos += 1;
                    stack.last_mut().unwrap().push(BfIr::MovePtr(-1));
                }
                '.' => {
                    self.pos += 1;
                    stack.last_mut().unwrap().push(BfIr::Output);
                }
                ',' => {
                    self.pos += 1;
                    stack.last_mut().unwrap().push(BfIr::Input);
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
                    let body = stack.pop().unwrap();
                    stack.last_mut().unwrap().push(BfIr::Loop(body));
                }
                _ => unreachable!(),
            }
        }

        if stack.len() != 1 {
            Err("unmatched '['".to_string())
        } else {
            Ok(stack.pop().unwrap())
        }
    }
}
