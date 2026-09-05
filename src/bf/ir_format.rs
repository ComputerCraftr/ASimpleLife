//! Backend-independent text formatting for structured Brainfuck IR.

use super::ir::BfIr;

fn indent(n: usize) -> String {
    " ".repeat(n)
}

pub fn format_ir(program: &[BfIr]) -> String {
    enum Frame<'a> {
        Seq {
            nodes: &'a [BfIr],
            index: usize,
            indent: usize,
        },
        Close {
            indent: usize,
        },
    }

    let mut out = String::new();
    let mut stack = vec![Frame::Seq {
        nodes: program,
        index: 0,
        indent: 0,
    }];

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Seq {
                nodes,
                mut index,
                indent: ind,
            } => {
                if index >= nodes.len() {
                    continue;
                }
                let pad = indent(ind);
                let node = &nodes[index];
                index += 1;
                stack.push(Frame::Seq {
                    nodes,
                    index,
                    indent: ind,
                });
                match node {
                    BfIr::MovePtr(n) => out.push_str(&format!("{pad}MovePtr({n})\n")),
                    BfIr::Add(n) => out.push_str(&format!("{pad}Add({n})\n")),
                    BfIr::Input => out.push_str(&format!("{pad}Input\n")),
                    BfIr::Output => out.push_str(&format!("{pad}Output\n")),
                    BfIr::Clear => out.push_str(&format!("{pad}Clear\n")),
                    BfIr::ClearAt { offset } => {
                        out.push_str(&format!("{pad}ClearAt({offset})\n"));
                    }
                    BfIr::Scan { stride } => {
                        out.push_str(&format!("{pad}Scan {{ stride: {stride} }}\n"))
                    }
                    BfIr::Shift {
                        src,
                        dst,
                        amount,
                        dir,
                        preserve_src,
                        set_dst,
                    } => out.push_str(&format!(
                        "{pad}Shift {{ src: {src}, dst: {dst}, amount: {amount}, dir: {dir:?}, preserve_src: {preserve_src}, set_dst: {set_dst} }}\n"
                    )),
                    BfIr::Affine {
                        src,
                        dst,
                        coeff,
                        preserve_src,
                        set_dst,
                    } => out.push_str(&format!(
                        "{pad}Affine {{ src: {src}, dst: {dst}, coeff: {coeff}, preserve_src: {preserve_src}, set_dst: {set_dst} }}\n"
                    )),
                    BfIr::Square {
                        src,
                        dst,
                        preserve_src,
                        set_dst,
                    } => out.push_str(&format!(
                        "{pad}Square {{ src: {src}, dst: {dst}, preserve_src: {preserve_src}, set_dst: {set_dst} }}\n"
                    )),
                    BfIr::MulAdd {
                        lhs,
                        rhs,
                        dst,
                        preserve_lhs,
                        preserve_rhs,
                        set_dst,
                    } => out.push_str(&format!(
                        "{pad}MulAdd {{ lhs: {lhs}, rhs: {rhs}, dst: {dst}, preserve_lhs: {preserve_lhs}, preserve_rhs: {preserve_rhs}, set_dst: {set_dst} }}\n"
                    )),
                    BfIr::Diverge => out.push_str(&format!("{pad}Diverge\n")),
                    BfIr::Distribute {
                        targets,
                        preserve_src,
                    } => out.push_str(&format!(
                        "{pad}Distribute {{ targets: {targets:?}, preserve_src: {preserve_src} }}\n"
                    )),
                    BfIr::Loop(body) => {
                        out.push_str(&format!("{pad}Loop {{\n"));
                        stack.push(Frame::Close { indent: ind });
                        stack.push(Frame::Seq {
                            nodes: body,
                            index: 0,
                            indent: ind + 2,
                        });
                    }
                }
            }
            Frame::Close { indent: ind } => out.push_str(&format!(
                "{}}}
",
                indent(ind)
            )),
        }
    }
    out
}
