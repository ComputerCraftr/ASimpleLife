use crate::RequiredExt;
use serde::Serialize;
use std::fmt::Write;

use super::ir::{BfIr, ShiftDir};
use super::optimizer::{CellSign, CodegenOpts, IoMode};

pub(crate) const DEFAULT_IR_MAX_LINES: usize = 80;
pub(crate) const DEFAULT_IR_MAX_DEPTH: usize = 8;
const IR_REPORT_VERSION: &str = "bf-ir-report/v3";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IrOutputFormat {
    Text,
    Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IrSectionSelection {
    Parsed,
    Optimized,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IrRenderOpts {
    pub format: IrOutputFormat,
    pub section: IrSectionSelection,
    pub max_lines: usize,
    pub max_depth: usize,
    pub no_elide: bool,
}

impl Default for IrRenderOpts {
    fn default() -> Self {
        Self {
            format: IrOutputFormat::Text,
            section: IrSectionSelection::Both,
            max_lines: DEFAULT_IR_MAX_LINES,
            max_depth: DEFAULT_IR_MAX_DEPTH,
            no_elide: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct IrReport {
    pub version: &'static str,
    pub opts: IrReportOpts,
    pub summary: IrReportSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parsed: Option<IrSectionReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimized: Option<IrSectionReport>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct IrReportOpts {
    pub io_mode: &'static str,
    pub cell_bits: u32,
    pub input_bits: Option<u32>,
    pub output_bits: Option<u32>,
    pub cell_sign: &'static str,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct IrReportSummary {
    pub source_bytes: usize,
    pub parsed_node_count: usize,
    pub optimized_node_count: usize,
    pub removed_nodes: usize,
    pub reduction_percent: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct IrSectionReport {
    pub node_count: usize,
    pub max_depth: usize,
    pub truncated: bool,
    pub omitted_nodes: usize,
    pub nodes: Vec<IrNodeView>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum IrNodeView {
    MovePtr {
        amount: crate::bf::BfOffset,
    },
    Add {
        amount: i32,
    },
    Input,
    Output,
    Clear,
    ClearAt {
        offset: crate::bf::BfOffset,
    },
    Scan {
        stride: crate::bf::BfOffset,
    },
    Distribute {
        targets: Vec<(crate::bf::BfOffset, i32)>,
        preserve_src: bool,
    },
    Shift {
        src: crate::bf::BfOffset,
        dst: crate::bf::BfOffset,
        amount: u32,
        dir: &'static str,
        preserve_src: bool,
        set_dst: bool,
    },
    Affine {
        src: crate::bf::BfOffset,
        dst: crate::bf::BfOffset,
        coeff: i32,
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
    Diverge,
    Loop {
        truncated: bool,
        omitted_nodes: usize,
        body: Vec<IrNodeView>,
    },
    Elided {
        omitted_nodes: usize,
    },
}

pub(crate) fn build_ir_report(
    src: &str,
    parsed: &[BfIr],
    optimized: &[BfIr],
    codegen_opts: CodegenOpts,
    render_opts: IrRenderOpts,
) -> IrReport {
    let (parsed_count, _) = ir_stats(parsed);
    let (optimized_count, _) = ir_stats(optimized);
    let removed_nodes = parsed_count.saturating_sub(optimized_count);
    let reduction_percent = if parsed_count == 0 {
        0.0
    } else {
        (removed_nodes as f64 * 100.0) / parsed_count as f64
    };

    IrReport {
        version: IR_REPORT_VERSION,
        opts: IrReportOpts::from(codegen_opts),
        summary: IrReportSummary {
            source_bytes: src.len(),
            parsed_node_count: parsed_count,
            optimized_node_count: optimized_count,
            removed_nodes,
            reduction_percent,
        },
        parsed: match render_opts.section {
            IrSectionSelection::Parsed | IrSectionSelection::Both => {
                Some(build_section_report(parsed, render_opts))
            }
            IrSectionSelection::Optimized => None,
        },
        optimized: match render_opts.section {
            IrSectionSelection::Optimized | IrSectionSelection::Both => {
                Some(build_section_report(optimized, render_opts))
            }
            IrSectionSelection::Parsed => None,
        },
    }
}

pub(crate) fn render_ir_text(report: &IrReport) -> String {
    let mut out = String::new();
    out.push_str("IR Report\n");
    let _ = writeln!(
        out,
        "summary: source_bytes={} parsed_nodes={} optimized_nodes={} removed_nodes={} reduction_percent={:.2}\n",
        report.summary.source_bytes,
        report.summary.parsed_node_count,
        report.summary.optimized_node_count,
        report.summary.removed_nodes,
        report.summary.reduction_percent
    );
    let _ = write!(
        out,
        "opts: io={} cell_bits={} signed={} input_bits=",
        report.opts.io_mode, report.opts.cell_bits, report.opts.cell_sign,
    );
    match report.opts.input_bits {
        Some(bits) => {
            let _ = write!(out, "{bits}");
        }
        None => out.push_str("none"),
    }
    out.push_str(" output_bits=");
    match report.opts.output_bits {
        Some(bits) => {
            let _ = writeln!(out, "{bits}");
        }
        None => out.push_str("none\n"),
    }
    if let Some(parsed) = &report.parsed {
        push_section_text(&mut out, "Parsed IR", parsed);
    }
    if let Some(optimized) = &report.optimized {
        push_section_text(&mut out, "Optimized IR", optimized);
    }
    out
}

pub(crate) fn render_ir_json(report: &IrReport) -> String {
    serde_json::to_string_pretty(report).or_invariant("IR report JSON serialization must succeed")
}

fn push_section_text(out: &mut String, title: &str, section: &IrSectionReport) {
    out.push_str(&format!(
        "=== {title} ===\nmeta: node_count={} max_depth={} truncated={} omitted_nodes={}\n",
        section.node_count, section.max_depth, section.truncated, section.omitted_nodes
    ));
    push_node_lines(out, &section.nodes, 0);
}

fn push_node_lines(out: &mut String, nodes: &[IrNodeView], indent: usize) {
    enum Frame<'a> {
        Nodes {
            nodes: &'a [IrNodeView],
            index: usize,
            indent: usize,
        },
        CloseLoop {
            indent: usize,
        },
        Elided {
            indent: usize,
            omitted_nodes: usize,
        },
    }

    let mut stack = vec![Frame::Nodes {
        nodes,
        index: 0,
        indent,
    }];
    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Nodes {
                nodes,
                index,
                indent,
            } => {
                if index >= nodes.len() {
                    continue;
                }
                stack.push(Frame::Nodes {
                    nodes,
                    index: index + 1,
                    indent,
                });
                let pad = " ".repeat(indent);
                match &nodes[index] {
                    IrNodeView::MovePtr { amount } => {
                        out.push_str(&format!("{pad}MovePtr({amount})\n"))
                    }
                    IrNodeView::Add { amount } => out.push_str(&format!("{pad}Add({amount})\n")),
                    IrNodeView::Input => out.push_str(&format!("{pad}Input\n")),
                    IrNodeView::Output => out.push_str(&format!("{pad}Output\n")),
                    IrNodeView::Clear => out.push_str(&format!("{pad}Clear\n")),
                    IrNodeView::ClearAt { offset } => {
                        out.push_str(&format!("{pad}ClearAt({offset})\n"));
                    }
                    IrNodeView::Scan { stride } => {
                        out.push_str(&format!("{pad}Scan {{ stride: {stride} }}\n"))
                    }
                    IrNodeView::Distribute {
                        targets,
                        preserve_src,
                    } => out.push_str(&format!(
                        "{pad}Distribute {{ targets: {targets:?}, preserve_src: {preserve_src} }}\n"
                    )),
                    IrNodeView::Shift {
                        src,
                        dst,
                        amount,
                        dir,
                        preserve_src,
                        set_dst,
                    } => out.push_str(&format!(
                        "{pad}Shift {{ src: {src}, dst: {dst}, amount: {amount}, dir: {dir}, preserve_src: {preserve_src}, set_dst: {set_dst} }}\n"
                    )),
                    IrNodeView::Affine {
                        src,
                        dst,
                        coeff,
                        preserve_src,
                        set_dst,
                    } => out.push_str(&format!(
                        "{pad}Affine {{ src: {src}, dst: {dst}, coeff: {coeff}, preserve_src: {preserve_src}, set_dst: {set_dst} }}\n"
                    )),
                    IrNodeView::Square {
                        src,
                        dst,
                        preserve_src,
                        set_dst,
                    } => out.push_str(&format!(
                        "{pad}Square {{ src: {src}, dst: {dst}, preserve_src: {preserve_src}, set_dst: {set_dst} }}\n"
                    )),
                    IrNodeView::MulAdd {
                        lhs,
                        rhs,
                        dst,
                        preserve_lhs,
                        preserve_rhs,
                        set_dst,
                    } => out.push_str(&format!(
                        "{pad}MulAdd {{ lhs: {lhs}, rhs: {rhs}, dst: {dst}, preserve_lhs: {preserve_lhs}, preserve_rhs: {preserve_rhs}, set_dst: {set_dst} }}\n"
                    )),
                    IrNodeView::Diverge => out.push_str(&format!("{pad}Diverge\n")),
                    IrNodeView::Elided { omitted_nodes } => {
                        out.push_str(&format!("{pad}... <{omitted_nodes} nodes omitted>\n"))
                    }
                    IrNodeView::Loop {
                        truncated,
                        omitted_nodes,
                        body,
                    } => {
                        out.push_str(&format!("{pad}Loop {{\n"));
                        stack.push(Frame::CloseLoop { indent });
                        if *truncated && *omitted_nodes > 0 {
                            stack.push(Frame::Elided {
                                indent: indent + 2,
                                omitted_nodes: *omitted_nodes,
                            });
                        }
                        if !body.is_empty() {
                            stack.push(Frame::Nodes {
                                nodes: body,
                                index: 0,
                                indent: indent + 2,
                            });
                        }
                    }
                }
            }
            Frame::CloseLoop { indent } => {
                out.push_str(&format!("{}}}\n", " ".repeat(indent)));
            }
            Frame::Elided {
                indent,
                omitted_nodes,
            } => {
                out.push_str(&format!(
                    "{}... <{} nodes omitted>\n",
                    " ".repeat(indent),
                    omitted_nodes
                ));
            }
        }
    }
}

fn build_section_report(program: &[BfIr], render_opts: IrRenderOpts) -> IrSectionReport {
    let (node_count, max_depth) = ir_stats(program);
    if render_opts.no_elide {
        return IrSectionReport {
            node_count,
            max_depth,
            truncated: false,
            omitted_nodes: 0,
            nodes: build_view_forest(program, None),
        };
    }

    let depth_limited = build_view_forest(program, Some(render_opts.max_depth));
    let depth_omitted_nodes = omitted_from_truncation(&depth_limited);
    let line_limited = apply_line_limit(depth_limited, render_opts.max_lines);

    IrSectionReport {
        node_count,
        max_depth,
        truncated: line_limited.truncated || depth_omitted_nodes > 0,
        omitted_nodes: line_limited.omitted_nodes + depth_omitted_nodes,
        nodes: line_limited.nodes,
    }
}

struct LineLimitedNodes {
    nodes: Vec<IrNodeView>,
    truncated: bool,
    omitted_nodes: usize,
}

fn apply_line_limit(nodes: Vec<IrNodeView>, max_lines: usize) -> LineLimitedNodes {
    if max_lines == 0 {
        let omitted_nodes = omitted_view_nodes(&nodes);
        return LineLimitedNodes {
            nodes: vec![IrNodeView::Elided { omitted_nodes }],
            truncated: true,
            omitted_nodes,
        };
    }

    let mut remaining_nodes = omitted_view_nodes(&nodes);
    let mut kept = Vec::new();
    let mut used = 0usize;
    let mut iter = nodes.into_iter().peekable();
    while let Some(node) = iter.next() {
        let line_cost = node_line_count(&node);
        let node_nodes = omitted_view_nodes(std::slice::from_ref(&node));
        let reserve_elision = usize::from(iter.peek().is_some());
        if used + line_cost + reserve_elision > max_lines {
            kept.push(IrNodeView::Elided {
                omitted_nodes: remaining_nodes,
            });
            return LineLimitedNodes {
                nodes: kept,
                truncated: true,
                omitted_nodes: remaining_nodes,
            };
        }
        remaining_nodes -= node_nodes;
        used += line_cost;
        kept.push(node);
    }

    LineLimitedNodes {
        nodes: kept,
        truncated: false,
        omitted_nodes: 0,
    }
}

fn build_view_forest(nodes: &[BfIr], max_depth: Option<usize>) -> Vec<IrNodeView> {
    enum Frame<'a> {
        Node {
            node: &'a BfIr,
            depth: usize,
        },
        FinishLoop {
            start: usize,
            truncated: bool,
            omitted_nodes: usize,
        },
    }

    let mut frames = Vec::new();
    let mut built = Vec::new();
    for node in nodes.iter().rev() {
        frames.push(Frame::Node { node, depth: 0 });
    }

    while let Some(frame) = frames.pop() {
        match frame {
            Frame::Node { node, depth } => match node {
                BfIr::MovePtr(amount) => built.push(IrNodeView::MovePtr { amount: *amount }),
                BfIr::Add(amount) => built.push(IrNodeView::Add { amount: *amount }),
                BfIr::Input => built.push(IrNodeView::Input),
                BfIr::Output => built.push(IrNodeView::Output),
                BfIr::Clear => built.push(IrNodeView::Clear),
                BfIr::ClearAt { offset } => {
                    built.push(IrNodeView::ClearAt { offset: *offset });
                }
                BfIr::Scan { stride } => built.push(IrNodeView::Scan { stride: *stride }),
                BfIr::Distribute {
                    targets,
                    preserve_src,
                } => built.push(IrNodeView::Distribute {
                    targets: targets.clone(),
                    preserve_src: *preserve_src,
                }),
                BfIr::Shift {
                    src,
                    dst,
                    amount,
                    dir,
                    preserve_src,
                    set_dst,
                } => built.push(IrNodeView::Shift {
                    src: *src,
                    dst: *dst,
                    amount: *amount,
                    dir: shift_dir_name(*dir),
                    preserve_src: *preserve_src,
                    set_dst: *set_dst,
                }),
                BfIr::Affine {
                    src,
                    dst,
                    coeff,
                    preserve_src,
                    set_dst,
                } => built.push(IrNodeView::Affine {
                    src: *src,
                    dst: *dst,
                    coeff: *coeff,
                    preserve_src: *preserve_src,
                    set_dst: *set_dst,
                }),
                BfIr::Square {
                    src,
                    dst,
                    preserve_src,
                    set_dst,
                } => built.push(IrNodeView::Square {
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
                } => built.push(IrNodeView::MulAdd {
                    lhs: *lhs,
                    rhs: *rhs,
                    dst: *dst,
                    preserve_lhs: *preserve_lhs,
                    preserve_rhs: *preserve_rhs,
                    set_dst: *set_dst,
                }),
                BfIr::Diverge => built.push(IrNodeView::Diverge),
                BfIr::Loop(body) => {
                    if max_depth.is_some_and(|limit| depth >= limit) {
                        built.push(IrNodeView::Loop {
                            truncated: !body.is_empty(),
                            omitted_nodes: ir_stats(body).0,
                            body: Vec::new(),
                        });
                    } else {
                        let start = built.len();
                        frames.push(Frame::FinishLoop {
                            start,
                            truncated: false,
                            omitted_nodes: 0,
                        });
                        for child in body.iter().rev() {
                            frames.push(Frame::Node {
                                node: child,
                                depth: depth + 1,
                            });
                        }
                    }
                }
            },
            Frame::FinishLoop {
                start,
                truncated,
                omitted_nodes,
            } => {
                let body = built.split_off(start);
                built.push(IrNodeView::Loop {
                    truncated,
                    omitted_nodes,
                    body,
                });
            }
        }
    }

    built
}

fn ir_stats(nodes: &[BfIr]) -> (usize, usize) {
    let mut count = 0usize;
    let mut max_depth = 0usize;
    let mut stack = Vec::new();
    for node in nodes.iter().rev() {
        stack.push((node, 1usize));
    }

    while let Some((node, depth)) = stack.pop() {
        count += 1;
        max_depth = max_depth.max(depth);
        if let BfIr::Loop(body) = node {
            for child in body.iter().rev() {
                stack.push((child, depth + 1));
            }
        }
    }

    (count, max_depth)
}

fn omitted_view_nodes(nodes: &[IrNodeView]) -> usize {
    let mut total = 0usize;
    let mut stack = vec![nodes];
    while let Some(cur) = stack.pop() {
        for node in cur {
            match node {
                IrNodeView::Loop { body, .. } => {
                    total += 1;
                    stack.push(body);
                }
                IrNodeView::Elided { omitted_nodes } => total += *omitted_nodes,
                _ => total += 1,
            }
        }
    }
    total
}

fn omitted_from_truncation(nodes: &[IrNodeView]) -> usize {
    let mut total = 0usize;
    let mut stack = vec![nodes];
    while let Some(cur) = stack.pop() {
        for node in cur {
            if let IrNodeView::Loop {
                truncated,
                omitted_nodes,
                body,
            } = node
            {
                if *truncated {
                    total += *omitted_nodes;
                }
                stack.push(body);
            }
        }
    }
    total
}

fn node_line_count(node: &IrNodeView) -> usize {
    match node {
        IrNodeView::Loop {
            truncated,
            omitted_nodes,
            body,
        } => {
            let mut lines = 2;
            if !body.is_empty() {
                lines += body.iter().map(node_line_count).sum::<usize>();
            }
            if *truncated && *omitted_nodes > 0 {
                lines += 1;
            }
            lines
        }
        _ => 1,
    }
}

fn shift_dir_name(dir: ShiftDir) -> &'static str {
    match dir {
        ShiftDir::Left => "Left",
        ShiftDir::Right => "Right",
    }
}

impl From<CodegenOpts> for IrReportOpts {
    fn from(value: CodegenOpts) -> Self {
        Self {
            io_mode: match value.io_mode {
                IoMode::Char => "char",
                IoMode::Number => "number",
            },
            cell_bits: value.cell_bits,
            input_bits: value.input_bits,
            output_bits: value.output_bits,
            cell_sign: match value.cell_sign {
                CellSign::Signed => "signed",
                CellSign::Unsigned => "unsigned",
            },
        }
    }
}
