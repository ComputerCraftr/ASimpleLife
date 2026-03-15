use std::io::{self, Read};

use super::c_backend::emit_c;
use super::c_super_backend::emit_c_super;
use super::ir::Parser;
use super::ir_report::{
    DEFAULT_IR_MAX_DEPTH, DEFAULT_IR_MAX_LINES, IrOutputFormat, IrRenderOpts, IrSectionSelection,
    build_ir_report, render_ir_json, render_ir_text,
};
use super::life_backend::{
    compile_to_life_circuit, serialize_life_circuit, serialize_life_circuit_hashlife,
};
use super::optimizer::{CellSign, CodegenOpts, IoMode, optimize_with_opts};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OutputMode {
    EmitIr,
    EmitC,
    EmitCSuper,
    EmitLife,
    EmitLifeHashLife,
}

fn print_help() {
    println!(
        "usage: bf_life [--emit-ir|--emit-c|--emit-c-super|--emit-life|--emit-life-hashlife] [opts] [-- <src>|<file>]"
    );
    println!("  --emit-ir      print parsed and optimized IR (default)");
    println!("  --emit-c       emit a C translation");
    println!("  --emit-c-super emit the symbolic-memo C backend");
    println!("  --emit-life    unavailable until the emitted grid independently executes BF");
    println!("  --emit-life-hashlife unavailable until the emitted grid independently executes BF");
    println!("opts: --cell-bits N  --io char|number  --signed-cells true|false");
    println!("ir opts: --emit-ir-format text|json  --emit-ir-section parsed|optimized|both");
    println!("         --emit-ir-max-lines N  --emit-ir-max-depth N  --emit-ir-no-elide");
}

pub fn run() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        std::process::exit(0);
    }

    let (mode, opts, ir_opts, src) = match read_input(&args) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    let parsed = match Parser::new(&src).parse() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("parse error: {e}");
            std::process::exit(1);
        }
    };

    if matches!(mode, OutputMode::EmitIr) {
        let optimized = optimize_with_opts(parsed.clone(), opts);
        let report = build_ir_report(&src, &parsed, &optimized, opts, ir_opts);
        match ir_opts.format {
            IrOutputFormat::Text => print!("{}", render_ir_text(&report)),
            IrOutputFormat::Json => print!("{}", render_ir_json(&report)),
        }
        return;
    }

    let optimized = optimize_with_opts(parsed, opts);
    match mode {
        OutputMode::EmitIr => crate::invariant_failure!(),
        OutputMode::EmitC => print!("{}", emit_c(&optimized, opts)),
        OutputMode::EmitCSuper => print!("{}", emit_c_super(&optimized, opts)),
        OutputMode::EmitLife | OutputMode::EmitLifeHashLife => {
            let artifact = match compile_to_life_circuit(&optimized, opts) {
                Ok(artifact) => artifact,
                Err(error) => {
                    eprintln!("error: {error}");
                    std::process::exit(1);
                }
            };
            let serialized = match mode {
                OutputMode::EmitLife => serialize_life_circuit(&artifact),
                OutputMode::EmitLifeHashLife => serialize_life_circuit_hashlife(&artifact),
                _ => crate::invariant_failure!(),
            };
            match serialized {
                Ok(output) => print!("{output}"),
                Err(error) => {
                    eprintln!("error: {error}");
                    std::process::exit(1);
                }
            }
        }
    }
}

pub(super) fn read_input(
    args: &[String],
) -> Result<(OutputMode, CodegenOpts, IrRenderOpts, String), String> {
    let mut mode = OutputMode::EmitIr;
    let mut rest = args;

    if let Some(first) = rest.first() {
        match first.as_str() {
            "--emit-c" => {
                mode = OutputMode::EmitC;
                rest = &rest[1..];
            }
            "--emit-c-super" => {
                mode = OutputMode::EmitCSuper;
                rest = &rest[1..];
            }
            "--emit-ir" => {
                mode = OutputMode::EmitIr;
                rest = &rest[1..];
            }
            "--emit-life" => {
                mode = OutputMode::EmitLife;
                rest = &rest[1..];
            }
            "--emit-life-hashlife" => {
                mode = OutputMode::EmitLifeHashLife;
                rest = &rest[1..];
            }
            _ => {}
        }
    }

    let (opts, ir_opts, rest) = parse_cli_opts(rest)?;

    if rest.is_empty() {
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("failed to read stdin: {e}"))?;
        return Ok((mode, opts, ir_opts, buf));
    }
    if rest[0] == "--" {
        return Ok((mode, opts, ir_opts, rest[1..].join(" ")));
    }
    if rest.len() == 1 {
        let arg = &rest[0];
        if std::path::Path::new(arg).exists() {
            let src =
                std::fs::read_to_string(arg).map_err(|e| format!("failed to read '{arg}': {e}"))?;
            return Ok((mode, opts, ir_opts, src));
        }
        return Ok((mode, opts, ir_opts, arg.clone()));
    }
    Ok((mode, opts, ir_opts, rest.join(" ")))
}

#[cfg(test)]
pub(super) fn parse_opts(args: &[String]) -> Result<(CodegenOpts, &[String]), String> {
    let (opts, _, rest) = parse_cli_opts(args)?;
    Ok((opts, rest))
}

fn parse_cli_opts(args: &[String]) -> Result<(CodegenOpts, IrRenderOpts, &[String]), String> {
    let mut opts = CodegenOpts {
        io_mode: IoMode::Char,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let mut ir_opts = IrRenderOpts::default();
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--emit-ir-format" => {
                let v = args
                    .get(i + 1)
                    .ok_or("missing value after --emit-ir-format")?;
                ir_opts.format = match v.as_str() {
                    "text" => IrOutputFormat::Text,
                    "json" => IrOutputFormat::Json,
                    other => {
                        return Err(format!(
                            "unsupported --emit-ir-format value '{other}'; expected 'text' or 'json'"
                        ));
                    }
                };
                i += 2;
            }
            "--emit-ir-section" => {
                let v = args
                    .get(i + 1)
                    .ok_or("missing value after --emit-ir-section")?;
                ir_opts.section = match v.as_str() {
                    "parsed" => IrSectionSelection::Parsed,
                    "optimized" => IrSectionSelection::Optimized,
                    "both" => IrSectionSelection::Both,
                    other => {
                        return Err(format!(
                            "unsupported --emit-ir-section value '{other}'; expected 'parsed', 'optimized', or 'both'"
                        ));
                    }
                };
                i += 2;
            }
            "--emit-ir-max-lines" => {
                let v = args
                    .get(i + 1)
                    .ok_or("missing value after --emit-ir-max-lines")?;
                ir_opts.max_lines = v
                    .parse()
                    .map_err(|_| format!("invalid --emit-ir-max-lines value '{v}'"))?;
                i += 2;
            }
            "--emit-ir-max-depth" => {
                let v = args
                    .get(i + 1)
                    .ok_or("missing value after --emit-ir-max-depth")?;
                ir_opts.max_depth = v
                    .parse()
                    .map_err(|_| format!("invalid --emit-ir-max-depth value '{v}'"))?;
                i += 2;
            }
            "--emit-ir-no-elide" => {
                ir_opts.no_elide = true;
                i += 1;
            }
            "--io" => {
                let v = args.get(i + 1).ok_or("missing value after --io")?;
                opts.io_mode = match v.as_str() {
                    "char" => IoMode::Char,
                    "number" => IoMode::Number,
                    other => {
                        return Err(format!(
                            "unsupported --io value '{other}'; expected 'char' or 'number'"
                        ));
                    }
                };
                i += 2;
            }
            "--cell-bits" => {
                let v = args.get(i + 1).ok_or("missing value after --cell-bits")?;
                let bits: u32 = v
                    .parse()
                    .map_err(|_| format!("invalid --cell-bits value '{v}'"))?;
                if bits > 63 {
                    return Err(format!(
                        "unsupported --cell-bits value '{bits}'; expected 0..=63"
                    ));
                }
                opts.cell_bits = bits;
                i += 2;
            }
            "--input-bits" => {
                let v = args.get(i + 1).ok_or("missing value after --input-bits")?;
                let bits: u32 = v
                    .parse()
                    .map_err(|_| format!("invalid --input-bits value '{v}'"))?;
                if bits > 63 {
                    return Err(format!(
                        "unsupported --input-bits value '{bits}'; expected 0..=63"
                    ));
                }
                opts.input_bits = Some(bits);
                i += 2;
            }
            "--output-bits" => {
                let v = args.get(i + 1).ok_or("missing value after --output-bits")?;
                let bits: u32 = v
                    .parse()
                    .map_err(|_| format!("invalid --output-bits value '{v}'"))?;
                if bits > 63 {
                    return Err(format!(
                        "unsupported --output-bits value '{bits}'; expected 0..=63"
                    ));
                }
                opts.output_bits = Some(bits);
                i += 2;
            }
            "--signed-cells" => {
                let v = args
                    .get(i + 1)
                    .ok_or("missing value after --signed-cells")?;
                let signed: bool = v
                    .parse()
                    .map_err(|_| format!("invalid --signed-cells value '{v}'"))?;
                opts.cell_sign = if signed {
                    CellSign::Signed
                } else {
                    CellSign::Unsigned
                };
                i += 2;
            }
            "--" => {
                break;
            }
            arg if arg.starts_with("--") => {
                return Err(format!(
                    "unexpected option '{arg}'; use --help to see supported BF and IR emit flags"
                ));
            }
            _ => {
                break;
            }
        }
    }
    if ir_opts.no_elide {
        ir_opts.max_lines = usize::MAX;
        ir_opts.max_depth = usize::MAX;
    } else {
        if ir_opts.max_lines == 0 {
            ir_opts.max_lines = DEFAULT_IR_MAX_LINES;
        }
        if ir_opts.max_depth == 0 {
            ir_opts.max_depth = DEFAULT_IR_MAX_DEPTH;
        }
    }
    Ok((opts, ir_opts, &args[i..]))
}
