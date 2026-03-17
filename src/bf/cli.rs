use std::io::{self, Read};

use super::codegen::{emit_c, format_ir, serialize_life_grid};
use super::ir::Parser;
use super::optimizer::{CellSign, CodegenOpts, IoMode, optimize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputMode {
    DumpIr,
    EmitC,
    EmitLife,
}

pub fn run() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!(
            "usage: bf_life [--dump-ir|--emit-c|--emit-life] [opts] [-- <src> | <file> | <src>]"
        );
        eprintln!("  --dump-ir      print parsed and optimized IR (default)");
        eprintln!("  --emit-c       emit a C translation");
        eprintln!("  --emit-life    emit a life grid file (tape state as live cells)");
        eprintln!("opts: --cell-bits N  --io char|number  --signed-cells  --unsigned-cells");
        return;
    }

    let (mode, opts, src) = match read_input(&args) {
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

    match mode {
        OutputMode::DumpIr => {
            let optimized = optimize(parsed.clone());
            println!("=== Parsed IR ===");
            print!("{}", format_ir(&parsed));
            println!("=== Optimized IR ===");
            print!("{}", format_ir(&optimized));
        }
        OutputMode::EmitC => {
            print!("{}", emit_c(&optimize(parsed), opts));
        }
        OutputMode::EmitLife => {
            print!("{}", serialize_life_grid(&optimize(parsed), opts));
        }
    }
}

fn read_input(args: &[String]) -> Result<(OutputMode, CodegenOpts, String), String> {
    let mut mode = OutputMode::DumpIr;
    let mut rest = args;

    if let Some(first) = rest.first() {
        match first.as_str() {
            "--emit-c" => {
                mode = OutputMode::EmitC;
                rest = &rest[1..];
            }
            "--dump-ir" => {
                mode = OutputMode::DumpIr;
                rest = &rest[1..];
            }
            "--emit-life" => {
                mode = OutputMode::EmitLife;
                rest = &rest[1..];
            }
            _ => {}
        }
    }

    let (opts, rest) = parse_opts(rest)?;

    if rest.is_empty() {
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("failed to read stdin: {e}"))?;
        return Ok((mode, opts, buf));
    }
    if rest[0] == "--" {
        return Ok((mode, opts, rest[1..].join(" ")));
    }
    if rest.len() == 1 {
        let arg = &rest[0];
        if std::path::Path::new(arg).exists() {
            let src =
                std::fs::read_to_string(arg).map_err(|e| format!("failed to read '{arg}': {e}"))?;
            return Ok((mode, opts, src));
        }
        return Ok((mode, opts, arg.clone()));
    }
    Ok((mode, opts, rest.join(" ")))
}

pub(super) fn parse_opts(args: &[String]) -> Result<(CodegenOpts, &[String]), String> {
    let mut opts = CodegenOpts {
        io_mode: IoMode::Char,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Signed,
    };
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
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
                opts.cell_sign = CellSign::Signed;
                i += 1;
            }
            "--unsigned-cells" => {
                opts.cell_sign = CellSign::Unsigned;
                i += 1;
            }
            _ => break,
        }
    }
    Ok((opts, &args[i..]))
}
