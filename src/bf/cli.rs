use std::io::{self, Read};

use super::c_backend::{emit_c, format_ir};
use super::c_super_backend::emit_c_super;
use super::ir::Parser;
use super::life_backend::{
    compile_to_life_circuit, serialize_life_circuit, serialize_life_circuit_hashlife,
};
use super::optimizer::{CellSign, CodegenOpts, IoMode, optimize};

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
    println!("  --emit-life    emit the BF Life circuit as a standard life-grid export");
    println!("  --emit-life-hashlife emit the BF Life circuit as a HashLife snapshot export");
    println!("opts: --cell-bits N  --io char|number  --signed-cells true|false");
}

pub fn run() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        std::process::exit(0);
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
        OutputMode::EmitIr => {
            let optimized = optimize(parsed.clone());
            println!("=== Parsed IR ===");
            print!("{}", format_ir(&parsed));
            println!("=== Optimized IR ===");
            print!("{}", format_ir(&optimized));
        }
        OutputMode::EmitC => {
            print!("{}", emit_c(&optimize(parsed), opts));
        }
        OutputMode::EmitCSuper => {
            print!("{}", emit_c_super(&optimize(parsed), opts));
        }
        OutputMode::EmitLife => match compile_to_life_circuit(&optimize(parsed), opts) {
            Ok(circuit) => print!("{}", serialize_life_circuit(&circuit)),
            Err(err) => {
                eprintln!("error: {err}");
                std::process::exit(1);
            }
        },
        OutputMode::EmitLifeHashLife => match compile_to_life_circuit(&optimize(parsed), opts) {
            Ok(circuit) => print!("{}", serialize_life_circuit_hashlife(&circuit)),
            Err(err) => {
                eprintln!("error: {err}");
                std::process::exit(1);
            }
        },
    }
}

pub(super) fn read_input(args: &[String]) -> Result<(OutputMode, CodegenOpts, String), String> {
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
        cell_sign: CellSign::Unsigned,
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
            arg => {
                eprintln!("unexpected argument: '{arg}'");
                print_help();
                std::process::exit(2);
            }
        }
    }
    Ok((opts, &args[i..]))
}
