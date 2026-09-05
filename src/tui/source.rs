use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use crate::app::initial_grid;
use crate::bf::{
    CellSign, CodegenOpts, IoMode, Parser, compile_to_life_circuit, optimize_with_opts,
};
use crate::cli::Config;
use crate::engine::SimulationSession;
use crate::generators::{pattern_by_name, random_soup};
use crate::persistence::{
    HASHLIFE_SNAPSHOT_MAGIC, LIFE_GRID_MAGIC, PersistenceFormat, deserialize_life_grid_from_reader,
};

use super::protocol::PreparedSource;

pub fn prepare_config_source(config: &Config) -> Result<PreparedSource, String> {
    let mut prepared = if let Some(path) = config.load.as_deref() {
        prepare_file_source(path)?
    } else if let Some(source) = config.bf.as_deref() {
        prepare_bf_source(source)?
    } else {
        prepare_grid(initial_grid(config), config.pattern.clone(), 0)?
    };
    if let Some(target) = config.target_generation {
        let current = prepared.session.hashlife_generation();
        if target < current {
            return Err(format!(
                "target generation {target} is before loaded generation {current}"
            ));
        }
        prepared
            .session
            .advance_hashlife_root(target - current)
            .map_err(|error| format!("failed to reach target generation {target}: {error:?}"))?;
    }
    Ok(prepared)
}

pub fn prepare_named_source(name: &str, config: &Config) -> Result<PreparedSource, String> {
    let name = name.trim();
    let grid = if name == "random" {
        let width = crate::bitgrid::Coord::try_from(config.width)
            .map_err(|_| "configured width exceeds the Life coordinate domain".to_string())?;
        let height = crate::bitgrid::Coord::try_from(config.height)
            .map_err(|_| "configured height exceeds the Life coordinate domain".to_string())?;
        random_soup(width, height, 30, config.seed)
    } else {
        pattern_by_name(name).ok_or_else(|| format!("unknown Life seed {name:?}"))?
    };
    prepare_grid(grid, name.to_string(), 0)
}

pub fn prepare_file_source(path: &str) -> Result<PreparedSource, String> {
    let format = detect_file_format(path)?;
    let file = File::open(path).map_err(|error| format!("failed to open {path:?}: {error}"))?;
    match format {
        PersistenceFormat::HashLifeSnapshot => {
            let mut session = SimulationSession::new();
            session
                .load_hashlife_snapshot_reader(BufReader::new(file))
                .map_err(|error| format!("failed to load HashLife snapshot: {error:?}"))?;
            Ok(PreparedSource {
                session,
                label: path.to_string(),
            })
        }
        PersistenceFormat::LifeGrid => {
            let grid = deserialize_life_grid_from_reader(BufReader::new(file))
                .map_err(|error| error.to_string())?;
            prepare_grid(grid, path.to_string(), 0)
        }
    }
}

fn detect_file_format(path: &str) -> Result<PersistenceFormat, String> {
    const MAX_HEADER_BYTES: u64 = 256;
    let file = File::open(path).map_err(|error| format!("failed to open {path:?}: {error}"))?;
    let mut header = Vec::new();
    BufReader::new(file)
        .take(MAX_HEADER_BYTES + 1)
        .read_until(b'\n', &mut header)
        .map_err(|error| format!("failed to read {path:?}: {error}"))?;
    if header.len() > usize::try_from(MAX_HEADER_BYTES).map_err(|_| "invalid header limit")? {
        return Err(format!(
            "persistence header exceeds {MAX_HEADER_BYTES} bytes"
        ));
    }
    let header = std::str::from_utf8(&header)
        .map_err(|_| "persistence header is not valid UTF-8".to_string())?
        .trim_end_matches(['\r', '\n']);
    match header {
        LIFE_GRID_MAGIC => Ok(PersistenceFormat::LifeGrid),
        HASHLIFE_SNAPSHOT_MAGIC => Ok(PersistenceFormat::HashLifeSnapshot),
        _ => Err(format!("unrecognized persistence header: {header:?}")),
    }
}

pub fn prepare_bf_source(source_or_path: &str) -> Result<PreparedSource, String> {
    let source = if Path::new(source_or_path).is_file() {
        fs::read_to_string(source_or_path)
            .map_err(|error| format!("failed to read BF source {source_or_path:?}: {error}"))?
    } else {
        source_or_path.to_string()
    };
    let parsed = Parser::new(&source)
        .parse()
        .map_err(|error| format!("BF parse error: {error}"))?;
    let opts = CodegenOpts {
        io_mode: IoMode::Char,
        cell_bits: 8,
        input_bits: None,
        output_bits: None,
        cell_sign: CellSign::Unsigned,
    };
    let optimized = optimize_with_opts(parsed, opts);
    let program = compile_to_life_circuit(&optimized, opts)
        .map_err(|error| format!("BF-to-Life compilation failed: {error}"))?;
    prepare_grid(program.initial_grid().clone(), "brainfuck".to_string(), 0)
}

fn prepare_grid(
    grid: crate::bitgrid::BitGrid,
    label: String,
    generation: u64,
) -> Result<PreparedSource, String> {
    let mut session = SimulationSession::new();
    session
        .try_load_hashlife_state_at_generation(&grid, generation)
        .map_err(|error| format!("failed to embed source in HashLife: {error:?}"))?;
    Ok(PreparedSource { session, label })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequiredExt;

    #[test]
    fn named_source_rejects_unknown_seed_without_touching_a_session() {
        let error = prepare_named_source("not-a-life-seed", &Config::default())
            .err()
            .or_invariant("unknown seed should fail preparation");
        assert!(
            error.contains("unknown Life seed"),
            "unexpected preparation error: {error}"
        );
    }

    #[test]
    fn config_source_reaches_requested_generation_before_publication() {
        let config = Config {
            pattern: "block".to_string(),
            target_generation: Some(1_000),
            ..Config::default()
        };
        let prepared = prepare_config_source(&config).or_invariant("block should prepare");
        assert_eq!(prepared.session.hashlife_generation(), 1_000);
    }

    #[test]
    fn snapshot_target_is_absolute_not_relative_to_saved_generation() {
        let mut saved =
            prepare_named_source("block", &Config::default()).or_invariant("block should prepare");
        saved
            .session
            .advance_hashlife_root(100)
            .or_invariant("saved block should advance");
        let snapshot = saved
            .session
            .export_hashlife_snapshot()
            .or_invariant("snapshot should export")
            .or_invariant("prepared source should be loaded");
        let path = std::env::temp_dir().join(format!(
            "asimplelife-tui-source-{}-100.hls",
            std::process::id()
        ));
        fs::write(&path, snapshot).or_invariant("snapshot fixture should write");
        let config = Config {
            load: Some(
                path.to_str()
                    .or_invariant("temporary path should be UTF-8")
                    .to_string(),
            ),
            target_generation: Some(1_000),
            ..Config::default()
        };

        let prepared = prepare_config_source(&config).or_invariant("snapshot should prepare");
        let _ = fs::remove_file(path);
        assert_eq!(prepared.session.hashlife_generation(), 1_000);
    }
}
