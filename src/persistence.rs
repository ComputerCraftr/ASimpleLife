use crate::bitgrid::BitGrid;
use crate::hashlife;
#[cfg(test)]
use crate::{RequiredErrorExt, RequiredExt};
use std::io::{BufRead, BufReader, Cursor, Read, Write};

pub const LIFE_GRID_MAGIC: &str = "# life-grid v1";
pub const HASHLIFE_SNAPSHOT_MAGIC: &str = "# hashlife-snapshot v1";
const MAX_PERSISTENCE_HEADER_BYTES: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PersistenceFormat {
    LifeGrid,
    HashLifeSnapshot,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersistenceError {
    message: String,
}

impl PersistenceError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl From<std::io::Error> for PersistenceError {
    fn from(error: std::io::Error) -> Self {
        Self::new(format!("persistence I/O failed: {error}"))
    }
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for PersistenceError {}

pub fn detect_format(s: &str) -> Result<PersistenceFormat, PersistenceError> {
    let Some(header) = s.lines().next() else {
        return Err(PersistenceError::new("empty persistence payload"));
    };
    match header {
        LIFE_GRID_MAGIC => Ok(PersistenceFormat::LifeGrid),
        HASHLIFE_SNAPSHOT_MAGIC => Ok(PersistenceFormat::HashLifeSnapshot),
        _ => Err(PersistenceError::new(format!(
            "unrecognized persistence header: {header:?}"
        ))),
    }
}

pub fn serialize_life_grid(grid: &BitGrid) -> String {
    let mut cells = grid.live_cells();
    cells.sort_unstable();
    let mut out = String::from(LIFE_GRID_MAGIC);
    out.push('\n');
    for (x, y) in cells {
        out.push_str(&format!("{x} {y}\n"));
    }
    out
}

pub fn serialize_life_grid_to_writer(
    grid: &BitGrid,
    writer: &mut impl Write,
) -> Result<(), PersistenceError> {
    writeln!(writer, "{LIFE_GRID_MAGIC}")?;
    let mut cells = grid.live_cells();
    cells.sort_unstable();
    for (x, y) in cells {
        writeln!(writer, "{x} {y}")?;
    }
    Ok(())
}

pub fn deserialize_life_grid(s: &str) -> Result<BitGrid, PersistenceError> {
    deserialize_life_grid_from_reader(Cursor::new(s.as_bytes()))
}

pub fn deserialize_life_grid_from_reader(reader: impl Read) -> Result<BitGrid, PersistenceError> {
    let mut lines = BufReader::new(reader).lines();
    match lines.next().transpose()? {
        Some(line) if line == LIFE_GRID_MAGIC => {}
        Some(other) => {
            return Err(PersistenceError::new(format!(
                "unrecognized life grid header: {other:?}"
            )));
        }
        None => return Err(PersistenceError::new("empty life grid file")),
    }
    let mut cells = Vec::new();
    for (lineno, line) in lines.enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (x_str, y_str) = line.split_once(' ').ok_or_else(|| {
            PersistenceError::new(format!(
                "line {}: expected \"x y\", got {line:?}",
                lineno + 2
            ))
        })?;
        let x = x_str.parse::<i64>().map_err(|_| {
            PersistenceError::new(format!(
                "line {}: invalid x coordinate {x_str:?}",
                lineno + 2
            ))
        })?;
        let y = y_str.parse::<i64>().map_err(|_| {
            PersistenceError::new(format!(
                "line {}: invalid y coordinate {y_str:?}",
                lineno + 2
            ))
        })?;
        cells.push((x, y));
    }
    Ok(BitGrid::from_cells(&cells))
}

pub fn serialize_grid(
    grid: &BitGrid,
    format: PersistenceFormat,
) -> Result<String, PersistenceError> {
    match format {
        PersistenceFormat::LifeGrid => Ok(serialize_life_grid(grid)),
        PersistenceFormat::HashLifeSnapshot => hashlife::serialize_grid_snapshot(grid)
            .map_err(|error| PersistenceError::new(error.to_string())),
    }
}

pub fn serialize_grid_to_writer(
    grid: &BitGrid,
    format: PersistenceFormat,
    writer: &mut impl Write,
) -> Result<(), PersistenceError> {
    match format {
        PersistenceFormat::LifeGrid => serialize_life_grid_to_writer(grid, writer),
        PersistenceFormat::HashLifeSnapshot => {
            hashlife::serialize_grid_snapshot_to_writer(grid, writer)
                .map_err(|error| PersistenceError::new(error.to_string()))
        }
    }
}

pub fn deserialize_grid(s: &str) -> Result<BitGrid, PersistenceError> {
    deserialize_grid_from_reader(Cursor::new(s.as_bytes()))
}

pub fn deserialize_grid_from_reader(reader: impl Read) -> Result<BitGrid, PersistenceError> {
    let mut reader = BufReader::new(reader);
    let header = read_format_header(&mut reader)?;
    let format = detect_format_bytes(&header)?;
    let replay = Cursor::new(header).chain(reader);
    match format {
        PersistenceFormat::LifeGrid => deserialize_life_grid_from_reader(replay),
        PersistenceFormat::HashLifeSnapshot => hashlife::deserialize_snapshot_from_reader(replay)
            .map_err(|error| PersistenceError::new(error.to_string())),
    }
}

fn read_format_header(reader: &mut impl BufRead) -> Result<Vec<u8>, PersistenceError> {
    let mut header = Vec::with_capacity(32);
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            break;
        }
        let consumed = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |index| index + 1);
        if header.len().saturating_add(consumed) > MAX_PERSISTENCE_HEADER_BYTES {
            return Err(PersistenceError::new(format!(
                "persistence header exceeds {MAX_PERSISTENCE_HEADER_BYTES} bytes"
            )));
        }
        header.extend_from_slice(&available[..consumed]);
        reader.consume(consumed);
        if header.last() == Some(&b'\n') {
            break;
        }
    }
    Ok(header)
}

fn detect_format_bytes(bytes: &[u8]) -> Result<PersistenceFormat, PersistenceError> {
    let header = std::str::from_utf8(bytes)
        .map_err(|_| PersistenceError::new("persistence header is not valid UTF-8"))?
        .trim_end_matches('\n')
        .trim_end_matches('\r');
    match header {
        LIFE_GRID_MAGIC => Ok(PersistenceFormat::LifeGrid),
        HASHLIFE_SNAPSHOT_MAGIC => Ok(PersistenceFormat::HashLifeSnapshot),
        "" => Err(PersistenceError::new("empty persistence payload")),
        _ => Err(PersistenceError::new(format!(
            "unrecognized persistence header: {header:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitgrid::BitGrid;
    use crate::generators::pattern_by_name;

    #[test]
    fn life_grid_roundtrip_empty_grid() {
        let grid = BitGrid::empty();
        let serialized = serialize_life_grid(&grid);
        assert_eq!(
            deserialize_life_grid(&serialized).or_invariant("required value"),
            grid
        );
    }

    #[test]
    fn life_grid_roundtrip_multiple_cells() {
        let grid = BitGrid::from_cells(&[(0, 0), (1, 0), (-5, 10), (100, -200)]);
        let serialized = serialize_life_grid(&grid);
        assert_eq!(
            deserialize_life_grid(&serialized).or_invariant("required value"),
            grid
        );
    }

    #[test]
    fn life_grid_rejects_wrong_magic() {
        let err =
            deserialize_life_grid("# not-a-circuit\n0 0\n").error_or_invariant("expected error");
        assert!(err.to_string().contains("unrecognized life grid header"));
    }

    #[test]
    fn detects_both_persistence_formats() {
        assert_eq!(
            detect_format(&serialize_life_grid(&BitGrid::from_cells(&[(0, 0)])))
                .or_invariant("required value"),
            PersistenceFormat::LifeGrid
        );
        assert_eq!(
            detect_format(
                &hashlife::serialize_grid_snapshot(
                    &pattern_by_name("glider").or_invariant("required value"),
                )
                .or_invariant("HashLife snapshot should serialize"),
            )
            .or_invariant("required value"),
            PersistenceFormat::HashLifeSnapshot
        );
    }

    #[test]
    fn auto_deserialize_accepts_hashlife_snapshot() {
        let grid = pattern_by_name("glider").or_invariant("required value");
        let serialized = hashlife::serialize_grid_snapshot(&grid)
            .or_invariant("HashLife snapshot should serialize");
        assert_eq!(
            deserialize_grid(&serialized).or_invariant("required value"),
            grid
        );
    }

    #[test]
    fn streaming_hashlife_persistence_retains_v1_format() {
        let grid = pattern_by_name("glider").or_invariant("glider fixture");
        let mut encoded = Vec::new();
        serialize_grid_to_writer(&grid, PersistenceFormat::HashLifeSnapshot, &mut encoded)
            .or_invariant("streaming HashLife persistence should write");

        assert!(encoded.starts_with(HASHLIFE_SNAPSHOT_MAGIC.as_bytes()));
        assert_eq!(
            deserialize_grid_from_reader(Cursor::new(encoded))
                .or_invariant("streaming HashLife persistence should read"),
            grid
        );
    }
}
