use crate::bitgrid::BitGrid;
use crate::hashlife;

pub const LIFE_GRID_MAGIC: &str = "# life-grid v1";
pub const HASHLIFE_SNAPSHOT_MAGIC: &str = "# hashlife-snapshot v1";

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

pub fn deserialize_life_grid(s: &str) -> Result<BitGrid, PersistenceError> {
    let mut lines = s.lines();
    match lines.next() {
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
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (x_str, y_str) = line
            .split_once(' ')
            .ok_or_else(|| {
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

pub fn serialize_grid(grid: &BitGrid, format: PersistenceFormat) -> String {
    match format {
        PersistenceFormat::LifeGrid => serialize_life_grid(grid),
        PersistenceFormat::HashLifeSnapshot => hashlife::serialize_grid_snapshot(grid),
    }
}

pub fn deserialize_grid(s: &str) -> Result<BitGrid, PersistenceError> {
    match detect_format(s)? {
        PersistenceFormat::LifeGrid => deserialize_life_grid(s),
        PersistenceFormat::HashLifeSnapshot => hashlife::deserialize_snapshot_to_grid(s)
            .map_err(|err| PersistenceError::new(err.to_string())),
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
        assert_eq!(deserialize_life_grid(&serialized).unwrap(), grid);
    }

    #[test]
    fn life_grid_roundtrip_multiple_cells() {
        let grid = BitGrid::from_cells(&[(0, 0), (1, 0), (-5, 10), (100, -200)]);
        let serialized = serialize_life_grid(&grid);
        assert_eq!(deserialize_life_grid(&serialized).unwrap(), grid);
    }

    #[test]
    fn life_grid_rejects_wrong_magic() {
        let err = deserialize_life_grid("# not-a-circuit\n0 0\n").unwrap_err();
        assert!(err.to_string().contains("unrecognized life grid header"));
    }

    #[test]
    fn detects_both_persistence_formats() {
        assert_eq!(
            detect_format(&serialize_life_grid(&BitGrid::from_cells(&[(0, 0)]))).unwrap(),
            PersistenceFormat::LifeGrid
        );
        assert_eq!(
            detect_format(&hashlife::serialize_grid_snapshot(&pattern_by_name("glider").unwrap()))
                .unwrap(),
            PersistenceFormat::HashLifeSnapshot
        );
    }

    #[test]
    fn auto_deserialize_accepts_hashlife_snapshot() {
        let grid = pattern_by_name("glider").unwrap();
        let serialized = hashlife::serialize_grid_snapshot(&grid);
        assert_eq!(deserialize_grid(&serialized).unwrap(), grid);
    }
}
