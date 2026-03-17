use crate::bitgrid::BitGrid;

const MAGIC: &str = "# life-grid v1";

pub fn serialize(grid: &BitGrid) -> String {
    let mut cells = grid.live_cells();
    cells.sort_unstable();
    let mut out = String::from(MAGIC);
    out.push('\n');
    for (x, y) in cells {
        out.push_str(&format!("{x} {y}\n"));
    }
    out
}

pub fn deserialize(s: &str) -> Result<BitGrid, String> {
    let mut lines = s.lines();
    match lines.next() {
        Some(line) if line == MAGIC => {}
        Some(other) => return Err(format!("unrecognized life grid header: {other:?}")),
        None => return Err("empty life grid file".to_string()),
    }
    let mut cells = Vec::new();
    for (lineno, line) in lines.enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (x_str, y_str) = line
            .split_once(' ')
            .ok_or_else(|| format!("line {}: expected \"x y\", got {line:?}", lineno + 2))?;
        let x = x_str
            .parse::<i64>()
            .map_err(|_| format!("line {}: invalid x coordinate {x_str:?}", lineno + 2))?;
        let y = y_str
            .parse::<i64>()
            .map_err(|_| format!("line {}: invalid y coordinate {y_str:?}", lineno + 2))?;
        cells.push((x, y));
    }
    Ok(BitGrid::from_cells(&cells))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitgrid::BitGrid;

    #[test]
    fn roundtrip_empty_grid() {
        let grid = BitGrid::empty();
        let s = serialize(&grid);
        let back = deserialize(&s).unwrap();
        assert_eq!(back, grid);
    }

    #[test]
    fn roundtrip_single_cell() {
        let grid = BitGrid::from_cells(&[(3, -7)]);
        let s = serialize(&grid);
        let back = deserialize(&s).unwrap();
        assert_eq!(back, grid);
    }

    #[test]
    fn roundtrip_multiple_cells() {
        let cells = vec![(0, 0), (1, 0), (-5, 10), (100, -200)];
        let grid = BitGrid::from_cells(&cells);
        let s = serialize(&grid);
        let back = deserialize(&s).unwrap();
        assert_eq!(back, grid);
    }

    #[test]
    fn rejects_wrong_magic() {
        let err = deserialize("# not-a-circuit\n0 0\n").unwrap_err();
        assert!(err.contains("unrecognized life grid header"));
    }

    #[test]
    fn rejects_empty_input() {
        let err = deserialize("").unwrap_err();
        assert!(err.contains("empty life grid file"));
    }

    #[test]
    fn rejects_malformed_coordinate_line() {
        let s = format!("{MAGIC}\nnot-a-coord\n");
        let err = deserialize(&s).unwrap_err();
        assert!(err.contains("expected \"x y\""));
    }

    #[test]
    fn skips_blank_lines_and_comments() {
        let s = format!("{MAGIC}\n\n# a comment\n1 2\n\n3 4\n");
        let grid = deserialize(&s).unwrap();
        assert_eq!(grid, BitGrid::from_cells(&[(1, 2), (3, 4)]));
    }

    #[test]
    fn serialized_output_starts_with_magic() {
        let s = serialize(&BitGrid::from_cells(&[(0, 0)]));
        assert!(s.starts_with(MAGIC));
    }
}
