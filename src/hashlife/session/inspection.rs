use super::*;

impl HashLifeSession {
    pub(crate) fn viewport_root_bounds(&self) -> Option<crate::bitgrid::Bounds> {
        let root = self.current_root?;
        let span = 1_i64.checked_shl(self.engine.node_columns.level(root))?;
        Some((
            self.current_origin_x,
            self.current_origin_y,
            self.current_origin_x.checked_add(span - 1)?,
            self.current_origin_y.checked_add(span - 1)?,
        ))
    }
    /// Bounded, read-only discovery. None means unavailable or work exhausted,
    /// never proof that a region is empty. No node handles escape this call.
    pub(crate) fn viewport_region_occupied(
        &self,
        bounds: (i128, i128, i128, i128),
        remaining: &mut usize,
    ) -> Option<bool> {
        self.inspect_region(bounds, remaining, true, |_, _| {})
    }

    pub(crate) fn viewport_neighborhood(
        &self,
        tile: (Coord, Coord),
        remaining: &mut usize,
    ) -> Option<[u64; 9]> {
        let x = i128::from(tile.0) * 8 - 8;
        let y = i128::from(tile.1) * 8 - 8;
        let mut chunks = [0; 9];
        self.inspect_region((x, y, x + 23, y + 23), remaining, false, |cx, cy| {
            // Differences are within this fixed 24x24 inspection window.
            if let (Ok(dx), Ok(dy)) = (u32::try_from(cx - x), u32::try_from(cy - y)) {
                let index = usize::try_from((dy / 8) * 3 + dx / 8).unwrap_or(0);
                chunks[index] |= 1_u64 << ((dy % 8) * 8 + dx % 8);
            }
        })?;
        Some(chunks)
    }

    fn inspect_region(
        &self,
        bounds: (i128, i128, i128, i128),
        remaining: &mut usize,
        stop_on_occupied: bool,
        mut cell: impl FnMut(i128, i128),
    ) -> Option<bool> {
        let root = self.current_root?;
        let size = 1_i128 << self.engine.node_columns.level(root);
        let mut stack = [(NodeId::ZERO, 0_i128, 0_i128, 0_i128); 256];
        stack[0] = (
            root,
            i128::from(self.current_origin_x),
            i128::from(self.current_origin_y),
            size,
        );
        let mut len = 1;
        let mut occupied = false;
        while len != 0 {
            *remaining = remaining.checked_sub(1)?;
            len -= 1;
            let (node, x, y, size) = stack[len];
            if x > bounds.2
                || y > bounds.3
                || x + size <= bounds.0
                || y + size <= bounds.1
                || self.engine.node_columns.population(node) == 0
            {
                continue;
            }
            if stop_on_occupied
                && x >= bounds.0
                && y >= bounds.1
                && x + size - 1 <= bounds.2
                && y + size - 1 <= bounds.3
            {
                return Some(true);
            }
            if size == 1 {
                occupied = true;
                if stop_on_occupied {
                    return Some(true);
                }
                cell(x, y);
                continue;
            }
            let half = size / 2;
            let [nw, ne, sw, se] = self.engine.node_columns.quadrants(node);
            if len + 4 > stack.len() {
                return None;
            }
            for entry in [
                (se, x + half, y + half, half),
                (sw, x, y + half, half),
                (ne, x + half, y, half),
                (nw, x, y, half),
            ] {
                stack[len] = entry;
                len += 1;
            }
        }
        Some(occupied)
    }
}
