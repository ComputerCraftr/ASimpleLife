//! Bounded exact occurrence copies. No live arena handles escape a capture.
use super::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug)]
pub(crate) struct CaptureLimits {
    pub bytes: usize,
    pub visits: usize,
    pub residency: Duration,
}

impl Default for CaptureLimits {
    fn default() -> Self {
        Self {
            bytes: 16 * 1024 * 1024,
            visits: 65_536,
            residency: Duration::from_millis(8),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CaptureError {
    Unloaded,
    Cancelled,
    TooLarge,
    Deadline,
    Allocation,
    InvalidGeometry,
}

#[derive(Clone, Copy, Debug)]
struct Row {
    level: u32,
    children: [usize; 4],
}

#[derive(Debug)]
pub(crate) struct OwnedDag {
    rows: Vec<Row>,
    root: usize,
    level: u32,
    origin: (Coord, Coord),
}

#[derive(Clone, Copy)]
struct Frame {
    node: NodeId,
    x: i128,
    y: i128,
    next: usize,
    children: [usize; 4],
    inside: bool,
}

impl HashLifeSession {
    pub(crate) fn capture_analysis(
        &self,
        region: Option<crate::bitgrid::Bounds>,
        limits: CaptureLimits,
        cancelled: &AtomicBool,
    ) -> Result<OwnedDag, CaptureError> {
        let start = Instant::now();
        let root = self.current_root.ok_or(CaptureError::Unloaded)?;
        if region.is_some_and(|(x, y, r, b)| x > r || y > b) {
            return Err(CaptureError::InvalidGeometry);
        }
        // Reserve a conservative complete simultaneous capacity, including hash
        // control storage and allocator rounding, before traversal begins.
        let capacity = limits.visits.checked_add(2).ok_or(CaptureError::TooLarge)?;
        let requested = capacity.checked_mul(128).ok_or(CaptureError::TooLarge)?;
        if requested > limits.bytes {
            return Err(CaptureError::TooLarge);
        }
        if cancelled.load(Ordering::Relaxed) {
            return Err(CaptureError::Cancelled);
        }
        let mut rows = Vec::new();
        rows.try_reserve_exact(capacity)
            .map_err(|_| CaptureError::Allocation)?;
        rows.extend(
            [Row {
                level: 0,
                children: [0; 4],
            }; 2],
        );
        let mut memo = HashMap::<NodeId, usize>::new();
        memo.try_reserve(capacity)
            .map_err(|_| CaptureError::Allocation)?;
        let initial = Frame {
            node: root,
            x: i128::from(self.current_origin_x),
            y: i128::from(self.current_origin_y),
            next: 0,
            children: [0; 4],
            inside: false,
        };
        let mut stack = [initial; 256];
        let mut depth = 1;
        let mut visits = 0;
        let mut resolved = None;
        loop {
            if cancelled.load(Ordering::Relaxed) {
                return Err(CaptureError::Cancelled);
            }
            if start.elapsed() >= limits.residency {
                return Err(CaptureError::Deadline);
            }
            if let Some(value) = resolved.take() {
                if depth == 0 {
                    return Ok(OwnedDag {
                        rows,
                        root: value,
                        level: self.engine.node_columns.level(root),
                        origin: (self.current_origin_x, self.current_origin_y),
                    });
                }
                let parent = &mut stack[depth - 1];
                parent.children[parent.next - 1] = value;
            }
            let frame = &mut stack[depth - 1];
            let level = self.engine.node_columns.level(frame.node);
            let size = 1_i128
                .checked_shl(level)
                .ok_or(CaptureError::InvalidGeometry)?;
            if frame.next == 0 {
                visits += 1;
                if visits > limits.visits {
                    return Err(CaptureError::TooLarge);
                }
                let outside = region.is_some_and(|(x, y, r, b)| {
                    frame.x > i128::from(r)
                        || frame.y > i128::from(b)
                        || frame.x + size <= i128::from(x)
                        || frame.y + size <= i128::from(y)
                });
                if outside || self.engine.node_columns.population(frame.node) == 0 {
                    depth -= 1;
                    resolved = Some(0);
                    continue;
                }
                frame.inside = region.is_none_or(|(x, y, r, b)| {
                    frame.x >= i128::from(x)
                        && frame.y >= i128::from(y)
                        && frame.x + size - 1 <= i128::from(r)
                        && frame.y + size - 1 <= i128::from(b)
                });
                if level == 0 {
                    depth -= 1;
                    resolved = Some(1);
                    continue;
                }
                if frame.inside
                    && let Some(&value) = memo.get(&frame.node)
                {
                    depth -= 1;
                    resolved = Some(value);
                    continue;
                }
            }
            if frame.next == 4 {
                if rows.len() == capacity {
                    return Err(CaptureError::TooLarge);
                }
                let value = rows.len();
                rows.push(Row {
                    level,
                    children: frame.children,
                });
                if frame.inside {
                    memo.insert(frame.node, value);
                }
                depth -= 1;
                resolved = Some(value);
                continue;
            }
            let quadrant = frame.next;
            frame.next += 1;
            let child = Frame {
                node: self.engine.node_columns.quadrants(frame.node)[quadrant],
                x: frame.x + if quadrant % 2 == 1 { size / 2 } else { 0 },
                y: frame.y + if quadrant >= 2 { size / 2 } else { 0 },
                next: 0,
                children: [0; 4],
                inside: false,
            };
            if depth == stack.len() {
                return Err(CaptureError::TooLarge);
            }
            stack[depth] = child;
            depth += 1;
        }
    }
}

impl OwnedDag {
    pub(crate) fn allocated_bytes(&self) -> u128 {
        self.rows.capacity() as u128 * std::mem::size_of::<Row>() as u128
    }

    /// The private analysis clock starts at zero; capture time belongs to the descriptor.
    pub(crate) fn into_analysis_session(
        self,
        budget: u128,
        cancelled: &AtomicBool,
    ) -> Result<HashLifeSession, CaptureError> {
        let owned = self.allocated_bytes();
        let map_bytes = self.rows.len() as u128 * std::mem::size_of::<NodeId>() as u128;
        let available = budget
            .checked_sub(owned + map_bytes)
            .ok_or(CaptureError::TooLarge)?;
        let mut session = HashLifeSession::with_limits(HashLifeLimits {
            soft_memory_bytes: available - available / 4,
            hard_memory_bytes: available,
        });
        let mut ids = Vec::new();
        ids.try_reserve_exact(self.rows.len())
            .map_err(|_| CaptureError::Allocation)?;
        ids.extend([session.engine.dead_leaf, session.engine.live_leaf]);
        for row in self.rows.iter().skip(2) {
            if cancelled.load(Ordering::Relaxed) {
                return Err(CaptureError::Cancelled);
            }
            let empty = session.engine.empty(row.level - 1);
            let children = row.children.map(|id| if id == 0 { empty } else { ids[id] });
            let node = session
                .engine
                .join(children[0], children[1], children[2], children[3]);
            if session.engine.allocation_failed() {
                return Err(CaptureError::Allocation);
            }
            ids.push(node);
        }
        let root = if self.root == 0 {
            session.engine.empty(self.level)
        } else {
            ids[self.root]
        };
        if session.engine.allocation_failed() {
            return Err(CaptureError::Allocation);
        }
        RootGeometry::new(self.level, self.origin.0, self.origin.1)
            .map_err(|_| CaptureError::InvalidGeometry)?;
        session.current_root = Some(root);
        session.current_origin_x = self.origin.0;
        session.current_origin_y = self.origin.1;
        Ok(session)
    }
}
