//! Recurrence evidence capture and transactional session skipping.

use super::*;
use crate::recurrence::{DagWitness, Lineage, Observation, RecurrenceSkip, RecurrenceUnavailable};
use normalization::{
    BoundsMemo, ReblockFrame, ReblockMemo, bounded_relative_bounds, reblock_square,
};

mod normalization;
#[cfg(test)]
mod tests;

impl HashLifeSession {
    pub(crate) fn try_recurrence_observation(
        &mut self,
        lineage: Lineage,
    ) -> Result<Observation, RecurrenceUnavailable> {
        if self.engine.allocation_failed() {
            return Err(RecurrenceUnavailable::Allocation);
        }
        let root = self
            .current_root
            .ok_or(RecurrenceUnavailable::WitnessLimit)?;
        let generation = self.current_generation;
        let mut work = 0;
        let mut bounds_memo = BoundsMemo::new();
        let relative_bounds =
            bounded_relative_bounds(&self.engine, root, &mut bounds_memo, &mut work)?;
        let bounds = relative_bounds.map(|relative| {
            let origin_x = i128::from(self.current_origin_x);
            let origin_y = i128::from(self.current_origin_y);
            (
                origin_x + i128::from(relative.min_x),
                origin_y + i128::from(relative.min_y),
                origin_x + i128::from(relative.max_x),
                origin_y + i128::from(relative.max_y),
            )
        });
        let (anchor_x, anchor_y, level) = match bounds {
            Some((min_x, min_y, max_x, max_y)) => {
                let width = max_x - min_x + 1;
                let height = max_y - min_y + 1;
                let span = u128::try_from(width.max(height))
                    .map_err(|_| RecurrenceUnavailable::CoordinateOverflow)?;
                let size = span
                    .checked_next_power_of_two()
                    .ok_or(RecurrenceUnavailable::CoordinateOverflow)?;
                (min_x, min_y, size.trailing_zeros())
            }
            None => (0, 0, 0),
        };

        let saved_failure = self.engine.allocation_failure;
        let saved_reserved = self.engine.allocation_transient_reserved;
        let mut reblock_memo = ReblockMemo::new();
        let normalized = if bounds.is_none() {
            Ok(self.engine.dead_leaf)
        } else {
            reblock_square(
                &mut self.engine,
                ReblockFrame {
                    source: root,
                    source_x: i128::from(self.current_origin_x),
                    source_y: i128::from(self.current_origin_y),
                    target_x: anchor_x,
                    target_y: anchor_y,
                    target_level: level,
                },
                bounds.or_invariant("nonempty recurrence root must have exact bounds"),
                &bounds_memo,
                &mut reblock_memo,
                &mut work,
            )
        };
        let candidate_failed = self.engine.allocation_failure != saved_failure;
        self.engine.allocation_failure = saved_failure;
        self.engine.allocation_transient_reserved = saved_reserved;
        let retained = self.allocated_bytes();
        self.memory_budget.sync_retained(retained);
        if candidate_failed {
            return Err(RecurrenceUnavailable::Allocation);
        }
        let normalized = normalized?;
        Ok(Observation::from_dag(
            lineage,
            generation,
            (anchor_x, anchor_y),
            DagWitness::new(
                self.checkpoint_session,
                self.engine.arena_epoch,
                u64::from(normalized),
                level,
            ),
        ))
    }

    pub(crate) fn try_apply_recurrence_skip(
        &mut self,
        skip: RecurrenceSkip,
    ) -> Result<SessionAdvanceStats, RecurrenceUnavailable> {
        let root = self
            .current_root
            .ok_or(RecurrenceUnavailable::WitnessLimit)?;
        let starting_generation = self.current_generation;
        let reached_generation = starting_generation
            .checked_add(skip.committed_generations())
            .ok_or(RecurrenceUnavailable::CoordinateOverflow)?;
        let displacement = skip.displacement();
        let origin_x = i128::from(self.current_origin_x)
            .checked_add(displacement.0)
            .ok_or(RecurrenceUnavailable::CoordinateOverflow)?;
        let origin_y = i128::from(self.current_origin_y)
            .checked_add(displacement.1)
            .ok_or(RecurrenceUnavailable::CoordinateOverflow)?;
        let origin_x = recurrence_coord(origin_x)?;
        let origin_y = recurrence_coord(origin_y)?;
        RootGeometry::new(self.engine.node_columns.level(root), origin_x, origin_y)
            .map_err(|_| RecurrenceUnavailable::CoordinateOverflow)?;

        self.current_origin_x = origin_x;
        self.current_origin_y = origin_y;
        self.current_generation = reached_generation;
        self.clear_cached_samples();
        Ok(SessionAdvanceStats {
            requested_generations: skip.committed_generations(),
            completed_generations: skip.committed_generations(),
            starting_generation,
            reached_generation,
        })
    }
}

#[doc = "source-policy: checked-narrowing-boundary"]
fn recurrence_coord(value: i128) -> Result<Coord, RecurrenceUnavailable> {
    Coord::try_from(value).map_err(|_| RecurrenceUnavailable::CoordinateOverflow)
}
