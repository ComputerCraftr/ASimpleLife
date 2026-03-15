use super::*;
use crate::RequiredExt;

impl ReferenceLifeScaffold {
    pub fn placed_machine(&self) -> &PlacedLifeMachine {
        &self.placed_machine
    }

    pub fn macro_instances(&self) -> &[LifeMacroInstance] {
        &self.placed_machine.macro_instances
    }

    pub fn routed_rails(&self) -> &[RoutedRail] {
        &self.placed_machine.routed_rails
    }

    pub fn macro_timing_specs(&self) -> &[MacroTimingSpec] {
        &self.placed_machine.macro_timing_specs
    }

    pub fn debug_routed_signals(&self) -> Vec<String> {
        self.placed_machine
            .routed_rails
            .iter()
            .map(|rail| {
                format!(
                    "{:?}/{}: {} -> {}",
                    rail.group, rail.name, rail.source, rail.sink
                )
            })
            .collect()
    }

    pub fn reference_run_to_completion(&mut self) -> Result<(), BfLifeCircuitError> {
        while self.step()? {}
        Ok(())
    }

    /// Returns only the immutable placed-machine grid. Mutable reference-model
    /// tape, control, and output state must never affect production emission.
    pub fn compiled_grid(&self) -> BitGrid {
        macro_instance_grid(self.macro_instances())
    }
}

impl ReferenceLifeScaffold {
    pub fn step(&mut self) -> Result<bool, BfLifeCircuitError> {
        if self.state.phase == CircuitPhase::Halted {
            return Ok(false);
        }
        if self.state.steps >= CIRCUIT_STEP_BUDGET {
            return Err(BfLifeCircuitError::StepBudgetExceeded);
        }
        self.state.steps += 1;

        match self.state.phase {
            CircuitPhase::Fetch => {
                let instr = self
                    .program
                    .get(self.state.pc)
                    .cloned()
                    .unwrap_or(PhysicalBfInstr::Halt);
                self.state.latched_instr = Some(instr);
                self.state.phase = CircuitPhase::Decode;
                Ok(true)
            }
            CircuitPhase::Decode => {
                let instr = self
                    .state
                    .latched_instr
                    .clone()
                    .or_invariant("fetch must latch an instruction");
                if instr == PhysicalBfInstr::Halt {
                    self.state.phase = CircuitPhase::Halted;
                    self.state.pending = None;
                    return Ok(false);
                }
                self.state.phase = CircuitPhase::Evaluate;
                Ok(true)
            }
            CircuitPhase::Evaluate => {
                let instr = self
                    .state
                    .latched_instr
                    .clone()
                    .or_invariant("decode must retain the instruction");
                let cur = self.state.tape[self.state.head];
                let pending = match instr {
                    PhysicalBfInstr::Add(delta) => PendingAction {
                        next_pc: self.state.pc + 1,
                        next_head: self.state.head,
                        writes: vec![(
                            self.state.head,
                            wrap_u64(
                                cur.wrapping_add(i64::from(delta).cast_unsigned()),
                                self.cell_bits,
                            ),
                        )],
                        emit: None,
                    },
                    PhysicalBfInstr::MovePtr(delta) => PendingAction {
                        next_pc: self.state.pc + 1,
                        next_head: wrap_tape_index(self.state.head, delta, self.tape_len),
                        writes: Vec::new(),
                        emit: None,
                    },
                    PhysicalBfInstr::Clear => PendingAction {
                        next_pc: self.state.pc + 1,
                        next_head: self.state.head,
                        writes: vec![(self.state.head, 0)],
                        emit: None,
                    },
                    PhysicalBfInstr::ClearAt(offset) => PendingAction {
                        next_pc: self.state.pc + 1,
                        next_head: self.state.head,
                        writes: vec![(wrap_tape_index(self.state.head, offset, self.tape_len), 0)],
                        emit: None,
                    },
                    PhysicalBfInstr::Distribute {
                        targets,
                        preserve_src,
                    } => {
                        let mut writes =
                            Vec::with_capacity(targets.len() + usize::from(!preserve_src));
                        for (offset, coeff) in targets {
                            let target = wrap_tape_index(self.state.head, offset, self.tape_len);
                            let base = writes
                                .iter()
                                .rev()
                                .find_map(|&(index, value)| (index == target).then_some(value))
                                .unwrap_or(self.state.tape[target]);
                            let next = wrap_add_product(
                                base,
                                cur,
                                i64::from(coeff).cast_unsigned(),
                                self.cell_bits,
                            );
                            if let Some((_, value)) =
                                writes.iter_mut().find(|(index, _)| *index == target)
                            {
                                *value = next;
                            } else {
                                writes.push((target, next));
                            }
                        }
                        if !preserve_src {
                            writes.push((self.state.head, 0));
                        }
                        PendingAction {
                            next_pc: self.state.pc + 1,
                            next_head: self.state.head,
                            writes,
                            emit: None,
                        }
                    }
                    PhysicalBfInstr::Affine {
                        src,
                        dst,
                        coeff,
                        preserve_src,
                        set_dst,
                    } => {
                        let s = wrap_tape_index(self.state.head, src, self.tape_len);
                        let d = wrap_tape_index(self.state.head, dst, self.tape_len);
                        let src_v = self.state.tape[s];
                        let base = if set_dst { 0 } else { self.state.tape[d] };
                        let dst_next = wrap_add_product(
                            base,
                            src_v,
                            i64::from(coeff).cast_unsigned(),
                            self.cell_bits,
                        );
                        let mut writes = Vec::new();
                        if !preserve_src && s != d {
                            writes.push((s, 0));
                        }
                        writes.push((d, dst_next));
                        PendingAction {
                            next_pc: self.state.pc + 1,
                            next_head: self.state.head,
                            writes,
                            emit: None,
                        }
                    }
                    PhysicalBfInstr::Shift {
                        src,
                        dst,
                        amount,
                        dir,
                        preserve_src,
                        set_dst,
                    } => {
                        let s = wrap_tape_index(self.state.head, src, self.tape_len);
                        let d = wrap_tape_index(self.state.head, dst, self.tape_len);
                        let shifted = match dir {
                            ShiftDir::Left => {
                                wrap_shift_left(self.state.tape[s], amount, self.cell_bits)
                            }
                            ShiftDir::Right => {
                                wrap_shift_right(self.state.tape[s], amount, self.cell_bits)
                            }
                        };
                        let base = if set_dst { 0 } else { self.state.tape[d] };
                        let dst_next = wrap_u64(base.wrapping_add(shifted), self.cell_bits);
                        let mut writes = Vec::new();
                        if !preserve_src && s != d {
                            writes.push((s, 0));
                        }
                        writes.push((d, dst_next));
                        PendingAction {
                            next_pc: self.state.pc + 1,
                            next_head: self.state.head,
                            writes,
                            emit: None,
                        }
                    }
                    PhysicalBfInstr::Square {
                        src,
                        dst,
                        preserve_src,
                        set_dst,
                    } => {
                        let s = wrap_tape_index(self.state.head, src, self.tape_len);
                        let d = wrap_tape_index(self.state.head, dst, self.tape_len);
                        let src_v = self.state.tape[s];
                        let base = if set_dst { 0 } else { self.state.tape[d] };
                        let dst_next = wrap_add_product(base, src_v, src_v, self.cell_bits);
                        let mut writes = Vec::new();
                        if !preserve_src && s != d {
                            writes.push((s, 0));
                        }
                        writes.push((d, dst_next));
                        PendingAction {
                            next_pc: self.state.pc + 1,
                            next_head: self.state.head,
                            writes,
                            emit: None,
                        }
                    }
                    PhysicalBfInstr::MulAdd {
                        lhs,
                        rhs,
                        dst,
                        preserve_lhs,
                        preserve_rhs,
                        set_dst,
                    } => {
                        let l = wrap_tape_index(self.state.head, lhs, self.tape_len);
                        let r = wrap_tape_index(self.state.head, rhs, self.tape_len);
                        let d = wrap_tape_index(self.state.head, dst, self.tape_len);
                        let lhs_v = self.state.tape[l];
                        let rhs_v = self.state.tape[r];
                        let base = if set_dst { 0 } else { self.state.tape[d] };
                        let dst_next = wrap_add_product(base, lhs_v, rhs_v, self.cell_bits);
                        let mut writes = Vec::new();
                        if !preserve_lhs && l != d {
                            writes.push((l, 0));
                        }
                        if !preserve_rhs && r != d && (r != l || preserve_lhs) {
                            writes.push((r, 0));
                        }
                        writes.push((d, dst_next));
                        PendingAction {
                            next_pc: self.state.pc + 1,
                            next_head: self.state.head,
                            writes,
                            emit: None,
                        }
                    }
                    PhysicalBfInstr::Output => PendingAction {
                        next_pc: self.state.pc + 1,
                        next_head: self.state.head,
                        writes: Vec::new(),
                        emit: Some(cur),
                    },
                    PhysicalBfInstr::JumpIfZero(target) => PendingAction {
                        next_pc: if cur == 0 { target } else { self.state.pc + 1 },
                        next_head: self.state.head,
                        writes: Vec::new(),
                        emit: None,
                    },
                    PhysicalBfInstr::JumpIfNonZero(target) => PendingAction {
                        next_pc: if cur != 0 { target } else { self.state.pc + 1 },
                        next_head: self.state.head,
                        writes: Vec::new(),
                        emit: None,
                    },
                    PhysicalBfInstr::Diverge => PendingAction {
                        next_pc: self.state.pc,
                        next_head: self.state.head,
                        writes: Vec::new(),
                        emit: None,
                    },
                    PhysicalBfInstr::Halt => crate::invariant_failure!("halt handled in decode"),
                };
                self.state.pending = Some(pending);
                self.state.phase = CircuitPhase::Commit;
                Ok(true)
            }
            CircuitPhase::Commit => {
                let pending = self
                    .state
                    .pending
                    .take()
                    .or_invariant("evaluate must prepare a pending action");
                for (target, value) in pending.writes {
                    self.state.tape[target] = value;
                }
                self.state.output_latch = pending.emit;
                if let Some(value) = pending.emit {
                    self.state.outputs.push(value);
                }
                self.state.pc = pending.next_pc;
                self.state.head = pending.next_head;
                self.state.latched_instr = None;
                self.state.phase = CircuitPhase::Fetch;
                Ok(true)
            }
            CircuitPhase::Halted => Ok(false),
        }
    }

    pub fn output_row_settle_generations(&self) -> u64 {
        self.placed_machine.output_row_settle_generations
    }
}
