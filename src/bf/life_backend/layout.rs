use super::super::life_macro_library::life_macro_template;
use super::*;

fn push_macro(
    instances: &mut Vec<LifeMacroInstance>,
    next_id: &mut usize,
    kind: LifeMacroKind,
    origin: (i64, i64),
) {
    let template = life_macro_template(kind);
    instances.push(LifeMacroInstance {
        id: *next_id,
        kind,
        name: template.name,
        origin,
        orientation: LifeMacroOrientation::R0,
    });
    *next_id += 1;
}

pub(super) fn build_placed_machine(
    program: &[PhysicalBfInstr],
    tape_len: usize,
    cell_bits: u32,
) -> PlacedLifeMachine {
    let mut instances = Vec::new();
    let mut routed = Vec::new();
    let mut next_id = 0usize;
    let phases = vec![
        CircuitPhase::Fetch,
        CircuitPhase::Decode,
        CircuitPhase::Evaluate,
        CircuitPhase::Commit,
    ];
    let tape_stride_x = tape_stride_x(cell_bits);

    push_macro(
        &mut instances,
        &mut next_id,
        LifeMacroKind::Clock,
        (0, CONTROL_BASE_Y),
    );
    for phase_idx in 0..4 {
        push_macro(
            &mut instances,
            &mut next_id,
            LifeMacroKind::StateLatch,
            (6 + i64::from(phase_idx) * 5, CONTROL_BASE_Y),
        );
        routed.push(RoutedRail {
            name: format!("phase_{phase_idx}_tick"),
            group: RailGroup::Phase,
            source: "clock_tick".to_string(),
            sink: format!("phase_latch_{phase_idx}"),
            phase: CircuitPhase::Fetch,
            delay_generations: 1,
        });
    }
    for bit in 0..16 {
        push_macro(
            &mut instances,
            &mut next_id,
            LifeMacroKind::StateLatch,
            (i64::from(bit) * 4, CONTROL_BASE_Y + 10),
        );
    }
    for cell in 0..tape_len {
        let x = layout_coord(cell) * tape_stride_x;
        push_macro(
            &mut instances,
            &mut next_id,
            LifeMacroKind::HeadTokenMover,
            (x, TAPE_BASE_Y - 8),
        );
        push_macro(
            &mut instances,
            &mut next_id,
            LifeMacroKind::ZeroDetector,
            (x, TAPE_BASE_Y + 8),
        );
        for bit in 0..cell_bits.max(1) {
            let bit_x = x + i64::from(bit) * 4;
            push_macro(
                &mut instances,
                &mut next_id,
                LifeMacroKind::StateLatch,
                (bit_x, TAPE_BASE_Y),
            );
            push_macro(
                &mut instances,
                &mut next_id,
                LifeMacroKind::BitIncrement,
                (bit_x, TAPE_BASE_Y + 16),
            );
            push_macro(
                &mut instances,
                &mut next_id,
                LifeMacroKind::BitDecrement,
                (bit_x, TAPE_BASE_Y + 24),
            );
        }
    }
    push_macro(
        &mut instances,
        &mut next_id,
        LifeMacroKind::OutputLatch,
        (layout_coord(tape_len) * tape_stride_x + 20, TAPE_BASE_Y),
    );
    push_macro(
        &mut instances,
        &mut next_id,
        LifeMacroKind::DivergeLatch,
        (
            layout_coord(tape_len) * tape_stride_x + 20,
            TAPE_BASE_Y + 20,
        ),
    );

    for (pc, instr) in program.iter().enumerate() {
        let y = PROGRAM_BASE_Y + layout_coord(pc) * 12;
        push_macro(
            &mut instances,
            &mut next_id,
            LifeMacroKind::SplitterMerger,
            (0, y),
        );
        routed.push(RoutedRail {
            name: format!("pc{pc}_fetch"),
            group: RailGroup::ProgramControl,
            source: "program_counter".to_string(),
            sink: format!("pc{pc}_decode"),
            phase: CircuitPhase::Fetch,
            delay_generations: 1,
        });
        match instr {
            PhysicalBfInstr::Add(n) => routed.push(RoutedRail {
                name: format!("pc{pc}_add"),
                group: RailGroup::TapeData,
                source: format!("pc{pc}_decode"),
                sink: format!("cell_write_add_{n}"),
                phase: CircuitPhase::Evaluate,
                delay_generations: 1,
            }),
            PhysicalBfInstr::MovePtr(n) => routed.push(RoutedRail {
                name: format!("pc{pc}_move"),
                group: RailGroup::HeadMove,
                source: format!("pc{pc}_decode"),
                sink: format!("head_move_{n}"),
                phase: CircuitPhase::Commit,
                delay_generations: 1,
            }),
            PhysicalBfInstr::Clear => routed.push(RoutedRail {
                name: format!("pc{pc}_clear"),
                group: RailGroup::TapeData,
                source: format!("pc{pc}_decode"),
                sink: format!("cell_clear_{pc}"),
                phase: CircuitPhase::Commit,
                delay_generations: 1,
            }),
            PhysicalBfInstr::ClearAt(offset) => routed.push(RoutedRail {
                name: format!("pc{pc}_clear_at_{offset}"),
                group: RailGroup::TapeData,
                source: format!("pc{pc}_decode"),
                sink: format!("cell_clear_at_{offset}_{pc}"),
                phase: CircuitPhase::Commit,
                delay_generations: 1,
            }),
            PhysicalBfInstr::Distribute { .. } => routed.push(RoutedRail {
                name: format!("pc{pc}_distribute"),
                group: RailGroup::TapeData,
                source: format!("pc{pc}_decode"),
                sink: format!("cell_distribute_{pc}"),
                phase: CircuitPhase::Commit,
                delay_generations: 1,
            }),
            PhysicalBfInstr::Affine { .. } => routed.push(RoutedRail {
                name: format!("pc{pc}_affine"),
                group: RailGroup::TapeData,
                source: format!("pc{pc}_decode"),
                sink: format!("cell_affine_{pc}"),
                phase: CircuitPhase::Commit,
                delay_generations: 1,
            }),
            PhysicalBfInstr::Shift { dir, .. } => routed.push(RoutedRail {
                name: match dir {
                    ShiftDir::Left => format!("pc{pc}_shift_left"),
                    ShiftDir::Right => format!("pc{pc}_shift_right"),
                },
                group: RailGroup::TapeData,
                source: format!("pc{pc}_decode"),
                sink: match dir {
                    ShiftDir::Left => format!("cell_shift_left_{pc}"),
                    ShiftDir::Right => format!("cell_shift_right_{pc}"),
                },
                phase: CircuitPhase::Commit,
                delay_generations: 1,
            }),
            PhysicalBfInstr::Square { .. } => routed.push(RoutedRail {
                name: format!("pc{pc}_square"),
                group: RailGroup::TapeData,
                source: format!("pc{pc}_decode"),
                sink: format!("cell_square_{pc}"),
                phase: CircuitPhase::Commit,
                delay_generations: 1,
            }),
            PhysicalBfInstr::MulAdd { .. } => routed.push(RoutedRail {
                name: format!("pc{pc}_muladd"),
                group: RailGroup::TapeData,
                source: format!("pc{pc}_decode"),
                sink: format!("cell_muladd_{pc}"),
                phase: CircuitPhase::Commit,
                delay_generations: 1,
            }),
            PhysicalBfInstr::Output => routed.push(RoutedRail {
                name: format!("pc{pc}_output"),
                group: RailGroup::OutputTransducer,
                source: format!("pc{pc}_decode"),
                sink: format!("output_seed_row_{pc}"),
                phase: CircuitPhase::Commit,
                delay_generations: OUTPUT_ROW_SETTLE_GENERATIONS,
            }),
            PhysicalBfInstr::JumpIfZero(target) => routed.push(RoutedRail {
                name: format!("pc{pc}_jump_zero"),
                group: RailGroup::ZeroDetectBranch,
                source: format!("pc{pc}_zero_detect"),
                sink: format!("pc{target}_fetch"),
                phase: CircuitPhase::Evaluate,
                delay_generations: 1,
            }),
            PhysicalBfInstr::JumpIfNonZero(target) => routed.push(RoutedRail {
                name: format!("pc{pc}_jump_nonzero"),
                group: RailGroup::ZeroDetectBranch,
                source: format!("pc{pc}_zero_detect"),
                sink: format!("pc{target}_fetch"),
                phase: CircuitPhase::Evaluate,
                delay_generations: 1,
            }),
            PhysicalBfInstr::Diverge => routed.push(RoutedRail {
                name: format!("pc{pc}_diverge"),
                group: RailGroup::HaltDiverge,
                source: format!("pc{pc}_decode"),
                sink: "diverge_latch".to_string(),
                phase: CircuitPhase::Commit,
                delay_generations: 1,
            }),
            PhysicalBfInstr::Halt => routed.push(RoutedRail {
                name: format!("pc{pc}_halt"),
                group: RailGroup::HaltDiverge,
                source: format!("pc{pc}_decode"),
                sink: "halt_latch".to_string(),
                phase: CircuitPhase::Commit,
                delay_generations: 1,
            }),
        }
    }
    let macro_timing_specs = vec![
        MacroTimingSpec {
            kind: LifeMacroKind::Clock,
            active_phase: CircuitPhase::Fetch,
            settle_generations: 0,
        },
        MacroTimingSpec {
            kind: LifeMacroKind::StateLatch,
            active_phase: CircuitPhase::Commit,
            settle_generations: 0,
        },
        MacroTimingSpec {
            kind: LifeMacroKind::HeadTokenMover,
            active_phase: CircuitPhase::Commit,
            settle_generations: 1,
        },
        MacroTimingSpec {
            kind: LifeMacroKind::BitIncrement,
            active_phase: CircuitPhase::Evaluate,
            settle_generations: 1,
        },
        MacroTimingSpec {
            kind: LifeMacroKind::BitDecrement,
            active_phase: CircuitPhase::Evaluate,
            settle_generations: 1,
        },
        MacroTimingSpec {
            kind: LifeMacroKind::ZeroDetector,
            active_phase: CircuitPhase::Evaluate,
            settle_generations: 1,
        },
        MacroTimingSpec {
            kind: LifeMacroKind::OutputLatch,
            active_phase: CircuitPhase::Commit,
            settle_generations: OUTPUT_ROW_SETTLE_GENERATIONS,
        },
        MacroTimingSpec {
            kind: LifeMacroKind::OutputBitTransducer,
            active_phase: CircuitPhase::Commit,
            settle_generations: OUTPUT_ROW_SETTLE_GENERATIONS,
        },
        MacroTimingSpec {
            kind: LifeMacroKind::DivergeLatch,
            active_phase: CircuitPhase::Commit,
            settle_generations: 0,
        },
    ];

    PlacedLifeMachine {
        phases,
        macro_instances: instances,
        routed_rails: routed,
        macro_timing_specs,
        output_row_settle_generations: OUTPUT_ROW_SETTLE_GENERATIONS,
    }
}
