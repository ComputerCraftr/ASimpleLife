use super::*;

#[test]
fn emit_ir_default_text_report_is_summary_aware() {
    let parsed = parse_only("+++[->++<]>.<.");
    let optimized = optimize_with_opts(parsed.clone(), life_opts());
    let report = build_ir_report(
        "+++[->++<]>.<.",
        &parsed,
        &optimized,
        life_opts(),
        IrRenderOpts::default(),
    );
    let rendered = render_ir_text(&report);
    assert!(rendered.starts_with("IR Report\nsummary:"));
    assert!(rendered.contains("=== Parsed IR ==="));
    assert!(rendered.contains("=== Optimized IR ==="));
    assert!(rendered.contains("reduction_percent="));
    assert!(rendered.contains("Shift { src: 0, dst: 1, amount: 1, dir: Left"));
}

#[test]
fn emit_ir_text_report_truncates_long_linear_programs() {
    let src = "+".repeat(32);
    let parsed = parse_only(&src);
    let optimized = optimize_with_opts(parsed.clone(), life_opts());
    let report = build_ir_report(
        &src,
        &parsed,
        &optimized,
        life_opts(),
        IrRenderOpts {
            max_lines: 4,
            ..IrRenderOpts::default()
        },
    );
    let rendered = render_ir_text(&report);
    assert!(rendered.contains("... <"));
    assert!(
        report
            .parsed
            .as_ref()
            .or_invariant("required value")
            .truncated
    );
}

#[test]
fn emit_ir_no_elide_restores_full_dump() {
    let parsed = parse_only("+++[->++<]>.<.");
    let optimized = optimize_with_opts(parsed.clone(), life_opts());
    let report = build_ir_report(
        "+++[->++<]>.<.",
        &parsed,
        &optimized,
        life_opts(),
        IrRenderOpts {
            no_elide: true,
            ..IrRenderOpts::default()
        },
    );
    let rendered = render_ir_text(&report);
    assert!(rendered.contains(&format_ir(&parsed)));
    assert!(rendered.contains(&format_ir(&optimized)));
    assert!(
        !report
            .parsed
            .as_ref()
            .or_invariant("required value")
            .truncated
    );
    assert!(
        !report
            .optimized
            .as_ref()
            .or_invariant("required value")
            .truncated
    );
}

#[test]
fn emit_ir_json_report_contains_schema_and_sections() {
    let parsed = parse_only("+++[->++<]>.<.");
    let optimized = optimize_with_opts(parsed.clone(), life_opts());
    let report = build_ir_report(
        "+++[->++<]>.<.",
        &parsed,
        &optimized,
        life_opts(),
        IrRenderOpts {
            format: IrOutputFormat::Json,
            section: IrSectionSelection::Both,
            ..IrRenderOpts::default()
        },
    );
    let rendered = render_ir_json(&report);
    assert!(rendered.contains("\"version\": \"bf-ir-report/v3\""));
    assert!(rendered.contains("\"summary\""));
    assert!(rendered.contains("\"parsed\""));
    assert!(rendered.contains("\"optimized\""));
    assert!(rendered.contains("\"kind\": \"shift\""));
}
