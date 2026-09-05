use super::*;
use crate::RequiredErrorExt;

fn fixture(records: &str, count: usize) -> String {
    format!(
        "{HASHLIFE_SNAPSHOT_MAGIC}\ngeneration 7\norigin 0 0\nroot N0@0\nnodes {count}\n{records}"
    )
}

#[test]
fn snapshot_parser_rejects_excess_records_and_mismatched_child_levels() {
    for (text, expected) in [
        (
            fixture("node 1 D@0 L@0 D@0 D@0\n", 0),
            "exceeds declared count",
        ),
        (
            fixture("node 2 D@0 L@0 D@0 D@0\n", 1),
            "children one level lower",
        ),
    ] {
        let error = read_snapshot(Cursor::new(text.as_bytes()))
            .error_or_invariant("malformed records must fail before reconstruction");
        assert!(
            error.to_string().contains(expected),
            "expected {expected}: {error}"
        );
    }
}

#[test]
fn snapshot_parser_rejects_scratch_budget_before_reading() {
    let text = fixture("node 1 D@0 L@0 D@0 D@0\n", 1);
    let mut reader = Cursor::new(text.as_bytes());
    let error = read_snapshot_with_limit(&mut reader, SNAPSHOT_PARSE_SCRATCH_BYTES - 1)
        .error_or_invariant("parser scratch must fit before reading");
    assert!(error.allocation_bytes().is_some(), "wrong failure: {error}");
    assert_eq!(
        reader.position(),
        0,
        "budget rejection consumed external input"
    );
}

#[test]
fn snapshot_import_accounts_for_owned_records_and_mapping_simultaneously() {
    let text = fixture("node 1 D@0 L@0 D@0 D@0\n", 1);
    let parsed = read_snapshot(Cursor::new(text.as_bytes())).or_invariant("fixture parses");
    let mut engine = HashLifeEngine::default();
    let before = engine.node_count();
    let retained = engine.allocated_bytes() as u128;
    engine.begin_allocation_transaction(retained + parsed.allocated_bytes());
    let error = engine
        .import_snapshot(parsed)
        .error_or_invariant("mapping cannot reuse the owned record allocation's budget");
    assert!(
        error.allocation_bytes().is_some(),
        "wrong import error: {error}"
    );
    assert_eq!(
        engine.node_count(),
        before,
        "mapping rejection published nodes"
    );
    assert_eq!(
        engine.allocation_transient_reserved, 0,
        "aborted import retained scratch charges"
    );
    assert!(engine.take_allocation_failure().is_some());

    engine.begin_allocation_transaction(u128::MAX);
    let parsed = read_snapshot(Cursor::new(text.as_bytes())).or_invariant("retry fixture parses");
    let (_, _, _, generation) = engine.import_snapshot(parsed).or_invariant("retry fits");
    assert_eq!(generation, 7);
    assert_eq!(engine.take_allocation_failure(), None);
}
