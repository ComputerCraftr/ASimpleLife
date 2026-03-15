use super::*;

#[test]
fn super_c_payload_split_uses_explicit_stats_sentinel() {
    let output = "memo hits: user payload\nstill payload\n=== BF SUPER RUNTIME STATS ===\nwork dispatches: 5\nwork loop iterations: 2\nwork ops: 6\nmemo hits:   3\nmemo misses: 4\n";
    assert_eq!(
        split_super_c_payload(output).trim_end(),
        "memo hits: user payload\nstill payload"
    );
    assert_eq!(super_work_stat(output, "work dispatches:"), 5);
    assert_eq!(super_work_stat(output, "work loop iterations:"), 2);
    assert_eq!(super_work_stat(output, "work ops:"), 6);
    assert_eq!(memo_stat(output, "memo hits:"), 3);
}

#[test]
fn super_c_payload_split_accepts_windows_line_endings() {
    let output = "user payload\r\n\r\n=== BF SUPER RUNTIME STATS ===\r\nwork dispatches: 5\r\nmemo hits: 3\r\n";
    assert_eq!(split_super_c_payload(output).trim_end(), "user payload");
    assert_eq!(super_work_stat(output, "work dispatches:"), 5);
    assert_eq!(memo_stat(output, "memo hits:"), 3);
}

#[test]
fn plain_c_payload_split_uses_explicit_stats_sentinel() {
    let output = "plain payload\n=== BF PLAIN RUNTIME STATS ===\nwork dispatches: 9\nwork loop iterations: 4\nwork ops: 7\n";
    assert_eq!(split_plain_c_payload(output).trim_end(), "plain payload");
    assert_eq!(plain_stat(output, "work dispatches:"), 9);
    assert_eq!(plain_stat(output, "work loop iterations:"), 4);
    assert_eq!(plain_stat(output, "work ops:"), 7);
}

#[test]
fn plain_c_payload_split_accepts_windows_line_endings() {
    let output = "plain payload\r\n=== BF PLAIN RUNTIME STATS ===\r\nwork ops: 7\r\n";
    assert_eq!(split_plain_c_payload(output).trim_end(), "plain payload");
    assert_eq!(plain_stat(output, "work ops:"), 7);
}
