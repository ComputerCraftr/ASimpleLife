use terminal_size::{Height, Width};

pub fn terminal_size(default_width: usize, default_height: usize) -> (usize, usize) {
    let probed = terminal_size::terminal_size()
        .map(|(Width(width), Height(height))| (usize::from(width), usize::from(height)));
    let columns = std::env::var("COLUMNS").ok();
    let lines = std::env::var("LINES").ok();
    resolve_terminal_size(
        probed,
        columns.as_deref(),
        lines.as_deref(),
        (default_width, default_height),
    )
}

fn resolve_terminal_size(
    probed: Option<(usize, usize)>,
    columns: Option<&str>,
    lines: Option<&str>,
    fallback: (usize, usize),
) -> (usize, usize) {
    let (probed_width, probed_height) = probed.unwrap_or_default();
    let width = positive_dimension(probed_width)
        .or_else(|| columns.and_then(parse_positive_dimension))
        .unwrap_or(fallback.0.max(1));
    let height = positive_dimension(probed_height)
        .or_else(|| lines.and_then(parse_positive_dimension))
        .unwrap_or(fallback.1.max(1));
    (width, height)
}

fn positive_dimension(value: usize) -> Option<usize> {
    (value > 0).then_some(value)
}

fn parse_positive_dimension(value: &str) -> Option<usize> {
    value.trim().parse().ok().and_then(positive_dimension)
}

#[cfg(test)]
mod tests {
    use super::resolve_terminal_size;

    #[test]
    fn native_terminal_size_wins_over_environment_fallbacks() {
        assert_eq!(
            resolve_terminal_size(Some((132, 43)), Some("90"), Some("30"), (80, 24)),
            (132, 43)
        );
    }

    #[test]
    fn non_tty_environment_dimensions_support_container_terminals() {
        assert_eq!(
            resolve_terminal_size(None, Some(" 101 "), Some("37"), (80, 24)),
            (101, 37)
        );
    }

    #[test]
    fn invalid_and_partial_dimensions_fall_back_independently() {
        assert_eq!(
            resolve_terminal_size(Some((0, 55)), Some("0"), Some("bad"), (80, 24)),
            (80, 55)
        );
        assert_eq!(
            resolve_terminal_size(None, Some("120"), None, (80, 24)),
            (120, 24)
        );
        assert_eq!(resolve_terminal_size(None, None, None, (0, 0)), (1, 1));
    }
}
