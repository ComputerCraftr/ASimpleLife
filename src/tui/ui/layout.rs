use super::*;
use ratatui::widgets::Wrap;

const HELP: &str = "Space: pause/resume\n.: exact single step\n+/-: change pace\n[/]: halve/double generation quantum\nf: advance by generations\ng: go to absolute generation\nArrows: manual pan\nShift+Arrows: pan ten cells\nTab / Shift+Tab: next / previous active group\nAuto navigation pins the selected group; manual navigation moves once.\na: toggle auto tracking (off/on releases pin)\nHome: recenter once\nc: classify/cancel classification\nn: new Life seed\nb: Brainfuck source\no: open file\nCtrl-S: save snapshot\nPgUp/PgDn: scroll status/help\nq / Ctrl-C: quit\nEsc: close dialog/help, otherwise quit";

fn panels(area: Rect) -> [Rect; 3] {
    let status_height = (area.height / 3).clamp(4, 10);
    let footer_height = if area.width >= 72 { 5 } else { 4 }.min(area.height.saturating_sub(7));
    let universe_height = area.height.saturating_sub(status_height + footer_height);
    [
        Rect::new(area.x, area.y, area.width, status_height),
        Rect::new(area.x, area.y + status_height, area.width, universe_height),
        Rect::new(
            area.x,
            area.y + status_height + universe_height,
            area.width,
            footer_height,
        ),
    ]
}

pub(super) fn universe_content(area: Rect) -> Rect {
    if area.width < 24 || area.height < 12 {
        return Rect::default();
    }
    Block::default()
        .borders(Borders::ALL)
        .inner(panels(area)[1])
}

pub(super) fn draw_ui(frame: &mut ratatui::Frame<'_>, state: &mut UiState) {
    let area = frame.area();
    if area.width < 24 || area.height < 12 {
        frame.render_widget(
            Paragraph::new("Terminal too small\nResize or q to quit").wrap(Wrap { trim: true }),
            area,
        );
        return;
    }
    if state.help {
        frame.render_widget(ratatui::widgets::Clear, area);
        state.status_scroll = draw_scrollable(
            frame,
            area,
            HELP,
            "Controls | PgUp/PgDn | Esc close",
            state.status_scroll,
        );
        return;
    }
    let [status, universe, footer] = panels(area);
    let text = if let Some(snapshot) = state.frame.as_ref() {
        format!(
            "generation={} | view generation={}\n{}\npopulation={}\nbackend={} {} quantum={} pace={}ms\nsource={}\nclassification={}\n{}{}{}",
            state
                .authoritative
                .as_ref()
                .map_or(snapshot.generation, |current| current.generation),
            snapshot.generation,
            state.notice,
            snapshot.population,
            snapshot.backend,
            if state
                .authoritative
                .as_ref()
                .map_or(snapshot.running, |current| current.running)
            {
                "running"
            } else {
                "paused"
            },
            state
                .authoritative
                .as_ref()
                .map_or(snapshot.quantum, |current| current.quantum),
            FRAME_INTERVALS_MS[state.speed_index],
            snapshot.source,
            state
                .analysis
                .as_ref()
                .map_or_else(|| "not requested".to_string(), |update| update.describe()),
            snapshot.status,
            if snapshot.output.is_empty() {
                ""
            } else {
                "\noutput="
            },
            snapshot.output
        )
    } else {
        state.notice.clone()
    };
    state.status_scroll = draw_scrollable(
        frame,
        status,
        &text,
        if area.width >= 40 {
            "ASimpleLife | PgUp/PgDn"
        } else {
            "Status PgUp/PgDn"
        },
        state.status_scroll,
    );
    // Keep the last accepted sample visible while a new camera request is
    // pending or fails. Its generation is labelled independently above.
    let snapshot = state.frame.as_ref();
    draw_universe(frame, universe, snapshot);
    draw_footer(frame, footer, state);
}

fn draw_scrollable(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    text: &str,
    title: &str,
    scroll: u16,
) -> u16 {
    let paragraph = Paragraph::new(text).wrap(Wrap { trim: false });
    let inner = Block::default().borders(Borders::ALL).inner(area);
    let maximum = paragraph
        .line_count(inner.width)
        .saturating_sub(usize::from(inner.height));
    let scroll = scroll.min(u16::try_from(maximum).unwrap_or(u16::MAX));
    frame.render_widget(
        paragraph
            .scroll((scroll, 0))
            .block(Block::default().borders(Borders::ALL).title(title)),
        area,
    );
    scroll
}

fn draw_footer(frame: &mut ratatui::Frame<'_>, area: Rect, state: &UiState) {
    if let Some(prompt) = &state.prompt {
        let text = format!("{:?}> {}", prompt.kind, prompt.value);
        // Keep the input's tail visible, including wide Unicode and wrapped paths.
        draw_scrollable(frame, area, &text, "Enter / Esc", u16::MAX);
    } else {
        let text = if area.width >= 72 {
            "Space pause  . step  +/- pace  [] quantum  f/g advance\narrows pan  a auto  Home center  Tab active group\nn/b/o source  Ctrl-S save  ? help  q quit"
        } else {
            "Space pause . step\n? all controls q quit"
        };
        draw_scrollable(frame, area, text, "Controls", 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequiredExt;
    use ratatui::backend::TestBackend;

    fn screen(terminal: &Terminal<TestBackend>) -> String {
        let buffer = terminal.backend().buffer();
        let mut text = String::new();
        for row in 0..buffer.area.height {
            for column in 0..buffer.area.width {
                text.push_str(buffer[(column, row)].symbol());
            }
            text.push('\n');
        }
        text
    }

    #[test]
    fn resizing_layout_uses_exact_bordered_content_and_keeps_footer_visible() {
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).or_invariant("test terminal");
        let mut state = UiState::new(&Config::default());
        let mut snapshot = super::super::tests::snapshot(1, (0, 0));
        snapshot.generation = u64::MAX;
        snapshot.population = u128::MAX;
        state.frame = Some(snapshot);
        for (width, height) in [(80, 24), (120, 40), (40, 18), (80, 24)] {
            terminal.backend_mut().resize(width, height);
            terminal
                .resize(Rect::new(0, 0, width, height))
                .or_invariant("resize");
            let content = universe_content(Rect::new(0, 0, width, height));
            let snapshot = state.frame.as_mut().or_invariant("fixture frame");
            snapshot.grid = crate::bitgrid::BitGrid::from_cells(&[
                (0, 0),
                (
                    Coord::from(content.width - 1),
                    Coord::from(content.height - 1) * 2 + 1,
                ),
            ]);
            terminal
                .draw(|frame| draw_ui(frame, &mut state))
                .or_invariant("draw");
            let rendered = screen(&terminal);
            assert!(
                rendered.contains("18446744073709551615"),
                "generation cut off at {width}x{height}:\n{rendered}"
            );
            assert!(
                rendered.contains("q quit"),
                "quit help missing at {width}x{height}:\n{rendered}"
            );
            assert_eq!(
                terminal.backend().buffer()[(content.x, content.y)].symbol(),
                "▀"
            );
            assert_eq!(
                terminal.backend().buffer()[(content.right() - 1, content.bottom() - 1)].symbol(),
                "▄",
                "sampled bottom/right edge hidden behind borders at {width}x{height}"
            );
        }
    }

    #[test]
    fn long_status_and_unicode_prompt_tails_are_reachable() {
        let mut terminal = Terminal::new(TestBackend::new(40, 18)).or_invariant("test terminal");
        let mut state = UiState::new(&Config::default());
        state.notice = format!("{} END-OF-STATUS", "detail ".repeat(60));
        state.status_scroll = u16::MAX;
        state.prompt = Some(Prompt {
            kind: PromptKind::Open,
            value: format!("{} /final-file.hls", "世界/".repeat(30)),
        });
        terminal
            .draw(|frame| draw_ui(frame, &mut state))
            .or_invariant("draw");
        let rendered = screen(&terminal);
        assert!(
            rendered.contains("END-OF-STATUS"),
            "status tail unreachable:\n{rendered}"
        );
        assert!(
            rendered.contains("/final-file.hls"),
            "input cursor tail hidden:\n{rendered}"
        );
        assert!(
            state.status_scroll < u16::MAX,
            "scroll position remained beyond the content"
        );
        state.status_scroll = state.status_scroll.saturating_sub(3);
        terminal
            .draw(|frame| draw_ui(frame, &mut state))
            .or_invariant("scroll up");
        assert!(
            !screen(&terminal).contains("END-OF-STATUS"),
            "scrolling up from the clamped end had no effect"
        );
    }

    #[test]
    fn tiny_terminal_does_not_request_an_invalid_universe() {
        for (width, height) in [(0, 0), (1, 1), (20, 8)] {
            let area = Rect::new(0, 0, width, height);
            assert_eq!(universe_content(area), Rect::default());
            let mut terminal =
                Terminal::new(TestBackend::new(width, height)).or_invariant("tiny terminal");
            terminal
                .draw(|frame| draw_ui(frame, &mut UiState::new(&Config::default())))
                .or_invariant("tiny draw");
        }
    }

    #[test]
    fn source_error_identifies_the_rejected_input_without_scrolling_at_eighty_columns() {
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).or_invariant("test terminal");
        let mut state = UiState::new(&Config::default());
        let mut snapshot = super::super::tests::snapshot(1, (0, 0));
        snapshot.generation = 575;
        snapshot.population = 4;
        snapshot.source = "block".to_string();
        snapshot.status = "running | viewport=auto largest active active=0/0".to_string();
        state.frame = Some(snapshot);
        state.notice = "unknown Life seed \"not-a-life-seed\"".to_string();
        terminal
            .draw(|frame| draw_ui(frame, &mut state))
            .or_invariant("source failure frame");
        let rendered = screen(&terminal);
        for expected in [
            "generation=575",
            "unknown Life seed \"not-a-life-seed\"",
            "source=block",
            "population=4",
        ] {
            assert!(
                rendered.contains(expected),
                "missing {expected:?} after source failure:\n{rendered}"
            );
        }
    }
}
