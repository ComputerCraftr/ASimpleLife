use super::*;
use crate::render::ViewportController;

pub(super) fn request_viewport(worker: &WorkerHandle, state: &mut UiState, content: Rect) -> bool {
    if let Some(request) = next_request(state, content) {
        worker.set_viewport(request);
        return true;
    }
    false
}

fn next_request(state: &mut UiState, content: Rect) -> Option<ViewportRequest> {
    if content.width == 0 || content.height == 0 {
        return None;
    }
    let previous = state.viewport_request;
    if let Some(previous) = previous
        && !state.auto_viewport
        && let Some(origin) = state.manual_origin
        && (previous.width != content.width || previous.height != content.height)
    {
        let resized =
            ViewportController::new(usize::from(previous.width), usize::from(previous.height))
                .and_then(|mut viewport| {
                    viewport.set_origin(origin);
                    viewport.resize(usize::from(content.width), usize::from(content.height))?;
                    Ok(viewport.origin())
                });
        match resized {
            Ok(origin) => state.manual_origin = origin,
            Err(error) => state.notice = error.to_string(),
        }
    }
    let origin = if state.auto_viewport {
        None
    } else {
        state.manual_origin
    };
    let recenter_acknowledged = previous.is_some_and(|request| {
        state
            .frame
            .as_ref()
            .is_some_and(|frame| frame.viewport_revision == request.revision)
    });
    if previous.is_some_and(|request| {
        request.width == content.width
            && request.height == content.height
            && request.origin == origin
            && request.auto == state.auto_viewport
            && (!state.recenter || (request.recenter && !recenter_acknowledged))
    }) {
        return None;
    }
    let Some(revision) = previous.map_or(Some(1), |request| request.revision.checked_add(1)) else {
        state.notice = "camera request revisions exhausted".into();
        return None;
    };
    let request = ViewportRequest {
        revision,
        width: content.width,
        height: content.height,
        origin,
        auto: state.auto_viewport,
        recenter: state.recenter,
    };
    state.viewport_request = Some(request);
    Some(request)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequiredExt;

    #[test]
    fn manual_resize_roundtrip_preserves_center_without_replaying_old_origin() {
        let mut state = UiState::new(&Config::default());
        state.auto_viewport = false;
        state.manual_origin = Some((100, 200));
        state.recenter = false;
        next_request(&mut state, Rect::new(0, 0, 20, 10)).or_invariant("initial request");
        let resized =
            next_request(&mut state, Rect::new(0, 0, 30, 15)).or_invariant("resize request");
        assert_eq!(resized.origin, Some((95, 195)));
        assert!(
            next_request(&mut state, Rect::new(0, 0, 30, 15)).is_none(),
            "idle UI resent stale viewport"
        );
        let restored =
            next_request(&mut state, Rect::new(0, 0, 20, 10)).or_invariant("restore request");
        assert_eq!(restored.origin, Some((100, 200)));
    }

    #[test]
    fn pending_recenter_survives_mailbox_replacement_and_rejects_stale_frames() {
        let mut state = UiState::new(&Config::default());
        let first =
            next_request(&mut state, Rect::new(0, 0, 20, 10)).or_invariant("initial request");
        assert!(next_request(&mut state, Rect::new(0, 0, 20, 10)).is_none());
        let second =
            next_request(&mut state, Rect::new(0, 0, 30, 15)).or_invariant("resize request");
        assert!(second.recenter, "resize lost unacknowledged Home action");
        let mut stale = super::super::tests::snapshot(1, (50, 50));
        stale.viewport_revision = first.revision;
        accept_frame(&mut state, stale);
        assert!(
            state.frame.is_none(),
            "old viewport frame replaced resized state"
        );
        let mut current = super::super::tests::snapshot(1, (0, 0));
        current.viewport_revision = second.revision;
        accept_frame(&mut state, current);
        assert!(!state.recenter);
        assert!(next_request(&mut state, Rect::new(0, 0, 30, 15)).is_none());
    }

    #[test]
    fn manual_focus_ack_is_not_replaced_by_the_previous_origin() {
        let mut state = UiState::new(&Config::default());
        state.auto_viewport = false;
        state.recenter = false;
        let request =
            next_request(&mut state, Rect::new(0, 0, 40, 15)).or_invariant("navigation request");
        let mut frame = super::super::tests::snapshot(1, (900, 900));
        frame.viewport_revision = request.revision;
        accept_frame(&mut state, frame);
        assert_eq!(state.manual_origin, Some((900, 900)));
        assert!(
            next_request(&mut state, Rect::new(0, 0, 40, 15)).is_none(),
            "ack generated stale origin replay"
        );
        state.manual_origin = Some((901, 900));
        let newer = next_request(&mut state, Rect::new(0, 0, 40, 15)).or_invariant("newer pan");
        let mut stale = super::super::tests::snapshot(1, (0, 0));
        stale.viewport_revision = request.revision;
        accept_frame(&mut state, stale);
        assert_eq!(
            state.manual_origin,
            Some((901, 900)),
            "stale navigation ack replaced pan"
        );
        assert!(newer.revision > request.revision);
    }
}
