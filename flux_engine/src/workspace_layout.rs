//! Shared geometry for the inspector, optional Explorer, and tab rail.
pub(crate) const RAIL_WIDTH: f32 = 148.0;
const GAP: f32 = 12.0;

pub(crate) struct Layout {
    pub inspector: egui::Rect,
    pub explorer: egui::Rect,
    pub dual: bool,
}
impl Layout {
    pub fn new(workspace: egui::Rect) -> Self {
        let right = (workspace.right() - RAIL_WIDTH - GAP).max(workspace.left() + 100.0);
        let width = (workspace.width() * 0.36)
            .clamp(340.0, 600.0)
            .min((right - workspace.left() - GAP).max(80.0));
        let inspector = egui::Rect::from_min_max(
            egui::pos2(right - width, workspace.top() + GAP),
            egui::pos2(
                right,
                (workspace.bottom() - GAP).max(workspace.top() + GAP + 60.0),
            ),
        );
        let left_width = inspector.left() - workspace.left() - 2.0 * GAP;
        let dual = left_width >= 300.0;
        let explorer = egui::Rect::from_min_size(
            egui::pos2(workspace.left() + GAP, inspector.top()),
            egui::vec2(
                if dual { left_width.min(620.0) } else { width },
                inspector.height(),
            ),
        );
        Self {
            inspector,
            explorer,
            dual,
        }
    }
}
pub(crate) fn available(context: &egui::Context) -> egui::Rect {
    context
        .data(|data| data.get_temp::<egui::Rect>(egui::Id::new("editor_workspace")))
        .unwrap_or_else(|| crate::timeline::workspace_rect(context))
}
pub(crate) fn viewport_left(context: &egui::Context) -> f32 {
    if context
        .data(|data| data.get_temp::<bool>(egui::Id::new("explorer_visible")))
        .unwrap_or(false)
    {
        Layout::new(available(context)).explorer.right() + GAP
    } else {
        available(context).left() + GAP
    }
}
pub(crate) fn bottom_inset(context: &egui::Context) -> f32 {
    context.content_rect().bottom() - available(context).bottom()
}
fn window(title: &'static str, context: &egui::Context, rect: egui::Rect) -> egui::Window<'static> {
    egui::Window::new(title)
        .id(egui::Id::new(("workspace_panel_v1", title)))
        .fixed_pos(rect.min)
        // Window sizing describes its body, excluding the title and frame.
        .fixed_size((rect.size() - egui::vec2(16.0, 44.0)).max(egui::vec2(64.0, 24.0)))
        .constrain_to(available(context))
        .collapsible(false)
        .resizable(false)
        .vscroll(true)
}
pub(crate) fn inspector(title: &'static str, context: &egui::Context) -> egui::Window<'static> {
    window(title, context, Layout::new(available(context)).inspector)
}
pub(crate) fn explorer(context: &egui::Context) -> egui::Window<'static> {
    window(
        "Asset Explorer",
        context,
        Layout::new(available(context)).explorer,
    )
    .vscroll(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn panels_fit_and_explorer_never_overlaps_inspector_in_dual_mode() {
        for width in [480.0, 800.0, 1280.0, 1920.0, 3840.0] {
            for height in [320.0, 624.0, 984.0] {
                let workspace =
                    egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(width, height));
                let layout = Layout::new(workspace);
                assert!(workspace.contains_rect(layout.inspector));
                assert!(workspace.contains_rect(layout.explorer));
                if layout.dual {
                    assert!(layout.explorer.right() + GAP <= layout.inspector.left());
                }
                assert!(layout.inspector.right() <= width - RAIL_WIDTH);
            }
        }
    }
}
