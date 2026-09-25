//! Route native menu/Dock Quit through the editor's asynchronous save prompt.
//! Winit 0.30 handles window closes, but NSApplication's terminate: otherwise
//! terminates directly. Preserve and restore that method for the event-loop lifetime.
use objc2::{
    runtime::{AnyClass, AnyObject, Imp, Method, Sel},
    sel,
};
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicBool, Ordering},
};
use winit::window::Window;
static REQUESTED: AtomicBool = AtomicBool::new(false);
static WINDOW: Mutex<Option<Weak<Window>>> = Mutex::new(None);
type Terminate = extern "C" fn(*mut AnyObject, Sel, *mut AnyObject);
extern "C" fn request_quit(_: *mut AnyObject, _: Sel, _: *mut AnyObject) {
    REQUESTED.store(true, Ordering::Release);
    if let Ok(window) = WINDOW.lock() {
        if let Some(window) = window.as_ref().and_then(Weak::upgrade) {
            window.request_redraw();
        }
    }
}
pub(crate) fn register_window(window: &Arc<Window>) {
    if let Ok(mut slot) = WINDOW.lock() {
        *slot = Some(Arc::downgrade(window));
    }
}
pub(crate) fn take_request() -> bool {
    REQUESTED.swap(false, Ordering::AcqRel)
}
pub(crate) struct QuitHook {
    method: &'static Method,
    original: Imp,
}
pub(crate) fn install() -> anyhow::Result<QuitHook> {
    let class = AnyClass::get("NSApplication")
        .ok_or_else(|| anyhow::anyhow!("NSApplication is not initialized"))?;
    let method = class
        .instance_method(sel!(terminate:))
        .ok_or_else(|| anyhow::anyhow!("NSApplication has no terminate: method"))?;
    // SAFETY: `terminate:` has the Objective-C signature void(id, SEL, id).
    // The replacement uses that exact ABI, only sets a flag and wakes the window,
    // and never calls into the editor's mutable state from AppKit's callback.
    let replacement = unsafe { std::mem::transmute::<Terminate, Imp>(request_quit) };
    let original = unsafe { method.set_implementation(replacement) };
    Ok(QuitHook { method, original })
}
impl Drop for QuitHook {
    fn drop(&mut self) {
        // SAFETY: Restore the original implementation with its original signature.
        unsafe {
            self.method.set_implementation(self.original);
        }
        if let Ok(mut slot) = WINDOW.lock() {
            *slot = None;
        }
    }
}
