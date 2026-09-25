//! Opt-in frame timing; disabled unless LEARN_WGPU_FRAME_PROFILE names an output file.
use std::io::Write;
use std::{
    sync::{Mutex, OnceLock},
    time::Instant,
};
struct Capture {
    file: std::io::BufWriter<std::fs::File>,
    previous: Instant,
}
fn capture() -> &'static Option<Mutex<Capture>> {
    static CAPTURE: OnceLock<Option<Mutex<Capture>>> = OnceLock::new();
    CAPTURE.get_or_init(|| {
        let path = std::env::var_os("LEARN_WGPU_FRAME_PROFILE")?;
        let mut file = std::io::BufWriter::new(std::fs::File::create(path).ok()?);
        writeln!(
            file,
            "playing,interval_ms,ui,animation,prepare,acquire,encode,submit_present"
        )
        .ok()?;
        Some(Mutex::new(Capture {
            file,
            previous: Instant::now(),
        }))
    })
}
pub struct Frame {
    last: Option<Instant>,
    times: Vec<f64>,
}
impl Frame {
    pub fn new() -> Self {
        Self {
            last: capture().as_ref().map(|_| Instant::now()),
            times: Vec::new(),
        }
    }
    pub fn mark(&mut self, _name: &str) {
        if let Some(last) = self.last {
            let now = Instant::now();
            self.times
                .push(now.duration_since(last).as_secs_f64() * 1000.0);
            self.last = Some(now);
        }
    }
    pub fn finish(self, playing: bool) {
        if let Some(capture) = capture() {
            if let Ok(mut c) = capture.lock() {
                let now = Instant::now();
                let interval = now.duration_since(c.previous).as_secs_f64() * 1000.0;
                c.previous = now;
                let _ = write!(c.file, "{playing},{interval:.3}");
                for value in self.times {
                    let _ = write!(c.file, ",{value:.3}");
                }
                let _ = writeln!(c.file);
            }
        }
    }
}
