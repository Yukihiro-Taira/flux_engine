//! Crash recovery and atomic project writes. Recovery never overwrites a user's project.
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const VERSIONS: usize = 5;
static SERIAL: AtomicU64 = AtomicU64::new(0);
fn stamp() -> String {
    format!(
        "{:020}-{:010}-{:06}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    )
}

pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let temporary = parent.join(format!(".fx-write-{}.tmp", stamp()));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)
            .with_context(|| format!("Cannot replace {}", path.display()))?;
        // Persist the directory entry as well as the file on Unix platforms.
        #[cfg(unix)]
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}
fn versions(directory: &Path, extension: &str) -> Result<Vec<PathBuf>> {
    let mut paths = fs::read_dir(directory)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some(extension))
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}
fn prune(directory: &Path, extension: &str) -> Result<()> {
    let paths = versions(directory, extension)?;
    for path in paths.iter().take(paths.len().saturating_sub(VERSIONS)) {
        fs::remove_file(path)?;
    }
    Ok(())
}
fn backup_directory(root: &Path, path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    // Stable FNV-1a key: separate histories for equal filenames in different folders.
    let hash = absolute
        .as_os_str()
        .as_encoded_bytes()
        .iter()
        .fold(0xcbf29ce484222325u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        });
    Ok(root.join("backups").join(format!("{hash:016x}")))
}
pub(crate) fn save_project(root: &Path, path: &Path, bytes: &[u8]) -> Result<()> {
    if path.exists() {
        let directory = backup_directory(root, path)?;
        fs::create_dir_all(&directory)?;
        let previous = fs::read(path).context("Could not read the previous project for backup")?;
        atomic_write(&directory.join(format!("{}.fx", stamp())), &previous)?;
        // A failed backup aborts the save, preserving the previous project.
        prune(&directory, "fx")?;
    }
    atomic_write(path, bytes)
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Snapshot {
    schema: u32,
    pub created_ms: u64,
    pub project_path: String,
    pub project: crate::FxProject,
}
impl Snapshot {
    fn new(project: crate::FxProject, project_path: String) -> Self {
        Self {
            schema: 1,
            created_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            project_path,
            project,
        }
    }
}
pub(crate) fn read_snapshot(path: &Path) -> Result<Snapshot> {
    let snapshot: Snapshot = serde_json::from_slice(&fs::read(path)?)?;
    if snapshot.schema != 1
        || snapshot.project.format != "FX Scene Project"
        || snapshot.project.version > crate::FX_PROJECT_VERSION
    {
        bail!("Unsupported recovery snapshot version");
    }
    Ok(snapshot)
}
fn write_snapshot(directory: &Path, snapshot: &Snapshot) -> Result<()> {
    let bytes = serde_json::to_vec(snapshot)?;
    atomic_write(&directory.join(format!("{}.json", stamp())), &bytes)?;
    prune(directory, "json")
}
pub(crate) struct Candidate {
    pub directory: PathBuf,
    pub path: PathBuf,
    pub project_path: String,
    pub created_ms: u64,
    _lock: File,
}
fn mark_directory_clean(directory: &Path) -> Result<()> {
    let through = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    atomic_write(&directory.join("clean"), through.to_string().as_bytes())
}
fn clean_through(directory: &Path) -> u64 {
    match fs::read_to_string(directory.join("clean")) {
        Ok(value) => value.parse().unwrap_or(u64::MAX),
        Err(_) => 0,
    }
}
fn scan(root: &Path) -> Result<Vec<Candidate>> {
    let mut found = Vec::new();
    let sessions = root.join("sessions");
    fs::create_dir_all(&sessions)?;
    for entry in fs::read_dir(sessions)? {
        let directory = entry?.path();
        if !directory.is_dir() {
            continue;
        }
        let Ok(lock) = OpenOptions::new()
            .read(true)
            .write(true)
            .open(directory.join("lock"))
        else {
            continue;
        };
        if lock.try_lock().is_err() {
            continue;
        } // Another running app owns this session.
        let clean_through = clean_through(&directory);
        for path in versions(&directory, "json")?.into_iter().rev() {
            // An incomplete/corrupt newest snapshot does not hide older good versions.
            if let Ok(snapshot) = read_snapshot(&path) {
                if snapshot.created_ms <= clean_through {
                    continue;
                }
                found.push(Candidate {
                    directory,
                    path,
                    project_path: snapshot.project_path,
                    created_ms: snapshot.created_ms,
                    _lock: lock,
                });
                break;
            }
        }
    }
    found.sort_by_key(|entry| std::cmp::Reverse(entry.created_ms));
    Ok(found)
}

pub(crate) enum Action {
    Close,
    Open(PathBuf),
    Import(PathBuf, Option<f64>),
    Clear,
    Recover(usize),
    Backup(PathBuf),
}
struct Job {
    snapshot: Snapshot,
    fingerprint: Vec<u8>,
    generation: u64,
}
struct Completed {
    fingerprint: Vec<u8>,
    generation: u64,
    result: Result<(), String>,
}

pub(crate) struct Recovery {
    pub root: Option<PathBuf>,
    directory: Option<PathBuf>,
    _lock: Option<File>,
    sender: Option<mpsc::SyncSender<Job>>,
    receiver: Option<mpsc::Receiver<Completed>>,
    pending: bool,
    generation: u64,
    saved: Vec<u8>,
    last_autosaved: Vec<u8>,
    pub dirty: bool,
    pub last_check: Instant,
    last_write: Instant,
    pub message: String,
    pub candidates: Vec<Candidate>,
    pub show_recovery: bool,
    pub action: Option<Action>,
    pub exit_requested: bool,
    pub approved_import: bool,
    pub loading_path: Option<String>,
    pub settle_loaded: bool,
    pub recovered_source: Option<PathBuf>,
    pub recovering: bool,
}
impl Recovery {
    pub fn new() -> Self {
        Self::new_at(crate::user_data::data_dir().map(|root| root.join("recovery")))
    }
    fn new_at(root: Result<PathBuf>) -> Self {
        let mut state = Self {
            root: None,
            directory: None,
            _lock: None,
            sender: None,
            receiver: None,
            pending: false,
            generation: 0,
            saved: Vec::new(),
            last_autosaved: Vec::new(),
            dirty: false,
            last_check: Instant::now(),
            last_write: Instant::now(),
            message: String::new(),
            candidates: Vec::new(),
            show_recovery: false,
            action: None,
            exit_requested: false,
            approved_import: false,
            loading_path: None,
            settle_loaded: false,
            recovering: false,
            recovered_source: None,
        };
        let setup = (|| -> Result<()> {
            let root = root?;
            fs::create_dir_all(&root)?;
            state.candidates = scan(&root)?;
            state.show_recovery = !state.candidates.is_empty();
            let directory = root.join("sessions").join(stamp());
            fs::create_dir_all(&directory)?;
            let lock = OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(directory.join("lock"))?;
            lock.try_lock()?;
            let (sender, jobs) = mpsc::sync_channel::<Job>(1);
            let (completed, receiver) = mpsc::channel();
            let worker_directory = directory.clone();
            std::thread::spawn(move || {
                while let Ok(job) = jobs.recv() {
                    let result = write_snapshot(&worker_directory, &job.snapshot)
                        .map_err(|e| format!("{e:#}"));
                    if completed
                        .send(Completed {
                            fingerprint: job.fingerprint,
                            generation: job.generation,
                            result,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            });
            state.root = Some(root);
            state.directory = Some(directory);
            state._lock = Some(lock);
            state.sender = Some(sender);
            state.receiver = Some(receiver);
            Ok(())
        })();
        if let Err(error) = setup {
            state.message = format!("Recovery unavailable: {error:#}");
        }
        state
    }
    pub fn modal_open(&self) -> bool {
        self.action.is_some() || self.show_recovery
    }
    pub fn mark_clean(&mut self) {
        if let Some(directory) = &self.directory {
            if let Err(error) = mark_directory_clean(directory) {
                self.message = format!("Could not mark recovery session clean: {error:#}");
            }
        }
    }
    pub fn baseline(&mut self, fingerprint: Vec<u8>) {
        self.generation = self.generation.wrapping_add(1);
        self.saved = fingerprint;
        self.dirty = false;
        self.last_autosaved.clear();
        self.last_write = Instant::now();
        self.mark_clean();
    }
    pub fn track(&mut self, fingerprint: &[u8]) {
        self.dirty = self.saved != fingerprint;
    }
    pub fn poll(&mut self) {
        if let Some(receiver) = &self.receiver {
            match receiver.try_recv() {
                Ok(completed) => {
                    self.pending = false;
                    if completed.generation != self.generation {
                        return;
                    }
                    match completed.result {
                        Ok(()) => {
                            self.last_autosaved = completed.fingerprint;
                            self.last_write = Instant::now();
                            self.message = "Recovery copy saved".into();
                        }
                        Err(error) => {
                            self.last_write = Instant::now();
                            self.message = format!("Autosave failed: {error}");
                        }
                    }
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.pending = false;
                    self.sender = None;
                    self.message = "Autosave worker stopped".into();
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
    }
    pub fn due(&self, seconds: u64) -> bool {
        self.dirty
            && !self.pending
            && self.sender.is_some()
            && self.last_write.elapsed() >= Duration::from_secs(seconds)
    }
    pub fn enqueue(&mut self, project: crate::FxProject, path: String, fingerprint: Vec<u8>) {
        if fingerprint == self.last_autosaved || self.pending {
            return;
        }
        if let Some(sender) = &self.sender {
            match sender.try_send(Job {
                snapshot: Snapshot::new(project, path),
                fingerprint,
                generation: self.generation,
            }) {
                Ok(()) => {
                    self.pending = true;
                    self.message = "Saving recovery copy…".into();
                }
                Err(error) => self.message = format!("Autosave could not start: {error}"),
            }
        }
    }
}

/// Ignore transient playback/selection and whether an inspector tab is open.
/// The actual recovery file still retains the current timeline sample and view.
pub(crate) fn fingerprint(project: &mut crate::FxProject) -> Result<Vec<u8>> {
    let time = project
        .imported_model
        .as_mut()
        .and_then(|model| model.usd_time_code.take());
    let selected = project
        .imported_model
        .as_ref()
        .map(|model| model.selected_instance);
    if let Some(model) = &mut project.imported_model {
        model.selected_instance = 0;
    }
    let selected_light = project.lights.selected_light.take();
    let uv_open = std::mem::replace(&mut project.viewport.show_uv_map, false);
    let bytes = serde_json::to_vec(project).map_err(Into::into);
    if let Some(model) = &mut project.imported_model {
        model.usd_time_code = time;
        model.selected_instance = selected.unwrap_or(0);
    }
    project.lights.selected_light = selected_light;
    project.viewport.show_uv_map = uv_open;
    bytes
}

impl crate::State {
    fn document_busy(&self) -> bool {
        self.pending_model_load.is_some()
            || self.editor.pending_asset.is_some()
            || self.pending_project.is_some()
            || self.editor.pending_hdri.is_some()
            || !self.pending_uv_textures.is_empty()
            || self.editor.pending_texture.is_some()
            || self.editor.pending_normal.is_some()
            || self.editor.pending_metallic_roughness.is_some()
            || self.editor.pending_emissive.is_some()
    }
    pub(crate) fn refresh_document_dirty(&mut self) {
        let mut project = self.capture_fx_project();
        match fingerprint(&mut project) {
            Ok(bytes) => self.recovery.track(&bytes),
            Err(error) => {
                self.recovery.dirty = true;
                self.recovery.message = format!("Could not inspect unsaved changes: {error:#}");
            }
        }
    }
    pub(crate) fn request_document_action(&mut self, action: Action) {
        if self.recovery.action.is_some() {
            return;
        }
        self.refresh_document_dirty();
        if self.recovery.dirty || self.document_busy() {
            self.timeline.pause_for_edit();
            self.houdini_navigation = Default::default();
            self.recovery.action = Some(action);
        } else {
            self.perform_document_action(action);
        }
        self.window.request_redraw();
    }
    fn perform_document_action(&mut self, action: Action) {
        match action {
            Action::Close => {
                self.recovery.mark_clean();
                self.recovery.exit_requested = true;
            }
            Action::Open(path) => self.load_fx_project_now(path),
            Action::Import(path, time) => {
                self.recovery.approved_import = true;
                self.editor.usd_use_stage_start = time.is_none();
                self.editor.usd_time_code = time.unwrap_or(0.0);
                self.editor.pending_asset = Some(path);
            }
            Action::Clear => self.clear_imported_model_now(),
            Action::Recover(index) => {
                let Some(candidate) = self.recovery.candidates.get(index) else {
                    return;
                };
                match read_snapshot(&candidate.path) {
                    Ok(snapshot) => {
                        self.recovery.recovered_source = Some(candidate.directory.clone());
                        self.recovery.recovering = true;
                        self.recovery.show_recovery = false;
                        self.stage_fx_project(snapshot.project, snapshot.project_path);
                    }
                    Err(error) => {
                        self.recovery.message = format!("Recovery could not be opened: {error:#}")
                    }
                }
            }
            Action::Backup(path) => {
                let result = fs::read(&path)
                    .map_err(anyhow::Error::from)
                    .and_then(|bytes| {
                        serde_json::from_slice::<crate::FxProject>(&bytes).map_err(Into::into)
                    });
                match result {
                    Ok(project)
                        if project.format == "FX Scene Project"
                            && project.version <= crate::FX_PROJECT_VERSION =>
                    {
                        self.recovery.recovering = true;
                        self.stage_fx_project(project, self.project_path.clone());
                    }
                    Ok(_) => self.recovery.message = "Unsupported backup project version".into(),
                    Err(error) => {
                        self.recovery.message = format!("Backup could not be opened: {error:#}")
                    }
                }
            }
        }
    }
    pub(crate) fn stage_fx_project(&mut self, project: crate::FxProject, path: String) {
        self.fx_load_started = Some(Instant::now());
        self.recovery.loading_path = Some(path);
        self.recovery.settle_loaded = false;
        if let Some(model) = &project.imported_model
            && !model.path.is_empty()
        {
            self.editor.pending_asset = Some(PathBuf::from(&model.path));
        }
        self.pending_project = Some(project);
        self.editor.status = "FX project queued for loading".into();
    }
    pub(crate) fn recovery_load_failed(&mut self) {
        if self.recovery.recovering {
            self.recovery.show_recovery = true;
        }
        self.recovery.message = self.editor.status.clone();
        self.recovery.loading_path = None;
        self.recovery.settle_loaded = false;
        self.recovery.recovering = false;
        self.recovery.recovered_source = None;
    }
    pub(crate) fn update_recovery(&mut self) {
        self.recovery.poll();
        if self.document_busy() {
            return;
        }
        if self.recovery.settle_loaded {
            self.recovery.settle_loaded = false;
            if let Some(path) = self.recovery.loading_path.take() {
                self.project_path = path;
            }
            let mut project = self.capture_fx_project();
            if let Ok(bytes) = fingerprint(&mut project) {
                if self.recovery.recovering {
                    self.recovery.recovering = false;
                    self.recovery.generation = self.recovery.generation.wrapping_add(1);
                    self.recovery.saved.clear();
                    self.recovery.last_autosaved.clear();
                    self.recovery.dirty = true;
                    self.recovery.last_write = Instant::now() - Duration::from_secs(3600);
                    // Keep the original interrupted session until a new snapshot or manual save succeeds.
                    self.recovery.message = "Recovered scene — save it to keep your changes".into();
                } else {
                    self.recovery.baseline(bytes);
                }
            }
        }
        if self.recovery.last_check.elapsed() < Duration::from_secs(1) {
            return;
        }
        self.recovery.last_check = Instant::now();
        let mut project = self.capture_fx_project();
        match fingerprint(&mut project) {
            Ok(bytes) => {
                self.recovery.track(&bytes);
                if self.preferences.autosave_enabled
                    && self.recovery.due(self.preferences.autosave_seconds)
                {
                    self.recovery
                        .enqueue(project, self.project_path.clone(), bytes);
                }
            }
            Err(error) => {
                self.recovery.message = format!("Autosave could not inspect the project: {error:#}")
            }
        }
        if self.recovery.recovered_source.is_some()
            && !self.recovery.pending
            && !self.recovery.last_autosaved.is_empty()
        {
            self.finish_recovered_session();
        }
    }
    pub(crate) fn finish_recovered_session(&mut self) {
        if let Some(directory) = self.recovery.recovered_source.take() {
            match mark_directory_clean(&directory) {
                Ok(()) => self
                    .recovery
                    .candidates
                    .retain(|candidate| candidate.directory != directory),
                Err(error) => {
                    self.recovery.message =
                        format!("Recovered, but could not retire old snapshot: {error:#}")
                }
            }
        }
    }
    pub(crate) fn save_document(&mut self, save_as: bool) -> bool {
        if self.document_busy() {
            self.editor.status = "Wait for the current import before saving".into();
            return false;
        }
        let path = if !save_as && !self.project_path.is_empty() {
            Some(PathBuf::from(&self.project_path))
        } else {
            rfd::FileDialog::new()
                .add_filter("FX Project", &["fx"])
                .set_file_name("scene.fx")
                .save_file()
        };
        path.is_some_and(|path| self.save_fx_project(path))
    }
    pub(crate) fn recovery_settings_ui(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.heading("Autosave & Recovery");
        ui.checkbox(
            &mut self.preferences.autosave_enabled,
            "Save recovery copies automatically",
        );
        ui.add_enabled(
            self.preferences.autosave_enabled,
            egui::Slider::new(&mut self.preferences.autosave_seconds, 10..=300)
                .text("Interval (seconds)"),
        );
        ui.small("Keeps five recovery versions per session and five previous versions of each saved project. Models and textures remain external references.");
        ui.label(if self.recovery.dirty {
            "Unsaved changes"
        } else {
            "No unsaved changes"
        });
        if !self.recovery.message.is_empty() {
            ui.add(egui::Label::new(&self.recovery.message).wrap());
        }
        if ui
            .add_enabled(
                !self.document_busy(),
                egui::Button::new("Save recovery copy now"),
            )
            .clicked()
        {
            let mut project = self.capture_fx_project();
            if let Ok(bytes) = fingerprint(&mut project) {
                self.recovery.track(&bytes);
                if self.recovery.dirty {
                    self.recovery
                        .enqueue(project, self.project_path.clone(), bytes);
                } else {
                    self.recovery.message = "No unsaved changes to recover".into();
                }
            }
        }
        if ui.button("Review interrupted sessions…").clicked() {
            if let Some(root) = &self.recovery.root {
                match scan(root) {
                    Ok(mut found) => self.recovery.candidates.append(&mut found),
                    Err(error) => {
                        self.recovery.message = format!("Could not scan recovery: {error:#}")
                    }
                }
            }
            self.recovery.show_recovery = true;
        }
        if ui
            .add_enabled(
                !self.project_path.is_empty(),
                egui::Button::new("Restore previous save…"),
            )
            .clicked()
        {
            if let Some(root) = &self.recovery.root {
                match backup_directory(root, Path::new(&self.project_path)) {
                    Ok(directory) if directory.exists() => {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("FX backup", &["fx"])
                            .set_directory(directory)
                            .pick_file()
                        {
                            self.request_document_action(Action::Backup(path));
                        }
                    }
                    _ => {
                        self.recovery.message =
                            "No previous saves are available for this project yet".into()
                    }
                }
            }
        }
    }
    pub(crate) fn recovery_ui(&mut self, context: &egui::Context) {
        if self.recovery.action.is_some() {
            let mut decision = 0;
            let busy = self.document_busy();
            egui::Modal::new(egui::Id::new("unsaved_document_prompt")).show(context, |ui| {
                ui.set_max_width(420.0);
                ui.heading("Save your changes?");
                ui.label("Save before continuing, or discard the current unsaved changes.");
                if busy {
                    ui.label("An import is still running. Wait to save its completed result.");
                }
                if !self.editor.status.is_empty() {
                    ui.add(egui::Label::new(&self.editor.status).wrap());
                }
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(!busy, egui::Button::new("Save and continue"))
                        .clicked()
                    {
                        decision = 1;
                    }
                    if ui.button("Discard changes").clicked() {
                        decision = 2;
                    }
                    if ui.button("Cancel").clicked() {
                        decision = 3;
                    }
                });
            });
            if decision == 3 {
                self.recovery.action = None;
                if let Some(path) = &self.loaded_model_path {
                    self.editor.model_path_input = path.display().to_string();
                }
            } else if decision == 2 || (decision == 1 && self.save_document(false)) {
                if let Some(action) = self.recovery.action.take() {
                    self.perform_document_action(action);
                }
            }
            return;
        }
        if !self.recovery.show_recovery {
            return;
        }
        let mut recover = None;
        let mut older = None;
        let mut discard = None;
        let mut later = false;
        egui::Modal::new(egui::Id::new("recover_interrupted_session")).show(context, |ui| {
            ui.set_max_width(480.0);
            ui.heading("Recover interrupted work");
            ui.label("Recovery copies were found from earlier sessions. Recover a copy to inspect it before saving.");
            if self.recovery.candidates.is_empty() { ui.label("No interrupted sessions found."); }
            egui::ScrollArea::vertical().max_height(280.0).show(ui, |ui| {
                for (index, candidate) in self.recovery.candidates.iter().enumerate() {
                    ui.push_id(index, |ui| {
                        ui.group(|ui| {
                            let name = Path::new(&candidate.project_path).file_name().and_then(|name|name.to_str()).unwrap_or("Untitled scene");
                            ui.strong(name);
                            let age = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis().saturating_sub(u128::from(candidate.created_ms)) / 60_000;
                            ui.small(format!("Recovery saved {age} minutes ago"));
                            ui.horizontal(|ui| {
                                if ui.button("Recover").clicked() { recover = Some(index); }
                                if ui.button("Older copies…").clicked() { older = Some(index); }
                                if ui.button("Discard recovery").clicked() { discard = Some(index); }
                            });
                        });
                    });
                }
            });
            if !self.recovery.message.is_empty() { ui.add(egui::Label::new(&self.recovery.message).wrap()); }
            if ui.button("Keep for later / Close").clicked() { later = true; }
        });
        if later {
            self.recovery.show_recovery = false;
        }
        if let Some(index) = discard {
            let candidate = &self.recovery.candidates[index];
            match mark_directory_clean(&candidate.directory) {
                Ok(()) => {
                    self.recovery.candidates.remove(index);
                }
                Err(error) => {
                    self.recovery.message = format!("Could not discard recovery: {error:#}")
                }
            }
        }
        if let Some(index) = older {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Recovery copy", &["json"])
                .set_directory(&self.recovery.candidates[index].directory)
                .pick_file()
            {
                match read_snapshot(&path) {
                    Ok(snapshot) => {
                        self.recovery.candidates[index].path = path;
                        self.recovery.candidates[index].project_path = snapshot.project_path;
                        self.recovery.candidates[index].created_ms = snapshot.created_ms;
                        recover = Some(index);
                    }
                    Err(error) => {
                        self.recovery.message = format!("Cannot read recovery copy: {error:#}")
                    }
                }
            }
        }
        if let Some(index) = recover {
            self.request_document_action(Action::Recover(index));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("fx-recovery-test-{}", stamp()));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn project() -> crate::FxProject {
        // A self-contained scene schema; no external models or textures are needed.
        serde_json::from_value(serde_json::json!({
            "format":"FX Scene Project", "version":1, "imported_model":null, "generated_objects":[],
            "material": {"base_color":[1,1,1,1],"properties":[0,0.5,1,1],"options":[0,0,1,0],"inspection":[0,0,0,0],"base_color_path":"","normal_path":"","roughness_path":""},
            "material_graph": serde_json::to_value(crate::material_graph::MaterialGraph::default()).unwrap(),
            "lights":{"mode":serde_json::to_value(crate::lighting::ViewportLightingMode::DefaultLight).unwrap(),"lights":[],"selected_light":null,"area_samples":16,"environment_samples":64},
            "grid":{"visible":true,"size":10,"spacing":1,"color":[1,1,1,1]},
            "viewport":{"background":[0,0,0,1],"gizmo_always_at_bottom":false,"wireframe":false,"show_points":false,"point_size":7,"point_color":[1,1,1,1],"show_normals":false,"normal_mode":0,"normal_length":0.01,"normal_color":[1,1,1,1],"show_uv_map":false,"show_uv_overlay":false,"show_uv_texture":true,"show_uv_lines":true,"camera_eye":[1,1,1],"camera_target":[0,0,0],"camera_up":[0,0,1],"camera_fovy":45,"orthographic":false,"ortho_scale":1},
            "environment":{"path":"","disabled":false,"intensity":1,"exposure":0,"rotation":0}
        })).unwrap()
    }
    #[test]
    fn atomic_save_keeps_five_previous_versions_and_distinct_project_histories() {
        let tmp = Temp::new();
        let root = tmp.0.join("recovery");
        let path = tmp.0.join("scene.fx");
        for version in 0..8 {
            save_project(&root, &path, format!("version {version}").as_bytes()).unwrap();
        }
        assert_eq!(fs::read_to_string(&path).unwrap(), "version 7");
        let history = versions(&backup_directory(&root, &path).unwrap(), "fx").unwrap();
        assert_eq!(history.len(), 5);
        let contents: Vec<_> = history
            .iter()
            .map(|p| fs::read_to_string(p).unwrap())
            .collect();
        assert_eq!(
            contents,
            vec![
                "version 2",
                "version 3",
                "version 4",
                "version 5",
                "version 6"
            ]
        );
        assert_ne!(
            backup_directory(&root, &path).unwrap(),
            backup_directory(&root, &tmp.0.join("other/scene.fx")).unwrap()
        );
        assert!(!fs::read_dir(&tmp.0).unwrap().any(|e| {
            e.unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".fx-write-")
        }));
    }
    #[test]
    fn failed_backup_preserves_the_saved_project() {
        let tmp = Temp::new();
        let path = tmp.0.join("scene.fx");
        let root = tmp.0.join("blocked");
        fs::write(&path, b"original").unwrap();
        fs::write(&root, b"not a directory").unwrap();
        assert!(save_project(&root, &path, b"replacement").is_err());
        assert_eq!(fs::read(&path).unwrap(), b"original");
    }
    #[test]
    fn interrupted_session_recovers_and_live_sessions_are_not_offered() {
        let tmp = Temp::new();
        let root = tmp.0.join("recovery");
        let mut manager = Recovery::new_at(Ok(root.clone()));
        let mut scene = project();
        let fingerprint = fingerprint(&mut scene).unwrap();
        manager.track(&fingerprint);
        manager.enqueue(scene, "original.fx".into(), fingerprint);
        for _ in 0..1000 {
            manager.poll();
            if !manager.pending {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(!manager.pending, "background save should finish");
        assert!(manager.message.contains("saved"), "{}", manager.message);
        assert!(
            scan(&root).unwrap().is_empty(),
            "live session lock prevents recovery by another instance"
        );
        drop(manager); // Simulate an interrupted session: no clean marker.
        let candidates = scan(&root).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(
            read_snapshot(&candidates[0].path).unwrap().project_path,
            "original.fx"
        );
        assert!(
            scan(&root).unwrap().is_empty(),
            "recovery dialog owns its candidate lock too"
        );
        atomic_write(&candidates[0].directory.join("clean"), b"discarded").unwrap();
        drop(candidates);
        assert!(scan(&root).unwrap().is_empty());
    }
    #[test]
    fn corrupt_latest_copy_falls_back_and_recovery_versions_are_bounded() {
        let tmp = Temp::new();
        let root = tmp.0.join("recovery");
        let manager = Recovery::new_at(Ok(root.clone()));
        let directory = manager.directory.clone().unwrap();
        for index in 0..9 {
            write_snapshot(
                &directory,
                &Snapshot::new(project(), format!("project-{index}.fx")),
            )
            .unwrap();
        }
        assert_eq!(versions(&directory, "json").unwrap().len(), 5);
        fs::write(directory.join("zzzz.json"), b"{partial").unwrap();
        drop(manager);
        let candidates = scan(&root).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].project_path, "project-8.fx");
    }
    #[test]
    fn stale_autosave_completion_cannot_override_a_manual_save_baseline() {
        let tmp = Temp::new();
        let mut manager = Recovery::new_at(Ok(tmp.0.join("recovery")));
        let old_generation = manager.generation;
        manager.baseline(b"saved".to_vec());
        let (sender, receiver) = mpsc::channel();
        manager.receiver = Some(receiver);
        manager.pending = true;
        sender
            .send(Completed {
                fingerprint: b"old recovery".to_vec(),
                generation: old_generation,
                result: Ok(()),
            })
            .unwrap();
        manager.poll();
        assert!(!manager.pending);
        assert!(manager.last_autosaved.is_empty());
        manager.track(b"old recovery");
        assert!(manager.dirty);
    }
    #[test]
    fn playback_and_inspector_selection_do_not_dirty_a_scene_but_edits_do() {
        let mut scene = project();
        scene.imported_model = Some(crate::FxImportedModel {
            usd_time_code: Some(0.0),
            path: "example.usda".into(),
            instances: vec![],
            visible_instance_count: 0,
            selected_instance: 0,
            gizmo_at_bottom: vec![],
            group_visibility: vec![],
            group_texture_enabled: vec![],
            textures: vec![],
        });
        let original = fingerprint(&mut scene).unwrap();
        scene.imported_model.as_mut().unwrap().usd_time_code = Some(120.0);
        scene.imported_model.as_mut().unwrap().selected_instance = 3;
        scene.viewport.show_uv_map = true;
        assert_eq!(fingerprint(&mut scene).unwrap(), original);
        assert_eq!(
            scene.imported_model.as_ref().unwrap().usd_time_code,
            Some(120.0),
            "snapshot must keep its actual playback sample"
        );
        scene.material.base_color[0] = 0.2;
        assert_ne!(fingerprint(&mut scene).unwrap(), original);
    }
    #[test]
    fn old_preferences_enable_autosave_with_a_safe_default_interval() {
        let prefs: crate::user_data::Preferences = serde_json::from_str("{}").unwrap();
        assert!(prefs.autosave_enabled);
        assert_eq!(prefs.autosave_seconds, 30);
    }
}
