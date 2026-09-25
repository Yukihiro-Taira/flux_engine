<div align="center">

# learn_wgpu

### A real-time 3D viewport and material editor built in Rust.

Import geometry. Shape the lighting. Explore materials.

![Rust](https://img.shields.io/badge/Rust-2024_edition-f97316?style=flat-square&logo=rust)
![wgpu](https://img.shields.io/badge/wgpu-29-6366f1?style=flat-square)
![egui](https://img.shields.io/badge/UI-egui-38bdf8?style=flat-square)
![Status](https://img.shields.io/badge/status-in_development-eab308?style=flat-square)

[Features](#features) · [Getting started](#getting-started) · [Controls](#controls) · [Development](#development)

</div>

---

`learn_wgpu` is an experimental desktop 3D editor for exploring real-time rendering, materials, and lighting. It combines a GPU-rendered viewport with an interactive editor, Houdini-style camera navigation, and a workflow centered on importing your own assets.

Built with **Rust**, **wgpu**, **WGSL**, **egui**, and **winit**. The project is under active development, and workflows and save formats may change.

## Features

- **Interactive viewport** — perspective and orthographic views, camera framing, a world-space grid, and navigation and transform gizmos.
- **Geometry import** — load OBJ, FBX, USD, USDA, USDC, and USDZ files, inspect mesh groups, and work with generated primitives and a ground plane.
- **Materials and textures** — edit PBR materials, assign materials to meshes and faces, preview results, and experiment with a node-based material graph.
- **Transparency** — texture alpha, opacity controls, cutout masks, double-sided surfaces, transparent previews, and alpha-aware shadows.
- **Lighting and environments** — edit scene lights and load HDR or EXR environments with intensity, exposure, and rotation controls.
- **Visual demos** — create reversible product deconstruction views with equal-distance group separation and repeatable random arrangements.
- **Geometry inspection** — explore UV layouts, normals, point overlays, and mesh statistics. Authored UVs retain their scale and placement; overlapping groups use whole-tile offsets without repacking texture islands.
- **Personal material library** — import textures, save reusable material presets with owned texture copies, and add them to any scene from Explorer.
- **Persistent preferences** — viewport settings and Explorer view choices survive restarts; restore defaults from Settings without deleting materials.
- **Project persistence** — save and reopen `.fx` scenes; export and import material graphs as JSON.
- **Asset caching** — cache processed geometry, textures, and environments to reduce repeated import work.

## Getting started

You need a Rust toolchain with support for the 2024 edition, your platform's native build tools, and a GPU with a compatible graphics driver. Run these commands from the directory containing `Cargo.toml`:

```sh
cargo run --release
```

The desktop app opens in a borderless fullscreen window with an empty scene. No bundled models or textures are required.

### Your first scene

1. Import an OBJ, FBX, or USD-family model using the geometry import controls.
2. Adjust its materials and load your own textures.
3. Add scene lighting or choose an `.hdr` or `.exr` environment.
4. Explore the model using the viewport controls.
5. Save your work as an `.fx` project.

You can also open an existing project at startup:

```sh
cargo run --release -- /path/to/scene.fx
```

> **Houdini RAT files:** `.rat` environment conversion requires Houdini's `iconvert` utility. The app looks for a local installation; you can also set `HFS` to your Houdini installation directory. HDR and EXR imports do not require Houdini.

## USD import

USD support uses the official [OpenUSD](https://openusd.org/release/index.html) runtime, including its composition engine and binary/package readers. Install its isolated Python backend once (Python 3.10–3.14):

```sh
python3 scripts/setup_usd.py
```

The installer creates `.usd-runtime/`, which is ignored by Git. OBJ/FBX import does not require this backend. For a packaged app, put `usd-runtime` beside the executable, install it in the personal data directory, or set `LEARN_WGPU_USD_PYTHON` to the Python executable of an environment containing the pinned packages in `scripts/usd-requirements.txt`. The app never downloads dependencies during import.

Use **Object → Model Import**, drag a file from Explorer, drop a file onto the Object panel, or pass its path at startup. `.usd` detects its underlying format; `.usda`, `.usdc`, and `.usdz` use the same workflow. Imports run in the background, preserve the current scene if decoding fails, and frame the imported model automatically.

Supported data includes:

- Composed layers, references, payloads, and authored variant selections.
- Hierarchical transforms, units, up-axis, visibility, mesh orientation, authored normals, indexed/interpolated UVs, polygon triangulation, holes, and material subsets.
- Native instances, nested point instancers, and cube/sphere/cylinder/cone/capsule/plane primitives.
- UsdPreviewSurface base color, opacity, metallic, roughness, normal, and emission, including texture channels, color spaces, scale/bias, and a UV set/transform per material.
- Playback and scrubbing from the fixed bottom timeline, including sampled transforms, points, visibility, and baked skeletal deformation. The chosen time is saved with `.fx` projects.
- Imported cameras under **USD cameras & lights → View…**; supported scene lights can be added to the existing Lighting tools from the same section. HDR/EXR dome textures use the environment importer.

USDZ texture bytes are resolved through OpenUSD and copied into content-addressed `usd-assets/` storage in the personal data directory. The importer does not extract archive paths or modify source USD layers. These textures work with scene saves and **Save to User Library**. Back up the personal data directory along with projects.

This is an evaluated import into the editor, not a live USD stage editor or a USD exporter. Subdivision surfaces currently use their control mesh; curves, volumes, renderer-specific shaders, multiple UV sets on one material, lens shift, and some lighting/shading features cannot be reproduced exactly. **Import notes** reports unsupported or approximated features and missing assets. Animation frames are evaluated in the background and cached for playback. USD scenes are recomposed on every import so changes to referenced layers are not hidden by a root-file-only cache.

```sh
# Real format, composition, material, animation, and package tests
.usd-runtime/bin/python -m unittest discover -s tests -p 'test_usd*.py' -v

# Include the Rust → OpenUSD bridge integration test
cargo test -- --include-ignored
```

To check the 20-second CPU import budget against a local model (OpenUSD conversion, JSON transfer, and Rust mesh preparation):

```sh
LEARN_WGPU_USD_BENCHMARK_ASSET="/path/to/model.usdz" cargo test --offline benchmark_usd_preparation -- --ignored --nocapture
```

The benchmark does not include GPU upload or the first rendered frame. Import time depends on geometry, textures, storage, and hardware. Geometry conversion reuses transformed source values, and texture channel conversion uses native lookup tables without reducing mesh detail or changing the texture resolution limit.

## Autosave, backups, and crash recovery

Desktop builds create a recovery copy every **30 seconds** while a scene has unsaved changes. Configure the interval (10–300 seconds), disable automatic copies, or save a copy immediately under **Settings → Autosave & Recovery**. Recovery writes run in the background and never overwrite your `.fx` project. Playback time and inspector selection alone do not mark a scene dirty.

Closing the app, loading another project, replacing an imported model, or clearing it prompts **Save and continue / Discard changes / Cancel** when necessary. **Save** updates the current project; **Save As…** chooses a new file. Use **Cmd/Ctrl+S** to save and **Cmd/Ctrl+Shift+S** for Save As when the material graph is closed (the graph retains its own save shortcut).

Project saves use a temporary file and atomic replacement. Before replacing an existing project, the app keeps its previous contents as a backup. Five previous saves are retained per project; choose **Settings → Autosave & Recovery → Restore previous save…** to inspect an older version. Restoring a backup marks the scene unsaved and leaves the current project file unchanged until you save.

After an interrupted session, the next launch offers recovery. Choose **Recover**, select an older copy, discard that session's recovery, or keep it for later. Five autosave versions are retained per session, and a damaged newest copy falls back to an older readable copy. Active sessions are locked so one running app instance cannot recover another's live work. Recovery data and backups live under `recovery/` in the personal app data directory, outside the repository.

Recovery restores the latest completed snapshot; changes since that snapshot may be lost after a crash. Models, textures, and environments remain external references and must still be available. Autosave pauses while an import is incomplete so a partial load does not replace the recovery copy.

## Workspace and materials

The right-hand tabs share one inspector position and size: Object, Geometry Inspection, Lighting, Scene Outliner, Materials, Demos, UV Map, and Settings. Part selection adds a Selected Part tab. Switch tabs to change tools; click the active tab to hide the inspector. Panels stay above the timeline and the open material graph. Explorer docks separately on the left when space permits; narrow windows show one panel at a time.

Materials uses a searchable scrolling catalog and a separate selected-material editor. Only visible thumbnail rows allocate previews; thumbnails share their rendering pipeline. Mesh and face assignments have bounded scrolling lists under **Assignments**. Long names truncate in the catalog and show their full text on hover. Importing maps, saving user presets, duplication, and drag-and-drop assignment remain available.

## Playback performance

Normal `cargo run` enables optimization while keeping debug information. Use `cargo run --release` for the fastest build; the first optimized build takes longer.

Texture-group controls edit one selected group at a time, so large imports do not build hundreds of offscreen editors each frame. Camera, lighting, and material uploads run once after UI and animation changes. Playback follows elapsed time and skips source samples when needed instead of slowing the animation clock.

60 FPS requires each displayed frame to stay within 16.7 ms. Very large scenes, deformation, transparency, high-resolution displays, and shadows can exceed that budget; a universal 60 FPS guarantee is not possible. For local profiling, run `LEARN_WGPU_FRAME_PROFILE=/tmp/viewer-frames.csv cargo run --release`. The optional CSV records frame intervals and CPU stages (UI, animation, preparation, surface acquisition, encoding, submission/presentation); it does not measure GPU execution directly or include asset paths. Profiling is disabled by default.

## Animation timeline

The fixed bottom timeline uses a Houdini-style playbar with compact transport buttons, a numbered frame ruler, playhead, and editable playback range. Editor windows, the orientation gizmo, settings button, and material graph stay above its reserved dock. Animated USD and FBX imports expose their source range and frame rate, with play/pause, first/last frame, stepping, scrubbing, looping, and playback speed controls. USD uses the composed stage, including skeletal and blend-shape baking; FBX evaluates the default take, skinning, blend shapes, and available baked geometry caches. OBJ assets remain static.

Transform-only USD animation keeps local mesh buffers on the GPU and updates per-group matrices, including shapes that start at zero scale. Playback follows elapsed time at the source rate, skipping display frames when needed instead of slowing the animation. Deforming assets use background geometry evaluation; uncached frames display an evaluation indicator. A bounded 128 MiB / 240-frame memory cache accelerates revisiting frames. The viewport camera and edited materials remain under user control: animated camera/light parameters and animated material inputs are not currently applied during playback. Reimport to refresh externally changed source files. Material overrides are matched by mesh name; face overrides require stable topology.

## Selecting model parts

Select an imported model and press **T** to enter part selection. Click geometry to highlight its group in cyan. A model with one mesh group uses connected UV islands instead; UV seams separate the selectable regions.

The **Selected Geometry** panel applies a material override to that part, creates an independent material for editing, or removes the override. Open the material editor from this panel to import textures and adjust material settings. Overrides persist in project saves and apply to all instances of that model. Press **T** again to return to object selection; material changes stay applied.

## Controls

Press **T** with an imported model selected to toggle group / UV-island selection and its material override panel.

Press **G** in the viewport to hide or show the grid. This shortcut is inactive while typing in a text field. Grid visibility is saved with `.fx` projects.

Hold **Space** or **Alt** while interacting with the viewport:

- **Left drag** — tumble around the scene.
- **Middle drag** — pan.
- **Right drag** — dolly.
- **Ctrl + left drag** — tilt.
- **Ctrl + right drag** — lens zoom.
- **Shift** — use finer navigation movement.

While holding **Space** or **Alt**, use:

- **1 / 2 / 3 / 4** — perspective / top / front / right view.
- **O** — toggle perspective and orthographic projection.
- **A** — frame all; **F** or **G** — frame the selection.
- **H** — return to the home grid view.
- **Z** — set the tumble pivot from the cursor.

## Transparency

Open **Transparency** in the material controls (Materials, Object, generated primitives, or texture groups):

- **Automatic / Alpha blend** uses base texture alpha and the **Opacity** multiplier. Zero opacity makes the surface invisible.
- **Opaque** ignores opacity and alpha, useful for images with unwanted alpha channels.
- **Cutout** uses **Alpha cutoff** for crisp holes in foliage, fences, and decals.
- **Dithered** uses patterned coverage when overlapping transparent geometry causes sorting artifacts.
- **Use base texture alpha** can be disabled independently. **Double sided** renders both sides of thin surfaces.
- **Reset transparency** restores the default controls.

Texture alpha works through the existing texture import workflows. USD imports preserve opacity, opacity thresholds, and double-sided mesh settings; OBJ/MTL imports honor `d`, `Tr`, and grayscale `map_d`; FBX imports retain material opacity and available opacity maps. Separate imported opacity maps are folded into RGBA textures in personal storage. Settings persist in `.fx` scenes, duplicated materials, and the user library. Group controls edit an assigned scene material when one is present.

Transparent surfaces blend after opaque geometry, sorted by group and instance depth. Intersecting surfaces or triangles inside one group can still have sorting artifacts; use Cutout or Dithered where appropriate. Translucent shadows use filtered coverage, and transparency does not simulate glass refraction.

## Local assets and saves

Bring your own models, textures, and environments. Assets are selected at runtime and are not copied into the build.

The repository's `.gitignore` excludes local asset folders, common model and image formats, `.fx` projects, material graph JSON files, generated caches, and build output. Keep personal work in `res/`, `assets/`, or `saves/` so related files stay together and out of version control.

Project saves include asset paths. Keep referenced assets available when reopening a scene; do not assume a save is a self-contained asset bundle.

## Product deconstruction demo

Import a model with at least two mesh groups, then open **Demos → Product Deconstruction**:

1. Turn on **Enable deconstruction**.
2. Adjust **Distance** to move every group by the same world-space distance.
3. Toggle **Random directions** and use **New random arrangement** to explore variations.
4. Use **Frame result** to fit the separated model in view.
5. Choose **Reassemble** to return to the original geometry.

Shading, shadows, wireframes, and inspection overlays follow the separated parts. Save an `.fx` project to preserve the demo controls and random seed. The effect does not modify the source model; a single merged mesh must be split into groups before import.

## Your material library

In **Materials**, choose **Import Texture as Material…**, or create a material and import its base-color, normal, metal/roughness, and emissive maps. Adjust the material, then choose **Save to User Library**. Each save creates a reusable preset and copies its texture maps into personal app storage.

Emissive maps have intensity, tint, and hue controls, alongside a separate base-color hue adjustment. These controls are available for scene materials, generated shapes, model groups, and the UV editor; projects and user-library presets retain them. Choose the texture type when importing a new material or importing from the UV editor. OBJ/MTL imports also recognize `map_Ke` emissive maps.

Open **Explorer → User Library** to search presets and choose **Add to Scene**. Scene edits do not change a saved preset; save again to keep another version. Moving the original imported textures does not break saved library materials.

Viewport settings save automatically. They remain active when importing a model, opening a project, or restarting the app. Use **Settings → Reset settings to default: Reset** to restore preferences; saved materials remain intact.

Personal data is stored outside the repository:

- **macOS:** `~/Library/Application Support/learn_wgpu/`
- **Windows:** `%APPDATA%/learn_wgpu/`
- **Linux:** `$XDG_DATA_HOME/learn_wgpu/`, or `~/.local/share/learn_wgpu/`

Back up this directory to preserve your settings and library. For isolated development or testing, set `LEARN_WGPU_USER_DATA_DIR` to a separate local directory.

## Development

```sh
# Check the project
cargo check

# Run the Rust tests
cargo test

# Build an optimized executable
cargo build --release
```

<details>
<summary><strong>Source layout</strong></summary>

- `src/lib.rs` — application lifecycle, viewport rendering, editor workflows, and scene persistence.
- `src/resources.rs` — model import, geometry processing, and model caches.
- `src/usd_import.rs` and `scripts/usd_import.py` — official OpenUSD bridge, composed scene conversion, and persistent packaged textures.
- `src/model.rs` — mesh data and rendering helpers.
- `src/material.rs` — GPU materials and texture handling.
- `src/material_library.rs` — scene materials and assignments.
- `src/material_graph.rs` — node-based material editor.
- `src/environment.rs` — environment loading and processing.
- `src/lighting/` — light data, editor controls, gizmos, and shadows.
- `src/houdini_navigation.rs` — viewport camera interaction.
- `src/*.wgsl` and `src/lighting/*.wgsl` — GPU shaders.

</details>

Desktop is the focus of these instructions. The code also contains WebAssembly support, but desktop file import and dialogs are platform-specific; a browser build needs its own hosting setup.

## Contributing

Bug reports, focused improvements, and rendering experiments are welcome. For bugs, include reproduction steps, your operating system, GPU, and any relevant error output. Share assets only when you have permission to distribute them.
