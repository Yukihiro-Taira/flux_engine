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
- **Geometry import** — load OBJ and FBX files, inspect mesh groups, and work with generated primitives and a ground plane.
- **Materials and textures** — edit PBR materials, assign materials to meshes and faces, preview results, and experiment with a node-based material graph.
- **Lighting and environments** — edit scene lights and load HDR or EXR environments with intensity, exposure, and rotation controls.
- **Geometry inspection** — explore UV layouts, normals, point overlays, and mesh statistics.
- **Project persistence** — save and reopen `.fx` scenes; export and import material graphs as JSON.
- **Asset caching** — cache processed geometry, textures, and environments to reduce repeated import work.

## Getting started

You need a Rust toolchain with support for the 2024 edition, your platform's native build tools, and a GPU with a compatible graphics driver. Run these commands from the directory containing `Cargo.toml`:

```sh
cargo run --release
```

The desktop app opens in a borderless fullscreen window with an empty scene. No bundled models or textures are required.

### Your first scene

1. Import an `.obj` or `.fbx` model using the geometry import controls.
2. Adjust its materials and load your own textures.
3. Add scene lighting or choose an `.hdr` or `.exr` environment.
4. Explore the model using the viewport controls.
5. Save your work as an `.fx` project.

You can also open an existing project at startup:

```sh
cargo run --release -- /path/to/scene.fx
```

> **Houdini RAT files:** `.rat` environment conversion requires Houdini's `iconvert` utility. The app looks for a local installation; you can also set `HFS` to your Houdini installation directory. HDR and EXR imports do not require Houdini.

## Controls

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

## Local assets and saves

Bring your own models, textures, and environments. Assets are selected at runtime and are not copied into the build.

The repository's `.gitignore` excludes local asset folders, common model and image formats, `.fx` projects, material graph JSON files, generated caches, and build output. Keep personal work in `res/`, `assets/`, or `saves/` so related files stay together and out of version control.

Project saves include asset paths. Keep referenced assets available when reopening a scene; do not assume a save is a self-contained asset bundle.

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
