Feature roadmap
version 0.0.1

TODO
-Combine controls
-streamline workflow
-make the .fx file load faster or figure out why it takes so long to load

---current implemented features---
<Viewport>
-added grid lines
-added length markers(world space metric)
-added navigation (Houdini navigation)
-add transport gizmo
<Modeling>
-add point 
-add simple shape inserts
-add ground plain
<UI>
<Diagnostic tools>
<Texture tools>
<Camera/Lighting>
<Quality of life>
<uncatogorizable>

---Future implemented/WIP features---
<Viewport>

<Model>
-add studio/environment presets
<UI>
-add multiview
<Diagnostic tools>
-add point numbers(view port)
-add point graph(NEW PANEL)
-add attribute graph
<Texture tools>
<Camera/Lighting>
-add Camera
-add camera presets
-add Lighting presets
<Quality of life>
-Skeleton insert/viewing tools
<uncatogorizable>
-add a renderer

---Nice to have features---
<Viewport>
<Model>
<UI>
<Diagnostic tools>
<Texture tools>
<Camera/Lighting>
<Quality of life>
<uncatogorizable>

---Crash logs---
Caused by:
  In Device::create_buffer, label = 'egui_index_buffer'
    Buffer size 300658536 is greater than the maximum buffer size (268435456)





• Based on the app’s current direction—a Houdini-style workflow with
  Blender-like viewport controls—these are the strongest missing
  features to implement next:

  1. Scene outliner

  A proper hierarchical scene tree for imported models, generated
  geometry, lights, cameras, and environments. It should support
  selection, renaming, visibility, locking, duplication, and deletion.

  2. Multiple generated objects

  The current generated-geometry system appears centered around one
  active generated shape. Support any number of cubes, planes, spheres,
  and other primitives, each with an independent transform, material,
  gizmo, subdivisions, and snapping state.

  3. Save and load projects

  Serialize the entire scene:

  - Objects and transforms
  - Generated geometry
  - Materials and texture paths
  - Lights and look-at targets
  - Grid and viewport settings
  - HDRI configuration

  4. Redo and complete undo

  Add Cmd+Shift+Z on macOS and Ctrl+Y or Ctrl+Shift+Z elsewhere. Extend
  history to model imports, texture changes, and clearing imported
  models.

  5. Duplicate, delete, copy, and paste objects

  Implement:

  - Cmd/Ctrl+C — copy
  - Cmd/Ctrl+V — paste
  - Cmd/Ctrl+D — duplicate
  - Delete/Backspace — delete selection

  This becomes especially important once multiple generated objects are
  supported.

  6. Transform modes

  Add explicit toolbar shortcuts:

  - G — move
  - R — rotate
  - S — scale
  - X, Y, or Z — constrain the active transformation
  - Numeric input for exact values
  - Local, world, parent, and view coordinate spaces

  7. Better snapping

  Extend bottom/grid snapping with:

  - Vertex snapping
  - Edge snapping
  - Face snapping
  - Increment snapping
  - Surface placement
  - Rotation angle snapping
  - Hold Ctrl/Cmd for temporary snapping

  8. Object parenting and constraints

  Allow objects and lights to follow other scene objects. Useful
  constraints include:

  - Parent
  - Look At
  - Position
  - Rotation
  - Scale
  - Maintain Offset

  Use stable object IDs rather than array indexes for constraint
  targets.

  9. Camera objects

  Add actual scene cameras with focal length, sensor size, clipping
  controls, depth of field, camera locking, and a “View Through Camera”
  mode.

  10. Real shadow rendering

  Lights already have shadow-related properties, but the next major
  visual improvement would be actual shadow maps for point, spot,
  directional, and area lights.

  11. Selection improvements

  Add:

  - Multi-selection
  - Box selection
  - Lasso selection
  - Select all/invert
  - Selection outlines
  - Active object versus selected objects
  - Hide selected/isolate selection

  12. Material library and assignments

  Support multiple materials per object, face-level material
  assignments, reusable materials, drag-and-drop assignment, and a
  material browser.

  13. Mesh editing mode

  A basic edit mode would add substantial value:

  - Vertex, edge, and face selection
  - Extrude
  - Inset
  - Bevel
  - Loop cut
  - Merge
  - Delete/dissolve
  - Recalculate normals

  14. Transform pivot controls

  Provide object origin, selection center, bounding-box center, 3D
  cursor, individual origins, and custom pivot placement.

  15. Performance tools

  Add an optional profiler showing CPU frame time, GPU frame time, draw
  calls, triangles, UI tessellation cost, and memory usage. This would
  make issues like the UV-panel slowdown much easier to catch.

  The three I would prioritize are:

  1. Multiple generated objects with stable IDs
  2. A scene outliner
  3. Project save/load

  Those features establish the scene architecture needed for reliable
  undo, copying, parenting, constraints, and future mesh editing.



  -need to move the cache files into a seperate folder in the app data
  -need to have a material save state so that it saves or creates a save file that can be loaded to have material libraries streamlined
  -fix the material assignment for generated basic shapes

  need to keep on working on the explorer
  -list view only
  -add favorites
  -add path feild
  -move path addy bar to the bottom of the window
  -make so it will open bigger
  -create a home button that can be set in the settings.
