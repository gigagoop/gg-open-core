# SpaceGraph review plan (entrypoint: examples/space_graph/ex_overview.py)

## Entrypoint trace (what I walked)
- `examples/space_graph/ex_overview.py`: creates a `SpaceGraph`, then calls `scatter`, `plot`, `mesh`, `add_sphere`, `text`, `text3d`, `add_image`, `add_arrow`, `lit_mesh`, `show`.
- `gigagoop/viz/space_graph/space_graph.py`: client-side API, request serialization, process management, validation helpers.
- `gigagoop/viz/space_graph/engine/server_engine.py`: ZMQ server, message handling, node construction.
- `gigagoop/viz/space_graph/engine/base_engine.py`: window, render loop, UI, camera, and event handlers.
- `gigagoop/viz/space_graph/nodes/*`: OpenGL resources and per-primitive rendering.
- `gigagoop/viz/space_graph/primitives/builder.py`: mesh building utilities for primitives and Unreal OBJ import.

## Bugs and correctness issues (actionable)
1. **Grayscale image handling crashes.**
   - File: `gigagoop/viz/space_graph/nodes/image.py`
   - `np.dstack(image, image, image)` is invalid; `np.dstack` expects a sequence. This throws a `TypeError` when `image.ndim == 2`.
   - Fix: `np.dstack([image, image, image])` or `np.repeat(image[..., None], 3, axis=2)`.

2. **Vector/position length mismatch check is broken.**
   - File: `gigagoop/viz/space_graph/space_graph.py`
   - In `vectors()`, `assert positions.shape == positions.shape` is a no-op. Mismatched lengths silently drop vectors in `zip`.
   - Fix: `assert positions.shape == vectors.shape`.

3. **Per-arrow zero-length direction can divide by zero.**
   - File: `gigagoop/viz/space_graph/nodes/arrow.py` (also `nodes/cylinder.py` for similar logic)
   - `length = np.linalg.norm(X)` followed by `X = X / length` without guarding `length == 0`. Client-side validation only checks *max* length.
   - Fix: skip or warn for any zero-length vector before normalization.

4. **Material flags are inconsistent and can raise `AttributeError`.**
   - File: `gigagoop/viz/space_graph/mesh.py`
   - `__init__` sets `self.material`, but `has_materials` / `valid_textures` read `self.materials`. If `materials` was never set, access fails.
   - Fix: initialize `self.materials = material` or update properties to use `self.material`.

5. **SizedScatter is wired to a missing shader.**
   - File: `gigagoop/viz/space_graph/nodes/sized_scatter.py`
   - Shader name `sized_points` is referenced but no `sized_points.vert/frag` exists under `gigagoop/viz/space_graph/shaders/`.
   - Fix: add the shader or change to `points` and add a size attribute in the existing shader.

6. **Camera node is not safe to instantiate.**
   - File: `gigagoop/viz/space_graph/nodes/camera.py`
   - `get_vbo()` / `get_vao()` are `pass`. `Node.__init__` expects buffers and will store `None`, and `Node.destroy()` will call `.release()` on `None`.
   - Fix: implement VBO/VAO properly or make this class not inherit `Node` and avoid `Node.__init__`.

7. **Texture dtype mismatch risk.**
   - Files: `gigagoop/viz/space_graph/nodes/image.py`, `nodes/text3d.py`, `nodes/unreal_mesh.py`
   - `engine.ctx.texture(..., data=uint8, dtype='f1')` uses an 8-bit *float* dtype flag while data is `uint8`. If the backend does not treat `'f1'` as `GL_UNSIGNED_BYTE`, colors or alpha can be wrong.
   - Fix: verify the moderngl dtype mapping; likely should be `dtype='u1'` for `uint8`.

8. **Validation in MaterialMesh is too strict for common data.**
   - File: `gigagoop/viz/space_graph/nodes/material_mesh.py`
   - `assert vertices.dtype == np.float64` rejects `float32` inputs. This can cause surprising failures for users who cast earlier.
   - Fix: accept `float32`/`float64` or cast internally.

9. **Documentation mismatch on mouse wheel step size.**
   - File: `gigagoop/viz/space_graph/engine/base_engine.py`
   - Comment says fine step is `0.01` but code uses `0.1`. Minor, but misleading.

## Easy speedups (low-risk, high ROI)
1. **Avoid repeated `zmq.Poller()` allocation per message.**
   - File: `gigagoop/viz/space_graph/space_graph.py` (`_send_message`)
   - Create a poller once and reuse it, or use `self._socket.poll()` to avoid per-call allocations.

2. **Use `np.asarray(..., dtype=...)` to avoid copies.**
   - File: `gigagoop/viz/space_graph/engine/server_engine.py`
   - Replace `np.array(...).astype(...)` with `np.asarray(..., dtype=...)` to avoid copies when already correct dtype.

3. **Batch text3d rendering where possible.**
   - File: `gigagoop/viz/space_graph/space_graph.py` (`text3d`)
   - Each label generates a new texture and node. For many labels, cache rendered glyphs or build a single atlas + one node.

4. **Wireframe edge generation can be vectorized.**
   - File: `gigagoop/viz/space_graph/space_graph.py` (`wireframe`)
   - `edges_as_lines` uses a Python loop; can use `vertices[unique_edges].reshape(-1, 3)` for a pure numpy path.

5. **Reduce per-node uniform writes when camera is static.**
   - File: `gigagoop/viz/space_graph/nodes/node.py`
   - `render()` always updates view/projection. Cache a camera revision counter and only update when changed.

6. **Consider multipart ZMQ with zero-copy for large arrays.**
   - File: `gigagoop/viz/space_graph/space_graph.py` / `engine/server_engine.py`
   - Current pickle path copies large arrays. ZMQ multipart with `memoryview` (or pickle protocol 5 buffers) can cut transfer time.

## Feature ideas (10 total)
1. **Node lifecycle & updates.** Create IDs for nodes so users can update positions/colors, remove nodes, or clear scenes without restarting.
2. **Interactive picking.** Click to select a node/triangle/point and display metadata; optionally highlight selection.
3. **Camera bookmarks & animation.** Save/recall camera poses, add smooth fly-throughs or keyframed paths.
4. **Scene I/O.** Save/load scenes to JSON or glTF; export meshes/point clouds to PLY/OBJ/glTF.
5. **Screenshot + video capture.** One-shot capture, plus simple recording with frame capture to disk.
6. **Layer/collection UI.** Group nodes into layers with visibility toggles, opacity sliders, and color overrides.
7. **Lighting controls.** UI for ambient/directional lights, toggles for flat/smooth shading, and light gizmos.
8. **Large point cloud support.** LOD decimation, progressive streaming, or chunked rendering for million+ points.
9. **Primitive expansion.** Add boxes, planes, capsules, arrows with labels, and bounding boxes for quick debugging.
10. **Rich text & labels.** Text billboarding, background plates, alignment helpers, and auto-collision for label overlaps.

## Additional observations / tech debt
- There are two engines (`gigagoop/viz/space_graph/engine.py` and `engine/base_engine.py` + `server_engine.py`). The former appears legacy and is not referenced by `SpaceGraph`. Consider deprecating or clearly documenting which engine is current to reduce drift.
- Mesh loading utilities are duplicated (`gigagoop/viz/space_graph/mesh.py` vs `primitives/builder.py`), with different BOM handling behavior. Consolidating would reduce inconsistencies.
- Test coverage is focused on `gigagoop/coord`; visualization code lacks tests (especially message handling, node creation, and texture paths).
