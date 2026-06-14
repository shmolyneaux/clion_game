# SHIM Engine

<img src="shimlang/logo_optimized.svg" alt="Shimlang logo" width="200"/>

The Shim Engine is a prototype game engine focused on quickly iterating on 3D games.

The interface is heavily influenced by the LÖVE game framework in terms of its
simplicity and easy-of-use. It expands on this with powerful time-travelling debugging,
data inspection, hot reloading, and deterministic replays.

Should include mesh/texture/audio authoring within the editor.

How much is done: 0%

The following are goals of the project:
- Windows and WASM support

The following are explicit non-goals:
- Advanced graphical features
- 

## Building on Linux?

SDL3 is built from source automatically via CMake's FetchContent (it isn't
packaged in apt yet), so you only need GLEW and a Rust toolchain installed:

```
sudo apt-get install libglew-dev
# Recommended for headless rendering (see below): EGL + Mesa software renderer
sudo apt-get install libegl1-mesa-dev libgles2-mesa-dev libgl1-mesa-dri
# Rust (if not already installed): https://rustup.rs

# For a release build
cmake -B build
# ...or for a debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug

cmake --build build -j

# Run it (uses the default game.shm in the working directory)
./build/game

# ...or run a specific script
./build/game --script path/to/my_game.shm
```

The first configure clones and builds SDL3 (and Tracy), so it takes a while;
subsequent builds reuse the cached copies under `build/_deps`.

## Choosing a script

By default the engine loads `game.shm` from the working directory. Use `--script`
to run a different script instead. This works in both interactive and headless
modes:

```
./build/game --script examples/demo.shm
```

## Running headless

For automation / CI graphics validation the engine can render without an
interactive window and write each composited frame (game render + ImGui overlay)
to disk as a PNG:

```
./build/game --headless --frames 5 --screenshot-dir frames
# render a specific script headless
./build/game --headless --frames 5 --screenshot-dir frames --script examples/demo.shm
```

This selects SDL's `offscreen` video driver (EGL surfaceless), so no X server /
`DISPLAY` is required. Each frame is rendered with a fixed delta (1/60s) for
reproducibility, then the process exits. In headless mode the script is loaded
synchronously at startup (and the hot-reload watcher is skipped), so frame output
is deterministic from frame 0 — suitable for golden-image comparison.

Flags:
- `--script PATH` — script to run (default `game.shm`); applies in all modes.
- `--headless` — enable offscreen mode.
- `--frames N` — number of frames to render before exiting (default 1).
- `--screenshot-dir DIR` — output directory for `frame_%04d.png` (default `frames`).
- `--width W` / `--height H` — render resolution (default 1920x1080).

Requirements: the EGL + Mesa packages above must be installed **before** SDL3 is
configured so SDL builds in its EGL/offscreen GL support. If SDL was already built
without them, delete `build/_deps/sdl3-*` and reconfigure. `libgl1-mesa-dri`
provides the `llvmpipe` software renderer used when there's no GPU.
