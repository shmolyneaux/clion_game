# SHIM Engine

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
# Rust (if not already installed): https://rustup.rs

# For a release build
cmake -B build
# ...or for a debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug

cmake --build build -j

# Run it
./build/game
```

The first configure clones and builds SDL3 (and Tracy), so it takes a while;
subsequent builds reuse the cached copies under `build/_deps`.
