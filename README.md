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

```
sudo apt-get install libsdl2-dev libglew-dev
mkdir build && cd build
# For release build
cmake ..
# For debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build .
```
