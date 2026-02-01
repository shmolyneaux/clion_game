cmake_minimum_required(VERSION 3.16)
project(clion_game)

# --- Dependencies ---
include(ExternalProject) # For Rust
include(FetchContent)    # For Tracy

FetchContent_Declare(
    tracy
    GIT_REPOSITORY https://github.com/wolfpld/tracy.git
    GIT_TAG v0.11.1
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(tracy)

# --- Rust Configuration ---
set(CARGO_PROFILE_DIR  "$<$<CONFIG:Debug>:debug>$<$<NOT:$<CONFIG:Debug>>:release>")
set(CARGO_RELEASE_FLAG "$<$<NOT:$<CONFIG:Debug>>:--release>")
set(OBJECT_FILE ${CMAKE_BINARY_DIR}/cmake_this_is_a_test3.o)

# Detect OS for Rust Output Path
if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set(RUST_TARGET "wasm32-unknown-emscripten")
    set(INTERMEDIATE_FILE ${CMAKE_SOURCE_DIR}/target/${RUST_TARGET}/${CARGO_PROFILE_DIR}/libclion_game.a)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(RUST_TARGET "x86_64-unknown-linux-gnu")
    # Linux uses lib<name>.a
    set(INTERMEDIATE_FILE ${CMAKE_SOURCE_DIR}/target/${RUST_TARGET}/${CARGO_PROFILE_DIR}/libclion_game.a)
else()
    set(RUST_TARGET "x86_64-pc-windows-msvc")
    # Windows uses <name>.lib
    set(INTERMEDIATE_FILE ${CMAKE_SOURCE_DIR}/target/${RUST_TARGET}/${CARGO_PROFILE_DIR}/clion_game.lib)
endif()

# Build Rust Object
add_custom_target(
    generate_object ALL
    COMMAND cargo build ${CARGO_RELEASE_FLAG} --target=${RUST_TARGET}
    COMMAND ${CMAKE_COMMAND} -E copy ${INTERMEDIATE_FILE} ${OBJECT_FILE}
    BYPRODUCTS ${OBJECT_FILE}
    OUTPUT ${OBJECT_FILE}
    COMMAND_EXPAND_LISTS
    VERBATIM
)

add_custom_target(regen_obj ALL DEPENDS ${OBJECT_FILE})

# --- C++ Configuration ---
set(CMAKE_CXX_STANDARD 20)

# Define Sources (ImGui + Main)
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/imgui-docking/")

set(SOURCES
    main.cpp
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl2.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# Global Includes for ImGui headers
include_directories(includes)
include_directories(${IMGUI_DIR})
include_directories(${IMGUI_DIR}/backends)

# Define Executable
add_executable(game ${SOURCES} ${OBJECT_FILE})
add_dependencies(game generate_object regen_obj)

# --- Linking & Libraries ---

# 1. Common Native Setup (Linux & Windows)
if (NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    # Tracy
    target_include_directories(game PRIVATE "${tracy_SOURCE_DIR}/public/tracy")
    target_compile_definitions(game PRIVATE TRACY_ENABLE)
    find_package(Threads REQUIRED) 
endif()

# 2. OS Specifics
if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    # ... (Keep your existing Emscripten block here) ...
    set(WEB_DIR ${CMAKE_BINARY_DIR}/web)
    file(MAKE_DIRECTORY ${WEB_DIR})
    set(CMAKE_EXECUTABLE_SUFFIX ".html")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -sUSE_SDL=2")
    target_link_options(game PRIVATE 
        "-sWASM=1" "-sUSE_SDL=2" "-sMIN_WEBGL_VERSION=2" "-sMAX_WEBGL_VERSION=2" 
        "-sALLOW_MEMORY_GROWTH=1" "-sNO_EXIT_RUNTIME=0" "-sASSERTIONS=1" "-sNO_FILESYSTEM=1"
        "SHELL:--shell-file ${IMGUI_DIR}/examples/libs/emscripten/shell_minimal.html"
    )
    target_compile_definitions(game PRIVATE IMGUI_IMPL_OPENGL_ES3)

elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # --- LINUX SETUP ---
    find_package(SDL2 REQUIRED)
    find_package(GLEW REQUIRED)
    
    # Add includes (This fixes "SDL.h not found" for ImGui too)
    target_include_directories(game PRIVATE ${SDL2_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})
    
    # Link Libraries
    target_link_libraries(game 
        PRIVATE 
        ${SDL2_LIBRARIES} 
        ${GLEW_LIBRARIES} 
        GL 
        TracyClient 
        Threads::Threads
        ${CMAKE_DL_LIBS}
    )

else()
    # --- WINDOWS SETUP ---
    set(SDL2_PATH "C:/lib/SDL2-2.30.11-VC")
    set(GLEW_PATH "C:/lib/glew-2.1.0")
    
    target_include_directories(game PRIVATE ${SDL2_PATH}/include ${GLEW_PATH}/include)
    link_directories(${SDL2_PATH}/lib/x64 ${GLEW_PATH}/lib/Release/x64)

    target_link_libraries(game SDL2 SDL2main opengl32 glew32 ws2_32 userenv ntdll TracyClient)

    # DLL Copying
    set(SDL2_DLL "${SDL2_PATH}/lib/x64/SDL2.dll")
    set(GLEW_DLL "${GLEW_PATH}/bin/Release/x64/glew32.dll")
    add_custom_command(TARGET game POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SDL2_DLL}" "$<TARGET_FILE_DIR:game>/SDL2.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GLEW_DLL}" "$<TARGET_FILE_DIR:game>/glew32.dll"
    )
endif()
