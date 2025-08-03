cmake_minimum_required(VERSION 3.16)
project(clion_game)

# For Rust support
include(ExternalProject)

set(OBJECT_FILE ${CMAKE_BINARY_DIR}/cmake_this_is_a_test3.o)      # Output object file

# Custom command to generate the .o file
if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set(INTERMEDIATE_FILE ${CMAKE_SOURCE_DIR}/target/wasm32-unknown-emscripten/debug/libclion_game.a)
    add_custom_target(
        generate_object ALL
        COMMAND cargo build --target=wasm32-unknown-emscripten
        COMMAND ${CMAKE_COMMAND} -E copy ${INTERMEDIATE_FILE} ${OBJECT_FILE}
        BYPRODUCTS ${OBJECT_FILE}
        OUTPUT ${OBJECT_FILE}
        VERBATIM
    )
    # target\wasm32-unknown-emscripten\release\libclion_game.a
else()
    set(INTERMEDIATE_FILE ${CMAKE_SOURCE_DIR}/target/x86_64-pc-windows-msvc/debug/clion_game.lib)
    add_custom_target(
        generate_object ALL
        COMMAND cargo build --target=x86_64-pc-windows-msvc
        COMMAND ${CMAKE_COMMAND} -E copy ${INTERMEDIATE_FILE} ${OBJECT_FILE}
        BYPRODUCTS ${OBJECT_FILE}
        OUTPUT ${OBJECT_FILE}
        VERBATIM
    )
endif()

# Custom target to ensure the object file is built
#add_custom_target(generate_object DEPENDS ${OBJECT_FILE})

add_custom_target(
    regen_obj ALL
    DEPENDS ${OBJECT_FILE}
)

set(CMAKE_CXX_STANDARD 20)

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

include_directories(includes)
include_directories(${IMGUI_DIR})
include_directories(${IMGUI_DIR}/backends)

if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
   set(CMAKE_VERBOSE_MAKEFILE ON)

    set(WEB_DIR ${CMAKE_BINARY_DIR}/web)
    file(MAKE_DIRECTORY ${WEB_DIR})
    set(CMAKE_EXECUTABLE_SUFFIX ".html")

    # Compiler and linker flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -sUSE_SDL=2")

    # AFAIK a number of these flags should be the same as the defaults, but removing them increases the resulting file
    # size, so I'm just going to leave them for now
    set(EM_LIBS
        "-sWASM=1"
        "-sMIN_WEBGL_VERSION=2"
        "-sMAX_WEBGL_VERSION=2"
        "-sALLOW_MEMORY_GROWTH=1"
        "-sNO_EXIT_RUNTIME=0"
        "-sASSERTIONS=1"
        "-sNO_FILESYSTEM=1"
        "SHELL:--shell-file ${IMGUI_DIR}/examples/libs/emscripten/shell_minimal.html"
    )
else()
    # TODO: update this to use find_package
    set(SDL2_PATH "C:/lib/SDL2-2.30.11-VC")
    include_directories(${SDL2_PATH}/include)
    link_directories(${SDL2_PATH}/lib/x64)

    set(GLEW_PATH "C:/lib/glew-2.1.0")
    include_directories(${GLEW_PATH}/include)
    link_directories(${GLEW_PATH}/lib/Release/x64)
endif()

add_executable(game ${SOURCES}
    ${OBJECT_FILE}
)

add_dependencies(game generate_object)
add_dependencies(game regen_obj)

if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    target_link_options(game PRIVATE ${EM_LIBS})
    set_target_properties(game PROPERTIES
        OUTPUT_NAME "index"
        RUNTIME_OUTPUT_DIRECTORY ${WEB_DIR}
    )
    target_compile_definitions(game PRIVATE IMGUI_IMPL_OPENGL_ES3)
else()
    target_link_libraries(game SDL2 SDL2main opengl32 glew32 ws2_32 userenv ntdll)
    # from rustc: native-static-libs: kernel32.lib advapi32.lib ntdll.lib userenv.lib ws2_32.lib dbghelp.lib /defaultlib:msvcrt

    set(SDL2_DLL "${SDL2_PATH}/lib/x64/SDL2.dll")
    set(GLEW_DLL "${GLEW_PATH}/bin/Release/x64/glew32.dll")

    add_custom_command(TARGET game POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SDL2_DLL}" "$<TARGET_FILE_DIR:game>/SDL2.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GLEW_DLL}" "$<TARGET_FILE_DIR:game>/glew32.dll"
    )
endif()
