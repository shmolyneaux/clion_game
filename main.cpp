// Dear ImGui: standalone example application for SDL2 + OpenGL
// (SDL is a cross-platform general purpose library for handling windows, inputs, OpenGL/Vulkan/Metal graphics context creation, etc.)

// Learn about Dear ImGui:
// - FAQ                  https://dearimgui.com/faq
// - Getting Started      https://dearimgui.com/getting-started
// - Documentation        https://dearimgui.com/docs (same as your local docs/ folder).
// - Introduction, links and more at the top of imgui.cpp

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#include <SDL.h>
#include <GL/glew.h>

#include "Tracy.hpp"
#include "TracyC.h"

#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL_opengles2.h>
#else
#include <SDL_opengl.h>
#endif

// This example can also compile and run with Emscripten! See 'Makefile.emscripten' for details.
#ifdef __EMSCRIPTEN__
#include "imgui-docking/examples/libs/emscripten/emscripten_mainloop_stub.h"

char _whatever[] = "empscripten/has/no/exe";

char* exe_path() {
    return _whatever;
}


#endif

#ifndef __EMSCRIPTEN__
#include <windows.h>
char* exe_path() {
    static char path[MAX_PATH];
    GetModuleFileNameA(NULL, path, MAX_PATH);
    std::cout << "Executable path: " << path << std::endl;
    return path;
}
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Global vector to store error logs
std::vector<std::string> logs;

SDL_Window* window;

extern "C" int rust_init();
extern "C" int rust_frame(float delta);

extern "C" {
    bool igBegin(const char* name, bool* p_open, ImGuiWindowFlags flags)
    {
        ZoneScoped;
        return ImGui::Begin(name, p_open, flags);
    }

    void igEnd()
    {
        ZoneScoped;
        return ImGui::End();
    }

    void igBeginDisabled() {
        ZoneScoped;
        ImGui::BeginDisabled();
    }

    void igEndDisabled() {
        ZoneScoped;
        ImGui::EndDisabled();
    }

    // Returns true if the last item is hovered (with optional flags)
    bool igIsItemHovered(unsigned int flags) {
        ZoneScoped;
        return ImGui::IsItemHovered(static_cast<ImGuiHoveredFlags>(flags));
    }

    // Sets a tooltip for the last item
    void igSetTooltip(const char* text) {
        ZoneScoped;
        ImGui::SetTooltip("%s", text);
    }

    bool igInputText(const char* label, char* buffer, int buffer_size, int flags) {
        ZoneScoped;
        return ImGui::InputText(label, buffer, buffer_size, (ImGuiInputTextFlags)flags);
    }

    void igTextC(const char* fmt, ...) {
        ZoneScoped;
        va_list args;
        va_start(args, fmt);
        ImGui::TextV(fmt,args);
        va_end(args);
    }

    void igTextColoredC(float r, float g, float b, float a, const char* fmt, ...) {
        ZoneScoped;
        va_list args;
        va_start(args, fmt);
        ImGui::TextColoredV(ImVec4(r, g, b, a), fmt, args);
        va_end(args);
    }

    bool igButton(const char* label) {
        ZoneScoped;
        return ImGui::Button(label);
    }

    void igSliderFloat(const char* label, float* v, float v_min, float v_max, const char* format) {
        ZoneScoped;
        ImGui::SliderFloat(label, v, v_min, v_max, format);
    }

    bool igCheckbox(const char* label, bool* v) {
        ZoneScoped;
        return ImGui::Checkbox(label, v);
    }

    bool igWantCaptureKeyboard() {
        ZoneScoped;
        return ImGui::GetIO().WantCaptureKeyboard;
    }

    bool igWantCaptureMouse() {
        ZoneScoped;
        return ImGui::GetIO().WantCaptureMouse;
    }

    void SHM_GetDrawableSize(int *display_w, int *display_h) {
        ZoneScoped;
        SDL_GL_GetDrawableSize(window, display_w, display_h);
    }

    void igSHMNextItemOpenOnce() {
        ZoneScoped;
        ImGui::SetNextItemOpen(true, ImGuiCond_Once);
    }

    bool igTreeNode(const char* label) {
        ZoneScoped;
        return ImGui::TreeNode(label);
    }

    void igTreePop() {
        ZoneScoped;
        return ImGui::TreePop();
    }

    void igSameLine() {
        ZoneScoped;
        return ImGui::SameLine();
    }

    bool igBeginTable(const char* label, int columns) {
        ZoneScoped;
        return ImGui::BeginTable(label, columns, ImGuiTableFlags_SizingStretchProp);
    }

    void igTableSetupColumn(const char* label) {
        ZoneScoped;
        ImGui::TableSetupColumn(label);
    }

    void igTableHeadersRow() {
        ZoneScoped;
        ImGui::TableHeadersRow();
    }

    void igTableNextRow() {
        ZoneScoped;
        ImGui::TableNextRow();
    }

    void igTableSetColumnIndex(int index) {
        ZoneScoped;
        ImGui::TableSetColumnIndex(index);
    }
    void igEndTable() {
        ZoneScoped;
        return ImGui::EndTable();
    }
    void igSetKeyboardFocusHere() {
        ZoneScoped;
        ImGui::SetKeyboardFocusHere();
    }

    TracyCZoneCtx tracy_zone_begin_n(const char* name, int active) {
        TracyCZoneN(ctx, name, active);
        return ctx;
    }

    TracyCZoneCtx tracy_zone_begin_ns(const char* name, int depth, int active) {
        TracyCZoneNS(ctx, name, depth, active);
        return ctx;
    }

    void tracy_zone_end(TracyCZoneCtx ctx) {
        TracyCZoneEnd(ctx);
    }

    void tracy_zone_text(TracyCZoneCtx ctx, const char* txt, unsigned len) {
        TracyCZoneText(ctx, txt, len);
    }

    void tracy_zone_name(TracyCZoneCtx ctx, const char* txt, unsigned len) {
        TracyCZoneName(ctx, txt, len);
    }

    void tracy_zone_color(TracyCZoneCtx ctx, unsigned color) {
        TracyCZoneColor(ctx, color);
    }
}



// Function to log error messages
void log_error(const char *error_message) {
    printf("%s\n", error_message);
    // Append a copy of the error_message to the global vector
    logs.emplace_back(error_message);
}

void log(const char *message) {
    printf("%s\n", message);
    logs.emplace_back(message);
}

void example_window(bool *show_another_window) {
    ImGui::Begin("Another Window", show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
    ImGui::Text("Hello from another window!");
    if (ImGui::Button("Close Me"))
        *show_another_window = false;
    ImGui::End();
}

void simple_window(ImVec4 *clear_color, bool *show_demo_window, bool *show_another_window) {
    static float f = 0.0f;
    static int counter = 0;

    const ImGuiIO& io = ImGui::GetIO();

    ImGui::Begin("Hello, world 6!");

    ImGui::Text("This is some useful text.");
    ImGui::Checkbox("Demo Window", show_demo_window);
    ImGui::Checkbox("Another Window", show_another_window);

    ImGui::SliderFloat("float", &f, 0.0f, 1.0f);
    ImGui::ColorEdit3("clear color", (float *) clear_color);

    if (ImGui::Button("Button")) {
        counter++;
    }
    ImGui::SameLine();
    ImGui::Text("counter = %d", counter);

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
    ImGui::End();
}

void error_window() {
    ImGui::SetNextWindowSize(ImVec2(600, 300), ImGuiCond_FirstUseEver);       // Set default size: 400x300
    ImGui::SetNextWindowPos(ImVec2(20, 350), ImGuiCond_FirstUseEver);        // Set default position: (100, 100)

    ImGui::Begin("Message log", nullptr, ImGuiWindowFlags_NoFocusOnAppearing);

    for (const auto &error : logs) {
        ImGui::Text("%s", error.c_str());
    }

    ImGui::End();
}

void checkShaderCompilation(GLuint shader) {
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        log_error("Error: Shader Compilation Failed");
        log_error(infoLog);
    }
}

void checkProgramLinking(GLuint program) {
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        log_error("Error: Shader Compilation Failed");
        log_error(infoLog);
    }
}

glm::mat4 view;
glm::mat4 projection;
unsigned int debugViewLoc;
unsigned int debugProjectionLoc;
GLuint debugShaderProgram;

GLuint BoxVAO, BoxVBO, BoxEBO;

std::vector<glm::vec3> debug_verts;
std::vector<unsigned int> debug_vert_indices;

#define END_PRIMITIVE 0xFFFFFFFF

void init_debug_drawing() {
    glGenVertexArrays(1, &BoxVAO);
    glGenBuffers(1, &BoxVBO);
    glGenBuffers(1, &BoxEBO);
}

void draw_debug_shapes() {
    glBindVertexArray(BoxVAO);
    glBindBuffer(GL_ARRAY_BUFFER, BoxVBO);
    glBufferData(
        GL_ARRAY_BUFFER,
        debug_verts.size() * sizeof(glm::vec3),
        debug_verts.data(),
        GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, BoxEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, debug_vert_indices.size() * sizeof(unsigned int), debug_vert_indices.data(), GL_DYNAMIC_DRAW);

    // Vertex Position Attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glUseProgram(debugShaderProgram);

    glUniformMatrix4fv(debugViewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(debugProjectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

    glDrawElements(GL_LINE_STRIP, debug_vert_indices.size(), GL_UNSIGNED_INT, 0);

    debug_verts.clear();
    debug_vert_indices.clear();
}




template <typename T>
struct Slice {
    T* data;
    std::size_t length;

    Slice(T* ptr, std::size_t len) : data(ptr), length(len) {}

    template <std::size_t N>
    Slice(T (&arr)[N]) : data(arr), length(N) {} // Constructor from C-array

    T& operator[](std::size_t index) {
        return data[index];
    }

    const T& operator[](std::size_t index) const {
        return data[index];
    }

    T* begin() { return data; }
    T* end() { return data + length; }

    const T* begin() const { return data; }
    const T* end() const { return data + length; }

    std::size_t size() const { return length; }
    bool empty() const { return length == 0; }

    // Nested Iterator Class for `enumerate()`
    struct EnumerateIterator {
        std::size_t index;
        T* ptr;

        EnumerateIterator(std::size_t i, T* p) : index(i), ptr(p) {}

        bool operator!=(const EnumerateIterator& other) const { return index != other.index; }
        void operator++() { ++index; ++ptr; }
        std::pair<std::size_t, T&> operator*() { return {index, *ptr}; }
    };

    // Enumerate Wrapper
    struct EnumerateRange {
        Slice& slice;
        EnumerateIterator begin() { return {0, slice.data}; }
        EnumerateIterator end() { return {slice.length, slice.data + slice.length}; }
    };

    // Method to return EnumerateRange
    EnumerateRange enumerate() { return EnumerateRange{*this}; }
};



void debug_lines(Slice<glm::vec3> points, bool connect_last) {
    int start_offset = debug_verts.size();
    int offset = debug_verts.size();
    for (glm::vec3 p : points) {
        debug_vert_indices.push_back(offset);
        debug_verts.push_back(p);
        offset += 1;
    }
    if (connect_last) {
        debug_vert_indices.push_back(start_offset);
    }
    debug_vert_indices.push_back(END_PRIMITIVE);
}

void debug_lines(Slice<glm::vec3> points) {
    debug_lines(points, false);
    int offset = debug_verts.size();
    for (glm::vec3 p : points) {
        debug_vert_indices.push_back(offset);
        debug_verts.push_back(p);
        offset += 1;
    }
    debug_vert_indices.push_back(END_PRIMITIVE);
}

void debug_box(glm::vec3 position, glm::vec3 size, glm::vec3 color) {
    int offset = debug_verts.size();

    glm::vec3 debug_verts_here[] = {
        position + glm::vec3(+size.x, +size.y, +size.z)/2.0f,
        position + glm::vec3(+size.x, +size.y, -size.z)/2.0f,
        position + glm::vec3(+size.x, -size.y, +size.z)/2.0f,
        position + glm::vec3(+size.x, -size.y, -size.z)/2.0f,
        position + glm::vec3(-size.x, +size.y, +size.z)/2.0f,
        position + glm::vec3(-size.x, +size.y, -size.z)/2.0f,
        position + glm::vec3(-size.x, -size.y, +size.z)/2.0f,
        position + glm::vec3(-size.x, -size.y, -size.z)/2.0f,
    };

    for (int i = 0; i < sizeof(debug_verts_here) / sizeof(glm::vec3); i++) {
        debug_verts.push_back(debug_verts_here[i]);
    }

    unsigned int debug_vert_indices_here[] = {
        0, 1, 3, 2, 0,
        4, 5, 7, 6, 4,
        END_PRIMITIVE,
        1, 5,
        END_PRIMITIVE,
        3, 7,
        END_PRIMITIVE,
        2, 6,
        END_PRIMITIVE,
    };

    for (int i = 0; i < sizeof(debug_vert_indices_here) / sizeof(unsigned int); i++) {
        unsigned int index = debug_vert_indices_here[i];
        if (index == END_PRIMITIVE) {
            debug_vert_indices.push_back(index);
        } else {
            debug_vert_indices.push_back(index+offset);
        }
    }
}

void debug_box(glm::vec3 position, glm::vec3 size) {
    debug_box(position, size, glm::vec3(0.0f, 1.0f, 0.0f));
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BUFFER_SIZE (100 * 1024 * 1024) // 100 MB

// Main code
int main(int, char**)
{
    bool startup_error = false;
    log_error("Starting game....");

    // Setup SDL
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)
    {
        log_error("SDL_Init Error:");
        log_error(SDL_GetError());
        SDL_Quit();
        return -1;
    }

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    log_error("Using OpenGL ES2");
    // GL ES 2.0 + GLSL 100 (WebGL 1.0)
    const char* glsl_version = "#version 100";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#elif defined(IMGUI_IMPL_OPENGL_ES3)
    log_error("Using OpenGL ES3");
    // GL ES 3.0 + GLSL 300 es (WebGL 2.0)
    const char* glsl_version = "#version 300 es";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#elif defined(__APPLE__)
    log_error("Apple or something?");
    // GL 3.2 Core + GLSL 150
    const char* glsl_version = "#version 150";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG); // Always required on Mac
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
#else
    log_error("Using OpenGL ES3 else case");
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 300 es";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#endif

    // From 2.0.18: Enable native IME.
#ifdef SDL_HINT_IME_SHOW_UI
    SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");
#endif

    // Create window with graphics context
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
    SDL_WindowFlags window_flags = (SDL_WindowFlags)(
            SDL_WINDOW_OPENGL
            | SDL_WINDOW_RESIZABLE
            | SDL_WINDOW_ALLOW_HIGHDPI
            );
    window = SDL_CreateWindow("CLion Game", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1920, 1080, window_flags);
    if (window == nullptr)
    {
        log_error("SDL_CreateWindow Error:");
        log_error(SDL_GetError());
        SDL_Quit();
        return -1;
    }

    bool mouse_captured = false;
    #ifndef __EMSCRIPTEN__
    SDL_SetRelativeMouseMode(SDL_TRUE);
    mouse_captured = true;
    #endif

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (gl_context == nullptr)
    {
        log_error("SDL_GL_CreateContext Error:");
        log_error(SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }


    glEnable(GL_DEPTH_TEST);

    #ifndef __EMSCRIPTEN__
    glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
    #endif

    // CHECK: Is GLEW not needed for the web?
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        log_error("Failed to initialize GLEW");
        SDL_GL_DeleteContext(gl_context);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    SDL_GL_MakeCurrent(window, gl_context);
    SDL_GL_SetSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // Enable Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;       // Enable Multi-Viewport / Platform Windows

    ImGui::StyleColorsDark();

    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;
    // Docs suggest making windows opaque so the transition to a platform window isn't noticeable. I don't care about
    // that and much prefer the usefulness of seeing behind normal windows.
    style.Colors[ImGuiCol_WindowBg].w = 0.85f;

    // Setup Platform/Renderer backends
    ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Our state
    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.102f, 0.102f, 0.114f, 1.00f);
    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);

    Uint64 frequency = SDL_GetPerformanceFrequency();
    Uint64 ticks = 0;

    init_debug_drawing();

    double elapsed = 0.0f;

    if (rust_init()) {
        printf("Rust failed to initialize!\n");
        return 1;
    }

    // Main loop
    bool done = false;
#ifdef __EMSCRIPTEN__
    // For an Emscripten build we are disabling file-system access, so let's not attempt to do a fopen() of the imgui.ini file.
    // You may manually call LoadIniSettingsFromMemory() to load settings from your own storage.
    io.IniFilename = nullptr;
    EMSCRIPTEN_MAINLOOP_BEGIN
#else
    while (!done)
#endif
    {
        Uint64 current_ticks = SDL_GetPerformanceCounter();
        // Increase the current time monotonically
        if (current_ticks <= ticks) {
            current_ticks = ticks + 1;
        }
        float delta = ticks > 0 ? (float)((double)(current_ticks - ticks) / frequency) : (float)(1.0f / 60.0f);

        elapsed += delta;

        ticks = current_ticks;

        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) {
                done = true;
            }
            if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE && event.window.windowID == SDL_GetWindowID(window)) {
                done = true;
            }
        }

        // Skip rendering when minimized
        if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED)
        {
            SDL_Delay(10);
            continue;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        int display_w, display_h;
        SDL_GL_GetDrawableSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        // glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        error_window();

        if (!startup_error) {
            if (show_demo_window) {
                ImGui::ShowDemoWindow(&show_demo_window);
            }

            if (show_another_window) {
                example_window(&show_another_window);
            }

            {
                ZoneScoped;
                TracyCZoneN(ctx, "rust_frame", 1);
                rust_frame(delta);
                TracyCZoneEnd(ctx);
            }
        } else {
            ImGui::Begin("STARTUP ERROR");
            ImGui::Text("Startup Error");
            ImGui::End();
        }

        // Rendering
        ImGui::Render();


        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Update and Render additional Platform Windows
        // (Platform functions may change the current OpenGL context, so we save/restore it to make it easier to paste this code elsewhere.
        //  For this specific demo app we could also call SDL_GL_MakeCurrent(window, gl_context) directly)
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            SDL_Window* backup_current_window = SDL_GL_GetCurrentWindow();
            SDL_GLContext backup_current_context = SDL_GL_GetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            SDL_GL_MakeCurrent(backup_current_window, backup_current_context);
        }

        SDL_GL_SwapWindow(window);
        FrameMark;
    }
#ifdef __EMSCRIPTEN__
    EMSCRIPTEN_MAINLOOP_END;
#endif

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    SDL_SCANCODE_LSHIFT;

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}

#undef END_PRIMITIVE