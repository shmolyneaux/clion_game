#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#include <SDL3/SDL.h>
#include <GL/glew.h>

#if defined(_WIN32)
#include "Tracy.hpp"
#include "TracyC.h"
#elif defined(__linux__)
#include "Tracy.hpp"
#include "TracyC.h"
#elif defined(__EMSCRIPTEN__)
#define ZoneScoped
#endif
#include <vector>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL3/SDL_opengles2.h>
#elif !defined(IMGUI_IMPL_OPENGL_ES3)
#include <SDL3/SDL_opengl.h>
#endif

#if defined(__EMSCRIPTEN__)
#include "emscripten_mainloop_stub.h"
#endif

// Global vector to store error logs
std::vector<std::string> logs;

SDL_Window* window;

extern "C" int rust_init();
extern "C" int rust_frame(float delta);
extern "C" void rust_audio_callback(void* userdata, Uint8* stream, int len);
extern "C" int rust_save_screenshot(const char* path, int w, int h);
extern "C" void rust_set_headless(bool enabled);
extern "C" void rust_set_script_path(const char* path);

static constexpr int kAudioSampleRate = 44100;

// Command-line configuration, populated from argv.
//
// `script_path` selects which script the engine runs and applies in all modes
// (interactive and headless); when empty the engine uses its built-in default
// ("game.shm").
//
// The headless options are only meaningful when `enabled` is set: the engine then
// renders offscreen (no visible window) for `frames` frames and writes each
// composited frame to `screenshot_dir` as a PNG, then exits. Intended for
// automation / CI graphics validation.
struct CliConfig {
    bool enabled = false;            // --headless
    int frames = 1;                  // --frames
    std::string screenshot_dir = "frames"; // --screenshot-dir
    int width = 1920;                // --width
    int height = 1080;               // --height
    std::string script_path;         // --script (empty => engine default)
};

static CliConfig parse_cli_config(int argc, char** argv) {
    CliConfig cfg;
    auto next_int = [&](int& i, int fallback) -> int {
        if (i + 1 < argc) {
            return SDL_atoi(argv[++i]);
        }
        return fallback;
    };
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--headless") {
            cfg.enabled = true;
        } else if (arg == "--frames") {
            cfg.frames = next_int(i, cfg.frames);
        } else if (arg == "--screenshot-dir") {
            if (i + 1 < argc) cfg.screenshot_dir = argv[++i];
        } else if (arg == "--width") {
            cfg.width = next_int(i, cfg.width);
        } else if (arg == "--height") {
            cfg.height = next_int(i, cfg.height);
        } else if (arg == "--script") {
            if (i + 1 < argc) cfg.script_path = argv[++i];
        } else if (arg == "--fixed-delta") {
            // Deterministic delta is implied in headless mode; accepted for clarity.
        }
    }
    if (cfg.frames < 1) cfg.frames = 1;
    return cfg;
}


// Function to log error messages
void log_error(const char *error_message) {
    printf("%s\n", error_message);
    // Append a copy of the error_message to the global vector
    logs.emplace_back(error_message);
}

void error_window() {
    ImGui::SetNextWindowSize(ImVec2(600, 300), ImGuiCond_FirstUseEver);       // Set default size: 400x300
    ImGui::SetNextWindowPos(ImVec2(20, 350), ImGuiCond_FirstUseEver);        // Set default position: (100, 100)

    if (!logs.empty()) {
        ImGui::Begin("C++ message log", nullptr, ImGuiWindowFlags_NoFocusOnAppearing);
        for (const auto &error : logs) {
            ImGui::Text("%s", error.c_str());
        }
        ImGui::End();
    }
}

// Main code
int main(int argc, char** argv)
{
    bool startup_error = false;

    CliConfig headless = parse_cli_config(argc, argv);

    if (headless.enabled) {
        // Use the offscreen (EGL surfaceless) video driver so a GL context can be
        // created with no X server / display, and the dummy audio driver so we
        // don't need an audio device.
        SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "offscreen");
        SDL_SetHint(SDL_HINT_AUDIO_DRIVER, "dummy");
    }

    // Setup SDL
    Uint32 init_flags = SDL_INIT_VIDEO | SDL_INIT_GAMEPAD;
    if (!headless.enabled) {
        init_flags |= SDL_INIT_AUDIO;
    }
    if (!SDL_Init(init_flags))
    {
        log_error("SDL_Init Error:");
        log_error(SDL_GetError());
        SDL_Quit();
        return -1;
    }

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100 (WebGL 1.0)
    const char* glsl_version = "#version 100";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#elif defined(IMGUI_IMPL_OPENGL_ES3)
    // GL ES 3.0 + GLSL 300 es (WebGL 2.0)
    const char* glsl_version = "#version 300 es";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#elif defined(__APPLE__)
    // GL 3.2 Core + GLSL 150
    const char* glsl_version = "#version 150";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG); // Always required on Mac
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
#else
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
    SDL_WindowFlags window_flags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    int window_w = 1920;
    int window_h = 1080;
    if (headless.enabled) {
        window_flags |= SDL_WINDOW_HIDDEN;
        window_w = headless.width;
        window_h = headless.height;
    }
    window = SDL_CreateWindow("CLion Game", window_w, window_h, window_flags);
    if (window == nullptr)
    {
        log_error("SDL_CreateWindow Error:");
        log_error(SDL_GetError());
        SDL_Quit();
        return -1;
    }

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
        // Under the offscreen/EGL driver GLEW may fail to initialize, but the
        // core GL calls this file makes resolve via libGL directly, so don't
        // treat it as fatal in headless mode.
        if (headless.enabled) {
            log_error("glewInit failed under headless mode; continuing");
        } else {
            log_error("Failed to initialize GLEW");
            SDL_GL_DestroyContext(gl_context);
            SDL_DestroyWindow(window);
            SDL_Quit();
            return -1;
        }
    }
    // glewInit() with glewExperimental calls glGetString(GL_EXTENSIONS) which
    // is invalid in a core profile context, leaving a spurious INVALID_ENUM in
    // the error queue. Clear it before handing control to Rust.
    glGetError();

    SDL_GL_MakeCurrent(window, gl_context);
#ifndef __EMSCRIPTEN__
    SDL_GL_SetSwapInterval(headless.enabled ? 0 : 1); // Enable vsync (off in headless)
#endif

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // Enable Docking
    if (!headless.enabled) {
        // Multi-viewport spawns real OS windows, which is impossible offscreen.
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;   // Enable Multi-Viewport / Platform Windows
    }

    ImGui::StyleColorsDark();

    {
        ImGuiStyle& style = ImGui::GetStyle();
        style.WindowRounding = 0.0f;
        // Docs suggest making windows opaque so the transition to a platform window isn't noticeable. I don't care about
        // that and much prefer the usefulness of seeing behind normal windows.
        auto transparency = 0.35f;
        style.Colors[ImGuiCol_WindowBg].w *= transparency;
        // style.Colors[ImGuiCol_Text].w *= transparency;
        // style.Colors[ImGuiCol_TextDisabled].w *= transparency;
        style.Colors[ImGuiCol_WindowBg].w *= transparency;
        style.Colors[ImGuiCol_ChildBg].w *= transparency;
        style.Colors[ImGuiCol_PopupBg].w *= transparency;
        style.Colors[ImGuiCol_Border].w *= transparency;
        style.Colors[ImGuiCol_BorderShadow].w *= transparency;
        style.Colors[ImGuiCol_FrameBg].w *= transparency;
        style.Colors[ImGuiCol_FrameBgHovered].w *= transparency;
        style.Colors[ImGuiCol_FrameBgActive].w *= transparency;
        style.Colors[ImGuiCol_TitleBg].w *= transparency;
        style.Colors[ImGuiCol_TitleBgActive].w *= transparency;
        style.Colors[ImGuiCol_TitleBgCollapsed].w *= transparency;
        style.Colors[ImGuiCol_MenuBarBg].w *= transparency;
        style.Colors[ImGuiCol_ScrollbarBg].w *= transparency;
        style.Colors[ImGuiCol_ScrollbarGrab].w *= transparency;
        style.Colors[ImGuiCol_ScrollbarGrabHovered].w *= transparency;
        style.Colors[ImGuiCol_ScrollbarGrabActive].w *= transparency;
        style.Colors[ImGuiCol_CheckMark].w *= transparency;
        style.Colors[ImGuiCol_SliderGrab].w *= transparency;
        style.Colors[ImGuiCol_SliderGrabActive].w *= transparency;
        style.Colors[ImGuiCol_Button].w *= transparency;
        style.Colors[ImGuiCol_ButtonHovered].w *= transparency;
        style.Colors[ImGuiCol_ButtonActive].w *= transparency;
        style.Colors[ImGuiCol_Header].w *= transparency;
        style.Colors[ImGuiCol_HeaderHovered].w *= transparency;
        style.Colors[ImGuiCol_HeaderActive].w *= transparency;
        style.Colors[ImGuiCol_Separator].w *= transparency;
        style.Colors[ImGuiCol_SeparatorHovered].w *= transparency;
        style.Colors[ImGuiCol_SeparatorActive].w *= transparency;
        style.Colors[ImGuiCol_ResizeGrip].w *= transparency;
        style.Colors[ImGuiCol_ResizeGripHovered].w *= transparency;
        style.Colors[ImGuiCol_ResizeGripActive].w *= transparency;
        style.Colors[ImGuiCol_TabHovered].w *= transparency;
        style.Colors[ImGuiCol_Tab].w *= transparency;
        style.Colors[ImGuiCol_TabSelected].w *= transparency;
        style.Colors[ImGuiCol_TabSelectedOverline].w *= transparency;
        style.Colors[ImGuiCol_TabDimmed].w *= transparency;
        style.Colors[ImGuiCol_TabDimmedSelected].w *= transparency;
        style.Colors[ImGuiCol_TabDimmedSelectedOverline].w *= transparency;
        style.Colors[ImGuiCol_DockingPreview].w *= transparency;
        style.Colors[ImGuiCol_DockingEmptyBg].w *= transparency;
        style.Colors[ImGuiCol_PlotLines].w *= transparency;
        style.Colors[ImGuiCol_PlotLinesHovered].w *= transparency;
        style.Colors[ImGuiCol_PlotHistogram].w *= transparency;
        style.Colors[ImGuiCol_PlotHistogramHovered].w *= transparency;
        style.Colors[ImGuiCol_TableHeaderBg].w *= transparency;
        style.Colors[ImGuiCol_TableBorderStrong].w *= transparency;
        style.Colors[ImGuiCol_TableBorderLight].w *= transparency;
        style.Colors[ImGuiCol_TableRowBg].w *= transparency;
        style.Colors[ImGuiCol_TableRowBgAlt].w *= transparency;
        style.Colors[ImGuiCol_TextLink].w *= transparency;
        style.Colors[ImGuiCol_TextSelectedBg].w *= transparency;
        style.Colors[ImGuiCol_DragDropTarget].w *= transparency;
        style.Colors[ImGuiCol_NavCursor].w *= transparency;
        style.Colors[ImGuiCol_NavWindowingHighlight].w *= transparency;
        style.Colors[ImGuiCol_NavWindowingDimBg].w *= transparency;
        style.Colors[ImGuiCol_ModalWindowDimBg].w *= transparency;
    }

    // Setup Platform/Renderer backends
    ImGui_ImplSDL3_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init(glsl_version);
    // ImGui's internal GL loader probes extensions and can leave spurious errors.
    while (glGetError() != GL_NO_ERROR) {}

    // Our state
    ImVec4 clear_color = ImVec4(0.102f, 0.102f, 0.114f, 1.00f);
    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);

    Uint64 frequency = SDL_GetPerformanceFrequency();
    Uint64 ticks = 0;

    // SDL3 audio uses a push-model stream callback. We reuse rust_audio_callback
    // to fill a stack buffer, then push it into the stream each call.
    auto sdl3_audio_callback = [](void*, SDL_AudioStream* stream, int additional_amount, int) {
        constexpr int kChunk = 4096;
        Uint8 buf[kChunk];
        while (additional_amount > 0) {
            int bytes = additional_amount < kChunk ? additional_amount : kChunk;
            rust_audio_callback(nullptr, buf, bytes);
            SDL_PutAudioStreamData(stream, buf, bytes);
            additional_amount -= bytes;
        }
    };
    SDL_AudioStream* audio_stream = nullptr;
    if (!headless.enabled) {
        SDL_AudioSpec want{};
        want.freq = kAudioSampleRate;
        want.format = SDL_AUDIO_S16;
        want.channels = 2;
        audio_stream = SDL_OpenAudioDeviceStream(
            SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &want, sdl3_audio_callback, nullptr);
        if (!audio_stream) {
            log_error("SDL_OpenAudioDeviceStream Error:");
            log_error(SDL_GetError());
        } else {
            SDL_ResumeAudioStreamDevice(audio_stream);
        }
    }

    if (headless.enabled && !SDL_CreateDirectory(headless.screenshot_dir.c_str())) {
        log_error("Failed to create screenshot directory:");
        log_error(SDL_GetError());
    }

    // Must be set before rust_init(): ScriptBridge::new() reads these. The headless
    // flag controls synchronous script loading / skipping the hot-reload watcher;
    // the script path (when provided) overrides the engine's default script.
    rust_set_headless(headless.enabled);
    if (!headless.script_path.empty()) {
        rust_set_script_path(headless.script_path.c_str());
    }

    if (rust_init()) {
        printf("Rust failed to initialize!\n");
        return 1;
    }

    // Main loop
    bool done = false;
    int headless_frame = 0;
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

        ticks = current_ticks;

        // Headless renders use a fixed delta so frames are reproducible run-to-run.
        if (headless.enabled) {
            delta = 1.0f / 60.0f;
        }

        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (event.type == SDL_EVENT_QUIT) {
                done = true;
            }
            if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && event.window.windowID == SDL_GetWindowID(window)) {
                done = true;
            }
        }

        // Skip rendering when minimized (never happens for a hidden headless window)
        if (!headless.enabled && (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED))
        {
            SDL_Delay(10);
            continue;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        // Fullscreen DockSpace
        {
            ImGuiWindowFlags dockspace_flags =
                ImGuiWindowFlags_NoDocking |
                ImGuiWindowFlags_NoTitleBar |
                ImGuiWindowFlags_NoCollapse |
                ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoBringToFrontOnFocus |
                ImGuiWindowFlags_NoNavFocus |
                ImGuiWindowFlags_NoBackground;

            ImGuiViewport* viewport = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(viewport->WorkPos);
            ImGui::SetNextWindowSize(viewport->WorkSize);
            ImGui::SetNextWindowViewport(viewport->ID);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

            ImGui::Begin("DockSpaceWindow", nullptr, dockspace_flags);
            ImGui::PopStyleVar(3);

            ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
            ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);

            ImGui::End();
        }

        int display_w, display_h;
        SDL_GetWindowSizeInPixels(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        // glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        error_window();

        if (!startup_error) {
            {
                ZoneScoped;
#ifndef __EMSCRIPTEN__
                TracyCZoneN(ctx, "rust_frame", 1);
#endif
                rust_frame(delta);
#ifndef __EMSCRIPTEN__
                TracyCZoneEnd(ctx);
#endif
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

        // Capture the composited back buffer before swapping. Exit once we've
        // written the requested number of frames.
        if (headless.enabled) {
            char path[1024];
            SDL_snprintf(path, sizeof(path), "%s/frame_%04d.png",
                         headless.screenshot_dir.c_str(), headless_frame);
            if (rust_save_screenshot(path, display_w, display_h)) {
                log_error("rust_save_screenshot failed");
                log_error(path);
            }
            headless_frame++;
            if (headless_frame >= headless.frames) {
                done = true;
            }
        }

        SDL_GL_SwapWindow(window);
#ifndef __EMSCRIPTEN__
        FrameMark;
#endif
    }
#ifdef __EMSCRIPTEN__
    EMSCRIPTEN_MAINLOOP_END;
#endif

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    if (audio_stream) {
        SDL_DestroyAudioStream(audio_stream);
    }

    SDL_GL_DestroyContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
