#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#include <SDL.h>
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
#include <SDL_opengles2.h>
#else
#include <SDL_opengl.h>
#endif

#if defined(__linux__) || defined(__EMSCRIPTEN__)
#include "imgui-docking/examples/libs/emscripten/emscripten_mainloop_stub.h"
#endif

// Global vector to store error logs
std::vector<std::string> logs;

SDL_Window* window;

extern "C" int rust_init();
extern "C" int rust_frame(float delta);


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
int main(int, char**)
{
    bool startup_error = false;

    // Setup SDL
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER | SDL_INIT_AUDIO) != 0)
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

    #ifndef __EMSCRIPTEN__
    SDL_SetRelativeMouseMode(SDL_TRUE);
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
    // glewInit() with glewExperimental calls glGetString(GL_EXTENSIONS) which
    // is invalid in a core profile context, leaving a spurious INVALID_ENUM in
    // the error queue. Clear it before handing control to Rust.
    glGetError();

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

    // Setup Platform/Renderer backends
    ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init(glsl_version);
    // ImGui's internal GL loader probes extensions and can leave spurious errors.
    while (glGetError() != GL_NO_ERROR) {}

    // Our state
    bool show_demo_window = false;
    ImVec4 clear_color = ImVec4(0.102f, 0.102f, 0.114f, 1.00f);
    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);

    Uint64 frequency = SDL_GetPerformanceFrequency();
    Uint64 ticks = 0;

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
        SDL_GL_GetDrawableSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        // glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        error_window();

        if (!startup_error) {
            if (show_demo_window) {
                ImGui::ShowDemoWindow(&show_demo_window);
            }

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
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
