#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_opengl3.h"
#include <SDL.h>

#if defined(_WIN32)
#include "Tracy.hpp"
#include "TracyC.h"
#elif defined(__linux__)
#include "Tracy.hpp"
#include "TracyC.h"
#elif defined(__EMSCRIPTEN__)
#define ZoneScoped
#endif

extern SDL_Window* window;

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

    void igRemoveSpacingH() {
        // Move back to the same line with 0 pixels of horizontal spacing
        ImGui::SameLine(0, 0); 
    }

    void igTextColoredC(float r, float g, float b, float a, const char* fmt, ...) {
        ZoneScoped;
        va_list args;
        va_start(args, fmt);
        ImGui::TextColoredV(ImVec4(r, g, b, a), fmt, args);
        va_end(args);
    }

    void igGetCursorScreenPos(ImVec2* pOut) {
        if (pOut) *pOut = ImGui::GetCursorScreenPos();
    }

    void igDrawRectFilled(ImVec2 min, ImVec2 max, ImU32 col) {
        // Access the current window's draw list and add the rect
        ImGui::GetWindowDrawList()->AddRectFilled(min, max, col);
    }

    void igDummy(ImVec2 size) {
        ImGui::Dummy(size);
    }

    void igBeginTooltip() {
        ImGui::BeginTooltip();
    }

    void igEndTooltip() {
        ImGui::EndTooltip();
    }

    void igGetMousePos(ImVec2* pOut) {
        if (pOut) *pOut = ImGui::GetIO().MousePos;
    }

    bool igIsMouseHoveringRect(ImVec2 min, ImVec2 max, bool clip) {
        return ImGui::IsMouseHoveringRect(min, max, clip);
    }

    void igTextColoredBC(float r, float g, float b, float a, float br, float bg, float bb, float ba, const char* text) {
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImVec2 textSize = ImGui::CalcTextSize(text);

        ImVec4 bgColor = ImVec4(br, bg, bb, ba);
        
        ImGui::GetWindowDrawList()->AddRectFilled(
            pos, 
            ImVec2(pos.x + textSize.x, pos.y + textSize.y), 
            ImGui::ColorConvertFloat4ToU32(bgColor)
        );

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(r, g, b, a));
        ImGui::TextUnformatted(text);
        ImGui::PopStyleColor();
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

    void SHM_SetCursorVisible(bool visible) {
        ZoneScoped;
        ImGui::SetMouseCursor(visible ? ImGuiMouseCursor_Arrow : ImGuiMouseCursor_None);
    }

    void SHM_SetWindowTitle(const char* title) {
        ZoneScoped;
        SDL_SetWindowTitle(window, title);
    }

    uint32_t SHM_GetWindowFlags() {
        ZoneScoped;
        return SDL_GetWindowFlags(window);
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

    void igSeparator() {
        ImGui::Separator();
    }

    bool igBeginTable(const char* label, int columns) {
        ZoneScoped;
        return ImGui::BeginTable(label, columns, ImGuiTableFlags_SizingStretchProp);
    }


    bool igBeginChild(const char* str_id, const ImVec2& size_arg, ImGuiChildFlags child_flags, ImGuiWindowFlags window_flags) {
        return ImGui::BeginChild(str_id, size_arg, child_flags, window_flags);
    }

    float shmConsoleFooterHeight() {
        return ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();
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

    float igFrameRate() {
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        return io.Framerate;
    }
}
