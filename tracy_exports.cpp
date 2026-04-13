#if defined(_WIN32)
#include "Tracy.hpp"
#include "TracyC.h"
#elif defined(__linux__)
#include "Tracy.hpp"
#include "TracyC.h"
#elif defined(__EMSCRIPTEN__)
#include <stdint.h>
struct ___tracy_c_zone_context
{
    uint32_t id;
    int active;
};
typedef struct ___tracy_c_zone_context TracyCZoneCtx;
#endif

extern "C" {
    TracyCZoneCtx tracy_zone_begin_n(const char* name, int active) {
#ifdef __EMSCRIPTEN__
        return TracyCZoneCtx {0, 0};
#else
        TracyCZoneN(ctx, name, active);
        return ctx;
#endif
    }

    TracyCZoneCtx tracy_zone_begin_ns(const char* name, int depth, int active) {
#ifdef __EMSCRIPTEN__
        return TracyCZoneCtx {0, 0};
#else
        TracyCZoneNS(ctx, name, depth, active);
        return ctx;
#endif
    }

    void tracy_zone_end(TracyCZoneCtx ctx) {
#ifndef __EMSCRIPTEN__
        TracyCZoneEnd(ctx);
#endif
    }

    void tracy_zone_text(TracyCZoneCtx ctx, const char* txt, unsigned len) {
#ifndef __EMSCRIPTEN__
        TracyCZoneText(ctx, txt, len);
#endif
    }

    void tracy_zone_name(TracyCZoneCtx ctx, const char* txt, unsigned len) {
#ifndef __EMSCRIPTEN__
        TracyCZoneName(ctx, txt, len);
#endif
    }

    void tracy_zone_color(TracyCZoneCtx ctx, unsigned color) {
#ifndef __EMSCRIPTEN__
        TracyCZoneColor(ctx, color);
#endif
    }
}
