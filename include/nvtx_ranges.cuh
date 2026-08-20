#pragma once

// NVTX ranges for the host side of the hybrid versions, so that an nsys timeline shows *why* the
// GPU rows have gaps in them: which levels went to the CPU, where the copies sit, and how long the
// syncs actually block. Nsight already draws the kernels and the memcpys on their own rows, so the
// ranges here only mark what the host thread is doing.
//
// Header only (nvtx3), nothing to link. Build with -DNO_NVTX to compile them out entirely.

#ifndef NO_NVTX

#include <nvtx3/nvToolsExt.h>
#include <stdarg.h>
#include <stdio.h>

// One colour per kind of work, so the timeline is readable without reading any of the labels.
#define NVTX_COL_SETUP     0xFF9E9E9E // grey
#define NVTX_COL_GPU       0xFF43A047 // green
#define NVTX_COL_CPU       0xFF1E88E5 // blue
#define NVTX_COL_COPY      0xFFFB8C00 // orange
#define NVTX_COL_SYNC      0xFFE53935 // red   - this is the one that should worry you
#define NVTX_COL_REDUCE    0xFF00897B // teal
#define NVTX_COL_TRACEBACK 0xFF8E24AA // purple

static inline void nvtx_push(const char* name, uint32_t colour)
{
    nvtxEventAttributes_t attr;
    memset(&attr, 0, sizeof(attr));

    attr.version       = NVTX_VERSION;
    attr.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType     = NVTX_COLOR_ARGB;
    attr.color         = colour;
    attr.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;

    nvtxRangePushEx(&attr);
}

// For the per level ranges, where the level index and its width are the whole point of the label.
static inline void nvtx_pushf(uint32_t colour, const char* fmt, ...)
{
    char buf[128];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    nvtx_push(buf, colour);
}

#define NVTX_PUSH(name, colour) nvtx_push((name), (colour))
#define NVTX_PUSHF(colour, ...)  nvtx_pushf((colour), __VA_ARGS__)
#define NVTX_POP()               nvtxRangePop()

#else

#define NVTX_PUSH(name, colour) ((void)0)
#define NVTX_PUSHF(colour, ...) ((void)0)
#define NVTX_POP()              ((void)0)

#endif
