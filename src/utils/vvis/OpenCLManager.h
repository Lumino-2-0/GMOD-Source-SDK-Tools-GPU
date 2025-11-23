#pragma once

// ============================================================================
// OpenCLManager.h — Manager OpenCL complet pour VVIS_GPU ULTRA
// ============================================================================

#include <CL/cl.h>
#include "tier0/dbg.h"
#include "utlvector.h"
#include "mathlib/vector.h"

#define CL_TARGET_OPENCL_VERSION 120

// ============================================================================
// Structure principale du manager OpenCL
// ============================================================================
struct OpenCLManager
{
    bool ok = false;

    cl_context        context = nullptr;
    cl_command_queue  queue = nullptr;
    cl_device_id      device = nullptr;
    cl_program        program = nullptr;

    // === KERNELS FLOWGPU ===
    cl_kernel kernel_raycast = nullptr;  // Preset 1
    cl_kernel kernel_cone = nullptr;  // Preset 2
    cl_kernel kernel_hlod = nullptr;  // Preset 3
    cl_kernel kernel_bfs = nullptr;  // BFS core

    // =============================
    // API du manager
    // =============================
    void init_once();
    void destroy();
};


// ============================================================================
// Global unique
// ============================================================================
extern OpenCLManager g_clManager;


// ============================================================================
// Helpers internes
// ============================================================================
bool LoadFileToString(const char* path, CUtlVector<char>& data);
