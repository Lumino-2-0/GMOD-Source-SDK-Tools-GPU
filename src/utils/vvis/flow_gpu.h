#pragma once
#include <CL/cl.h>
#include "bsplib.h"

// ============================
// FLOAT3 / FLOAT4 STRUCTURES
// ============================

typedef struct { float x, y, z; } float3;
typedef struct { float x, y, z, w; } float4;

// ============================
// KERNEL INLINE (OPTION A + B1)
// ============================

static const char* g_gpuPortalFlowKernels = R"CLC(

//
// ===============================================
// KERNEL 1 — GEOMETRIC PRUNE (optionnel)
// ===============================================
//
__kernel void separators(
    __global const float3* origins,
    __global const float* radius,
    __global const float4* planes,
    __global const float3* winding4,
    __global int* outMask,
    int portalIndex,
    int portalCount,
    int portalLongs
)
{
    int j = get_global_id(0);
    if(j >= portalCount) return;

    int word = j >> 5;
    int bit  = 1 << (j & 31);
    int offset = portalIndex * portalLongs + word;

    int old = outMask[offset];
    old |= bit;

    float3 a = origins[portalIndex];
    float3 b = origins[j];

    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float dz = b.z - a.z;

    float d2 = dx*dx + dy*dy + dz*dz;
    float rsum = radius[portalIndex] + radius[j];

    if(d2 > (rsum*rsum))
        old &= ~bit;

    outMask[offset] = old;
}

//
// ===============================================
// KERNEL 2 — FULL PVS FLOOD-FILL (IDENTIQUE CPU)
// ===============================================
//
__kernel void portalFlowIter(
    __global int* mask,
    __global int* mightsee,
    __global int* changed,
    int longs,
    int pCount)
{
    int p = get_global_id(0);
    if (p >= pCount) return;

    int baseP = p * longs;
    int updated = 0;

    // pour chaque q POSSIBLE
    for (int q = 0; q < pCount; q++)
    {
        int byte = q >> 5;
        int bit  = 1 << (q & 31);

        // si q n'est PAS dans mightSee[p], skip
        if ((mightsee[baseP + byte] & bit) == 0)
            continue;

        int baseQ = q * longs;
        int visible = 1;

        // test d'intersection
        for (int i = 0; i < longs; i++)
        {
            if ((mask[baseP + i] & mask[baseQ + i]) == 0)
            {
                visible = 0;
                break;
            }
        }

        if (!visible)
            continue;

        // SETBIT(mask[p], q)
        int old = mask[baseP + byte];
        int newv = old | bit;

        if (newv != old)
        {
            mask[baseP + byte] = newv;
            updated = 1;
        }
    }

    if (updated)
        atomic_add(changed, 1);
}

)CLC";



// ========================================================
// STRUCTURES GPU
// ========================================================

struct GPUPortalFlowCLContext
{
    bool initialized;

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;

    cl_kernel k_separators;
    cl_kernel k_merge;
    cl_kernel k_flowIter;

    cl_mem d_portalVis;
    cl_mem d_origins;
    cl_mem d_radius;
    cl_mem d_planes;
    cl_mem d_winding4;

    cl_mem d_frontier;
    cl_mem d_nextFrontier;
    cl_mem d_mightSee;
    cl_mem d_changed;

    int portalCount;
    int portalLongs;
};

extern GPUPortalFlowCLContext g_gpuPF;

// ========================================================
// API GPU PUBLIC
// ========================================================
bool InitOpenCL_PortalFlow();
void ShutdownOpenCL_PortalFlow();
bool AllocatePortalFlowBuffers();

bool PortalFlow_GPU(int portalIdx, portal_t* p);
void PortalFlow_GPU_Wrapper(int thread, int portalIndex);
void PortalFlow_FullGPU();

// For -TryGPU comparison, implemented in flow.cpp
void GPU_CPU_SampleCompare();

// ======================================================================
// GPU FULL PORTAL FLOW SUPPORT (Pipeline BFS)
// ======================================================================

typedef struct {
    cl_mem d_frontier;      // int[portalCount] : actifs à cette iteration
    cl_mem d_nextFrontier;  // int[portalCount]
    cl_mem d_changed;       // int(1) : indique si propagation continue
    cl_mem d_mightSee;      // int[portalCount * longs]
} GPUPortalFlowFull;

