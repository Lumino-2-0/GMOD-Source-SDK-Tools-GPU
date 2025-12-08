#pragma once
#include <CL/cl.h>
#include "bsplib.h"

// ============================================================================
// CONSTANTES GLOBALES
// ============================================================================
#define MAX_WINDING_POINTS 128        // CPU-accurate
#define GPU_WINDING_POOL_MAX 8000000  // 8 million float3 (~96 MB)
#define GPU_MAX_PLANES 4096           // sufficient for separators

// ============================================================================
// TYPES GPU SÉCURISÉS (pas de float3 OpenCL natif côté host)
// ============================================================================
typedef struct { float x, y, z; } float3;
typedef struct { float x, y, z, w; } float4;

// ============================================================================
// GPUFlowState - ÉTAT COMPLET DE CHAQUE PORTAL DURANT LE BFS
// Doit correspondre *exactement* à la version du kernel GPU (portalFlowExpand)
// ============================================================================
typedef struct
{
    int leaf;
    int portal;
    int mightOffset;
    int firstPass;

    int srcOffset;
    int srcCount;

    int passOffset;
    int passCount;
} GPUFlowState;

// ============================================================================
// STRUCTURE GLOBALE POUR LE FULL GPU BFS
// ============================================================================
struct GPUFlowFixed
{
    cl_mem d_stateCur = nullptr;       // array<GPUFlowState>
    cl_mem d_stateNext = nullptr;      // array<GPUFlowState>
    cl_mem d_stateCount = nullptr;     // int
    cl_mem d_stateNextCount = nullptr; // int

    cl_mem d_mightSee = nullptr;       // mightSee[] (same as CPU)
};
extern GPUFlowFixed g_gpuFF;

// ============================================================================
// CONTEXTE GPU PORTAL FLOW (OpenCL complet)
// ============================================================================
struct GPUPortalFlowCLContext
{
    // OpenCL objects
    cl_platform_id platform = nullptr;
    cl_device_id   device = nullptr;
    cl_context     context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program     program = nullptr;

    // =============================================
    // ACTIVE KERNELS
    // =============================================
    cl_kernel k_resetPool = nullptr;   // BLOC 2
    cl_kernel k_expand = nullptr;   // BLOC 6 (PortalFlowExpand)

    // Ces kernels ne sont plus utilisés dans ton pipeline,
    // MAIS on les déclare pour compatibilité avec ton InitOpenCL :
    cl_kernel k_gpuClipWinding = nullptr;  // BLOC 3
    cl_kernel k_gpuGenerateSep = nullptr;  // BLOC 4
    cl_kernel k_gpuClipToSep = nullptr;  // BLOC 5

    // =============================================
    // STATIC PORTAL BUFFERS
    // =============================================
    cl_mem d_portalVis = nullptr;
    cl_mem d_origins = nullptr;
    cl_mem d_radius = nullptr;
    cl_mem d_planes = nullptr;
    cl_mem d_winding4 = nullptr;
    cl_mem d_portalLeaf = nullptr;

    // =============================================
    // LEAF ADJACENCY
    // =============================================
    cl_mem d_leafPortalCount = nullptr;
    cl_mem d_leafPortalList = nullptr;

    int portalCount = 0;
    int portalLongs = 0;
    int numLeaves = 0;
    int maxPerLeaf = 0;

    bool initialized = false;

    // =============================================
    // GLOBAL WINDING POOL + OFFSETS
    // =============================================
    cl_mem d_windPool = nullptr;
    cl_mem d_windPoolCount = nullptr;

    cl_mem d_initSrcOffset = nullptr;
    cl_mem d_initSrcCount = nullptr;
};


extern GPUPortalFlowCLContext g_gpuPF;

// ============================================================================
// INTERFACES PUBLIQUES
// ============================================================================
bool InitOpenCL_PortalFlow();
void ShutdownOpenCL_PortalFlow();
void RecursiveLeafFlow_CPU(int leafnum, threaddata_t* thread, pstack_t* prevstack);
bool AllocatePortalFlowBuffers();
void BuildLeafPortalTable();
void PortalFlow_FullGPU();
void GPU_CPU_SampleCompare();

// ============================================================================
// KERNELS SOURCE (définis dans flow_gpu_kernels.cpp ou bloc intégré)
// ============================================================================
extern const char* g_gpuPortalFlowKernels;




static const char* g_gpuPortalFlowKernels = R"CLC(

#define MAX_WINDING_POINTS 128
#define EPS_PLANE 1e-5f
#define EPS_CLIP  1e-5f
#define EPS_WIND  1e-6f
#define EPS_DOT   1e-6f

// =============================================================
// STRUCT CPU <-> GPU IDENTIQUE
// =============================================================
typedef struct {
    int leaf;
    int portal;
    int mightOffset;
    int firstPass;
    int srcOffset;
    int srcCount;
    int passOffset;
    int passCount;
} GPUFlowState;

// =============================================================
// MATH UTILS
// =============================================================
inline float dot3(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline float3 sub3(float3 a, float3 b) {
    return (float3)(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline float3 mul3(float3 a, float s) {
    return (float3)(a.x*s, a.y*s, a.z*s);
}

inline float3 cross3(float3 a, float3 b)
{
    return (float3)(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}

// =============================================================
// POOL ALLOCATION : renvoie base index pour un winding
// =============================================================
inline int allocW(__global int* poolCount)
{
    return atomic_add(poolCount, MAX_WINDING_POINTS);
}

// =============================================================
// CHOP (CPU ChopWinding répliqué exactement ordre FP32)
// =============================================================
inline int chopExact(
    float3* outPts,
    const float3* inPts,
    int inCount,
    float3 N,
    float  D
){
    int o = 0;

    for (int i = 0; i < inCount; i++)
    {
        int ni = (i + 1) % inCount;

        float3 P1 = inPts[i];
        float3 P2 = inPts[ni];

        float d1 = dot3(P1, N) - D;
        float d2 = dot3(P2, N) - D;

        if (d1 >= -EPS_CLIP) {
            outPts[o++] = P1;
        }

        int diff = (d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0);
        if (diff)
        {
            float t = d1 / (d1 - d2);
            float3 mid = (float3)(
                P1.x + t*(P2.x - P1.x),
                P1.y + t*(P2.y - P1.y),
                P1.z + t*(P2.z - P1.z)
            );
            outPts[o++] = mid;
        }
    }

    return o;
}

// =============================================================
// CPU ClipToSeparators – VERSION B – 100% logique CPU
// =============================================================
inline int clipToSeparatorsExact(
    __global float3* pool,
    __global int*    poolCount,

    const float3* source,
    int srcCount,
    const float3* pass,
    int passCount,
    const float3* target,
    int tgtCount
){
    float3 A[MAX_WINDING_POINTS];
    float3 B[MAX_WINDING_POINTS];

    // init target -> A
    for (int i = 0; i < tgtCount; i++)
        A[i] = target[i];

    int aCount = tgtCount;

    // === CPU nested loops ===
    for (int i = 0; i < srcCount; i++)
    {
        int ni = (i + 1) % srcCount;
        float3 E = sub3(source[ni], source[i]);

        for (int j = 0; j < passCount; j++)
        {
            float3 V = sub3(pass[j], source[i]);
            float3 N = cross3(E, V);

            float len2 = dot3(N, N);
            if (len2 < EPS_WIND)
                continue;

            float inv = 1.0f / sqrt(len2);
            N = mul3(N, inv);

            float D = dot3(pass[j], N);

            // === EXACT CPU CLIP ===
            int o = chopExact(B, A, aCount, N, D);
            if (o == 0)
                return 0;

            // copy back
            for (int k = 0; k < o; k++)
                A[k] = B[k];
            aCount = o;
        }
    }

    // === allocate pool ===
    int base = allocW(poolCount);
    for (int i = 0; i < aCount; i++)
        pool[base+i] = A[i];

    return aCount;
}

// =============================================================
// KERNEL PRINCIPAL : portalFlowExpand
// =============================================================
__kernel void portalFlowExpand(

    __global const float3* origins,
    __global const float*  radius,
    __global const float4* planes,

    __global int*          visMask,
    __global const int*    mightSee,

    __global GPUFlowState* cur,
    __global GPUFlowState* next,
    __global int*          curCount,
    __global int*          nextCount,

    __global const int* leafPortalCount,
    __global const int* leafPortalList,

    int longs,
    int portalCount,
    int maxPerLeaf,

    __global const int* portalLeaf,

    // POOL
    __global float3* pool,
    __global int*    poolCount,

    // SRC INIT
    __global const int* initSrcOffset,
    __global const int* initSrcCount
){
    int idx = get_global_id(0);
    int active = *curCount;
    if (idx >= active) return;

    GPUFlowState st = cur[idx];
    int P = st.portal;
    int leaf = st.leaf;

    int base = st.mightOffset;

    // params of P
    float3 oP = origins[P];
    float4 plP = planes[P];
    float3 NP = (float3)(plP.x, plP.y, plP.z);
    float DP = plP.w;

    int num = leafPortalCount[leaf];

    for (int k = 0; k < num; k++)
    {
        int Q = leafPortalList[leaf*maxPerLeaf + k];
        if (Q < 0) continue;

        int byte = Q >> 5;
        int bit  = 1 << (Q & 31);

        if ((mightSee[base + byte] & bit) == 0)
            continue;

        int old = visMask[base + byte];
        if (old & bit)
            continue;

        // radius test
        float3 oQ = origins[Q];
        float3 dv = sub3(oQ, oP);
        float d2 = dot3(dv, dv);
        float rr = radius[P] + radius[Q];
        if (d2 > rr*rr)
            continue;

        // plane tests
        float side1 = dot3(oQ, NP) - DP;
        if (side1 <= -radius[Q])
            continue;

        float4 plQ = planes[Q];
        float3 NQ = (float3)(plQ.x, plQ.y, plQ.z);
        float DQ = plQ.w;

        float side2 = dot3(oP, NQ) - DQ;
        if (side2 <= -radius[P])
            continue;

        // =====================================================
        // FIRST PASS (CPU rule: skip separator clipping)
        // =====================================================
        if (st.firstPass == 1)
        {
            visMask[base + byte] = old | bit;

            int out = atomic_add(nextCount, 1);
            if (out < portalCount)
            {
                next[out].portal = Q;
                next[out].leaf   = portalLeaf[Q];

                next[out].mightOffset = Q * longs;
                next[out].firstPass   = 0;

                next[out].srcOffset = initSrcOffset[P];
                next[out].srcCount  = initSrcCount[P];

                next[out].passOffset = -1;
                next[out].passCount  = 0;
            }
            continue;
        }

        // =====================================================
        // FULL CLIP-TO-SEPARATORS (VERSION B)
        // =====================================================
        const float3* src  = pool + st.srcOffset;
        int srcCnt = st.srcCount;

        const float3* pass = (st.passOffset >= 0 ? pool + st.passOffset : src);
        int passCnt = (st.passOffset >= 0 ? st.passCount : srcCnt);

        int tgtOff = initSrcOffset[Q];
        int tgtCnt = initSrcCount[Q];
        const float3* tgt = pool + tgtOff;

        int newCount = clipToSeparatorsExact(
            pool,
            poolCount,
            src,  srcCnt,
            pass, passCnt,
            tgt,  tgtCnt
        );

        if (newCount == 0)
            continue;

        // mark visible
        visMask[base + byte] = old | bit;

        // push next BFS node
        int out = atomic_add(nextCount, 1);
        if (out < portalCount)
        {
            next[out].portal = Q;
            next[out].leaf   = portalLeaf[Q];

            next[out].mightOffset = Q * longs;
            next[out].firstPass   = 0;

            // keep same source
            next[out].srcOffset = st.srcOffset;
            next[out].srcCount  = st.srcCount;

            int newOffset = (*poolCount) - MAX_WINDING_POINTS;
            next[out].passOffset = newOffset;
            next[out].passCount  = newCount;
        }
    }
}

)CLC";
