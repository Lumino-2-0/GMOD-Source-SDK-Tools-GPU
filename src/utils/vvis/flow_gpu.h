#pragma once
#include <CL/cl.h>
#include "bsplib.h"
#include <vector>

// ============================================================================
// CONSTANTES GLOBALES
// ============================================================================
#define MAX_WINDING_POINTS 256        // CPU-accurate
#define GPU_WINDING_POOL_MAX 8000000  // 8 million float3 (~96 MB)
#define GPU_MAX_PLANES 4096           // sufficient for separators

// ============================================================================
// TYPES GPU SÉCURISÉS (pas de float3 OpenCL natif côté host)
// ============================================================================
<<<<<<< Updated upstream
typedef struct { float x, y, z; } float3;
typedef struct { float x, y, z, w; } float4;
=======
#pragma pack(push,1)
struct float2 { float x, y; };
#pragma pack(pop)

// 16-byte aligned float3/float4
struct float3 { float x, y, z, w; };
struct float4 { float x, y, z, w; };

// ============================================================================
// HYBRID VIS — CONSTANTES
// ============================================================================

// Nombre maximum de portails (sécurité)
#define MAX_PORTALS_GPU 8192

// Alignement 32 bits (comme VIS CPU)
typedef uint32_t visword_t;


// ============================================================================
// HYBRID VIS — GPU RESULT (MightSeeGPU)
// ============================================================================

// GPU produces this: a conservative visibility candidate mask
struct GPUMightSeeResult
{
    int portalCount;     // = g_numportals * 2
    int portalLongs;     // = portallongs

    // Bitmask: [portalCount][portalLongs]
    // Same layout as CPU portalflood / portalvis
    cl_mem d_mightSeeMask = nullptr;
};


// =======================================================
// GPU-assisted exact separator test (NO false positives)
// =======================================================
// Core (indices)
bool GPU_CanPassSeparators(
    winding_t* source,
    winding_t* pass,
    winding_t* target,
    int leafA,
    int leafB
);

// ============================================================================
// HYBRID VIS — GPU FILTER JOB
// ============================================================================

struct GPUFilterJob
{
    int portalCount;
    int portalLongs;

    // Inputs
    cl_mem d_portalOrigins = nullptr;   // float3[portalCount]
    cl_mem d_portalRadius = nullptr;   // float[portalCount]
    cl_mem d_portalPlanes = nullptr;   // float4[portalCount]

    // World data
    cl_mem d_worldTris = nullptr;
    cl_mem d_worldBVH = nullptr;
    int    worldTriCount = 0;
    int    worldBVHCount = 0;

    // Output
    GPUMightSeeResult result;
};

// ============================================================================
// HYBRID VIS — GLOBAL CONTEXT
// ============================================================================

struct GPUHybridVisContext
{
    bool enabled = false;

    // GPU job
    GPUFilterJob filterJob;

    // CPU-visible buffer (copie depuis GPU)
    std::vector<visword_t> cpuMightSeeMask;

    // Stats / debug
    int rejectedPairs = 0;
    int keptPairs = 0;
};

extern GPUHybridVisContext g_gpuHybridVis;

// ============================================================================
// LEAF HYBRID VIS — GPU RESULT
// ============================================================================

struct GPULeafMightSeeResult
{
    int leafCount;
    int leafLongs;

    cl_mem d_leafMightSee = nullptr;   // GPU bitmask
    std::vector<visword_t> cpuLeafMightSee; // CPU mirror
};

// ============================================================================
// LEAF HYBRID VIS — CONTEXT
// ============================================================================

struct GPULeafHybridVisContext
{
    bool enabled = false;

    int leafCount = 0;
    int leafLongs = 0;

    GPULeafMightSeeResult result;
};

extern GPULeafHybridVisContext g_gpuLeafHybridVis;



// =============================================
// LEAF AABB (GPU)
// =============================================
struct LeafAABBGPU
{
    float3 mins;
    float3 maxs;
};

// =======================================================
// WORLD BRUSH AABB (GPU)
// =======================================================
typedef struct
{
    float3 mins;
    float3 maxs;
} WorldAABBGPU;


// BVH node (global + leaf)
struct BVHNodeGPU
{
    float3 aabbMin;       // 16 bytes
    float3 aabbMax;       // 16 bytes
    int left;             // -1 = leaf
    int right;
    int firstPrim;        // index dans primitive table
    int primCount;        // 0 si inner node
};


// Primitive SDF simple (spheres only for fast tests)
struct SDFPrimGPU
{
    float3 pos;       // center
    float radius;     // sphere radius
};

// ============================================================================
// WORLD TRIANGLE (CPU -> GPU, PresetGPU 3)
// ============================================================================
struct WorldTriCPU
{
    Vector a;
    Vector b;
    Vector c;
};

// ============================================================================
// World triangle — GPU format (PresetGPU 3)
// ============================================================================
struct WorldTriGPU
{
    float3 a;
    float3 b;
    float3 c;
};


// RayHit
struct RayHitGPU
{
    float t;
    int hit;
    float pad0;
    float pad1;
};

>>>>>>> Stashed changes

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
    cl_kernel k_rayTriangleBVH = nullptr;
    cl_kernel k_hybridFilter = nullptr;
	cl_kernel k_separatorReject = nullptr;

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
<<<<<<< Updated upstream
=======
    cl_mem d_portalArea = nullptr;   // NEW: portal -> area index
    cl_mem d_leafArea; // leaf -> area
	cl_mem d_portalCenters = nullptr;
    cl_mem d_portalNormals = nullptr;
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
=======
    cl_mem d_ultraMask = nullptr;
    cl_mem d_ultraMaskTemp = nullptr;

    cl_mem d_areaportals = nullptr;
    int areaportalCount = 0;

    cl_mem d_leafVis;
    int leafLongs;

    cl_mem d_visitedExpand = nullptr;


    // LEAF HYBRID VIS KERNEL
    cl_kernel k_leafHybridFilter = nullptr;

    // LEAF DATA
    cl_mem d_leafAABBs = nullptr;

    // PresetGPU 2 (World occlusion) — ray tests + BVH
    cl_kernel k_ultraWorldOcc = nullptr;

    // ============================================================================
    // ULTRA GPU BUFFERS
    // ============================================================================
    cl_mem d_sceneSDF = nullptr;
    cl_mem d_cones = nullptr;

    // PresetGPU 2: World occluders (AABB list) + BVH nodes (optional)
    cl_mem d_worldAABBs = nullptr;
    int worldAABBCount = 0;
    int worldBrushCount;

    // ============================================================================
    // PresetGPU 3 — World triangles (GPU)
    // ============================================================================
    cl_mem d_worldTris = nullptr;
    int worldTriCount = 0;

    cl_mem d_worldBVH = nullptr;
    int worldBVHCount = 0;

    cl_mem d_ultraRejectMask = nullptr;

>>>>>>> Stashed changes
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
// void PortalFlow_FullGPU(); NE DOIT PLUS ETRE APPELÉ
void GPU_CPU_SampleCompare();
void BuildPortalOrderByMightSee();
void PortalFlow_CPU_Ordered(int iThread, int workIndex);

<<<<<<< Updated upstream
// ============================================================================
// KERNELS SOURCE (définis dans flow_gpu_kernels.cpp ou bloc intégré)
// ============================================================================
extern const char* g_gpuPortalFlowKernels;
=======
// Ultra preset kernels + calls
// void PortalFlow_ULTRA_GPU(); NE DOIT PLUS ETRE APPELÉ
void RunKernel_LeafAABBOcclusion();

// ============================================================================
// HYBRID VIS — PUBLIC API
// ============================================================================

// GPU side
bool InitGPUHybridVis();
bool RunGPUHybridFilter();
void ShutdownGPUHybridVis();
bool InitGPULeafHybridVis();
bool RunGPULeafHybridFilter();
void ApplyLeafHybridToPortals();



// CPU side
void ApplyHybridMightSeeToCPU();
>>>>>>> Stashed changes




static const char* g_gpuPortalFlowKernels = R"CLC(

<<<<<<< Updated upstream
#define MAX_WINDING_POINTS 128
#define EPS_PLANE 1e-5f
#define EPS_CLIP  1e-5f
#define EPS_WIND  1e-6f
#define EPS_DOT   1e-6f

// =============================================================
// STRUCT CPU <-> GPU IDENTIQUE
// =============================================================
typedef struct {
=======
#define MAX_WINDING_POINTS 256
#define ON_EPSILON         0.01f
#define EPS_CLIP           0.01f
#define BVH_STACK_MAX 128

// ============================================================================
// GPU TYPES REQUIRED BY ULTRA & RBVH
// ============================================================================
typedef struct {
    float3 aabbMin;
    float3 aabbMax;
    int left;
    int right;
    int firstPrim;
    int primCount;
} BVHNodeGPU;

typedef struct {
    float3 pos;
    float radius;
} SDFPrimGPU;

// =======================================================
// World triangle (GPU) — MUST MATCH C++ LAYOUT
// =======================================================
typedef struct
{
    float3 a;
    float3 b;
    float3 c;
} WorldTriGPU;

// =========================
//      GPUWinding
// =========================
typedef struct {
    int numpoints;
    float3 points[MAX_WINDING_POINTS];
} GPUWinding;

// =========================
//    GPUFlowState
// =========================
typedef struct {
    int portal;
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
// =============================================================
// POOL ALLOCATION : renvoie base index pour un winding
// =============================================================
inline int allocW(__global int* poolCount)
=======
inline float3 add3s(float3 a, float3 b) {
    return (float3)(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline float3 sub3s(float3 a, float3 b) {
    return (float3)(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline float3 mul3s(float3 v, float s) {
    return (float3)(v.x*s, v.y*s, v.z*s);
}

inline float3 normalize3s(float3 v) {
    float len2 = dot(v,v);
    if (len2 <= 0.0f) return v;
    float inv = native_rsqrt(len2);
    return (float3)(v.x*inv, v.y*inv, v.z*inv);
}

// ============================================================================
//    ClipWindingEpsilon_CPU
// ============================================================================

inline GPUWinding ClipWindingEpsilon_CPU(
    GPUWinding inW,
    float3 normal,
    float  dist,
    int* ok)
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
            float3 mid = (float3)(
                P1.x + t*(P2.x - P1.x),
                P1.y + t*(P2.y - P1.y),
                P1.z + t*(P2.z - P1.z)
            );
            outPts[o++] = mid;
=======
            float3 mid;

            // CPU HACK EXACT — COMPOSANTS INDIVIDUELS
            float n0 = normal.x, n1 = normal.y, n2 = normal.z;

            float p1x=p1.x, p1y=p1.y, p1z=p1.z;
            float p2x=p2.x, p2y=p2.y, p2z=p2.z;

            // X
            if (n0==1.0f) mid.x =  dist;
            else if (n0==-1.0f) mid.x = -dist;
            else mid.x = p1x + t*(p2x - p1x);

            // Y
            if (n1==1.0f) mid.y =  dist;
            else if (n1==-1.0f) mid.y = -dist;
            else mid.y = p1y + t*(p2y - p1y);

            // Z
            if (n2==1.0f) mid.z =  dist;
            else if (n2==-1.0f) mid.z = -dist;
            else mid.z = p1z + t*(p2z - p1z);

            front.points[newCount++] = mid;

            if (newCount >= MAX_WINDING_POINTS)
            {
                *ok = 0;
                GPUWinding empty; empty.numpoints = 0;
                return empty;
            }
>>>>>>> Stashed changes
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
)CLC"

<<<<<<< Updated upstream
// =============================================================
// KERNEL PRINCIPAL : portalFlowExpand
// =============================================================
__kernel void portalFlowExpand(
=======
/*
	HERE GOES TO THE RAY VS TRIANGLE + BVH INTERSECTION CODE + MAIN PortalFlowExpand KERNEL !
*/

>>>>>>> Stashed changes

R"CLC(

// =======================================================
// Ray vs Triangle (Möller–Trumbore)
// =======================================================
inline int RayTriangleIntersect(
    float3 ro,
    float3 rd,
    float3 a,
    float3 b,
    float3 c,
    float* tOut
)
{
    float3 e1 = b - a;
    float3 e2 = c - a;

    float3 p = cross(rd, e2);
    float det = dot(e1, p);

    if (fabs(det) < 1e-6f)
        return 0;

    float invDet = 1.0f / det;

    float3 s = ro - a;
    float u = dot(s, p) * invDet;
    if (u < 0.0f || u > 1.0f)
        return 0;

<<<<<<< Updated upstream
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
=======
    float3 q = cross(s, e1);
    float v = dot(rd, q) * invDet;
    if (v < 0.0f || u + v > 1.0f)
        return 0;

    float t = dot(e2, q) * invDet;
    if (t > 0.0001f)
    {
        *tOut = t;
        return 1;
    }

    return 0;
}

// =======================================================
// Ray vs AABB (slab)
// =======================================================
inline int RayAABB(
    float3 ro,
    float3 invDir,
    float3 bmin,
    float3 bmax,
    float tMax
)
{
    float3 t1 = (bmin - ro) * invDir;
    float3 t2 = (bmax - ro) * invDir;

    float3 tmin = fmin(t1, t2);
    float3 tmax = fmax(t1, t2);

    float lo = fmax(fmax(tmin.x, tmin.y), tmin.z);
    float hi = fmin(fmin(tmax.x, tmax.y), tmax.z);

    return (hi >= lo && lo <= tMax);
}

inline int RayBlockedBVH_Internal(
    float3 ro,
    float3 rq,
    __global const WorldTriGPU* tris,
    __global const BVHNodeGPU* bvhNodes,
    int triCount,
    int bvhCount
)
{
    float3 dir = rq - ro;
    float dist = length(dir);

    float len2 = dot(dir, dir);
    if (len2 < 1e-8f) 
        return false;
    dir *= native_rsqrt(len2);

    dir.x = fabs(dir.x) < 1e-6f ? 1e-6f : dir.x;
    dir.y = fabs(dir.y) < 1e-6f ? 1e-6f : dir.y;
    dir.z = fabs(dir.z) < 1e-6f ? 1e-6f : dir.z;

    float3 invDir = (float3)(
        1.0f / dir.x,
        1.0f / dir.y,
        1.0f / dir.z
    );

    int stack[BVH_STACK_MAX];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0)
    {
        int n = stack[--sp];
        if (n < 0 || n >= bvhCount)
>>>>>>> Stashed changes
            continue;

<<<<<<< Updated upstream
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
=======
        BVHNodeGPU node = bvhNodes[n];

        if (!RayAABB(ro, invDir, node.aabbMin, node.aabbMax, dist))
            continue;

        if (node.primCount > 0)
        {
            for (int i = 0; i < node.primCount; i++)
            {
                int triIdx = node.firstPrim + i;
                if (triIdx >= triCount)
                    continue;

                WorldTriGPU t = tris[triIdx];
                float tHit;

                if (RayTriangleIntersect(ro, dir, t.a, t.b, t.c, &tHit))
                {
                    if (tHit < dist)
                        return 1; // BLOQUÉ
                }
            }
        }
        else
        {
            if (sp + 2 < BVH_STACK_MAX)
            {
                stack[sp++] = node.left;
                stack[sp++] = node.right;
            }
        }
    }

    return 0; // PAS BLOQUÉ
}


)CLC"

/* 
    THE FIRST PART OF THE VVIS IS COMPLETE BUT I CANNOT REALLY PROVIDE THE SAME "GOOD" QUALITY AS CPU ON GPU WITHOUT THE NEXT KERNELS.
    THEY ARE NOT USED IN THE CURRENT PIPELINE, BUT THEY FORM THE BASIS FOR FUTURE IMPROVEMENTS
    WITH BVH + SDF ACCELERATION.
    Why ? Because the current portal flow GPU kernel is a strict mirror of the CPU version,
    And the GPU cannot do RecursiveLeafFlow with the same quality as CPU without acceleration structures. GPU use BFS system for PVS propagation !
*/


R"CLC(
// =======================================================
// WORLD BRUSH AABB (GPU)
// =======================================================
typedef struct
{
    float3 mins;
    float3 maxs;
} WorldAABBGPU;


// =======================================================
// WORLD BRUSH OCCLUSION
// =======================================================
__kernel void ultra_worldOcclusion(
    __global const int* portalLeaf,
    __global const float4* portalCenter, // xyz = center
    __global const WorldAABBGPU* brushes,
    int brushCount,
    __global int* ultraRejectMask,
    int portalCount,
    int longs
)
{
    int P = get_global_id(0);
    if (P >= portalCount) return;

    float3 p0 = portalCenter[P].xyz;
    int base = P * longs;

    for (int Q = 0; Q < portalCount; Q++)
    {
        int word = Q >> 5;
        int bit  = 1 << (Q & 31);

        if (!(ultraRejectMask[base + word] & bit))
            continue;

        float3 p1 = portalCenter[Q].xyz;
        float3 dir = p1 - p0;

        for (int b = 0; b < brushCount; b++)
        {
            WorldAABBGPU a = brushes[b];

            // Ray-AABB slab test (conservatif)
            float tmin = 0.0f;
            float tmax = 1.0f;

            for (int ax = 0; ax < 3; ax++)
            {
                float o = get3(p0, ax);
                float d = get3(dir, ax);

                float inv = 1.0f / d;

                float t1 = (get3(a.mins, ax) - o) * inv;
                float t2 = (get3(a.maxs, ax) - o) * inv;


                float lo = fmin(t1, t2);
                float hi = fmax(t1, t2);

                tmin = fmax(tmin, lo);
                tmax = fmin(tmax, hi);

                if (tmin > tmax)
                    break;
            }

            if (tmin <= tmax)
            {
                ultraRejectMask[base + word] |= bit;
                break;
            }
>>>>>>> Stashed changes
        }
    }
}

<<<<<<< Updated upstream
)CLC";
=======


// =======================================================
// PresetGPU 3 — RayTriangleBVH
// =======================================================
__kernel void rayTriangleBVH(
    __global const float3* portalOrigins,
    __global int*          portalVis,
    __global int*          ultraRejectMask,

    __global const WorldTriGPU* tris,
    __global const BVHNodeGPU* bvhNodes,

    int triCount,
    int bvhCount,
    int portalCount,
    int longs
)
{
    int P = get_global_id(0);
    if (P >= portalCount)
        return;

    float3 ro = portalOrigins[P];
    int base = P * longs;

    for (int Q = 0; Q < portalCount; Q++)
    {
        int word = Q >> 5;
        int bit  = 1 << (Q & 31);


        // ===== DEBUG FORCÉ =====
        if (P == 0 && Q == 1)
        {
            ultraRejectMask[base + word] |= bit;
            continue; // IMPORTANT : on stoppe ici
        }
        // =======================


        float3 rq = portalOrigins[Q];
        float3 dir = rq - ro;
        float dist = length(dir);

        float len2 = dot(dir, dir);
        if (len2 < 1e-8f) continue;
        dir *= native_rsqrt(len2);

        dir.x = fabs(dir.x) < 1e-6f ? 1e-6f : dir.x;
        dir.y = fabs(dir.y) < 1e-6f ? 1e-6f : dir.y;
        dir.z = fabs(dir.z) < 1e-6f ? 1e-6f : dir.z;

        float3 invDir = (float3)(
            1.0f / dir.x,
            1.0f / dir.y,
            1.0f / dir.z
        );

        // Explicit stack (small)
        int stack[BVH_STACK_MAX];
        int sp = 0;
        stack[sp++] = 0;

        int blocked = 0;

        while (sp > 0)
        {
            int n = stack[--sp];
            if (n < 0 || n >= bvhCount)
                continue;

            BVHNodeGPU node = bvhNodes[n];

            if (!RayAABB(ro, invDir, node.aabbMin, node.aabbMax, dist))
                continue;

            if (node.primCount > 0)
            {
                for (int i = 0; i < node.primCount; i++)
                {
                    int triIdx = node.firstPrim + i;
                    if (triIdx >= triCount)
                        continue;

                    WorldTriGPU t = tris[triIdx];
                    float tHit;

                    if (RayTriangleIntersect(ro, dir, t.a, t.b, t.c, &tHit))
                    {
                        if (tHit < dist)
                        {
                            blocked = 1;
                            break;
                        }
                    }
                }
                if (blocked) break;
            }
            else
            {
                if (sp + 2 < BVH_STACK_MAX)
                {
                    stack[sp++] = node.left;
                    stack[sp++] = node.right;
                }
            }
        }
        
        if (blocked)
        {
            ultraRejectMask[base + word] |= bit;
        }
    }
}



// =======================================================
// HYBRID VIS
// =======================================================

__kernel void Hybrid_MightSee_Filter(
    __global const float3* portalOrigins,
    __global const float3* portalNormals,
    __global const WorldTriGPU* tris,
    __global const BVHNodeGPU* bvhNodes,

    __global uint* outMask,   // [portalCount * longs]
    int triCount,
    int bvhCount,
    int portalCount,
    int longs
)
{
    int gid = get_global_id(0);
    int P = gid / portalCount;
    int Q = gid % portalCount;

    if (P >= portalCount || Q >= portalCount)
        return;

    int word = Q >> 5;
    uint bit = 1u << (Q & 31);
    int base = P * longs;

    // self visible
    if (P == Q)
    {
        atomic_or(&outMask[base + word], bit);
        return;
    }

    float3 ro = portalOrigins[P];
    float3 rq = portalOrigins[Q];

    float3 nP = normalize(portalNormals[P]);
    float3 nQ = normalize(portalNormals[Q]);

    float3 dirBase = rq - ro;
    float distBase = length(dirBase);

    if (distBase < 1e-4f)
    {
        atomic_or(&outMask[base + word], bit);
        return;
    }

    // ray origins offsets (8 rays)
    float eps = 4.0f;

    float3 rayOrig[4];
    rayOrig[0] = ro;
    rayOrig[1] = ro + nP * eps;
    rayOrig[2] = ro - nP * eps;
    rayOrig[3] = ro + cross(nP, (float3)(1,0,0)) * eps;

    int visible = 0;

    // for each ray
    for (int r = 0; r < 4 && !visible; r++)
    {
        float3 ro_r = rayOrig[r];
        float3 rq_r = rq + nQ * eps;

        float3 dir = rq_r - ro_r;
        float dist = length(dir);

        float len2 = dot(dir, dir);
        if (len2 < 1e-8f) 
            return;
        dir *= native_rsqrt(len2);

        dir.x = fabs(dir.x) < 1e-6f ? 1e-6f : dir.x;
        dir.y = fabs(dir.y) < 1e-6f ? 1e-6f : dir.y;
        dir.z = fabs(dir.z) < 1e-6f ? 1e-6f : dir.z;

        float3 invDir = (float3)(
            1.0f / dir.x,
            1.0f / dir.y,
            1.0f / dir.z
        );

        int stack[BVH_STACK_MAX];
        int sp = 0;
        stack[sp++] = 0;

        int blocked = 0;

        while (sp > 0)
        {
            int n = stack[--sp];
            if (n < 0 || n >= bvhCount)
                continue;

            BVHNodeGPU node = bvhNodes[n];

            float3 t1 = (node.aabbMin - ro_r) * invDir;
            float3 t2 = (node.aabbMax - ro_r) * invDir;

            float3 tmin = fmin(t1, t2);
            float3 tmax = fmax(t1, t2);

            float lo = fmax(fmax(tmin.x, tmin.y), tmin.z);
            float hi = fmin(fmin(tmax.x, tmax.y), tmax.z);

            if (hi < 0.0f || lo > dist || lo > hi)
                continue;

            if (node.primCount > 0)
            {
                for (int i = 0; i < node.primCount; i++)
                {
                    int triIdx = node.firstPrim + i;
                    if (triIdx >= triCount)
                        continue;

                    WorldTriGPU t = tris[triIdx];
                    float tHit;

                    if (RayTriangleIntersect(ro_r, dir, t.a, t.b, t.c, &tHit))
                    {
                        if (tHit > 0.0001f && tHit < dist)
                        {
                            blocked = 1;
                            break;
                        }
                    }
                }
                if (blocked)
                    break;
            }
            else
            {
                if (sp + 2 < BVH_STACK_MAX)
                {
                    stack[sp++] = node.left;
                    stack[sp++] = node.right;
                }
            }
        }

        if (!blocked)
            visible = 1;
    }

    if (visible)
    {
        atomic_or(&outMask[base + word], bit);
    }
}


// =======================================================
// LEAF HYBRID VIS — AABB CONSERVATIVE FILTER
// =======================================================
__kernel void Leaf_MightSee_Filter(
    __global const LeafAABBGPU* leafAABBs,
    __global uint* outMask,   // [leafCount * leafLongs]
    int leafCount,
    int leafLongs
)
{
    int gid = get_global_id(0);
    int A = gid / leafCount;
    int B = gid % leafCount;

    if (A >= leafCount || B >= leafCount)
        return;

    int word = B >> 5;
    int bit  = 1u << (B & 31);
    int base = A * leafLongs;

    // Self always visible
    if (A == B)
    {
        atomic_or(&outMask[base + word], bit);
        return;
    }

    LeafAABBGPU a = leafAABBs[A];
    LeafAABBGPU b = leafAABBs[B];

    // AABB overlap / adjacency (VERY conservative)
    if (
        a.maxs.x < b.mins.x || a.mins.x > b.maxs.x ||
        a.maxs.y < b.mins.y || a.mins.y > b.maxs.y ||
        a.maxs.z < b.mins.z || a.mins.z > b.maxs.z
    )
    {
        // Separated -> MAY be occluded -> reject
        return;
    }

    // Keep visibility
    atomic_or(&outMask[base + word], bit);
}

// =======================================================
// GPU EXACT SEPARATOR EARLY-REJECT (MULTI-RAY)
// =======================================================
// Retourne 0 = IMPOSSIBLE (reject CPU)
// Retourne 1 = POSSIBLE (laisser CPU faire)
// =======================================================

__kernel void SeparatorReject_MultiRay(
    __global const float3* srcPts,   int srcCount,
    __global const float3* passPts,  int passCount,
    __global const float3* tgtPts,   int tgtCount,

    __global const WorldTriGPU* tris,
    __global const BVHNodeGPU* bvh,
    int triCount,
    int bvhCount,

    volatile __global int* outResult
)
{
    int tid = get_global_id(0);
    if (tid >= srcCount * tgtCount)
        return;

    int i = tid / tgtCount;
    int j = tid % tgtCount;

    float3 ro = srcPts[i];
    float3 rq = tgtPts[j];
    float3 dir = rq - ro;
    float dist = length(dir);

    if (dist < 1e-4f)
        return;

    float len2 = dot(dir, dir);

    if (len2 < 1e-8f) 
        return;

    dir *= native_rsqrt(len2);

    // évite div0
    dir.x = fabs(dir.x) < 1e-6f ? 1e-6f : dir.x;
    dir.y = fabs(dir.y) < 1e-6f ? 1e-6f : dir.y;
    dir.z = fabs(dir.z) < 1e-6f ? 1e-6f : dir.z;

    float3 invDir = (float3)(
        1.0f / dir.x,
        1.0f / dir.y,
        1.0f / dir.z
    );

    int stack[BVH_STACK_MAX];
    int sp = 0;
    stack[sp++] = 0;
    
    if (*outResult)
        return;
    
    // Si UN SEUL rayon passe -> POSSIBLE
    int blocked = 0;

    while (sp > 0)
    {
        int n = stack[--sp];
        if (n < 0 || n >= bvhCount)
            continue;

        BVHNodeGPU node = bvh[n];

        // Ray vs AABB
        float3 t1 = (node.aabbMin - ro) * invDir;
        float3 t2 = (node.aabbMax - ro) * invDir;

        float3 tmin = fmin(t1, t2);
        float3 tmax = fmax(t1, t2);

        float lo = fmax(fmax(tmin.x, tmin.y), tmin.z);
        float hi = fmin(fmin(tmax.x, tmax.y), tmax.z);

        if (hi < 0.0f || lo > dist || lo > hi)
            continue;

        if (node.primCount > 0)
        {
            for (int k = 0; k < node.primCount; k++)
            {
                int triIdx = node.firstPrim + k;
                if (triIdx >= triCount)
                    continue;

                WorldTriGPU t = tris[triIdx];
                float tHit;

                if (RayTriangleIntersect(ro, dir, t.a, t.b, t.c, &tHit))
                {
                    if (tHit > 0.0001f && tHit < dist)
                    {
                        blocked = 1;
                        break;
                    }
                }
            }
            if (blocked)
                break;
        }
        else
        {
            if (sp + 2 < BVH_STACK_MAX)
            {
                stack[sp++] = node.left;
                stack[sp++] = node.right;
            }
        }

        if (*outResult)
            return;
    }



    // Si UN rayon n'est PAS bloqué -> POSSIBLE
    if (!blocked)
        atomic_or(outResult, 1);
}



)CLC";
>>>>>>> Stashed changes
