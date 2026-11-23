// flow_gpu.h
//
// Interface between VVIS (CPU) and the GPU visibility computation.
//
// Three kinds of ray engines work together on the same pass:
//   - CUDA   : nvcuda.dll, embedded PTX kernel (flow_gpu_kernel.cu)
//   - OpenCL : OpenCL.dll, kernel embedded as source text (flow_gpu_kernel.h);
//              used for non-NVIDIA GPUs, including integrated GPUs
//   - CPU    : the same kernel source compiled natively (flow_cpu_ray*.cpp),
//              SSE2 or AVX2 build picked at run time
// All engines compute bit-identical results, so the PVS does not depend on which
// engine processed which pair. GPU_Init() returns false only if no engine at all
// can be created; VVIS then falls back to the original CPU flow.

#pragma once

// --- GPUPortal ----------------------------------------------------------------
// Flat copy of a portal_t: plane, destination leaf, bounding sphere and
// location of the winding vertices in the winding_pts array.
// Only 4-byte fields, so the structure has the same layout on the CPU, in
// CUDA and in OpenCL C. Do not change the field order.

typedef struct
{
	float  plane_normal[3];
	float  plane_dist;
	int    leaf;
	float  origin[3];
	float  radius;
	int    winding_start;
	int    winding_count;
	int    portalflood_offset;
	int    status;
	int    nummightsee;
} GPUPortal;

// --- GPULeaf --------------------------------------------------------------------
// A leaf knows its outgoing portals: they are stored in the leaf_portals array,
// between indices portals_start and portals_start + portals_count.

typedef struct
{
	int portals_start;
	int portals_count;
} GPULeaf;

// --- Host interface -----------------------------------------------------------

bool GPU_Init( void );
void GPU_Shutdown( void );
// One raycast pass.
//   flags    : RAY_FLAG_* below.
//   samples  : without HALTON, cap on the vertices used per ring; with HALTON,
//              number of points spread over the whole portal surface.
//   ringmask : enabled rings (without HALTON): bit 0 = 90 %, bit 1 = 50 %,
//              bit 2 = 99 % (corners). Points per portal = 1 + popcount * vertices.
//   budgetRel, budgetMin : see RAY_FLAG_AUTO.
// Passes > 0 only reprocess the candidate pairs not yet proven visible;
// the results are merged with OR.
typedef struct
{
	int samples;
	int flags;
	int ringmask;
	float budgetRel;
	float budgetMin;
} GPUPassConfig;

bool GPU_RunPortalFlowRaycast( int sorted_portal_count, const GPUPassConfig *passes, int numPasses );

// Kernel flags (same values as in flow_gpu_kernel.cu / flow_gpu_kernel.h).
#define RAY_FLAG_EXTEND     1
#define RAY_FLAG_EXHAUSTIVE 2
#define RAY_FLAG_HALTON     4
#define RAY_FLAG_MIRROR     8
// Host side only: work-budgeted dense pass. `samples` is then only a cap; the host
// picks the largest number of points per portal n such that
//   candidates * n^2 <= max( budgetMin * 1e6, budgetRel * previous pass work )
// where the work of a pass is candidates * points^2 (ray-units). The budget depends
// only on the map, never on the hardware or the measured speed, so the PVS is the
// same on every machine. Small maps get a fine dense pass, very large maps a light
// one (or none, below 6 points).
#define RAY_FLAG_AUTO       16

// Engine selection (vvis.cpp options): -gpuonly disables the CPU ray engine,
// -nogpu uses only the CPU ray engine.
extern bool g_bRayGpuOnly;
extern bool g_bRayNoGpu;
