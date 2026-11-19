// flow_gpu.h
//
// Interface between VVIS (CPU) and the GPU visibility computation.
//
// Two backends, loaded dynamically (no static linking):
//   - CUDA   : nvcuda.dll, embedded PTX kernel (flow_gpu_kernel.cu)
//   - OpenCL : OpenCL.dll, kernel embedded as source text (flow_gpu_kernel.h)
// Both kernels perform exactly the same computation. If no GPU is usable,
// GPU_Init() returns false and VVIS falls back to the original CPU flow.

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
// Host side only: time-budgeted dense pass. `samples` is then only a cap; the host
// picks the number of points per portal so that the pass lasts at most
// max( budgetMin, budgetRel * previous pass duration ) seconds, based on the rate
// measured on the previous pass. Small maps thus get a fine dense pass, very large
// maps a light one (or none).
#define RAY_FLAG_AUTO       16
