// flow_cpu_ray.h
//
// CPU engine of the raycast. It runs the very same kernel as the GPUs
// (flow_gpu_kernel.cu compiled as C++) on ranges of (src, dst) pairs, so the CPU
// threads can share the work with the GPU(s) and produce bit-identical results.
// Two builds of the same code: SSE2 (any x64 CPU) and AVX2 (picked at run time
// when the CPU and the OS support it).

#pragma once

#include "flow_gpu.h"

// Kernel arguments, host-side copy (same meaning as the GPU kernel arguments).
struct RayKernelArgs
{
	const GPUPortal     *portals;
	int                  numPortals;
	const GPULeaf       *leafs;
	int                  numLeafs;
	const int           *leafPortals;
	const float         *windingPts;
	const float         *windingEdges;
	const unsigned char *portalflood;
	unsigned int        *portalvisWords;
	int                  portalbytes;
	int                  samples;
	int                  maxSteps;
	int                  flags;
	int                  ringmask;
};

// Casts the rays of every candidate pair of the pair index range [begin, end)
// (index = src * numPortals + dst). Thread-safe: bits are set with atomic ORs.
typedef void ( *FlowCPU_RaycastRangeFn )( const RayKernelArgs &args, unsigned long long begin, unsigned long long end );

// Best build for this CPU (*pbAVX2 = true if the AVX2 build was selected).
// VVIS_CPU_ISA=sse2 forces the SSE2 build (testing).
FlowCPU_RaycastRangeFn FlowCPU_GetRaycastRange( bool *pbAVX2 );

// Precomputes, for every winding vertex k, the edge k -> k+1 and its margin:
// outEdges[4*i] = ( edge x, edge y, edge z, edgeEps * edge length ).
// Computed once here (IEEE /fp:precise) and read by every engine.
void FlowCPU_BuildEdgeTable( const GPUPortal *portals, int numPortals, const float *windingPts,
	float edgeEps, float *outEdges );