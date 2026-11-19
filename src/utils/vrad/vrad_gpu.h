// vrad_gpu.h
//
// Interface of the VRAD CUDA backend: transfer visibility, direct-light shadow
// rays and static prop indirect lighting.
// Implementation: vrad_gpu.cpp. Kernels: vrad_gpu_kernel.cu (embedded PTX).

#ifndef VRAD_GPU_H
#define VRAD_GPU_H
#ifdef _WIN32
#pragma once
#endif

#include "tier1/utlvector.h"

extern bool g_bUseCUDA;		// -cuda: enable the GPU backend
extern bool g_bCUDAVerify;	// -cudaverify: check every GPU ray against the CPU

// Loads CUDA and uploads the k-d tree and triangles of g_RtEnv. Must be called AFTER
// g_RtEnv.SetupAccelerationStructure(). Returns false (and clears g_bUseCUDA) if the
// GPU cannot be used; VRAD then falls back to the CPU.
bool VradGPU_Init();
void VradGPU_Shutdown();

// Traces numRays rays (6 floats each: origin, then end point) and writes one result
// byte per ray to out. skipId is a triangle ID to ignore (-1 for none).
// Thread-safe (internal locking).
//   mode 0: 1 = unobstructed, 0 = blocked (transfers).
//   mode 1: direct light; only the nearest hit counts and sky does not block.
//   mode 2: like mode 0, but 2 = no opaque hit and a transparent triangle was
//           crossed (the ray must be redone on the CPU).
bool VradGPU_TraceRays( const float *rays, int numRays, unsigned char *out, int mode, int skipId );

// Mode 0 (transfer visibility).
inline bool VradGPU_TraceVisibility( const float *rays, int numRays, unsigned char *out )
{
	return VradGPU_TraceRays( rays, numRays, out, 0, -1 );
}

// Statistics: -cudaverify counters and GPU totals. VradProf_PrintLights is in lightmap.cpp.
void VradGPU_AddVerifyStats( int rays, int mismatches );
void VradGPU_PrintStats();
void VradProf_PrintLights();

// Per-thread queue of deferred shadow rays (standard direct lights).
// mode 1 = record: TestLine stores its 4 rays and reports "visible".
// mode 2 = replay: TestLine reads the GPU results back in the same order; if the rays
// do not match the recorded ones, or the GPU returned 2 (transparent triangle), it
// falls back to the CPU. rays = 6 floats per ray, 4 rays per TestLine call.
struct VradRayQueue_t
{
	int mode;
	int pos;		// current TestLine call (replay)
	int misses;		// calls redone on the CPU
	CUtlVector<float> rays;
	CUtlVector<unsigned char> res;
	VradRayQueue_t() : mode( 0 ), pos( 0 ), misses( 0 ) {}
};
extern __declspec( thread ) VradRayQueue_t *g_pRayQueue;
void VradGPU_AddDirectStats( int rays, int misses );

// Static prop indirect lighting on the GPU. VradGPU_IndirectPrepare uploads the BSP and
// lightmaps (once, AFTER the final lightmaps have been computed). VradGPU_Indirect: rays =
// 6 floats (origin, delta); out = 4 floats (rgb, w); w = 0 no contribution, 1 contribution,
// 2 redo on the CPU.
extern bool g_bIndirectGPU;	// true once the BSP/lightmap data has been uploaded to the GPU
bool VradGPU_IndirectPrepare();
bool VradGPU_Indirect( const float *rays, int numRays, float *out );

#endif
