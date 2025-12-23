//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose: GPU-accelerated portal flow (OpenCL)
// $NoKeywords: $
//=============================================================================//

#include "vis.h"
#include "flow_gpu.h"
#include <CL/cl.h>
#include <vector>
#include <mutex>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <iomanip>
#include <icommandline.h>
#include "threads.h"
#include <unordered_map>
#include <cstdint>

// =============================================================
// UNIFIED EPSILON CONSTANTS (CPU <-> GPU CONSISTENCY)
// =============================================================
#define VIS_EPSILON_PLANE       1e-5f
#define VIS_EPSILON_DOT         1e-5f
#define VIS_EPSILON_CLIP        1e-5f
#define VIS_EPSILON_WINDING     1e-5f
#define VIS_EPSILON_COLINEAR    1e-6f


static std::unordered_map<uint64_t, bool> g_gpuSeparatorCache;

// Nombre total de points dynamiques
static int g_totalWindingPoints = 0;

// Offsets CPU -> GPU (CPU side)
static std::vector<int> g_windingOffsetsCPU;
static std::vector<float3> g_windingPointsCPU;


static std::vector<int> g_portalOrderCPU;


// Buffers GPU
static cl_mem d_windingPoints = nullptr;   // float3[]
static cl_mem d_windingOffsets = nullptr;  // int[]
static cl_mem d_windingCounts = nullptr;   // int[]

cl_mem d_poolPoints = nullptr;         // float3[N * MAX_WINDING_POINTS]
cl_mem d_poolCount = nullptr;         // int[1]

cl_mem d_initSrcOffset = nullptr;      // int[portalCount]
cl_mem d_initSrcCount = nullptr;      // int[portalCount]


// Fonction d’upload (définie plus bas)
bool UploadDynamicWindingsToGPU(cl_context ctx, cl_command_queue q, cl_int* errOut);

// =======================================================
// Patch GPU protos
// =======================================================
bool AllocatePortalFlowBuffers();
void BuildLeafPortalTable();

// Utilisé pour reconstruire les adjacency tables
void BuildFlatLeafPortalArrays(std::vector<int>& outCount, std::vector<int>& outList);

static std::mutex g_trace_mutex;
static std::ofstream g_trace_file;
static std::atomic<bool> g_trace_inited{ false };

// ========= GPU PRUNE TUNING =========

int g_gpuPreset = 3;




GPUPortalFlowCLContext g_gpuPF = {};
GPUFlowFixed g_gpuFF = {};
GPUHybridVisContext g_gpuHybridVis = {};
GPULeafHybridVisContext g_gpuLeafHybridVis = {};

// =============================================================================
// GLOBAL WINDING POOL (8 million float3 = ~96 MB GPU)
// =============================================================================
static std::vector<int> g_initSrcOffsetCPU;
static std::vector<int> g_initSrcCountCPU;
static std::vector<float3> g_initWindingCPU;

static std::vector<std::vector<int>> g_leafPortals;
static int g_maxPerLeaf = 256;

// ============================================================================
// PresetGPU 3 — World solid triangles (CPU)
// ============================================================================
static std::vector<WorldTriCPU> g_worldTrisCPU;


inline uint64_t MakeLeafKey(int a, int b)
{
	if (a > b) std::swap(a, b);
	return ((uint64_t)a << 32) | (uint64_t)b;
}

static cl_mem UploadTempWinding(winding_t* w)
{
	if (!w || w->numpoints <= 0)
		return nullptr;

	std::vector<float3> pts;
	pts.resize(w->numpoints);

	for (int i = 0; i < w->numpoints; i++)
	{
		pts[i].x = w->points[i].x;
		pts[i].y = w->points[i].y;
		pts[i].z = w->points[i].z;
		pts[i].w = 0.0f;
	}

	cl_int err = CL_SUCCESS;
	cl_mem buf = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float3) * pts.size(),
		pts.data(),
		&err
	);

	if (err != CL_SUCCESS)
		return nullptr;

	return buf;
}

// ============================================================================
// PresetGPU 3 — CPU BVH construction (temporary)
// ============================================================================
struct BVHBuildNode
{
	Vector mins;
	Vector maxs;

	int left = -1;
	int right = -1;

	int firstTri = 0;
	int triCount = 0;
};

bool InitOpenCL_PortalFlow()
{
	if (g_gpuPF.initialized)
		return true;

	cl_int err = 0;

	// ===========================================================
	// PLATFORM
	// ===========================================================
	cl_uint platformCount = 0;
	err = clGetPlatformIDs(0, nullptr, &platformCount);
	if (err != CL_SUCCESS || platformCount == 0)
	{
		Warning("[GPU-VIS] No OpenCL platform found.\n");
		return false;
	}

	std::vector<cl_platform_id> platforms(platformCount);
	clGetPlatformIDs(platformCount, platforms.data(), nullptr);
	g_gpuPF.platform = platforms[0];

	// ===========================================================
	// DEVICE (GPU then fallback CPU)
	// ===========================================================
	cl_uint deviceCount = 0;
	err = clGetDeviceIDs(g_gpuPF.platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);

	if (err != CL_SUCCESS || deviceCount == 0)
	{
		Warning("[GPU-VIS] No GPU found, trying CPU...\n");
		err = clGetDeviceIDs(g_gpuPF.platform, CL_DEVICE_TYPE_CPU, 0, nullptr, &deviceCount);
		if (err != CL_SUCCESS || deviceCount == 0)
		{
			Warning("[GPU-VIS] No usable OpenCL device.\n");
			return false;
		}
	}

	std::vector<cl_device_id> devices(deviceCount);
	clGetDeviceIDs(g_gpuPF.platform, CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);
	g_gpuPF.device = devices[0];

	// ===========================================================
	// CONTEXT
	// ===========================================================
	g_gpuPF.context = clCreateContext(nullptr, 1, &g_gpuPF.device, nullptr, nullptr, &err);
	if (err != CL_SUCCESS)
	{
		Warning("[GPU-VIS] Failed to create context.\n");
		return false;
	}

	// ===========================================================
	// COMMAND QUEUE
	// ===========================================================
#if defined(CL_VERSION_2_0)
	const cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, 0, 0 };
	g_gpuPF.queue = clCreateCommandQueueWithProperties(
		g_gpuPF.context, g_gpuPF.device, props, &err);
#else
	g_gpuPF.queue = clCreateCommandQueue(g_gpuPF.context, g_gpuPF.device, 0, &err);
#endif

	if (err != CL_SUCCESS)
	{
		Warning("[GPU-VIS] Failed to create queue.\n");
		return false;
	}

	// ===========================================================
	// BUILD PROGRAM
	// ===========================================================
	const char* src = g_gpuPortalFlowKernels;
	size_t srcSize = strlen(g_gpuPortalFlowKernels);

	g_gpuPF.program = clCreateProgramWithSource(
		g_gpuPF.context,
		1,
		&src,
		&srcSize,
		&err
	);
	if (err != CL_SUCCESS)
	{
		Warning("[GPU-VIS] Failed program creation.\n");
		return false;
	}

	const char* opts = "-cl-fast-relaxed-math -cl-std=CL2.0";
	err = clBuildProgram(g_gpuPF.program, 1, &g_gpuPF.device, opts, nullptr, nullptr);

	{
		size_t logSize = 0;
		clGetProgramBuildInfo(g_gpuPF.program, g_gpuPF.device,
			CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

		if (logSize > 1)
		{
			std::vector<char> log(logSize);
			clGetProgramBuildInfo(g_gpuPF.program, g_gpuPF.device,
				CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
			Msg("[GPU-VIS] Build log:\n%s\n", log.data());
		}
	}

	if (err != CL_SUCCESS)
	{
		Warning("[GPU-VIS] Failed to build GPU kernels.\n");
		return false;
	}

	cl_int kerr = 0;

	// ===========================================================
	// LOAD ALL KERNELS A1 -> A6
	// ===========================================================

	

	g_gpuPF.k_ultraWorldOcc = clCreateKernel(g_gpuPF.program, "ultra_worldOcclusion", &kerr);
	if (kerr != CL_SUCCESS)
	{
		Warning("[GPU-VIS] Failed to create ultra_worldOcclusion kernel\n");
		return false;
	}

	g_gpuPF.k_rayTriangleBVH = clCreateKernel(g_gpuPF.program, "rayTriangleBVH", &kerr);
	if (kerr != CL_SUCCESS)
	{
		Warning("[GPU-VIS] Failed to create rayTriangleBVH kernel\n");
		return false;
	}

	g_gpuPF.k_hybridFilter = clCreateKernel(g_gpuPF.program, "Hybrid_MightSee_Filter", &kerr);
	if (kerr != CL_SUCCESS)
	{
		Warning("[GPU-VIS] Failed to create Hybrid_MightSee_Filter kernel\n");
		return false;
	}

	g_gpuPF.k_leafHybridFilter = clCreateKernel(g_gpuPF.program, "Leaf_MightSee_Filter", &kerr);
	if (kerr != CL_SUCCESS)
	{
		Warning("[LEAF-HYBRID] Failed to create Leaf_MightSee_Filter kernel\n");
		return false;
	}

	g_gpuPF.k_separatorReject = clCreateKernel(g_gpuPF.program, "SeparatorReject_MultiRay", &kerr);
	if (kerr != CL_SUCCESS)
	{
		Warning("[GPU-VIS] Failed to create SeparatorReject_MultiRay kernel\n");
		return false;
	}

	g_gpuPF.initialized = true;

	Msg("[GPU-VIS] OpenCL PortalFlow READY (all kernels loaded).\n");
	return true;
}


// ============================================================================
// HYBRID VIS — INIT
// ============================================================================

bool InitGPUHybridVis()
{
	if (!InitOpenCL_PortalFlow())
		return false;

	if (!g_gpuPF.d_worldTris || !g_gpuPF.d_worldBVH)
	{
		Warning("[HYBRID-VIS] World data missing, hybrid disabled\n");
		g_gpuHybridVis.enabled = false;
		return false;
	}

	g_gpuHybridVis.enabled = true;

	g_gpuHybridVis.filterJob.portalCount = g_numportals * 2;
	g_gpuHybridVis.filterJob.portalLongs = portallongs;

	int portalCount = g_gpuHybridVis.filterJob.portalCount;
	int longs = g_gpuHybridVis.filterJob.portalLongs;

	size_t maskBytes = portalCount * longs * sizeof(visword_t);

	// Allocate GPU might-see result
	g_gpuHybridVis.filterJob.result.portalCount = portalCount;
	g_gpuHybridVis.filterJob.result.portalLongs = longs;

	cl_int err = 0;
	g_gpuHybridVis.filterJob.result.d_mightSeeMask =
		clCreateBuffer(
			g_gpuPF.context,
			CL_MEM_READ_WRITE,
			maskBytes,
			nullptr,
			&err
		);

	if (err != CL_SUCCESS)
	{
		Warning("[HYBRID-VIS] Failed to allocate GPU mightSee mask\n");
		return false;
	}

	// Init mask to ZERO
	int zero = 0;
	clEnqueueFillBuffer(
		g_gpuPF.queue,
		g_gpuHybridVis.filterJob.result.d_mightSeeMask,
		&zero,
		sizeof(int),
		0,
		maskBytes,
		0, nullptr, nullptr
	);

	clFinish(g_gpuPF.queue);

	// Allocate CPU mirror
	g_gpuHybridVis.cpuMightSeeMask.resize(portalCount * longs);

	Msg("[HYBRID-VIS] GPU Hybrid VIS initialized (%d portals)\n", portalCount);
	return true;
}

bool InitGPULeafHybridVis()
{
	if (!InitOpenCL_PortalFlow())
		return false;

	g_gpuLeafHybridVis.enabled = true;

	g_gpuLeafHybridVis.leafCount = portalclusters;
	g_gpuLeafHybridVis.leafLongs = (portalclusters + 31) >> 5;

	int leafCount = g_gpuLeafHybridVis.leafCount;
	int longs = g_gpuLeafHybridVis.leafLongs;

	size_t bytes = leafCount * longs * sizeof(visword_t);

	cl_int err = 0;
	g_gpuLeafHybridVis.result.d_leafMightSee =
		clCreateBuffer(
			g_gpuPF.context,
			CL_MEM_READ_WRITE,
			bytes,
			nullptr,
			&err
		);

	if (err != CL_SUCCESS)
	{
		Warning("[LEAF-HYBRID] Failed to allocate leafMightSee buffer\n");
		return false;
	}

	int zero = 0;
	clEnqueueFillBuffer(
		g_gpuPF.queue,
		g_gpuLeafHybridVis.result.d_leafMightSee,
		&zero,
		sizeof(int),
		0,
		bytes,
		0, nullptr, nullptr
	);

	clFinish(g_gpuPF.queue);

	g_gpuLeafHybridVis.result.cpuLeafMightSee.resize(leafCount * longs);

	Msg("[LEAF-HYBRID] Initialized GPU leaf hybrid vis (%d leaves)\n", leafCount);
	return true;
}


void ShutdownOpenCL_PortalFlow()
{
	if (!g_gpuPF.initialized)
		return;

#define REL(x) if(x){ clReleaseMemObject(x); x=nullptr; }

	REL(g_gpuPF.d_planes);
	REL(g_gpuPF.d_portalVis);
	REL(g_gpuPF.d_origins);
	REL(g_gpuPF.d_radius);
	REL(g_gpuPF.d_winding4);
	REL(g_gpuPF.d_portalLeaf);
	REL(g_gpuPF.d_leafPortalCount);
	REL(g_gpuPF.d_leafPortalList);
	REL(g_gpuFF.d_stateCur);
	REL(g_gpuFF.d_stateNext);
	REL(g_gpuFF.d_stateCount);
	REL(g_gpuFF.d_stateNextCount);
	REL(g_gpuFF.d_mightSee);
	REL(d_windingPoints);
	REL(d_windingOffsets);
	REL(d_windingCounts);
	REL(g_gpuPF.d_worldAABBs);
	REL(g_gpuPF.d_worldBVH);
	REL(g_gpuPF.d_portalArea);
	REL(g_gpuPF.d_areaportals);
	REL(g_gpuPF.d_worldTris);

#undef REL

	if (g_gpuPF.program) clReleaseProgram(g_gpuPF.program);
	if (g_gpuPF.queue)   clReleaseCommandQueue(g_gpuPF.queue);
	if (g_gpuPF.context) clReleaseContext(g_gpuPF.context);


	g_gpuPF = {};
	Msg("[GPU-VIS] OpenCL shutdown complete.\n");

}



int g_TraceClusterStart = -1;
int g_TraceClusterStop = -1;
/*

  each portal will have a list of all possible to see from first portal

  if (!thread->portalmightsee[portalnum])

  portal mightsee

  for p2 = all other portals in leaf
	get sperating planes
	for all portals that might be seen by p2
		mark as unseen if not present in seperating plane
	flood fill a new mightsee
	save as passagemightsee


  void CalcMightSee (leaf_t *leaf,
*/


int CountBits(byte* bits, int numbits)
{
	int		i;
	int		c;

	c = 0;
	for (i = 0; i < numbits; i++)
		if (CheckBit(bits, i))
			c++;

	return c;
}

int		c_fullskip;
int		c_portalskip, c_leafskip;
int		c_vistest, c_mighttest;

int		c_chop, c_nochop;

int		active;



void CheckStack(leaf_t* leaf, threaddata_t* thread)
{
	pstack_t* p, * p2;

	for (p = thread->pstack_head.next; p; p = p->next)
	{
		//		Msg ("=");
		if (p->leaf == leaf)
			Error("CheckStack: leaf recursion");
		for (p2 = thread->pstack_head.next; p2 != p; p2 = p2->next)
			if (p2->leaf == p->leaf)
				Error("CheckStack: late leaf recursion");
	}
	//	Msg ("\n");
}


winding_t* AllocStackWinding(pstack_t* stack)
{
	int		i;

	for (i = 0; i < 3; i++)
	{
		if (stack->freewindings[i])
		{
			stack->freewindings[i] = 0;
			return &stack->windings[i];
		}
	}

	Error("Out of memory. AllocStackWinding: failed");

	return NULL;
}

void FreeStackWinding(winding_t* w, pstack_t* stack)
{
	int		i;

	i = w - stack->windings;

	if (i < 0 || i>2)
		return;		// not from local

	if (stack->freewindings[i])
		Error("FreeStackWinding: allready free");
	stack->freewindings[i] = 1;
}

/*
==============
ChopWinding

==============
*/

#ifdef _WIN32
#pragma warning (disable:4701)
#endif

winding_t* ChopWinding(winding_t* in, pstack_t* stack, plane_t* split)
{
	vec_t	dists[128];
	int		sides[128];
	int		counts[3];
	vec_t	dot;
	int		i, j;
	Vector	mid;
	winding_t* neww;

	counts[0] = counts[1] = counts[2] = 0;

	// determine sides for each point
	for (i = 0; i < in->numpoints; i++)
	{
		dot = DotProduct(in->points[i], split->normal);
		dot -= split->dist;
		dists[i] = dot;
		if (dot > ON_VIS_EPSILON)
			sides[i] = SIDE_FRONT;
		else if (dot < -ON_VIS_EPSILON)
			sides[i] = SIDE_BACK;
		else
		{
			sides[i] = SIDE_ON;
		}
		counts[sides[i]]++;
	}

	if (!counts[1])
		return in;		// completely on front side

	if (!counts[0])
	{
		FreeStackWinding(in, stack);
		return NULL;
	}

	sides[i] = sides[0];
	dists[i] = dists[0];

	neww = AllocStackWinding(stack);

	neww->numpoints = 0;

	for (i = 0; i < in->numpoints; i++)
	{
		Vector& p1 = in->points[i];

		if (neww->numpoints == MAX_POINTS_ON_FIXED_WINDING)
		{
			FreeStackWinding(neww, stack);
			return in;		// can't chop -- fall back to original
		}

		if (sides[i] == SIDE_ON)
		{
			VectorCopy(p1, neww->points[neww->numpoints]);
			neww->numpoints++;
			continue;
		}

		if (sides[i] == SIDE_FRONT)
		{
			VectorCopy(p1, neww->points[neww->numpoints]);
			neww->numpoints++;
		}

		if (sides[i + 1] == SIDE_ON || sides[i + 1] == sides[i])
			continue;

		if (neww->numpoints == MAX_POINTS_ON_FIXED_WINDING)
		{
			FreeStackWinding(neww, stack);
			return in;		// can't chop -- fall back to original
		}

		// generate a split point
		Vector& p2 = in->points[(i + 1) % in->numpoints];

		dot = dists[i] / (dists[i] - dists[i + 1]);
		for (j = 0; j < 3; j++)
		{	// avoid round off error when possible
			if (split->normal[j] == 1)
				mid[j] = split->dist;
			else if (split->normal[j] == -1)
				mid[j] = -split->dist;
			else
				mid[j] = p1[j] + dot * (p2[j] - p1[j]);
		}

		VectorCopy(mid, neww->points[neww->numpoints]);
		neww->numpoints++;
	}

	// free the original winding
	FreeStackWinding(in, stack);

	return neww;
}

#ifdef _WIN32
#pragma warning (default:4701)
#endif

/*
==============
ClipToSeperators

Source, pass, and target are an ordering of portals.

Generates seperating planes canidates by taking two points from source and one
point from pass, and clips target by them.

If target is totally clipped away, that portal can not be seen through.

Normal clip keeps target on the same side as pass, which is correct if the
order goes source, pass, target.  If the order goes pass, source, target then
flipclip should be set.
==============
*/
winding_t* ClipToSeperators(winding_t* source, winding_t* pass, winding_t* target, bool flipclip, pstack_t* stack)
{
	int			i, j, k, l;
	plane_t		plane;
	Vector		v1, v2;
	float		d;
	vec_t		length;
	int			counts[3];
	bool		fliptest;

	// check all combinations	
	for (i = 0; i < source->numpoints; i++)
	{
		l = (i + 1) % source->numpoints;
		VectorSubtract(source->points[l], source->points[i], v1);

		// fing a vertex of pass that makes a plane that puts all of the
		// vertexes of pass on the front side and all of the vertexes of
		// source on the back side
		for (j = 0; j < pass->numpoints; j++)
		{
			VectorSubtract(pass->points[j], source->points[i], v2);

			plane.normal[0] = v1[1] * v2[2] - v1[2] * v2[1];
			plane.normal[1] = v1[2] * v2[0] - v1[0] * v2[2];
			plane.normal[2] = v1[0] * v2[1] - v1[1] * v2[0];

			// if points don't make a valid plane, skip it

			length = plane.normal[0] * plane.normal[0]
				+ plane.normal[1] * plane.normal[1]
				+ plane.normal[2] * plane.normal[2];

			if (length < ON_VIS_EPSILON)
				continue;

			length = 1 / sqrt(length);

			plane.normal[0] *= length;
			plane.normal[1] *= length;
			plane.normal[2] *= length;

			plane.dist = DotProduct(pass->points[j], plane.normal);

			//
			// find out which side of the generated seperating plane has the
			// source portal
			//
#if 1
			fliptest = false;
			for (k = 0; k < source->numpoints; k++)
			{
				if (k == i || k == l)
					continue;
				d = DotProduct(source->points[k], plane.normal) - plane.dist;
				if (d < -ON_VIS_EPSILON)
				{	// source is on the negative side, so we want all
					// pass and target on the positive side
					fliptest = false;
					break;
				}
				else if (d > ON_VIS_EPSILON)
				{	// source is on the positive side, so we want all
					// pass and target on the negative side
					fliptest = true;
					break;
				}
			}
			if (k == source->numpoints)
				continue;		// planar with source portal
#else
			fliptest = flipclip;
#endif
			//
			// flip the normal if the source portal is backwards
			//
			if (fliptest)
			{
				VectorSubtract(vec3_origin, plane.normal, plane.normal);
				plane.dist = -plane.dist;
			}
#if 1
			//
			// if all of the pass portal points are now on the positive side,
			// this is the seperating plane
			//
			counts[0] = counts[1] = counts[2] = 0;
			for (k = 0; k < pass->numpoints; k++)
			{
				if (k == j)
					continue;
				d = DotProduct(pass->points[k], plane.normal) - plane.dist;
				if (d < -ON_VIS_EPSILON)
					break;
				else if (d > ON_VIS_EPSILON)
					counts[0]++;
				else
					counts[2]++;
			}
			if (k != pass->numpoints)
				continue;	// points on negative side, not a seperating plane

			if (!counts[0])
				continue;	// planar with seperating plane
#else
			k = (j + 1) % pass->numpoints;
			d = DotProduct(pass->points[k], plane.normal) - plane.dist;
			if (d < -ON_VIS_EPSILON)
				continue;
			k = (j + pass->numpoints - 1) % pass->numpoints;
			d = DotProduct(pass->points[k], plane.normal) - plane.dist;
			if (d < -ON_VIS_EPSILON)
				continue;
#endif
			//
			// flip the normal if we want the back side
			//
			if (flipclip)
			{
				VectorSubtract(vec3_origin, plane.normal, plane.normal);
				plane.dist = -plane.dist;
			}

			//
			// clip target by the seperating plane
			//
			target = ChopWinding(target, stack, &plane);
			if (!target)
				return NULL;		// target is not visible

			// JAY: End the loop, no need to find additional separators on this edge ?
//			j = pass->numpoints;
		}
	}

	return target;
}


class CPortalTrace
{
public:
	CUtlVector<Vector>	m_list;
	CThreadFastMutex	m_mutex;
} g_PortalTrace;

void WindingCenter(winding_t* w, Vector& center)
{
	int		i;
	float	scale;

	VectorCopy(vec3_origin, center);
	for (i = 0; i < w->numpoints; i++)
		VectorAdd(w->points[i], center, center);

	scale = 1.0 / w->numpoints;
	VectorScale(center, scale, center);
}

Vector ClusterCenter(int cluster)
{
	Vector mins, maxs;
	ClearBounds(mins, maxs);
	int count = leafs[cluster].portals.Count();
	for (int i = 0; i < count; i++)
	{
		winding_t* w = leafs[cluster].portals[i]->winding;
		for (int j = 0; j < w->numpoints; j++)
		{
			AddPointToBounds(w->points[j], mins, maxs);
		}
	}
	return (mins + maxs) * 0.5f;
}


void DumpPortalTrace(pstack_t* pStack)
{
	AUTO_LOCK(g_PortalTrace.m_mutex);
	if (g_PortalTrace.m_list.Count())
		return;

	Warning("Dumped cluster trace!!!\n");
	Vector	mid;
	mid = ClusterCenter(g_TraceClusterStart);
	g_PortalTrace.m_list.AddToTail(mid);
	for (; pStack != NULL; pStack = pStack->next)
	{
		winding_t* w = pStack->pass ? pStack->pass : pStack->portal->winding;
		WindingCenter(w, mid);
		g_PortalTrace.m_list.AddToTail(mid);
		for (int i = 0; i < w->numpoints; i++)
		{
			g_PortalTrace.m_list.AddToTail(w->points[i]);
			g_PortalTrace.m_list.AddToTail(mid);
		}
		for (int i = 0; i < w->numpoints; i++)
		{
			g_PortalTrace.m_list.AddToTail(w->points[i]);
		}
		g_PortalTrace.m_list.AddToTail(w->points[0]);
		g_PortalTrace.m_list.AddToTail(mid);
	}
	mid = ClusterCenter(g_TraceClusterStop);
	g_PortalTrace.m_list.AddToTail(mid);
}

void WritePortalTrace(const char* source)
{
	Vector	mid;
	FILE* linefile;
	char	filename[1024];

	if (!g_PortalTrace.m_list.Count())
	{
		Warning("No trace generated from %d to %d\n", g_TraceClusterStart, g_TraceClusterStop);
		return;
	}

	sprintf(filename, "%s.lin", source);
	linefile = fopen(filename, "w");
	if (!linefile)
		Error("Couldn't open %s\n", filename);

	for (int i = 0; i < g_PortalTrace.m_list.Count(); i++)
	{
		Vector p = g_PortalTrace.m_list[i];
		fprintf(linefile, "%f %f %f\n", p[0], p[1], p[2]);
	}
	fclose(linefile);
	Warning("Wrote %s!!!\n", filename);
}

static std::mutex g_gpuSeparatorCacheMutex;


/*
==================
RecursiveLeafFlow

Flood fill through the leafs
If src_portal is NULL, this is the originating leaf
==================
*/

void RecursiveLeafFlow_CPU(int leafnum, threaddata_t* thread, pstack_t* prevstack)
{
	pstack_t	stack;
	portal_t* p;
	plane_t		backplane;
	leaf_t* leaf;
	int			i, j;
	long* test, * might, * vis, more;
	int			pnum;


	if (leafnum == g_TraceClusterStop)
	{
		DumpPortalTrace(&thread->pstack_head);
		return;
	}
	thread->c_chains++;

	leaf = &leafs[leafnum];

	prevstack->next = &stack;

	stack.next = NULL;
	stack.leaf = leaf;
	stack.portal = NULL;

	might = (long*)stack.mightsee;
	vis = (long*)thread->base->portalvis;

	// check all portals for flowing into other leafs	
	for (i = 0; i < leaf->portals.Count(); i++)
	{

		p = leaf->portals[i];
		pnum = p - portals;

		if (!(prevstack->mightsee[pnum >> 3] & (1 << (pnum & 7))))
		{
			continue;	// can't possibly see it
		}

		// if the portal can't see anything we haven't allready seen, skip it
		if (p->status == stat_done)
		{
			test = (long*)p->portalvis;
		}
		else
		{
			test = (long*)p->portalflood;
		}

		more = 0;
		for (j = 0; j < portallongs; j++)
		{
			might[j] = ((long*)prevstack->mightsee)[j] & test[j];
			more |= (might[j] & ~vis[j]);
		}

		if (!more && CheckBit(thread->base->portalvis, pnum))
		{	// can't see anything new
			continue;
		}

		// get plane of portal, point normal into the neighbor leaf
		stack.portalplane = p->plane;
		VectorSubtract(vec3_origin, p->plane.normal, backplane.normal);
		backplane.dist = -p->plane.dist;

		stack.portal = p;
		stack.next = NULL;
		stack.freewindings[0] = 1;
		stack.freewindings[1] = 1;
		stack.freewindings[2] = 1;

		float d = DotProduct(p->origin, thread->pstack_head.portalplane.normal);
		d -= thread->pstack_head.portalplane.dist;
		if (d < -p->radius)
		{
			continue;
		}
		else if (d > p->radius)
		{
			stack.pass = p->winding;
		}
		else
		{
			stack.pass = ChopWinding(p->winding, &stack, &thread->pstack_head.portalplane);
			if (!stack.pass)
				continue;
		}


		d = DotProduct(thread->base->origin, p->plane.normal);
		d -= p->plane.dist;
		if (d > thread->base->radius)
		{
			continue;
		}
		else if (d < -thread->base->radius)
		{
			stack.source = prevstack->source;
		}
		else
		{
			stack.source = ChopWinding(prevstack->source, &stack, &backplane);
			if (!stack.source)
				continue;
		}

		if (!prevstack->pass)
		{	// the second leaf can only be blocked if coplanar

			// mark the portal as visible
			SetBit(thread->base->portalvis, pnum);

			RecursiveLeafFlow_CPU(p->leaf, thread, &stack);
			continue;
		}

		// ============================================
		// GPU EARLY REJECT — exact & safe
		// ============================================
		if (
			g_gpuPF.initialized &&
			g_gpuPreset >= 3 &&
			prevstack == &thread->pstack_head
			)
		{
			if (!GPU_CanPassSeparators(
				stack.source,
				prevstack->pass,
				stack.pass,
				p->leaf,
				int(prevstack->leaf - leafs)
			))
			{
				continue;
			}
		}


		stack.pass = ClipToSeperators(stack.source, prevstack->pass, stack.pass, false, &stack);
		if (!stack.pass)
			continue;

		stack.pass = ClipToSeperators(prevstack->pass, stack.source, stack.pass, true, &stack);
		if (!stack.pass)
			continue;

		// mark the portal as visible
		SetBit(thread->base->portalvis, pnum);

		// flow through it for real
		RecursiveLeafFlow_CPU(p->leaf, thread, &stack);
	}
}

bool GPU_CanPassSeparators(
	winding_t* source,
	winding_t* pass,
	winding_t* target,
	int leafA,
	int leafB
)
{
	// ================= FAIL SAFE =================
	if (!g_gpuPF.initialized)
		return true;

	if (!g_gpuPF.d_worldTris || !g_gpuPF.d_worldBVH)
		return true;

	// Trop petit → CPU déjà exact
	if (source->numpoints < 3 || target->numpoints < 3)
		return true;

	// Protection runaway kernel
	if (source->numpoints * target->numpoints > 256)
		return true;

	if (!g_gpuPF.k_separatorReject)
		return true;


	if (!source || !target || source->numpoints <= 0 || target->numpoints <= 0)
		return true;
	// ============================================


	// ---------------- CACHE ----------------
	const uint64_t key =
		((uint64_t)leafA << 48) ^
		((uint64_t)leafB << 32) ^
		((uint64_t)source->numpoints << 16) ^
		(uint64_t)target->numpoints;
	{
		std::lock_guard<std::mutex> lock(g_gpuSeparatorCacheMutex);
		auto it = g_gpuSeparatorCache.find(key);
		if (it != g_gpuSeparatorCache.end())
			return it->second;
	}
	// ---------------------------------------

	cl_mem d_src = UploadTempWinding(source);
	cl_mem d_tgt = UploadTempWinding(target);

	if (!d_src || !d_tgt)
	{
		if (d_src) clReleaseMemObject(d_src);
		if (d_tgt) clReleaseMemObject(d_tgt);
		return true;
	}

	int srcCount = source->numpoints;
	int tgtCount = target->numpoints;

	cl_int err = CL_SUCCESS;
	int zero = 0;
	cl_mem d_result = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(int),
		&zero,
		&err
	);

	if (err != CL_SUCCESS || !d_result)
	{
		clReleaseMemObject(d_src);
		clReleaseMemObject(d_tgt);
		return true;
	}

	size_t global = (size_t)srcCount * (size_t)tgtCount;

	err = clSetKernelArg(g_gpuPF.k_separatorReject, 0, sizeof(cl_mem), &d_src);
	err |= clSetKernelArg(g_gpuPF.k_separatorReject, 1, sizeof(int), &srcCount);
	err |= clSetKernelArg(g_gpuPF.k_separatorReject, 2, sizeof(cl_mem), &d_tgt);
	err |= clSetKernelArg(g_gpuPF.k_separatorReject, 3, sizeof(int), &tgtCount);
	err |= clSetKernelArg(g_gpuPF.k_separatorReject, 4, sizeof(cl_mem), &g_gpuPF.d_worldTris);
	err |= clSetKernelArg(g_gpuPF.k_separatorReject, 5, sizeof(cl_mem), &g_gpuPF.d_worldBVH);
	err |= clSetKernelArg(g_gpuPF.k_separatorReject, 6, sizeof(int), &g_gpuPF.worldTriCount);
	err |= clSetKernelArg(g_gpuPF.k_separatorReject, 7, sizeof(int), &g_gpuPF.worldBVHCount);
	err |= clSetKernelArg(g_gpuPF.k_separatorReject, 8, sizeof(cl_mem), &d_result);

	if (err == CL_SUCCESS)
	{
		err = clEnqueueNDRangeKernel(
			g_gpuPF.queue,
			g_gpuPF.k_separatorReject,
			1, nullptr, &global, nullptr,
			0, nullptr, nullptr
		);
		clFinish(g_gpuPF.queue);
	}

	int result = 1; // fallback = POSSIBLE
	if (err == CL_SUCCESS)
	{
		clEnqueueReadBuffer(
			g_gpuPF.queue,
			d_result,
			CL_TRUE,
			0,
			sizeof(int),
			&result,
			0, nullptr, nullptr
		);
	}

	clReleaseMemObject(d_src);
	clReleaseMemObject(d_tgt);
	clReleaseMemObject(d_result);

	bool canPass = (result != 0);

	{
		std::lock_guard<std::mutex> lock(g_gpuSeparatorCacheMutex);
		g_gpuSeparatorCache[key] = canPass;
	}

	return canPass;
}


void BuildFlatLeafPortalArrays(std::vector<int>& outCount, std::vector<int>& outList);

/*
// --------------------
// PortalFlow + All fonctions for GPU
// --------------------
//
*/

// ============================================================================
// PresetGPU 3 — Extract solid world triangles from BSP (CPU)
// ============================================================================
static void ExtractWorldSolidTriangles()
{
	g_worldTrisCPU.clear();

	for (int i = 0; i < numfaces; i++)
	{
		const dface_t& face = dfaces[i];

		// Skip faces with no edges
		if (face.numedges < 3)
			continue;

		// Texture info
		const texinfo_t& ti = texinfo[face.texinfo];

		// Skip sky / tools / non-solid
		if (ti.flags & (SURF_SKY | SURF_NODRAW))
			continue;

		// Get plane
		const dplane_t& plane = dplanes[face.planenum];

		// Skip non-solid planes (parano)
		if (!(plane.type >= 0))
			continue;

		// Collect vertices of the face
		std::vector<Vector> verts;
		verts.reserve(face.numedges);

		for (int e = 0; e < face.numedges; e++)
		{
			int surfEdge = dsurfedges[face.firstedge + e];
			int edgeIdx = abs(surfEdge);
			const dedge_t& edge = dedges[edgeIdx];

			int vertIdx = (surfEdge >= 0) ? edge.v[0] : edge.v[1];
			const Vector& v = *(Vector*)&dvertexes[vertIdx];

			verts.push_back(v);
		}

		if (verts.size() < 3)
			continue;

		// Triangulation fan: (v0, v[i], v[i+1])
		const Vector& v0 = verts[0];
		for (size_t t = 1; t + 1 < verts.size(); t++)
		{
			WorldTriCPU tri;
			tri.a = v0;
			tri.b = verts[t];
			tri.c = verts[t + 1];

			g_worldTrisCPU.push_back(tri);
		}
	}

	Msg("[GPU-VIS][PRESET3] Extracted %zu world solid triangles\n",
		g_worldTrisCPU.size());
}

static void ComputeTriAABB(const WorldTriCPU& t, Vector& mins, Vector& maxs)
{
	mins = t.a;
	maxs = t.a;

	AddPointToBounds(t.b, mins, maxs);
	AddPointToBounds(t.c, mins, maxs);
}

static float SurfaceArea(const Vector& mins, const Vector& maxs)
{
	Vector e = maxs - mins;
	return 2.0f * (e.x * e.y + e.x * e.z + e.y * e.z);
}

static std::vector<BVHBuildNode> g_bvhBuildNodes;
static std::vector<int> g_bvhTriIndices;

static int BuildBVHNode(int start, int count)
{
	BVHBuildNode node;
	Vector mins(1e30f, 1e30f, 1e30f);
	Vector maxs(-1e30f, -1e30f, -1e30f);

	for (int i = 0; i < count; i++)
	{
		Vector tmin, tmax;
		ComputeTriAABB(g_worldTrisCPU[g_bvhTriIndices[start + i]], tmin, tmax);
		AddPointToBounds(tmin, mins, maxs);
		AddPointToBounds(tmax, mins, maxs);
	}

	node.mins = mins;
	node.maxs = maxs;
	node.firstTri = start;
	node.triCount = count;

	int nodeIndex = (int)g_bvhBuildNodes.size();
	g_bvhBuildNodes.push_back(node);

	// Leaf condition
	if (count <= 4)
		return nodeIndex;

	// Choose split axis (largest extent)
	Vector ext = maxs - mins;
	int axis = (ext.x > ext.y && ext.x > ext.z) ? 0 :
		(ext.y > ext.z) ? 1 : 2;

	float splitPos = 0.5f * (mins[axis] + maxs[axis]);

	int mid = start;
	for (int i = start; i < start + count; i++)
	{
		const WorldTriCPU& t = g_worldTrisCPU[g_bvhTriIndices[i]];
		float c = (t.a[axis] + t.b[axis] + t.c[axis]) / 3.0f;
		if (c < splitPos)
			std::swap(g_bvhTriIndices[i], g_bvhTriIndices[mid++]);
	}

	int leftCount = mid - start;
	int rightCount = count - leftCount;

	// Fallback if split failed
	if (leftCount == 0 || rightCount == 0)
		return nodeIndex;

	node.left = BuildBVHNode(start, leftCount);
	node.right = BuildBVHNode(mid, rightCount);

	g_bvhBuildNodes[nodeIndex] = node;
	return nodeIndex;
}

bool AllocatePortalFlowBuffers()
{
	if (!g_gpuPF.initialized)
		return false;

	cl_int err = 0;
	int zero = 0;

	// ---------------------------------------------------------
	// PORTAL PARAMS
	// ---------------------------------------------------------
	g_gpuPF.portalCount = g_numportals * 2;
	g_gpuPF.portalLongs = portallongs;
	g_gpuFF.startPortal = 0;
	const int portalCount = g_gpuPF.portalCount;
	const int longs = g_gpuPF.portalLongs;

	const size_t maskBytes = portalCount * longs * sizeof(int);
	const size_t originBytes = portalCount * sizeof(float3);
	const size_t radiusBytes = portalCount * sizeof(float);
	const size_t planeBytes = portalCount * sizeof(float4);


	// ---------------------------------------------------------
	// RELEASE PREVIOUS BUFFERS
	// ---------------------------------------------------------
#define REL(x) if(x){ clReleaseMemObject(x); x=nullptr; }

	REL(g_gpuPF.d_portalVis);
	REL(g_gpuPF.d_origins);
	REL(g_gpuPF.d_radius);
	REL(g_gpuPF.d_planes);
	REL(g_gpuPF.d_winding4);
	REL(g_gpuPF.d_portalLeaf);
	REL(g_gpuPF.d_leafPortalCount);
	REL(g_gpuPF.d_leafPortalList);

	REL(g_gpuFF.d_stateCur);
	REL(g_gpuFF.d_stateNext);
	REL(g_gpuFF.d_stateCount);
	REL(g_gpuFF.d_stateNextCount);
	REL(g_gpuFF.d_mightSee);

	REL(g_gpuPF.d_windPool);
	REL(g_gpuPF.d_windPoolCount);
	REL(g_gpuPF.d_initSrcOffset);
	REL(g_gpuPF.d_initSrcCount);

	REL(g_gpuPF.d_ultraRejectMask);

#undef REL

	// =========================================================
	// ULTRA REJECT MASK (PresetGPU 2)
	// =========================================================


	g_gpuPF.d_ultraRejectMask = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_WRITE,
		maskBytes,
		nullptr,
		&err
	);
	if (err != CL_SUCCESS)
	{
		Warning("[GPU-VIS][ULTRA] Failed to create ultraRejectMask\n");
		return false;
	}

	clEnqueueFillBuffer(
		g_gpuPF.queue,
		g_gpuPF.d_ultraRejectMask,
		&zero,
		sizeof(int),
		0,
		maskBytes,
		0, nullptr, nullptr
	);

	clFinish(g_gpuPF.queue);



	// CREATE GPU portalVis[] : START EMPTY (GPU BUILDS VIS)
	g_gpuPF.d_portalVis = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_WRITE,
		maskBytes,
		nullptr,
		&err
	);

	std::vector<int> init(portalCount * longs, 0);
	clEnqueueWriteBuffer(
		g_gpuPF.queue,
		g_gpuPF.d_portalVis,
		CL_TRUE,
		0,
		init.size() * sizeof(int),
		init.data(),
		0, nullptr, nullptr
	);

	// FORCE SELF-VISIBILITY
	for (int p = 0; p < portalCount; p++)
	{
		int byte = p >> 5;
		int bit = 1 << (p & 31);
		int word = 0;

		size_t off = p * longs * sizeof(int) + byte * sizeof(int);

		clEnqueueReadBuffer(
			g_gpuPF.queue, g_gpuPF.d_portalVis,
			CL_TRUE, off, sizeof(int), &word,
			0, nullptr, nullptr
		);

		word |= bit;

		clEnqueueWriteBuffer(
			g_gpuPF.queue, g_gpuPF.d_portalVis,
			CL_TRUE, off, sizeof(int), &word,
			0, nullptr, nullptr
		);
	}



	std::vector<float3> h_orig(portalCount);
	std::vector<float>  h_rad(portalCount);
	std::vector<float4> h_pl(portalCount);
	std::vector<float3> h_normals(portalCount);
	std::vector<int>	h_leaf(portalCount);
	std::vector<int>	h_area(portalCount, 0);


	// =========================================================
	// CORE BUFFERS : origins / radius / planes
	// =========================================================
	g_gpuPF.d_origins = clCreateBuffer(
		g_gpuPF.context, CL_MEM_READ_ONLY, originBytes, nullptr, &err);
	if (err) return false;

	g_gpuPF.d_portalNormals = clCreateBuffer(
		g_gpuPF.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, portalCount * sizeof(float3),	h_normals.data(),&err); 
	if (err) return false;

	g_gpuPF.d_radius = clCreateBuffer(
		g_gpuPF.context, CL_MEM_READ_ONLY, radiusBytes, nullptr, &err);
	if (err) return false;

	g_gpuPF.d_planes = clCreateBuffer(
		g_gpuPF.context, CL_MEM_READ_ONLY, planeBytes, nullptr, &err);
	if (err) return false;

	g_gpuPF.d_winding4 = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_ONLY,
		portalCount * 4 * sizeof(float3),
		nullptr,
		&err
	);

	for (int p = 0; p < portalCount; p++)
	{
		portal_t* P = sorted_portals[p];

		h_orig[p] = { P->origin.x, P->origin.y, P->origin.z };
		h_rad[p] = P->radius;
		h_pl[p] = { P->plane.normal.x, P->plane.normal.y, P->plane.normal.z, P->plane.dist };
		h_normals[p] = {P->plane.normal.x, P->plane.normal.y, P->plane.normal.z, 0.f};


		h_leaf[p] = P->leaf;

		// IMPORTANT: l’area est stockée sur le dleaf BSP correspondant au cluster/leaf du portal
		// (le portal “leaf” ici correspond à un portalcluster)
		int leafIdx = h_leaf[p];
		int area = 0;
		if (leafIdx >= 0 && leafIdx < numleafs)
			area = dleafs[leafIdx].area;   // dleaf_t::area

		h_area[p] = area;
	}

	// =========================================================
	// PORTAL -> AREA table (leaf -> area)
	// =========================================================

	for (int p = 0; p < portalCount; p++)
	{
		int leaf = h_leaf[p];
		if (leaf >= 0 && leaf < numleafs)
		{
			// dleaf_t has an 'area' in Source BSPs
			h_area[p] = dleafs[leaf].area;
		}
		else
		{
			h_area[p] = 0;
		}
	}


	// Nouveau buffer : portal -> area
	g_gpuPF.d_portalArea = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		portalCount * sizeof(int),
		h_area.data(),
		&err
	);
	if (err) return false;

	clEnqueueWriteBuffer(g_gpuPF.queue, g_gpuPF.d_origins, CL_TRUE,
		0, originBytes, h_orig.data(), 0, nullptr, nullptr);

	clEnqueueWriteBuffer(g_gpuPF.queue, g_gpuPF.d_radius, CL_TRUE,
		0, radiusBytes, h_rad.data(), 0, nullptr, nullptr);

	clEnqueueWriteBuffer(g_gpuPF.queue, g_gpuPF.d_planes, CL_TRUE,
		0, planeBytes, h_pl.data(), 0, nullptr, nullptr);


	// =========================================================
	// PORTAL -> LEAF table
	// =========================================================
	g_gpuPF.d_portalLeaf = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		portalCount * sizeof(int),
		h_leaf.data(),
		&err
	);

	if (err != CL_SUCCESS)
	{
		Warning("[GPU-VIS] ERROR creating d_portalLeaf buffer (err=%d)\n", err);
		return false;
	}

	Msg("[GPU-VIS] portalLeaf buffer uploaded (%d entries)\n", portalCount);

	// =========================================================
	// PORTAL -> AREA BUFFER
	// =========================================================
	g_gpuPF.d_portalArea = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		portalCount * sizeof(int),
		h_area.data(),
		&err
	);

	if (err != CL_SUCCESS)
	{
		Warning("[GPU-VIS] ERROR creating d_portalArea buffer (err=%d)\n", err);
		return false;
	}

	Msg("[GPU-VIS] portalArea buffer uploaded (%d entries)\n", portalCount);

	// =========================================================
	// DYNAMIC WINDINGS -> POOL GLOBAL
	// =========================================================
	g_initSrcOffsetCPU.resize(portalCount);
	g_initSrcCountCPU.resize(portalCount);
	g_initWindingCPU.clear();
	g_totalWindingPoints = 0;

	for (int p = 0; p < portalCount; p++)
	{
		winding_t* W = sorted_portals[p]->winding;
		int cnt = (W ? W->numpoints : 0);

		g_initSrcOffsetCPU[p] = g_totalWindingPoints;
		g_initSrcCountCPU[p] = cnt;

		for (int k = 0; k < cnt; k++)
		{
			Vector& v = W->points[k];
			g_initWindingCPU.push_back({ v.x, v.y, v.z });
		}

		g_totalWindingPoints += cnt;
	}

	g_gpuPF.d_initSrcOffset = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int) * portalCount,
		g_initSrcOffsetCPU.data(),
		&err
	);
	if (err) return false;

	g_gpuPF.d_initSrcCount = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int) * portalCount,
		g_initSrcCountCPU.data(),
		&err
	);
	if (err) return false;


	// ========= POOL GLOBAL =========
	g_gpuPF.d_windPool = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_WRITE,
		sizeof(float3) * GPU_WINDING_POOL_MAX,
		nullptr,
		&err
	);
	if (err) return false;

	// ===== POOL COUNTER (MANQUANT) =====
	g_gpuPF.d_windPoolCount = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_WRITE,
		sizeof(int),
		nullptr,
		&err
	);
	if (err) return false;

	// init pool counter
	int poolStart = g_totalWindingPoints;
	clEnqueueWriteBuffer(
		g_gpuPF.queue,
		g_gpuPF.d_windPoolCount,
		CL_TRUE,
		0,
		sizeof(int),
		&poolStart,
		0, nullptr, nullptr
	);


	if (g_totalWindingPoints > 0)
	{
		clEnqueueWriteBuffer(
			g_gpuPF.queue,
			g_gpuPF.d_windPool,
			CL_TRUE,
			0,
			g_totalWindingPoints * sizeof(float3),
			g_initWindingCPU.data(),
			0, nullptr, nullptr
		);
	}



	if (err) return false;

	// =========================================================
	// LEAF ADJ TABLE
	// =========================================================
	BuildLeafPortalTable();

	// =========================================================
	// PresetGPU 3 — Extract world solid triangles (CPU)
	// =========================================================
	if (g_gpuPreset >= 3)
	{
		ExtractWorldSolidTriangles();
	}

	// =========================================================
	// PresetGPU 3 — Upload world triangles to GPU
	// =========================================================
	if (g_gpuPreset >= 3 && !g_worldTrisCPU.empty())
	{
		std::vector<WorldTriGPU> h_tris;
		h_tris.reserve(g_worldTrisCPU.size());

		for (const WorldTriCPU& t : g_worldTrisCPU)
		{
			WorldTriGPU gt;
			gt.a = { t.a.x, t.a.y, t.a.z, 0.f };
			gt.b = { t.b.x, t.b.y, t.b.z, 0.f };
			gt.c = { t.c.x, t.c.y, t.c.z, 0.f };
			h_tris.push_back(gt);
		}

		g_gpuPF.worldTriCount = (int)h_tris.size();

		cl_int err2 = 0;
		g_gpuPF.d_worldTris = clCreateBuffer(
			g_gpuPF.context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			sizeof(WorldTriGPU) * g_gpuPF.worldTriCount,
			h_tris.data(),
			&err2
		);

		if (err2 != CL_SUCCESS)
		{
			Warning("[GPU-VIS][PRESET3] Failed to upload world triangles\n");
			return false;
		}

		Msg("[GPU-VIS][PRESET3] Uploaded %d world triangles to GPU\n",
			g_gpuPF.worldTriCount);
	}


	// =========================================================
	// PresetGPU 3 — Build BVH (CPU)
	// =========================================================
	if (g_gpuPreset >= 3 && !g_worldTrisCPU.empty())
	{
		g_bvhTriIndices.resize(g_worldTrisCPU.size());
		for (int i = 0; i < (int)g_worldTrisCPU.size(); i++)
			g_bvhTriIndices[i] = i;

		g_bvhBuildNodes.clear();
		BuildBVHNode(0, (int)g_worldTrisCPU.size());

		Msg("[GPU-VIS][PRESET3] Built BVH with %zu nodes\n",
			g_bvhBuildNodes.size());
	}

	// =========================================================
	// PresetGPU 3 — Upload BVH to GPU
	// =========================================================
	if (g_gpuPreset >= 3 && !g_bvhBuildNodes.empty())
	{
		std::vector<BVHNodeGPU> h_bvh;
		h_bvh.resize(g_bvhBuildNodes.size());

		for (size_t i = 0; i < g_bvhBuildNodes.size(); i++)
		{
			const BVHBuildNode& n = g_bvhBuildNodes[i];
			BVHNodeGPU& g = h_bvh[i];

			g.aabbMin = { n.mins.x, n.mins.y, n.mins.z, 0.f };
			g.aabbMax = { n.maxs.x, n.maxs.y, n.maxs.z, 0.f };

			g.left = n.left;
			g.right = n.right;

			g.firstPrim = n.firstTri;
			g.primCount = n.triCount;
		}

		g_gpuPF.worldBVHCount = (int)h_bvh.size();

		cl_int err3 = 0;
		g_gpuPF.d_worldBVH = clCreateBuffer(
			g_gpuPF.context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			sizeof(BVHNodeGPU) * g_gpuPF.worldBVHCount,
			h_bvh.data(),
			&err3
		);

		if (err3 != CL_SUCCESS)
		{
			Warning("[GPU-VIS][PRESET3] Failed to upload BVH to GPU\n");
			return false;
		}

		Msg("[GPU-VIS][PRESET3] Uploaded BVH to GPU (%d nodes)\n",
			g_gpuPF.worldBVHCount);
	}


	for (int i = 0; i < g_leafPortals.size(); i++)
	{
		if (g_leafPortals[i].size() > 50)
		{
			Msg("[DEBUG] Leaf %d has %zu portals\n", i, g_leafPortals[i].size());
			break;
		}
	}

	// =========================================================
	// LEAF AABB EXTRACTION (CPU -> GPU)
	// =========================================================
	std::vector<LeafAABBGPU> h_leafAABBs;

	// =========================================================
	// PRESETGPU 2 — WORLD OCCLUDERS FROM SOLID LEAFS
	// =========================================================

	std::vector<WorldAABBGPU> h_worldAABBs;
	h_worldAABBs.reserve(numleafs);

	for (int i = 0; i < numleafs; i++)
	{
		const dleaf_t& L = dleafs[i];

		// Skip non-solid leaves
		if (!(L.contents & CONTENTS_SOLID))
			continue;

		// Skip sky / water / special contents
		if (L.contents & CONTENTS_WATER)
			continue;

		WorldAABBGPU box;
		box.mins = {
			(float)L.mins[0],
			(float)L.mins[1],
			(float)L.mins[2]
		};
		box.maxs = {
			(float)L.maxs[0],
			(float)L.maxs[1],
			(float)L.maxs[2]
		};

		// Reject degenerate leaves
		if (box.mins.x >= box.maxs.x ||
			box.mins.y >= box.maxs.y ||
			box.mins.z >= box.maxs.z)
			continue;

		h_worldAABBs.push_back(box);
	}

	g_gpuPF.worldAABBCount = (int)h_worldAABBs.size();

	if (g_gpuPF.worldAABBCount > 0)
	{
		g_gpuPF.d_worldAABBs = clCreateBuffer(
			g_gpuPF.context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			sizeof(WorldAABBGPU) * g_gpuPF.worldAABBCount,
			h_worldAABBs.data(),
			&err
		);

		Msg("[GPU-VIS][ULTRA] Uploaded %d solid leaf occluders\n",
			g_gpuPF.worldAABBCount);
	}
	else
	{
		g_gpuPF.d_worldAABBs = nullptr;
	}

	h_leafAABBs.reserve(numleafs);

	for (int i = 0; i < numleafs; i++)
	{
		dleaf_t* L = &dleafs[i];

		LeafAABBGPU aabb;
		aabb.mins = {
			(float)L->mins[0],
			(float)L->mins[1],
			(float)L->mins[2]
		};
		aabb.maxs = {
			(float)L->maxs[0],
			(float)L->maxs[1],
			(float)L->maxs[2]
		};

		h_leafAABBs.push_back(aabb);
	}

	// =========================================================
	// LEAF AABB → GPU (HYBRID VIS)
	// =========================================================
	g_gpuPF.d_leafAABBs = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(LeafAABBGPU) * h_leafAABBs.size(),
		h_leafAABBs.data(),
		&err
	);

	if (err != CL_SUCCESS)
	{
		Warning("[LEAF-HYBRID] Failed to upload leaf AABBs to GPU\n");
		return false;
	}

	Msg("[LEAF-HYBRID] Uploaded %zu leaf AABBs to GPU\n", h_leafAABBs.size());

	std::vector<int> h_leafCount;
	std::vector<int> h_leafList;
	BuildFlatLeafPortalArrays(h_leafCount, h_leafList);

	g_gpuPF.numLeaves = (int)h_leafCount.size();
	g_gpuPF.maxPerLeaf = g_maxPerLeaf;

	g_gpuPF.d_leafPortalCount = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		h_leafCount.size() * sizeof(int),
		h_leafCount.data(),
		&err
	);

	g_gpuPF.d_leafPortalList = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		h_leafList.size() * sizeof(int),
		h_leafList.data(),
		&err
	);

	if (err) return false;


	// =========================================================
	// MIGHTSEE (CPU -> GPU)
	// =========================================================
	std::vector<int> h_might(portalCount * longs);
	for (int p = 0; p < portalCount; p++)
		memcpy(&h_might[p * longs], sorted_portals[p]->portalflood,
			longs * sizeof(int));

	g_gpuFF.d_mightSee = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		h_might.size() * sizeof(int),
		h_might.data(),
		&err
	);
	if (err) return false;



	// =========================================================
	// BFS STATES
	// =========================================================
	const size_t stateBytes = portalCount * sizeof(GPUFlowState);

	g_gpuFF.d_stateCur = clCreateBuffer(
		g_gpuPF.context, CL_MEM_READ_WRITE, stateBytes, nullptr, &err);
	g_gpuFF.d_stateNext = clCreateBuffer(
		g_gpuPF.context, CL_MEM_READ_WRITE, stateBytes, nullptr, &err);
	g_gpuFF.d_stateCount = clCreateBuffer(
		g_gpuPF.context, CL_MEM_READ_WRITE, sizeof(int), nullptr, &err);
	g_gpuFF.d_stateNextCount = clCreateBuffer(
		g_gpuPF.context, CL_MEM_READ_WRITE, sizeof(int), nullptr, &err);

	// NextCount = 0 (normal)
	clEnqueueWriteBuffer(
		g_gpuPF.queue,
		g_gpuFF.d_stateNextCount,
		CL_TRUE,
		0,
		sizeof(int),
		&zero,
		0, nullptr, nullptr
	);

	Msg("[GPU-VIS] PortalFlow GPU buffers allocated (%d portals, %d winding verts).\n",
		portalCount, g_totalWindingPoints);


	// =======================================================
	// PRESETGPU 2 — WORLD BRUSH EXTRACTION (AABB)
	// =======================================================
	if (g_gpuPreset >= 2)
	{
		std::vector<WorldAABBGPU> h_worldAABBs;

		for (int i = 0; i < numbrushes; i++)
		{
			const dbrush_t& brush = dbrushes[i];

			// uniquement WORLD + SOLID
			if (!(brush.contents & CONTENTS_SOLID))
				continue;

			Vector mins(1e30f, 1e30f, 1e30f);
			Vector maxs(-1e30f, -1e30f, -1e30f);

			for (int s = 0; s < brush.numsides; s++)
			{
				const dbrushside_t& side = dbrushsides[brush.firstside + s];
				const dplane_t& plane = dplanes[side.planenum];

				// On approxime via normales planes (conservatif)
				Vector n = plane.normal * plane.dist;
				mins.x = min(mins.x, n.x);
				mins.y = min(mins.y, n.y);
				mins.z = min(mins.z, n.z);
				maxs.x = max(maxs.x, n.x);
				maxs.y = max(maxs.y, n.y);
				maxs.z = max(maxs.z, n.z);
			}

			WorldAABBGPU gpu;
			gpu.mins = { mins.x, mins.y, mins.z, 0.f };
			gpu.maxs = { maxs.x, maxs.y, maxs.z, 0.f };
			h_worldAABBs.push_back(gpu);
		}

		g_gpuPF.worldBrushCount = (int)h_worldAABBs.size();

		if (g_gpuPF.worldBrushCount > 0)
		{
			cl_int err;
			g_gpuPF.d_worldAABBs = clCreateBuffer(
				g_gpuPF.context,
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				sizeof(WorldAABBGPU) * g_gpuPF.worldBrushCount,
				h_worldAABBs.data(),
				&err
			);

			if (err != CL_SUCCESS)
			{
				Warning("[GPU-VIS][WORLD] Failed to upload world AABBs\n");
				return false;
			}

			Msg("[GPU-VIS][WORLD] Uploaded %d world brush AABBs\n",
				g_gpuPF.worldBrushCount);
		}
	}

	return true;
}


void DumpGPUPoolPolygon(int offset, int count)
{
	std::vector<float3> pts(count);
	size_t bytes = count * sizeof(float3);

	clEnqueueReadBuffer(
		g_gpuPF.queue,
		d_poolPoints,
		CL_TRUE,
		offset * sizeof(float3),
		bytes,
		pts.data(),
		0, nullptr, nullptr
	);

	Msg("---- GPU POLYGON (%d verts) ----\n", count);
	for (int i = 0; i < count; i++)
		Msg("[%d] %.3f %.3f %.3f\n",
			i, pts[i].x, pts[i].y, pts[i].z);
}

void GPU_CPU_SampleCompare()
{
	Msg("[TryGPU] Comparing GPU vs CPU results...\n");

	const int portalCount = g_gpuPF.portalCount;
	const int longs = g_gpuPF.portalLongs;

	int mismatches = 0;

	for (int p = 0; p < portalCount; p++)
	{
		int* cpu = (int*)sorted_portals[p]->portalvis;
		int* gpu = (int*)sorted_portals[p]->portalvisGPU;

		for (int w = 0; w < longs; w++)
		{
			int cm = cpu[w];
			int gm = gpu[w];

			if (w == longs - 1)
			{
				int leftover = portalCount & 31;
				if (leftover != 0)
				{
					int mask = (1 << leftover) - 1;
					cm &= mask;
					gm &= mask;
				}
			}

			if (w == (p >> 5))
			{
				int bit = 1 << (p & 31);
				cm |= bit;
				gm |= bit;
			}

			if (cm != gm)
			{
				Msg("[TryGPU] MISMATCH portal %d word %d : CPU=%08x GPU=%08x\n",
					p, w, cm, gm);
				mismatches++;
			}
		}
	}

	if (mismatches == 0)
		Msg("[TryGPU] PERFECT MATCH - GPU identical to CPU.\n");
	else
		Msg("[TryGPU] %d mismatches detected.\n", mismatches);
}


void RunKernel_WorldOcclusion()
{
	if (!g_gpuPF.k_ultraWorldOcc) return;
	if (!g_gpuPF.d_worldAABBs || g_gpuPF.worldBrushCount <= 0) return;

	size_t gsz = g_gpuPF.portalCount;

	clSetKernelArg(g_gpuPF.k_ultraWorldOcc, 0, sizeof(cl_mem), &g_gpuPF.d_portalLeaf);
	clSetKernelArg(g_gpuPF.k_ultraWorldOcc, 1, sizeof(cl_mem), &g_gpuPF.d_origins);
	clSetKernelArg(g_gpuPF.k_ultraWorldOcc, 2, sizeof(cl_mem), &g_gpuPF.d_worldAABBs);
	clSetKernelArg(g_gpuPF.k_ultraWorldOcc, 3, sizeof(int), &g_gpuPF.worldBrushCount);
	clSetKernelArg(g_gpuPF.k_ultraWorldOcc, 4, sizeof(cl_mem), &g_gpuPF.d_ultraRejectMask);
	clSetKernelArg(g_gpuPF.k_ultraWorldOcc, 5, sizeof(int), &g_gpuPF.portalCount);
	clSetKernelArg(g_gpuPF.k_ultraWorldOcc, 6, sizeof(int), &g_gpuPF.portalLongs);

	clEnqueueNDRangeKernel(
		g_gpuPF.queue,
		g_gpuPF.k_ultraWorldOcc,
		1, nullptr, &gsz, nullptr,
		0, nullptr, nullptr
	);

	clFinish(g_gpuPF.queue);
}

inline void CopyUltraTempToFinal()
{
	size_t bytes = g_gpuPF.portalCount * g_gpuPF.portalLongs * sizeof(int);
	clEnqueueCopyBuffer(
		g_gpuPF.queue,
		g_gpuPF.d_ultraMaskTemp,
		g_gpuPF.d_ultraMask,
		0, 0, bytes,
		0, nullptr, nullptr
	);
	clFinish(g_gpuPF.queue);
}



void PortalFlow_ULTRA_GPU()
{
	if (g_gpuPreset < 2)
		return;

	if (g_gpuPreset >= 2)
	{
		Msg("[GPU-VIS][ULTRA] World brush occlusion pass...\n");
		RunKernel_WorldOcclusion();
	}

	if (g_gpuPreset >= 3)
	{
		Msg("[GPU-VIS][ULTRA] Ray/BVH occlusion pass...\n");

		size_t gsz = g_gpuPF.portalCount;
		cl_int err = 0;

		// UTILISER LES VRAIS BUFFERS
		err |= clSetKernelArg(g_gpuPF.k_rayTriangleBVH, 0, sizeof(cl_mem), &g_gpuPF.d_portalCenters);
		err |= clSetKernelArg(g_gpuPF.k_rayTriangleBVH, 1, sizeof(cl_mem), &g_gpuFF.d_mightSee);
		err |= clSetKernelArg(g_gpuPF.k_rayTriangleBVH, 2, sizeof(cl_mem), &g_gpuPF.d_ultraRejectMask);

		// REMPLACE PAR LE NOM RÉEL DE TON BUFFER TRIANGLE
		err |= clSetKernelArg(g_gpuPF.k_rayTriangleBVH, 3, sizeof(cl_mem), &g_gpuPF.d_worldTris);
		err |= clSetKernelArg(g_gpuPF.k_rayTriangleBVH, 4, sizeof(int), &g_gpuPF.worldTriCount);

		err |= clSetKernelArg(g_gpuPF.k_rayTriangleBVH, 5, sizeof(cl_mem), &g_gpuPF.d_worldBVH);
		err |= clSetKernelArg(g_gpuPF.k_rayTriangleBVH, 6, sizeof(int), &g_gpuPF.worldBVHCount);
		err |= clSetKernelArg(g_gpuPF.k_rayTriangleBVH, 7, sizeof(cl_mem), &g_gpuPF.d_portalVis);
		err |= clSetKernelArg(g_gpuPF.k_rayTriangleBVH, 8, sizeof(int), &g_gpuPF.portalCount);
		err |= clSetKernelArg(g_gpuPF.k_rayTriangleBVH, 9, sizeof(int), &g_gpuPF.portalLongs);

		if (err != CL_SUCCESS)
		{
			Warning("[GPU-VIS][ULTRA] rayTriangleBVH arg error %d\n", err);
			return;
		}

		err = clEnqueueNDRangeKernel(
			g_gpuPF.queue,
			g_gpuPF.k_rayTriangleBVH,
			1, nullptr,
			&gsz, nullptr,
			0, nullptr, nullptr
		);

		if (err != CL_SUCCESS)
		{
			Warning("[GPU-VIS][ULTRA] rayTriangleBVH launch failed %d\n", err);
			return;
		}

		clFinish(g_gpuPF.queue);
	}
}


// ============================================================================
// HYBRID VIS — RUN GPU FILTER
// ============================================================================

bool RunGPUHybridFilter()
{
	if (!g_gpuHybridVis.enabled)
		return false;

	int portalCount = g_gpuHybridVis.filterJob.portalCount;
	int longs = g_gpuHybridVis.filterJob.portalLongs;

	size_t maskBytes = portalCount * longs * sizeof(uint32_t);
	uint32_t zero = 0;

	// Reset GPU mask
	clEnqueueFillBuffer(
		g_gpuPF.queue,
		g_gpuHybridVis.filterJob.result.d_mightSeeMask,
		&zero,
		sizeof(uint32_t),
		0,
		maskBytes,
		0, nullptr, nullptr
	);

	size_t global = portalCount;

	cl_int err = 0;
	err |= clSetKernelArg(g_gpuPF.k_hybridFilter, 0, sizeof(cl_mem), &g_gpuPF.d_origins);
	err |= clSetKernelArg(g_gpuPF.k_hybridFilter, 1, sizeof(cl_mem), &g_gpuPF.d_portalNormals);
	err |= clSetKernelArg(g_gpuPF.k_hybridFilter, 2, sizeof(cl_mem), &g_gpuPF.d_worldTris);
	err |= clSetKernelArg(g_gpuPF.k_hybridFilter, 3, sizeof(cl_mem), &g_gpuPF.d_worldBVH);
	err |= clSetKernelArg(g_gpuPF.k_hybridFilter, 4, sizeof(cl_mem), &g_gpuHybridVis.filterJob.result.d_mightSeeMask);
	err |= clSetKernelArg(g_gpuPF.k_hybridFilter, 5, sizeof(int), &g_gpuPF.worldTriCount);
	err |= clSetKernelArg(g_gpuPF.k_hybridFilter, 6, sizeof(int), &g_gpuPF.worldBVHCount);
	err |= clSetKernelArg(g_gpuPF.k_hybridFilter, 7, sizeof(int), &portalCount);
	err |= clSetKernelArg(g_gpuPF.k_hybridFilter, 8, sizeof(int), &longs);

	if (err != CL_SUCCESS)
	{
		Warning("[HYBRID-VIS] Kernel arg error %d\n", err);
		return false;
	}

	err = clEnqueueNDRangeKernel(
		g_gpuPF.queue,
		g_gpuPF.k_hybridFilter,
		1, nullptr,
		&global, nullptr,
		0, nullptr, nullptr
	);

	if (err != CL_SUCCESS)
	{
		Warning("[HYBRID-VIS] Kernel launch failed %d\n", err);
		return false;
	}

	clFinish(g_gpuPF.queue);

	// Read back result
	clEnqueueReadBuffer(
		g_gpuPF.queue,
		g_gpuHybridVis.filterJob.result.d_mightSeeMask,
		CL_TRUE,
		0,
		maskBytes,
		g_gpuHybridVis.cpuMightSeeMask.data(),
		0, nullptr, nullptr
	);


	// ================= SAFETY CHECK =================
	int totalBits = portalCount * longs * 32;
	int keptBits = 0;

	for (int i = 0; i < portalCount * longs; i++)
	{
		keptBits += __popcnt(g_gpuHybridVis.cpuMightSeeMask[i]);
	}

	float ratio = (float)keptBits / (float)totalBits;

	Msg("[HYBRID-VIS] GPU kept %.2f%% visibility\n", ratio * 100.0f);


	// Section desactivée pour l'instant car on veut être "agressif"
	if (ratio < 0.005f)
	{
		Warning("[HYBRID-VIS] GPU rejected too much (%.2f%%), fallback CPU\n",
			ratio * 100.0f);
		return false;
	}
	// =================================================


	Msg("[HYBRID-VIS] GPU hybrid filter completed\n");
	return true;
}


bool RunGPULeafHybridFilter()
{
	if (!g_gpuLeafHybridVis.enabled)
		return false;

	int leafCount = g_gpuLeafHybridVis.leafCount;
	int longs = g_gpuLeafHybridVis.leafLongs;

	size_t maskBytes = leafCount * longs * sizeof(visword_t);
	int zero = 0;

	clEnqueueFillBuffer(
		g_gpuPF.queue,
		g_gpuLeafHybridVis.result.d_leafMightSee,
		&zero,
		sizeof(int),
		0,
		maskBytes,
		0, nullptr, nullptr
	);

	size_t global = (size_t)leafCount * (size_t)leafCount;

	clSetKernelArg(g_gpuPF.k_leafHybridFilter, 0, sizeof(cl_mem), &g_gpuPF.d_leafAABBs);
	clSetKernelArg(g_gpuPF.k_leafHybridFilter, 1, sizeof(cl_mem),
		&g_gpuLeafHybridVis.result.d_leafMightSee);
	clSetKernelArg(g_gpuPF.k_leafHybridFilter, 2, sizeof(int), &leafCount);
	clSetKernelArg(g_gpuPF.k_leafHybridFilter, 3, sizeof(int), &longs);

	clEnqueueNDRangeKernel(
		g_gpuPF.queue,
		g_gpuPF.k_leafHybridFilter,
		1, nullptr, &global, nullptr,
		0, nullptr, nullptr
	);

	clFinish(g_gpuPF.queue);

	clEnqueueReadBuffer(
		g_gpuPF.queue,
		g_gpuLeafHybridVis.result.d_leafMightSee,
		CL_TRUE,
		0,
		maskBytes,
		g_gpuLeafHybridVis.result.cpuLeafMightSee.data(),
		0, nullptr, nullptr
	);

	Msg("[LEAF-HYBRID] GPU leaf filter completed\n");
	return true;
}



// ============================================================================
// HYBRID VIS — APPLY FILTER TO CPU
// ============================================================================

void ApplyHybridMightSeeToCPU()
{
	int portalCount = g_gpuHybridVis.filterJob.portalCount;
	int longs = g_gpuHybridVis.filterJob.portalLongs;

	for (int p = 0; p < portalCount; p++)
	{
		portal_t* P = sorted_portals[p];

		uint32_t* gpu = &g_gpuHybridVis.cpuMightSeeMask[p * longs];
		uint32_t* dst = (uint32_t*)P->portalHybridMask;

		for (int w = 0; w < longs; w++)
			dst[w] &= gpu[w];
	}

	Msg("[HYBRID-VIS] GPU hybrid mask stored (non destructive)\n");
}



void ApplyLeafHybridToPortals()
{
	int portalCount = g_numportals * 2;
	int portalLongs = portallongs;

	for (int p = 0; p < portalCount; p++)
	{
		portal_t* P = sorted_portals[p];
		int leafA = P->leaf;

		uint32_t* cpu = (uint32_t*)P->portalflood;

		for (int q = 0; q < portalCount; q++)
		{
			portal_t* Q = sorted_portals[q];
			int leafB = Q->leaf;

			int word = leafB >> 5;
			int bit = 1 << (leafB & 31);

			if (!(g_gpuLeafHybridVis.result.cpuLeafMightSee[leafA * g_gpuLeafHybridVis.leafLongs + word] & bit))
			{
				// Leaf cannot see → portal cannot see
				cpu[q >> 5] &= ~(1 << (q & 31));
			}
		}
	}

	Msg("[LEAF-HYBRID] Leaf visibility applied to portal flood\n");
}


void PortalFlow_CPU_Ordered(int iThread, int workIndex)
{
	PortalFlow_CPU(iThread, g_portalOrderCPU[workIndex]);
}


void BuildPortalOrderByMightSee()
{
	int portalCount = g_numportals * 2;
	g_portalOrderCPU.resize(portalCount);

	for (int i = 0; i < portalCount; i++)
		g_portalOrderCPU[i] = i;

	std::sort(g_portalOrderCPU.begin(), g_portalOrderCPU.end(),
		[](int a, int b)
		{
			return sorted_portals[a]->nummightsee > sorted_portals[b]->nummightsee;
		});
}

void PortalFlow_CPU(int iThread, int portalnum)
{
	threaddata_t	data;
	int				i;
	portal_t* p;
	int				c_might, c_can;

	p = sorted_portals[portalnum];
	p->status = stat_working;

	c_might = CountBits(p->portalflood, g_numportals * 2);

	memset(&data, 0, sizeof(data));
	data.base = p;

	data.pstack_head.portal = p;
	data.pstack_head.source = p->winding;
	data.pstack_head.portalplane = p->plane;
	for (i = 0; i < portallongs; i++)
		((long*)data.pstack_head.mightsee)[i] = ((long*)p->portalflood)[i];

	RecursiveLeafFlow_CPU(p->leaf, &data, &data.pstack_head);


	p->status = stat_done;

	c_can = CountBits(p->portalvis, g_numportals * 2);

	qprintf("portal:%4i  mightsee:%4i  cansee:%4i (%i chains)\n",
		(int)(p - portals), c_might, c_can, data.c_chains);
}


int		c_flood, c_vis;

/*
==================
SimpleFlood

==================
*/
void SimpleFlood(portal_t* srcportal, int leafnum)
{
	int		i;
	leaf_t* leaf;
	portal_t* p;
	int		pnum;

	leaf = &leafs[leafnum];

	for (i = 0; i < leaf->portals.Count(); i++)
	{
		p = leaf->portals[i];
		pnum = p - portals;
		if (!CheckBit(srcportal->portalfront, pnum))
			continue;

		if (CheckBit(srcportal->portalflood, pnum))
			continue;

		SetBit(srcportal->portalflood, pnum);

		SimpleFlood(srcportal, p->leaf);
	}
}

/*
==============
BasePortalVis
==============
*/

void BasePortalVis(int iThread, int portalnum)
{
	int			j, k;
	portal_t* tp, * p;
	float		d;
	winding_t* w;
	Vector		segment;
	double		dist2, minDist2;

	// get the portal
	p = portals + portalnum;

	//
	// allocate memory for bitwise vis solutions for this portal
	//
	p->portalfront = (byte*)malloc(portalbytes);
	memset(p->portalfront, 0, portalbytes);

	p->portalflood = (byte*)malloc(portalbytes);
	memset(p->portalflood, 0, portalbytes);

	p->portalvis = (byte*)malloc(portalbytes);
	memcpy(p->portalvis, p->portalflood, portalbytes);

	p->portalvis_cpu = (byte*)malloc(portalbytes);
	memcpy(p->portalvis_cpu, p->portalflood, portalbytes);

	p->portalvisGPU = (byte*)malloc(portalbytes);
	memset(p->portalvisGPU, 0, portalbytes);

	p->portalHybridMask = (byte*)malloc(portalbytes);
	memset(p->portalHybridMask, 0xFF, portalbytes); // tout visible par défaut

	//
	// test the given portal against all of the portals in the map
	//
	for (j = 0, tp = portals; j < g_numportals * 2; j++, tp++)
	{
		// don't test against itself
		// don't test against itself
		if (j == portalnum)
			continue;

		//
		//
		//
		w = tp->winding;
		for (k = 0; k < w->numpoints; k++)
		{
			d = DotProduct(w->points[k], p->plane.normal) - p->plane.dist;
			if (d > ON_VIS_EPSILON)
				break;
		}
		if (k == w->numpoints)
			continue;	// no points on front

		//
		//
		//
		w = p->winding;
		for (k = 0; k < w->numpoints; k++)
		{
			d = DotProduct(w->points[k], tp->plane.normal) - tp->plane.dist;
			if (d < -ON_VIS_EPSILON)
				break;
		}
		if (k == w->numpoints)
			continue;	// no points on front

		//
		// if using radius visibility -- check to see if any portal points lie inside of the
		// radius given
		//
		if (g_bUseRadius)
		{
			w = tp->winding;
			minDist2 = 1024000000.0;			// 32000^2
			for (k = 0; k < w->numpoints; k++)
			{
				VectorSubtract(w->points[k], p->origin, segment);
				dist2 = (segment[0] * segment[0]) + (segment[1] * segment[1]) + (segment[2] * segment[2]);
				if (dist2 < minDist2)
				{
					minDist2 = dist2;
				}
			}

			if (minDist2 > g_VisRadius)
				continue;
		}

		// add current portal to given portal's list of visible portals
		SetBit(p->portalfront, j);
	}

	SimpleFlood(p, p->leaf);

	p->nummightsee = CountBits(p->portalflood, g_numportals * 2);
	//	Msg ("portal %i: %i mightsee\n", portalnum, p->nummightsee);
	c_flood += p->nummightsee;
}



void BuildLeafPortalTable()
{
	int numLeaves = portalclusters;

	g_leafPortals.clear();
	g_leafPortals.resize(numLeaves);

	int portalCount = g_numportals * 2;

	// -------------------------------------------
	// Construire un mapping rawIndex -> sortedIndex
	// -------------------------------------------
	static std::vector<int> rawToSorted;
	rawToSorted.resize(portalCount);

	for (int s = 0; s < portalCount; s++)
	{
		portal_t* P = sorted_portals[s];
		int rawIdx = P - portals;
		rawToSorted[rawIdx] = s;
	}

	// -------------------------------------------
	// Ajouter chaque portail dans la liste de son leaf
	// mais en utilisant l’index TRIÉ
	// -------------------------------------------
	for (int raw = 0; raw < portalCount; raw++)
	{
		portal_t* P = portals + raw;
		int leaf = P->leaf;

		if (leaf < 0 || leaf >= numLeaves)
			continue;

		int sortedIdx = rawToSorted[raw];
		g_leafPortals[leaf].push_back(sortedIdx);
	}

	Msg("[GPU-VIS] Built leaf->portal adjacency table (sorted indices, %d leaves)\n", numLeaves);
}

// ======================================================================
// Flatten leafPortal adjacency table for GPU upload
// ======================================================================

void BuildFlatLeafPortalArrays(std::vector<int>& outCount,
	std::vector<int>& outList)
{
	int numLeaves = g_leafPortals.size();
	outCount.resize(numLeaves);
	outList.resize(numLeaves * g_maxPerLeaf);

	for (int leaf = 0; leaf < numLeaves; leaf++)
	{
		const auto& vec = g_leafPortals[leaf];
		int count = vec.size();
		if (count > g_maxPerLeaf)
			count = g_maxPerLeaf;

		outCount[leaf] = count;

		// Copier les portails triés
		for (int i = 0; i < count; i++)
			outList[leaf * g_maxPerLeaf + i] = vec[i];

		// Padding à -1
		for (int i = count; i < g_maxPerLeaf; i++)
			outList[leaf * g_maxPerLeaf + i] = -1;
	}

	Msg("[GPU-VIS] Flattened adjacency (leaf -> portal sorted) uploaded to GPU.\n");
}
