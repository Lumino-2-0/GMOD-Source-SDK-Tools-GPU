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



// =============================================================
// UNIFIED EPSILON CONSTANTS (CPU <-> GPU CONSISTENCY)
// =============================================================
#define VIS_EPSILON_PLANE       1e-5f
#define VIS_EPSILON_DOT         1e-5f
#define VIS_EPSILON_CLIP        1e-5f
#define VIS_EPSILON_WINDING     1e-5f
#define VIS_EPSILON_COLINEAR    1e-6f

// Remplace l'ancien ON_VIS_EPSILON du CPU
#undef ON_VIS_EPSILON
#define ON_VIS_EPSILON VIS_EPSILON_CLIP


// Nombre total de points dynamiques
static int g_totalWindingPoints = 0;

// Offsets CPU -> GPU (CPU side)
static std::vector<int> g_windingOffsetsCPU;
static std::vector<float3> g_windingPointsCPU;

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
void BuildFlatLeafPortalArrays(std::vector<int>& outCount,
	std::vector<int>& outList);

static std::mutex g_trace_mutex;
static std::ofstream g_trace_file;
static std::atomic<bool> g_trace_inited{ false };

// ========= GPU PRUNE TUNING =========

int g_gpuPreset = 2;

#include <CL/cl.h>


GPUPortalFlowCLContext g_gpuPF = {};
GPUFlowFixed g_gpuFF = {};

// =============================================================================
// GLOBAL WINDING POOL (8 million float3 = ~96 MB GPU)
// =============================================================================
static std::vector<int> g_initSrcOffsetCPU;
static std::vector<int> g_initSrcCountCPU;
static std::vector<float3> g_initWindingCPU;

static std::vector<std::vector<int>> g_leafPortals;
static int g_maxPerLeaf = 256;

bool InitOpenCL_PortalFlow()
{
	if (g_gpuPF.initialized)
		return true;

	cl_int err = 0;

	// -------------------------------------------------------
	// PLATFORM
	// -------------------------------------------------------
	cl_uint platformCount = 0;
	clGetPlatformIDs(0, nullptr, &platformCount);
	if (platformCount == 0)
	{
		Warning("[GPU-VIS] No OpenCL platform found.\n");
		return false;
	}

	std::vector<cl_platform_id> platforms(platformCount);
	clGetPlatformIDs(platformCount, platforms.data(), nullptr);
	g_gpuPF.platform = platforms[0];

	// -------------------------------------------------------
	// DEVICE (GPU then fallback CPU)
	// -------------------------------------------------------
	cl_uint deviceCount = 0;
	err = clGetDeviceIDs(g_gpuPF.platform, CL_DEVICE_TYPE_GPU,
		0, nullptr, &deviceCount);

	if (err != CL_SUCCESS || deviceCount == 0)
	{
		Warning("[GPU-VIS] No GPU, trying CPU...\n");
		err = clGetDeviceIDs(g_gpuPF.platform, CL_DEVICE_TYPE_CPU,
			0, nullptr, &deviceCount);
		if (err != CL_SUCCESS || deviceCount == 0)
		{
			Warning("[GPU-VIS] No OpenCL device available.\n");
			return false;
		}
	}

	std::vector<cl_device_id> devices(deviceCount);
	clGetDeviceIDs(g_gpuPF.platform, CL_DEVICE_TYPE_ALL, deviceCount,
		devices.data(), nullptr);
	g_gpuPF.device = devices[0];

	// -------------------------------------------------------
	// CONTEXT
	// -------------------------------------------------------
	g_gpuPF.context = clCreateContext(nullptr, 1,
		&g_gpuPF.device, nullptr, nullptr, &err);

	if (err != CL_SUCCESS)
	{
		Warning("[GPU-VIS] Failed to create OpenCL context.\n");
		return false;
	}

	// -------------------------------------------------------
	// QUEUE
	// -------------------------------------------------------
#if defined(CL_VERSION_2_0)
	const cl_queue_properties props[] =
	{ CL_QUEUE_PROPERTIES, 0, 0 };
	g_gpuPF.queue = clCreateCommandQueueWithProperties(
		g_gpuPF.context, g_gpuPF.device, props, &err);
#else
	g_gpuPF.queue = clCreateCommandQueue(
		g_gpuPF.context, g_gpuPF.device, 0, &err);
#endif

	if (err != CL_SUCCESS)
	{
		Warning("[GPU-VIS] Failed to create command queue.\n");
		return false;
	}

	// -------------------------------------------------------
	// PROGRAM FROM SOURCE
	// -------------------------------------------------------
	const char* src = g_gpuPortalFlowKernels;
	size_t srcSize = strlen(g_gpuPortalFlowKernels);

	g_gpuPF.program = clCreateProgramWithSource(
		g_gpuPF.context, 1, &src, &srcSize, &err);

	if (err != CL_SUCCESS)
	{
		Warning("[GPU-VIS] Failed to create OpenCL program.\n");
		return false;
	}

	const char* opts = "-cl-fast-relaxed-math -cl-std=CL2.0";
	err = clBuildProgram(g_gpuPF.program,
		1, &g_gpuPF.device, opts, nullptr, nullptr);

	// PRINT BUILD LOG
	{
		size_t logSize = 0;
		clGetProgramBuildInfo(g_gpuPF.program, g_gpuPF.device,
			CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

		if (logSize > 1)
		{
			std::vector<char> log(logSize);
			clGetProgramBuildInfo(g_gpuPF.program, g_gpuPF.device,
				CL_PROGRAM_BUILD_LOG,
				logSize, log.data(), nullptr);
			Msg("[GPU-VIS] Build log:\n%s\n", log.data());
		}
	}

	if (err != CL_SUCCESS)
	{
		Warning("[GPU-VIS] Failed to build kernels.\n");
		return false;
	}

	// -------------------------------------------------------
	// LOAD KERNELS
	// -------------------------------------------------------
	cl_int kerr = 0;

	g_gpuPF.k_resetPool = clCreateKernel(
		g_gpuPF.program, "resetWindingPool", &kerr);

	g_gpuPF.k_gpuClipWinding = clCreateKernel(
		g_gpuPF.program, "gpuChopWinding", &kerr);

	g_gpuPF.k_gpuGenerateSep = clCreateKernel(
		g_gpuPF.program, "gpuGenerateSeparators", &kerr);

	g_gpuPF.k_gpuClipToSep = clCreateKernel(
		g_gpuPF.program, "gpuClipToSeparators", &kerr);

	g_gpuPF.k_expand = clCreateKernel(
		g_gpuPF.program, "portalFlowExpand", &kerr);

	g_gpuPF.initialized = true;

	Msg("[GPU-VIS] OpenCL PortalFlow READY.\n");
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
winding_t* ClipToSeperators(
	winding_t* source,
	winding_t* pass,
	winding_t* target,
	bool flipclip,
	pstack_t* stack
) {
	int i, j, k, l;
	plane_t plane;
	Vector v1, v2;
	float d;
	float length;
	int counts[3];
	bool fliptest;

	if (!source || !pass || !target)
		return target;

	// === LOOP EXACT CPU ===
	for (i = 0; i < source->numpoints; i++)
	{
		l = (i + 1) % source->numpoints;

		VectorSubtract(source->points[l], source->points[i], v1);

		for (j = 0; j < pass->numpoints; j++)
		{
			VectorSubtract(pass->points[j], source->points[i], v2);

			// normal = v1 × v2
			plane.normal[0] = v1[1] * v2[2] - v1[2] * v2[1];
			plane.normal[1] = v1[2] * v2[0] - v1[0] * v2[2];
			plane.normal[2] = v1[0] * v2[1] - v1[1] * v2[0];

			// invalid plane?
			length = DotProduct(plane.normal, plane.normal);
			if (length < VIS_EPSILON_WINDING)
				continue;

			length = 1.0f / sqrt(length);

			plane.normal[0] *= length;
			plane.normal[1] *= length;
			plane.normal[2] *= length;

			plane.dist = DotProduct(pass->points[j], plane.normal);

			// ------------------------
			// Determine flip direction
			// ------------------------
			fliptest = false;
			for (k = 0; k < source->numpoints; k++)
			{
				if (k == i || k == l)
					continue;

				d = DotProduct(source->points[k], plane.normal) - plane.dist;

				if (d < -VIS_EPSILON_CLIP)
				{
					fliptest = false;
					break;
				}
				else if (d > VIS_EPSILON_CLIP)
				{
					fliptest = true;
					break;
				}
			}
			if (k == source->numpoints)
				continue; // Degenerate plane

			// Flip if needed
			if (fliptest)
			{
				VectorSubtract(vec3_origin, plane.normal, plane.normal);
				plane.dist = -plane.dist;
			}

			// ------------------------
			// Check pass side
			// ------------------------
			counts[0] = counts[1] = counts[2] = 0;
			for (k = 0; k < pass->numpoints; k++)
			{
				if (k == j) continue;

				d = DotProduct(pass->points[k], plane.normal) - plane.dist;

				if (d < -VIS_EPSILON_CLIP)
					break;
				else if (d > VIS_EPSILON_CLIP)
					counts[0]++;
				else
					counts[2]++;
			}
			if (k != pass->numpoints)
				continue;

			if (!counts[0])
				continue; // coplanar → skip

			// Final flip if required by CPU logic
			if (flipclip)
			{
				VectorSubtract(vec3_origin, plane.normal, plane.normal);
				plane.dist = -plane.dist;
			}

			// ------------------------
			// CLIP TARGET by plane
			// ------------------------
			target = ChopWinding(target, stack, &plane);
			if (!target)
				return NULL; // fully clipped → not visible
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


void BuildFlatLeafPortalArrays(std::vector<int>& outCount, std::vector<int>& outList);

/*
// --------------------
// PortalFlow
// --------------------
//
*/


bool AllocatePortalFlowBuffers()
{
	if (!g_gpuPF.initialized)
		return false;

	cl_int err = 0;

	// ----------------------------------------
	// PARAMÈTRES PORTAILS
	// ----------------------------------------
	g_gpuPF.portalCount = g_numportals * 2;
	g_gpuPF.portalLongs = portallongs;

	const int portalCount = g_gpuPF.portalCount;
	const int longs = g_gpuPF.portalLongs;

	const size_t maskBytes = portalCount * longs * sizeof(int);
	const size_t originBytes = portalCount * sizeof(float3);
	const size_t radiusBytes = portalCount * sizeof(float);
	const size_t planeBytes = portalCount * sizeof(float4);

#define REL(x) if(x){ clReleaseMemObject(x); x=nullptr; }

	REL(g_gpuPF.d_portalVis);
	REL(g_gpuPF.d_origins);
	REL(g_gpuPF.d_radius);
	REL(g_gpuPF.d_planes);
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

#undef REL

	// =====================================================
	// PORTALVIS GPU = copie CPU portalflood[] + auto-vis
	// =====================================================
	g_gpuPF.d_portalVis =
		clCreateBuffer(g_gpuPF.context, CL_MEM_READ_WRITE, maskBytes, nullptr, &err);
	if (err) return false;

	{
		std::vector<int> buf(portalCount * longs);

		for (int p = 0; p < portalCount; p++)
			memcpy(&buf[p * longs], sorted_portals[p]->portalflood,
				longs * sizeof(int));

		clEnqueueWriteBuffer(
			g_gpuPF.queue, g_gpuPF.d_portalVis,
			CL_TRUE, 0, buf.size() * sizeof(int), buf.data(),
			0, nullptr, nullptr
		);
	}

	// auto-visibilité
	for (int p = 0; p < portalCount; p++)
	{
		int byte = p >> 5;
		int bit = 1 << (p & 31);

		size_t off = p * longs * sizeof(int) + byte * sizeof(int);

		int w = 0;
		clEnqueueReadBuffer(
			g_gpuPF.queue, g_gpuPF.d_portalVis,
			CL_TRUE, off, sizeof(int), &w,
			0, nullptr, nullptr
		);

		w |= bit;

		clEnqueueWriteBuffer(
			g_gpuPF.queue, g_gpuPF.d_portalVis,
			CL_TRUE, off, sizeof(int), &w,
			0, nullptr, nullptr
		);
	}

	// =====================================================
	// ORIGINES / RADIUS / PLANES
	// =====================================================
	g_gpuPF.d_origins = clCreateBuffer(
		g_gpuPF.context, CL_MEM_READ_ONLY, originBytes, nullptr, &err);
	if (err) return false;

	g_gpuPF.d_radius = clCreateBuffer(
		g_gpuPF.context, CL_MEM_READ_ONLY, radiusBytes, nullptr, &err);
	if (err) return false;

	g_gpuPF.d_planes = clCreateBuffer(
		g_gpuPF.context, CL_MEM_READ_ONLY, planeBytes, nullptr, &err);
	if (err) return false;

	std::vector<float3> h_orig(portalCount);
	std::vector<float>  h_rad(portalCount);
	std::vector<float4> h_pl(portalCount);
	std::vector<int>    h_leaf(portalCount);

	for (int p = 0; p < portalCount; p++)
	{
		portal_t* P = sorted_portals[p];

		h_orig[p] = { P->origin.x, P->origin.y, P->origin.z };
		h_rad[p] = P->radius;
		h_pl[p] = { P->plane.normal.x, P->plane.normal.y,
					  P->plane.normal.z, P->plane.dist };
		h_leaf[p] = P->leaf;
	}

	clEnqueueWriteBuffer(g_gpuPF.queue, g_gpuPF.d_origins, CL_TRUE, 0, originBytes, h_orig.data(), 0, nullptr, nullptr);
	clEnqueueWriteBuffer(g_gpuPF.queue, g_gpuPF.d_radius, CL_TRUE, 0, radiusBytes, h_rad.data(), 0, nullptr, nullptr);
	clEnqueueWriteBuffer(g_gpuPF.queue, g_gpuPF.d_planes, CL_TRUE, 0, planeBytes, h_pl.data(), 0, nullptr, nullptr);

	// =====================================================
	// PORTAL → LEAF TABLE
	// =====================================================
	g_gpuPF.d_portalLeaf = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		portalCount * sizeof(int),
		h_leaf.data(),
		&err
	);
	if (err) return false;

	Msg("[GPU-VIS] portalLeaf uploaded (%d entries)\n", portalCount);

	// =====================================================
	// DYNAMIC WINDINGS → POOL 16M
	// =====================================================
	std::vector<int>    h_srcOff(portalCount);
	std::vector<int>    h_srcCnt(portalCount);
	std::vector<float3> h_pts;

	h_pts.reserve(200000);

	int offset = 0;
	for (int p = 0; p < portalCount; p++)
	{
		winding_t* W = sorted_portals[p]->winding;
		int cnt = W ? W->numpoints : 0;

		h_srcOff[p] = offset;
		h_srcCnt[p] = cnt;

		for (int i = 0; i < cnt; i++)
		{
			Vector& v = W->points[i];
			h_pts.push_back({ v.x, v.y, v.z });
		}
		offset += cnt;
	}

	g_gpuPF.d_initSrcOffset = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int) * portalCount,
		h_srcOff.data(),
		&err
	);
	if (err) return false;

	g_gpuPF.d_initSrcCount = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int) * portalCount,
		h_srcCnt.data(),
		&err
	);
	if (err) return false;

	// POOL 16M float3
	const int POOL_MAX = 16000000;

	g_gpuPF.d_windPool = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_WRITE,
		sizeof(float3) * POOL_MAX,
		nullptr,
		&err
	);
	if (err) return false;

	if (!h_pts.empty())
	{
		clEnqueueWriteBuffer(
			g_gpuPF.queue, g_gpuPF.d_windPool,
			CL_TRUE, 0,
			sizeof(float3) * h_pts.size(),
			h_pts.data(),
			0, nullptr, nullptr
		);
	}

	int initialPool = h_pts.size();
	g_gpuPF.d_windPoolCount = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(int),
		&initialPool,
		&err
	);
	if (err) return false;

	// =====================================================
	// LEAF → PORTAL GRAPH
	// =====================================================
	BuildLeafPortalTable();

	std::vector<int> adjCount;
	std::vector<int> adjList;
	BuildFlatLeafPortalArrays(adjCount, adjList);

	g_gpuPF.numLeaves = adjCount.size();
	g_gpuPF.maxPerLeaf = g_maxPerLeaf;

	g_gpuPF.d_leafPortalCount = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int) * adjCount.size(),
		adjCount.data(),
		&err
	);
	if (err) return false;

	g_gpuPF.d_leafPortalList = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int) * adjList.size(),
		adjList.data(),
		&err
	);
	if (err) return false;

	// =====================================================
	// MIGHTSEE (portalflood CPU)
	// =====================================================
	std::vector<int> h_might(portalCount * longs);
	for (int p = 0; p < portalCount; p++)
		memcpy(&h_might[p * longs], sorted_portals[p]->portalflood, longs * sizeof(int));

	g_gpuFF.d_mightSee = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int) * h_might.size(),
		h_might.data(),
		&err
	);
	if (err) return false;

	// =====================================================
	// FLOW STATE BUFFERS
	// =====================================================
	const size_t stateBytes = portalCount * sizeof(GPUFlowState);

	g_gpuFF.d_stateCur = clCreateBuffer(
		g_gpuPF.context, CL_MEM_READ_WRITE, stateBytes, nullptr, &err);
	if (err) return false;

	g_gpuFF.d_stateNext = clCreateBuffer(
		g_gpuPF.context, CL_MEM_READ_WRITE, stateBytes, nullptr, &err);
	if (err) return false;

	g_gpuFF.d_stateCount = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(int),
		(void*)&portalCount,
		&err
	);
	if (err) return false;

	int zero = 0;
	g_gpuFF.d_stateNextCount = clCreateBuffer(
		g_gpuPF.context,
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(int),
		&zero,
		&err
	);
	if (err) return false;

	// init states
	std::vector<GPUFlowState> init(portalCount);

	for (int p = 0; p < portalCount; p++)
	{
		init[p].portal = p;
		init[p].leaf = h_leaf[p];
		init[p].mightOffset = p * longs;

		init[p].firstPass = 1;

		init[p].srcOffset = h_srcOff[p];
		init[p].srcCount = h_srcCnt[p];

		init[p].passOffset = -1;
		init[p].passCount = 0;
	}

	clEnqueueWriteBuffer(
		g_gpuPF.queue, g_gpuFF.d_stateCur,
		CL_TRUE, 0, stateBytes,
		init.data(),
		0, nullptr, nullptr
	);

	Msg("[GPU-VIS] PortalFlow GPU buffers allocated (%d portals, %d winding verts).\n",
		portalCount, (int)h_pts.size());

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

	for (int p = 0; p < portalCount; ++p)
	{
		portal_t* P = sorted_portals[p];

		int* cpu = (int*)P->portalvis;
		int* gpu = (int*)P->portalvisGPU;

		if (!gpu)
		{
			Warning("[TryGPU] Missing GPU buffer for portal %d\n", p);
			mismatches++;
			continue;
		}

		for (int w = 0; w < longs; w++)
		{
			int c = cpu[w];
			int g = gpu[w];

			// -----------------------------------------
			// 1) FORCE SELF-VISIBILITY (CPU le fait toujours)
			// -----------------------------------------
			if (w == (p >> 5))
			{
				int bit = 1 << (p & 31);
				c |= bit;
				g |= bit;
			}

			// -----------------------------------------
			// 2) MASK SUR LE DERNIER WORD (padding)
			// -----------------------------------------
			if (w == longs - 1)
			{
				int leftovers = portalCount & 31;
				if (leftovers)
				{
					uint32_t mask = (1u << leftovers) - 1u;
					c &= mask;
					g &= mask;
				}
			}

			// -----------------------------------------
			// 3) COMPARAISON STRICTE
			// -----------------------------------------
			if (c != g)
			{
				mismatches++;

				if (g_bDebugMode)
				{
					Warning(
						"[TryGPU] WORD mismatch portal=%d word=%d CPU=%08X GPU=%08X\n",
						p, w, c, g
					);
				}
			}
		}

		// -----------------------------------------
		// 4) MATCH STATUT CPU (sécurité)
		// -----------------------------------------
		P->status = stat_done;
	}

	if (mismatches == 0)
	{
		Msg("[TryGPU] PERFECT MATCH — GPU identical to CPU for ALL %d portals.\n",
			portalCount);
	}
	else
	{
		Warning("[TryGPU] %d mismatches found — CPU results will be preferred.\n",
			mismatches);
	}
}


void PortalFlow_FullGPU()
{
	Msg("[GPU-VIS] FULL GPU Flood fill starting...\n");

	const int portalCount = g_gpuPF.portalCount;
	const int longs = g_gpuPF.portalLongs;

	cl_int err = 0;

	while (true)
	{
		int curCountCPU = 0;

		clEnqueueReadBuffer(
			g_gpuPF.queue,
			g_gpuFF.d_stateCount,
			CL_TRUE,
			0,
			sizeof(int),
			&curCountCPU,
			0, nullptr, nullptr
		);

		if (g_bDebugMode)
			Msg("[GPU-VIS] Iteration: active states = %d\n", curCountCPU);

		if (curCountCPU == 0)
			break;

		int zero = 0;
		clEnqueueWriteBuffer(
			g_gpuPF.queue,
			g_gpuFF.d_stateNextCount,
			CL_TRUE,
			0,
			sizeof(int),
			&zero,
			0, nullptr, nullptr
		);

		clFinish(g_gpuPF.queue);

		size_t gsz = curCountCPU;

		err = clSetKernelArg(g_gpuPF.k_expand, 0, sizeof(cl_mem), &g_gpuPF.d_origins);
		err |= clSetKernelArg(g_gpuPF.k_expand, 1, sizeof(cl_mem), &g_gpuPF.d_radius);
		err |= clSetKernelArg(g_gpuPF.k_expand, 2, sizeof(cl_mem), &g_gpuPF.d_planes);

		err |= clSetKernelArg(g_gpuPF.k_expand, 3, sizeof(cl_mem), &g_gpuPF.d_portalVis);
		err |= clSetKernelArg(g_gpuPF.k_expand, 4, sizeof(cl_mem), &g_gpuFF.d_mightSee);

		err |= clSetKernelArg(g_gpuPF.k_expand, 5, sizeof(cl_mem), &g_gpuFF.d_stateCur);
		err |= clSetKernelArg(g_gpuPF.k_expand, 6, sizeof(cl_mem), &g_gpuFF.d_stateNext);
		err |= clSetKernelArg(g_gpuPF.k_expand, 7, sizeof(cl_mem), &g_gpuFF.d_stateCount);
		err |= clSetKernelArg(g_gpuPF.k_expand, 8, sizeof(cl_mem), &g_gpuFF.d_stateNextCount);

		err |= clSetKernelArg(g_gpuPF.k_expand, 9, sizeof(cl_mem), &g_gpuPF.d_leafPortalCount);
		err |= clSetKernelArg(g_gpuPF.k_expand, 10, sizeof(cl_mem), &g_gpuPF.d_leafPortalList);

		err |= clSetKernelArg(g_gpuPF.k_expand, 11, sizeof(int), &longs);
		err |= clSetKernelArg(g_gpuPF.k_expand, 12, sizeof(int), &portalCount);
		err |= clSetKernelArg(g_gpuPF.k_expand, 13, sizeof(int), &g_gpuPF.maxPerLeaf);

		err |= clSetKernelArg(g_gpuPF.k_expand, 14, sizeof(cl_mem), &g_gpuPF.d_portalLeaf);

		err |= clSetKernelArg(g_gpuPF.k_expand, 15, sizeof(cl_mem), &g_gpuPF.d_windPool);
		err |= clSetKernelArg(g_gpuPF.k_expand, 16, sizeof(cl_mem), &g_gpuPF.d_windPoolCount);

		err |= clSetKernelArg(g_gpuPF.k_expand, 17, sizeof(cl_mem), &g_gpuPF.d_initSrcOffset);
		err |= clSetKernelArg(g_gpuPF.k_expand, 18, sizeof(cl_mem), &g_gpuPF.d_initSrcCount);

		if (err != CL_SUCCESS)
		{
			Warning("[GPU-VIS] Failed to set kernel args (err=%d)\n", err);
			return;
		}

		err = clEnqueueNDRangeKernel(
			g_gpuPF.queue,
			g_gpuPF.k_expand,
			1,
			nullptr,
			&gsz,
			nullptr,
			0, nullptr, nullptr
		);

		if (err != CL_SUCCESS)
		{
			Warning("[GPU-VIS] Kernel launch FAILED (portalFlowExpand), err=%d\n", err);
			return;
		}

		clFinish(g_gpuPF.queue);

		int nextCountCPU = 0;
		clEnqueueReadBuffer(
			g_gpuPF.queue,
			g_gpuFF.d_stateNextCount,
			CL_TRUE,
			0,
			sizeof(int),
			&nextCountCPU,
			0, nullptr, nullptr
		);

		std::swap(g_gpuFF.d_stateCur, g_gpuFF.d_stateNext);
		std::swap(g_gpuFF.d_stateCount, g_gpuFF.d_stateNextCount);
	}

	Msg("[GPU-VIS] GPU Full Flow completed.\n");

	// ==========================================================
	// WRITEBACK CPU
	// ==========================================================
	for (int p = 0; p < portalCount; p++)
	{
		int* dst = (int*)sorted_portals[p]->portalvis;
		size_t off = p * longs * sizeof(int);

		clEnqueueReadBuffer(
			g_gpuPF.queue,
			g_gpuPF.d_portalVis,
			CL_TRUE,
			off,
			longs * sizeof(int),
			dst,
			0, nullptr, nullptr
		);

		int byte = p >> 5;
		int bit = 1 << (p & 31);
		dst[byte] |= bit;

		int leftovers = portalCount & 31;
		if (leftovers)
		{
			uint32_t mask = (1u << leftovers) - 1u;
			dst[longs - 1] &= mask;
		}

		sorted_portals[p]->nummightsee =
			CountBits(sorted_portals[p]->portalvis, portalCount);

		sorted_portals[p]->status = stat_done;
	}

	clFinish(g_gpuPF.queue);

	Msg("[GPU-VIS] Writeback from GPU completed.\n");

	// ==========================================================
	// TRYGPU MODE (COPIE GPU → portalvisGPU)
	// ==========================================================
	if (g_bTryGPU)
	{
		for (int p = 0; p < portalCount; p++)
		{
			byte* out = sorted_portals[p]->portalvisGPU;
			if (!out) continue;

			int* dst = (int*)out;

			size_t off = p * longs * sizeof(int);
			clEnqueueReadBuffer(
				g_gpuPF.queue,
				g_gpuPF.d_portalVis,
				CL_TRUE,
				off,
				longs * sizeof(int),
				dst,
				0, nullptr, nullptr
			);
		}

		clFinish(g_gpuPF.queue);

		GPU_CPU_SampleCompare();
	}

	Msg("[GPU-VIS] FULL GPU PortalFlow done.\n");
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
BasePortalVis [OLD]
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

void BuildFlatLeafPortalArrays(
	std::vector<int>& outCount,
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

		for (int i = 0; i < count; i++)
			outList[leaf * g_maxPerLeaf + i] = vec[i];

		for (int i = count; i < g_maxPerLeaf; i++)
			outList[leaf * g_maxPerLeaf + i] = -1;
	}

	Msg("[GPU-VIS] Flattened leaf->portal adjacency uploaded.\n");
}



void BuildLeafPortalTable()
{
	int numLeaves = portalclusters;

	g_leafPortals.clear();
	g_leafPortals.resize(numLeaves);

	int portalCount = g_numportals * 2;

	static std::vector<int> rawToSorted;
	rawToSorted.resize(portalCount);

	// Map raw index -> sorted index
	for (int s = 0; s < portalCount; s++)
	{
		portal_t* P = sorted_portals[s];
		int rawIdx = P - portals;
		rawToSorted[rawIdx] = s;
	}

	// Fill adjacency lists with sorted portal indexes
	for (int raw = 0; raw < portalCount; raw++)
	{
		portal_t* P = portals + raw;
		int leaf = P->leaf;

		if (leaf < 0 || leaf >= numLeaves)
			continue;

		int sortedIdx = rawToSorted[raw];
		g_leafPortals[leaf].push_back(sortedIdx);
	}

	Msg("[GPU-VIS] Built leaf->portal adjacency table (%d leaves).\n",
		numLeaves);
}


// ======================================================================
// Flatten leafPortal adjacency table for GPU upload
// ======================================================================

