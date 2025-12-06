//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose: GPU-accelerated portal flow (OpenCL)
// $NoKeywords: $
//=============================================================================//

#define CL_TARGET_OPENCL_VERSION 200
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

static std::mutex g_trace_mutex;
static std::ofstream g_trace_file;
static std::atomic<bool> g_trace_inited{ false };

// ========= GPU PRUNE TUNING =========

int g_gpuPreset = 2;

#include <CL/cl.h>


GPUPortalFlowCLContext g_gpuPF = {};
GPUPortalFlowFull g_gpuFF = {};

bool InitOpenCL_PortalFlow()
{
	if (g_gpuPF.initialized)
		return true;

	cl_int err = 0;

	// ===========================================================
	// 1) PLATFORM
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
	// 2) DEVICE (GPU -> fallback CPU)
	// ===========================================================
	cl_uint deviceCount = 0;
	err = clGetDeviceIDs(g_gpuPF.platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);

	if (err != CL_SUCCESS || deviceCount == 0)
	{
		Warning("[GPU-VIS] No GPU found, trying CPU device...\n");
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
	// 3) CONTEXT
	// ===========================================================
	g_gpuPF.context = clCreateContext(nullptr, 1, &g_gpuPF.device, nullptr, nullptr, &err);
	if (err != CL_SUCCESS || !g_gpuPF.context)
	{
		Warning("[GPU-VIS] Failed to create OpenCL context.\n");
		return false;
	}

	// ===========================================================
	// 4) COMMAND QUEUE
	// ===========================================================
#if defined(CL_VERSION_2_0)
	const cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, 0, 0 };
	g_gpuPF.queue = clCreateCommandQueueWithProperties(g_gpuPF.context, g_gpuPF.device, props, &err);
#else
	g_gpuPF.queue = clCreateCommandQueue(g_gpuPF.context, g_gpuPF.device, 0, &err);
#endif
	if (err != CL_SUCCESS || !g_gpuPF.queue)
	{
		Warning("[GPU-VIS] Failed to create OpenCL queue.\n");
		return false;
	}

	// ===========================================================
	// 5) BUILD PROGRAM (INLINE KERNELS)
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
		Warning("[GPU-VIS] Failed to create OpenCL program.\n");
		return false;
	}

	err = clBuildProgram(g_gpuPF.program, 1, &g_gpuPF.device, nullptr, nullptr, nullptr);

	// Always dump compilation log
	{
		size_t logSize = 0;
		clGetProgramBuildInfo(g_gpuPF.program, g_gpuPF.device,
			CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

		if (logSize > 1)
		{
			std::vector<char> log(logSize);
			clGetProgramBuildInfo(g_gpuPF.program, g_gpuPF.device,
				CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);

			Msg("[GPU-VIS] KERNEL BUILD LOG:\n%s\n", log.data());
		}
	}

	if (err != CL_SUCCESS)
	{
		Warning("[GPU-VIS] Kernel build failed.\n");
		return false;
	}

	g_gpuPF.k_separators = clCreateKernel(g_gpuPF.program, "separators", &err);
	if (err != CL_SUCCESS) Msg("[GPU-VIS] Failed kernel separators\n");
	else Msg("[GPU-VIS] Kernel separators OK.\n");

	g_gpuPF.k_flowIter = clCreateKernel(g_gpuPF.program, "portalFlowIter", &err);
	if (err != CL_SUCCESS) Msg("[GPU-VIS] Failed kernel portalFlowIter\n");
	else Msg("[GPU-VIS] Kernel portalFlowIter OK.\n");

	// (Pas de buffers alloués ici — patch 3 gérera selon pcount)

	g_gpuPF.initialized = true;

	Msg("[GPU-VIS] OpenCL PortalFlow initialized OK.\n");
	return true;
}

void ShutdownOpenCL_PortalFlow()
{
	if (!g_gpuPF.initialized)
		return;

#define REL(x) if(x){ clReleaseMemObject(x); x=nullptr; }

	REL(g_gpuPF.d_planes);
	REL(g_gpuPF.d_portalVis);

#undef REL
	if (g_gpuPF.k_flowIter) clReleaseKernel(g_gpuPF.k_flowIter);
	if (g_gpuPF.k_merge) clReleaseKernel(g_gpuPF.k_merge);
	if (g_gpuPF.k_separators) clReleaseKernel(g_gpuPF.k_separators);

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

    g_gpuPF.portalCount = g_numportals * 2;
    g_gpuPF.portalLongs = portallongs;

    int portalCount = g_gpuPF.portalCount;
    int longs = g_gpuPF.portalLongs;

    size_t maskBytes = portalCount * longs * sizeof(int);
    size_t originsBytes = portalCount * sizeof(float3);
    size_t radiusBytes  = portalCount * sizeof(float);
    size_t planesBytes  = portalCount * sizeof(float4);
    size_t windingBytes4 = portalCount * 4 * sizeof(float3);

    // ============
    // CLEAN PREVIOUS
    // ============
#define REL(x) if(x){ clReleaseMemObject(x); x=nullptr; }
    REL(g_gpuPF.d_portalVis);
    REL(g_gpuPF.d_origins);
    REL(g_gpuPF.d_radius);
    REL(g_gpuPF.d_planes);
    REL(g_gpuPF.d_winding4);
#undef REL

    // ============
    // CREATE BUFFERS
    // ============

    g_gpuPF.d_portalVis = clCreateBuffer(
        g_gpuPF.context,
        CL_MEM_READ_WRITE,
        maskBytes,
        nullptr,
        &err
    );
    if (err != CL_SUCCESS) { Warning("[GPU-VIS] FAILED alloc portalVis\n"); return false; }

    g_gpuPF.d_origins = clCreateBuffer(
        g_gpuPF.context,
        CL_MEM_READ_ONLY,
        originsBytes,
        nullptr,
        &err
    );
    if (err != CL_SUCCESS) { Warning("[GPU-VIS] FAILED alloc origins\n"); return false; }

    g_gpuPF.d_radius = clCreateBuffer(
        g_gpuPF.context,
        CL_MEM_READ_ONLY,
        radiusBytes,
        nullptr,
        &err
    );
    if (err != CL_SUCCESS) { Warning("[GPU-VIS] FAILED alloc radius\n"); return false; }

    g_gpuPF.d_planes = clCreateBuffer(
        g_gpuPF.context,
        CL_MEM_READ_ONLY,
        planesBytes,
        nullptr,
        &err
    );
    if (err != CL_SUCCESS) { Warning("[GPU-VIS] FAILED alloc planes\n"); return false; }

    g_gpuPF.d_winding4 = clCreateBuffer(
        g_gpuPF.context,
        CL_MEM_READ_ONLY,
        windingBytes4,
        nullptr,
        &err
    );
    if (err != CL_SUCCESS) { Warning("[GPU-VIS] FAILED alloc winding4\n"); return false; }

	// === FULL GPU FLOW BUFFERS ===
	size_t frontierBytes = g_gpuPF.portalCount * sizeof(int);
	size_t visBytes = g_gpuPF.portalCount * g_gpuPF.portalLongs * sizeof(int);

	cl_int errF = 0;

	g_gpuFF.d_frontier = clCreateBuffer(
		g_gpuPF.context, CL_MEM_READ_WRITE, frontierBytes, nullptr, &errF
	);
	g_gpuFF.d_nextFrontier = clCreateBuffer(
		g_gpuPF.context, CL_MEM_READ_WRITE, frontierBytes, nullptr, &errF
	);
	g_gpuFF.d_changed = clCreateBuffer(
		g_gpuPF.context, CL_MEM_READ_WRITE, sizeof(int), nullptr, &errF
	);
	g_gpuFF.d_mightSee = clCreateBuffer(
		g_gpuPF.context, CL_MEM_READ_WRITE, visBytes, nullptr, &errF
	);

	if (errF != CL_SUCCESS)
	{
		Warning("[GPU-VIS] FAILED allocating full-flow BFS buffers.\n");
		return false;
	}


	// Init mask = full visible bits & mightSee = portalflood CPU
	for (int p = 0; p < portalCount; p++)
	{
		int* src = (int*)sorted_portals[p]->portalflood;
		clEnqueueWriteBuffer(
			g_gpuPF.queue,
			g_gpuPF.d_portalVis,
			CL_TRUE,
			p * longs * sizeof(int),
			longs * sizeof(int),
			src,
			0, nullptr, nullptr
		);

		clEnqueueWriteBuffer(
			g_gpuPF.queue,
			g_gpuFF.d_mightSee,
			CL_TRUE,
			p * longs * sizeof(int),
			longs * sizeof(int),
			src,
			0, nullptr, nullptr
		);
	}


    // ============
    // UPLOAD CPU -> GPU (one-time)
    // ============

    std::vector<float3> h_origins(portalCount);
    std::vector<float> h_radius(portalCount);
    std::vector<float4> h_planes(portalCount);
    std::vector<float3> h_winding4(portalCount * 4);

    for (int i = 0; i < portalCount; i++)
    {
        portal_t* p = sorted_portals[i];

        // origins
        h_origins[i] = float3{ p->origin.x, p->origin.y, p->origin.z };

        // radius
        h_radius[i] = p->radius;

        // plane
        h_planes[i] = float4{
            p->plane.normal.x,
            p->plane.normal.y,
            p->plane.normal.z,
            p->plane.dist
        };

        // extract 4 points from winding
        winding_t* w = p->winding;
        int wcount = w ? w->numpoints : 0;

        for (int k = 0; k < 4; k++)
        {
            int src = (k < wcount) ? k : (wcount ? 0 : 0);
            Vector& v = w ? w->points[src] : p->origin;

            h_winding4[i * 4 + k] = float3{
                v.x,
                v.y,
                v.z
            };
        }
    }

    // ---- Upload to GPU ----

    clEnqueueWriteBuffer(g_gpuPF.queue, g_gpuPF.d_origins,  CL_TRUE, 0, originsBytes,  h_origins.data(), 0,nullptr,nullptr);
    clEnqueueWriteBuffer(g_gpuPF.queue, g_gpuPF.d_radius,   CL_TRUE, 0, radiusBytes,   h_radius.data(),  0,nullptr,nullptr);
    clEnqueueWriteBuffer(g_gpuPF.queue, g_gpuPF.d_planes,   CL_TRUE, 0, planesBytes,   h_planes.data(),  0,nullptr,nullptr);
    clEnqueueWriteBuffer(g_gpuPF.queue, g_gpuPF.d_winding4, CL_TRUE, 0, windingBytes4, h_winding4.data(),0,nullptr,nullptr);

    Msg("[GPU-VIS] PortalFlow GPU buffers allocated & uploaded (%d portals).\n", portalCount);
    return true;
}

void GPU_CPU_SampleCompare()
{
	Msg("[TryGPU] Sample comparison disabled (stub).\n");
}


void PortalFlow_FullGPU()
{
	Msg("[GPU-VIS] FULL GPU Flood fill starting...\n");

	const int portalCount = g_gpuPF.portalCount;
	const int portalLongs = g_gpuPF.portalLongs;

	cl_int err;
	size_t gsz = portalCount;

	int changed = 1;
	int iter = 0;

	while (changed && iter < 128)   // 128 = sécurité
	{
		iter++;

		// Reset flag changed = 0
		changed = 0;
		clEnqueueWriteBuffer(
			g_gpuPF.queue,
			g_gpuPF.d_changed,
			CL_TRUE,
			0,
			sizeof(int),
			&changed,
			0, nullptr, nullptr
		);

		// Set kernel args (FULL GPU PVS)
		err = clSetKernelArg(g_gpuPF.k_flowIter, 0, sizeof(cl_mem), &g_gpuPF.d_portalVis);
		err |= clSetKernelArg(g_gpuPF.k_flowIter, 1, sizeof(cl_mem), &g_gpuFF.d_mightSee);
		err |= clSetKernelArg(g_gpuPF.k_flowIter, 2, sizeof(cl_mem), &g_gpuFF.d_changed);
		err |= clSetKernelArg(g_gpuPF.k_flowIter, 3, sizeof(int), &portalLongs);
		err |= clSetKernelArg(g_gpuPF.k_flowIter, 4, sizeof(int), &portalCount);

		if (err != CL_SUCCESS)
		{
			Warning("[GPU-VIS] Failed to set args for portalFlowIter\n");
			return;
		}

		// Run kernel
		clEnqueueNDRangeKernel(
			g_gpuPF.queue,
			g_gpuPF.k_flowIter,
			1, nullptr, &gsz, nullptr,
			0, nullptr, nullptr
		);

		clFinish(g_gpuPF.queue);

		// Read "changed"
		clEnqueueReadBuffer(
			g_gpuPF.queue,
			g_gpuPF.d_changed,
			CL_TRUE,
			0,
			sizeof(int),
			&changed,
			0, nullptr, nullptr
		);

		Msg("[GPU-VIS] Iteration %d complete (changed=%d)\n", iter, changed);
	}

	Msg("[GPU-VIS] FULL GPU flood fill finished in %d iterations.\n", iter);

	// ================================
// WRITEBACK GPU → CPU
// ================================
	for (int p = 0; p < portalCount; p++)
	{
		int* cpuVis = (int*)sorted_portals[p]->portalvis;
		size_t offset = p * portalLongs * sizeof(int);

		clEnqueueReadBuffer(
			g_gpuPF.queue,
			g_gpuPF.d_portalVis,
			CL_TRUE,
			offset,
			portalLongs * sizeof(int),
			cpuVis,
			0, nullptr, nullptr
		);

		sorted_portals[p]->nummightsee =
			CountBits(sorted_portals[p]->portalvis, portalCount);
	}


	// Mark all portals as done
	for (int i = 0; i < portalCount; i++)
		portals[i].status = stat_done;
}

void PortalFlow_GPU_Wrapper(int thread, int p)
{
	PortalFlow_GPU(p, sorted_portals[p]);
}

bool PortalFlow_GPU(int pIdx, portal_t* p)
{
	if (!g_gpuPF.initialized)
		return false;

	int portalCount = g_gpuPF.portalCount;
	int longs = g_gpuPF.portalLongs;
	size_t maskBytes = longs * sizeof(int);
	size_t maskOffset = pIdx * maskBytes;

	cl_int err = 0;

	// Set kernel args
	err = clSetKernelArg(g_gpuPF.k_separators, 0, sizeof(cl_mem), &g_gpuPF.d_origins);
	err |= clSetKernelArg(g_gpuPF.k_separators, 1, sizeof(cl_mem), &g_gpuPF.d_radius);
	err |= clSetKernelArg(g_gpuPF.k_separators, 2, sizeof(cl_mem), &g_gpuPF.d_planes);
	err |= clSetKernelArg(g_gpuPF.k_separators, 3, sizeof(cl_mem), &g_gpuPF.d_winding4);
	err |= clSetKernelArg(g_gpuPF.k_separators, 4, sizeof(cl_mem), &g_gpuPF.d_portalVis);
	err |= clSetKernelArg(g_gpuPF.k_separators, 5, sizeof(int), &pIdx);
	err |= clSetKernelArg(g_gpuPF.k_separators, 6, sizeof(int), &portalCount);
	err |= clSetKernelArg(g_gpuPF.k_separators, 7, sizeof(int), &longs);

	if (err != CL_SUCCESS)
	{
		Warning("[GPU-VIS] Failed to set kernel args\n");
		return false;
	}

	size_t gsz = portalCount;
	err = clEnqueueNDRangeKernel(g_gpuPF.queue, g_gpuPF.k_separators, 1, nullptr, &gsz, nullptr, 0, nullptr, nullptr);

	if (err != CL_SUCCESS)
	{
		Warning("[GPU-VIS] Kernel launch failed\n");
		return false;
	}

	clFinish(g_gpuPF.queue);

	// Read mask back
	std::vector<int> gpuMask(longs);
	err = clEnqueueReadBuffer(g_gpuPF.queue, g_gpuPF.d_portalVis, CL_TRUE, maskOffset, maskBytes, gpuMask.data(), 0, nullptr, nullptr);

	if (err != CL_SUCCESS)
	{
		Warning("[GPU-VIS] Failed to read prune mask\n");
		return false;
	}

	// Apply prune
	int* flood = (int*)p->portalflood;
	for (int i = 0; i < longs; i++)
		flood[i] &= gpuMask[i];

	p->nummightsee = CountBits(p->portalflood, portalCount);

	return true;
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

	memccpy(p->portalvis, p->portalflood, 0, portalbytes);

	//
	// test the given portal against all of the portals in the map
	//
	for (j = 0, tp = portals; j < g_numportals * 2; j++, tp++)
	{
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



/*
===============================================================================

This is a second order aproximation

Calculates portalvis bit vector

WAAAAAAY too slow.

===============================================================================
*/

/*
==================
RecursiveLeafBitFlow
[OLD]
==================

void RecursiveLeafBitFlow(int leafnum, byte* mightsee, byte* cansee)
{
	portal_t	*p;
	leaf_t 		*leaf;
	int			i, j;
	long		more;
	int			pnum;
	byte		newmight[MAX_PORTALS/8];

	leaf = &leafs[leafnum];

// check all portals for flowing into other leafs
	for (i=0 ; i<leaf->portals.Count(); i++)
	{
		p = leaf->portals[i];
		pnum = p - portals;

		// if some previous portal can't see it, skip
		if ( !CheckBit( mightsee, pnum ) )
			continue;

		// if this portal can see some portals we mightsee, recurse
		more = 0;
		for (j=0 ; j<portallongs ; j++)
		{
			((long *)newmight)[j] = ((long *)mightsee)[j]
				& ((long *)p->portalflood)[j];
			more |= ((long *)newmight)[j] & ~((long *)cansee)[j];
		}

		if (!more)
			continue;	// can't see anything new

		SetBit( cansee, pnum );

		RecursiveLeafBitFlow (p->leaf, newmight, cansee);
	}
}
*/
/*
==============
BetterPortalVis [OLD]
==============


void BetterPortalVis (int portalnum)
{
	portal_t	*p;

	p = portals+portalnum;

	RecursiveLeafBitFlow (p->leaf, p->portalflood, p->portalvis);

	// build leaf vis information
	p->nummightsee = CountBits (p->portalvis, g_numportals*2);
	c_vis += p->nummightsee;
}
*/
