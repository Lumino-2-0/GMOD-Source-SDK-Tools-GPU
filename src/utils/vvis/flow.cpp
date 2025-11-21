//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose: GPU-accelerated portal flow (OpenCL)
// $NoKeywords: $
//=============================================================================//

#define CL_TARGET_OPENCL_VERSION 200
#include "vis.h"
#include "vmpi.h"
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

static std::mutex g_trace_mutex;
static std::ofstream g_trace_file;
static std::atomic<bool> g_trace_inited{ false };


// Déclaration globale partagée pour debug & kernels
std::vector<cl_float3> g_portal_origin;

// ========= GPU PRUNE TUNING =========

int g_gpuPreset = 2;


// actual parameters (remplis automatiquement selon preset)
float prune_min_radius = 8.0f;
float prune_backface_dot = -0.1f;
float prune_angle_dot = 0.25f;
float prune_convex_dot = 0.0f;
float prune_frustum_dot = -0.05f;

// Backface culling strict
static const float DOT_BACKFACE = 0.0f;          // cos(90°)

// Angle culling moderately strict
static const float DOT_ANGLE = -0.25f;           // cos(105°)

// Multi-wall convexity culling
static const float DOT_CONVEXITY = -0.15f;       // cos(98°)

// Tiny portal removal
static const float PORTAL_MIN_RADIUS = 12.0f;

// Portal frustum limits (universal)
static const float PORTAL_FRUSTUM_DOT_SIDE = -0.35f; // cos(110°)
static const float PORTAL_FRUSTUM_DOT_UP = -0.45f; // cos(117°)

// cluster fusion distance
static const float CLUSTER_MERGE_DIST = 64.0f;

static void RunTinyPortalPrune(int numportals, int portallongs, float minRadius)
{
	cl_kernel K = g_clManager.tiny_kernel;

	// param #4 = minRadius
	clSetKernelArg(K, 4, sizeof(float), &minRadius);

	size_t global = (size_t)numportals;
	cl_int err = clEnqueueNDRangeKernel(
		g_clManager.queue, K, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);

	if (err != CL_SUCCESS)
		Msg("[GPU] RunTinyPortalPrune failed (%d)\n", err);
	clFinish(g_clManager.queue);
}

static void RunBackfacePrune(int numportals, int portallongs, float dotMin)
{
	cl_kernel K = g_clManager.backface_kernel;

	clSetKernelArg(K, 5, sizeof(float), &dotMin);

	size_t global = (size_t)numportals;
	cl_int err = clEnqueueNDRangeKernel(
		g_clManager.queue, K, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);

	if (err != CL_SUCCESS)
		Msg("[GPU] RunBackfacePrune failed (%d)\n", err);
	clFinish(g_clManager.queue);
}

static void RunAnglePrune(int numportals, int portallongs, float dotMin)
{
	cl_kernel K = g_clManager.angle_kernel;

	clSetKernelArg(K, 4, sizeof(float), &dotMin);

	size_t global = (size_t)numportals;
	cl_int err = clEnqueueNDRangeKernel(
		g_clManager.queue, K, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);

	if (err != CL_SUCCESS)
		Msg("[GPU] RunAnglePrune failed (%d)\n", err);
	clFinish(g_clManager.queue);
}

static void RunConvexityPrune(int numportals, int portallongs, float dotMin)
{
	cl_kernel K = g_clManager.convexity_kernel;

	clSetKernelArg(K, 4, sizeof(float), &dotMin);

	size_t global = (size_t)numportals;
	cl_int err = clEnqueueNDRangeKernel(
		g_clManager.queue, K, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);

	if (err != CL_SUCCESS)
		Msg("[GPU] RunConvexityPrune failed (%d)\n", err);
	clFinish(g_clManager.queue);
}

static void RunFrustumPrune(int numportals, int portallongs, float dotMin)
{
	cl_kernel K = g_clManager.frustum_kernel;

	clSetKernelArg(K, 5, sizeof(float), &dotMin);

	size_t global = (size_t)numportals;
	cl_int err = clEnqueueNDRangeKernel(
		g_clManager.queue, K, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);

	if (err != CL_SUCCESS)
		Msg("[GPU] RunFrustumPrune failed (%d)\n", err);
	clFinish(g_clManager.queue);
}


inline void InitTrace()
{
	bool expected = false;
	if (g_trace_inited.compare_exchange_strong(expected, true)) {
		g_trace_file.open("vvis_gpu_trace.log", std::ios::out | std::ios::trunc);
	}
}

inline static void TracePrint(const char* fmt, ...)
{
	// Respecter flag global -debug (défini dans vvis.cpp / vis.h)
	extern bool g_bDebugMode;
	if (!g_bDebugMode) {
		return; // désactiver tout logging Trace si -debug non fourni
	}

	InitTrace();
	std::lock_guard<std::mutex> lk(g_trace_mutex);
	va_list ap;
	va_start(ap, fmt);
	char buf[2048];
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	// add timestamp
	auto now = std::chrono::system_clock::now();
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
	std::ostringstream oss;
	oss << "[" << ms << "] " << buf << "\n";
	std::string out = oss.str();
	// stdout
	std::fwrite(out.c_str(), 1, out.size(), stdout);
	fflush(stdout);
	// file
	if (g_trace_file.is_open()) {
		g_trace_file << out;
		g_trace_file.flush();
	}
}

struct TraceScope {
	const char* name;
	std::chrono::steady_clock::time_point t;
	TraceScope(const char* n) : name(n), t(std::chrono::steady_clock::now()) {
		TracePrint("ENTER %s", name);
	}
	~TraceScope() {
		auto d = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t).count();
		TracePrint("EXIT  %s (us=%lld)", name, (long long)d);
	}
};

#define TRACE_FN() TraceScope _trace_scope_obj(__FUNCTION__)
#define TRACE_MSG(fmt, ...) TracePrint(fmt, ##__VA_ARGS__)

// Helper de logging OpenCL : écrit via TracePrint si -debug, sinon sur cerr.
inline void CLCheckAndLog(cl_int err, const char* msg)
{
	extern bool g_bDebugMode; // défini dans vvis.cpp
	if (err != CL_SUCCESS) {
		if (g_bDebugMode) {
			TracePrint("[CL ERR] %s => %d", msg, (int)err);
		}
		else {
			std::cerr << "[CL ERR] " << msg << " => " << (int)err << std::endl;
		}
	}
	else {
		if (g_bDebugMode) {
			TracePrint("[CL OK]  %s", msg);
		}
	}
}
#define CL_CHECK_ERR(err, msg) CLCheckAndLog((err), (msg))

// Kernel OpenCL optimise (convergence device-side, logs via flags)
static const char* floodfill_kernel_src = R"CL(

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

// Constantes utilisees (adaptees via options de build)
#ifndef MAX_STACK_DEPTH
#define MAX_STACK_DEPTH 64
#endif

#ifndef MAX_PORTAL_LONGS
#define MAX_PORTAL_LONGS 1024
#endif

// Types correspondant aux structs C++ (doivent etre identiques a flow_gpu.h)
typedef struct { float normal[3]; float dist; } cl_plane_t;
typedef struct { int numpoints; float points[16][3]; } cl_winding_t;
typedef struct { cl_plane_t plane; int leaf; float origin[3]; float radius; int winding_idx; } cl_portal_t;
typedef struct { int first_portal; int num_portals; } cl_leaf_t;



__kernel void distance_prune(
    __global uint* portalvis,
    __global const float3* portal_origin,
    float maxDistSq,
    int numportals,
    int portallongs)
{
    int P = get_global_id(0);
    if (P >= numportals) return;

    float3 O = portal_origin[P];
    uint* outmask = portalvis + (P * portallongs);

    for (int j = 0; j < numportals; ++j)
    {
        float3 Oj = portal_origin[j];
        float dx = Oj.x - O.x;
        float dy = Oj.y - O.y;
        float dz = Oj.z - O.z;

        float d2 = dx*dx + dy*dy + dz*dz;
        if (d2 > maxDistSq)
        {
            int w = j >> 5;
            int b = j & 31;
            outmask[w] &= ~(1u << b);
        }
    }
}


__kernel void opposite_facing_prune(
    __global uint* portalvis,
    __global const float3* portal_normal,
    int numportals,
    int portallongs,
    float dotLimit)
{
    int P = get_global_id(0);
    if (P >= numportals) return;

    float3 N = portal_normal[P];
    uint* outmask = portalvis + P * portallongs;

    for (int j = 0; j < numportals; ++j)
    {
        float3 Nj = portal_normal[j];

        float dotv = N.x*Nj.x + N.y*Nj.y + N.z*Nj.z;
        if (dotv < dotLimit)
        {
            int w = j >> 5;
            int b = j & 31;
            outmask[w] &= ~(1u << b);
        }
    }
}

__kernel void sector_prune(
    __global uint* portalvis,
    __global const float3* portal_origin,
    __global const float3* portal_normal,
    int numportals,
    int portallongs,
    float angleLimit)
{
    int P = get_global_id(0);
    if (P >= numportals) return;

    float3 O = portal_origin[P];
    float3 N = portal_normal[P];

    uint* outmask = portalvis + P * portallongs;

    for (int j = 0; j < numportals; ++j)
    {
        float3 Oj = portal_origin[j];
        float3 dir = (float3)(Oj.x - O.x, Oj.y - O.y, Oj.z - O.z);

        float len = sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z) + 1e-5f;
        dir = dir / len;

        float dotv = N.x*dir.x + N.y*dir.y + N.z*dir.z;

        if (dotv < angleLimit)
        {
            int w = j >> 5;
            int b = j & 31;
            outmask[w] &= ~(1u << b);
        }
    }
}



// ===================================================
// 1. TINY PORTAL PRUNE 
// ===================================================
__kernel void tiny_portal_prune(
    __global uint* portalvis,
    __global const float* portal_radius,
    int numportals,
    int portallongs,
    float min_radius
){
    int i = get_global_id(0);
    if (i >= numportals) return;

    int base = i * portallongs;

    for (int j = 0; j < numportals; ++j)
    {
        if (portal_radius[j] < min_radius)
        {
            int w = j >> 5;
            uint bit = 1u << (j & 31);
            portalvis[base + w] &= ~bit;
        }
    }
}



// ===================================================
// 2. BACKFACE PRUNE 
// ===================================================
__kernel void backface_prune(
    __global uint* portalvis,
    __global const float3* portal_origin,
    __global const float3* portal_normal,
    int numportals,
    int portallongs,
    float dot_threshold
){
    int i = get_global_id(0);
    if (i >= numportals) return;

    int base = i * portallongs;

    float3 Oi = portal_origin[i];
    float3 Ni = portal_normal[i];

    for (int j = 0; j < numportals; ++j)
    {
        int w = j >> 5;
        uint bit = 1u << (j & 31);
        uint m = portalvis[base + w];
        if (!(m & bit)) continue;

        float3 Dir = portal_origin[j] - Oi;
        float len = length(Dir);
        if (len < 0.001f) { continue; }
        Dir /= len;

        float dotv = dot(Ni, Dir);

        if (dotv < dot_threshold) 
        {
            portalvis[base + w] = m & ~bit;
        }
    }
}



// ===================================================
// 3. ANGLE PRUNE 
// ===================================================
__kernel void angle_prune(
    __global uint* portalvis,
    __global const float3* portal_normal,
    int numportals,
    int portallongs,
    float dot_threshold
){
    int i = get_global_id(0);
    if (i >= numportals) return;

    int base = i * portallongs;
    float3 Ni = portal_normal[i];

    for (int j = 0; j < numportals; ++j)
    {
        int w = j >> 5;
        uint bit = 1u << (j & 31);
        uint m = portalvis[base + w];
        if (!(m & bit)) continue;

        float dotv = dot(Ni, portal_normal[j]);

        if (dotv < dot_threshold) 
            portalvis[base + w] = m & ~bit;
    }
}



// ===================================================
// 4. CONVEXITY PRUNE 
//     (Double-wall occlusion heuristic)
// ===================================================
__kernel void convexity_prune(
    __global uint* portalvis,
    __global const float3* portal_normal,
    int numportals,
    int portallongs,
    float dot_threshold
){
    int i = get_global_id(0);
    if (i >= numportals) return;

    int base_i = i * portallongs;
    float3 Ni = portal_normal[i];

    for (int j = 0; j < numportals; ++j)
    {
        int wj = j >> 5;
        uint bitj = 1u << (j & 31);
        uint mj = portalvis[base_i + wj];
        if (!(mj & bitj)) continue;

        float3 Nj = portal_normal[j];

        // Search any K that blocks I->J via convexity
        for (int k = 0; k < numportals; ++k)
        {
            if (k == i || k == j) continue;

            float3 Nk = portal_normal[k];

            float d1 = dot(Ni, Nk);
            float d2 = dot(Nk, Nj);

            if (d1 < dot_threshold && d2 < dot_threshold)
            {
                portalvis[base_i + wj] = mj & ~bitj;
                break;
            }
        }
    }
}




// ===================================================
// 5. FRUSTUM PRUNE 
//     Simple directional frustum per-portal
// ===================================================
__kernel void frustum_prune(
    __global uint* portalvis,
    __global const float3* portal_origin,
    __global const float3* portal_normal,
    int numportals,
    int portallongs,
    float dot_threshold
){
    int i = get_global_id(0);
    if (i >= numportals) return;

    int base_i = i * portallongs;

    float3 Oi = portal_origin[i];
    float3 Ni = portal_normal[i];

    for (int j = 0; j < numportals; ++j)
    {
        int wj = j >> 5;
        uint bitj = 1u << (j & 31);
        uint mj = portalvis[base_i + wj];
        if (!(mj & bitj)) continue;

        float3 Dir = portal_origin[j] - Oi;
        float len = length(Dir);
        if (len < 0.001f) continue;
        Dir /= len;

        float d = dot(Ni, Dir);
        if (d < dot_threshold)
            portalvis[base_i + wj] = mj & ~bitj;
    }
}


__kernel void z_occlusion_prune(
    __global uint* portalvis,
    __global const float3* portal_origin,
    __global const float3* portal_normal,
    int numportals,
    int portallongs,
    float zLimit)
{
    int P = get_global_id(0);
    if (P >= numportals) return;

    float3 O = portal_origin[P];
    float3 N = portal_normal[P];

    uint* outmask = portalvis + P * portallongs;

    for (int j = 0; j < numportals; ++j)
    {
        float3 D = portal_origin[j] - O;
        float d = dot(D, N); // profondeur dans l’axe du portail

        if (d > zLimit)   // Trop profond = invisible
        {
            int w = j >> 5;
            int b = j & 31;
            outmask[w] &= ~(1u << b);
        }
    }
}

__kernel void solid_angle_prune(
    __global uint* portalvis,
    __global const float3* portal_origin,
    __global const float* portal_radius,
    int numportals,
    int portallongs,
    float minAngle)
{
    int P = get_global_id(0);
    if (P >= numportals) return;

    float3 O = portal_origin[P];
    uint* outmask = portalvis + P*portallongs;

    for (int j = 0; j < numportals; j++)
    {
        float3 D = portal_origin[j] - O;
        float d = length(D) + 0.0001f;
        float angle = portal_radius[j] / d;

        if (angle < minAngle)
        {
            int w = j >> 5;
            int b = j & 31;
            outmask[w] &= ~(1u << b);
        }
    }
}

__kernel void pyramid_sector_prune(
    __global uint* portalvis,
    __global const float3* portal_origin,
    __global const float3* portal_normal,
    int numportals,
    int portallongs,
    float cosLimit)
{
    int P = get_global_id(0);
    if (P >= numportals) return;

    float3 O = portal_origin[P];
    float3 N = portal_normal[P];
    uint* outmask = portalvis + P * portallongs;

    for (int j = 0; j < numportals; j++)
    {
        float3 D = normalize(portal_origin[j] - O);
        float c = dot(N, D);

        if (c < cosLimit)
        {
            int w = j >> 5;
            int b = j & 31;
            outmask[w] &= ~(1u << b);
        }
    }
}




// Kernel de propagation de visibilité entre portails (une itération) NOUVEAU !
__kernel void pvs_leaf_propagate(
    __global const uint* portalflood,
    __global const int* leaf_first,
    __global const int* leaf_count,
    __global const int* leaf_portals,
    __global const int* portal_leaf,   // new
    __global uint* visleaf,            // [leafclusters][leaflongs]
    __global uint* next_visleaf,
    __global int* changed,
    int leafclusters,
    int leaflongs,
    int portallongs,
    int max_iters)   // <-- nouveau param : nombre d'itérations internes
{
    int L = get_global_id(0);
    if (L >= leafclusters) return;

    int lf = leaf_first[L];
    int lc = leaf_count[L];

    // pointeur vers le bitset de la leaf courante (in-place)
    __global uint* vis = visleaf + L * leaflongs;

    // boucle interne : plusieurs itérations de propagation dans un seul lancement
    for (int iter = 0; iter < max_iters; ++iter)
    {
        int local_changed = 0;

        // Pour chaque portail dans cette leaf
        for (int i = 0; i < lc; ++i)
        {
            int P = leaf_portals[lf + i];

            // Pour chaque mot de portalflood[P]
            for (int w = 0; w < portallongs; ++w)
            {
                uint mask = portalflood[P * portallongs + w];
                if (!mask) continue;

                // pour chaque bit dans le mot
                uint baseIndex = (uint)w << 5;
                for (int b = 0; b < 32; ++b)
                {
                    uint bit = (1u << b);
                    if (!(mask & bit)) continue;

                    int P2 = (int)(baseIndex + b);
                    int L2 = portal_leaf[P2];
                    int dstWord = L2 >> 5;
                    uint bitmask = 1u << (L2 & 31);

                    __global uint* dstAddr = vis + dstWord;

                    // écriture monotone via atomic_or pour permettre converger en place
                    uint old = atomic_or((volatile __global uint*)dstAddr, bitmask);
                    if (!(old & bitmask))
                        local_changed = 1;
                }
            }
        }

        // si rien de nouveau dans cette itération, on peut sortir tôt
        if (!local_changed)
            break;

        // informer le host (optionnel) qu'un changement est intervenu
        atomic_or((volatile __global int*)changed, 1);
        // continue next iter to let propagation ripple further within device
    }
}
)CL";

// Gestionnaire OpenCL (singleton)
OpenCLManager g_clManager;


cl_mem d_portal_origin = nullptr;
cl_mem d_portal_normal = nullptr;
cl_mem d_portal_radius = nullptr;

// Initialisation OpenCL
void OpenCLManager::init_once() {
	TRACE_FN();
	TRACE_MSG("OpenCLManager:: starting ! ");
	std::lock_guard<std::mutex> lock(init_mutex);
	if (ok) return;

	cl_int err = CL_SUCCESS;
	// 1. Choisir la plateforme et le device GPU
	err = clGetPlatformIDs(1, &platform, nullptr);
	CL_CHECK_ERR(err, "clGetPlatformIDs");
	if (err != CL_SUCCESS) {
		std::cerr << "[OpenCL|GPU-Mod] Plateforme OpenCL introuvable, fallback CPU. (veuillez verifier que vous avez les pilotes OpenCL ou une carte graphique compatible :/ )\n";
		ok = false;
		return;
	}

	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
	CL_CHECK_ERR(err, "clGetDeviceIDs");
	if (err != CL_SUCCESS) {
		std::cerr << "[OpenCL|GPU-Mod] Aucun device GPU OpenCL trouve, fallback CPU.\n";
		ok = false;
		return;
	}

	// 2. Creer contexte et queue
	context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
	CL_CHECK_ERR(err, "clCreateContext");
	if (err != CL_SUCCESS) {
		std::cerr << "[OpenCL|GPU-Mod] echec creation contexte, fallback CPU.\n";
		ok = false;
		return;
	}

	queue = clCreateCommandQueue(context, device, 0, &err);
	CL_CHECK_ERR(err, "clCreateCommandQueue");
	if (err != CL_SUCCESS) {
		std::cerr << "[OpenCL|GPU-Mod] echec creation queue, fallback CPU.\n";
		ok = false;
		return;
	}

	// 3. Compiler le programme OpenCL
	const char* kernelSrc = floodfill_kernel_src;
	program = clCreateProgramWithSource(context, 1, &kernelSrc, nullptr, &err);
	CL_CHECK_ERR(err, "clCreateProgramWithSource");
	if (err != CL_SUCCESS) {
		std::cerr << "[OpenCL|GPU-Mod] Erreur clCreateProgramWithSource, fallback CPU.\n";
		ok = false;
		return;
	}

	std::ostringstream options;
	options << "-DMAX_PORTAL_LONGS=" << portallongs;
	options << " -DMAX_STACK_DEPTH=" << 4;
	options << " -DMAX_POINTS_ON_FIXED_WINDING=" << MAX_POINTS_ON_FIXED_WINDING;
	err = clBuildProgram(program, 1, &device, options.str().c_str(), nullptr, nullptr);
	if (err != CL_SUCCESS) {
		// Compilation echouee : obtenir log et fallback
		size_t log_size = 0;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
		std::vector<char> build_log(log_size ? log_size : 1);
		if (log_size)
			clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
		std::cerr << "[OpenCL|GPU-Mod] Erreur compilation OpenCL, fallback CPU.\n";
		std::cerr << (build_log.size() ? build_log.data() : std::string("No build log")) << "\n";
		ok = false;
		return;
	}
	CL_CHECK_ERR(err, "clBuildProgram");

	// 4. Creer le kernel
	floodfill_kernel = clCreateKernel(program, "pvs_leaf_propagate", &err);
	CL_CHECK_ERR(err, "clCreateKernel pvs_leaf_propagate");
	if (err != CL_SUCCESS) {
		std::cerr << "[OpenCL|GPU-Mod] Kernel introuvable ou corrompu (pvs_leaf_propagate), fallback CPU.\n";
		ok = false;
		return;
	}
	ok = true;
	TRACE_MSG("OpenCLManager::init_once() done, OK");

	// =====================================================================
	// CREATE PRUNE KERNELS (NO ARGS SET HERE)
	// =====================================================================
	g_clManager.tiny_kernel = clCreateKernel(program, "tiny_portal_prune", &err);
	CL_CHECK_ERR(err, "create tiny_portal_prune");

	g_clManager.backface_kernel = clCreateKernel(program, "backface_prune", &err);
	CL_CHECK_ERR(err, "create backface_prune");

	g_clManager.angle_kernel = clCreateKernel(program, "angle_prune", &err);
	CL_CHECK_ERR(err, "create angle_prune");

	g_clManager.convexity_kernel = clCreateKernel(program, "convexity_prune", &err);
	CL_CHECK_ERR(err, "create convexity_prune");

	g_clManager.frustum_kernel = clCreateKernel(program, "frustum_prune", &err);
	CL_CHECK_ERR(err, "create frustum_prune");

	g_clManager.leaf_kernel = clCreateKernel(program, "pvs_leaf_propagate", &err);
	CL_CHECK_ERR(err, "create pvs_leaf_propagate");

	g_clManager.distance_prune_kernel = clCreateKernel(program, "distance_prune", &err);
	CL_CHECK_ERR(err, "create distance_prune");

	g_clManager.opposite_prune_kernel = clCreateKernel(program, "opposite_facing_prune", &err);
	CL_CHECK_ERR(err, "create opposite_facing_prune");

	g_clManager.sector_prune_kernel = clCreateKernel(program, "sector_prune", &err);
	CL_CHECK_ERR(err, "create sector_prune");

	g_clManager.z_occlusion_kernel = clCreateKernel(program, "z_occlusion_prune", &err);
	CL_CHECK_ERR(err, "create z_occlusion_prune");

	g_clManager.solid_angle_kernel = clCreateKernel(program, "solid_angle_prune", &err);
	CL_CHECK_ERR(err, "create solid_angle_prune");

	g_clManager.pyramid_sector_kernel = clCreateKernel(program, "pyramid_sector_prune", &err);
	CL_CHECK_ERR(err, "create pyramid_sector_prune");

}

void OpenCLManager::cleanup() {
	if (floodfill_kernel) { clReleaseKernel(floodfill_kernel); floodfill_kernel = nullptr; }
	if (countbits_kernel) { clReleaseKernel(countbits_kernel); countbits_kernel = nullptr; }
	if (program) { clReleaseProgram(program); program = nullptr; }
	if (queue) { clReleaseCommandQueue(queue); queue = nullptr; }
	if (context) { clReleaseContext(context); context = nullptr; }
	platform = nullptr; device = nullptr; ok = false;
	std::cout << "[OpenCL|GPU-Mod] Nettoyage OpenCL termine.\n";
}

bool TinyPortal(const portal_t* P)
{
	return (P->radius < PORTAL_MIN_RADIUS);
}

bool Backface(const Vector& n, const Vector& dir)
{
	return (DotProduct(n, dir) < DOT_BACKFACE);
}

bool AngleCull(const Vector& A, const Vector& B)
{
	return (DotProduct(A, B) < DOT_ANGLE);
}

bool ConvexityCull(const Vector& A, const Vector& B)
{
	return (DotProduct(A, B) < DOT_CONVEXITY);
}

bool FrustumCull(const Vector& forward, const Vector& toPortal)
{
	float dot = DotProduct(forward, toPortal);
	return (dot < PORTAL_FRUSTUM_DOT_SIDE);
}


static void GPU_SetPreset(int mode)
{

	switch (mode)
	{
		// 0 = soft : quasi identique VVIS original
	case 0:
		prune_min_radius = 2.0f;
		prune_backface_dot = -0.8f;
		prune_angle_dot = 0.05f;
		prune_convex_dot = -0.5f;
		prune_frustum_dot = -0.9f;
		break;

		// 1 = normal (nos valeurs par défaut)
	case 1:
		prune_min_radius = 8.0f;
		prune_backface_dot = -0.1f;
		prune_angle_dot = 0.25f;
		prune_convex_dot = 0.0f;
		prune_frustum_dot = -0.05f;
		break;

		// 2 = aggressive (pour map semi-ouvertes)
	case 2:
		prune_min_radius = 16.0f;
		prune_backface_dot = 0.0f;
		prune_angle_dot = 0.35f;
		prune_convex_dot = 0.2f;
		prune_frustum_dot = 0.1f;
		break;

		// 3 = ultra (pour maps géantes, villes, Kindercity ❤️)
	case 3:
		prune_min_radius = 24.0f;
		prune_backface_dot = 0.15f;
		prune_angle_dot = 0.45f;
		prune_convex_dot = 0.35f;
		prune_frustum_dot = 0.25f;
		break;
	}

	Msg("[GPU-VIS] Preset %d loaded\n", mode);
}



static inline bool GeometricOcclusionCull(
	const Vector& originA, const Vector& normalA,
	const Vector& originB, const Vector& normalB,
	float distAB, float dotAB,
	int preset)
{
	// RULE 1 : Si les portails se tournent le dos (fort)
	if (dotAB < -0.25f) return true;

	// RULE 2 : Si distance trop grande selon preset
	if (preset >= 2 && distAB > 2000.0f) return true;
	if (preset >= 3 && distAB > 900.0f)  return true;

	// RULE 3 : L’un est “derrière” le plan de l’autre
	float dA = DotProduct(normalB, originA - originB);
	float dB = DotProduct(normalA, originB - originA);

	if (preset >= 1 && (dA < -20.0f || dB < -20.0f))
		return true;

	// RULE 4 : Angle trop fermé (porte intérieure)
	if (preset >= 2)
	{
		if (dotAB < 0.10f) return true;
	}

	// RULE 5 : Ultra aggressive : coupe tout ce qui n’est pas quasi-aligné
	if (preset >= 3)
	{
		if (dotAB < 0.35f) return true;
		if (fabs(dA) > 60.0f || fabs(dB) > 60.0f) return true;
	}

	return false;
}


void GPU_ZOcclusionPrune(cl_mem d_portalvis,
	cl_mem d_portal_origin,
	cl_mem d_portal_normal,
	int numportals, int portallongs)
{
	float limit = 4096.0f;

	cl_kernel k = g_clManager.z_occlusion_kernel;

	clSetKernelArg(k, 0, sizeof(cl_mem), &d_portalvis);
	clSetKernelArg(k, 1, sizeof(cl_mem), &d_portal_origin);
	clSetKernelArg(k, 2, sizeof(cl_mem), &d_portal_normal);
	clSetKernelArg(k, 3, sizeof(int), &numportals);
	clSetKernelArg(k, 4, sizeof(int), &portallongs);
	clSetKernelArg(k, 5, sizeof(float), &limit);

	size_t global = numportals;
	clEnqueueNDRangeKernel(g_clManager.queue, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
	clFinish(g_clManager.queue);
}

void GPU_SolidAnglePrune(cl_mem d_portalvis,
	cl_mem d_portal_origin,
	cl_mem d_portal_radius,
	int numportals, int portallongs)
{
	float minA = 0.005f;

	cl_kernel k = g_clManager.solid_angle_kernel;

	clSetKernelArg(k, 0, sizeof(cl_mem), &d_portalvis);
	clSetKernelArg(k, 1, sizeof(cl_mem), &d_portal_origin);
	clSetKernelArg(k, 2, sizeof(cl_mem), &d_portal_radius);
	clSetKernelArg(k, 3, sizeof(int), &numportals);
	clSetKernelArg(k, 4, sizeof(int), &portallongs);
	clSetKernelArg(k, 5, sizeof(float), &minA);

	size_t global = numportals;
	clEnqueueNDRangeKernel(g_clManager.queue, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
	clFinish(g_clManager.queue);
}

void GPU_PyramidSectorPrune(cl_mem d_portalvis,
	cl_mem d_portal_origin,
	cl_mem d_portal_normal,
	int numportals, int portallongs)
{
	float cosLimit = 0.2f;  // 78°

	cl_kernel k = g_clManager.pyramid_sector_kernel;

	clSetKernelArg(k, 0, sizeof(cl_mem), &d_portalvis);
	clSetKernelArg(k, 1, sizeof(cl_mem), &d_portal_origin);
	clSetKernelArg(k, 2, sizeof(cl_mem), &d_portal_normal);
	clSetKernelArg(k, 3, sizeof(int), &numportals);
	clSetKernelArg(k, 4, sizeof(int), &portallongs);
	clSetKernelArg(k, 5, sizeof(float), &cosLimit);

	size_t global = numportals;
	clEnqueueNDRangeKernel(g_clManager.queue, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
	clFinish(g_clManager.queue);
}


void GPU_DistancePrune(cl_mem d_portalvis, cl_mem d_portal_origin,
	int numportals, int portallongs)
{
	float maxDist = 20000.0f; // règle à ajuster
	float maxDistSq = maxDist * maxDist;

	cl_kernel k = g_clManager.distance_prune_kernel;

	clSetKernelArg(k, 0, sizeof(cl_mem), &d_portalvis);
	clSetKernelArg(k, 1, sizeof(cl_mem), &d_portal_origin);
	clSetKernelArg(k, 2, sizeof(float), &maxDistSq);
	clSetKernelArg(k, 3, sizeof(int), &numportals);
	clSetKernelArg(k, 4, sizeof(int), &portallongs);

	size_t global = numportals;
	clEnqueueNDRangeKernel(g_clManager.queue, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
	clFinish(g_clManager.queue);
}


void GPU_OppositeFacingPrune(cl_mem d_portalvis, cl_mem d_portal_normal,
	int numportals, int portallongs)
{
	float dotLimit = -0.3f;  // portails opposés strictement

	cl_kernel k = g_clManager.opposite_prune_kernel;

	clSetKernelArg(k, 0, sizeof(cl_mem), &d_portalvis);
	clSetKernelArg(k, 1, sizeof(cl_mem), &d_portal_normal);
	clSetKernelArg(k, 2, sizeof(int), &numportals);
	clSetKernelArg(k, 3, sizeof(int), &portallongs);
	clSetKernelArg(k, 4, sizeof(float), &dotLimit);

	size_t global = numportals;
	clEnqueueNDRangeKernel(g_clManager.queue, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
	clFinish(g_clManager.queue);
}

void GPU_SectorPrune(cl_mem d_portalvis,
	cl_mem d_portal_origin,
	cl_mem d_portal_normal,
	int numportals, int portallongs)
{
	float angleLimit = -0.1f; // 90° max

	cl_kernel k = g_clManager.sector_prune_kernel;

	clSetKernelArg(k, 0, sizeof(cl_mem), &d_portalvis);
	clSetKernelArg(k, 1, sizeof(cl_mem), &d_portal_origin);
	clSetKernelArg(k, 2, sizeof(cl_mem), &d_portal_normal);
	clSetKernelArg(k, 3, sizeof(int), &numportals);
	clSetKernelArg(k, 4, sizeof(int), &portallongs);
	clSetKernelArg(k, 5, sizeof(float), &angleLimit);

	size_t global = numportals;
	clEnqueueNDRangeKernel(g_clManager.queue, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
	clFinish(g_clManager.queue);
}



// Flood fill global sur GPU avec convergence
void MassiveFloodFillGPU()
{
	TRACE_FN();

	// choose preset

	GPU_SetPreset(g_gpuPreset); // g_gpuPreset = défini via l'argument -PresetGPU

	assert(g_clManager.ok && "OpenCL non initialisé !");

	const int numportals = g_numportals * 2;
	const int portallongs = ::portallongs;
	const size_t totalWords = (size_t)numportals * portallongs;

	// ============================================================
	// Build flat arrays
	// ============================================================
	std::vector<uint> portalflood_flat(totalWords);
	std::vector<uint> portalvis_flat(totalWords, 0);
	std::vector<uint> frontier_flat(totalWords);
	std::vector<uint> next_frontier_flat(totalWords, 0);

	for (int i = 0; i < numportals; ++i)
	{
		uint* src = (uint*)portals[i].portalflood;
		for (int j = 0; j < portallongs; ++j)
		{
			portalflood_flat[i * portallongs + j] = src[j];
			frontier_flat[i * portallongs + j] = src[j];   // initial frontier = flood
		}
	}

	// ============================================================
	// OpenCL buffers
	// ============================================================
	cl_int err = CL_SUCCESS;

	cl_mem d_portalflood = clCreateBuffer(g_clManager.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint) * totalWords, portalflood_flat.data(), &err);
	cl_mem d_portalvis = clCreateBuffer(g_clManager.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(uint) * totalWords, portalvis_flat.data(), &err);
	cl_mem d_frontier = clCreateBuffer(g_clManager.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(uint) * totalWords, frontier_flat.data(), &err);
	cl_mem d_next_frontier = clCreateBuffer(g_clManager.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(uint) * totalWords, next_frontier_flat.data(), &err);
	cl_mem d_changed = clCreateBuffer(g_clManager.context, CL_MEM_READ_WRITE, sizeof(int), nullptr, &err);

	// PREPARE BUFFERS FOR PRUNE KERNELS
	g_portal_origin.resize(numportals);
	std::vector<cl_float3> portal_normal(numportals);
	std::vector<float>     portal_radius(numportals);

	for (int i = 0; i < numportals; ++i)
	{
		portal_t* P = &portals[i];

		g_portal_origin[i].x = (float)P->origin[0];
		g_portal_origin[i].y = (float)P->origin[1];
		g_portal_origin[i].z = (float)P->origin[2];

		portal_normal[i].x = (float)P->plane.normal[0];
		portal_normal[i].y = (float)P->plane.normal[1];
		portal_normal[i].z = (float)P->plane.normal[2];

		portal_radius[i] = P->radius;
	}

	g_clManager.buf_portal_origin = clCreateBuffer(g_clManager.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_float3) * numportals, g_portal_origin.data(), &err);
	g_clManager.buf_portal_normal = clCreateBuffer(g_clManager.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_float3) * numportals, portal_normal.data(), &err);
	g_clManager.buf_portal_radius = clCreateBuffer(g_clManager.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * numportals, portal_radius.data(), &err);

	g_clManager.buf_portalvis = d_portalvis; // IMPORTANT

	// ============================================================
	// BUILD LEAF -> PORTAL MAPPING (GPU VERSION)
	// ============================================================

	std::vector<int> leaf_first(portalclusters);
	std::vector<int> leaf_count(portalclusters);
	std::vector<int> leaf_portals;

	for (int L = 0; L < portalclusters; ++L)
	{
		leaf_first[L] = (int)leaf_portals.size();

		int cnt = leafs[L].portals.Count();
		leaf_count[L] = cnt;

		for (int i = 0; i < cnt; ++i)
		{
			// store portal index (portal_t pointer minus base pointer)
			leaf_portals.push_back(leafs[L].portals[i] - portals);
		}
	}

	// ========================= GPU BUFFERS =========================

	cl_mem d_leaf_first = clCreateBuffer(
		g_clManager.context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int) * portalclusters,
		leaf_first.data(),
		&err
	);

	cl_mem d_leaf_count = clCreateBuffer(
		g_clManager.context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int) * portalclusters,
		leaf_count.data(),
		&err
	);

	cl_mem d_leaf_portals = clCreateBuffer(
		g_clManager.context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int) * leaf_portals.size(),
		leaf_portals.data(),
		&err
	);

	// Buffer for portal → leaf mapping
	std::vector<int> portal_leaf(numportals);
	for (int i = 0; i < numportals; ++i)
		portal_leaf[i] = portals[i].leaf;

	cl_mem d_portal_leaf = clCreateBuffer(
		g_clManager.context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int) * numportals,
		portal_leaf.data(),
		&err
	);

	// Store for kernel usage
	g_clManager.buf_leaf_first = d_leaf_first;
	g_clManager.buf_leaf_count = d_leaf_count;
	g_clManager.buf_leaf_portals = d_leaf_portals;
	g_clManager.buf_portal_leaf = d_portal_leaf;

	// ============================
	// BIND PRUNE KERNEL ARGS HERE
	// ============================

	// TINY PORTAL PRUNE
	clSetKernelArg(g_clManager.tiny_kernel, 0, sizeof(cl_mem), &d_portalvis);
	clSetKernelArg(g_clManager.tiny_kernel, 1, sizeof(cl_mem), &g_clManager.buf_portal_radius);
	clSetKernelArg(g_clManager.tiny_kernel, 2, sizeof(int), &numportals);
	clSetKernelArg(g_clManager.tiny_kernel, 3, sizeof(int), &portallongs);

	// BACKFACE PRUNE
	clSetKernelArg(g_clManager.backface_kernel, 0, sizeof(cl_mem), &d_portalvis);
	clSetKernelArg(g_clManager.backface_kernel, 1, sizeof(cl_mem), &g_clManager.buf_portal_origin);
	clSetKernelArg(g_clManager.backface_kernel, 2, sizeof(cl_mem), &g_clManager.buf_portal_normal);
	clSetKernelArg(g_clManager.backface_kernel, 3, sizeof(int), &numportals);
	clSetKernelArg(g_clManager.backface_kernel, 4, sizeof(int), &portallongs);

	// ANGLE PRUNE
	clSetKernelArg(g_clManager.angle_kernel, 0, sizeof(cl_mem), &d_portalvis);
	clSetKernelArg(g_clManager.angle_kernel, 1, sizeof(cl_mem), &g_clManager.buf_portal_normal);
	clSetKernelArg(g_clManager.angle_kernel, 2, sizeof(int), &numportals);
	clSetKernelArg(g_clManager.angle_kernel, 3, sizeof(int), &portallongs);

	// CONVEXITY PRUNE
	clSetKernelArg(g_clManager.convexity_kernel, 0, sizeof(cl_mem), &d_portalvis);
	clSetKernelArg(g_clManager.convexity_kernel, 1, sizeof(cl_mem), &g_clManager.buf_portal_normal);
	clSetKernelArg(g_clManager.convexity_kernel, 2, sizeof(int), &numportals);
	clSetKernelArg(g_clManager.convexity_kernel, 3, sizeof(int), &portallongs);

	// FRUSTUM PRUNE
	clSetKernelArg(g_clManager.frustum_kernel, 0, sizeof(cl_mem), &d_portalvis);
	clSetKernelArg(g_clManager.frustum_kernel, 1, sizeof(cl_mem), &g_clManager.buf_portal_origin);
	clSetKernelArg(g_clManager.frustum_kernel, 2, sizeof(cl_mem), &g_clManager.buf_portal_normal);
	clSetKernelArg(g_clManager.frustum_kernel, 3, sizeof(int), &numportals);
	clSetKernelArg(g_clManager.frustum_kernel, 4, sizeof(int), &portallongs);

	// ============================================================
        // ============================================================
        // BFS PROPAGATION (CPU loop driving GPU)
        // ============================================================
	// Nombre réel de clusters (leafs)
	// Nombre réel de leafs
	const int leafclusters = portalclusters;

	// Nombre réel de portails (bits nécessaires dans leafvis)
	const int real_portal_count = g_numportals * 2;

	// Taille d’un bitfield leafvis en uint32_t
	const int leaflongs = (real_portal_count + 31) / 32;

	// Taille totale du buffer leafvis
	const size_t leafvis_size_bytes = (size_t)leafclusters * leaflongs * sizeof(uint32_t);

	// Alloue les bitfields CPU
        std::vector<uint32_t> leafvis_flat;

        leafvis_flat.resize((size_t)leafclusters* leaflongs);

        std::fill(leafvis_flat.begin(), leafvis_flat.end(), 0u);

	// Initial state: each leaf sees itself
	for (int L = 0; L < leafclusters; ++L)
	{
		leafvis_flat[L * leaflongs + (L >> 5)] |= (1u << (L & 31));
	}

	// GPU buffers
        cl_mem d_leafvis = clCreateBuffer(
                g_clManager.context,
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                leafvis_size_bytes,
                leafvis_flat.data(),
                &err
        );

        // Conserver un pointeur global pour la lecture finale (evite nullptr)
        g_clManager.buf_leafvis = d_leafvis;

	// Loop until convergence
	int changed = 1;
	cl_mem d_changed_leaf = clCreateBuffer(
		g_clManager.context,
		CL_MEM_READ_WRITE,
		sizeof(int),
		nullptr,
		&err
	);

	size_t leaf_global = leafclusters;

	while (changed)
	{
		changed = 0;
		clEnqueueWriteBuffer(
			g_clManager.queue,
			d_changed_leaf,
			CL_TRUE,
			0,
			sizeof(int),
			&changed,
			0,
			nullptr,
			nullptr
		);

		clSetKernelArg(g_clManager.leaf_kernel, 0, sizeof(cl_mem), &d_portalflood);
		clSetKernelArg(g_clManager.leaf_kernel, 1, sizeof(cl_mem), &g_clManager.buf_leaf_first);
		clSetKernelArg(g_clManager.leaf_kernel, 2, sizeof(cl_mem), &g_clManager.buf_leaf_count);
		clSetKernelArg(g_clManager.leaf_kernel, 3, sizeof(cl_mem), &g_clManager.buf_leaf_portals);
		clSetKernelArg(g_clManager.leaf_kernel, 4, sizeof(cl_mem), &g_clManager.buf_portal_leaf);
                clSetKernelArg(g_clManager.leaf_kernel, 5, sizeof(cl_mem), &d_leafvis);
                // kernel ecrit en place dans d_leafvis : passer le meme buffer pour next_visleaf
                clSetKernelArg(g_clManager.leaf_kernel, 6, sizeof(cl_mem), &d_leafvis);
		clSetKernelArg(g_clManager.leaf_kernel, 7, sizeof(cl_mem), &d_changed_leaf);
		clSetKernelArg(g_clManager.leaf_kernel, 8, sizeof(int), &leafclusters);
		clSetKernelArg(g_clManager.leaf_kernel, 9, sizeof(int), &leaflongs);
		clSetKernelArg(g_clManager.leaf_kernel, 10, sizeof(int), &portallongs);

		// ============================================================
		// PHASE 3 — BFS GPU MULTI-STEP PROPAGATION
		// ============================================================

		int changed_gpu = 1;
		cl_mem d_changed_gpu = clCreateBuffer(
			g_clManager.context,
			CL_MEM_READ_WRITE,
			sizeof(int),
			nullptr,
			&err
		);

		while (changed_gpu)
		{
			changed_gpu = 0;

			clEnqueueWriteBuffer(
				g_clManager.queue,
				d_changed_gpu,
				CL_TRUE,
				0,
				sizeof(int),
				&changed_gpu,
				0,
				nullptr,
				nullptr
			);

			size_t global = leafclusters;
			clSetKernelArg(g_clManager.leaf_kernel, 7, sizeof(cl_mem), &d_changed_gpu);

			// Ajuster max_iters en fonction du preset pour équilibrer overhead CPU↔GPU vs travail GPU
			int max_iters = 8;
			if (g_gpuPreset >= 3) max_iters = 16;
			else if (g_gpuPreset == 2) max_iters = 8;
			else max_iters = 4;

			clSetKernelArg(g_clManager.leaf_kernel, 11, sizeof(int), &max_iters);
			CL_CHECK_ERR(
				clEnqueueNDRangeKernel(g_clManager.queue, g_clManager.leaf_kernel,
					1, nullptr, &global, nullptr, 0, nullptr, nullptr),
				"launch pvs_leaf_propagate"
			);

			clFinish(g_clManager.queue);

			// Lire si quelque chose a changé
			clEnqueueReadBuffer(
				g_clManager.queue,
				d_changed_gpu,
				CL_TRUE,
				0,
				sizeof(int),
				&changed_gpu,
				0,
				nullptr,
				nullptr
			);
		}

		clEnqueueReadBuffer(
			g_clManager.queue,
			d_changed_leaf,
			CL_TRUE,
			0,
			sizeof(int),
			&changed,
			0,
			nullptr,
			nullptr
		);

                // Pas d'echange de buffers : la propagation se fait en place et
                // on conserve d_leafvis intact pour la lecture finale.
                clFinish(g_clManager.queue);
	}
	// ============================================================
	// Copy back final GPU result
	// ============================================================
	clEnqueueReadBuffer(g_clManager.queue, d_portalvis, CL_TRUE, 0, sizeof(uint) * totalWords, portalvis_flat.data(), 0, nullptr, nullptr);

	// ==============================
	// READ BACK leafvis FROM GPU
	// ==============================


	err = clEnqueueReadBuffer(
		g_clManager.queue,
		g_clManager.buf_leafvis,
		CL_TRUE,
		0,
		sizeof(uint) * leafclusters * leaflongs,
		leafvis_flat.data(),
		0,
		nullptr,
		nullptr
	);
	if (err != CL_SUCCESS) {
		printf("[ERROR] clEnqueueReadBuffer leafvis failed: %d\n", err);
	}
	else {
		printf("[DEBUG] leafvis buffer fetched from GPU.\n");
	}

	// ============================================================
	// GPU PRUNE PHASE (toujours exécuté)
	// ============================================================



	// TEMPORAIREMENT DÉSACTIVÉS POUR TEST SANS PRUNE !
	
	//RunTinyPortalPrune(numportals, portallongs, prune_min_radius); 
	//RunBackfacePrune(numportals, portallongs, prune_backface_dot);
	//RunAnglePrune(numportals, portallongs, prune_angle_dot);
	//RunConvexityPrune(numportals, portallongs, prune_convex_dot);
	//RunFrustumPrune(numportals, portallongs, prune_frustum_dot);


	//if (g_gpuPreset >= 1)  // activation pour preset >=1
	//{
	//	TRACE_MSG("GPU-PVS: Geometric Occlusion CPU-Pass...");

	//	for (int i = 0; i < numportals; ++i)
	//	{
	//		portal_t* A = &portals[i];
	//		Vector Ao = A->origin;
	//		Vector An = A->plane.normal;

	//		uint* visA = portalvis_flat.data() + i * portallongs;

	//		for (int j = 0; j < numportals; ++j)
	//		{
	//			if (i == j) continue;

	//			uint* visword = visA + (j >> 5);
	//			uint bit = 1u << (j & 31);
	//			if (!(*visword & bit)) continue;

	//			portal_t* B = &portals[j];
	//			Vector Bo = B->origin;
	//			Vector Bn = B->plane.normal;

	//			Vector AB = Bo - Ao;
	//			float distAB = VectorLength(AB);
	//			float dotAB = DotProduct(An, (Bo - Ao).Normalized());

	//			if (GeometricOcclusionCull(Ao, An, Bo, Bn, distAB, dotAB, g_gpuPreset))
	//			{
	//				*visword &= ~bit;
	//			}
	//		}
	//	}

	//	TRACE_MSG("Geometric Occlusion CPU-PASS DONE.");
	//}

	//// ============================================================
	//// CPU HARD PRUNE (désactivé pour preset >= 2)
	//// ============================================================
	//if (g_gpuPreset <= 1)
	//{
	//	TRACE_MSG("GPU-PVS: PRE-PROPAGATION PRUNE START");
	//	RunTinyPortalPrune(numportals, portallongs, prune_min_radius);
	//	RunBackfacePrune(numportals, portallongs, prune_backface_dot);
	//	RunAnglePrune(numportals, portallongs, prune_angle_dot);
	//	RunConvexityPrune(numportals, portallongs, prune_convex_dot);
	//	RunFrustumPrune(numportals, portallongs, prune_frustum_dot);
	//	GPU_DistancePrune(d_portalvis, g_clManager.buf_portal_origin, numportals, portallongs);
	//	GPU_OppositeFacingPrune(d_portalvis, g_clManager.buf_portal_normal, numportals, portallongs);
	//	GPU_SectorPrune(d_portalvis, g_clManager.buf_portal_origin, g_clManager.buf_portal_normal, numportals, portallongs);
	//	clFinish(g_clManager.queue);
	//	TRACE_MSG("GPU-PVS: PRE-PROPAGATION PRUNE END");


	//	TRACE_MSG("Leaf→Leaf propagation start (BFS-GPU)...");
	//	// launch kernel floodfill_kernel (already set up elsewhere)
	//	size_t global = numleafs;
	//	cl_kernel kernel = g_clManager.leaf_kernel;
	//	clSetKernelArg(kernel, 0, sizeof(cl_mem), &g_clManager.buf_leaf_first);
	//	clSetKernelArg(kernel, 1, sizeof(cl_mem), &g_clManager.buf_leaf_count);
	//	clSetKernelArg(kernel, 2, sizeof(cl_mem), &g_clManager.buf_leaf_portals);
	//	clSetKernelArg(kernel, 3, sizeof(cl_mem), &g_clManager.buf_portal_leaf);
	//	clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_portalflood);
	//	clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_leafvis);
	//	CL_CHECK_ERR(clEnqueueNDRangeKernel(g_clManager.queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "launch pvs_leaf_propagate");
	//	clFinish(g_clManager.queue);
	//	TRACE_MSG("Leaf→Leaf propagation done.");
	//}
	//else
	//{
	//	TRACE_MSG("Skipping HARD-HP PVS PRUNE for preset >= 2 (large maps)");
	//}


	TRACE_MSG("Cluster merge pass...");

	// ============================================================
	// PHASE 3 — CLUSTERMERGE V2
	// ============================================================

	for (int i = 0; i < numportals; i++)
	{
		for (int j = i + 1; j < numportals; j++)
		{
			// Distance < 64m
			float dx = g_portal_origin[i].s[0] - g_portal_origin[j].s[0];
			float dy = g_portal_origin[i].s[1] - g_portal_origin[j].s[1];
			float dz = g_portal_origin[i].s[2] - g_portal_origin[j].s[2];

			float d2 = dx * dx + dy * dy + dz * dz;

			if (d2 < 4096.0f * 4096.0f)
			{
				uint32_t* A = (uint32_t*)portals[i].portalvis;
				uint32_t* B = (uint32_t*)portals[j].portalvis;

				for (int w = 0; w < portallongs; w++)
				{
					uint32_t m = A[w] | B[w];
					A[w] = B[w] = m;
				}
			}
		}
	}

	TRACE_MSG("ClusterMergeV2 OK");


	// Read leafvis from GPU
	clEnqueueReadBuffer(
		g_clManager.queue,
		g_clManager.buf_leafvis,   // bon : buffer global
		CL_TRUE,
		0,
		leafvis_size_bytes,
		leafvis_flat.data(),
		0,
		nullptr,
		nullptr
	);

	for (int P = 0; P < numportals; ++P)
	{
		portal_t* pa = &portals[P];
		int Lsrc = pa->leaf;



		// Réinitialiser le portalvis
		if (!pa->portalvis) {
			pa->portalvis = (byte*)malloc(portalbytes);
		}
		memset(pa->portalvis, 0, portalbytes);

		// LeafVis GPU du leaf source
		uint32_t* leafbm = &leafvis_flat[Lsrc * leaflongs];

		// Pour chaque leaf visible depuis Lsrc
		for (int Ldst = 0; Ldst < leafclusters; ++Ldst)
		{
			if (!(leafbm[Ldst >> 5] & (1u << (Ldst & 31))))
				continue;

			int first = leaf_first[Ldst];
			int count = leaf_count[Ldst];

			// SAFE GUARD : skip leafs without portals
			if (count <= 0) continue;

			// SAFE GUARD : avoid overflow
			if (first < 0 || first + count >(int)leaf_portals.size())
			{
				printf("[GPU-WARN] leaf %d has invalid portal list (first=%d count=%d size=%d)\n",
					Ldst, first, count, (int)leaf_portals.size());
				continue;
			}

			for (int k = 0; k < count; ++k)
			{
				int Pdst = leaf_portals[first + k];

				// SAFE GUARD : portal index must exist
				if (Pdst < 0 || Pdst >= numportals)
				{
					printf("[GPU-WARN] invalid Pdst=%d for leaf %d\n", Pdst, Ldst);
					continue;
				}

				// Ensure destination portal visibility buffer exists
				if (!portals[Pdst].portalvis)
				{
					portals[Pdst].portalvis = (byte*)calloc(1, portalbytes);
				}

				// Marquer le portail distant comme visible
				SetBit(pa->portalvis, Pdst);

				// Fusion réciproque (important pour corriger mismatches)
				SetBit(portals[Pdst].portalvis, P);
			}
		}
		// Compter bits visibles pour debug
		int visible = CountBits(pa->portalvis, numportals);
		if (visible == 0)
			printf("[GPU-WARN] portal %d has 0 bits set!\n", P);
		else
			printf("[GPU-OK] portal %d : %d portals visible\n", P, visible);

		pa->nummightsee = CountBits(pa->portalvis, numportals);
		pa->status = stat_done;
	}

	TRACE_MSG("Leaf->Portal remap completed (PHASE 3 FIX)");



	TRACE_MSG("GPU PVS propagation completed.");
}

void GPU_CPU_SampleCompare()
{
	extern bool g_bTryGPU;
	if (!g_bTryGPU)
		return;

	TRACE_FN();

	int numportals = g_numportals * 2;
	int portallongs = ::portallongs;
	if (numportals <= 0 || portallongs <= 0) {
		Msg("[GPU Test] Aucun portail ou configuration invalide.\n");
		return;
	}

	// Mode exhaustif avec -TryGPUAll
	bool exhaustive = (CommandLine()->FindParm("-TryGPUAll") != 0);

	int sampleCount = exhaustive ? numportals : 32;
	if (sampleCount > numportals) sampleCount = numportals;
	int stride = exhaustive ? 1 : std::max(1, numportals / sampleCount);

	Msg("[GPU Test] Demarrage comparaison CPU vs GPU pour %d échantillons (stride=%d) %s\n",
		sampleCount, stride, exhaustive ? "(exhaustif)" : "");

	int mismatches = 0;
	int checked = 0;
	// Statistiques : compte de bits GPU globalement
	uint64_t totalBitsGPU = 0;
	uint64_t totalBitsCPU = 0;

	for (int s = 0, idx = 0; s < sampleCount; ++s, idx += stride) {
		if (idx >= numportals) idx = numportals - 1;

		portal_t* p = &portals[idx];
		if (!p) {
			Msg("[GPU Test] portail %d introuvable\n", idx);
			continue;
		}
		if (!p->portalvis) {
			Msg("[GPU Test] portail %d : portalvis non alloue — ignorer\n", idx);
			continue;
		}
		if (!p->portalflood) {
			Msg("[GPU Test] portail %d : portalflood non alloue — ignorer\n", idx);
			continue;
		}

		// Lire GPU bits (tel que stocké après MassiveFloodFillGPU)
		std::vector<uint32_t> gpu_bits(portallongs);
		for (int w = 0; w < portallongs; ++w) {
			gpu_bits[w] = ((uint32_t*)p->portalvis)[w];
		}

		// Sauvegarder portalflood pour dump si besoin
		std::vector<uint32_t> flood_bits(portallongs);
		for (int w = 0; w < portallongs; ++w) {
			flood_bits[w] = ((uint32_t*)p->portalflood)[w];
		}

		auto GetBitFromWords = [&](const std::vector<uint32_t>& words, int bit) -> bool {
			int word = bit >> 5;
			if (word < 0 || word >= (int)words.size()) {
				return false;
			}
			return (words[word] & (1u << (bit & 31))) != 0;
			};

		// Trouver l'indice dans sorted_portals correspondant à &portals[idx]
		int sortedIndex = -1;
		for (int si = 0; si < g_numportals * 2; ++si) {
			if (sorted_portals[si] == &portals[idx]) { sortedIndex = si; break; }
		}
		if (sortedIndex == -1) {
			Msg("[GPU Test] portail %d : introuvable dans sorted_portals, ignorer\n", idx);
			continue;
		}

		// Sauvegarder une copie GPU avant d'appeler PortalFlow (PortalFlow peut écrire p->portalvis)
		std::vector<uint32_t> gpu_before = gpu_bits;

		// Calcul CPU local pour ce portail (appelant PortalFlow_CPU)
		PortalFlow_CPU(0, sortedIndex);

		// Lire CPU bits (après PortalFlow)
		std::vector<uint32_t> cpu_bits(portallongs);
		for (int w = 0; w < portallongs; ++w) {
			cpu_bits[w] = ((uint32_t*)p->portalvis)[w];
		}

		// Compte bits pour stats
		auto CountBitsVector = [&](const std::vector<uint32_t>& v)->uint64_t {
			uint64_t c = 0;
			for (uint32_t x : v) c += (uint64_t)__popcnt(x);
			return c;
			};
		uint64_t gpuCount = CountBitsVector(gpu_before);
		uint64_t cpuCount = CountBitsVector(cpu_bits);
		totalBitsGPU += gpuCount;
		totalBitsCPU += cpuCount;

		// Comparer mot à mot
		bool equal = true;
		int first_diff_word = -1;
		for (int w = 0; w < portallongs; ++w) {
			if (cpu_bits[w] != gpu_before[w]) { equal = false; first_diff_word = w; break; }
		}

		++checked;
		if (equal) {
			Msg("[GPU Test] portail %6d : OK (bits GPU=%llu CPU=%llu)\n", idx, (unsigned long long)gpuCount, (unsigned long long)cpuCount);
		}
		else {
			++mismatches;
			Msg("[GPU Test] portail %6d : MISMATCH (premier mot diff = %d) GPUbits=%llu CPUbits=%llu\n",
				idx, first_diff_word, (unsigned long long)gpuCount, (unsigned long long)cpuCount);

			// ============================================================
			// TRYGPU PRO — DETAILS ÉTENDUS
			// ============================================================

			// Leaf source
			int leaf_src = p->leaf;

			// Liste des leafs visibles CPU/GPU
			std::vector<int> cpuLeafs;
			std::vector<int> gpuLeafs;

			for (int L = 0; L < portalclusters; ++L)
			{
				if (GetBitFromWords(cpu_bits, L)) cpuLeafs.push_back(L);
				if (GetBitFromWords(gpu_bits, L)) gpuLeafs.push_back(L);
			}

			printf("    >> Leaf source           : %d\n", leaf_src);
			printf("    >> Leaf visible CPU (%d) : ", (int)cpuLeafs.size());
			for (int L : cpuLeafs) printf("%d ", L);
			printf("\n");

			printf("    >> Leaf visible GPU (%d) : ", (int)gpuLeafs.size());
			for (int L : gpuLeafs) printf("%d ", L);
			printf("\n");

			// Mismatch par portail
			for (int Pmiss = 0; Pmiss < numportals; ++Pmiss)
			{
				bool c = GetBitFromWords(cpu_bits, Pmiss);
				bool g = GetBitFromWords(gpu_bits, Pmiss);

				if (c != g)
				{
					cl_float3 A = g_portal_origin[idx];
					cl_float3 B = g_portal_origin[Pmiss];

					float dx = B.x - A.x;
					float dy = B.y - A.y;
					float dz = B.z - A.z;

					float dist = sqrt(dx * dx + dy * dy + dz * dz);

					printf("    >> Diff portal %d  (CPU=%d GPU=%d)  dist=%.1f\n",
						Pmiss, (int)c, (int)g, dist);
				}
			}

			// Dump binaire pour analyse (GPU, CPU, portalflood)
			char fname[256];
			snprintf(fname, sizeof(fname), "pvis_gpu_%d.bin", idx);
			{
				std::ofstream f(fname, std::ios::binary);
				if (f.is_open()) f.write(reinterpret_cast<const char*>(gpu_before.data()), portallongs * sizeof(uint32_t));
			}
			snprintf(fname, sizeof(fname), "pvis_cpu_%d.bin", idx);
			{
				std::ofstream f(fname, std::ios::binary);
				if (f.is_open()) f.write(reinterpret_cast<const char*>(cpu_bits.data()), portallongs * sizeof(uint32_t));
			}
			snprintf(fname, sizeof(fname), "portalflood_%d.bin", idx);
			{
				std::ofstream f(fname, std::ios::binary);
				if (f.is_open()) f.write(reinterpret_cast<const char*>(flood_bits.data()), portallongs * sizeof(uint32_t));
			}

			// Print a small hex window around first difference for quick reading
			int start = std::max(0, first_diff_word - 4);
			int end = std::min(portallongs - 1, first_diff_word + 4);
			Msg("   mot#    GPU(hex)     CPU(hex)\n");
			for (int w = start; w <= end; ++w) {
				Msg("   %5d  %08x  %08x\n", w, gpu_before[w], cpu_bits[w]);
			}
			Msg("   Dumps écrits: pvis_gpu_%d.bin pvis_cpu_%d.bin portalflood_%d.bin\n", idx, idx, idx);
			// If not exhaustive, continue checking others; if exhaustive, still continue to produce full report.
		}

		// Restaurer la valeur GPU dans p->portalvis pour préserver l'état (important)
		for (int w = 0; w < portallongs; ++w) {
			((uint32_t*)p->portalvis)[w] = gpu_before[w];
		}
		p->status = stat_done;

		// Si on n'est pas en mode exhaustif et trop de mismatches, on s'arrête pour investigation
		if (!exhaustive && mismatches > 10) {
			Msg("[GPU Test] Trop de mismatches (%d) - arrêt precoce du test\n", mismatches);
			break;
		}
	}


	Msg("[GPU Test] Comparaison terminee : %d mismatches sur %d vérifiés\n", mismatches, checked);
	Msg("[GPU Test] Bits totaux (GPU=%llu CPU=%llu)\n", (unsigned long long)totalBitsGPU, (unsigned long long)totalBitsCPU);

	if (mismatches == 0) Msg("[GPU Test] Aucune difference detectee sur les echantillons.\n");
	else Msg("[GPU Test] %d differences detectees. Dumps générés pour les cas.\n", mismatches);

	TRACE_MSG("EXIT  GPU_CPU_SampleCompare");
}

// Compte le nombre de bits a 1 pour chaque portail (GPU)
void CountBitsGPU(std::vector<unsigned int>& portalvis_flat, std::vector<int>& out_counts, int numportals, int portallongs)
{
	g_clManager.init_once();
	assert(g_clManager.ok);

	cl_int err;
	cl_mem d_portalvis = clCreateBuffer(g_clManager.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int) * portalvis_flat.size(), portalvis_flat.data(), &err);
	cl_mem d_counts = clCreateBuffer(g_clManager.context, CL_MEM_WRITE_ONLY, sizeof(int) * numportals, nullptr, &err);

	cl_kernel kernel = g_clManager.countbits_kernel;
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_portalvis);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_counts);
	clSetKernelArg(kernel, 2, sizeof(int), &portallongs);

	size_t global = numportals;
	clEnqueueNDRangeKernel(g_clManager.queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
	clFinish(g_clManager.queue);
	TRACE_MSG("Kernel finished");


	// ============================
	// GPU PRUNE PIPELINE
	// ============================

	RunTinyPortalPrune(numportals, portallongs, prune_min_radius);
	RunBackfacePrune(numportals, portallongs, prune_backface_dot);
	RunAnglePrune(numportals, portallongs, prune_angle_dot);
	RunConvexityPrune(numportals, portallongs, prune_convex_dot);
	RunFrustumPrune(numportals, portallongs, prune_frustum_dot);

	clFinish(g_clManager.queue);
	TRACE_MSG("PRUNE FINISHED");

	clEnqueueReadBuffer(g_clManager.queue, d_counts, CL_TRUE, 0, sizeof(int) * numportals, out_counts.data(), 0, nullptr, nullptr);
	if (portalvis_flat.size() >= 2) {
		TRACE_MSG("Read portalvis_flat first 2 words: %08X %08X ...", portalvis_flat[0], portalvis_flat[1]);
	}
	else {
		TRACE_MSG("Read portalvis_flat size: %zu", portalvis_flat.size());
	}
	clReleaseMemObject(d_portalvis);
	clReleaseMemObject(d_counts);
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

#ifdef MPI
extern bool g_bVMPIEarlyExit;
#endif


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

#ifdef MPI
	// Early-out if we're a VMPI worker that's told to exit. If we don't do this here, then the
	// worker might spin its wheels for a while on an expensive work unit and not be available to the pool.
	// This is pretty common in vis.
	if (g_bVMPIEarlyExit)
		return;
#endif

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

		if (!CheckBit(thread->base->portalvis, pnum))
			continue;

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
// Version optimisee de PortalFlow
*/

void PortalFlow(int iThread, int portalnum)
{
	// Recuperation du portail courant (attention : sorted_portals !)
	portal_t* p = sorted_portals[portalnum];

	// Si portalvis non alloue (cas où GPU/host n'a pas encore rempli),
	// on le recouvre temporairement par portalflood pour éviter erreurs en aval.
	if (!p->portalvis && p->portalflood) {
		p->portalvis = p->portalflood;
	}

	// Marquer comme en cours puis termine (comportement attendu par le reste du pipeline)
	p->status = stat_working;

	int c_might = CountBits(p->portalflood, g_numportals * 2);
	int c_can = CountBits(p->portalvis, g_numportals * 2);

	int c_chains = 1;

	qprintf("portal:%4i  mightsee:%4i  cansee:%4i (%i chains)\n",
		(int)(p - portals), c_might, c_can, c_chains);

	// Indiquer que ce portail est traité
	p->status = stat_done;
}

void PortalFlow_CPU(int iThread, int portalnum)
{
	threaddata_t data;
	int i;
	portal_t* p;
	int c_might, c_can;

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

	// ============================================================
	// GPU ASSIST FILTER (Phase 2)
	// ============================================================
	uint32_t* gpuMask = (uint32_t*)p->portalvis;

	for (int j = 0; j < g_numportals * 2; ++j)
	{
		if (!CheckBit(gpuMask, j))
		{
			ClearBit(data.pstack_head.mightsee, j);
		}
	}

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
	Msg ("portal %i: %i mightsee\n", portalnum, p->nummightsee);
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


struct OpenCLCleanup {
	~OpenCLCleanup() { g_clManager.cleanup(); }
} g_OpenCLCleanup;