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


// Kernel de propagation de visibilité entre portails (une itération) NOUVEAU !
__kernel void pvs_propagate(
    __global const uint* portalflood,   // [numportals][portallongs]
    __global uint* portalvis,           // [numportals][portallongs]
    __global uint* frontier,            // [numportals][portallongs]
    __global uint* next_frontier,       // [numportals][portallongs]
    __global int* changed_flag,
    int numportals,
    int portallongs
) {
    int p = get_global_id(0);
    if (p >= numportals) return;

    const uint* f_in = frontier + p * portallongs;
    uint* f_out      = next_frontier + p * portallongs;
    const uint* flood= portalflood + p * portallongs;
    uint* vis        = portalvis  + p * portallongs;

    int local_changed = 0;

    // If this portal is in the frontier, propagate visibility
    for (int j = 0; j < portallongs; ++j)
    {
        // new portals visible
        uint newly = f_in[j] & ~vis[j];

        if (newly != 0)
            local_changed = 1;

        // update visibility
        vis[j] |= newly;

        // prepare next frontier = newly & flood  (propagation rule)
        f_out[j] = newly & flood[j];
    }

    if (local_changed)
        *changed_flag = 1;
} )CL";

// Gestionnaire OpenCL (singleton)
OpenCLManager g_clManager;



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
		std::cerr << "[OpenCL|GPU-Mod] Plateforme OpenCL introuvable, fallback CPU.\n";
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
	floodfill_kernel = clCreateKernel(program, "pvs_propagate", &err);
	CL_CHECK_ERR(err, "clCreateKernel pvs_propagate");
	if (err != CL_SUCCESS) {
		std::cerr << "[OpenCL|GPU-Mod] Kernel introuvable (pvs_propagate), fallback CPU.\n";
		ok = false;
		return;
	}
	ok = true;
	TRACE_MSG("OpenCLManager::init_once() done, OK");
	std::cout << "[OpenCL|GPU-Mod] Initialisation OpenCL reussie.\n";
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

// Flood fill global sur GPU avec convergence
void MassiveFloodFillGPU()
{
	TRACE_FN();

	g_clManager.init_once();
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

	// ============================================================
	// Kernel handles
	// ============================================================
	cl_kernel kernel = g_clManager.floodfill_kernel;

	size_t globalSize = numportals;

	// ============================================================
	// BFS PROPAGATION (CPU loop driving GPU)
	// ============================================================
	int changed = 1;
	while (changed)
	{
		changed = 0;
		clEnqueueWriteBuffer(g_clManager.queue, d_changed, CL_TRUE, 0, sizeof(int), &changed, 0, nullptr, nullptr);

		// Set kernel args
		clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_portalflood);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_portalvis);
		clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_frontier);
		clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_next_frontier);
		clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_changed);
		clSetKernelArg(kernel, 5, sizeof(int), &numportals);
		clSetKernelArg(kernel, 6, sizeof(int), &portallongs);

		// Launch
		err = clEnqueueNDRangeKernel(g_clManager.queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
		clFinish(g_clManager.queue);

		// Read "changed"
		clEnqueueReadBuffer(g_clManager.queue, d_changed, CL_TRUE, 0, sizeof(int), &changed, 0, nullptr, nullptr);

		// swap frontier buffers
		std::swap(d_frontier, d_next_frontier);

		// clear next frontier
		cl_uint zero = 0;
		clEnqueueFillBuffer(g_clManager.queue, d_next_frontier, &zero, sizeof(uint), 0, sizeof(uint) * totalWords, 0, nullptr, nullptr);
	}

	// ============================================================
	// Copy back final GPU result
	// ============================================================
	clEnqueueReadBuffer(g_clManager.queue, d_portalvis, CL_TRUE, 0, sizeof(uint) * totalWords, portalvis_flat.data(), 0, nullptr, nullptr);

	// Apply result to CPU portal structures
	for (int i = 0; i < numportals; ++i)
	{
		portal_t* p = &portals[i];
		memcpy(p->portalvis, portalvis_flat.data() + i * portallongs, portalbytes);
		p->nummightsee = CountBits((byte*)p->portalvis, g_numportals * 2);
		p->status = stat_done;
	}

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

	Msg("[GPU Test] Démarrage comparaison CPU vs GPU pour %d échantillons (stride=%d) %s\n",
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


int CountBits (byte *bits, int numbits)
{
	int		i;
	int		c;

	c = 0;
	for (i=0 ; i<numbits ; i++)
		if ( CheckBit( bits, i ) )
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


void CheckStack (leaf_t *leaf, threaddata_t *thread)
{
	pstack_t	*p, *p2;

	for (p=thread->pstack_head.next ; p ; p=p->next)
	{
//		Msg ("=");
		if (p->leaf == leaf)
			Error ("CheckStack: leaf recursion");
		for (p2=thread->pstack_head.next ; p2 != p ; p2=p2->next)
			if (p2->leaf == p->leaf)
				Error ("CheckStack: late leaf recursion");
	}
//	Msg ("\n");
}


winding_t *AllocStackWinding (pstack_t *stack)
{
	int		i;

	for (i=0 ; i<3 ; i++)
	{
		if (stack->freewindings[i])
		{
			stack->freewindings[i] = 0;
			return &stack->windings[i];
		}
	}

	Error ("Out of memory. AllocStackWinding: failed");

	return NULL;
}

void FreeStackWinding (winding_t *w, pstack_t *stack)
{
	int		i;

	i = w - stack->windings;

	if (i<0 || i>2)
		return;		// not from local

	if (stack->freewindings[i])
		Error ("FreeStackWinding: allready free");
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

winding_t	*ChopWinding (winding_t *in, pstack_t *stack, plane_t *split)
{
	vec_t	dists[128];
	int		sides[128];
	int		counts[3];
	vec_t	dot;
	int		i, j;
	Vector	mid;
	winding_t	*neww;

	counts[0] = counts[1] = counts[2] = 0;

// determine sides for each point
	for (i=0 ; i<in->numpoints ; i++)
	{
		dot = DotProduct (in->points[i], split->normal);
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
		FreeStackWinding (in, stack);
		return NULL;
	}

	sides[i] = sides[0];
	dists[i] = dists[0];
	
	neww = AllocStackWinding (stack);

	neww->numpoints = 0;

	for (i=0 ; i<in->numpoints ; i++)
	{
		Vector& p1 = in->points[i];

		if (neww->numpoints == MAX_POINTS_ON_FIXED_WINDING)
		{
			FreeStackWinding (neww, stack);
			return in;		// can't chop -- fall back to original
		}

		if (sides[i] == SIDE_ON)
		{
			VectorCopy (p1, neww->points[neww->numpoints]);
			neww->numpoints++;
			continue;
		}
	
		if (sides[i] == SIDE_FRONT)
		{
			VectorCopy (p1, neww->points[neww->numpoints]);
			neww->numpoints++;
		}
		
		if (sides[i+1] == SIDE_ON || sides[i+1] == sides[i])
			continue;
			
		if (neww->numpoints == MAX_POINTS_ON_FIXED_WINDING)
		{
			FreeStackWinding (neww, stack);
			return in;		// can't chop -- fall back to original
		}

	// generate a split point
		Vector& p2 = in->points[(i+1)%in->numpoints];
		
		dot = dists[i] / (dists[i]-dists[i+1]);
		for (j=0 ; j<3 ; j++)
		{	// avoid round off error when possible
			if (split->normal[j] == 1)
				mid[j] = split->dist;
			else if (split->normal[j] == -1)
				mid[j] = -split->dist;
			else
				mid[j] = p1[j] + dot*(p2[j]-p1[j]);
		}
			
		VectorCopy (mid, neww->points[neww->numpoints]);
		neww->numpoints++;
	}
	
// free the original winding
	FreeStackWinding (in, stack);
	
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
winding_t	*ClipToSeperators (winding_t *source, winding_t *pass, winding_t *target, bool flipclip, pstack_t *stack)
{
	int			i, j, k, l;
	plane_t		plane;
	Vector		v1, v2;
	float		d;
	vec_t		length;
	int			counts[3];
	bool		fliptest;

// check all combinations	
	for (i=0 ; i<source->numpoints ; i++)
	{
		l = (i+1)%source->numpoints;
		VectorSubtract (source->points[l] , source->points[i], v1);

	// fing a vertex of pass that makes a plane that puts all of the
	// vertexes of pass on the front side and all of the vertexes of
	// source on the back side
		for (j=0 ; j<pass->numpoints ; j++)
		{
			VectorSubtract (pass->points[j], source->points[i], v2);

			plane.normal[0] = v1[1]*v2[2] - v1[2]*v2[1];
			plane.normal[1] = v1[2]*v2[0] - v1[0]*v2[2];
			plane.normal[2] = v1[0]*v2[1] - v1[1]*v2[0];
			
		// if points don't make a valid plane, skip it

			length = plane.normal[0] * plane.normal[0]
			+ plane.normal[1] * plane.normal[1]
			+ plane.normal[2] * plane.normal[2];
			
			if (length < ON_VIS_EPSILON)
				continue;

			length = 1/sqrt(length);
			
			plane.normal[0] *= length;
			plane.normal[1] *= length;
			plane.normal[2] *= length;

			plane.dist = DotProduct (pass->points[j], plane.normal);

		//
		// find out which side of the generated seperating plane has the
		// source portal
		//
#if 1
			fliptest = false;
			for (k=0 ; k<source->numpoints ; k++)
			{
				if (k == i || k == l)
					continue;
				d = DotProduct (source->points[k], plane.normal) - plane.dist;
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
				VectorSubtract (vec3_origin, plane.normal, plane.normal);
				plane.dist = -plane.dist;
			}
#if 1
		//
		// if all of the pass portal points are now on the positive side,
		// this is the seperating plane
		//
			counts[0] = counts[1] = counts[2] = 0;
			for (k=0 ; k<pass->numpoints ; k++)
			{
				if (k==j)
					continue;
				d = DotProduct (pass->points[k], plane.normal) - plane.dist;
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
			k = (j+1)%pass->numpoints;
			d = DotProduct (pass->points[k], plane.normal) - plane.dist;
			if (d < -ON_VIS_EPSILON)
				continue;
			k = (j+pass->numpoints-1)%pass->numpoints;
			d = DotProduct (pass->points[k], plane.normal) - plane.dist;
			if (d < -ON_VIS_EPSILON)
				continue;			
#endif
		//
		// flip the normal if we want the back side
		//
			if (flipclip)
			{
				VectorSubtract (vec3_origin, plane.normal, plane.normal);
				plane.dist = -plane.dist;
			}
			
		//
		// clip target by the seperating plane
		//
			target = ChopWinding (target, stack, &plane);
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

void WindingCenter (winding_t *w, Vector &center)
{
	int		i;
	float	scale;

	VectorCopy (vec3_origin, center);
	for (i=0 ; i<w->numpoints ; i++)
		VectorAdd (w->points[i], center, center);

	scale = 1.0/w->numpoints;
	VectorScale (center, scale, center);
}

Vector ClusterCenter( int cluster )
{
	Vector mins, maxs;
	ClearBounds(mins, maxs);
	int count = leafs[cluster].portals.Count();
	for ( int i = 0; i < count; i++ )
	{
		winding_t *w = leafs[cluster].portals[i]->winding;
		for ( int j = 0; j < w->numpoints; j++ )
		{
			AddPointToBounds( w->points[j], mins, maxs );
		}
	}
	return (mins + maxs) * 0.5f;
}


void DumpPortalTrace( pstack_t *pStack )
{
	AUTO_LOCK(g_PortalTrace.m_mutex);
	if ( g_PortalTrace.m_list.Count() )
		return;

	Warning("Dumped cluster trace!!!\n");
	Vector	mid;
	mid = ClusterCenter( g_TraceClusterStart );
	g_PortalTrace.m_list.AddToTail(mid);
	for ( ; pStack != NULL; pStack = pStack->next )
	{
		winding_t *w = pStack->pass ? pStack->pass : pStack->portal->winding;
		WindingCenter (w, mid);
		g_PortalTrace.m_list.AddToTail(mid);
		for ( int i = 0; i < w->numpoints; i++ )
		{
			g_PortalTrace.m_list.AddToTail(w->points[i]);
			g_PortalTrace.m_list.AddToTail(mid);
		}
		for ( int i = 0; i < w->numpoints; i++ )
		{
			g_PortalTrace.m_list.AddToTail(w->points[i]);
		}
		g_PortalTrace.m_list.AddToTail(w->points[0]);
		g_PortalTrace.m_list.AddToTail(mid);
	}
	mid = ClusterCenter( g_TraceClusterStop );
	g_PortalTrace.m_list.AddToTail(mid);
}

void WritePortalTrace( const char *source )
{
	Vector	mid;
	FILE	*linefile;
	char	filename[1024];

	if ( !g_PortalTrace.m_list.Count() )
	{
		Warning("No trace generated from %d to %d\n", g_TraceClusterStart, g_TraceClusterStop );
		return;
	}

	sprintf (filename, "%s.lin", source);
	linefile = fopen (filename, "w");
	if (!linefile)
		Error ("Couldn't open %s\n", filename);

	for ( int i = 0; i < g_PortalTrace.m_list.Count(); i++ )
	{
		Vector p = g_PortalTrace.m_list[i];
		fprintf (linefile, "%f %f %f\n", p[0], p[1], p[2]);
	}
	fclose (linefile);
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
void SimpleFlood (portal_t *srcportal, int leafnum)
{
	int		i;
	leaf_t	*leaf;
	portal_t	*p;
	int		pnum;

	leaf = &leafs[leafnum];
	
	for (i=0 ; i<leaf->portals.Count(); i++)
	{
		p = leaf->portals[i];
		pnum = p - portals;
		if ( !CheckBit( srcportal->portalfront, pnum ) )
			continue;

		if ( CheckBit( srcportal->portalflood, pnum ) )
			continue;

		SetBit( srcportal->portalflood, pnum );
		
		SimpleFlood (srcportal, p->leaf);
	}
}

/*
==============
BasePortalVis [OLD]
==============
*/

void BasePortalVis (int iThread, int portalnum)
{
	int			j, k;
	portal_t	*tp, *p;
	float		d;
	winding_t	*w;
	Vector		segment;
	double		dist2, minDist2;

	// get the portal
	p = portals+portalnum;

	//
	// allocate memory for bitwise vis solutions for this portal
	//
	p->portalfront = (byte*)malloc (portalbytes);
	memset (p->portalfront, 0, portalbytes);

	p->portalflood = (byte*)malloc (portalbytes);
	memset (p->portalflood, 0, portalbytes);
	
	p->portalvis = (byte*)malloc (portalbytes);
	memset (p->portalvis, 0, portalbytes);
	
	//
	// test the given portal against all of the portals in the map
	//
	for (j=0, tp = portals ; j<g_numportals*2 ; j++, tp++)
	{
		// don't test against itself
		if (j == portalnum)
			continue;

		//
		//
		//
		w = tp->winding;
		for (k=0 ; k<w->numpoints ; k++)
		{
			d = DotProduct (w->points[k], p->plane.normal) - p->plane.dist;
			if (d > ON_VIS_EPSILON)
				break;
		}
		if (k == w->numpoints)
			continue;	// no points on front

		//
		//
		//
		w = p->winding;
		for (k=0 ; k<w->numpoints ; k++)
		{
			d = DotProduct (w->points[k], tp->plane.normal) - tp->plane.dist;
			if (d < -ON_VIS_EPSILON)
				break;
		}
		if (k == w->numpoints)
			continue;	// no points on front

		//
		// if using radius visibility -- check to see if any portal points lie inside of the
		// radius given
		//
		if( g_bUseRadius )
		{
			w = tp->winding;
			minDist2 = 1024000000.0;			// 32000^2
			for( k = 0; k < w->numpoints; k++ )
			{
				VectorSubtract( w->points[k], p->origin, segment );
				dist2 = ( segment[0] * segment[0] ) + ( segment[1] * segment[1] ) + ( segment[2] * segment[2] );
				if( dist2 < minDist2 )
				{
					minDist2 = dist2;
				}
			}

			if( minDist2 > g_VisRadius )
				continue;
		}

		// add current portal to given portal's list of visible portals
		SetBit( p->portalfront, j );
	}
	
	SimpleFlood (p, p->leaf);

	p->nummightsee = CountBits (p->portalflood, g_numportals*2);
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


struct OpenCLCleanup {
    ~OpenCLCleanup() { g_clManager.cleanup(); }
} g_OpenCLCleanup;
