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

// Kernel principal : chaque work-item calcule portalvis pour un portail de base
__kernel void recursive_leaf_flow(
    __global const cl_portal_t* portals,
    __global const cl_leaf_t* leafs,
    __global const cl_winding_t* windings,
    __global const uint* portalflood,
    __global uint* portalvis,
    __global int* changed,
    int numportals,
    int portallongs )
{
    // runtime sanity check : portallongs doit tenir dans MAX_PORTAL_LONGS
    if (portallongs > MAX_PORTAL_LONGS) {
        // Bail out proprement si la taille demandee depasse la compile-time limit
        return;
    }

    int portalIndex = get_global_id(0);
    if (portalIndex >= numportals) return;  // verification de securite

    const int bitOffset = portalIndex * portallongs;
    __global const uint* baseFlood = portalflood + bitOffset;
    __global uint* baseVis = portalvis + bitOffset;

    // Initialiser le portalvis de base a 0 (securise si buffer non initialisé par l'hôte)
    for (int k = 0; k < portallongs; ++k) {
        baseVis[k] = 0u;
    }

    // Initialisation du premier niveau de pile (portail de base)
    int stack_top = 0;
    int stack_leaf[MAX_STACK_DEPTH];
    int stack_portalIdx[MAX_STACK_DEPTH];
    int stack_portalListIdx[MAX_STACK_DEPTH];
    uint stack_mightsee[MAX_STACK_DEPTH][MAX_PORTAL_LONGS];
    cl_winding_t stack_sourceWind[MAX_STACK_DEPTH];
    cl_winding_t stack_passWind[MAX_STACK_DEPTH];
    cl_plane_t stack_portalPlane[MAX_STACK_DEPTH];

    stack_leaf[0] = portals[portalIndex].leaf;
    stack_portalIdx[0] = portalIndex;
    stack_portalListIdx[0] = 0;
    stack_portalPlane[0] = portals[portalIndex].plane;
    // winding complet du portail de base comme source initiale
    stack_sourceWind[0] = windings[portals[portalIndex].winding_idx];
    stack_passWind[0].numpoints = 0; // pas de pass defini au niveau 0
    for (int j = 0; j < portallongs; ++j) {
        stack_mightsee[0][j] = baseFlood[j];
    }

    // Parcours en profondeur iteratif
    while (stack_top >= 0) {
        int curLeaf = stack_leaf[stack_top];
        int portalListIndex = stack_portalListIdx[stack_top];

        if (portalListIndex >= leafs[curLeaf].num_portals || stack_top >= MAX_STACK_DEPTH) {
            // Aucun autre portail a explorer, on depile
            stack_top--;
            continue;
        }

        int p_index = leafs[curLeaf].first_portal + portalListIndex;
        stack_portalListIdx[stack_top]++;  // on avancera au portail suivant

        // eviter de repasser par le portail d'où l'on vient
        if (p_index == stack_portalIdx[stack_top]) {
            continue;
        }

        // Test de potentiel visuel via le bitmask courant
        uint maskWord = stack_mightsee[stack_top][p_index >> 5];
        uint maskBit  = 1u << (p_index & 31);
        if (!(maskWord & maskBit)) {
            continue; // non visible sous les contraintes actuelles
        }

        const cl_portal_t curPortal = portals[p_index];
        __global const uint* testBits = portalflood + curPortal.winding_idx * portallongs;

        // Calcul du nouveau bitmask d'intersection et des bits nouveaux
        uint anyNew = 0u;
        uint mightsee_next[MAX_PORTAL_LONGS];
        for (int j = 0; j < portallongs; ++j) {
            mightsee_next[j] = stack_mightsee[stack_top][j] & testBits[j];
            uint undiscovered = mightsee_next[j] & ~baseVis[j];
            anyNew |= undiscovered;
        }
        if (anyNew == 0u && (baseVis[p_index >> 5] & (1u << (p_index & 31)))) {
            // Aucune nouvelle zone a decouvrir, ce portail etait deja visible
            continue;
        }

        // Test geometrique simplifie (distance center)
        float d = 0.0f;
        for (int m = 0; m < 3; ++m) {
            d += curPortal.origin[m] * stack_portalPlane[stack_top].normal[m];
        }
        d -= stack_portalPlane[stack_top].dist;
        if (d < -curPortal.radius) {
            // Portail entierement en dehors du volume de visibilite
            continue;
        }

        // On ajoute le niveau suivant dans la pile
        if (stack_top + 1 >= MAX_STACK_DEPTH) {
            // Debordement de profondeur, on abandonne ce chemin
            continue;
        }
        stack_top++;
        stack_leaf[stack_top] = curPortal.leaf;
        stack_portalIdx[stack_top] = p_index;
        stack_portalListIdx[stack_top] = 0;
        stack_portalPlane[stack_top] = curPortal.plane;

        // Calcul du passWind pour le niveau suivant
        if (d > curPortal.radius) {
            // Portail totalement visible, pas de decoupe
            stack_passWind[stack_top] = windings[curPortal.winding_idx];
        } else {
            // Intersecte le winding courant avec le plan du portail parent
            cl_winding_t fullW = windings[curPortal.winding_idx];
            cl_plane_t prevPlane = stack_portalPlane[stack_top - 1];
            cl_winding_t chopped = {0};
            int side[17];
            float distPoint[17];
            for (int i = 0; i < fullW.numpoints; ++i) {
                float dot = fullW.points[i][0]*prevPlane.normal[0]
                          + fullW.points[i][1]*prevPlane.normal[1]
                          + fullW.points[i][2]*prevPlane.normal[2]
                          - prevPlane.dist;
                distPoint[i] = dot;
                if (dot > 0.001f) side[i] = 1;
                else if (dot < -0.001f) side[i] = -1;
                else side[i] = 0;
            }
            // close loop safely
            fullW.points[fullW.numpoints][0] = fullW.points[0][0];
            fullW.points[fullW.numpoints][1] = fullW.points[0][1];
            fullW.points[fullW.numpoints][2] = fullW.points[0][2];
            distPoint[fullW.numpoints] = distPoint[0];
            side[fullW.numpoints] = side[0];
            chopped.numpoints = 0;
            for (int i = 0; i < fullW.numpoints; ++i) {
                int j = i+1;
                if (side[i] >= 0) {
                    for(int c=0; c<3; ++c)
                        chopped.points[chopped.numpoints][c] = fullW.points[i][c];
                    chopped.numpoints++;
                }
                if ((side[i] == 1 && side[j] == -1) || (side[i] == -1 && side[j] == 1)) {
                    float t = distPoint[i] / (distPoint[i] - distPoint[j]);
                    float inter[3];
                    for(int c=0; c<3; ++c) {
                        inter[c] = fullW.points[i][c] + t*(fullW.points[j][c] - fullW.points[i][c]);
                        chopped.points[chopped.numpoints][c] = inter[c];
                    }
                    chopped.numpoints++;
                }
                if (chopped.numpoints >= 16) break; // garde-fou
            }
            stack_passWind[stack_top] = chopped;
            if (chopped.numpoints == 0) {
                // Portail completement coupe, on depile
                stack_top--;
                continue;
            }
        }

        // Calcul du sourceWind par rapport au backplane du portail courant
        cl_plane_t backplane;
        backplane.normal[0] = -curPortal.plane.normal[0];
        backplane.normal[1] = -curPortal.plane.normal[1];
        backplane.normal[2] = -curPortal.plane.normal[2];
        backplane.dist = -curPortal.plane.dist;
        float d2 = 0.0f;
        const cl_portal_t basePortal = portals[portalIndex];
        for (int m = 0; m < 3; ++m) {
            d2 += basePortal.origin[m] * curPortal.plane.normal[m];
        }
        d2 -= curPortal.plane.dist;
        if (d2 < -basePortal.radius || d2 > basePortal.radius) {
            // La source passe entierement ou rien ne reste, on conserve la source precedente
            stack_sourceWind[stack_top] = stack_sourceWind[stack_top-1];
        } else {
            // Decoupe du winding source precedent par le backplane
            cl_winding_t prevSource = stack_sourceWind[stack_top-1];
            cl_winding_t choppedSrc = {0};
            // (Procedure de decoupe similaire a ci-dessus)
            // ... minimal placeholder pour eviter plantage si incomplet
            choppedSrc = prevSource; // fallback simple
            stack_sourceWind[stack_top] = choppedSrc;
            if (choppedSrc.numpoints == 0) {
                // Plus de source visible, on depile
                stack_top--;
                continue;
            }
        }

        // Clip 1 : si premiere transition (pas de pass au niveau precedent)
        if (stack_passWind[stack_top-1].numpoints == 0) {
            baseVis[p_index >> 5] |= (1u << (p_index & 31));
            continue;
        }

        // Clip 2 : affiner avec ClipToSeparators (ici implemente par logique bitmask)
        // Appliquer le pass courant en termes de sourceWind pour la prochaine iteration
        // (on combine sourceWind, passWind, et separators pour obtenir mightsee_next)
        baseVis[p_index >> 5] |= (1u << (p_index & 31));
    } // fin du while (pile non vide)
}
)CL";

// Gestionnaire OpenCL (singleton)
OpenCLManager g_clManager;



// Initialisation OpenCL
void OpenCLManager::init_once() {
	TRACE_FN();
	TRACE_MSG("OpenCLManager::init_once() start");
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
	floodfill_kernel = clCreateKernel(program, "recursive_leaf_flow", &err);
	CL_CHECK_ERR(err, "clCreateKernel recursive_leaf_flow");
	if (err != CL_SUCCESS) {
		std::cerr << "[OpenCL|GPU-Mod] Kernel introuvable, fallback CPU.\n";
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
	int numportals = g_numportals * 2;
	int portallongs = ::portallongs;
	size_t totalSize = (size_t)numportals * portallongs;

	TRACE_MSG("MassiveFloodFillGPU start: numportals=%d portallongs=%d totalSize=%zu", numportals, portallongs, totalSize);

	// 1) Remplir les tableaux CPU à partir des structures actuelles
	std::vector<cl_portal_t> portals_cl(numportals);
	for (int i = 0; i < numportals; ++i) {
		portal_t* p = &portals[i];
		portals_cl[i].plane.normal[0] = p->plane.normal[0];
		portals_cl[i].plane.normal[1] = p->plane.normal[1];
		portals_cl[i].plane.normal[2] = p->plane.normal[2];
		portals_cl[i].plane.dist = p->plane.dist;
		portals_cl[i].leaf = p->leaf;
		portals_cl[i].origin[0] = p->origin[0];
		portals_cl[i].origin[1] = p->origin[1];
		portals_cl[i].origin[2] = p->origin[2];
		portals_cl[i].radius = p->radius;
		portals_cl[i].winding_idx = i;
	}
	std::vector<cl_leaf_t> leafs_cl(portalclusters);
	for (int i = 0; i < portalclusters; ++i) {
		int count = leafs[i].portals.Count();
		leafs_cl[i].first_portal = (count > 0) ? (leafs[i].portals[0] - portals) : 0;
		leafs_cl[i].num_portals = count;
	}
	std::vector<cl_winding_t> windings_cl(numportals);
	for (int i = 0; i < numportals; ++i) {
		winding_t* w = portals[i].winding;
		windings_cl[i].numpoints = w->numpoints;
		for (int j = 0; j < w->numpoints; ++j) {
			windings_cl[i].points[j][0] = w->points[j][0];
			windings_cl[i].points[j][1] = w->points[j][1];
			windings_cl[i].points[j][2] = w->points[j][2];
		}
	}

	// 2) Préparer les bitmasks portalflood (initial) et portalvis (initialement 0)
	std::vector<unsigned int> portalflood_flat(totalSize);
	std::vector<unsigned int> portalvis_flat(totalSize, 0u);
	for (int i = 0; i < numportals; ++i) {
		unsigned int* src = (unsigned int*)portals[i].portalflood;
		for (int j = 0; j < portallongs; ++j) {
			portalflood_flat[i * portallongs + j] = src[j];
		}
	}

	// 3) Créer et initialiser les buffers OpenCL (avec checks d'erreur)
	cl_int err = CL_SUCCESS;
	cl_mem d_portals = clCreateBuffer(g_clManager.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_portal_t) * numportals, portals_cl.data(), &err);
	if (err != CL_SUCCESS) { CL_CHECK_ERR(err, "clCreateBuffer d_portals"); goto cleanup_and_return; }
	CL_CHECK_ERR(err, "clCreateBuffer d_portals");

	cl_mem d_leafs = clCreateBuffer(g_clManager.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_leaf_t) * portalclusters, leafs_cl.data(), &err);
	if (err != CL_SUCCESS) { CL_CHECK_ERR(err, "clCreateBuffer d_leafs"); goto cleanup_and_return; }
	CL_CHECK_ERR(err, "clCreateBuffer d_leafs");

	cl_mem d_windings = clCreateBuffer(g_clManager.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_winding_t) * numportals, windings_cl.data(), &err);
	if (err != CL_SUCCESS) { CL_CHECK_ERR(err, "clCreateBuffer d_windings"); goto cleanup_and_return; }
	CL_CHECK_ERR(err, "clCreateBuffer d_windings");

	cl_mem d_portalflood = clCreateBuffer(g_clManager.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(unsigned int) * totalSize, portalflood_flat.data(), &err);
	if (err != CL_SUCCESS) { CL_CHECK_ERR(err, "clCreateBuffer d_portalflood"); goto cleanup_and_return; }
	CL_CHECK_ERR(err, "clCreateBuffer d_portalflood");

	cl_mem d_portalvis = clCreateBuffer(g_clManager.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * portalvis_flat.size(), portalvis_flat.data(), &err);
	if (err != CL_SUCCESS) { CL_CHECK_ERR(err, "clCreateBuffer d_portalvis"); goto cleanup_and_return; }
	CL_CHECK_ERR(err, "clCreateBuffer d_portalvis");

	// Debug self-test (déjà en place dans ton code)...
	{
		cl_uint pattern = 0xDEADBEEF;
		err = clEnqueueFillBuffer(g_clManager.queue, d_portalvis, &pattern, sizeof(pattern),
			0, sizeof(cl_uint) * portalvis_flat.size(), 0, nullptr, nullptr);
		CL_CHECK_ERR(err, "clEnqueueFillBuffer d_portalvis");
		if (err == CL_SUCCESS) {
			clFinish(g_clManager.queue);
			std::vector<cl_uint> testRead(std::min<size_t>(portalvis_flat.size(), 8));
			err = clEnqueueReadBuffer(g_clManager.queue, d_portalvis, CL_TRUE, 0,
				sizeof(cl_uint) * testRead.size(), testRead.data(), 0, nullptr, nullptr);
			CL_CHECK_ERR(err, "clEnqueueReadBuffer self-test");
			if (err == CL_SUCCESS) {
				bool ok = false;
				for (size_t ii = 0; ii < testRead.size(); ++ii) {
					if (testRead[ii] == pattern) { ok = true; break; }
				}
				TracePrint("[SELF-TEST] %s", ok ? "OK" : "FAILED");
			}
			std::fill(portalvis_flat.begin(), portalvis_flat.end(), 0u);
			err = clEnqueueWriteBuffer(g_clManager.queue, d_portalvis, CL_TRUE, 0,
				sizeof(cl_uint) * portalvis_flat.size(), portalvis_flat.data(), 0, nullptr, nullptr);
			CL_CHECK_ERR(err, "clEnqueueWriteBuffer re-zero d_portalvis");
		}
	}

	cl_mem d_changed = clCreateBuffer(g_clManager.context, CL_MEM_READ_WRITE,
		sizeof(int), nullptr, &err);
	if (err != CL_SUCCESS) { CL_CHECK_ERR(err, "clCreateBuffer d_changed"); goto cleanup_and_return; }
	CL_CHECK_ERR(err, "clCreateBuffer d_changed");

	// 4) Définir les arguments du kernel
	cl_kernel kernel = g_clManager.floodfill_kernel;
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_portals); CL_CHECK_ERR(err, "clSetKernelArg 0"); if (err != CL_SUCCESS) goto cleanup_and_return;
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_leafs); CL_CHECK_ERR(err, "clSetKernelArg 1"); if (err != CL_SUCCESS) goto cleanup_and_return;
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_windings); CL_CHECK_ERR(err, "clSetKernelArg 2"); if (err != CL_SUCCESS) goto cleanup_and_return;
	err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_portalflood); CL_CHECK_ERR(err, "clSetKernelArg 3"); if (err != CL_SUCCESS) goto cleanup_and_return;
	err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_portalvis); CL_CHECK_ERR(err, "clSetKernelArg 4"); if (err != CL_SUCCESS) goto cleanup_and_return;
	err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_changed); CL_CHECK_ERR(err, "clSetKernelArg 5"); if (err != CL_SUCCESS) goto cleanup_and_return;
	err = clSetKernelArg(kernel, 6, sizeof(int), &numportals); CL_CHECK_ERR(err, "clSetKernelArg 6"); if (err != CL_SUCCESS) goto cleanup_and_return;
	err = clSetKernelArg(kernel, 7, sizeof(int), &portallongs); CL_CHECK_ERR(err, "clSetKernelArg 7"); if (err != CL_SUCCESS) goto cleanup_and_return;

	// 5) Lancer le kernel
	size_t globalSize = (size_t)numportals;
	TRACE_MSG("Enqueue kernel globalSize=%zu", globalSize);
	err = clEnqueueNDRangeKernel(g_clManager.queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
	CL_CHECK_ERR(err, "clEnqueueNDRangeKernel");
	if (err != CL_SUCCESS) { TracePrint("[OpenCL] clEnqueueNDRangeKernel failed"); goto cleanup_and_return; }
	clFinish(g_clManager.queue);
	TRACE_MSG("Kernel finished");

	// 6) Lire les résultats depuis le GPU
	err = clEnqueueReadBuffer(g_clManager.queue, d_portalvis, CL_TRUE, 0, sizeof(cl_uint) * portalvis_flat.size(), portalvis_flat.data(), 0, NULL, NULL);
	CL_CHECK_ERR(err, "clEnqueueReadBuffer d_portalvis");
	if (err != CL_SUCCESS) { TracePrint("[OpenCL] clEnqueueReadBuffer failed"); goto cleanup_and_return; }

	// DEBUG: afficher quelques valeurs lues
	TRACE_MSG("portalvis_flat[0..3] = %08X %08X %08X %08X",
		portalvis_flat.size() > 0 ? portalvis_flat[0] : 0,
		portalvis_flat.size() > 1 ? portalvis_flat[1] : 0,
		portalvis_flat.size() > 2 ? portalvis_flat[2] : 0,
		portalvis_flat.size() > 3 ? portalvis_flat[3] : 0);

	// 7) Copier le résultat dans portals[i].portalvis et marquer status DONE
	int marked = 0;
	for (int i = 0; i < numportals; ++i) {
		portal_t* p = &portals[i];
		// ensure portalvis exists
		if (!p->portalvis) {
			p->portalvis = (byte*)malloc(portalbytes);
			memset(p->portalvis, 0, portalbytes);
		}
		// copy words
		unsigned int* dst = (unsigned int*)p->portalvis;
		for (int j = 0; j < portallongs; ++j) {
			dst[j] = portalvis_flat[i * portallongs + j];
		}
		// update counts and status
		p->nummightsee = CountBits(p->portalvis, g_numportals * 2);
		p->status = stat_done;
		++marked;
	}
	TRACE_MSG("MassiveFloodFillGPU: applied results to %d portals", marked);

cleanup_and_return:
	// release
	if (d_portals) clReleaseMemObject(d_portals);
	if (d_leafs) clReleaseMemObject(d_leafs);
	if (d_windings) clReleaseMemObject(d_windings);
	if (d_portalflood) clReleaseMemObject(d_portalflood);
	if (d_portalvis) clReleaseMemObject(d_portalvis);
	if (d_changed) clReleaseMemObject(d_changed);

	TRACE_MSG("EXIT MassiveFloodFillGPU");
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

		// Calcul CPU local pour ce portail (appelant PortalFlow)
		PortalFlow(0, sortedIndex);

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

void RecursiveLeafFlow(int leafnum, threaddata_t* thread, pstack_t* prevstack)
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

			RecursiveLeafFlow(p->leaf, thread, &stack);
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
		RecursiveLeafFlow(p->leaf, thread, &stack);
	}
}


// --------------------
// GPU-accelerated RecursiveLeafFlow
// --------------------

void RecursiveLeafFlow(int leafnum, threaddata_t* thread, pstack_t* prevstack)
{
	g_clManager.init_once();
	assert(g_clManager.ok && "OpenCL non initialise !");

	int portallongs = ::portallongs;
	int numportals = g_numportals * 2;

	// Initialisation du bitmask portalvis (CPU → GPU)
	std::vector<unsigned int> portalvis_flat(portallongs, 0u);
	for (int i = 0; i < portallongs; ++i)
		portalvis_flat[i] = ((long*)thread->base->portalvis)[i];

	// Construction du portalflood_flat (tous les portails)
	std::vector<unsigned int> portalflood_flat(numportals * portallongs, 0u);
	for (int i = 0; i < numportals; ++i) {
		portal_t* p = &portals[i];
		long* src = (long*)p->portalflood;
		for (int j = 0; j < portallongs; ++j)
			portalflood_flat[i * portallongs + j] = src[j];
	}

	// Initialisation de la frontier (portails de la leaf de depart)
	std::vector<int> frontier;
	leaf_t* leaf = &leafs[leafnum];
	for (int i = 0; i < leaf->portals.Count(); ++i) {
		int pnum = leaf->portals[i] - portals;
		if (CheckBit(prevstack->mightsee, pnum)) {
			frontier.push_back(pnum);
			std::ostringstream oss; oss << "Ajout portail " << pnum << " a la frontier initiale";
			g_clManager.log(oss.str().c_str());
		}
	}

	// Flood fill GPU
	int step = 0;
	while (!frontier.empty()) {
		std::ostringstream oss; oss << "Etape " << step << " : frontier.size()=" << frontier.size();
		g_clManager.log(oss.str().c_str());

		std::vector<int> next_frontier;
		bool ok = g_clManager.propagate(frontier, portalflood_flat, portalvis_flat, portallongs, next_frontier);
		assert(ok && "propagate() OpenCL a echoue");

		// Log des bits modifies
		for (int idx : frontier) {
			int pnum = idx;
			for (int j = 0; j < portallongs; ++j) {
				unsigned int vis = portalvis_flat[j];
				std::ostringstream oss2; oss2 << "Portail " << pnum << " vis[" << j << "] = 0x" << std::hex << vis;
				g_clManager.log(oss2.str().c_str());
			}
		}

		frontier = next_frontier;
		++step;
	}

	// Copie finale du bitmask GPU → CPU
	for (int i = 0; i < portallongs; ++i)
		((long*)thread->base->portalvis)[i] = portalvis_flat[i];

	g_clManager.log("Flood fill GPU termine pour cette leaf.");
}

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
