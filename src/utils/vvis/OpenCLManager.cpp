// ============================================================================
// OpenCLManager.cpp — Manager OpenCL complet pour VVIS_GPU ULTRA
// ============================================================================

#include "OpenCLManager.h"
#include "tier0/platform.h"
#include "tier0/dbg.h"
#include "tier1/strtools.h"

#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// GLOBAL UNIQUE
// ============================================================================
OpenCLManager g_clManager;

// ============================================================================
// Helper — charger un fichier texte dans une CUtlVector<char>
// ============================================================================
bool LoadFileToString(const char* path, CUtlVector<char>& data)
{
    FILE* f = fopen(path, "rb");
    if (!f)
    {
        Msg("[OpenCL] ERREUR : Impossible d’ouvrir %s\n", path);
        return false;
    }

    fseek(f, 0, SEEK_END);
    int size = ftell(f);
    fseek(f, 0, SEEK_SET);

    data.SetSize(size + 1);
    fread(data.Base(), 1, size, f);
    fclose(f);

    data[size] = '\0';
    return true;
}


// ============================================================================
// Initialisation OpenCL (Appelée UNE FOIS par vvis.cpp)
// ============================================================================
void OpenCLManager::init_once()
{
    if (ok)
        return;

    Msg("[OpenCL] Initialisation...\n");

    cl_int err;

    // ===============================================================
    // 1. Choix du device GPU
    // ===============================================================
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0)
    {
        Msg("[OpenCL] Aucune plateforme détectée.\n");
        return;
    }

    CUtlVector<cl_platform_id> platforms;
    platforms.SetSize(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.Base(), nullptr);

    cl_platform_id chosenPlatform = platforms[0];

    cl_uint numDevices = 0;
    err = clGetDeviceIDs(chosenPlatform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0)
    {
        Msg("[OpenCL] Aucun GPU OpenCL détecté. Fallback CPU.\n");
        return;
    }

    CUtlVector<cl_device_id> devices;
    devices.SetSize(numDevices);
    clGetDeviceIDs(chosenPlatform, CL_DEVICE_TYPE_GPU, numDevices, devices.Base(), nullptr);

    device = devices[0];

    // ===============================================================
    // 2. Création du contexte
    // ===============================================================
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (!context || err != CL_SUCCESS)
    {
        Msg("[OpenCL] ERREUR : Impossible de créer un contexte.\n");
        return;
    }

    // ===============================================================
    // 3. Création de la queue
    // ===============================================================
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (!queue || err != CL_SUCCESS)
    {
        Msg("[OpenCL] ERREUR : Impossible de créer la commande queue.\n");
        return;
    }

    // ===============================================================
    // 4. Charger les kernels depuis flow_kernels.cl
    // ===============================================================
    CUtlVector<char> kernelSrc;
    if (!LoadFileToString("flow_kernels.cl", kernelSrc))
    {
        Msg("[OpenCL] ERREUR : Impossible de charger flow_kernels.cl\n");
        return;
    }

    const char* src = kernelSrc.Base();
    program = clCreateProgramWithSource(context, 1, &src, nullptr, &err);
    if (!program || err != CL_SUCCESS)
    {
        Msg("[OpenCL] ERREUR : échec clCreateProgramWithSource\n");
        return;
    }

    // ===============================================================
    // 5. Compiler le programme OpenCL
    // ===============================================================
    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);

    if (err != CL_SUCCESS)
    {
        Msg("[OpenCL] ERREUR : échec clBuildProgram\n");

        // --- Afficher le log de compilation ---
        size_t logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

        CUtlVector<char> log;
        log.SetSize(logSize + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.Base(), nullptr);
        log[logSize] = '\0';

        Msg("=== LOG BUILD OPENCL ===\n%s\n=== FIN LOG ===\n", log.Base());
        return;
    }

    Msg("[OpenCL] Compilation OK.\n");

    // ===============================================================
    // 6. Récupérer les kernels FlowGPU
    // ===============================================================

#define LOAD_KERNEL(var, name)                                       \
        var = clCreateKernel(program, name, &err);                        \
        if (!var || err != CL_SUCCESS)                                   \
        {                                                                 \
            Msg("[OpenCL] ERREUR : kernel introuvable : %s\n", name);     \
            return;                                                       \
        }

    LOAD_KERNEL(kernel_raycast, "RaycastPreset");
    LOAD_KERNEL(kernel_cone, "ConePreset");
    LOAD_KERNEL(kernel_hlod, "HLODPreset");
    LOAD_KERNEL(kernel_bfs, "LeafBFSStep");

#undef LOAD_KERNEL

    Msg("[OpenCL] Kernels chargés avec succès.\n");

    ok = true;
}


// ============================================================================
// Destruction (non indispensable car vvis termine après utilisation)
// ============================================================================
void OpenCLManager::destroy()
{
    if (!ok)
        return;

    if (kernel_raycast) clReleaseKernel(kernel_raycast);
    if (kernel_cone)    clReleaseKernel(kernel_cone);
    if (kernel_hlod)    clReleaseKernel(kernel_hlod);
    if (kernel_bfs)     clReleaseKernel(kernel_bfs);

    if (program) clReleaseProgram(program);
    if (queue)   clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);

    ok = false;
}
