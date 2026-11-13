// vrad_gpu.h
//
// Interface du backend CUDA de VRAD (visibilite des transferts).
// Implementation : vrad_gpu.cpp. Noyau : vrad_gpu_kernel.cu (PTX embarque).

#ifndef VRAD_GPU_H
#define VRAD_GPU_H
#ifdef _WIN32
#pragma once
#endif

#include "tier1/utlvector.h"

extern bool g_bUseCUDA;		// -cuda : active le backend GPU

extern bool g_bUseCUDA;		// -cuda : active le backend GPU
extern bool g_bCUDAVerify;	// -cudaverify : compare chaque rayon GPU au CPU

// Charge CUDA et envoie k-d tree + triangles de g_RtEnv. A appeler APRES
// g_RtEnv.SetupAccelerationStructure(). Retourne false (et desactive g_bUseCUDA)
// si le GPU n'est pas utilisable : VRAD retombe alors sur le CPU.
bool VradGPU_Init();
void VradGPU_Shutdown();

// Trace numRays rayons (6 floats : origine puis point d'arrivee). out[i] = 1 si
// le rayon est libre, 0 s'il est bloque. Thread-safe (verrou interne).
bool VradGPU_TraceRays( const float *rays, int numRays, unsigned char *out, int mode, int skipId );

// mode 0 : libre/bloque (transferts). mode 1 : lumiere directe, le ciel ne bloque pas, skipId ignore.
inline bool VradGPU_TraceVisibility( const float *rays, int numRays, unsigned char *out )
{
	return VradGPU_TraceRays( rays, numRays, out, 0, -1 );
}

// Compteurs de diagnostic (mode -cudaverify)
void VradGPU_AddVerifyStats( int rays, int mismatches );
void VradGPU_PrintStats();
void VradProf_PrintLights();

// File de rayons d'ombre differes, par thread (lumiere directe standard).
// mode 1 = enregistrement : TestLine memorise ses 4 rayons et renvoie "visible".
// mode 2 = rejeu : TestLine relit les resultats GPU dans le meme ordre ; si le rayon
// ne correspond pas a l'enregistrement, ou si le GPU renvoie 2 (triangle transparent),
// il retombe sur le CPU. rays = 6 floats par rayon, 4 rayons par appel.
struct VradRayQueue_t
{
	int mode;
	int pos;		// rayon courant (rejeu)
	int misses;		// appels repasses au CPU
	CUtlVector<float> rays;
	CUtlVector<unsigned char> res;
	VradRayQueue_t() : mode( 0 ), pos( 0 ), misses( 0 ) {}
};
extern __declspec( thread ) VradRayQueue_t *g_pRayQueue;
void VradGPU_AddDirectStats( int rays, int misses );

// Eclairage indirect des props statiques sur GPU. Prepare envoie BSP + lightmaps (une fois,
// APRES le calcul final des lightmaps). Indirect : rays = 6 floats (origine, delta) ; out = 4 floats
// (rgb, w) ; w = 0 rien, 1 contribution, 2 a refaire sur le CPU.
extern bool g_bIndirectGPU;	// vrai une fois les donnees BSP/lightmaps envoyees au GPU
bool VradGPU_IndirectPrepare();
bool VradGPU_Indirect( const float *rays, int numRays, float *out );

#endif
