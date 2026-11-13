// vrad_gpu.cpp
//
// Partie CPU du backend CUDA de VRAD. Meme principe que VVIS (flow_gpu_host.cpp) :
// nvcuda.dll chargee dynamiquement, noyau PTX embarque, aucun SDK a l'execution.
//
// Donnees envoyees une seule fois au GPU (geometrie statique de g_RtEnv) :
//   - noeuds du k-d tree (Children + SplittingPlaneValue)
//   - liste d'indices de triangles
//   - triangles au format "intersection" (plan + equations d'aretes projetees)
//   - boite englobante
// Puis, par lot : rayons -> GPU, resultats (1 octet/rayon) <- GPU.

#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "vrad.h"
#include "vrad_gpu.h"
#include "vrad_gpu_kernel_ptx.h"

bool g_bUseCUDA = false;
bool g_bCUDAVerify = false;
bool g_bIndirectGPU = false;
__declspec( thread ) VradRayQueue_t *g_pRayQueue = NULL;
static volatile LONG s_directRays = 0, s_directMiss = 0;

void VradGPU_AddDirectStats( int rays, int misses )
{
	InterlockedExchangeAdd( &s_directRays, rays );
	InterlockedExchangeAdd( &s_directMiss, misses );
}

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;

#define CU_CC_MAJOR 75
#define CU_CC_MINOR 76

#define CUAPI __stdcall
typedef CUresult (CUAPI *PFN_cuInit)( unsigned int );
typedef CUresult (CUAPI *PFN_cuDeviceGetCount)( int * );
typedef CUresult (CUAPI *PFN_cuDeviceGet)( CUdevice *, int );
typedef CUresult (CUAPI *PFN_cuDeviceGetName)( char *, int, CUdevice );
typedef CUresult (CUAPI *PFN_cuDeviceGetAttribute)( int *, int, CUdevice );
typedef CUresult (CUAPI *PFN_cuCtxCreate)( CUcontext *, unsigned int, CUdevice );
typedef CUresult (CUAPI *PFN_cuCtxDestroy)( CUcontext );
typedef CUresult (CUAPI *PFN_cuCtxSetCurrent)( CUcontext );
typedef CUresult (CUAPI *PFN_cuCtxSynchronize)( void );
typedef CUresult (CUAPI *PFN_cuModuleLoadData)( CUmodule *, const void * );
typedef CUresult (CUAPI *PFN_cuModuleUnload)( CUmodule );
typedef CUresult (CUAPI *PFN_cuModuleGetFunction)( CUfunction *, CUmodule, const char * );
typedef CUresult (CUAPI *PFN_cuMemAlloc)( CUdeviceptr *, size_t );
typedef CUresult (CUAPI *PFN_cuMemFree)( CUdeviceptr );
typedef CUresult (CUAPI *PFN_cuMemcpyHtoD)( CUdeviceptr, const void *, size_t );
typedef CUresult (CUAPI *PFN_cuMemcpyDtoH)( void *, CUdeviceptr, size_t );
typedef CUresult (CUAPI *PFN_cuLaunchKernel)( CUfunction, unsigned int, unsigned int, unsigned int,
	unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void **, void ** );
typedef CUresult (CUAPI *PFN_cuStreamCreate)( CUstream *, unsigned int );
typedef CUresult (CUAPI *PFN_cuStreamDestroy)( CUstream );
typedef CUresult (CUAPI *PFN_cuStreamSynchronize)( CUstream );
typedef CUresult (CUAPI *PFN_cuMemcpyHtoDAsync)( CUdeviceptr, const void *, size_t, CUstream );
typedef CUresult (CUAPI *PFN_cuMemcpyDtoHAsync)( void *, CUdeviceptr, size_t, CUstream );

static struct
{
	HMODULE hLib;
	PFN_cuInit Init;
	PFN_cuDeviceGetCount DeviceGetCount;
	PFN_cuDeviceGet DeviceGet;
	PFN_cuDeviceGetName DeviceGetName;
	PFN_cuDeviceGetAttribute DeviceGetAttribute;
	PFN_cuCtxCreate CtxCreate;
	PFN_cuCtxDestroy CtxDestroy;
	PFN_cuCtxSetCurrent CtxSetCurrent;
	PFN_cuCtxSynchronize CtxSynchronize;
	PFN_cuModuleLoadData ModuleLoadData;
	PFN_cuModuleUnload ModuleUnload;
	PFN_cuModuleGetFunction ModuleGetFunction;
	PFN_cuMemAlloc MemAlloc;
	PFN_cuMemFree MemFree;
	PFN_cuMemcpyHtoD MemcpyHtoD;
	PFN_cuMemcpyDtoH MemcpyDtoH;
	PFN_cuLaunchKernel LaunchKernel;
	PFN_cuStreamCreate StreamCreate;
	PFN_cuStreamDestroy StreamDestroy;
	PFN_cuStreamSynchronize StreamSynchronize;
	PFN_cuMemcpyHtoDAsync MemcpyHtoDAsync;
	PFN_cuMemcpyDtoHAsync MemcpyDtoHAsync;
} cu;

// Doit correspondre a struct GpuTri de vrad_gpu_kernel.cu
struct GpuTri
{
	float nx, ny, nz, d;
	float e[6];
	int cs0, cs1;
	int flags;
	int id;
};

static CRITICAL_SECTION s_lock;
static bool s_lockInit = false;
static CUcontext s_ctx = NULL;
static CUmodule s_mod = NULL;
static CUfunction s_kernel = NULL;
static CUdeviceptr s_dChildren = 0, s_dSplit = 0, s_dTriIdx = 0, s_dTris = 0, s_dBounds = 0;
static CUdeviceptr s_dRays = 0, s_dOut = 0;
static size_t s_rayCap = 0;

#define NUM_GPU_SLOTS 4
struct GpuSlot_t
{
	CRITICAL_SECTION cs;
	CUstream stream;
	CUdeviceptr dRays, dOut;
	size_t cap;
	CUdeviceptr dIndRays, dIndOut;
	size_t indCap;
};
static CUfunction s_indKernel = NULL;
static bool s_indReady = false;
static CUdeviceptr s_indNodes = 0, s_indLeaves = 0, s_indLeafFaces = 0, s_indFaces = 0, s_indSky = 0, s_indLight = 0, s_indPow2 = 0;
static int s_indLightBytes = 0;
static GpuSlot_t s_slots[NUM_GPU_SLOTS];
static volatile LONG s_slotCounter = 0;

static volatile LONG64 s_totalRays = 0;
static volatile LONG s_totalBatches = 0, s_verifyRays = 0, s_verifyMismatch = 0;

static bool LoadCUDA()
{
	cu.hLib = LoadLibraryA( "nvcuda.dll" );
	if ( !cu.hLib )
		return false;
#define CU_LOAD( member, symbol ) \
	cu.member = (PFN_cu##member)GetProcAddress( cu.hLib, symbol ); \
	if ( !cu.member ) return false;
	CU_LOAD( Init, "cuInit" );
	CU_LOAD( DeviceGetCount, "cuDeviceGetCount" );
	CU_LOAD( DeviceGet, "cuDeviceGet" );
	CU_LOAD( DeviceGetName, "cuDeviceGetName" );
	CU_LOAD( DeviceGetAttribute, "cuDeviceGetAttribute" );
	CU_LOAD( CtxCreate, "cuCtxCreate_v2" );
	CU_LOAD( CtxDestroy, "cuCtxDestroy_v2" );
	CU_LOAD( CtxSetCurrent, "cuCtxSetCurrent" );
	CU_LOAD( CtxSynchronize, "cuCtxSynchronize" );
	CU_LOAD( ModuleLoadData, "cuModuleLoadData" );
	CU_LOAD( ModuleUnload, "cuModuleUnload" );
	CU_LOAD( ModuleGetFunction, "cuModuleGetFunction" );
	CU_LOAD( MemAlloc, "cuMemAlloc_v2" );
	CU_LOAD( MemFree, "cuMemFree_v2" );
	CU_LOAD( MemcpyHtoD, "cuMemcpyHtoD_v2" );
	CU_LOAD( MemcpyDtoH, "cuMemcpyDtoH_v2" );
	CU_LOAD( LaunchKernel, "cuLaunchKernel" );
	CU_LOAD( StreamCreate, "cuStreamCreate" );
	CU_LOAD( StreamDestroy, "cuStreamDestroy_v2" );
	CU_LOAD( StreamSynchronize, "cuStreamSynchronize" );
	CU_LOAD( MemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2" );
	CU_LOAD( MemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2" );
#undef CU_LOAD
	return true;
}

static void FreeAll()
{
	CUdeviceptr *p[] = { &s_dChildren, &s_dSplit, &s_dTriIdx, &s_dTris, &s_dBounds, &s_dRays, &s_dOut };
	if ( cu.MemFree )
	{
		for ( int i = 0; i < 7; ++i )
			if ( *p[i] ) { cu.MemFree( *p[i] ); *p[i] = 0; }
	}
	s_rayCap = 0;
	for ( int i = 0; i < NUM_GPU_SLOTS; ++i )
	{
		if ( cu.MemFree )
		{
			if ( s_slots[i].dRays ) cu.MemFree( s_slots[i].dRays );
			if ( s_slots[i].dOut ) cu.MemFree( s_slots[i].dOut );
		}
		if ( cu.MemFree )
		{
			if ( s_slots[i].dIndRays ) cu.MemFree( s_slots[i].dIndRays );
			if ( s_slots[i].dIndOut ) cu.MemFree( s_slots[i].dIndOut );
		}
		if ( s_slots[i].stream && cu.StreamDestroy ) cu.StreamDestroy( s_slots[i].stream );
		s_slots[i].dRays = s_slots[i].dOut = 0;
		s_slots[i].dIndRays = s_slots[i].dIndOut = 0;
		s_slots[i].stream = NULL;
		s_slots[i].cap = 0;
		s_slots[i].indCap = 0;
	}
	{
		CUdeviceptr *ind[] = { &s_indNodes, &s_indLeaves, &s_indLeafFaces, &s_indFaces, &s_indSky, &s_indLight, &s_indPow2 };
		if ( cu.MemFree )
			for ( int i = 0; i < 7; ++i )
				if ( *ind[i] ) { cu.MemFree( *ind[i] ); *ind[i] = 0; }
		s_indReady = false;
		s_indKernel = NULL;
	}
	if ( s_mod ) { cu.ModuleUnload( s_mod ); s_mod = NULL; }
	if ( s_ctx ) { cu.CtxDestroy( s_ctx ); s_ctx = NULL; }
	s_kernel = NULL;
	if ( cu.hLib ) { FreeLibrary( cu.hLib ); }
	memset( &cu, 0, sizeof( cu ) );
}

static bool Upload( CUdeviceptr *dst, const void *src, size_t bytes )
{
	if ( bytes == 0 )
		bytes = 4;
	if ( cu.MemAlloc( dst, bytes ) != 0 )
		return false;
	return cu.MemcpyHtoD( *dst, src, bytes ) == 0;
}

bool VradGPU_Init()
{
	if ( !s_lockInit )
	{
		InitializeCriticalSection( &s_lock );
		for ( int i = 0; i < NUM_GPU_SLOTS; ++i )
			InitializeCriticalSection( &s_slots[i].cs );
		s_lockInit = true;
	}

	if ( !LoadCUDA() )
	{
		Warning( "[VRAD GPU] nvcuda.dll introuvable : repli CPU.\n" );
		FreeAll(); g_bUseCUDA = false; return false;
	}
	int count = 0;
	if ( cu.Init( 0 ) != 0 || cu.DeviceGetCount( &count ) != 0 || count <= 0 )
	{
		Warning( "[VRAD GPU] aucun GPU CUDA : repli CPU.\n" );
		FreeAll(); g_bUseCUDA = false; return false;
	}

	CUdevice dev = 0;
	int chosen = -1, bestSM = -1;
	for ( int i = 0; i < count; ++i )
	{
		CUdevice d; int major = 0, minor = 0, sm = 0;
		if ( cu.DeviceGet( &d, i ) != 0 ) continue;
		cu.DeviceGetAttribute( &major, CU_CC_MAJOR, d );
		cu.DeviceGetAttribute( &minor, CU_CC_MINOR, d );
		cu.DeviceGetAttribute( &sm, 16, d );
		if ( major < 7 || ( major == 7 && minor < 5 ) ) continue;
		if ( sm > bestSM ) { bestSM = sm; chosen = i; dev = d; }
	}
	if ( chosen < 0 )
	{
		Warning( "[VRAD GPU] GPU < compute 7.5 : repli CPU.\n" );
		FreeAll(); g_bUseCUDA = false; return false;
	}

	char name[256] = { 0 };
	cu.DeviceGetName( name, 255, dev );

	if ( cu.CtxCreate( &s_ctx, 0, dev ) != 0 )
	{
		Warning( "[VRAD GPU] creation du contexte CUDA impossible : repli CPU.\n" );
		FreeAll(); g_bUseCUDA = false; return false;
	}
	CUresult r = cu.ModuleLoadData( &s_mod, g_VradTracePTX );
	if ( r != 0 || cu.ModuleGetFunction( &s_kernel, s_mod, "TraceVisKernel" ) != 0 )
	{
		Warning( "[VRAD GPU] chargement du noyau PTX refuse (%d) : repli CPU.\n", r );
		FreeAll(); g_bUseCUDA = false; return false;
	}

	// --- Geometrie ---
	int nNodes = g_RtEnv.OptimizedKDTree.Count();
	int nIdx = g_RtEnv.TriangleIndexList.Count();
	int nTris = g_RtEnv.OptimizedTriangleList.Count();
	if ( nNodes == 0 )
	{
		Warning( "[VRAD GPU] k-d tree vide : repli CPU.\n" );
		FreeAll(); g_bUseCUDA = false; return false;
	}

	int *children = (int *)malloc( nNodes * sizeof( int ) );
	float *split = (float *)malloc( nNodes * sizeof( float ) );
	for ( int i = 0; i < nNodes; ++i )
	{
		children[i] = g_RtEnv.OptimizedKDTree[i].Children;
		split[i] = g_RtEnv.OptimizedKDTree[i].SplittingPlaneValue;
	}
	GpuTri *tris = (GpuTri *)malloc( ( nTris ? nTris : 1 ) * sizeof( GpuTri ) );
	for ( int i = 0; i < nTris; ++i )
	{
		const TriIntersectData_t &t = g_RtEnv.OptimizedTriangleList[i].m_Data.m_IntersectData;
		tris[i].nx = t.m_flNx; tris[i].ny = t.m_flNy; tris[i].nz = t.m_flNz; tris[i].d = t.m_flD;
		for ( int k = 0; k < 6; ++k ) tris[i].e[k] = t.m_ProjectedEdgeEquations[k];
		tris[i].cs0 = t.m_nCoordSelect0; tris[i].cs1 = t.m_nCoordSelect1;
		tris[i].flags = t.m_nFlags; tris[i].id = t.m_nTriangleID;
	}
	float bounds[6] = {
		g_RtEnv.m_MinBound.x, g_RtEnv.m_MinBound.y, g_RtEnv.m_MinBound.z,
		g_RtEnv.m_MaxBound.x, g_RtEnv.m_MaxBound.y, g_RtEnv.m_MaxBound.z };

	bool ok = Upload( &s_dChildren, children, nNodes * sizeof( int ) )
		&& Upload( &s_dSplit, split, nNodes * sizeof( float ) )
		&& Upload( &s_dTriIdx, nIdx ? (const void *)g_RtEnv.TriangleIndexList.Base() : (const void *)children, nIdx * sizeof( int ) )
		&& Upload( &s_dTris, tris, ( nTris ? nTris : 1 ) * sizeof( GpuTri ) )
		&& Upload( &s_dBounds, bounds, sizeof( bounds ) );
	free( children ); free( split ); free( tris );
	if ( !ok )
	{
		Warning( "[VRAD GPU] envoi de la geometrie impossible : repli CPU.\n" );
		FreeAll(); g_bUseCUDA = false; return false;
	}

	Msg( "[VRAD GPU] CUDA : %s | %d noeuds kd, %d triangles\n", name, nNodes, nTris );
	return true;
}

void VradGPU_Shutdown()
{
	if ( s_ctx )
		FreeAll();
}

bool VradGPU_TraceRays( const float *rays, int numRays, unsigned char *out, int mode, int skipId )
{
	if ( !s_ctx || numRays <= 0 )
		return numRays <= 0;

	// Choix d'un emplacement libre (flux + tampons propres) : les threads recouvrent copies et calcul.
	int start = (int)( (unsigned)InterlockedIncrement( &s_slotCounter ) % NUM_GPU_SLOTS );
	int si = -1;
	for ( int k = 0; k < NUM_GPU_SLOTS; ++k )
	{
		int c = ( start + k ) % NUM_GPU_SLOTS;
		if ( TryEnterCriticalSection( &s_slots[c].cs ) ) { si = c; break; }
	}
	if ( si < 0 )
	{
		si = start;
		EnterCriticalSection( &s_slots[si].cs );
	}
	GpuSlot_t &sl = s_slots[si];
	bool ok = ( cu.CtxSetCurrent( s_ctx ) == 0 );

	if ( ok && !sl.stream )
		ok = cu.StreamCreate( &sl.stream, 0 ) == 0;
	if ( ok && (size_t)numRays > sl.cap )
	{
		if ( sl.dRays ) cu.MemFree( sl.dRays );
		if ( sl.dOut ) cu.MemFree( sl.dOut );
		sl.dRays = sl.dOut = 0;
		size_t cap = (size_t)numRays * 2;
		if ( cap < 65536 ) cap = 65536;
		ok = cu.MemAlloc( &sl.dRays, cap * 6 * sizeof( float ) ) == 0
			&& cu.MemAlloc( &sl.dOut, cap ) == 0;
		sl.cap = ok ? cap : 0;
	}
	if ( ok )
		ok = cu.MemcpyHtoDAsync( sl.dRays, rays, (size_t)numRays * 6 * sizeof( float ), sl.stream ) == 0;
	if ( ok )
	{
		int n = numRays;
		void *args[] = { &s_dChildren, &s_dSplit, &s_dTriIdx, &s_dTris, &s_dBounds, &sl.dRays, &sl.dOut, &n, &mode, &skipId };
		unsigned int block = 128, grid = ( n + block - 1 ) / block;
		ok = cu.LaunchKernel( s_kernel, grid, 1, 1, block, 1, 1, 0, sl.stream, args, NULL ) == 0;
	}
	if ( ok )
		ok = cu.MemcpyDtoHAsync( out, sl.dOut, numRays, sl.stream ) == 0 && cu.StreamSynchronize( sl.stream ) == 0;
	LeaveCriticalSection( &sl.cs );

	if ( ok )
	{
		InterlockedExchangeAdd64( &s_totalRays, numRays );
		InterlockedIncrement( &s_totalBatches );
	}
	return ok;
}

// ---------------------------------------------------------------------------
// Eclairage indirect des props statiques : donnees BSP + lightmaps sur le GPU.
// Les structures doivent correspondre a celles de vrad_gpu_kernel.cu.
// ---------------------------------------------------------------------------
struct GpuBspNode { float nx, ny, nz, dist; int type; int ch0, ch1; int firstface, numfaces; };
struct GpuBspLeaf { int firstleafface, numleaffaces, hasDisp, pad; };
struct GpuBspFace
{
	float pn[4];
	float lv[8];
	int mins[2];
	int size[2];
	int flags;
	int skip;
	int onNode;
	int lightofs;
	int style0;
	float refl[3];
	int skyStart;
	int skyCount;
};
struct IndirectData
{
	CUdeviceptr nodes, leaves, leafFaces, faces, skyPts, light, pow2;
	int lightBytes;
	int pad;
};

extern void VradDisp_MarkLeaves( unsigned char *pLeafFlags );

bool VradGPU_IndirectPrepare()
{
	// Table identique a celle du CPU (TexLightToLinear : c * power2_n[exp + 128], 2^(i-128)/255).
	extern float power2_n[256];
	const float *s_pow2Local = power2_n;
	if ( s_indReady )
		return true;
	if ( !s_ctx || !pdlightdata || numnodes <= 0 || numleafs <= 0 || numfaces <= 0 )
		return false;
	if ( cu.CtxSetCurrent( s_ctx ) != 0 )
		return false;
	if ( cu.ModuleGetFunction( &s_indKernel, s_mod, "IndirectKernel" ) != 0 )
		return false;

	GpuBspNode *nodes = (GpuBspNode *)malloc( numnodes * sizeof( GpuBspNode ) );
	for ( int i = 0; i < numnodes; ++i )
	{
		const dplane_t &p = dplanes[dnodes[i].planenum];
		nodes[i].nx = p.normal.x; nodes[i].ny = p.normal.y; nodes[i].nz = p.normal.z;
		nodes[i].dist = p.dist; nodes[i].type = p.type;
		nodes[i].ch0 = dnodes[i].children[0]; nodes[i].ch1 = dnodes[i].children[1];
		nodes[i].firstface = dnodes[i].firstface; nodes[i].numfaces = dnodes[i].numfaces;
	}

	unsigned char *dispLeaf = (unsigned char *)calloc( numleafs, 1 );
	VradDisp_MarkLeaves( dispLeaf );
	GpuBspLeaf *leaves = (GpuBspLeaf *)malloc( numleafs * sizeof( GpuBspLeaf ) );
	for ( int i = 0; i < numleafs; ++i )
	{
		leaves[i].firstleafface = dleafs[i].firstleafface;
		leaves[i].numleaffaces = dleafs[i].numleaffaces;
		leaves[i].hasDisp = dispLeaf[i];
		leaves[i].pad = 0;
	}
	free( dispLeaf );

	CUtlVector<float> sky;
	GpuBspFace *faces = (GpuBspFace *)calloc( numfaces, sizeof( GpuBspFace ) );
	for ( int i = 0; i < numfaces; ++i )
	{
		dface_t *f = &g_pFaces[i];
		GpuBspFace &g = faces[i];
		const dplane_t &p = dplanes[f->planenum];
		g.pn[0] = p.normal.x; g.pn[1] = p.normal.y; g.pn[2] = p.normal.z; g.pn[3] = p.dist;
		texinfo_t *t = ( f->texinfo >= 0 && f->texinfo < texinfo.Count() ) ? &texinfo[f->texinfo] : NULL;
		g.skip = ( f->dispinfo != -1 || !t ) ? 1 : 0;
		g.onNode = f->onNode;
		g.lightofs = f->lightofs;
		g.style0 = f->styles[0];
		g.mins[0] = f->m_LightmapTextureMinsInLuxels[0]; g.mins[1] = f->m_LightmapTextureMinsInLuxels[1];
		g.size[0] = f->m_LightmapTextureSizeInLuxels[0]; g.size[1] = f->m_LightmapTextureSizeInLuxels[1];
		if ( t )
		{
			for ( int k = 0; k < 4; ++k ) { g.lv[k] = t->lightmapVecsLuxelsPerWorldUnits[0][k]; g.lv[4 + k] = t->lightmapVecsLuxelsPerWorldUnits[1][k]; }
			g.flags = t->flags;
			g.refl[0] = dtexdata[t->texdata].reflectivity.x;
			g.refl[1] = dtexdata[t->texdata].reflectivity.y;
			g.refl[2] = dtexdata[t->texdata].reflectivity.z;
			if ( !g.skip && g.onNode && ( t->flags & SURF_SKY ) )
			{
				Vector zeroOrg( 0.0f, 0.0f, 0.0f );
				winding_t *w = WindingFromFace( f, zeroOrg );
				if ( w )
				{
					g.skyStart = sky.Count() / 3;
					g.skyCount = w->numpoints;
					for ( int k = 0; k < w->numpoints; ++k )
					{
						sky.AddToTail( w->p[k].x ); sky.AddToTail( w->p[k].y ); sky.AddToTail( w->p[k].z );
					}
					FreeWinding( w );
				}
			}
		}
	}

	s_indLightBytes = pdlightdata->Count();
	int *lf = (int *)calloc( numleaffaces ? numleaffaces : 1, sizeof( int ) );
	for ( int i = 0; i < numleaffaces; ++i )
		lf[i] = dleaffaces[i];
	bool ok = Upload( &s_indNodes, nodes, numnodes * sizeof( GpuBspNode ) )
		&& Upload( &s_indLeaves, leaves, numleafs * sizeof( GpuBspLeaf ) )
		&& Upload( &s_indLeafFaces, lf, ( numleaffaces ? numleaffaces : 1 ) * sizeof( int ) )
		&& Upload( &s_indFaces, faces, numfaces * sizeof( GpuBspFace ) )
		&& Upload( &s_indSky, sky.Count() ? (const void *)sky.Base() : (const void *)nodes, sky.Count() * sizeof( float ) )
		&& Upload( &s_indLight, pdlightdata->Base(), s_indLightBytes )
		&& Upload( &s_indPow2, s_pow2Local, 256 * sizeof( float ) );
	free( lf ); free( nodes ); free( leaves ); free( faces );
	s_indReady = ok;
	if ( !ok )
		Warning( "[VRAD GPU] envoi des donnees BSP (indirect) impossible : repli CPU.\n" );
	return ok;
}

// rays : 6 floats par rayon (origine, delta). out : 4 floats par rayon (rgb, w).
// w = 0 : aucune contribution, 1 : contribution, 2 : a refaire sur le CPU.
bool VradGPU_Indirect( const float *rays, int numRays, float *out )
{
	if ( !s_ctx || !s_indReady || numRays <= 0 )
		return false;

	int start = (int)( (unsigned)InterlockedIncrement( &s_slotCounter ) % NUM_GPU_SLOTS );
	int si = -1;
	for ( int k = 0; k < NUM_GPU_SLOTS; ++k )
	{
		int c = ( start + k ) % NUM_GPU_SLOTS;
		if ( TryEnterCriticalSection( &s_slots[c].cs ) ) { si = c; break; }
	}
	if ( si < 0 )
	{
		si = start;
		EnterCriticalSection( &s_slots[si].cs );
	}
	GpuSlot_t &sl = s_slots[si];
	bool ok = ( cu.CtxSetCurrent( s_ctx ) == 0 );
	if ( ok && !sl.stream )
		ok = cu.StreamCreate( &sl.stream, 0 ) == 0;
	if ( ok && (size_t)numRays > sl.indCap )
	{
		if ( sl.dIndRays ) cu.MemFree( sl.dIndRays );
		if ( sl.dIndOut ) cu.MemFree( sl.dIndOut );
		sl.dIndRays = sl.dIndOut = 0;
		size_t cap = (size_t)numRays * 2;
		if ( cap < 16384 ) cap = 16384;
		ok = cu.MemAlloc( &sl.dIndRays, cap * 6 * sizeof( float ) ) == 0
			&& cu.MemAlloc( &sl.dIndOut, cap * 4 * sizeof( float ) ) == 0;
		sl.indCap = ok ? cap : 0;
	}
	if ( ok )
		ok = cu.MemcpyHtoDAsync( sl.dIndRays, rays, (size_t)numRays * 6 * sizeof( float ), sl.stream ) == 0;
	if ( ok )
	{
		IndirectData D;
		D.nodes = s_indNodes; D.leaves = s_indLeaves; D.leafFaces = s_indLeafFaces; D.faces = s_indFaces;
		D.skyPts = s_indSky; D.light = s_indLight; D.pow2 = s_indPow2;
		D.lightBytes = s_indLightBytes; D.pad = 0;
		int n = numRays;
		void *args[] = { &D, &sl.dIndRays, &sl.dIndOut, &n };
		unsigned int block = 64, grid = ( n + block - 1 ) / block;
		ok = cu.LaunchKernel( s_indKernel, grid, 1, 1, block, 1, 1, 0, sl.stream, args, NULL ) == 0;
	}
	if ( ok )
		ok = cu.MemcpyDtoHAsync( out, sl.dIndOut, (size_t)numRays * 4 * sizeof( float ), sl.stream ) == 0
			&& cu.StreamSynchronize( sl.stream ) == 0;
	LeaveCriticalSection( &sl.cs );
	return ok;
}

void VradGPU_AddVerifyStats( int rays, int mismatches )
{
	InterlockedExchangeAdd( &s_verifyRays, rays );
	InterlockedExchangeAdd( &s_verifyMismatch, mismatches );
}

void VradGPU_PrintStats()
{
	Msg( "[VRAD GPU] %lld rayons en %ld lots\n", (long long)s_totalRays, s_totalBatches );
	Msg( "[VRAD GPU] lumiere directe : %ld appels rejoues depuis le GPU, %ld repasses au CPU\n", s_directRays / 4, s_directMiss );
	if ( g_bCUDAVerify )
		Msg( "[VRAD GPU] verification : %ld rayons compares, %ld ecarts CPU/GPU\n", s_verifyRays, s_verifyMismatch );
}
