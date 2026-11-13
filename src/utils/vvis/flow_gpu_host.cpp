// flow_gpu_host.cpp
//
// Partie CPU du calcul de visibilite sur GPU (CUDA ou OpenCL).
//
// Place dans VVIS :
//   BasePortalVis (CPU)               -> portalflood, un sur-ensemble grossier
//   GPU_RunPortalFlowRaycast (GPU)    -> portalvis, remplace PortalFlow
//   ClusterMerge / CalcPAS (CPU)      -> inchanges
//
// Les deux API sont chargees dynamiquement (LoadLibrary), sans SDK ni lien
// statique : nvcuda.dll (pilote NVIDIA) et OpenCL.dll (pilotes NVIDIA, AMD,
// Intel). Les types et constantes OpenCL utilises sont redeclares ici. Sans GPU
// utilisable, GPU_Init() renvoie false et VVIS execute le flow CPU d'origine.
//
// Choix du backend :
//   - CUDA d'abord, si un GPU NVIDIA compute >= 7.5 est present. Le noyau PTX
//     (flow_gpu_kernel_ptx.h, genere par build_ptx.ps1) est compile par le pilote.
//   - sinon OpenCL.
//   - VVIS_GPU_BACKEND=opencl force OpenCL (option -opencl de vvis).
//
// Choix de la carte : VVIS_GPU_DEVICE=<n> impose la n-ieme carte ; sinon on prend
// celle qui a le plus de compute units (en pratique la carte dediee, pas l'iGPU).

#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "vis.h"
#include "flow_gpu.h"
#include "flow_gpu_kernel.h"
#include "flow_gpu_kernel_ptx.h"

// --- Sous-ensemble de l'API OpenCL 1.2 ---------------------------------------

typedef struct _cl_platform_id   *cl_platform_id;
typedef struct _cl_device_id     *cl_device_id;
typedef struct _cl_context       *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_mem           *cl_mem;
typedef struct _cl_program       *cl_program;
typedef struct _cl_kernel        *cl_kernel;
typedef int                cl_int;
typedef unsigned int       cl_uint;
typedef unsigned long long cl_ulong;
typedef cl_ulong           cl_bitfield;
typedef cl_bitfield        cl_device_type;
typedef cl_bitfield        cl_mem_flags;
typedef cl_bitfield        cl_command_queue_properties;
typedef intptr_t           cl_context_properties;
typedef cl_uint            cl_device_info;
typedef cl_uint            cl_platform_info;
typedef cl_uint            cl_program_build_info;
typedef cl_uint            cl_bool;

#define CL_SUCCESS                  0
#define CL_DEVICE_TYPE_GPU          (1 << 2)
#define CL_DEVICE_NAME              0x102B
#define CL_DEVICE_VENDOR            0x102C
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_DEVICE_GLOBAL_MEM_SIZE   0x101F
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE 0x1010
#define CL_PROGRAM_BUILD_LOG        0x1183
#define CL_MEM_READ_WRITE           (1 << 0)
#define CL_MEM_READ_ONLY            (1 << 2)
#define CL_MEM_COPY_HOST_PTR        (1 << 5)
#define CL_TRUE                     1

#define CLAPI __stdcall
typedef cl_int  (CLAPI *PFN_clGetPlatformIDs)( cl_uint, cl_platform_id *, cl_uint * );
typedef cl_int  (CLAPI *PFN_clGetDeviceIDs)( cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint * );
typedef cl_int  (CLAPI *PFN_clGetDeviceInfo)( cl_device_id, cl_device_info, size_t, void *, size_t * );
typedef cl_context (CLAPI *PFN_clCreateContext)( const cl_context_properties *, cl_uint, const cl_device_id *, void *, void *, cl_int * );
typedef cl_command_queue (CLAPI *PFN_clCreateCommandQueue)( cl_context, cl_device_id, cl_command_queue_properties, cl_int * );
typedef cl_program (CLAPI *PFN_clCreateProgramWithSource)( cl_context, cl_uint, const char **, const size_t *, cl_int * );
typedef cl_int  (CLAPI *PFN_clBuildProgram)( cl_program, cl_uint, const cl_device_id *, const char *, void *, void * );
typedef cl_int  (CLAPI *PFN_clGetProgramBuildInfo)( cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t * );
typedef cl_kernel (CLAPI *PFN_clCreateKernel)( cl_program, const char *, cl_int * );
typedef cl_mem  (CLAPI *PFN_clCreateBuffer)( cl_context, cl_mem_flags, size_t, void *, cl_int * );
typedef cl_int  (CLAPI *PFN_clSetKernelArg)( cl_kernel, cl_uint, size_t, const void * );
typedef cl_int  (CLAPI *PFN_clEnqueueNDRangeKernel)( cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const void *, void * );
typedef cl_int  (CLAPI *PFN_clEnqueueReadBuffer)( cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const void *, void * );
typedef cl_int  (CLAPI *PFN_clEnqueueFillBuffer)( cl_command_queue, cl_mem, const void *, size_t, size_t, size_t, cl_uint, const void *, void * );
typedef cl_int  (CLAPI *PFN_clFinish)( cl_command_queue );
typedef cl_int  (CLAPI *PFN_clReleaseMemObject)( cl_mem );
typedef cl_int  (CLAPI *PFN_clReleaseKernel)( cl_kernel );
typedef cl_int  (CLAPI *PFN_clReleaseProgram)( cl_program );
typedef cl_int  (CLAPI *PFN_clReleaseCommandQueue)( cl_command_queue );
typedef cl_int  (CLAPI *PFN_clReleaseContext)( cl_context );

static struct
{
	HMODULE                       hLib;
	PFN_clGetPlatformIDs          GetPlatformIDs;
	PFN_clGetDeviceIDs            GetDeviceIDs;
	PFN_clGetDeviceInfo           GetDeviceInfo;
	PFN_clCreateContext           CreateContext;
	PFN_clCreateCommandQueue      CreateCommandQueue;
	PFN_clCreateProgramWithSource CreateProgramWithSource;
	PFN_clBuildProgram            BuildProgram;
	PFN_clGetProgramBuildInfo     GetProgramBuildInfo;
	PFN_clCreateKernel            CreateKernel;
	PFN_clCreateBuffer            CreateBuffer;
	PFN_clSetKernelArg            SetKernelArg;
	PFN_clEnqueueNDRangeKernel    EnqueueNDRangeKernel;
	PFN_clEnqueueReadBuffer       EnqueueReadBuffer;
	PFN_clEnqueueFillBuffer       EnqueueFillBuffer;
	PFN_clFinish                  Finish;
	PFN_clReleaseMemObject        ReleaseMemObject;
	PFN_clReleaseKernel           ReleaseKernel;
	PFN_clReleaseProgram          ReleaseProgram;
	PFN_clReleaseCommandQueue     ReleaseCommandQueue;
	PFN_clReleaseContext          ReleaseContext;
} cl;

static cl_device_id     s_device  = NULL;
static cl_context       s_context = NULL;
static cl_command_queue s_queue   = NULL;
static cl_program       s_program = NULL;
static cl_kernel        s_kernel  = NULL;

static bool LoadOpenCL( void )
{
	cl.hLib = LoadLibraryA( "OpenCL.dll" );
	if ( !cl.hLib )
		return false;

#define CL_LOAD( name ) \
	cl.name = (PFN_cl##name)GetProcAddress( cl.hLib, "cl" #name ); \
	if ( !cl.name ) return false;

	CL_LOAD( GetPlatformIDs );
	CL_LOAD( GetDeviceIDs );
	CL_LOAD( GetDeviceInfo );
	CL_LOAD( CreateContext );
	CL_LOAD( CreateCommandQueue );
	CL_LOAD( CreateProgramWithSource );
	CL_LOAD( BuildProgram );
	CL_LOAD( GetProgramBuildInfo );
	CL_LOAD( CreateKernel );
	CL_LOAD( CreateBuffer );
	CL_LOAD( SetKernelArg );
	CL_LOAD( EnqueueNDRangeKernel );
	CL_LOAD( EnqueueReadBuffer );
	CL_LOAD( EnqueueFillBuffer );
	CL_LOAD( Finish );
	CL_LOAD( ReleaseMemObject );
	CL_LOAD( ReleaseKernel );
	CL_LOAD( ReleaseProgram );
	CL_LOAD( ReleaseCommandQueue );
	CL_LOAD( ReleaseContext );
#undef CL_LOAD
	return true;
}

// Liste tous les GPU de toutes les plateformes OpenCL.
static int EnumerateGPUs( cl_device_id *pOut, int maxOut )
{
	cl_platform_id platforms[16];
	cl_uint numPlatforms = 0;
	if ( cl.GetPlatformIDs( 16, platforms, &numPlatforms ) != CL_SUCCESS )
		return 0;
	if ( numPlatforms > 16 )
		numPlatforms = 16;

	int count = 0;
	for ( cl_uint i = 0; i < numPlatforms && count < maxOut; ++i )
	{
		cl_uint n = 0;
		if ( cl.GetDeviceIDs( platforms[i], CL_DEVICE_TYPE_GPU, (cl_uint)( maxOut - count ), pOut + count, &n ) != CL_SUCCESS )
			continue;
		count += ( n < (cl_uint)( maxOut - count ) ) ? (int)n : maxOut - count;
	}
	return count;
}

// --- Backend CUDA (API driver, chargee dynamiquement) -----------------------------

typedef int                CUresult;
typedef int                CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st    *CUcontext;
typedef struct CUmod_st    *CUmodule;
typedef struct CUfunc_st   *CUfunction;
typedef struct CUstream_st *CUstream;

#define CU_DEV_ATTR_MULTIPROCESSOR_COUNT 16
#define CU_DEV_ATTR_CC_MAJOR             75
#define CU_DEV_ATTR_CC_MINOR             76

typedef CUresult (CLAPI *PFN_cuInit)( unsigned int );
typedef CUresult (CLAPI *PFN_cuDeviceGetCount)( int * );
typedef CUresult (CLAPI *PFN_cuDeviceGet)( CUdevice *, int );
typedef CUresult (CLAPI *PFN_cuDeviceGetName)( char *, int, CUdevice );
typedef CUresult (CLAPI *PFN_cuDeviceTotalMem)( size_t *, CUdevice );
typedef CUresult (CLAPI *PFN_cuDeviceGetAttribute)( int *, int, CUdevice );
typedef CUresult (CLAPI *PFN_cuCtxCreate)( CUcontext *, unsigned int, CUdevice );
typedef CUresult (CLAPI *PFN_cuCtxDestroy)( CUcontext );
typedef CUresult (CLAPI *PFN_cuCtxSynchronize)( void );
typedef CUresult (CLAPI *PFN_cuModuleLoadData)( CUmodule *, const void * );
typedef CUresult (CLAPI *PFN_cuModuleUnload)( CUmodule );
typedef CUresult (CLAPI *PFN_cuModuleGetFunction)( CUfunction *, CUmodule, const char * );
typedef CUresult (CLAPI *PFN_cuMemAlloc)( CUdeviceptr *, size_t );
typedef CUresult (CLAPI *PFN_cuMemFree)( CUdeviceptr );
typedef CUresult (CLAPI *PFN_cuMemcpyHtoD)( CUdeviceptr, const void *, size_t );
typedef CUresult (CLAPI *PFN_cuMemcpyDtoH)( void *, CUdeviceptr, size_t );
typedef CUresult (CLAPI *PFN_cuMemsetD8)( CUdeviceptr, unsigned char, size_t );
typedef CUresult (CLAPI *PFN_cuLaunchKernel)( CUfunction, unsigned int, unsigned int, unsigned int,
	unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void **, void ** );

static struct
{
	HMODULE                  hLib;
	PFN_cuInit               Init;
	PFN_cuDeviceGetCount     DeviceGetCount;
	PFN_cuDeviceGet          DeviceGet;
	PFN_cuDeviceGetName      DeviceGetName;
	PFN_cuDeviceTotalMem     DeviceTotalMem;
	PFN_cuDeviceGetAttribute DeviceGetAttribute;
	PFN_cuCtxCreate          CtxCreate;
	PFN_cuCtxDestroy         CtxDestroy;
	PFN_cuCtxSynchronize     CtxSynchronize;
	PFN_cuModuleLoadData     ModuleLoadData;
	PFN_cuModuleUnload       ModuleUnload;
	PFN_cuModuleGetFunction  ModuleGetFunction;
	PFN_cuMemAlloc           MemAlloc;
	PFN_cuMemFree            MemFree;
	PFN_cuMemcpyHtoD         MemcpyHtoD;
	PFN_cuMemcpyDtoH         MemcpyDtoH;
	PFN_cuMemsetD8           MemsetD8;
	PFN_cuLaunchKernel       LaunchKernel;
} cu;

enum GPUBackend { BACKEND_NONE = 0, BACKEND_CUDA, BACKEND_OPENCL };
static GPUBackend  s_backend   = BACKEND_NONE;
static CUcontext   s_cuContext = NULL;
static CUmodule    s_cuModule  = NULL;
static CUfunction  s_cuKernel  = NULL;

static bool LoadCUDA( void )
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
	CU_LOAD( DeviceTotalMem, "cuDeviceTotalMem_v2" );
	CU_LOAD( DeviceGetAttribute, "cuDeviceGetAttribute" );
	CU_LOAD( CtxCreate, "cuCtxCreate_v2" );
	CU_LOAD( CtxDestroy, "cuCtxDestroy_v2" );
	CU_LOAD( CtxSynchronize, "cuCtxSynchronize" );
	CU_LOAD( ModuleLoadData, "cuModuleLoadData" );
	CU_LOAD( ModuleUnload, "cuModuleUnload" );
	CU_LOAD( ModuleGetFunction, "cuModuleGetFunction" );
	CU_LOAD( MemAlloc, "cuMemAlloc_v2" );
	CU_LOAD( MemFree, "cuMemFree_v2" );
	CU_LOAD( MemcpyHtoD, "cuMemcpyHtoD_v2" );
	CU_LOAD( MemcpyDtoH, "cuMemcpyDtoH_v2" );
	CU_LOAD( MemsetD8, "cuMemsetD8_v2" );
	CU_LOAD( LaunchKernel, "cuLaunchKernel" );
#undef CU_LOAD
	return true;
}

static void ShutdownCUDA( void )
{
	if ( s_cuModule )  { cu.ModuleUnload( s_cuModule );  s_cuModule = NULL; }
	if ( s_cuContext ) { cu.CtxDestroy( s_cuContext );   s_cuContext = NULL; }
	s_cuKernel = NULL;
	if ( cu.hLib )
	{
		FreeLibrary( cu.hLib );
		memset( &cu, 0, sizeof( cu ) );
	}
}

// Tente d'initialiser CUDA : GPU NVIDIA compute >= 7.5, contexte, module PTX.
// En cas d'echec (pas de pilote, GPU trop ancien, PTX refuse par un pilote
// trop vieux...) retourne false sans rien laisser d'alloue : OpenCL prend le relais.
static bool InitCUDA( void )
{
	if ( !LoadCUDA() )
	{
		ShutdownCUDA();
		return false;
	}
	int count = 0;
	if ( cu.Init( 0 ) != 0 || cu.DeviceGetCount( &count ) != 0 || count <= 0 )
	{
		ShutdownCUDA();
		return false;
	}

	int chosen = -1, bestSM = -1;
	const char *penv = getenv( "VVIS_GPU_DEVICE" );
	int forced = ( penv && *penv ) ? atoi( penv ) : -1;
	for ( int i = 0; i < count; ++i )
	{
		CUdevice dev;
		int major = 0, minor = 0, sm = 0;
		if ( cu.DeviceGet( &dev, i ) != 0 )
			continue;
		cu.DeviceGetAttribute( &major, CU_DEV_ATTR_CC_MAJOR, dev );
		cu.DeviceGetAttribute( &minor, CU_DEV_ATTR_CC_MINOR, dev );
		cu.DeviceGetAttribute( &sm, CU_DEV_ATTR_MULTIPROCESSOR_COUNT, dev );
		if ( major < 7 || ( major == 7 && minor < 5 ) )
			continue;
		if ( forced >= 0 )
		{
			if ( i == forced ) { chosen = i; break; }
			continue;
		}
		if ( sm > bestSM ) { bestSM = sm; chosen = i; }
	}
	if ( chosen < 0 )
	{
		ShutdownCUDA();
		return false;
	}

	CUdevice dev;
	char name[256] = { 0 };
	size_t mem = 0;
	cu.DeviceGet( &dev, chosen );
	cu.DeviceGetName( name, (int)sizeof( name ) - 1, dev );
	cu.DeviceTotalMem( &mem, dev );

	if ( cu.CtxCreate( &s_cuContext, 0, dev ) != 0 )
	{
		ShutdownCUDA();
		return false;
	}
	CUresult r = cu.ModuleLoadData( &s_cuModule, g_RaycastPTX );
	if ( r != 0 || cu.ModuleGetFunction( &s_cuKernel, s_cuModule, "RaycastStabKernel" ) != 0 )
	{
		fprintf( stderr, "[VVIS GPU] CUDA : chargement du noyau PTX refuse (%d), pilote trop ancien ? Repli OpenCL.\n", r );
		ShutdownCUDA();
		return false;
	}
	printf( "[VVIS GPU] CUDA device %d/%d : %s (%llu MB)\n", chosen, count, name, (unsigned long long)( mem / ( 1024 * 1024 ) ) );
	return true;
}

// Equivalent CUDA de l'etape "buffers + noyau + lecture" d'OpenCL. Meme noyau,
// memes tranches (anti-TDR), memes resultats.
static bool RunRaycastCUDA( const GPUPortal *hPortals, size_t portalsBytes, int numPortals,
	const GPULeaf *hLeafs, size_t leafsBytes,
	const int *hLeafPortals, size_t leafPortalsBytes,
	const float *hWinding, size_t windingBytes,
	const unsigned char *hFlood, unsigned char *hVis, size_t visBytesPadded,
	int portalbytes_, int samples_, int maxSteps_ )
{
	CUdeviceptr dPortals = 0, dLeafs = 0, dLeafPortals = 0, dWinding = 0, dFlood = 0, dVis = 0;
	bool ok = true;
	int nLeafs = portalclusters;

#define CU_TRY( expr, what ) \
	if ( ok ) { CUresult r_ = ( expr ); if ( r_ != 0 ) { fprintf( stderr, "[VVIS GPU] CUDA %s a echoue (%d).\n", what, r_ ); ok = false; } }
#define CU_UPLOAD( var, host, bytes, what ) \
	CU_TRY( cu.MemAlloc( &var, bytes ), what ); \
	CU_TRY( cu.MemcpyHtoD( var, host, bytes ), what );

	CU_UPLOAD( dPortals, hPortals, portalsBytes, "portals" );
	CU_UPLOAD( dLeafs, hLeafs, leafsBytes, "leafs" );
	CU_UPLOAD( dLeafPortals, hLeafPortals, leafPortalsBytes, "leaf_portals" );
	CU_UPLOAD( dWinding, hWinding, windingBytes, "winding_pts" );
	CU_UPLOAD( dFlood, hFlood, visBytesPadded, "portalflood" );
	CU_TRY( cu.MemAlloc( &dVis, visBytesPadded ), "portalvis" );
	CU_TRY( cu.MemsetD8( dVis, 0, visBytesPadded ), "memset" );

	// Statistiques de diagnostic : actives seulement si VVIS_GPU_STATS=1.
	const int statCount = 4 + 16;
	const char *stenv = getenv( "VVIS_GPU_STATS" );
	const bool wantStats = stenv && *stenv == '1';
	CUdeviceptr dStats = 0;
	if ( wantStats )
	{
		CU_TRY( cu.MemAlloc( &dStats, sizeof( unsigned long long ) * statCount ), "stats" );
		CU_TRY( cu.MemsetD8( dStats, 0, sizeof( unsigned long long ) * statCount ), "stats memset" );
	}

	const unsigned long long npairs = (unsigned long long)numPortals * (unsigned long long)numPortals;
	const unsigned int block = 128;
	const unsigned long long chunk = (unsigned long long)block * 16384;
	for ( unsigned long long start = 0; ok && start < npairs; start += chunk )
	{
		unsigned long long remaining = npairs - start;
		unsigned long long n = remaining < chunk ? remaining : chunk;
		unsigned int grid = (unsigned int)( ( n + block - 1 ) / block );
		unsigned long long startArg = start;
		void *args[] = { &dPortals, &numPortals, &dLeafs, &nLeafs, &dLeafPortals, &dWinding,
			&dFlood, &dVis, &portalbytes_, &samples_, &maxSteps_, &startArg, &dStats };
		CU_TRY( cu.LaunchKernel( s_cuKernel, grid, 1, 1, block, 1, 1, 0, NULL, args, NULL ), "cuLaunchKernel" );
		CU_TRY( cu.CtxSynchronize(), "cuCtxSynchronize" );
	}
	CU_TRY( cu.MemcpyDtoH( hVis, dVis, visBytesPadded ), "cuMemcpyDtoH" );

	if ( ok && wantStats )
	{
		unsigned long long st[4 + 16] = { 0 };
		CU_TRY( cu.MemcpyDtoH( st, dStats, sizeof( st ) ), "stats DtoH" );
		printf( "[VVIS GPU][STATS] paires testees %llu, visibles %llu, rayons %llu (dont %llu pour paires invisibles)\n",
			st[0], st[1], st[2], st[3] );
		printf( "[VVIS GPU][STATS] profondeur de mort des rayons rates (portails traverses : rayons) :\n" );
		for ( int h = 0; h < 16; ++h )
			if ( st[4 + h] )
				printf( "[VVIS GPU][STATS]   %s%2d : %llu\n", h == 15 ? ">=" : "  ", h, st[4 + h] );
	}

#undef CU_UPLOAD
#undef CU_TRY

	if ( dPortals )     cu.MemFree( dPortals );
	if ( dLeafs )       cu.MemFree( dLeafs );
	if ( dLeafPortals ) cu.MemFree( dLeafPortals );
	if ( dWinding )     cu.MemFree( dWinding );
	if ( dFlood )       cu.MemFree( dFlood );
	if ( dVis )         cu.MemFree( dVis );
	if ( dStats )       cu.MemFree( dStats );
	return ok;
}

// --- GPU_Init ----------------------------------------------------------------

bool GPU_Init( void )
{
	// CUDA d'abord (NVIDIA), OpenCL ensuite (NVIDIA ancien, AMD, Intel).
	const char *be = getenv( "VVIS_GPU_BACKEND" );
	const bool forceOpenCL = be && !_stricmp( be, "opencl" );
	if ( !forceOpenCL && InitCUDA() )
	{
		s_backend = BACKEND_CUDA;
		return true;
	}

	if ( !LoadOpenCL() )
	{
		fprintf( stderr, "[VVIS GPU] OpenCL indisponible (OpenCL.dll absente ou incomplete). Flow CPU.\n" );
		GPU_Shutdown();
		return false;
	}

	cl_device_id devices[32];
	int numDevices = EnumerateGPUs( devices, 32 );
	if ( numDevices == 0 )
	{
		fprintf( stderr, "[VVIS GPU] Aucun GPU OpenCL detecte. Flow CPU.\n" );
		GPU_Shutdown();
		return false;
	}

	int chosen = -1;
	const char *penv = getenv( "VVIS_GPU_DEVICE" );
	if ( penv && *penv )
	{
		int v = atoi( penv );
		if ( v >= 0 && v < numDevices )
			chosen = v;
	}
	if ( chosen < 0 )
	{
		cl_uint bestCU = 0;
		for ( int i = 0; i < numDevices; ++i )
		{
			cl_uint cu = 0;
			cl.GetDeviceInfo( devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof( cu ), &cu, NULL );
			if ( chosen < 0 || cu > bestCU )
			{
				chosen = i;
				bestCU = cu;
			}
		}
	}
	s_device = devices[chosen];

	char name[256] = { 0 }, vendor[256] = { 0 };
	cl_ulong mem = 0;
	cl.GetDeviceInfo( s_device, CL_DEVICE_NAME, sizeof( name ) - 1, name, NULL );
	cl.GetDeviceInfo( s_device, CL_DEVICE_VENDOR, sizeof( vendor ) - 1, vendor, NULL );
	cl.GetDeviceInfo( s_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof( mem ), &mem, NULL );
	printf( "[VVIS GPU] OpenCL device %d/%d : %s (%s, %llu MB)\n",
		chosen, numDevices, name, vendor, (unsigned long long)( mem / ( 1024 * 1024 ) ) );

	cl_int err = CL_SUCCESS;
	s_context = cl.CreateContext( NULL, 1, &s_device, NULL, NULL, &err );
	if ( err != CL_SUCCESS )
	{
		fprintf( stderr, "[VVIS GPU] clCreateContext a echoue (%d). Flow CPU.\n", err );
		GPU_Shutdown();
		return false;
	}
	s_queue = cl.CreateCommandQueue( s_context, s_device, 0, &err );
	if ( err != CL_SUCCESS )
	{
		fprintf( stderr, "[VVIS GPU] clCreateCommandQueue a echoue (%d). Flow CPU.\n", err );
		GPU_Shutdown();
		return false;
	}

	// Division IEEE et pas de contraction FMA implicite : resultats stables
	// d'un constructeur a l'autre.
	const char *src = g_szRaycastKernelSource;
	s_program = cl.CreateProgramWithSource( s_context, 1, &src, NULL, &err );
	if ( err == CL_SUCCESS )
		err = cl.BuildProgram( s_program, 1, &s_device, "-cl-fp32-correctly-rounded-divide-sqrt", NULL, NULL );
	if ( err != CL_SUCCESS )
	{
		char log[8192] = { 0 };
		if ( s_program )
			cl.GetProgramBuildInfo( s_program, s_device, CL_PROGRAM_BUILD_LOG, sizeof( log ) - 1, log, NULL );
		fprintf( stderr, "[VVIS GPU] Compilation du noyau OpenCL echouee (%d) :\n%s\nFlow CPU.\n", err, log );
		GPU_Shutdown();
		return false;
	}
	s_kernel = cl.CreateKernel( s_program, "RaycastStabKernel", &err );
	if ( err != CL_SUCCESS )
	{
		fprintf( stderr, "[VVIS GPU] clCreateKernel a echoue (%d). Flow CPU.\n", err );
		GPU_Shutdown();
		return false;
	}
	s_backend = BACKEND_OPENCL;
	return true;
}

// --- GPU_Shutdown --------------------------------------------------------------

void GPU_Shutdown( void )
{
	ShutdownCUDA();
	s_backend = BACKEND_NONE;
	if ( !cl.hLib )
		return;
	if ( s_kernel )  { cl.ReleaseKernel( s_kernel );        s_kernel = NULL; }
	if ( s_program ) { cl.ReleaseProgram( s_program );      s_program = NULL; }
	if ( s_queue )   { cl.ReleaseCommandQueue( s_queue );   s_queue = NULL; }
	if ( s_context ) { cl.ReleaseContext( s_context );      s_context = NULL; }
	s_device = NULL;
	if ( cl.hLib )
	{
		FreeLibrary( cl.hLib );
		memset( &cl, 0, sizeof( cl ) );
	}
}

// --- GPU_RunPortalFlowRaycast ---------------------------------------------------
// Remplit portals[i].portalvis pour chaque portail, a la place de PortalFlow.
// Renvoie false, sans rien modifier, si une operation GPU echoue : VVIS lance
// alors le flow CPU d'origine.
//
// Deroulement :
//   1. Mise "a plat" des portails et des sommets de leurs windings (GPUPortal).
//   2. Mise "a plat" de la topologie leaf -> portails sortants (GPULeaf).
//   3. Copie de portalflood : c'est le filtre des paires candidates.
//   4. Pour chaque passe : lancement du noyau par tranches de paires
//      (1 thread = 1 paire), puis fusion du resultat par OU binaire.
//   5. Copie du resultat dans portal_t et marquage stat_done.
//
// Passes : passSamples[k] est le nombre max de sommets utilises par anneau a la
// passe k (points par portail = 1 + 2 * min(n, sommets)). La passe 0 teste toutes
// les paires candidates. Les passes suivantes ne reprennent que les candidates
// pas encore prouvees visibles :
//
//   candidates   = portalflood
//   passe 0      : teste candidates            -> vis0
//   passe 1      : teste candidates et pas vis0 -> vis1
//   resultat     = vis0 OU vis1 ...
//
// Une passe ne retire jamais de visibilite, elle peut seulement en retrouver.
//
// Diagnostic : VVIS_GPU_STATS=1 (CUDA seulement) affiche le nombre de rayons
// tires et la profondeur a laquelle meurent les rayons rates.

// Compte les bits a 1 d'un tableau (pour les pourcentages affiches).
static size_t PopCountBytes( const unsigned char *p, size_t n )
{
	size_t c = 0;
	for ( size_t i = 0; i < n; i++ )
	{
		unsigned v = p[i];
		while ( v ) { c += v & 1; v >>= 1; }
	}
	return c;
}

bool GPU_RunPortalFlowRaycast( int sorted_portal_count, const int *passSamples, int numPasses )
{
	const int nportals = sorted_portal_count;
	const int numPortals = g_numportals * 2;
	if ( numPasses < 1 || !passSamples )
		return false;

	// -- 1. Portails + sommets ------------------------------------------------
	int totalWindingPts = 0;
	for ( int i = 0; i < numPortals; i++ )
		if ( portals[i].winding )
			totalWindingPts += portals[i].winding->numpoints;

	GPUPortal *hPortals = (GPUPortal *)calloc( numPortals > 0 ? numPortals : 1, sizeof( GPUPortal ) );
	float *hWindingPts = (float *)malloc( sizeof( float ) * 3 * ( totalWindingPts > 0 ? totalWindingPts : 1 ) );

	int wpt = 0;
	for ( int i = 0; i < numPortals; i++ )
	{
		portal_t *p = &portals[i];
		GPUPortal &gp = hPortals[i];
		gp.plane_normal[0]    = p->plane.normal[0];
		gp.plane_normal[1]    = p->plane.normal[1];
		gp.plane_normal[2]    = p->plane.normal[2];
		gp.plane_dist         = p->plane.dist;
		gp.leaf               = p->leaf;
		gp.origin[0]          = p->origin[0];
		gp.origin[1]          = p->origin[1];
		gp.origin[2]          = p->origin[2];
		gp.radius             = p->radius;
		gp.portalflood_offset = 0;
		gp.status             = (int)p->status;
		gp.nummightsee        = p->nummightsee;
		gp.winding_start      = wpt;
		gp.winding_count      = 0;
		if ( p->winding )
		{
			gp.winding_count = p->winding->numpoints;
			for ( int k = 0; k < p->winding->numpoints; k++ )
			{
				hWindingPts[( wpt + k ) * 3 + 0] = p->winding->points[k][0];
				hWindingPts[( wpt + k ) * 3 + 1] = p->winding->points[k][1];
				hWindingPts[( wpt + k ) * 3 + 2] = p->winding->points[k][2];
			}
			wpt += p->winding->numpoints;
		}
	}

	// -- 2. Topologie leafs -> portails ------------------------------------------
	int totalLeafPortals = 0;
	for ( int l = 0; l < portalclusters; l++ )
		totalLeafPortals += leafs[l].portals.Count();

	GPULeaf *hLeafs = (GPULeaf *)calloc( portalclusters > 0 ? portalclusters : 1, sizeof( GPULeaf ) );
	int *hLeafPortals = (int *)malloc( sizeof( int ) * ( totalLeafPortals > 0 ? totalLeafPortals : 1 ) );
	int lp = 0;
	for ( int l = 0; l < portalclusters; l++ )
	{
		hLeafs[l].portals_start = lp;
		hLeafs[l].portals_count = leafs[l].portals.Count();
		for ( int k = 0; k < leafs[l].portals.Count(); k++ )
			hLeafPortals[lp++] = (int)( leafs[l].portals[k] - portals );
	}

	// -- 3. portalflood -------------------------------------------------------------
	// Taille arrondie au multiple de 4 : le noyau ecrit portalvis par mots 32 bits.
	const size_t visBytes = (size_t)numPortals * portalbytes;
	const size_t visBytesPadded = ( visBytes + 3 ) & ~(size_t)3;
	unsigned char *hFlood = (unsigned char *)calloc( visBytesPadded > 0 ? visBytesPadded : 4, 1 );
	for ( int i = 0; i < numPortals; i++ )
		if ( portals[i].portalflood )
			memcpy( hFlood + (size_t)i * portalbytes, portals[i].portalflood, portalbytes );

	unsigned char *hPortalvis = (unsigned char *)malloc( visBytesPadded > 0 ? visBytesPadded : 4 );
	const size_t visAlloc = visBytesPadded > 0 ? visBytesPadded : 4;
	unsigned char *hFloodOrig = (unsigned char *)malloc( visAlloc );
	unsigned char *hAccum = (unsigned char *)calloc( visAlloc, 1 );
	memcpy( hFloodOrig, hFlood, visAlloc );
	const size_t floodBits = PopCountBytes( hFloodOrig, visAlloc );

	const bool useCuda = ( s_backend == BACKEND_CUDA );
	bool success = true;

	for ( int pass = 0; pass < numPasses && success; pass++ )
	{
	const int samples = passSamples[pass];
	const ULONGLONG t0 = GetTickCount64();
	if ( pass > 0 )
	{
		for ( size_t b = 0; b < visAlloc; b++ )
			hFlood[b] = hFloodOrig[b] & ~hAccum[b];
	}
	const size_t candBits = PopCountBytes( hFlood, visAlloc );
	printf( "[VVIS GPU] Pass %d/%d : samples=%d, %llu paires candidates...\n", pass + 1, numPasses, samples, (unsigned long long)candBits );

	// -- Buffers -------------------------------------------------------------------
	bool cudaOk = false;
	bool ok = true;
	if ( useCuda )
	{
		cudaOk = RunRaycastCUDA( hPortals, sizeof( GPUPortal ) * ( numPortals > 0 ? numPortals : 1 ), numPortals,
			hLeafs, sizeof( GPULeaf ) * ( portalclusters > 0 ? portalclusters : 1 ),
			hLeafPortals, sizeof( int ) * ( totalLeafPortals > 0 ? totalLeafPortals : 1 ),
			hWindingPts, sizeof( float ) * 3 * ( totalWindingPts > 0 ? totalWindingPts : 1 ),
			hFlood, hPortalvis, visBytesPadded > 0 ? visBytesPadded : 4,
			portalbytes, samples, portalclusters + 1 );
		ok = false; // saute toute la partie OpenCL ci-dessous
	}
	cl_int err = CL_SUCCESS;
	cl_mem dPortals = NULL, dLeafs = NULL, dLeafPortals = NULL, dWinding = NULL, dFlood = NULL, dVis = NULL;

#define CL_TRY( expr, what ) \
	if ( ok ) { err = ( expr ); if ( err != CL_SUCCESS ) { fprintf( stderr, "[VVIS GPU] %s a echoue (%d).\n", what, err ); ok = false; } }

#define CL_BUFFER( var, flags, size, host, what ) \
	if ( ok ) { var = cl.CreateBuffer( s_context, flags, size, host, &err ); if ( err != CL_SUCCESS ) { fprintf( stderr, "[VVIS GPU] clCreateBuffer(%s) a echoue (%d).\n", what, err ); ok = false; } }

	CL_BUFFER( dPortals, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof( GPUPortal ) * ( numPortals > 0 ? numPortals : 1 ), hPortals, "portals" );
	CL_BUFFER( dLeafs, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof( GPULeaf ) * ( portalclusters > 0 ? portalclusters : 1 ), hLeafs, "leafs" );
	CL_BUFFER( dLeafPortals, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof( int ) * ( totalLeafPortals > 0 ? totalLeafPortals : 1 ), hLeafPortals, "leaf_portals" );
	CL_BUFFER( dWinding, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof( float ) * 3 * ( totalWindingPts > 0 ? totalWindingPts : 1 ), hWindingPts, "winding_pts" );
	CL_BUFFER( dFlood, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, visBytesPadded > 0 ? visBytesPadded : 4, hFlood, "portalflood" );
	CL_BUFFER( dVis, CL_MEM_READ_WRITE, visBytesPadded > 0 ? visBytesPadded : 4, NULL, "portalvis" );

	const cl_uint zero = 0;
	CL_TRY( cl.EnqueueFillBuffer( s_queue, dVis, &zero, sizeof( zero ), 0, visBytesPadded > 0 ? visBytesPadded : 4, 0, NULL, NULL ), "clEnqueueFillBuffer" );

	// -- 4. Noyau ------------------------------------------------------------------
	const int maxSteps = portalclusters + 1;
	cl_uint arg = 0;
	CL_TRY( cl.SetKernelArg( s_kernel, arg++, sizeof( cl_mem ), &dPortals ), "clSetKernelArg" );
	CL_TRY( cl.SetKernelArg( s_kernel, arg++, sizeof( int ), &numPortals ), "clSetKernelArg" );
	CL_TRY( cl.SetKernelArg( s_kernel, arg++, sizeof( cl_mem ), &dLeafs ), "clSetKernelArg" );
	CL_TRY( cl.SetKernelArg( s_kernel, arg++, sizeof( int ), &portalclusters ), "clSetKernelArg" );
	CL_TRY( cl.SetKernelArg( s_kernel, arg++, sizeof( cl_mem ), &dLeafPortals ), "clSetKernelArg" );
	CL_TRY( cl.SetKernelArg( s_kernel, arg++, sizeof( cl_mem ), &dWinding ), "clSetKernelArg" );
	CL_TRY( cl.SetKernelArg( s_kernel, arg++, sizeof( cl_mem ), &dFlood ), "clSetKernelArg" );
	CL_TRY( cl.SetKernelArg( s_kernel, arg++, sizeof( cl_mem ), &dVis ), "clSetKernelArg" );
	CL_TRY( cl.SetKernelArg( s_kernel, arg++, sizeof( int ), &portalbytes ), "clSetKernelArg" );
	CL_TRY( cl.SetKernelArg( s_kernel, arg++, sizeof( int ), &samples ), "clSetKernelArg" );
	CL_TRY( cl.SetKernelArg( s_kernel, arg++, sizeof( int ), &maxSteps ), "clSetKernelArg" );

	// Tranches de paires : chaque lancement reste court pour ne pas declencher
	// le watchdog (TDR) de Windows sur les GPU qui affichent aussi l'ecran.
	const unsigned long long npairs = (unsigned long long)numPortals * (unsigned long long)numPortals;
	const size_t localSize = 128;
	const size_t chunk = (size_t)localSize * 16384;
	for ( unsigned long long start = 0; ok && start < npairs; start += chunk )
	{
		unsigned long long remaining = npairs - start;
		size_t global = (size_t)( remaining < chunk ? remaining : chunk );
		global = ( global + localSize - 1 ) / localSize * localSize;
		size_t offset = (size_t)start;
		CL_TRY( cl.EnqueueNDRangeKernel( s_queue, s_kernel, 1, &offset, &global, &localSize, 0, NULL, NULL ), "clEnqueueNDRangeKernel" );
		CL_TRY( cl.Finish( s_queue ), "clFinish" );
	}

	// -- 5. Resultat ---------------------------------------------------------------
	CL_TRY( cl.EnqueueReadBuffer( s_queue, dVis, CL_TRUE, 0, visBytesPadded > 0 ? visBytesPadded : 4, hPortalvis, 0, NULL, NULL ), "clEnqueueReadBuffer" );

	if ( dPortals )     cl.ReleaseMemObject( dPortals );
	if ( dLeafs )       cl.ReleaseMemObject( dLeafs );
	if ( dLeafPortals ) cl.ReleaseMemObject( dLeafPortals );
	if ( dWinding )     cl.ReleaseMemObject( dWinding );
	if ( dFlood )       cl.ReleaseMemObject( dFlood );
	if ( dVis )         cl.ReleaseMemObject( dVis );

	success = useCuda ? cudaOk : ok;
	if ( success )
	{
		for ( size_t b = 0; b < visAlloc; b++ )
			hAccum[b] |= hPortalvis[b];
		const size_t visBits = PopCountBytes( hAccum, visAlloc );
		const double sec = ( GetTickCount64() - t0 ) / 1000.0;
		const int pct = floodBits ? (int)( 100.0 * visBits / floodBits ) : 100;
		char bar[21];
		for ( int k = 0; k < 20; k++ )
			bar[k] = ( k < pct / 5 ) ? '#' : '.';
		bar[20] = 0;
		printf( "[VVIS GPU][%s] Pass %d/%d [%s] %3d%% visibles / candidates initiaux (%llu/%llu) - %.2f s\n",
			useCuda ? "CUDA" : "OpenCL", pass + 1, numPasses, bar, pct,
			(unsigned long long)visBits, (unsigned long long)floodBits, sec );
	}
	}

#undef CL_TRY
#undef CL_BUFFER

	if ( success )
	{
		for ( int i = 0; i < nportals; i++ )
		{
			portal_t *p = sorted_portals[i];
			if ( !p->portalvis )
				p->portalvis = (byte *)malloc( portalbytes );
			memcpy( p->portalvis, hAccum + (size_t)( p - portals ) * portalbytes, portalbytes );
			p->status = stat_done;
		}
		printf( "[VVIS GPU][RAYCAST][%s] PortalFlow complete : %d portails, %d pass.\n",
			useCuda ? "CUDA" : "OpenCL", nportals, numPasses );
	}

	free( hFloodOrig );
	free( hAccum );
	free( hPortals );
	free( hWindingPts );
	free( hLeafs );
	free( hLeafPortals );
	free( hFlood );
	free( hPortalvis );
	return success;
}
