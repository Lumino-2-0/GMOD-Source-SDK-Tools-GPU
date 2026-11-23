// flow_gpu_host.cpp
//
// Host side of the raycast visibility computation. It drives every compute engine of
// the machine (GPUs and CPU cores) on the same work.
//
// Where it fits in VVIS:
//   BasePortalVis (CPU)          -> portalflood, a coarse superset
//   GPU_RunPortalFlowRaycast     -> portalvis, replaces PortalFlow
//   ClusterMerge / CalcPAS (CPU) -> unchanged
//
// ENGINES. They all run the same kernel (flow_gpu_kernel.cu) and give bit-identical
// results for a given pair, so the PVS never depends on the hardware:
//   - main GPU    : CUDA (NVIDIA, compute capability >= 7.5, PTX compiled by the driver),
//                   otherwise the OpenCL GPU with the most compute units.
//                   VVIS_GPU_BACKEND=opencl (-opencl) forces OpenCL; VVIS_GPU_DEVICE=<n>
//                   forces the n-th device.
//   - helper GPUs : the other OpenCL GPUs, typically the integrated GPU of the processor.
//                   When CUDA drives the main GPU, NVIDIA OpenCL devices are skipped (same
//                   hardware; VVIS_GPU_HELPERS_ALL=1 keeps them, for testing).
//   - CPU threads : the kernel compiled as C++ (SSE2 or AVX2 build, flow_cpu_ray.cpp).
//   -gpuonly keeps only the main GPU, -nogpu only the CPU threads. Without a usable GPU
//   the CPU threads do all the work (same PVS, just slower).
// Both GPU APIs are loaded dynamically (LoadLibrary), with no SDK and no static linking:
// nvcuda.dll (NVIDIA driver) and OpenCL.dll (NVIDIA, AMD and Intel drivers). The types
// and constants used are redeclared here.
//
// SCHEDULING. The candidate pairs of a pass (portalflood bits) are cut into small work
// units: consecutive ranges of the pair index space (src * numPortals + dst) holding about
// the same amount of work. Each engine runs in its own thread and repeatedly claims a batch
// of consecutive units from a shared atomic counter. The batch size follows the throughput
// measured on that engine (a GPU batch lasts ~0.1 s, a CPU batch ~0.05 s) and shrinks near
// the end of the pass, so fast engines take most of the work and all engines finish at
// about the same time. GPUs keep two launches in flight on two streams (queues): the GPU
// never waits for the host, and the tail of one launch overlaps the next one. CPU threads
// write into the host result buffer (atomic OR); each GPU accumulates in its own buffer,
// merged with OR at the end of the pass. If an engine fails, the units it claimed are
// redone by the other engines.

#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <math.h>

#include "vis.h"
#include "threads.h"
#include "pacifier.h"
#include "flow_gpu.h"
#include "flow_cpu_ray.h"
#include "vvis_ui.h"
#include "flow_gpu_kernel.h"
#include "flow_gpu_kernel_ptx.h"

// --- OpenCL 1.2 API subset ---------------------------------------------------

typedef struct _cl_platform_id   *cl_platform_id;
typedef struct _cl_device_id     *cl_device_id;
typedef struct _cl_context       *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_mem           *cl_mem;
typedef struct _cl_program       *cl_program;
typedef struct _cl_kernel        *cl_kernel;
typedef struct _cl_event         *cl_event;
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

#define CL_SUCCESS                   0
#define CL_DEVICE_TYPE_GPU           (1 << 2)
#define CL_PLATFORM_NAME             0x0902
#define CL_PLATFORM_VENDOR           0x0903
#define CL_DEVICE_NAME               0x102B
#define CL_DEVICE_VENDOR             0x102C
#define CL_DEVICE_MAX_COMPUTE_UNITS  0x1002
#define CL_DEVICE_GLOBAL_MEM_SIZE    0x101F
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE 0x1010
#define CL_PROGRAM_BUILD_LOG         0x1183
#define CL_MEM_READ_WRITE            (1 << 0)
#define CL_MEM_READ_ONLY             (1 << 2)
#define CL_MEM_COPY_HOST_PTR         (1 << 5)
#define CL_TRUE                      1

#define CLAPI __stdcall
typedef cl_int  (CLAPI *PFN_clGetPlatformIDs)( cl_uint, cl_platform_id *, cl_uint * );
typedef cl_int  (CLAPI *PFN_clGetPlatformInfo)( cl_platform_id, cl_platform_info, size_t, void *, size_t * );
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
typedef cl_int  (CLAPI *PFN_clEnqueueNDRangeKernel)( cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event * );
typedef cl_int  (CLAPI *PFN_clEnqueueReadBuffer)( cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event * );
typedef cl_int  (CLAPI *PFN_clEnqueueWriteBuffer)( cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event * );
typedef cl_int  (CLAPI *PFN_clEnqueueFillBuffer)( cl_command_queue, cl_mem, const void *, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event * );
typedef cl_int  (CLAPI *PFN_clFlush)( cl_command_queue );
typedef cl_int  (CLAPI *PFN_clFinish)( cl_command_queue );
typedef cl_int  (CLAPI *PFN_clWaitForEvents)( cl_uint, const cl_event * );
typedef cl_int  (CLAPI *PFN_clReleaseEvent)( cl_event );
typedef cl_int  (CLAPI *PFN_clReleaseMemObject)( cl_mem );
typedef cl_int  (CLAPI *PFN_clReleaseKernel)( cl_kernel );
typedef cl_int  (CLAPI *PFN_clReleaseProgram)( cl_program );
typedef cl_int  (CLAPI *PFN_clReleaseCommandQueue)( cl_command_queue );
typedef cl_int  (CLAPI *PFN_clReleaseContext)( cl_context );

static struct
{
	HMODULE                       hLib;
	PFN_clGetPlatformIDs          GetPlatformIDs;
	PFN_clGetPlatformInfo         GetPlatformInfo;
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
	PFN_clEnqueueWriteBuffer      EnqueueWriteBuffer;
	PFN_clEnqueueFillBuffer       EnqueueFillBuffer;
	PFN_clFlush                   Flush;
	PFN_clFinish                  Finish;
	PFN_clWaitForEvents           WaitForEvents;
	PFN_clReleaseEvent            ReleaseEvent;
	PFN_clReleaseMemObject        ReleaseMemObject;
	PFN_clReleaseKernel           ReleaseKernel;
	PFN_clReleaseProgram          ReleaseProgram;
	PFN_clReleaseCommandQueue     ReleaseCommandQueue;
	PFN_clReleaseContext          ReleaseContext;
} cl;

static bool LoadOpenCL( void )
{
	if ( cl.hLib )
		return true;
	cl.hLib = LoadLibraryA( "OpenCL.dll" );
	if ( !cl.hLib )
		return false;

#define CL_LOAD( name ) \
	cl.name = (PFN_cl##name)GetProcAddress( cl.hLib, "cl" #name ); \
	if ( !cl.name ) { FreeLibrary( cl.hLib ); memset( &cl, 0, sizeof( cl ) ); return false; }

	CL_LOAD( GetPlatformIDs );
	CL_LOAD( GetPlatformInfo );
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
	CL_LOAD( EnqueueWriteBuffer );
	CL_LOAD( EnqueueFillBuffer );
	CL_LOAD( Flush );
	CL_LOAD( Finish );
	CL_LOAD( WaitForEvents );
	CL_LOAD( ReleaseEvent );
	CL_LOAD( ReleaseMemObject );
	CL_LOAD( ReleaseKernel );
	CL_LOAD( ReleaseProgram );
	CL_LOAD( ReleaseCommandQueue );
	CL_LOAD( ReleaseContext );
#undef CL_LOAD
	return true;
}

// Lists every GPU of every OpenCL platform. The Microsoft "OpenCLOn12" platform re-exposes,
// through Direct3D 12, GPUs that already have a native driver: it is only used when no
// native platform reports a GPU, so that no GPU is counted twice.
static int EnumerateGPUs( cl_device_id *pOut, int maxOut )
{
	cl_platform_id platforms[16];
	cl_uint numPlatforms = 0;
	if ( cl.GetPlatformIDs( 16, platforms, &numPlatforms ) != CL_SUCCESS )
		return 0;
	if ( numPlatforms > 16 )
		numPlatforms = 16;

	int count = 0;
	for ( int layered = 0; layered < 2 && count == 0; ++layered )
	{
		for ( cl_uint i = 0; i < numPlatforms && count < maxOut; ++i )
		{
			char name[128] = { 0 }, vendor[128] = { 0 };
			cl.GetPlatformInfo( platforms[i], CL_PLATFORM_NAME, sizeof( name ) - 1, name, NULL );
			cl.GetPlatformInfo( platforms[i], CL_PLATFORM_VENDOR, sizeof( vendor ) - 1, vendor, NULL );
			const bool isLayered = strstr( name, "OpenCLOn12" ) || strstr( vendor, "Microsoft" );
			if ( isLayered != ( layered == 1 ) )
				continue;
			cl_uint n = 0;
			if ( cl.GetDeviceIDs( platforms[i], CL_DEVICE_TYPE_GPU, (cl_uint)( maxOut - count ), pOut + count, &n ) != CL_SUCCESS )
				continue;
			count += ( n < (cl_uint)( maxOut - count ) ) ? (int)n : maxOut - count;
		}
	}
	return count;
}

// --- CUDA driver API subset (loaded dynamically) ------------------------------------

typedef int                CUresult;
typedef int                CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st    *CUcontext;
typedef struct CUmod_st    *CUmodule;
typedef struct CUfunc_st   *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st  *CUevent;

#define CU_DEV_ATTR_MULTIPROCESSOR_COUNT 16
#define CU_DEV_ATTR_CC_MAJOR             75
#define CU_DEV_ATTR_CC_MINOR             76
#define CU_CTX_SCHED_BLOCKING_SYNC       0x04 // waiting host threads sleep instead of spinning
#define CU_STREAM_NON_BLOCKING           0x01
#define CU_EVENT_BLOCKING_SYNC           0x01
#define CU_EVENT_DISABLE_TIMING          0x02

typedef CUresult (CLAPI *PFN_cuInit)( unsigned int );
typedef CUresult (CLAPI *PFN_cuDeviceGetCount)( int * );
typedef CUresult (CLAPI *PFN_cuDeviceGet)( CUdevice *, int );
typedef CUresult (CLAPI *PFN_cuDeviceGetName)( char *, int, CUdevice );
typedef CUresult (CLAPI *PFN_cuDeviceTotalMem)( size_t *, CUdevice );
typedef CUresult (CLAPI *PFN_cuDeviceGetAttribute)( int *, int, CUdevice );
typedef CUresult (CLAPI *PFN_cuCtxCreate)( CUcontext *, unsigned int, CUdevice );
typedef CUresult (CLAPI *PFN_cuCtxDestroy)( CUcontext );
typedef CUresult (CLAPI *PFN_cuCtxSetCurrent)( CUcontext );
typedef CUresult (CLAPI *PFN_cuCtxSynchronize)( void );
typedef CUresult (CLAPI *PFN_cuModuleLoadData)( CUmodule *, const void * );
typedef CUresult (CLAPI *PFN_cuModuleUnload)( CUmodule );
typedef CUresult (CLAPI *PFN_cuModuleGetFunction)( CUfunction *, CUmodule, const char * );
typedef CUresult (CLAPI *PFN_cuMemAlloc)( CUdeviceptr *, size_t );
typedef CUresult (CLAPI *PFN_cuMemFree)( CUdeviceptr );
typedef CUresult (CLAPI *PFN_cuMemcpyHtoD)( CUdeviceptr, const void *, size_t );
typedef CUresult (CLAPI *PFN_cuMemcpyDtoH)( void *, CUdeviceptr, size_t );
typedef CUresult (CLAPI *PFN_cuMemsetD8)( CUdeviceptr, unsigned char, size_t );
typedef CUresult (CLAPI *PFN_cuStreamCreate)( CUstream *, unsigned int );
typedef CUresult (CLAPI *PFN_cuStreamDestroy)( CUstream );
typedef CUresult (CLAPI *PFN_cuEventCreate)( CUevent *, unsigned int );
typedef CUresult (CLAPI *PFN_cuEventRecord)( CUevent, CUstream );
typedef CUresult (CLAPI *PFN_cuEventSynchronize)( CUevent );
typedef CUresult (CLAPI *PFN_cuEventDestroy)( CUevent );
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
	PFN_cuCtxSetCurrent      CtxSetCurrent;
	PFN_cuCtxSynchronize     CtxSynchronize;
	PFN_cuModuleLoadData     ModuleLoadData;
	PFN_cuModuleUnload       ModuleUnload;
	PFN_cuModuleGetFunction  ModuleGetFunction;
	PFN_cuMemAlloc           MemAlloc;
	PFN_cuMemFree            MemFree;
	PFN_cuMemcpyHtoD         MemcpyHtoD;
	PFN_cuMemcpyDtoH         MemcpyDtoH;
	PFN_cuMemsetD8           MemsetD8;
	PFN_cuStreamCreate       StreamCreate;
	PFN_cuStreamDestroy      StreamDestroy;
	PFN_cuEventCreate        EventCreate;
	PFN_cuEventRecord        EventRecord;
	PFN_cuEventSynchronize   EventSynchronize;
	PFN_cuEventDestroy       EventDestroy;
	PFN_cuLaunchKernel       LaunchKernel;
} cu;

static bool LoadCUDA( void )
{
	cu.hLib = LoadLibraryA( "nvcuda.dll" );
	if ( !cu.hLib )
		return false;

#define CU_LOAD( member, symbol ) \
	cu.member = (PFN_cu##member)GetProcAddress( cu.hLib, symbol ); \
	if ( !cu.member ) { FreeLibrary( cu.hLib ); memset( &cu, 0, sizeof( cu ) ); return false; }

	CU_LOAD( Init, "cuInit" );
	CU_LOAD( DeviceGetCount, "cuDeviceGetCount" );
	CU_LOAD( DeviceGet, "cuDeviceGet" );
	CU_LOAD( DeviceGetName, "cuDeviceGetName" );
	CU_LOAD( DeviceTotalMem, "cuDeviceTotalMem_v2" );
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
	CU_LOAD( MemsetD8, "cuMemsetD8_v2" );
	CU_LOAD( StreamCreate, "cuStreamCreate" );
	CU_LOAD( StreamDestroy, "cuStreamDestroy_v2" );
	CU_LOAD( EventCreate, "cuEventCreate" );
	CU_LOAD( EventRecord, "cuEventRecord" );
	CU_LOAD( EventSynchronize, "cuEventSynchronize" );
	CU_LOAD( EventDestroy, "cuEventDestroy_v2" );
	CU_LOAD( LaunchKernel, "cuLaunchKernel" );
#undef CU_LOAD
	return true;
}

// --- Engines -------------------------------------------------------------------------

enum RayEngineType { ENGINE_CUDA = 0, ENGINE_OPENCL, ENGINE_CPU };

// A batch of consecutive work units claimed by an engine: [first, first + count).
struct ClaimRange
{
	int first, count;
};

struct RayEngine
{
	RayEngineType type;
	bool          isMain;         // main GPU
	char          name[192];
	bool          dead;           // failed: excluded from every later round and pass
	volatile bool failed;         // failed during the current round
	bool          inRound;        // takes part in the current round
	bool          usedInPass;     // claimed work in the current pass (its buffer is merged)
	double        rate;           // measured throughput, in ray-units per second
	double        targetSec;      // wanted duration of one batch
	double        minBatchWork;   // GPU: smallest batch (ray-units) that fills every compute unit
	unsigned long long candDone;  // candidate pairs processed in the current pass (-debug)
	ClaimRange   *claims;         // batches claimed in the current round (redone on failure)
	int           numClaims, maxClaims;
	HANDLE        thread;

	// CUDA
	CUcontext   cuCtx;
	CUmodule    cuMod;
	CUfunction  cuFn;
	CUstream    cuStream[2];
	CUevent     cuEvent[2];
	CUdeviceptr dPortals, dLeafs, dLeafPortals, dWinding, dEdges, dFlood, dVis, dStats;

	// OpenCL
	cl_device_id       clDev;
	cl_context         clCtx;
	cl_command_queue   clQueue[2];
	cl_program         clProg;
	cl_kernel          clKernel;
	cl_mem             mPortals, mLeafs, mLeafPortals, mWinding, mEdges, mFlood, mVis;
	unsigned long long clMaxAlloc, clGlobalMem;
};

#define MAX_ENGINES 160
static RayEngine             *s_engines[MAX_ENGINES];
static int                    s_numEngines = 0;
static int                    s_numCpuEngines = 0;
static bool                   s_cpuAVX2 = false;
static FlowCPU_RaycastRangeFn s_cpuRange = NULL;
static double                 s_qpcPeriod = 0.0;

// Starting throughput guesses (ray-units per second), deliberately low: the first batches
// are short, then the measured rate takes over (it can grow 4x per batch).
static const double RATE_MAIN_GPU   = 5.0e7;
static const double RATE_HELPER_GPU = 5.0e6;
static const double RATE_CPU_THREAD = 1.0e6;
// A GPU launch lasts at least as long as its slowest warp, whatever its size: a batch that
// is too small leaves compute units idle, measures a low rate and shrinks the next batch.
// Every GPU batch therefore holds at least this much work per compute unit (ray-units).
static const double GPU_MIN_WORK_PER_CU = 1048576.0;
// Rays per grid row with RAY_FLAG_EXHAUSTIVE: must match RAY_CHUNK in the kernels.
#define RAY_CHUNK 256

// Diagnostic statistics (VVIS_GPU_STATS=1, CUDA only), see flow_gpu_kernel.cu.
#define RAY_STAT_COUNT ( 4 + 16 + 1 )

static double Now( void )
{
	LARGE_INTEGER c;
	QueryPerformanceCounter( &c );
	return (double)c.QuadPart * s_qpcPeriod;
}

static RayEngine *NewEngine( RayEngineType type )
{
	if ( s_numEngines >= MAX_ENGINES )
		return NULL;
	RayEngine *e = (RayEngine *)calloc( 1, sizeof( RayEngine ) );
	if ( !e )
		return NULL;
	e->type = type;
	s_engines[s_numEngines++] = e;
	return e;
}

// Frees the buffers of one run (scene, candidates, results).
static void ReleaseGpuRun( RayEngine *e )
{
	if ( e->type == ENGINE_CUDA && e->cuCtx )
	{
		cu.CtxSetCurrent( e->cuCtx );
		CUdeviceptr *bufs[] = { &e->dPortals, &e->dLeafs, &e->dLeafPortals, &e->dWinding, &e->dEdges, &e->dFlood, &e->dVis, &e->dStats };
		for ( int i = 0; i < (int)( sizeof( bufs ) / sizeof( bufs[0] ) ); i++ )
			if ( *bufs[i] ) { cu.MemFree( *bufs[i] ); *bufs[i] = 0; }
	}
	else if ( e->type == ENGINE_OPENCL )
	{
		cl_mem *bufs[] = { &e->mPortals, &e->mLeafs, &e->mLeafPortals, &e->mWinding, &e->mEdges, &e->mFlood, &e->mVis };
		for ( int i = 0; i < (int)( sizeof( bufs ) / sizeof( bufs[0] ) ); i++ )
			if ( *bufs[i] ) { cl.ReleaseMemObject( *bufs[i] ); *bufs[i] = NULL; }
	}
}

static void ReleaseEngine( RayEngine *e )
{
	ReleaseGpuRun( e );
	if ( e->type == ENGINE_CUDA && e->cuCtx )
	{
		cu.CtxSetCurrent( e->cuCtx );
		for ( int i = 0; i < 2; i++ )
		{
			if ( e->cuEvent[i] )  cu.EventDestroy( e->cuEvent[i] );
			if ( e->cuStream[i] ) cu.StreamDestroy( e->cuStream[i] );
		}
		if ( e->cuMod ) cu.ModuleUnload( e->cuMod );
		cu.CtxDestroy( e->cuCtx );
	}
	else if ( e->type == ENGINE_OPENCL )
	{
		if ( e->clKernel ) cl.ReleaseKernel( e->clKernel );
		if ( e->clProg )   cl.ReleaseProgram( e->clProg );
		for ( int i = 0; i < 2; i++ )
			if ( e->clQueue[i] ) cl.ReleaseCommandQueue( e->clQueue[i] );
		if ( e->clCtx )    cl.ReleaseContext( e->clCtx );
	}
	free( e->claims );
	free( e );
}

// Removes the last engine created (initialisation failure).
static void DropLastEngine( void )
{
	if ( s_numEngines > 0 )
		ReleaseEngine( s_engines[--s_numEngines] );
}

// Main GPU through CUDA: NVIDIA GPU with compute capability >= 7.5, context, PTX module,
// two streams. On failure (no driver, GPU too old, PTX rejected by an outdated driver...)
// returns NULL with nothing left allocated: OpenCL takes over.
static RayEngine *InitCUDA( void )
{
	if ( !LoadCUDA() )
		return NULL;
	int count = 0;
	if ( cu.Init( 0 ) != 0 || cu.DeviceGetCount( &count ) != 0 || count <= 0 )
		return NULL;

	int chosen = -1, bestSM = -1;
	const char *penv = getenv( "VVIS_GPU_DEVICE" );
	const int forced = ( penv && *penv ) ? atoi( penv ) : -1;
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
		return NULL;

	CUdevice dev;
	char name[128] = { 0 };
	size_t mem = 0;
	cu.DeviceGet( &dev, chosen );
	cu.DeviceGetName( name, (int)sizeof( name ) - 1, dev );
	cu.DeviceTotalMem( &mem, dev );

	RayEngine *e = NewEngine( ENGINE_CUDA );
	if ( !e )
		return NULL;
	if ( cu.CtxCreate( &e->cuCtx, CU_CTX_SCHED_BLOCKING_SYNC, dev ) != 0 )
	{
		e->cuCtx = NULL;
		DropLastEngine();
		return NULL;
	}
	CUresult r = cu.ModuleLoadData( &e->cuMod, g_RaycastPTX );
	if ( r != 0 || cu.ModuleGetFunction( &e->cuFn, e->cuMod, "RaycastStabKernel" ) != 0 )
	{
		fprintf( stderr, "[VVIS GPU] CUDA: PTX kernel load rejected (%d), driver too old? Falling back to OpenCL.\n", r );
		if ( r != 0 )
			e->cuMod = NULL;
		DropLastEngine();
		return NULL;
	}
	for ( int i = 0; i < 2; i++ )
	{
		if ( cu.StreamCreate( &e->cuStream[i], CU_STREAM_NON_BLOCKING ) != 0 ||
			cu.EventCreate( &e->cuEvent[i], CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING ) != 0 )
		{
			DropLastEngine();
			return NULL;
		}
	}
	int smCount = 0;
	cu.DeviceGetAttribute( &smCount, CU_DEV_ATTR_MULTIPROCESSOR_COUNT, dev );
	e->isMain = true;
	e->rate = RATE_MAIN_GPU;
	e->targetSec = 0.1;
	e->minBatchWork = GPU_MIN_WORK_PER_CU * ( smCount > 0 ? smCount : 16 );
	sprintf( e->name, "CUDA %s (%llu MB)", name, (unsigned long long)( mem / ( 1024 * 1024 ) ) );
	return e;
}

// One OpenCL GPU: context, two in-order queues, kernel compiled from source.
// Correctly rounded division/sqrt, and FP_CONTRACT OFF (no implicit FMA) in the kernel
// source, so the results match the other engines exactly.
static RayEngine *InitOpenCLEngine( cl_device_id dev, bool isMain )
{
	char name[128] = { 0 }, vendor[96] = { 0 };
	cl_ulong mem = 0, maxAlloc = 0;
	cl.GetDeviceInfo( dev, CL_DEVICE_NAME, sizeof( name ) - 1, name, NULL );
	cl.GetDeviceInfo( dev, CL_DEVICE_VENDOR, sizeof( vendor ) - 1, vendor, NULL );
	cl.GetDeviceInfo( dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof( mem ), &mem, NULL );
	cl.GetDeviceInfo( dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( maxAlloc ), &maxAlloc, NULL );

	RayEngine *e = NewEngine( ENGINE_OPENCL );
	if ( !e )
		return NULL;
	e->clDev = dev;
	e->clGlobalMem = mem;
	e->clMaxAlloc = maxAlloc;
	sprintf( e->name, "OpenCL %s (%s, %llu MB)", name, vendor, (unsigned long long)( mem / ( 1024 * 1024 ) ) );

	cl_int err = CL_SUCCESS;
	const char *what = "clCreateContext";
	e->clCtx = cl.CreateContext( NULL, 1, &dev, NULL, NULL, &err );
	for ( int i = 0; i < 2 && err == CL_SUCCESS; i++ )
	{
		what = "clCreateCommandQueue";
		e->clQueue[i] = cl.CreateCommandQueue( e->clCtx, dev, 0, &err );
	}
	if ( err == CL_SUCCESS )
	{
		what = "kernel compilation";
		const char *src = g_szRaycastKernelSource;
		e->clProg = cl.CreateProgramWithSource( e->clCtx, 1, &src, NULL, &err );
		if ( err == CL_SUCCESS )
			err = cl.BuildProgram( e->clProg, 1, &dev, "-cl-fp32-correctly-rounded-divide-sqrt", NULL, NULL );
		if ( err != CL_SUCCESS && ( isMain || g_bVisDebug ) )
		{
			char log[8192] = { 0 };
			if ( e->clProg )
				cl.GetProgramBuildInfo( e->clProg, dev, CL_PROGRAM_BUILD_LOG, sizeof( log ) - 1, log, NULL );
			fprintf( stderr, "[VVIS GPU] %s: OpenCL build log:\n%s\n", e->name, log );
		}
	}
	if ( err == CL_SUCCESS )
	{
		what = "clCreateKernel";
		e->clKernel = cl.CreateKernel( e->clProg, "RaycastStabKernel", &err );
	}
	if ( err != CL_SUCCESS )
	{
		if ( isMain || g_bVisDebug )
			fprintf( stderr, "[VVIS GPU] %s: %s failed (%d), device skipped.\n", e->name, what, err );
		DropLastEngine();
		return NULL;
	}
	cl_uint cuCount = 0;
	cl.GetDeviceInfo( dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof( cuCount ), &cuCount, NULL );
	e->isMain = isMain;
	e->rate = isMain ? RATE_MAIN_GPU : RATE_HELPER_GPU;
	e->minBatchWork = GPU_MIN_WORK_PER_CU * ( cuCount > 0 ? cuCount : 4 );
	// A helper is often the integrated GPU that also drives the display: shorter launches.
	e->targetSec = isMain ? 0.1 : 0.05;
	return e;
}

static void AddCpuEngines( int count )
{
	if ( !s_cpuRange )
		s_cpuRange = FlowCPU_GetRaycastRange( &s_cpuAVX2 );
	for ( int i = 0; i < count; i++ )
	{
		RayEngine *e = NewEngine( ENGINE_CPU );
		if ( !e )
			break;
		e->rate = RATE_CPU_THREAD;
		e->targetSec = 0.05;
		sprintf( e->name, "CPU thread %d", s_numCpuEngines );
		s_numCpuEngines++;
	}
}

// The VVIS thread pool is capped at MAX_TOOL_THREADS (numthreads is clamped there), but the
// ray threads are our own: at that cap, use every logical processor. A smaller value comes
// from -threads or from a small CPU and is kept.
static int CpuThreadCount( void )
{
	if ( numthreads > 0 && numthreads < MAX_TOOL_THREADS )
		return numthreads;
	const DWORD n = GetActiveProcessorCount( ALL_PROCESSOR_GROUPS );
	if ( n < 1 )
		return numthreads > 0 ? numthreads : 1;
	return n > 128 ? 128 : (int)n;
}

// Every engine failed (GPU only, e.g. -gpuonly): the CPU threads take over.
static void EnsureLiveEngine( void )
{
	for ( int i = 0; i < s_numEngines; i++ )
		if ( !s_engines[i]->dead )
			return;
	Warning( "[VVIS GPU] No GPU left: the CPU threads take over (same PVS, slower).\n" );
	AddCpuEngines( CpuThreadCount() );
}

static bool AnyGpuAlive( void )
{
	for ( int i = 0; i < s_numEngines; i++ )
		if ( s_engines[i]->type != ENGINE_CPU && !s_engines[i]->dead )
			return true;
	return false;
}

// --- GPU_Init / GPU_Shutdown ------------------------------------------------------------

bool GPU_Init( void )
{
	LARGE_INTEGER freq;
	QueryPerformanceFrequency( &freq );
	s_qpcPeriod = 1.0 / (double)freq.QuadPart;

	const char *be = getenv( "VVIS_GPU_BACKEND" );
	const bool forceOpenCL = be && !_stricmp( be, "opencl" );
	const char *ha = getenv( "VVIS_GPU_HELPERS_ALL" );
	const bool helpersAll = ha && *ha == '1';

	if ( !g_bRayNoGpu )
	{
		RayEngine *mainGpu = forceOpenCL ? NULL : InitCUDA();
		const bool mainIsCuda = mainGpu != NULL;
		const bool wantHelpers = !g_bRayGpuOnly;
		if ( ( !mainGpu || wantHelpers ) && LoadOpenCL() )
		{
			cl_device_id devices[32];
			const int numDevices = EnumerateGPUs( devices, 32 );
			int mainIdx = -1;
			if ( !mainGpu && numDevices > 0 )
			{
				const char *penv = getenv( "VVIS_GPU_DEVICE" );
				if ( penv && *penv && atoi( penv ) >= 0 && atoi( penv ) < numDevices )
					mainIdx = atoi( penv );
				cl_uint bestCU = 0;
				if ( mainIdx < 0 )
				{
					for ( int i = 0; i < numDevices; ++i )
					{
						cl_uint units = 0;
						cl.GetDeviceInfo( devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof( units ), &units, NULL );
						if ( mainIdx < 0 || units > bestCU )
						{
							mainIdx = i;
							bestCU = units;
						}
					}
				}
				mainGpu = InitOpenCLEngine( devices[mainIdx], true );
			}
			for ( int i = 0; wantHelpers && i < numDevices; ++i )
			{
				if ( i == mainIdx )
					continue;
				char vendor[96] = { 0 };
				cl.GetDeviceInfo( devices[i], CL_DEVICE_VENDOR, sizeof( vendor ) - 1, vendor, NULL );
				if ( mainIsCuda && !helpersAll && ( strstr( vendor, "NVIDIA" ) || strstr( vendor, "nvidia" ) ) )
					continue;
				InitOpenCLEngine( devices[i], false );
			}
		}
	}

	const int numGpu = s_numEngines;
	if ( !g_bRayGpuOnly || numGpu == 0 )
	{
		// With a GPU, one core stays free for the drivers and the GPU feeding threads.
		int n = CpuThreadCount();
		if ( numGpu > 0 && n > 1 )
			n--;
		AddCpuEngines( n );
	}
	if ( numGpu == 0 && !g_bRayNoGpu )
		Warning( "[VVIS GPU] No usable GPU found: raycast on the CPU threads (same PVS, slower).\n" );
	return s_numEngines > 0;
}

void GPU_Shutdown( void )
{
	for ( int i = 0; i < s_numEngines; i++ )
		ReleaseEngine( s_engines[i] );
	s_numEngines = 0;
	s_numCpuEngines = 0;
	if ( cu.hLib )
	{
		FreeLibrary( cu.hLib );
		memset( &cu, 0, sizeof( cu ) );
	}
	if ( cl.hLib )
	{
		FreeLibrary( cl.hLib );
		memset( &cl, 0, sizeof( cl ) );
	}
}

// --- Scene and work units ------------------------------------------------------------

// Host data of the current run, shared by every engine.
static struct
{
	const GPUPortal *portals;      size_t portalsBytes;
	const GPULeaf   *leafs;        size_t leafsBytes;
	const int       *leafPortals;  size_t leafPortalsBytes;
	const float     *winding;      size_t windingBytes;
	const float     *edges;        size_t edgesBytes;
	unsigned char   *flood;        // candidates of the current pass
	unsigned char   *vis;          // results of the current pass
	size_t           visBytes;     // multiple of 4 (32-bit atomics)
	int              numPortals, numLeafs, maxSteps;
	bool             wantStats;
} s_scene;

// Consecutive range [begin, end) of the pair index space holding `cand` candidate pairs.
struct WorkUnit
{
	unsigned long long begin, end;
	unsigned int       cand;
};

// State of the current round (a pass has one round, plus one per recovery).
static struct
{
	const WorkUnit *units;
	int             numUnits;
	volatile LONG   next;          // first unit not claimed yet
	volatile LONG   running;       // engine threads still running
	volatile LONG64 candDone;      // candidate pairs processed in the pass (progress)
	double          unitWork;      // average work of one unit, in ray-units
	double          costPerCand;   // ray-units per candidate pair (points per portal ^ 2)
	int             samples, flags, ringmask;
	unsigned int    chunksY;       // GPU grid rows: ray chunks per pair (1 without EXHAUSTIVE)
	RayKernelArgs   cpuArgs;
} s_round;

// Units of about UNIT_WORK ray-units: large enough to make claiming cheap, small enough for
// a CPU thread to process one in a few milliseconds. Built from the candidates only, so
// the split depends on the map, never on the hardware.
static WorkUnit *BuildUnits( const unsigned char *flood, int numPortals, int pb, double costPerCand, int *pNumUnits )
{
	const double UNIT_WORK = 32768.0;
	double uc = UNIT_WORK / ( costPerCand > 1.0 ? costPerCand : 1.0 );
	const unsigned int unitCand = uc < 1.0 ? 1u : uc > 4096.0 ? 4096u : (unsigned int)uc;

	int cap = 1024, num = 0;
	WorkUnit *units = (WorkUnit *)malloc( sizeof( WorkUnit ) * cap );
	const unsigned long long n = (unsigned long long)numPortals;
	const int nw = ( numPortals + 63 ) >> 6;
	unsigned long long begin = 0;
	unsigned int acc = 0;
	for ( int r = 0; r < numPortals && units; r++ )
	{
		const unsigned char *row = flood + (size_t)r * pb;
		for ( int w = 0; w < nw; w++ )
		{
			unsigned long long bits;
			memcpy( &bits, row + (size_t)w * 8, 8 );
			if ( !bits )
				continue;
			acc += (unsigned int)__popcnt64( bits );
			if ( acc < unitCand )
				continue;
			if ( num == cap )
			{
				cap *= 2;
				WorkUnit *grown = (WorkUnit *)realloc( units, sizeof( WorkUnit ) * cap );
				if ( !grown ) { free( units ); units = NULL; break; }
				units = grown;
			}
			const int dEnd = ( w + 1 ) * 64 < numPortals ? ( w + 1 ) * 64 : numPortals;
			units[num].begin = begin;
			units[num].end = (unsigned long long)r * n + (unsigned long long)dEnd;
			units[num].cand = acc;
			begin = units[num].end;
			acc = 0;
			num++;
		}
	}
	if ( units && begin < n * n )
	{
		if ( acc || num == 0 )
		{
			if ( num == cap )
			{
				WorkUnit *grown = (WorkUnit *)realloc( units, sizeof( WorkUnit ) * ( cap + 1 ) );
				if ( !grown ) { free( units ); units = NULL; }
				else units = grown;
			}
			if ( units )
			{
				units[num].begin = begin;
				units[num].end = n * n;
				units[num].cand = acc;
				num++;
			}
		}
		else
			units[num - 1].end = n * n;
	}
	*pNumUnits = units ? num : 0;
	return units;
}

// Number of units an engine claims at once: about targetSec of work at its measured rate,
// but at most half of its share of the units left (guided scheduling), so that every
// engine finishes at about the same time.
static int BatchSize( const RayEngine *e )
{
	const LONG left = s_round.numUnits - s_round.next;
	if ( left <= 0 )
		return 0;
	double totalRate = 0.0;
	for ( int i = 0; i < s_numEngines; i++ )
		if ( s_engines[i]->inRound && !s_engines[i]->failed )
			totalRate += s_engines[i]->rate;
	double want = e->targetSec * e->rate / s_round.unitWork;
	const double cap = 0.5 * (double)left * e->rate / ( totalRate > 1.0 ? totalRate : 1.0 );
	if ( want > cap )
		want = cap;
	const double minUnits = e->minBatchWork / s_round.unitWork;
	if ( want < minUnits )
		want = minUnits;
	if ( want > (double)left )
		want = (double)left;
	return want < 1.0 ? 1 : (int)want;
}

// Claims the next batch: pair range [*pBegin, *pEnd) and its candidate count. 0 = no work left.
static int ClaimBatch( RayEngine *e, unsigned long long *pBegin, unsigned long long *pEnd, unsigned long long *pCand )
{
	const int want = BatchSize( e );
	if ( want <= 0 )
		return 0;
	// Room for the record first: a claimed batch must always be recorded so it can be redone.
	if ( e->numClaims == e->maxClaims )
	{
		const int newMax = e->maxClaims ? e->maxClaims * 2 : 256;
		ClaimRange *grown = (ClaimRange *)realloc( e->claims, sizeof( ClaimRange ) * newMax );
		if ( !grown )
			return 0;
		e->claims = grown;
		e->maxClaims = newMax;
	}
	const LONG first = InterlockedExchangeAdd( &s_round.next, (LONG)want );
	if ( first >= s_round.numUnits )
		return 0;
	const int count = ( s_round.numUnits - (int)first < want ) ? s_round.numUnits - (int)first : want;
	unsigned long long cand = 0;
	for ( int u = 0; u < count; u++ )
		cand += s_round.units[first + u].cand;
	*pBegin = s_round.units[first].begin;
	*pEnd = s_round.units[first + count - 1].end;
	*pCand = cand;

	e->claims[e->numClaims].first = (int)first;
	e->claims[e->numClaims].count = count;
	e->numClaims++;
	e->usedInPass = true;
	return count;
}

// A batch is done: updates the engine's throughput (smoothed, at most x4 per batch) and progress.
static void RecordBatch( RayEngine *e, unsigned long long cand, double sec )
{
	if ( cand )
	{
		double r = (double)cand * s_round.costPerCand / ( sec > 1e-4 ? sec : 1e-4 );
		if ( r > 4.0 * e->rate )
			r = 4.0 * e->rate;
		e->rate = 0.5 * ( e->rate + r );
	}
	e->candDone += cand;
	InterlockedExchangeAdd64( &s_round.candDone, (LONG64)cand );
}

// --- Engine threads ----------------------------------------------------------------------

static DWORD WINAPI CpuEngineThread( LPVOID p )
{
	RayEngine *e = (RayEngine *)p;
	unsigned long long b, en, cand;
	while ( ClaimBatch( e, &b, &en, &cand ) )
	{
		const double t0 = Now();
		s_cpuRange( s_round.cpuArgs, b, en );
		RecordBatch( e, cand, Now() - t0 );
	}
	InterlockedDecrement( &s_round.running );
	return 0;
}

static bool CudaLaunch( RayEngine *e, int slot, unsigned long long begin, unsigned long long end )
{
	const unsigned int block = 128;
	const unsigned long long maxThreads = (unsigned long long)block << 22;
	CUdeviceptr dPortals = e->dPortals, dLeafs = e->dLeafs, dLeafPortals = e->dLeafPortals, dWinding = e->dWinding;
	CUdeviceptr dEdges = e->dEdges, dFlood = e->dFlood, dVis = e->dVis, dStats = e->dStats;
	int numPortals = s_scene.numPortals, numLeafs = s_scene.numLeafs, pb = portalbytes, maxSteps = s_scene.maxSteps;
	int samples = s_round.samples, flags = s_round.flags, rings = s_round.ringmask;
	for ( unsigned long long s = begin; s < end; s += maxThreads )
	{
		unsigned long long start = s, stop = ( end - s > maxThreads ) ? s + maxThreads : end;
		const unsigned int grid = (unsigned int)( ( stop - start + block - 1 ) / block );
		void *args[] = { &dPortals, &numPortals, &dLeafs, &numLeafs, &dLeafPortals, &dWinding, &dEdges,
			&dFlood, &dVis, &pb, &samples, &maxSteps, &flags, &rings, &start, &stop, &dStats };
		if ( cu.LaunchKernel( e->cuFn, grid, s_round.chunksY, 1, block, 1, 1, 0, e->cuStream[slot], args, NULL ) != 0 )
			return false;
	}
	return cu.EventRecord( e->cuEvent[slot], e->cuStream[slot] ) == 0;
}

// Arguments that do not change during a round.
static bool OpenCLSetArgs( RayEngine *e )
{
	int numPortals = s_scene.numPortals, numLeafs = s_scene.numLeafs, pb = portalbytes, maxSteps = s_scene.maxSteps;
	int samples = s_round.samples, flags = s_round.flags, rings = s_round.ringmask;
	cl_kernel k = e->clKernel;
	cl_int err = CL_SUCCESS;
	err |= cl.SetKernelArg( k, 0, sizeof( cl_mem ), &e->mPortals );
	err |= cl.SetKernelArg( k, 1, sizeof( int ), &numPortals );
	err |= cl.SetKernelArg( k, 2, sizeof( cl_mem ), &e->mLeafs );
	err |= cl.SetKernelArg( k, 3, sizeof( int ), &numLeafs );
	err |= cl.SetKernelArg( k, 4, sizeof( cl_mem ), &e->mLeafPortals );
	err |= cl.SetKernelArg( k, 5, sizeof( cl_mem ), &e->mWinding );
	err |= cl.SetKernelArg( k, 6, sizeof( cl_mem ), &e->mEdges );
	err |= cl.SetKernelArg( k, 7, sizeof( cl_mem ), &e->mFlood );
	err |= cl.SetKernelArg( k, 8, sizeof( cl_mem ), &e->mVis );
	err |= cl.SetKernelArg( k, 9, sizeof( int ), &pb );
	err |= cl.SetKernelArg( k, 10, sizeof( int ), &samples );
	err |= cl.SetKernelArg( k, 11, sizeof( int ), &maxSteps );
	err |= cl.SetKernelArg( k, 12, sizeof( int ), &flags );
	err |= cl.SetKernelArg( k, 13, sizeof( int ), &rings );
	return err == CL_SUCCESS;
}

static bool OpenCLLaunch( RayEngine *e, int slot, unsigned long long begin, unsigned long long end, cl_event *pEvent )
{
	const size_t local = 32; // = RAY_GROUP, required by the kernel (reqd_work_group_size)
	const unsigned long long maxThreads = 1ull << 28;
	*pEvent = NULL;
	for ( unsigned long long s = begin; s < end; s += maxThreads )
	{
		cl_ulong start = s, stop = ( end - s > maxThreads ) ? s + maxThreads : end;
		size_t global[2] = { (size_t)( ( stop - start + local - 1 ) / local * local ), s_round.chunksY };
		const size_t localSize[2] = { local, 1 };
		if ( cl.SetKernelArg( e->clKernel, 14, sizeof( cl_ulong ), &start ) != CL_SUCCESS ||
			cl.SetKernelArg( e->clKernel, 15, sizeof( cl_ulong ), &stop ) != CL_SUCCESS )
			return false;
		const bool last = stop == end;
		if ( cl.EnqueueNDRangeKernel( e->clQueue[slot], e->clKernel, 2, NULL, global, localSize, 0, NULL, last ? pEvent : NULL ) != CL_SUCCESS )
			return false;
	}
	return cl.Flush( e->clQueue[slot] ) == CL_SUCCESS;
}

// GPU feeding thread: keeps two launches in flight (one per stream/queue). It sleeps while
// the GPU works (blocking synchronisation), so it costs almost no CPU time.
static DWORD WINAPI GpuEngineThread( LPVOID p )
{
	RayEngine *e = (RayEngine *)p;
	bool ok = ( e->type == ENGINE_CUDA ) ? cu.CtxSetCurrent( e->cuCtx ) == 0 : OpenCLSetArgs( e );
	struct InFlight
	{
		bool               used;
		unsigned long long cand;
		double             tSubmit;
		cl_event           ev;
	} fl[2];
	memset( fl, 0, sizeof( fl ) );
	int head = 0;
	double tLast = Now();
	for ( ;; )
	{
		unsigned long long b = 0, en = 0, cand = 0;
		const int got = ok ? ClaimBatch( e, &b, &en, &cand ) : 0;
		if ( got )
		{
			const bool launched = ( e->type == ENGINE_CUDA ) ? CudaLaunch( e, head, b, en ) : OpenCLLaunch( e, head, b, en, &fl[head].ev );
			if ( launched )
			{
				fl[head].used = true;
				fl[head].cand = cand;
				fl[head].tSubmit = Now();
			}
			else
				ok = false;
			head ^= 1;
		}
		// Oldest launch in flight; with a single one, queue a second launch before waiting.
		int old = head;
		if ( !fl[old].used )
			old ^= 1;
		if ( !fl[old].used )
			break;
		if ( got && ok && !fl[head].used )
			continue;

		bool done;
		if ( e->type == ENGINE_CUDA )
			done = cu.EventSynchronize( e->cuEvent[old] ) == 0;
		else
		{
			done = !fl[old].ev || cl.WaitForEvents( 1, &fl[old].ev ) == CL_SUCCESS;
			if ( fl[old].ev )
				cl.ReleaseEvent( fl[old].ev );
			fl[old].ev = NULL;
		}
		const double t = Now();
		if ( !done )
			ok = false;
		else if ( ok )
			RecordBatch( e, fl[old].cand, t - ( fl[old].tSubmit > tLast ? fl[old].tSubmit : tLast ) );
		tLast = t;
		fl[old].used = false;
	}
	if ( !ok )
		e->failed = true;
	InterlockedDecrement( &s_round.running );
	return 0;
}

// --- GPU buffers --------------------------------------------------------------------------

// Uploads the scene and allocates the candidate/result buffers, once per run.
static bool PrepareGpuRun( RayEngine *e )
{
	const size_t vb = s_scene.visBytes;
	if ( e->type == ENGINE_CUDA )
	{
		if ( cu.CtxSetCurrent( e->cuCtx ) != 0 )
			return false;
#define CU_ALLOC_UPLOAD( var, host, bytes ) \
		if ( cu.MemAlloc( &var, bytes ) != 0 ) { var = 0; return false; } \
		if ( cu.MemcpyHtoD( var, host, bytes ) != 0 ) return false;
		CU_ALLOC_UPLOAD( e->dPortals, s_scene.portals, s_scene.portalsBytes );
		CU_ALLOC_UPLOAD( e->dLeafs, s_scene.leafs, s_scene.leafsBytes );
		CU_ALLOC_UPLOAD( e->dLeafPortals, s_scene.leafPortals, s_scene.leafPortalsBytes );
		CU_ALLOC_UPLOAD( e->dWinding, s_scene.winding, s_scene.windingBytes );
		CU_ALLOC_UPLOAD( e->dEdges, s_scene.edges, s_scene.edgesBytes );
#undef CU_ALLOC_UPLOAD
		if ( cu.MemAlloc( &e->dFlood, vb ) != 0 ) { e->dFlood = 0; return false; }
		if ( cu.MemAlloc( &e->dVis, vb ) != 0 ) { e->dVis = 0; return false; }
		if ( s_scene.wantStats && cu.MemAlloc( &e->dStats, sizeof( unsigned long long ) * RAY_STAT_COUNT ) != 0 ) { e->dStats = 0; return false; }
		return true;
	}

	const unsigned long long sceneBytes = s_scene.portalsBytes + s_scene.leafsBytes + s_scene.leafPortalsBytes + s_scene.windingBytes + s_scene.edgesBytes;
	if ( vb > e->clMaxAlloc || 2ull * vb + sceneBytes > e->clGlobalMem / 10 * 9 )
	{
		if ( g_bVisDebug || e->isMain )
			Warning( "[VVIS GPU] %s: not enough memory for this map (%llu MB needed), device skipped.\n",
				e->name, (unsigned long long)( ( 2ull * vb + sceneBytes ) >> 20 ) );
		return false;
	}
	cl_int err = CL_SUCCESS;
#define CL_BUF( var, flags, bytes, host ) \
	var = cl.CreateBuffer( e->clCtx, flags, bytes, (void *)( host ), &err ); \
	if ( err != CL_SUCCESS ) { var = NULL; return false; }
	CL_BUF( e->mPortals, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, s_scene.portalsBytes, s_scene.portals );
	CL_BUF( e->mLeafs, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, s_scene.leafsBytes, s_scene.leafs );
	CL_BUF( e->mLeafPortals, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, s_scene.leafPortalsBytes, s_scene.leafPortals );
	CL_BUF( e->mWinding, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, s_scene.windingBytes, s_scene.winding );
	CL_BUF( e->mEdges, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, s_scene.edgesBytes, s_scene.edges );
	CL_BUF( e->mFlood, CL_MEM_READ_ONLY, vb, NULL );
	CL_BUF( e->mVis, CL_MEM_READ_WRITE, vb, NULL );
#undef CL_BUF
	return true;
}

// Per pass: uploads the candidates and clears the results.
static bool PrepareGpuPass( RayEngine *e )
{
	const size_t vb = s_scene.visBytes;
	if ( e->type == ENGINE_CUDA )
	{
		if ( cu.CtxSetCurrent( e->cuCtx ) != 0 || cu.MemcpyHtoD( e->dFlood, s_scene.flood, vb ) != 0 ||
			cu.MemsetD8( e->dVis, 0, vb ) != 0 )
			return false;
		if ( e->dStats && cu.MemsetD8( e->dStats, 0, sizeof( unsigned long long ) * RAY_STAT_COUNT ) != 0 )
			return false;
		// The kernels run on non-blocking streams, which do not wait for the default stream.
		return cu.CtxSynchronize() == 0;
	}
	const cl_uint zero = 0;
	return cl.EnqueueWriteBuffer( e->clQueue[0], e->mFlood, CL_TRUE, 0, vb, s_scene.flood, 0, NULL, NULL ) == CL_SUCCESS &&
		cl.EnqueueFillBuffer( e->clQueue[0], e->mVis, &zero, sizeof( zero ), 0, vb, 0, NULL, NULL ) == CL_SUCCESS &&
		cl.Finish( e->clQueue[0] ) == CL_SUCCESS;
}

static void OrInto( unsigned char *dst, const unsigned char *src, size_t n )
{
	size_t i = 0;
	for ( ; i + 8 <= n; i += 8 )
	{
		unsigned long long a, b;
		memcpy( &a, dst + i, 8 );
		memcpy( &b, src + i, 8 );
		a |= b;
		memcpy( dst + i, &a, 8 );
	}
	for ( ; i < n; i++ )
		dst[i] |= src[i];
}

// Reads the GPU's results back in slices and ORs them into the host result buffer.
static bool MergeGpuResult( RayEngine *e, unsigned char *staging, size_t stagingBytes )
{
	const size_t vb = s_scene.visBytes;
	if ( e->type == ENGINE_CUDA && ( cu.CtxSetCurrent( e->cuCtx ) != 0 || cu.CtxSynchronize() != 0 ) )
		return false;
	for ( size_t off = 0; off < vb; off += stagingBytes )
	{
		const size_t n = ( vb - off < stagingBytes ) ? vb - off : stagingBytes;
		const bool ok = ( e->type == ENGINE_CUDA )
			? cu.MemcpyDtoH( staging, e->dVis + off, n ) == 0
			: cl.EnqueueReadBuffer( e->clQueue[0], e->mVis, CL_TRUE, off, n, staging, 0, NULL, NULL ) == CL_SUCCESS;
		if ( !ok )
			return false;
		OrInto( s_scene.vis + off, staging, n );
	}
	return true;
}

// --- Rounds ----------------------------------------------------------------------------

// Maps the current pass's progress onto the global bar (a single bar outside -debug).
static float s_progBase = 0.0f, s_progScale = 1.0f;
static void ReportProgress( float passFraction )
{
	UpdatePacifier( s_progBase + s_progScale * passFraction );
}

// Runs every live engine on `units` until they are all claimed and processed.
static void RunRound( const WorkUnit *units, int numUnits, unsigned long long passCand )
{
	double cand = 0.0;
	for ( int u = 0; u < numUnits; u++ )
		cand += units[u].cand;
	s_round.units = units;
	s_round.numUnits = numUnits;
	s_round.next = 0;
	s_round.unitWork = numUnits ? cand / numUnits * s_round.costPerCand : 1.0;
	if ( s_round.unitWork < 1.0 )
		s_round.unitWork = 1.0;

	int count = 0;
	for ( int i = 0; i < s_numEngines; i++ )
	{
		RayEngine *e = s_engines[i];
		e->inRound = !e->dead;
		e->failed = false;
		e->numClaims = 0;
		count += e->inRound ? 1 : 0;
	}
	s_round.running = count;
	for ( int i = 0; i < s_numEngines; i++ )
	{
		RayEngine *e = s_engines[i];
		if ( !e->inRound )
			continue;
		const bool cpu = e->type == ENGINE_CPU;
		e->thread = CreateThread( NULL, 0, cpu ? CpuEngineThread : GpuEngineThread, e, 0, NULL );
		if ( !e->thread )
		{
			e->inRound = false;
			InterlockedDecrement( &s_round.running );
			continue;
		}
		// The GPU feeding threads must react quickly; the CPU threads leave room for the rest.
		SetThreadPriority( e->thread, cpu ? THREAD_PRIORITY_BELOW_NORMAL : THREAD_PRIORITY_ABOVE_NORMAL );
	}
	while ( s_round.running > 0 )
	{
		Sleep( 30 );
		const double f = passCand ? (double)s_round.candDone / (double)passCand : 1.0;
		ReportProgress( (float)( f < 1.0 ? f : 1.0 ) );
	}
	for ( int i = 0; i < s_numEngines; i++ )
	{
		RayEngine *e = s_engines[i];
		if ( e->thread )
		{
			WaitForSingleObject( e->thread, INFINITE );
			CloseHandle( e->thread );
			e->thread = NULL;
		}
		e->inRound = false;
	}
}

// Runs one pass over the candidates in s_scene.flood; the results end up in s_scene.vis.
// The GPU results are merged at the end of each round. Units claimed by an engine that
// failed (and units nobody claimed) are redone in a new round by the remaining engines.
static bool RunPass( const WorkUnit *units, int numUnits, unsigned long long passCand, unsigned char *staging, size_t stagingBytes )
{
	memset( s_scene.vis, 0, s_scene.visBytes );
	for ( int i = 0; i < s_numEngines; i++ )
	{
		RayEngine *e = s_engines[i];
		e->candDone = 0;
		e->usedInPass = false;
		if ( e->type != ENGINE_CPU && !e->dead && !PrepareGpuPass( e ) )
		{
			Warning( "[VVIS GPU] %s: upload failed, device skipped.\n", e->name );
			e->dead = true;
		}
	}
	s_round.candDone = 0;

	const WorkUnit *cur = units;
	int curNum = numUnits;
	WorkUnit *redo = NULL;
	bool ok = true;
	for ( int round = 0; curNum > 0; round++ )
	{
		if ( round >= 4 )
		{
			ok = false;
			break;
		}
		EnsureLiveEngine();
		RunRound( cur, curNum, passCand );

		for ( int i = 0; i < s_numEngines; i++ )
		{
			RayEngine *e = s_engines[i];
			if ( e->type != ENGINE_CPU && !e->dead && !e->failed && e->numClaims > 0 && !MergeGpuResult( e, staging, stagingBytes ) )
				e->failed = true;
		}

		int redoNum = s_round.next < curNum ? curNum - (int)s_round.next : 0;
		for ( int i = 0; i < s_numEngines; i++ )
			if ( s_engines[i]->failed )
				for ( int c = 0; c < s_engines[i]->numClaims; c++ )
					redoNum += s_engines[i]->claims[c].count;
		WorkUnit *next = NULL;
		int n = 0;
		if ( redoNum > 0 )
		{
			next = (WorkUnit *)malloc( sizeof( WorkUnit ) * redoNum );
			if ( !next )
			{
				ok = false;
				break;
			}
			for ( int i = 0; i < s_numEngines; i++ )
			{
				RayEngine *e = s_engines[i];
				if ( !e->failed )
					continue;
				Warning( "[VVIS GPU] %s failed: its share of the work is redone by the other engines.\n", e->name );
				e->dead = true;
				for ( int c = 0; c < e->numClaims; c++ )
				{
					memcpy( next + n, cur + e->claims[c].first, sizeof( WorkUnit ) * e->claims[c].count );
					n += e->claims[c].count;
				}
			}
			for ( int u = (int)s_round.next; u < curNum; u++ )
				next[n++] = cur[u];
		}
		free( redo );
		redo = next;
		cur = next;
		curNum = n;
	}
	free( redo );
	return ok;
}

// --- GPU_RunPortalFlowRaycast ---------------------------------------------------
// Fills portals[i].portalvis for every portal, in place of PortalFlow.
// Returns false, without modifying anything, if the work could not be completed:
// VVIS then runs the original CPU flow.
//
// Steps:
//   1. Flatten the portals and their winding vertices (GPUPortal), precompute the edges.
//   2. Flatten the leaf -> outgoing portals topology (GPULeaf).
//   3. Copy portalflood: it is the candidate pair filter.
//   4. For each pass: cut the candidates into work units, run every engine on them
//      (on a GPU, 1 warp = 1 pair, split into ray chunks in the dense pass; on the
//      CPU, 1 thread walks its pairs one by one), then merge the results with a bitwise OR.
//   5. Copy the result into portal_t and mark it stat_done.
//
// Passes: passes[k] describes pass k (see GPUPassConfig in flow_gpu.h). Pass 0 tests
// every candidate pair. Later passes only retest the candidates not yet proven visible:
//
//   candidates   = portalflood
//   pass 0       : test candidates              -> vis0
//   pass 1       : test candidates and not vis0 -> vis1
//   result       = vis0 OR vis1 ...
//
// A pass never removes visibility, it can only add more. The work of every pass is
// derived from the map only (see RAY_FLAG_AUTO), so the PVS is the same on any machine.
//
// Diagnostics: VVIS_GPU_STATS=1 (CUDA only) prints the number of rays cast
// and the depth at which missed rays die.

// Counts the set bits of a buffer.
static size_t PopCountBytes( const unsigned char *p, size_t n )
{
	size_t c = 0, i = 0;
	for ( ; i + 8 <= n; i += 8 )
	{
		unsigned long long v;
		memcpy( &v, p + i, 8 );
		c += (size_t)__popcnt64( v );
	}
	for ( ; i < n; i++ )
		c += (size_t)__popcnt( p[i] );
	return c;
}

// Portals come in pairs (2k, 2k+1) = the two directions of the same portal. Since the
// line a -> b is also the line b -> a, a visible pair (i, j) implies that (j^1, i^1) is
// visible too, so it is copied over. Used with RAY_FLAG_MIRROR, where the kernel only
// tests one of the two directions.
static void SymmetrizeVis( unsigned char *vis, int numPortals, int pb )
{
	for ( int i = 0; i < numPortals; i++ )
	{
		const unsigned long long *row = (const unsigned long long *)( vis + (size_t)i * pb );
		const int nw = pb / 8;
		for ( int w = 0; w < nw; w++ )
		{
			unsigned long long v = row[w];
			while ( v )
			{
				unsigned long idx;
				_BitScanForward64( &idx, v );
				v &= v - 1;
				const int j = w * 64 + (int)idx;
				if ( j >= numPortals )
					continue;
				vis[(size_t)( j ^ 1 ) * pb + ( ( i ^ 1 ) >> 3 )] |= (unsigned char)( 1u << ( ( i ^ 1 ) & 7 ) );
			}
		}
	}
}

// Portal edge margin: negative = a ray must pass clearly inside every portal. A positive
// margin (0.01, like ON_VIS_EPSILON) let through rays grazing the walls and made hidden
// areas visible.
static const float g_EdgeEps = -0.003f;

// Points per portal of a pass (nominal: the cap, rings use min(cap, vertices) per ring).
static double PassPoints( int samples, int flags, int ringmask )
{
	if ( flags & RAY_FLAG_HALTON )
		return (double)samples;
	return 1.0 + (double)__popcnt( (unsigned)ringmask ) * (double)samples;
}

bool GPU_RunPortalFlowRaycast( int sorted_portal_count, const GPUPassConfig *passCfgIn, int numPasses )
{
	const int nportals = sorted_portal_count;
	const int numPortals = g_numportals * 2;
	if ( numPasses < 1 || numPasses > 16 || !passCfgIn || s_numEngines == 0 )
		return false;
	GPUPassConfig passes[16];
	for ( int k = 0; k < numPasses; k++ )
		passes[k] = passCfgIn[k];

	// Outside -debug: a single bar for all passes. An auto dense pass weighs ~1/4 of a pass.
	float passWeight[16], totalWeight = 0.0f;
	for ( int k = 0; k < numPasses; k++ )
	{
		passWeight[k] = ( passes[k].flags & RAY_FLAG_AUTO ) ? 0.25f : 1.0f;
		totalWeight += passWeight[k];
	}

	// -- 1. Portals + vertices + edges -------------------------------------------
	int totalWindingPts = 0;
	for ( int i = 0; i < numPortals; i++ )
		if ( portals[i].winding )
			totalWindingPts += portals[i].winding->numpoints;
	const int windingAlloc = totalWindingPts > 0 ? totalWindingPts : 1;

	GPUPortal *hPortals = (GPUPortal *)calloc( numPortals > 0 ? numPortals : 1, sizeof( GPUPortal ) );
	float *hWindingPts = (float *)calloc( (size_t)windingAlloc * 3, sizeof( float ) );
	float *hEdges = (float *)calloc( (size_t)windingAlloc * 4, sizeof( float ) );

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
	FlowCPU_BuildEdgeTable( hPortals, numPortals, hWindingPts, g_EdgeEps, hEdges );

	// -- 2. Leaf -> portal topology ---------------------------------------------
	int totalLeafPortals = 0;
	for ( int l = 0; l < portalclusters; l++ )
		totalLeafPortals += leafs[l].portals.Count();

	GPULeaf *hLeafs = (GPULeaf *)calloc( portalclusters > 0 ? portalclusters : 1, sizeof( GPULeaf ) );
	int *hLeafPortals = (int *)calloc( totalLeafPortals > 0 ? totalLeafPortals : 1, sizeof( int ) );
	int lp = 0;
	for ( int l = 0; l < portalclusters; l++ )
	{
		hLeafs[l].portals_start = lp;
		hLeafs[l].portals_count = leafs[l].portals.Count();
		for ( int k = 0; k < leafs[l].portals.Count(); k++ )
			hLeafPortals[lp++] = (int)( leafs[l].portals[k] - portals );
	}

	// -- 3. portalflood -------------------------------------------------------------
	// Size rounded up to a multiple of 4: the kernel writes portalvis as 32-bit words.
	const size_t visBytes = (size_t)numPortals * portalbytes;
	const size_t visAlloc = visBytes > 0 ? ( visBytes + 3 ) & ~(size_t)3 : 4;
	unsigned char *hFlood = (unsigned char *)calloc( visAlloc, 1 );
	for ( int i = 0; i < numPortals; i++ )
		if ( portals[i].portalflood )
			memcpy( hFlood + (size_t)i * portalbytes, portals[i].portalflood, portalbytes );
	unsigned char *hPortalvis = (unsigned char *)calloc( visAlloc, 1 );
	unsigned char *hFloodOrig = (unsigned char *)malloc( visAlloc );
	unsigned char *hAccum = (unsigned char *)calloc( visAlloc, 1 );
	const size_t stagingBytes = visAlloc < ( 64u << 20 ) ? visAlloc : ( 64u << 20 );
	unsigned char *staging = (unsigned char *)malloc( stagingBytes );
	if ( !hPortals || !hWindingPts || !hEdges || !hLeafs || !hLeafPortals || !hFlood || !hPortalvis || !hFloodOrig || !hAccum || !staging )
	{
		free( hPortals ); free( hWindingPts ); free( hEdges ); free( hLeafs ); free( hLeafPortals );
		free( hFlood ); free( hPortalvis ); free( hFloodOrig ); free( hAccum ); free( staging );
		return false;
	}
	memcpy( hFloodOrig, hFlood, visAlloc );
	const size_t floodBits = PopCountBytes( hFloodOrig, visAlloc );

	s_scene.portals = hPortals;          s_scene.portalsBytes = sizeof( GPUPortal ) * ( numPortals > 0 ? numPortals : 1 );
	s_scene.leafs = hLeafs;              s_scene.leafsBytes = sizeof( GPULeaf ) * ( portalclusters > 0 ? portalclusters : 1 );
	s_scene.leafPortals = hLeafPortals;  s_scene.leafPortalsBytes = sizeof( int ) * ( totalLeafPortals > 0 ? totalLeafPortals : 1 );
	s_scene.winding = hWindingPts;       s_scene.windingBytes = sizeof( float ) * 3 * windingAlloc;
	s_scene.edges = hEdges;              s_scene.edgesBytes = sizeof( float ) * 4 * windingAlloc;
	s_scene.flood = hFlood;
	s_scene.vis = hPortalvis;
	s_scene.visBytes = visAlloc;
	s_scene.numPortals = numPortals;
	s_scene.numLeafs = portalclusters;
	s_scene.maxSteps = portalclusters + 1;
	const char *stenv = getenv( "VVIS_GPU_STATS" );
	s_scene.wantStats = stenv && *stenv == '1';

	// -- GPU buffers for the whole run ------------------------------------------------
	for ( int i = 0; i < s_numEngines; i++ )
	{
		RayEngine *e = s_engines[i];
		if ( e->type != ENGINE_CPU && !PrepareGpuRun( e ) )
		{
			if ( g_bVisDebug || e->isMain )
				Warning( "[VVIS GPU] %s: buffer allocation failed, device skipped.\n", e->name );
			ReleaseGpuRun( e );
			e->dead = true;
		}
	}
	EnsureLiveEngine();
	const bool gpuUsed = AnyGpuAlive();
	if ( g_bVisDebug )
	{
		Msg( "[VVIS GPU] Engines:" );
		for ( int i = 0; i < s_numEngines; i++ )
			if ( s_engines[i]->type != ENGINE_CPU && !s_engines[i]->dead )
				Msg( " %s +", s_engines[i]->name );
		if ( s_numCpuEngines )
			Msg( " CPU %d threads (%s)\n", s_numCpuEngines, s_cpuAVX2 ? "AVX2" : "SSE2" );
		else
			Msg( " no CPU threads\n" );
	}

	const ULONGLONG tFlow0 = GetTickCount64();
	if ( !g_bVisDebug )
	{
		VisUI_Label( gpuUsed ? "PortalFlow_RayGPU:" : "PortalFlow_RayCPU:" );
		StartPacifier( "" );
	}

	s_round.cpuArgs.portals = hPortals;
	s_round.cpuArgs.numPortals = numPortals;
	s_round.cpuArgs.leafs = hLeafs;
	s_round.cpuArgs.numLeafs = portalclusters;
	s_round.cpuArgs.leafPortals = hLeafPortals;
	s_round.cpuArgs.windingPts = hWindingPts;
	s_round.cpuArgs.windingEdges = hEdges;
	s_round.cpuArgs.portalflood = hFlood;
	s_round.cpuArgs.portalvisWords = (unsigned int *)hPortalvis;
	s_round.cpuArgs.portalbytes = portalbytes;
	s_round.cpuArgs.maxSteps = portalclusters + 1;

	bool success = true;
	double prevWork = 0.0; // work of the previous pass, in ray-units (candidates x points^2)
	int passesRun = 0;
	for ( int pass = 0; pass < numPasses && success; pass++ )
	{
		const GPUPassConfig &cfg = passes[pass];
		int samples = cfg.samples;
		const int passFlags = cfg.flags, passRings = cfg.ringmask;
		const ULONGLONG t0 = GetTickCount64();
		{
			float before = 0.0f;
			for ( int k = 0; k < pass; k++ )
				before += passWeight[k];
			s_progBase = g_bVisDebug ? 0.0f : before / totalWeight;
			s_progScale = g_bVisDebug ? 1.0f : passWeight[pass] / totalWeight;
		}
		if ( pass > 0 )
		{
			for ( size_t b = 0; b < visAlloc; b++ )
				hFlood[b] = hFloodOrig[b] & ~hAccum[b];
		}
		const size_t candBits = PopCountBytes( hFlood, visAlloc );
		if ( passFlags & RAY_FLAG_AUTO )
		{
			// Points per portal: the largest count whose work fits in the budget. The budget
			// only depends on the map (candidate counts), never on the measured speed, so
			// every machine computes exactly the same PVS.
			int n = 0;
			if ( candBits > 0 )
			{
				const double budget = max( (double)cfg.budgetMin * 1.0e6, (double)cfg.budgetRel * prevWork );
				n = (int)sqrt( budget / (double)candBits );
				if ( n > cfg.samples )
					n = cfg.samples;
			}
			if ( n < 6 )
			{
				if ( g_bVisDebug )
					Msg( "[VVIS GPU] Pass %d/%d : dense pass skipped (%llu candidate pairs, work budget too small).\n", pass + 1, numPasses, (unsigned long long)candBits );
				continue;
			}
			samples = n;
		}
		const double pts = PassPoints( samples, passFlags, passRings );
		prevWork = (double)candBits * pts * pts;
		if ( g_bVisDebug )
		{
			if ( passFlags & RAY_FLAG_HALTON )
				Msg( "[VVIS GPU] Pass %d/%d : dense sampling, %d points/portal, %llu candidate pairs%s...\n", pass + 1, numPasses, samples, (unsigned long long)candBits, ( passFlags & RAY_FLAG_EXHAUSTIVE ) ? ", all rays" : "" );
			else
				Msg( "[VVIS GPU] Pass %d/%d : rings, samples=%d, %llu candidate pairs...\n", pass + 1, numPasses, samples, (unsigned long long)candBits );
			char prefix[64];
			sprintf( prefix, "[VVIS GPU] Pass %d/%d: ", pass + 1, numPasses );
			StartPacifier( prefix );
		}

		s_round.samples = samples;
		s_round.flags = passFlags;
		s_round.ringmask = passRings;
		s_round.costPerCand = pts * pts;
		s_round.chunksY = 1;
		if ( passFlags & RAY_FLAG_EXHAUSTIVE )
		{
			// pts * pts bounds the rays of a pair (rings: at most `samples` vertices each).
			const double chunks = ceil( pts * pts / (double)RAY_CHUNK );
			s_round.chunksY = chunks < 1.0 ? 1u : chunks > 65535.0 ? 65535u : (unsigned int)chunks;
		}
		s_round.cpuArgs.samples = samples;
		s_round.cpuArgs.flags = passFlags;
		s_round.cpuArgs.ringmask = passRings;

		if ( candBits > 0 )
		{
			int numUnits = 0;
			WorkUnit *units = BuildUnits( hFlood, numPortals, portalbytes, s_round.costPerCand, &numUnits );
			success = units && RunPass( units, numUnits, candBits, staging, stagingBytes );
			free( units );
		}
		else
			memset( hPortalvis, 0, visAlloc );
		passesRun++;
		if ( g_bVisDebug )
			EndPacifier( true );
		if ( !success )
			break;

		if ( s_scene.wantStats )
		{
			for ( int i = 0; i < s_numEngines; i++ )
			{
				RayEngine *e = s_engines[i];
				if ( e->type != ENGINE_CUDA || e->dead || !e->dStats )
					continue;
				unsigned long long st[RAY_STAT_COUNT] = { 0 };
				if ( cu.CtxSetCurrent( e->cuCtx ) != 0 || cu.MemcpyDtoH( st, e->dStats, sizeof( st ) ) != 0 )
					continue;
				printf( "[VVIS GPU][STATS] (pairs processed by CUDA) pairs tested %llu, visible %llu, rays %llu (%llu of them for invisible pairs), walk steps %llu\n",
					st[0], st[1], st[2], st[3], st[20] );
				printf( "[VVIS GPU][STATS] depth at which missed rays die (portals crossed: rays):\n" );
				for ( int h = 0; h < 16; ++h )
					if ( st[4 + h] )
						printf( "[VVIS GPU][STATS]   %s%2d : %llu\n", h == 15 ? ">=" : "  ", h, st[4 + h] );
			}
		}

		if ( passFlags & RAY_FLAG_MIRROR )
			SymmetrizeVis( hPortalvis, numPortals, portalbytes );
		// As in the CPU flow, portalvis is always a subset of portalflood. The mirror copy and
		// the portals crossed along the way can fall outside that bound (portalflood is not
		// symmetric): those bits could never be visible on the CPU side, so they are dropped.
		for ( size_t b = 0; b < visAlloc; b++ )
			hAccum[b] |= hPortalvis[b] & hFloodOrig[b];

		if ( g_bVisDebug )
		{
			const size_t visBits = PopCountBytes( hAccum, visAlloc );
			const double sec = ( GetTickCount64() - t0 ) / 1000.0;
			const int pct = floodBits ? (int)( 100.0 * visBits / floodBits ) : 100;
			char bar[21];
			for ( int k = 0; k < 20; k++ )
				bar[k] = ( k < pct / 5 ) ? '#' : '.';
			bar[20] = 0;
			Msg( "[VVIS GPU] Pass %d/%d [%s] %3d%% visible / initial candidates (%llu/%llu) - %.2f s\n",
				pass + 1, numPasses, bar, pct, (unsigned long long)visBits, (unsigned long long)floodBits, sec );
			unsigned long long cpuDone = 0, total = 0;
			for ( int i = 0; i < s_numEngines; i++ )
			{
				total += s_engines[i]->candDone;
				if ( s_engines[i]->type == ENGINE_CPU )
					cpuDone += s_engines[i]->candDone;
			}
			if ( total )
			{
				Msg( "[VVIS GPU]   work share:" );
				for ( int i = 0; i < s_numEngines; i++ )
					if ( s_engines[i]->type != ENGINE_CPU && s_engines[i]->candDone )
						Msg( " %s %.1f%% |", s_engines[i]->name, 100.0 * s_engines[i]->candDone / total );
				Msg( " CPU %.1f%%\n", 100.0 * cpuDone / total );
			}
		}
	}
	s_progBase = 0.0f;
	s_progScale = 1.0f;
	if ( !g_bVisDebug )
	{
		if ( success )
			EndPacifier( false );
		Msg( " (%d)\n", (int)( ( GetTickCount64() - tFlow0 ) / 1000 ) );
	}

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
		if ( g_bVisDebug )
			Msg( "[VVIS GPU][RAYCAST] PortalFlow complete : %d portals, %d of %d passes run.\n", nportals, passesRun, numPasses );
	}

	for ( int i = 0; i < s_numEngines; i++ )
		ReleaseGpuRun( s_engines[i] );
	memset( &s_scene, 0, sizeof( s_scene ) );
	free( staging );
	free( hFloodOrig );
	free( hAccum );
	free( hPortals );
	free( hWindingPts );
	free( hEdges );
	free( hLeafs );
	free( hLeafPortals );
	free( hFlood );
	free( hPortalvis );
	return success;
}
