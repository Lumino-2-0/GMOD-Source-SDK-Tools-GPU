// flow_cpu_ray.cpp
//
// CPU raycast engine, SSE2 build (runs on every x64 CPU), AVX2 dispatch and the
// winding edge table. See flow_cpu_ray.h.
//
// The project is compiled with /fp:fast; this file and flow_cpu_ray_avx2.cpp are
// compiled with /fp:precise (see the vcxproj) and the pragmas below forbid FMA
// contraction, so the float math matches the GPU kernels (-fmad=false) exactly.

#pragma float_control( precise, on )
#pragma fp_contract( off )

#include <intrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "flow_cpu_ray.h"

namespace FlowCpuSSE2
{
#include "flow_cpu_ray_impl.h"
}

namespace FlowCpuAVX2
{
void RaycastRange( const RayKernelArgs &a, unsigned long long begin, unsigned long long end );
}

// AVX2 usable: CPU flag (CPUID leaf 7, EBX bit 5) and YMM state saved by the OS (XCR0).
static bool CpuHasAVX2( void )
{
	int r[4];
	__cpuid( r, 0 );
	if ( r[0] < 7 )
		return false;
	__cpuid( r, 1 );
	const bool osxsave = ( r[2] & ( 1 << 27 ) ) != 0;
	const bool avx = ( r[2] & ( 1 << 28 ) ) != 0;
	if ( !osxsave || !avx )
		return false;
	if ( ( _xgetbv( 0 ) & 6 ) != 6 )
		return false;
	__cpuidex( r, 7, 0 );
	return ( r[1] & ( 1 << 5 ) ) != 0;
}

FlowCPU_RaycastRangeFn FlowCPU_GetRaycastRange( bool *pbAVX2 )
{
	const char *isa = getenv( "VVIS_CPU_ISA" );
	const bool forceSSE2 = isa && !_stricmp( isa, "sse2" );
	const bool avx2 = !forceSSE2 && CpuHasAVX2();
	if ( pbAVX2 )
		*pbAVX2 = avx2;
	return avx2 ? &FlowCpuAVX2::RaycastRange : &FlowCpuSSE2::RaycastRange;
}

void FlowCPU_BuildEdgeTable( const GPUPortal *portals, int numPortals, const float *windingPts,
	float edgeEps, float *outEdges )
{
	for ( int i = 0; i < numPortals; i++ )
	{
		const GPUPortal &gp = portals[i];
		const int n = gp.winding_count;
		const float *w = windingPts + (size_t)gp.winding_start * 3;
		float *e = outEdges + (size_t)gp.winding_start * 4;
		for ( int k = 0; k < n; ++k )
		{
			const int k1 = ( k + 1 == n ) ? 0 : k + 1;
			const float e0 = w[k1*3+0] - w[k*3+0], e1 = w[k1*3+1] - w[k*3+1], e2 = w[k1*3+2] - w[k*3+2];
			e[k*4+0] = e0;
			e[k*4+1] = e1;
			e[k*4+2] = e2;
			e[k*4+3] = edgeEps * sqrtf( e0*e0 + e1*e1 + e2*e2 );
		}
	}
}