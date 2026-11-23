// flow_cpu_ray_impl.h
//
// Body of the CPU raycast engine. Included inside a namespace by flow_cpu_ray.cpp
// (SSE2 build) and flow_cpu_ray_avx2.cpp (AVX2 build); not a standalone header.
// The includer must enable /fp:precise semantics and disable FP contraction (no FMA)
// so that results match the GPU kernels bit for bit.

#include "flow_gpu_kernel.cu"

void RaycastRange( const RayKernelArgs &a, unsigned long long begin, unsigned long long end )
{
	const unsigned long long n = (unsigned long long)a.numPortals;
	unsigned long long idx = begin;
	while ( idx < end )
	{
		const int src = (int)( idx / n );
		const unsigned long long rowBase = (unsigned long long)src * n;
		const unsigned long long rowEnd = ( rowBase + n < end ) ? rowBase + n : end;
		const int d0 = (int)( idx - rowBase ), d1 = (int)( rowEnd - rowBase );
		const unsigned char *row = a.portalflood + (size_t)src * a.portalbytes;

		// Only the candidate bits of the row are visited (64 at a time).
		for ( int w = d0 >> 6; w <= ( d1 - 1 ) >> 6; ++w )
		{
			unsigned long long bits;
			memcpy( &bits, row + (size_t)w * 8, 8 );
			const int lo = w * 64;
			if ( d0 > lo )
				bits &= ~0ull << ( d0 - lo );
			if ( d1 - lo < 64 )
				bits &= ( 1ull << ( d1 - lo ) ) - 1;
			while ( bits )
			{
				unsigned long b;
				_BitScanForward64( &b, bits );
				bits &= bits - 1;
				const int dst = lo + (int)b;
				if ( !RayPairIsCandidate( a.portalflood, a.numPortals, a.portalbytes, a.flags, src, dst ) )
					continue;
				unsigned long long rays, walked;
				RaycastPair( a.portals, a.leafs, a.numLeafs, a.leafPortals, a.windingPts, a.windingEdges,
					a.portalvisWords, a.portalbytes, a.samples, a.maxSteps, a.flags, a.ringmask, src, dst,
					&rays, &walked, (unsigned long long *)0 );
			}
		}
		idx = rowEnd;
	}
}