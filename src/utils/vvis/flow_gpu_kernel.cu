// flow_gpu_kernel.cu
//
// CUDA kernel for the portal-to-portal visibility computation ("stabbing lines").
// Same algorithm as the OpenCL version (flow_gpu_kernel.h): any change made here
// must be mirrored there, and vice versa.
//
// This file is not compiled by MSBuild. build_ptx.ps1 turns it into PTX
// (flow_gpu_kernel_ptx.h), which is embedded in the DLL. The NVIDIA driver then
// compiles the PTX for the installed GPU, so users do not need nvcc.
// Only rerun build_ptx.ps1 when this file changes.
//
// Parity with OpenCL: -fmad=false (no FMA) and IEEE division.
//
// PRINCIPLE
//
// For each "src" portal, VVIS provides a list of candidates (portalflood).
// For each candidate pair (src, dst), rays are cast between a point on the src
// portal and a point on the dst portal. A ray is valid if it crosses, portal
// by portal, a chain of leafs all the way to dst:
//
//   src          P1           P2           dst
//    |  leaf A   |   leaf B   |   leaf C   |
//  a o---------->|----------->|----------->o b
//    |           |            |            |
//    every portal crossed is marked visible from src (portalvis bit)
//
// If the ray no longer hits any portal ahead of it, it is lost. A single ray
// reaching dst is enough: the pair is then done.
//
// IMPORTANT: this computation only ADDS visibility. More rays = more pairs
// proven visible. It can never remove any.
//
// PARALLELISM: 1 thread = 1 (src, dst) pair. The kernel is launched in chunks
// so as not to trigger the Windows watchdog (TDR).

// Unused: the active rings are selected by the `ringmask` kernel argument.
#define RAY_NUM_RINGS 2
// Ray almost parallel to a portal plane: ignored.
#define RAY_EPS_DENOM 1e-6f
// Margin so the portal that was just crossed is not hit again.
#define RAY_EPS_T     1e-5f
// The ray goes from a (t = 0) to b (t = 1); small margin for rounding errors.
#define RAY_T_MAX     1.001f
// Kernel flags (`flags` argument).
//   RAY_FLAG_EXTEND     : the ray does not stop at dst, it keeps going as long as it
//                         crosses portals (also marking what lies behind).
//   RAY_FLAG_EXHAUSTIVE : cast every ray of the pair even after a hit
//                         (the extended rays then benefit other pairs).
//   RAY_FLAG_HALTON     : dense sampling: `samples_per_portal` points spread over the
//                         whole portal surface (Halton sequence) instead of the rings.
//   RAY_FLAG_MIRROR     : visibility is symmetric (the line a -> b is also the line
//                         b -> a). If the mirror pair (dst^1 -> src^1) is also a candidate,
//                         only one of the two threads tests it; the host copies the result.
#define RAY_FLAG_EXTEND     1
#define RAY_FLAG_EXHAUSTIVE 2
#define RAY_FLAG_HALTON     4
#define RAY_FLAG_MIRROR     8
#define RAY_T_INF           1.0e30f

typedef struct
{
	float  plane_normal[3];
	float  plane_dist;
	int    leaf;
	float  origin[3];
	float  radius;
	int    winding_start;
	int    winding_count;
	int    portalflood_offset;
	int    status;
	int    nummightsee;
} GPUPortal;

typedef struct
{
	int portals_start;
	int portals_count;
} GPULeaf;

// Sets the (src -> dst) bit in portalvis. Words are 32-bit and several threads
// may target the same word: atomicOr avoids losing a bit.
__device__ __forceinline__ void SetPortalvisBit( unsigned int *words, int src, int dst, int portalbytes )
{
	unsigned long long byte_index = (unsigned long long)src * (unsigned long long)portalbytes + (unsigned long long)( dst >> 3 );
	unsigned int shift = (unsigned int)( byte_index & 3 ) * 8u + (unsigned int)( dst & 7 );
	unsigned int mask = 1u << shift;
	volatile unsigned int *w = words + ( byte_index >> 2 );
	// Read first: most bits are already set, which avoids the atomic.
	if ( !( *w & mask ) )
		atomicOr( (unsigned int *)w, mask );
}

__device__ __forceinline__ int TestBit( const unsigned char *row, int bit )
{
	return ( row[bit >> 3] >> ( bit & 7 ) ) & 1;
}

// A point is inside the winding if it lies on the inner side of every edge, with a
// signed margin `eps` (map units): eps > 0 tolerates a point slightly outside, eps < 0
// requires it to be clearly inside. vvis_gpu uses eps < 0: a ray that grazes an edge
// runs along a wall, and accepting it made hidden areas visible (PVS too large).
// s = scalar triple product = signed distance to the edge * edge length.
__device__ int PointInWinding( const float *winding_pts, const GPUPortal *gp, const float x[3], float eps )
{
	int allpos = 1, allneg = 1;
	int n = gp->winding_count;
	const float *w = winding_pts + (size_t)gp->winding_start * 3;
	for ( int k = 0; k < n; ++k )
	{
		int k1 = ( k + 1 == n ) ? 0 : k + 1;
		float e0 = w[k1*3+0] - w[k*3+0], e1 = w[k1*3+1] - w[k*3+1], e2 = w[k1*3+2] - w[k*3+2];
		float d0 = x[0] - w[k*3+0], d1 = x[1] - w[k*3+1], d2 = x[2] - w[k*3+2];
		float c0 = e1*d2 - e2*d1, c1 = e2*d0 - e0*d2, c2 = e0*d1 - e1*d0;
		float s = c0*gp->plane_normal[0] + c1*gp->plane_normal[1] + c2*gp->plane_normal[2];
		float tol = eps * sqrtf( e0*e0 + e1*e1 + e2*e2 );
		if ( s < -tol ) allpos = 0;
		if ( s > tol ) allneg = 0;
	}
	return allpos || allneg;
}

__device__ void PortalCentroid( const float *winding_pts, const GPUPortal *gp, float out[3] )
{
	const float *w = winding_pts + (size_t)gp->winding_start * 3;
	float cx = 0.0f, cy = 0.0f, cz = 0.0f;
	for ( int k = 0; k < gp->winding_count; ++k )
	{
		cx += w[k*3+0]; cy += w[k*3+1]; cz += w[k*3+2];
	}
	float inv = 1.0f / (float)gp->winding_count;
	out[0] = cx * inv; out[1] = cy * inv; out[2] = cz * inv;
}

// Sample point number s of a portal (nv = vertices used per ring).
//   s = 0  : the centre
//   s >= 1 : rings enabled in `ringmask`, nv points each, in order:
//            bit 0 = 90 % of the way from the centre to the vertex, bit 1 = 50 %,
//            bit 2 = 99 % (corners). Not 100 %: a ray starting at a vertex runs along
//            the walls and would be rejected (or, worse, wrongly accepted) at the
//            edges of the following portals.
// Total per portal: 1 + popcount(ringmask) * nv points.
__device__ void PortalInsetSample( const float *winding_pts, const GPUPortal *gp,
	const float c[3], int nv, int ringmask, int s, float out[3] )
{
	if ( s <= 0 || nv <= 0 )
	{
		out[0] = c[0]; out[1] = c[1]; out[2] = c[2];
		return;
	}
	int idx = s - 1;
	int ring = idx / nv;
	int vi = idx - ring * nv;
	int m = ringmask;
	for ( int r = 0; r < ring; ++r )
		m &= m - 1;
	int bit = m & -m;
	float t = ( bit == 1 ) ? 0.9f : ( bit == 2 ) ? 0.5f : 0.99f;
	const float *w = winding_pts + (size_t)( gp->winding_start + vi ) * 3;
	out[0] = c[0] + t * ( w[0] - c[0] );
	out[1] = c[1] + t * ( w[1] - c[1] );
	out[2] = c[2] + t * ( w[2] - c[2] );
}

// Radical inverse (Halton sequence) of index i in the given base, in [0,1).
__device__ float RadicalInverse( int i, int base )
{
	float f = 1.0f, r = 0.0f;
	while ( i > 0 )
	{
		f /= (float)base;
		r += f * (float)( i % base );
		i /= base;
	}
	return r;
}

// Point number s (s >= 1) spread over the whole portal surface: a triangle of the
// fan (vertex 0, k, k+1) is picked, then a barycentric point inside it.
// The points are nested: the set for s <= N contains the set for s <= M < N.
__device__ void PortalHaltonSample( const float *winding_pts, const GPUPortal *gp, int s, float out[3] )
{
	int n = gp->winding_count;
	const float *w = winding_pts + (size_t)gp->winding_start * 3;
	int tri = (int)( RadicalInverse( s, 5 ) * (float)( n - 2 ) );
	if ( tri > n - 3 ) tri = n - 3;
	float r1 = RadicalInverse( s, 2 ), r2 = RadicalInverse( s, 3 );
	if ( r1 + r2 > 1.0f ) { r1 = 1.0f - r1; r2 = 1.0f - r2; }
	const float *p0 = w, *p1 = w + (size_t)( tri + 1 ) * 3, *p2 = w + (size_t)( tri + 2 ) * 3;
	for ( int k = 0; k < 3; ++k )
		out[k] = p0[k] + r1 * ( p1[k] - p0[k] ) + r2 * ( p2[k] - p0[k] );
}

// Follows a ray a -> b from portal to portal, starting in the leaf of the src portal.
// At each step: among the outgoing portals of the current leaf, keep the closest one
// ahead of the ray whose hit point lies inside its winding, mark that portal visible
// from src, then move on to the next leaf.
// Returns 1 if dst is reached, 0 otherwise. *steps_out = portals crossed.
// With RAY_FLAG_EXTEND the ray keeps going after dst (until it dies).
__device__ int WalkStabbingLine(
	const GPUPortal *portals, const GPULeaf *leafs, int num_leafs,
	const int *leaf_portals, const float *winding_pts,
	unsigned int *portalvis_words, int portalbytes, int max_steps, int flags, float edge_eps,
	int src, int dst, const float a[3], const float b[3], int *steps_out )
{
	float dir0 = b[0] - a[0], dir1 = b[1] - a[1], dir2 = b[2] - a[2];
	int leaf = portals[src].leaf;
	float tcur = 0.0f;
	const float tmax = ( flags & RAY_FLAG_EXTEND ) ? RAY_T_INF : RAY_T_MAX;
	int reached = 0;
	for ( int step = 0; step < max_steps; ++step )
	{
		if ( leaf < 0 || leaf >= num_leafs )
		{
			*steps_out = step;
			return 0;
		}
		GPULeaf L = leafs[leaf];
		float best_t = 3.0e38f;
		int best_q = -1;
		for ( int k = 0; k < L.portals_count; ++k )
		{
			int q = leaf_portals[L.portals_start + k];
			const GPUPortal *gq = portals + q;
			float denom = dir0*gq->plane_normal[0] + dir1*gq->plane_normal[1] + dir2*gq->plane_normal[2];
			if ( denom <= RAY_EPS_DENOM )
				continue;
			float da = a[0]*gq->plane_normal[0] + a[1]*gq->plane_normal[1] + a[2]*gq->plane_normal[2];
			float t = ( gq->plane_dist - da ) / denom;
			if ( t <= tcur + RAY_EPS_T || t > tmax || t >= best_t )
				continue;
			float x[3] = { a[0] + t*dir0, a[1] + t*dir1, a[2] + t*dir2 };
			if ( !PointInWinding( winding_pts, gq, x, edge_eps ) )
				continue;
			best_t = t;
			best_q = q;
		}
		if ( best_q < 0 )
		{
			*steps_out = step;
			return reached;
		}
		SetPortalvisBit( portalvis_words, src, best_q, portalbytes );
		if ( best_q == dst )
		{
			reached = 1;
			if ( !( flags & RAY_FLAG_EXTEND ) )
			{
				*steps_out = step;
				return 1;
			}
		}
		leaf = portals[best_q].leaf;
		tcur = best_t;
	}
	*steps_out = max_steps;
	return reached;
}

// Optional statistics (diagnostics). If stats == NULL, nothing is counted
// and the result is exactly the same.
//   stats[0] pairs tested            stats[1] pairs proven visible
//   stats[2] rays cast               stats[3] rays cast by invisible pairs
//   stats[4 + d] rays that died after crossing d portals (d >= 15: last bin)
//   stats[4 + STAT_BINS] total walk steps (leafs visited, all rays): a measure of the work
#define STAT_BINS 16

// 1 thread = 1 (src, dst) pair. `start` = index of the first pair of the chunk.
extern "C" __global__ void RaycastStabKernel(
	const GPUPortal *portals, int num_portals,
	const GPULeaf *leafs, int num_leafs,
	const int *leaf_portals,
	const float *winding_pts,
	const unsigned char *portalflood,
	unsigned int *portalvis_words,
	int portalbytes, int samples_per_portal, int max_steps, int flags, float edge_eps, int ringmask,
	unsigned long long start,
	unsigned long long *stats )
{
	unsigned long long idx = start + (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long npairs = (unsigned long long)num_portals * (unsigned long long)num_portals;
	if ( idx >= npairs )
		return;
	int src = (int)( idx / (unsigned long long)num_portals );
	int dst = (int)( idx - (unsigned long long)src * (unsigned long long)num_portals );
	if ( !TestBit( portalflood + (size_t)src * portalbytes, dst ) )
		return;
	if ( flags & RAY_FLAG_MIRROR )
	{
		unsigned long long midx = (unsigned long long)( dst ^ 1 ) * (unsigned long long)num_portals + (unsigned long long)( src ^ 1 );
		if ( midx < idx && TestBit( portalflood + (size_t)( dst ^ 1 ) * portalbytes, src ^ 1 ) )
			return;
	}

	const GPUPortal *sp = portals + src;
	const GPUPortal *dp = portals + dst;
	if ( sp->winding_count < 3 || dp->winding_count < 3 )
		return;

	int nvs = sp->winding_count < samples_per_portal ? sp->winding_count : samples_per_portal;
	int nvd = dp->winding_count < samples_per_portal ? dp->winding_count : samples_per_portal;
	float cs[3], cd[3];
	PortalCentroid( winding_pts, sp, cs );
	PortalCentroid( winding_pts, dp, cd );

	const bool halton = ( flags & RAY_FLAG_HALTON ) != 0;
	const int nrings = __popc( (unsigned int)ringmask );
	int ss = halton ? samples_per_portal : 1 + nrings * nvs;
	int sd = halton ? samples_per_portal : 1 + nrings * nvd;
	unsigned long long rays = 0, walked = 0;
	unsigned long long hist[STAT_BINS];
	if ( stats )
		for ( int h = 0; h < STAT_BINS; ++h )
			hist[h] = 0;
	int visible = 0;
	const bool stop_on_hit = !( flags & RAY_FLAG_EXHAUSTIVE );
	for ( int si = 0; si < ss && !( visible && stop_on_hit ); ++si )
	{
		float a[3];
		if ( halton ) PortalHaltonSample( winding_pts, sp, si + 1, a );
		else PortalInsetSample( winding_pts, sp, cs, nvs, ringmask, si, a );
		for ( int di = 0; di < sd; ++di )
		{
			float b[3];
			if ( halton ) PortalHaltonSample( winding_pts, dp, di + 1, b );
			else PortalInsetSample( winding_pts, dp, cd, nvd, ringmask, di, b );
			int steps = 0;
			++rays;
			const int hit = WalkStabbingLine( portals, leafs, num_leafs, leaf_portals, winding_pts,
					portalvis_words, portalbytes, max_steps, flags, edge_eps, src, dst, a, b, &steps );
			walked += (unsigned long long)steps + 1;
			if ( hit )
			{
				visible = 1;
				if ( stop_on_hit )
					break;
				continue;
			}
			if ( stats )
				++hist[steps < STAT_BINS ? steps : STAT_BINS - 1];
		}
	}
	if ( stats )
	{
		atomicAdd( stats + 0, 1ull );
		atomicAdd( stats + 1, (unsigned long long)visible );
		atomicAdd( stats + 2, rays );
		if ( !visible )
			atomicAdd( stats + 3, rays );
		for ( int h = 0; h < STAT_BINS; ++h )
			if ( hist[h] )
				atomicAdd( stats + 4 + h, hist[h] );
		atomicAdd( stats + 4 + STAT_BINS, walked );
	}
}
