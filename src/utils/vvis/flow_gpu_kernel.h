// flow_gpu_kernel.h
// OpenCL C source of the "stabbing lines" kernel, compiled at run time by the
// driver (clBuildProgram). Same computation as flow_gpu_kernel.cu: same
// sampling, same epsilons, same traversal.
//
// Principle: for each (src, dst) pair allowed by portalflood, segments are cast
// between interior points of src and dst, and each segment is followed from leaf
// to leaf through the portals it pierces:
//
//   src        q1         q2         dst
//    |  leaf A  |  leaf B  |  leaf C  |
//    a ---------x----------x--------- b
//
// In each leaf, the closest outgoing portal (smallest t > tcur) is taken whose
// plane is crossed along its normal and whose winding contains the hit point
// (with the signed edge margin edge_eps). q1, q2 and dst are then marked
// visible from src.
//
// Points per portal: centroid + the rings enabled in ringmask (t = 0.9, 0.5 and
// 0.99 towards the vertices, at most samples_per_portal vertices per ring), or,
// with RAY_FLAG_HALTON, samples_per_portal points spread over the whole surface.

#pragma once

static const char g_szRaycastKernelSource[] = R"CLC(
#pragma OPENCL FP_CONTRACT OFF
#define RAY_EPS_DENOM 1e-6f
#define RAY_EPS_T     1e-5f
#define RAY_T_MAX     1.001f
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

// Sets bit dst of row src. The buffer is viewed as 32-bit words
// (little-endian order) so that atomic_or can be used.
inline void SetPortalvisBit( __global volatile uint *portalvis_words, int src, int dst, int portalbytes )
{
	ulong byte_index = (ulong)src * (ulong)portalbytes + (ulong)( dst >> 3 );
	uint shift = (uint)( byte_index & 3 ) * 8u + (uint)( dst & 7 );
	uint mask = 1u << shift;
	__global volatile uint *w = portalvis_words + ( byte_index >> 2 );
	// Read first: most bits are already set, which avoids the atomic.
	if ( !( *w & mask ) )
		atomic_or( w, mask );
}

inline int TestBit( __global const uchar *row, int bit )
{
	return ( row[bit >> 3] >> ( bit & 7 ) ) & 1;
}

// A point is inside the winding if it lies on the inner side of every edge, with a
// signed margin `eps`: eps < 0 requires it to be clearly inside (see the .cu).
int PointInWinding( __global const float *winding_pts, __global const GPUPortal *gp, const float x[3], float eps )
{
	int allpos = 1, allneg = 1;
	int n = gp->winding_count;
	__global const float *w = winding_pts + (size_t)gp->winding_start * 3;
	for ( int k = 0; k < n; ++k )
	{
		int k1 = ( k + 1 == n ) ? 0 : k + 1;
		float e0 = w[k1*3+0] - w[k*3+0], e1 = w[k1*3+1] - w[k*3+1], e2 = w[k1*3+2] - w[k*3+2];
		float d0 = x[0] - w[k*3+0], d1 = x[1] - w[k*3+1], d2 = x[2] - w[k*3+2];
		float c0 = e1*d2 - e2*d1, c1 = e2*d0 - e0*d2, c2 = e0*d1 - e1*d0;
		float s = c0*gp->plane_normal[0] + c1*gp->plane_normal[1] + c2*gp->plane_normal[2];
		float tol = eps * sqrt( e0*e0 + e1*e1 + e2*e2 );
		if ( s < -tol ) allpos = 0;
		if ( s > tol ) allneg = 0;
	}
	return allpos || allneg;
}

void PortalCentroid( __global const float *winding_pts, __global const GPUPortal *gp, float out[3] )
{
	__global const float *w = winding_pts + (size_t)gp->winding_start * 3;
	float cx = 0.0f, cy = 0.0f, cz = 0.0f;
	for ( int k = 0; k < gp->winding_count; ++k )
	{
		cx += w[k*3+0]; cy += w[k*3+1]; cz += w[k*3+2];
	}
	float inv = 1.0f / (float)gp->winding_count;
	out[0] = cx * inv; out[1] = cy * inv; out[2] = cz * inv;
}

// s = 0: centroid; otherwise the rings enabled in ringmask (bit 0: t=0.9, bit 1: t=0.5,
// bit 2: t=0.99 = corners), nv points each.
void PortalInsetSample( __global const float *winding_pts, __global const GPUPortal *gp,
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
	__global const float *w = winding_pts + (size_t)( gp->winding_start + vi ) * 3;
	out[0] = c[0] + t * ( w[0] - c[0] );
	out[1] = c[1] + t * ( w[1] - c[1] );
	out[2] = c[2] + t * ( w[2] - c[2] );
}

// Radical inverse (Halton sequence) of index i in the given base, in [0,1).
float RadicalInverse( int i, int base )
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
void PortalHaltonSample( __global const float *winding_pts, __global const GPUPortal *gp, int s, float out[3] )
{
	int n = gp->winding_count;
	__global const float *w = winding_pts + (size_t)gp->winding_start * 3;
	int tri = (int)( RadicalInverse( s, 5 ) * (float)( n - 2 ) );
	if ( tri > n - 3 ) tri = n - 3;
	float r1 = RadicalInverse( s, 2 ), r2 = RadicalInverse( s, 3 );
	if ( r1 + r2 > 1.0f ) { r1 = 1.0f - r1; r2 = 1.0f - r2; }
	__global const float *p0 = w, *p1 = w + (size_t)( tri + 1 ) * 3, *p2 = w + (size_t)( tri + 2 ) * 3;
	for ( int k = 0; k < 3; ++k )
		out[k] = p0[k] + r1 * ( p1[k] - p0[k] ) + r2 * ( p2[k] - p0[k] );
}

// Walks a->b starting from the leaf of src. Sets the bit of every portal crossed.
// Returns 1 if dst is reached. With RAY_FLAG_EXTEND the ray keeps going after dst.
int WalkStabbingLine(
	__global const GPUPortal *portals, __global const GPULeaf *leafs, int num_leafs,
	__global const int *leaf_portals, __global const float *winding_pts,
	__global volatile uint *portalvis_words, int portalbytes, int max_steps, int flags, float edge_eps,
	int src, int dst, const float a[3], const float b[3] )
{
	float dir0 = b[0] - a[0], dir1 = b[1] - a[1], dir2 = b[2] - a[2];
	int leaf = portals[src].leaf;
	float tcur = 0.0f;
	const float tmax = ( flags & RAY_FLAG_EXTEND ) ? RAY_T_INF : RAY_T_MAX;
	int reached = 0;
	for ( int step = 0; step < max_steps; ++step )
	{
		if ( leaf < 0 || leaf >= num_leafs )
			return 0;
		GPULeaf L = leafs[leaf];
		float best_t = 3.0e38f;
		int best_q = -1;
		for ( int k = 0; k < L.portals_count; ++k )
		{
			int q = leaf_portals[L.portals_start + k];
			__global const GPUPortal *gq = portals + q;
			float denom = dir0*gq->plane_normal[0] + dir1*gq->plane_normal[1] + dir2*gq->plane_normal[2];
			if ( denom <= RAY_EPS_DENOM )
				continue; // only cross along the portal normal
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
			return reached;
		SetPortalvisBit( portalvis_words, src, best_q, portalbytes );
		if ( best_q == dst )
		{
			reached = 1;
			if ( !( flags & RAY_FLAG_EXTEND ) )
				return 1;
		}
		leaf = portals[best_q].leaf;
		tcur = best_t;
	}
	return reached; // step limit reached: nothing else is confirmed
}

// 1 work-item = 1 candidate (src, dst) pair (bit set in portalflood[src]).
// Launched in chunks via global_work_offset (avoids the Windows watchdog).
__kernel void RaycastStabKernel(
	__global const GPUPortal *portals, int num_portals,
	__global const GPULeaf *leafs, int num_leafs,
	__global const int *leaf_portals,
	__global const float *winding_pts,
	__global const uchar *portalflood,
	__global volatile uint *portalvis_words,
	int portalbytes, int samples_per_portal, int max_steps, int flags, float edge_eps, int ringmask )
{
	ulong idx = get_global_id( 0 );
	ulong npairs = (ulong)num_portals * (ulong)num_portals;
	if ( idx >= npairs )
		return;
	int src = (int)( idx / (ulong)num_portals );
	int dst = (int)( idx - (ulong)src * (ulong)num_portals );
	if ( !TestBit( portalflood + (size_t)src * portalbytes, dst ) )
		return;
	if ( flags & RAY_FLAG_MIRROR )
	{
		ulong midx = (ulong)( dst ^ 1 ) * (ulong)num_portals + (ulong)( src ^ 1 );
		if ( midx < idx && TestBit( portalflood + (size_t)( dst ^ 1 ) * portalbytes, src ^ 1 ) )
			return;
	}

	__global const GPUPortal *sp = portals + src;
	__global const GPUPortal *dp = portals + dst;
	if ( sp->winding_count < 3 || dp->winding_count < 3 )
		return;

	int nvs = sp->winding_count < samples_per_portal ? sp->winding_count : samples_per_portal;
	int nvd = dp->winding_count < samples_per_portal ? dp->winding_count : samples_per_portal;
	float cs[3], cd[3];
	PortalCentroid( winding_pts, sp, cs );
	PortalCentroid( winding_pts, dp, cd );

	const int halton = ( flags & RAY_FLAG_HALTON ) != 0;
	const int nrings = popcount( (uint)ringmask );
	int ss = halton ? samples_per_portal : 1 + nrings * nvs;
	int sd = halton ? samples_per_portal : 1 + nrings * nvd;
	int visible = 0;
	const int stop_on_hit = !( flags & RAY_FLAG_EXHAUSTIVE );
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
			if ( WalkStabbingLine( portals, leafs, num_leafs, leaf_portals, winding_pts,
					portalvis_words, portalbytes, max_steps, flags, edge_eps, src, dst, a, b ) )
			{
				visible = 1;
				if ( stop_on_hit )
					break;
			}
		}
	}
}
)CLC";
