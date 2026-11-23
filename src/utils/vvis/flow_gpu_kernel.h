// flow_gpu_kernel.h
// OpenCL C source of the "stabbing lines" kernel, compiled at run time by the
// driver (clBuildProgram). Same computation as flow_gpu_kernel.cu (also used by the
// CPU engine): same sampling, same epsilons, same traversal, bit-identical results.
// Any change made here must be mirrored in flow_gpu_kernel.cu, and vice versa.
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
// (with the signed edge margin baked into winding_edges). q1, q2 and dst are then
// marked visible from src.
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

// A point is inside the winding if it lies on the inner side of every edge, with the
// signed margin tol = eps * edge length stored in winding_edges (see the .cu).
// Stops as soon as the point is known to be outside.
int PointInWinding( __global const float *winding_pts, __global const float *winding_edges,
	__global const GPUPortal *gp, const float x[3] )
{
	int allpos = 1, allneg = 1;
	const int n = gp->winding_count;
	__global const float *w = winding_pts + (size_t)gp->winding_start * 3;
	__global const float *e = winding_edges + (size_t)gp->winding_start * 4;
	const float n0 = gp->plane_normal[0], n1 = gp->plane_normal[1], n2 = gp->plane_normal[2];
	for ( int k = 0; k < n; ++k )
	{
		float e0 = e[k*4+0], e1 = e[k*4+1], e2 = e[k*4+2], tol = e[k*4+3];
		float d0 = x[0] - w[k*3+0], d1 = x[1] - w[k*3+1], d2 = x[2] - w[k*3+2];
		float c0 = e1*d2 - e2*d1, c1 = e2*d0 - e0*d2, c2 = e0*d1 - e1*d0;
		float s = c0*n0 + c1*n1 + c2*n2;
		if ( s < -tol ) allpos = 0;
		if ( s > tol ) allneg = 0;
		if ( !allpos && !allneg )
			return 0;
	}
	return 1;
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
	__global const int *leaf_portals, __global const float *winding_pts, __global const float *winding_edges,
	__global volatile uint *portalvis_words, int portalbytes, int max_steps, int flags,
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
			// Bounding sphere (+1 unit margin): strictly looser than the winding test.
			float o0 = x[0] - gq->origin[0], o1 = x[1] - gq->origin[1], o2 = x[2] - gq->origin[2];
			float rr = gq->radius + 1.0f;
			if ( o0*o0 + o1*o1 + o2*o2 > rr*rr )
				continue;
			if ( !PointInWinding( winding_pts, winding_edges, gq, x ) )
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

// Rays of a pair are cast in groups of RAY_GROUP; the pair stops after the first group
// that contains a hit (same rule as RaycastPair in flow_gpu_kernel.cu).
#define RAY_GROUP 32
// With RAY_FLAG_EXHAUSTIVE: rays per row of the NDRange (dimension 1), see the host.
#define RAY_CHUNK 256

void RayPoint( __global const float *winding_pts, __global const GPUPortal *gp, const float c[3],
	int nv, int ringmask, int halton, int s, float out[3] )
{
	if ( halton ) PortalHaltonSample( winding_pts, gp, s + 1, out );
	else PortalInsetSample( winding_pts, gp, c, nv, ringmask, s, out );
}

// 1 work-item = 1 index of the range [start, end) of the pair index space. The 32 items
// of a work-group list the candidate pairs among their indices, then cast the rays of
// each pair together: item l casts ray base + l of each group. Every item reaches every
// barrier: all the branches around them only depend on local memory or kernel arguments.
// With RAY_FLAG_EXHAUSTIVE every ray is cast, and dimension 1 of the NDRange selects a
// chunk of RAY_CHUNK rays of each pair.
__kernel __attribute__(( reqd_work_group_size( RAY_GROUP, 1, 1 ) ))
void RaycastStabKernel(
	__global const GPUPortal *portals, int num_portals,
	__global const GPULeaf *leafs, int num_leafs,
	__global const int *leaf_portals,
	__global const float *winding_pts,
	__global const float *winding_edges,
	__global const uchar *portalflood,
	__global volatile uint *portalvis_words,
	int portalbytes, int samples_per_portal, int max_steps, int flags, int ringmask,
	ulong start, ulong end )
{
	__local int candSrc[RAY_GROUP];
	__local int candDst[RAY_GROUP];
	__local int groupHit[3];
	const int lid = (int)get_local_id( 0 );
	const ulong idx = start + get_global_id( 0 );
	int csrc = -1, cdst = 0;
	if ( idx < end )
	{
		int src = (int)( idx / (ulong)num_portals );
		int dst = (int)( idx - (ulong)src * (ulong)num_portals );
		int cand = TestBit( portalflood + (size_t)src * portalbytes, dst );
		if ( cand && ( flags & RAY_FLAG_MIRROR ) )
		{
			ulong midx = (ulong)( dst ^ 1 ) * (ulong)num_portals + (ulong)( src ^ 1 );
			if ( midx < idx && TestBit( portalflood + (size_t)( dst ^ 1 ) * portalbytes, src ^ 1 ) )
				cand = 0;
		}
		if ( cand )
		{
			csrc = src;
			cdst = dst;
		}
	}
	candSrc[lid] = csrc;
	candDst[lid] = cdst;
	if ( lid < 3 )
		groupHit[lid] = 0;
	barrier( CLK_LOCAL_MEM_FENCE );

	// groupHit is triple-buffered: group g uses slot g % 3 and zeroes the slot of group
	// g + 1, which nobody reads any more (its last readers passed the previous barrier).
	int slot = 0;
	const int stop_on_hit = !( flags & RAY_FLAG_EXHAUSTIVE );
	for ( int k = 0; k < RAY_GROUP; ++k )
	{
		const int src = candSrc[k];
		if ( src < 0 )
			continue;
		const int dst = candDst[k];
		__global const GPUPortal *sp = portals + src;
		__global const GPUPortal *dp = portals + dst;
		if ( sp->winding_count < 3 || dp->winding_count < 3 )
			continue;

		int nvs = sp->winding_count < samples_per_portal ? sp->winding_count : samples_per_portal;
		int nvd = dp->winding_count < samples_per_portal ? dp->winding_count : samples_per_portal;
		float cs[3], cd[3];
		PortalCentroid( winding_pts, sp, cs );
		PortalCentroid( winding_pts, dp, cd );
		const int halton = ( flags & RAY_FLAG_HALTON ) != 0;
		const int nrings = popcount( (uint)ringmask );
		const int ss = halton ? samples_per_portal : 1 + nrings * nvs;
		const int sd = halton ? samples_per_portal : 1 + nrings * nvd;
		const int total = ss * sd;
		int rbegin = 0, rend = total;
		if ( !stop_on_hit )
		{
			rbegin = (int)get_group_id( 1 ) * RAY_CHUNK;
			if ( rbegin >= total )
				continue;
			rend = rbegin + RAY_CHUNK < total ? rbegin + RAY_CHUNK : total;
		}
		int visible = 0;
		for ( int base = rbegin; base < rend; base += RAY_GROUP )
		{
			const int next = slot == 2 ? 0 : slot + 1;
			if ( lid == 0 )
				groupHit[next] = 0;
			const int r = base + lid;
			if ( r < rend )
			{
				const int si = r / sd, di = r - si * sd;
				float a[3], b[3];
				RayPoint( winding_pts, sp, cs, nvs, ringmask, halton, si, a );
				RayPoint( winding_pts, dp, cd, nvd, ringmask, halton, di, b );
				if ( WalkStabbingLine( portals, leafs, num_leafs, leaf_portals, winding_pts, winding_edges,
						portalvis_words, portalbytes, max_steps, flags, src, dst, a, b ) )
					groupHit[slot] = 1;
			}
			barrier( CLK_LOCAL_MEM_FENCE );
			if ( groupHit[slot] )
				visible = 1;
			slot = next;
			if ( visible && stop_on_hit )
				break;
		}
	}
}
)CLC";