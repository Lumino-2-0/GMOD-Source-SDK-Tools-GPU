// flow_gpu_kernel.cu
//
// Portal-to-portal visibility kernel ("stabbing lines"), shared by every engine:
//   - CUDA   : build_ptx.ps1 turns this file into PTX (flow_gpu_kernel_ptx.h), which is
//              embedded in the DLL. The NVIDIA driver compiles the PTX for the installed
//              GPU, so users do not need nvcc. Only rerun build_ptx.ps1 when this file changes.
//   - CPU    : flow_cpu_ray.cpp and flow_cpu_ray_avx2.cpp #include this file as plain C++.
//   - OpenCL : flow_gpu_kernel.h holds the same code as OpenCL C. Any change made here
//              must be mirrored there, and vice versa.
//
// Every engine must produce bit-identical results, whatever the share of the work it
// gets: no FMA (-fmad=false, FP_CONTRACT OFF, fp_contract(off) on the CPU), IEEE division.
// The edge lengths (the only square roots) are precomputed once on the CPU.
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
// PARALLELISM: the host splits the pair index space (src * num_portals + dst) into
// ranges [start, end) and hands them out to the GPU(s) and to the CPU threads.
// The rays of a pair are numbered r = si * sd + di and cast in groups of RAY_GROUP (32):
// the pair stops after the first group that contains a hit. On the GPU, the 32 rays of
// a group run on the 32 lanes of a warp (CUDA) or work-group (OpenCL), which keeps every
// lane busy even when candidate pairs are sparse; the CPU casts the same groups one ray
// after the other. Every engine therefore casts exactly the same rays.
// With RAY_FLAG_EXHAUSTIVE every ray is cast: the rays are then independent and the GPU
// kernels split them into chunks of RAY_CHUNK rays (grid dimension y) to spread the
// pairs that need many rays over many warps.

#ifdef __CUDACC__
#define RAY_FN                 __device__ __forceinline__
#define RAY_ATOMIC_OR( p, m )  atomicOr( (unsigned int *)( p ), ( m ) )
#define RAY_POPC( x )          __popc( x )
#else
// CPU build: flow_gpu.h (GPUPortal, GPULeaf, RAY_FLAG_*) is included before this file.
#define RAY_FN                 static __forceinline
#define RAY_ATOMIC_OR( p, m )  _InterlockedOr( (volatile long *)( p ), (long)( m ) )
#define RAY_POPC( x )          RayPopc32( x )
// Portable bit count (the POPCNT instruction is not guaranteed on every x64 CPU).
static __forceinline int RayPopc32( unsigned int v )
{
	int c = 0;
	for ( ; v; v &= v - 1 )
		++c;
	return c;
}
#endif

// Ray almost parallel to a portal plane: ignored.
#define RAY_EPS_DENOM 1e-6f
// Margin so the portal that was just crossed is not hit again.
#define RAY_EPS_T     1e-5f
// The ray goes from a (t = 0) to b (t = 1); small margin for rounding errors.
#define RAY_T_MAX     1.001f
#define RAY_T_INF     1.0e30f

// Kernel flags (`flags` argument, same values as flow_gpu.h).
//   RAY_FLAG_EXTEND     : the ray does not stop at dst, it keeps going as long as it
//                         crosses portals (also marking what lies behind).
//   RAY_FLAG_EXHAUSTIVE : cast every ray of the pair even after a hit.
//   RAY_FLAG_HALTON     : dense sampling: `samples_per_portal` points spread over the
//                         whole portal surface (Halton sequence) instead of the rings.
//   RAY_FLAG_MIRROR     : visibility is symmetric (the line a -> b is also the line
//                         b -> a). If the mirror pair (dst^1 -> src^1) is also a candidate,
//                         only one of the two is tested; the host copies the result.
#ifndef RAY_FLAG_EXTEND
#define RAY_FLAG_EXTEND     1
#define RAY_FLAG_EXHAUSTIVE 2
#define RAY_FLAG_HALTON     4
#define RAY_FLAG_MIRROR     8
#endif

#ifdef __CUDACC__
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
#endif

// Sets the (src -> dst) bit in portalvis. Words are 32-bit and several threads
// may target the same word: an atomic OR avoids losing a bit.
RAY_FN void SetPortalvisBit( unsigned int *words, int src, int dst, int portalbytes )
{
	unsigned long long byte_index = (unsigned long long)src * (unsigned long long)portalbytes + (unsigned long long)( dst >> 3 );
	unsigned int shift = (unsigned int)( byte_index & 3 ) * 8u + (unsigned int)( dst & 7 );
	unsigned int mask = 1u << shift;
	volatile unsigned int *w = words + ( byte_index >> 2 );
	// Read first: most bits are already set, which avoids the atomic.
	if ( !( *w & mask ) )
		RAY_ATOMIC_OR( w, mask );
}

RAY_FN int TestBit( const unsigned char *row, int bit )
{
	return ( row[bit >> 3] >> ( bit & 7 ) ) & 1;
}

// A point is inside the winding if it lies on the inner side of every edge, with a
// signed margin: winding_edges[k] = ( edge k vector, tol = eps * edge length ), see
// FlowCPU_BuildEdgeTable. vvis_gpu uses eps < 0 (tol < 0): the point must be clearly
// inside, because a ray that grazes an edge runs along a wall and accepting it made
// hidden areas visible. s = scalar triple product = signed distance to the edge * length.
// Stops as soon as the point is known to be outside (same result, less work).
RAY_FN int PointInWinding( const float *winding_pts, const float *winding_edges, const GPUPortal *gp, const float x[3] )
{
	int allpos = 1, allneg = 1;
	const int n = gp->winding_count;
	const float *w = winding_pts + (size_t)gp->winding_start * 3;
	const float *e = winding_edges + (size_t)gp->winding_start * 4;
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

RAY_FN void PortalCentroid( const float *winding_pts, const GPUPortal *gp, float out[3] )
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
RAY_FN void PortalInsetSample( const float *winding_pts, const GPUPortal *gp,
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
RAY_FN float RadicalInverse( int i, int base )
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
RAY_FN void PortalHaltonSample( const float *winding_pts, const GPUPortal *gp, int s, float out[3] )
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
RAY_FN int WalkStabbingLine(
	const GPUPortal *portals, const GPULeaf *leafs, int num_leafs,
	const int *leaf_portals, const float *winding_pts, const float *winding_edges,
	unsigned int *portalvis_words, int portalbytes, int max_steps, int flags,
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
			// Cheap rejection: a point inside the winding is inside the portal's bounding
			// sphere. The 1-unit margin keeps this test strictly looser than the winding test.
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

// Candidate filter: the bit is set in portalflood and, with RAY_FLAG_MIRROR, the pair is
// not the mirror of a candidate with a smaller index (that one is tested instead).
RAY_FN int RayPairIsCandidate( const unsigned char *portalflood, int num_portals, int portalbytes,
	int flags, int src, int dst )
{
	if ( !TestBit( portalflood + (size_t)src * portalbytes, dst ) )
		return 0;
	if ( flags & RAY_FLAG_MIRROR )
	{
		unsigned long long idx = (unsigned long long)src * (unsigned long long)num_portals + (unsigned long long)dst;
		unsigned long long midx = (unsigned long long)( dst ^ 1 ) * (unsigned long long)num_portals + (unsigned long long)( src ^ 1 );
		if ( midx < idx && TestBit( portalflood + (size_t)( dst ^ 1 ) * portalbytes, src ^ 1 ) )
			return 0;
	}
	return 1;
}

// Optional statistics (diagnostics, CUDA only):
//   stats[0] pairs tested            stats[1] pairs proven visible
//   stats[2] rays cast               stats[3] rays cast by invisible pairs
//   stats[4 + d] rays that died after crossing d portals (d >= 15: last bin)
//   stats[4 + STAT_BINS] total walk steps (leafs visited, all rays): a measure of the work
#define STAT_BINS 16

// Rays of a pair are cast in groups of RAY_GROUP; the pair stops after the first group
// that contains a hit. Must match the warp / work-group width of the GPU kernels.
#define RAY_GROUP 32
// Rays per grid row with RAY_FLAG_EXHAUSTIVE (multiple of RAY_GROUP). The host uses the
// same value to size the grid.
#define RAY_CHUNK 256

// Sample points of the pair: counts and centroids, shared by RaycastPair and the warp kernel.
typedef struct
{
	float cs[3], cd[3];
	int nvs, nvd, ss, sd, halton;
} RayPairSetup;

RAY_FN void RayPairInit( const float *winding_pts, const GPUPortal *sp, const GPUPortal *dp,
	int samples_per_portal, int flags, int ringmask, RayPairSetup *ps )
{
	ps->nvs = sp->winding_count < samples_per_portal ? sp->winding_count : samples_per_portal;
	ps->nvd = dp->winding_count < samples_per_portal ? dp->winding_count : samples_per_portal;
	PortalCentroid( winding_pts, sp, ps->cs );
	PortalCentroid( winding_pts, dp, ps->cd );
	ps->halton = ( flags & RAY_FLAG_HALTON ) != 0;
	const int nrings = RAY_POPC( (unsigned int)ringmask );
	ps->ss = ps->halton ? samples_per_portal : 1 + nrings * ps->nvs;
	ps->sd = ps->halton ? samples_per_portal : 1 + nrings * ps->nvd;
}

RAY_FN void RaySrcPoint( const float *winding_pts, const GPUPortal *sp, const RayPairSetup *ps, int ringmask, int si, float a[3] )
{
	if ( ps->halton ) PortalHaltonSample( winding_pts, sp, si + 1, a );
	else PortalInsetSample( winding_pts, sp, ps->cs, ps->nvs, ringmask, si, a );
}

RAY_FN void RayDstPoint( const float *winding_pts, const GPUPortal *dp, const RayPairSetup *ps, int ringmask, int di, float b[3] )
{
	if ( ps->halton ) PortalHaltonSample( winding_pts, dp, di + 1, b );
	else PortalInsetSample( winding_pts, dp, ps->cd, ps->nvd, ringmask, di, b );
}

// Casts the rays of one candidate pair, one after the other (CPU engine); returns 1 if
// a ray reached dst. hist: NULL, or STAT_BINS counters of the depth at which missed rays die.
RAY_FN int RaycastPair(
	const GPUPortal *portals, const GPULeaf *leafs, int num_leafs,
	const int *leaf_portals, const float *winding_pts, const float *winding_edges,
	unsigned int *portalvis_words, int portalbytes, int samples_per_portal, int max_steps,
	int flags, int ringmask, int src, int dst,
	unsigned long long *rays_out, unsigned long long *walked_out, unsigned long long *hist )
{
	const GPUPortal *sp = portals + src;
	const GPUPortal *dp = portals + dst;
	*rays_out = 0;
	*walked_out = 0;
	if ( sp->winding_count < 3 || dp->winding_count < 3 )
		return 0;

	RayPairSetup ps;
	RayPairInit( winding_pts, sp, dp, samples_per_portal, flags, ringmask, &ps );
	const int total = ps.ss * ps.sd;
	unsigned long long rays = 0, walked = 0;
	int visible = 0;
	const int stop_on_hit = !( flags & RAY_FLAG_EXHAUSTIVE );
	int lastSi = -1;
	float a[3] = { 0.0f, 0.0f, 0.0f };
	for ( int base = 0; base < total; base += RAY_GROUP )
	{
		const int rend = base + RAY_GROUP < total ? base + RAY_GROUP : total;
		for ( int r = base; r < rend; ++r )
		{
			const int si = r / ps.sd, di = r - si * ps.sd;
			if ( si != lastSi )
			{
				RaySrcPoint( winding_pts, sp, &ps, ringmask, si, a );
				lastSi = si;
			}
			float b[3];
			RayDstPoint( winding_pts, dp, &ps, ringmask, di, b );
			int steps = 0;
			++rays;
			const int hit = WalkStabbingLine( portals, leafs, num_leafs, leaf_portals, winding_pts, winding_edges,
				portalvis_words, portalbytes, max_steps, flags, src, dst, a, b, &steps );
			walked += (unsigned long long)steps + 1;
			if ( hit )
				visible = 1;
			else if ( hist )
				++hist[steps < STAT_BINS ? steps : STAT_BINS - 1];
		}
		if ( visible && stop_on_hit )
			break;
	}
	*rays_out = rays;
	*walked_out = walked;
	return visible;
}

#ifdef __CUDACC__
// Rays of one pair spread over the 32 lanes of the warp: lane l casts ray base + l of
// each group. Same rays and same stop rule as RaycastPair. Called by the whole warp.
// With RAY_FLAG_EXHAUSTIVE, only the chunk blockIdx.y of the rays is cast.
__device__ void RaycastPairWarp(
	const GPUPortal *portals, const GPULeaf *leafs, int num_leafs,
	const int *leaf_portals, const float *winding_pts, const float *winding_edges,
	unsigned int *portalvis_words, int portalbytes, int samples_per_portal, int max_steps,
	int flags, int ringmask, int src, int dst, int lane, unsigned long long *stats )
{
	const GPUPortal *sp = portals + src;
	const GPUPortal *dp = portals + dst;
	if ( sp->winding_count < 3 || dp->winding_count < 3 )
		return;
	RayPairSetup ps;
	RayPairInit( winding_pts, sp, dp, samples_per_portal, flags, ringmask, &ps );
	const int total = ps.ss * ps.sd;
	const int stop_on_hit = !( flags & RAY_FLAG_EXHAUSTIVE );
	int rbegin = 0, rend = total;
	if ( !stop_on_hit )
	{
		rbegin = (int)blockIdx.y * RAY_CHUNK;
		if ( rbegin >= total )
			return;
		rend = rbegin + RAY_CHUNK < total ? rbegin + RAY_CHUNK : total;
	}
	unsigned long long rays = 0, walked = 0;
	int visible = 0;
	for ( int base = rbegin; base < rend; base += RAY_GROUP )
	{
		const int r = base + lane;
		int hit = 0;
		if ( r < rend )
		{
			const int si = r / ps.sd, di = r - si * ps.sd;
			float a[3], b[3];
			RaySrcPoint( winding_pts, sp, &ps, ringmask, si, a );
			RayDstPoint( winding_pts, dp, &ps, ringmask, di, b );
			int steps = 0;
			hit = WalkStabbingLine( portals, leafs, num_leafs, leaf_portals, winding_pts, winding_edges,
				portalvis_words, portalbytes, max_steps, flags, src, dst, a, b, &steps );
			++rays;
			walked += (unsigned long long)steps + 1;
			if ( !hit && stats )
				atomicAdd( stats + 4 + ( steps < STAT_BINS ? steps : STAT_BINS - 1 ), 1ull );
		}
		if ( __any_sync( 0xffffffffu, hit ) )
			visible = 1;
		if ( visible && stop_on_hit )
			break;
	}
	if ( stats )
	{
		atomicAdd( stats + 2, rays );
		if ( !visible )
			atomicAdd( stats + 3, rays );
		atomicAdd( stats + 4 + STAT_BINS, walked );
		if ( lane == 0 && blockIdx.y == 0 )
		{
			atomicAdd( stats + 0, 1ull );
			atomicAdd( stats + 1, (unsigned long long)visible );
		}
	}
}

// 1 thread = 1 index of the range [start, end) of the pair index space. Each warp finds
// the candidate pairs among its 32 indices, then casts their rays together, pair by pair.
// The block size must be a multiple of 32; threads past `end` still take part.
// gridDim.y = 1, or the number of ray chunks with RAY_FLAG_EXHAUSTIVE.
extern "C" __global__ void RaycastStabKernel(
	const GPUPortal *portals, int num_portals,
	const GPULeaf *leafs, int num_leafs,
	const int *leaf_portals,
	const float *winding_pts,
	const float *winding_edges,
	const unsigned char *portalflood,
	unsigned int *portalvis_words,
	int portalbytes, int samples_per_portal, int max_steps, int flags, int ringmask,
	unsigned long long start, unsigned long long end,
	unsigned long long *stats )
{
	const unsigned long long idx = start + (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
	const int lane = threadIdx.x & 31;
	int src = 0, dst = 0, cand = 0;
	if ( idx < end )
	{
		src = (int)( idx / (unsigned long long)num_portals );
		dst = (int)( idx - (unsigned long long)src * (unsigned long long)num_portals );
		cand = RayPairIsCandidate( portalflood, num_portals, portalbytes, flags, src, dst );
	}
	unsigned int pending = __ballot_sync( 0xffffffffu, cand );
	while ( pending )
	{
		const int l = __ffs( pending ) - 1;
		pending &= pending - 1;
		const int ps = __shfl_sync( 0xffffffffu, src, l );
		const int pd = __shfl_sync( 0xffffffffu, dst, l );
		RaycastPairWarp( portals, leafs, num_leafs, leaf_portals, winding_pts, winding_edges,
			portalvis_words, portalbytes, samples_per_portal, max_steps, flags, ringmask, ps, pd, lane, stats );
	}
}
#endif
