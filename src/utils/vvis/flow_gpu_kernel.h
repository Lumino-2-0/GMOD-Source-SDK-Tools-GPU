// flow_gpu_kernel.h
// Source OpenCL C du noyau "stabbing lines", compile a l'execution par le
// pilote (clBuildProgram). Meme calcul que flow_gpu_kernel.cu : meme
// echantillonnage, memes epsilons, meme traversee.
//
// Principe : pour chaque paire (src, dst) autorisee par portalflood, on tire
// des segments entre des points interieurs de src et de dst, et on suit
// chaque segment de leaf en leaf a travers les portails qu'il perce :
//
//   src        q1         q2         dst
//    |  leaf A  |  leaf B  |  leaf C  |
//    a ---------x----------x--------- b
//
// Dans chaque leaf, on prend le portail sortant le plus proche (plus petit
// t > tcur) dont le plan est coupe dans le sens de sa normale et dont le
// winding contient strictement le point d'impact. q1, q2 et dst sont alors
// marques visibles depuis src.
//
// Points par portail : centroide + 2 anneaux "inset" (t = 0.9 et t = 0.5 vers
// les sommets, au plus samples_per_portal sommets par anneau).

#pragma once

static const char g_szRaycastKernelSource[] = R"CLC(
#pragma OPENCL FP_CONTRACT OFF
#define RAY_NUM_RINGS 2
#define RAY_EPS_DENOM 1e-6f
#define RAY_EPS_T     1e-5f
#define RAY_T_MAX     1.001f

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

// Pose le bit dst de la ligne src. Le tampon est vu en mots de 32 bits
// (ordre little-endian) pour pouvoir utiliser atomic_or.
inline void SetPortalvisBit( __global volatile uint *portalvis_words, int src, int dst, int portalbytes )
{
	ulong byte_index = (ulong)src * (ulong)portalbytes + (ulong)( dst >> 3 );
	uint shift = (uint)( byte_index & 3 ) * 8u + (uint)( dst & 7 );
	atomic_or( portalvis_words + ( byte_index >> 2 ), 1u << shift );
}

inline int TestBit( __global const uchar *row, int bit )
{
	return ( row[bit >> 3] >> ( bit & 7 ) ) & 1;
}

// Point strictement a l'interieur du winding convexe (tous les produits mixtes
// de meme signe, aucun nul). Strict = jamais de confirmation sur un bord.
int PointInWinding( __global const float *winding_pts, __global const GPUPortal *gp, const float x[3] )
{
	int pos = 0, neg = 0;
	int n = gp->winding_count;
	__global const float *w = winding_pts + (size_t)gp->winding_start * 3;
	for ( int k = 0; k < n; ++k )
	{
		int k1 = ( k + 1 == n ) ? 0 : k + 1;
		float e0 = w[k1*3+0] - w[k*3+0], e1 = w[k1*3+1] - w[k*3+1], e2 = w[k1*3+2] - w[k*3+2];
		float d0 = x[0] - w[k*3+0], d1 = x[1] - w[k*3+1], d2 = x[2] - w[k*3+2];
		float c0 = e1*d2 - e2*d1, c1 = e2*d0 - e0*d2, c2 = e0*d1 - e1*d0;
		float s = c0*gp->plane_normal[0] + c1*gp->plane_normal[1] + c2*gp->plane_normal[2];
		if ( s > 0.0f ) ++pos;
		else if ( s < 0.0f ) ++neg;
		else return 0;
	}
	return pos == n || neg == n;
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

// s = 0 : centroide ; sinon inset (anneau 0 : t=0.9, anneau 1 : t=0.5).
void PortalInsetSample( __global const float *winding_pts, __global const GPUPortal *gp,
	const float c[3], int nv, int s, float out[3] )
{
	if ( s <= 0 || nv <= 0 )
	{
		out[0] = c[0]; out[1] = c[1]; out[2] = c[2];
		return;
	}
	int idx = s - 1;
	int ring = idx / nv;
	int vi = idx - ring * nv;
	float t = ( ring == 0 ) ? 0.9f : 0.5f;
	__global const float *w = winding_pts + (size_t)( gp->winding_start + vi ) * 3;
	out[0] = c[0] + t * ( w[0] - c[0] );
	out[1] = c[1] + t * ( w[1] - c[1] );
	out[2] = c[2] + t * ( w[2] - c[2] );
}

// Traverse a->b depuis la leaf de src. Pose le bit de chaque portail traverse.
// Retourne 1 si dst est atteint.
int WalkStabbingLine(
	__global const GPUPortal *portals, __global const GPULeaf *leafs, int num_leafs,
	__global const int *leaf_portals, __global const float *winding_pts,
	__global volatile uint *portalvis_words, int portalbytes, int max_steps,
	int src, int dst, const float a[3], const float b[3] )
{
	float dir0 = b[0] - a[0], dir1 = b[1] - a[1], dir2 = b[2] - a[2];
	int leaf = portals[src].leaf;
	float tcur = 0.0f;
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
				continue; // on ne traverse que dans le sens de la normale
			float da = a[0]*gq->plane_normal[0] + a[1]*gq->plane_normal[1] + a[2]*gq->plane_normal[2];
			float t = ( gq->plane_dist - da ) / denom;
			if ( t <= tcur + RAY_EPS_T || t > RAY_T_MAX || t >= best_t )
				continue;
			float x[3] = { a[0] + t*dir0, a[1] + t*dir1, a[2] + t*dir2 };
			if ( !PointInWinding( winding_pts, gq, x ) )
				continue;
			best_t = t;
			best_q = q;
		}
		if ( best_q < 0 )
			return 0;
		SetPortalvisBit( portalvis_words, src, best_q, portalbytes );
		if ( best_q == dst )
			return 1;
		leaf = portals[best_q].leaf;
		tcur = best_t;
	}
	return 0; // borne atteinte : rien d'autre n'est confirme
}

// 1 work-item = 1 paire (src, dst) candidate (bit portalflood[src] pose).
// Lance par tranches via global_work_offset (evite le watchdog Windows).
__kernel void RaycastStabKernel(
	__global const GPUPortal *portals, int num_portals,
	__global const GPULeaf *leafs, int num_leafs,
	__global const int *leaf_portals,
	__global const float *winding_pts,
	__global const uchar *portalflood,
	__global volatile uint *portalvis_words,
	int portalbytes, int samples_per_portal, int max_steps )
{
	ulong idx = get_global_id( 0 );
	ulong npairs = (ulong)num_portals * (ulong)num_portals;
	if ( idx >= npairs )
		return;
	int src = (int)( idx / (ulong)num_portals );
	int dst = (int)( idx - (ulong)src * (ulong)num_portals );
	if ( !TestBit( portalflood + (size_t)src * portalbytes, dst ) )
		return;

	__global const GPUPortal *sp = portals + src;
	__global const GPUPortal *dp = portals + dst;
	if ( sp->winding_count < 3 || dp->winding_count < 3 )
		return;

	int nvs = sp->winding_count < samples_per_portal ? sp->winding_count : samples_per_portal;
	int nvd = dp->winding_count < samples_per_portal ? dp->winding_count : samples_per_portal;
	float cs[3], cd[3];
	PortalCentroid( winding_pts, sp, cs );
	PortalCentroid( winding_pts, dp, cd );

	int ss = 1 + RAY_NUM_RINGS * nvs;
	int sd = 1 + RAY_NUM_RINGS * nvd;
	for ( int si = 0; si < ss; ++si )
	{
		float a[3];
		PortalInsetSample( winding_pts, sp, cs, nvs, si, a );
		for ( int di = 0; di < sd; ++di )
		{
			float b[3];
			PortalInsetSample( winding_pts, dp, cd, nvd, di, b );
			if ( WalkStabbingLine( portals, leafs, num_leafs, leaf_portals, winding_pts,
					portalvis_words, portalbytes, max_steps, src, dst, a, b ) )
				return;
		}
	}
}
)CLC";
