// flow_gpu_kernel.cu
//
// Noyau CUDA du calcul de visibilite entre portails ("stabbing lines").
// Meme algorithme que la version OpenCL (flow_gpu_kernel.h) : toute modification
// ici doit etre reportee la-bas, et inversement.
//
// Ce fichier n'est pas compile par MSBuild. build_ptx.ps1 le transforme en PTX
// (flow_gpu_kernel_ptx.h), embarque dans la DLL. Le pilote NVIDIA compile ensuite
// le PTX pour la carte presente : pas besoin de nvcc chez l'utilisateur.
// A relancer uniquement quand ce fichier change.
//
// Parite avec OpenCL : -fmad=false (pas de FMA) et division IEEE.
//
// PRINCIPE
//
// VVIS fournit pour chaque portail "src" une liste de candidats (portalflood).
// Pour chaque paire (src, dst) candidate, on tire des rayons entre un point du
// portail src et un point du portail dst. Un rayon est valide s'il traverse,
// de portail en portail, une chaine de leafs jusqu'a dst :
//
//   src          P1           P2           dst
//    |  leaf A   |   leaf B   |   leaf C   |
//  a o---------->|----------->|----------->o b
//    |           |            |            |
//    chaque portail franchi est marque visible depuis src (bit portalvis)
//
// Si le rayon ne touche plus aucun portail devant lui, il est perdu. Un seul
// rayon qui arrive a dst suffit : on arrete alors la paire.
//
// IMPORTANT : ce calcul ne fait qu'AJOUTER de la visibilite. Plus de rayons =
// plus de paires prouvees visibles. Il ne peut jamais en retirer.
//
// PARALLELISME : 1 thread = 1 paire (src, dst). Le noyau est lance par tranches
// pour ne pas declencher le watchdog (TDR) de Windows.

// Nombre d'anneaux de points autour du centre (a 90 % et 50 % du centre
// vers chaque sommet).
#define RAY_NUM_RINGS 2
// Rayon presque parallele a un plan de portail : on l'ignore.
#define RAY_EPS_DENOM 1e-6f
// Marge pour ne pas retoucher le portail que l'on vient de franchir.
#define RAY_EPS_T     1e-5f
// Le rayon va de a (t = 0) a b (t = 1) ; petite marge pour les arrondis.
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

// Ecrit le bit (src -> dst) dans portalvis. Les mots sont de 32 bits et plusieurs
// threads peuvent viser le meme mot : atomicOr evite de perdre un bit.
__device__ __forceinline__ void SetPortalvisBit( unsigned int *words, int src, int dst, int portalbytes )
{
	unsigned long long byte_index = (unsigned long long)src * (unsigned long long)portalbytes + (unsigned long long)( dst >> 3 );
	unsigned int shift = (unsigned int)( byte_index & 3 ) * 8u + (unsigned int)( dst & 7 );
	atomicOr( words + ( byte_index >> 2 ), 1u << shift );
}

__device__ __forceinline__ int TestBit( const unsigned char *row, int bit )
{
	return ( row[bit >> 3] >> ( bit & 7 ) ) & 1;
}

// Un point est dans le winding s'il est du meme cote de toutes les aretes
// (tous les produits mixtes de meme signe). Sur une arete : on refuse.
__device__ int PointInWinding( const float *winding_pts, const GPUPortal *gp, const float x[3] )
{
	int pos = 0, neg = 0;
	int n = gp->winding_count;
	const float *w = winding_pts + (size_t)gp->winding_start * 3;
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

// Point d'echantillon numero s d'un portail (nv = sommets utilises par anneau).
//   s = 0              : le centre
//   s = 1 .. nv        : anneau exterieur (90 % du centre vers le sommet)
//   s = nv+1 .. 2*nv   : anneau interieur (50 %)
// Total par portail : 1 + 2 * nv points.
__device__ void PortalInsetSample( const float *winding_pts, const GPUPortal *gp,
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
	const float *w = winding_pts + (size_t)( gp->winding_start + vi ) * 3;
	out[0] = c[0] + t * ( w[0] - c[0] );
	out[1] = c[1] + t * ( w[1] - c[1] );
	out[2] = c[2] + t * ( w[2] - c[2] );
}

// Suit un rayon a -> b de portail en portail, en partant du leaf du portail src.
// A chaque etape : parmi les portails sortants du leaf courant, on garde le plus
// proche devant le rayon dont le point d'impact est dans le winding, on marque ce
// portail visible depuis src, puis on passe dans le leaf suivant.
// Retourne 1 si dst est atteint, 0 sinon. *steps_out = portails traverses.
__device__ int WalkStabbingLine(
	const GPUPortal *portals, const GPULeaf *leafs, int num_leafs,
	const int *leaf_portals, const float *winding_pts,
	unsigned int *portalvis_words, int portalbytes, int max_steps,
	int src, int dst, const float a[3], const float b[3], int *steps_out )
{
	float dir0 = b[0] - a[0], dir1 = b[1] - a[1], dir2 = b[2] - a[2];
	int leaf = portals[src].leaf;
	float tcur = 0.0f;
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
			if ( t <= tcur + RAY_EPS_T || t > RAY_T_MAX || t >= best_t )
				continue;
			float x[3] = { a[0] + t*dir0, a[1] + t*dir1, a[2] + t*dir2 };
			if ( !PointInWinding( winding_pts, gq, x ) )
				continue;
			best_t = t;
			best_q = q;
		}
		if ( best_q < 0 )
		{
			*steps_out = step;
			return 0;
		}
		SetPortalvisBit( portalvis_words, src, best_q, portalbytes );
		if ( best_q == dst )
		{
			*steps_out = step;
			return 1;
		}
		leaf = portals[best_q].leaf;
		tcur = best_t;
	}
	*steps_out = max_steps;
	return 0;
}

// Statistiques facultatives (diagnostic). Si stats == NULL, rien n'est compte
// et le resultat est strictement le meme.
//   stats[0] paires testees          stats[1] paires prouvees visibles
//   stats[2] rayons tires            stats[3] rayons tires par des paires invisibles
//   stats[4 + d] rayons morts apres d portails traverses (d >= 15 : dernier bin)
#define STAT_BINS 16

// 1 thread = 1 paire (src, dst). `start` = indice de la premiere paire de la tranche.
extern "C" __global__ void RaycastStabKernel(
	const GPUPortal *portals, int num_portals,
	const GPULeaf *leafs, int num_leafs,
	const int *leaf_portals,
	const float *winding_pts,
	const unsigned char *portalflood,
	unsigned int *portalvis_words,
	int portalbytes, int samples_per_portal, int max_steps,
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

	const GPUPortal *sp = portals + src;
	const GPUPortal *dp = portals + dst;
	if ( sp->winding_count < 3 || dp->winding_count < 3 )
		return;

	int nvs = sp->winding_count < samples_per_portal ? sp->winding_count : samples_per_portal;
	int nvd = dp->winding_count < samples_per_portal ? dp->winding_count : samples_per_portal;
	float cs[3], cd[3];
	PortalCentroid( winding_pts, sp, cs );
	PortalCentroid( winding_pts, dp, cd );

	int ss = 1 + RAY_NUM_RINGS * nvs;
	int sd = 1 + RAY_NUM_RINGS * nvd;
	unsigned long long rays = 0;
	unsigned long long hist[STAT_BINS];
	for ( int h = 0; h < STAT_BINS; ++h )
		hist[h] = 0;
	int visible = 0;
	for ( int si = 0; si < ss && !visible; ++si )
	{
		float a[3];
		PortalInsetSample( winding_pts, sp, cs, nvs, si, a );
		for ( int di = 0; di < sd; ++di )
		{
			float b[3];
			PortalInsetSample( winding_pts, dp, cd, nvd, di, b );
			int steps = 0;
			++rays;
			if ( WalkStabbingLine( portals, leafs, num_leafs, leaf_portals, winding_pts,
					portalvis_words, portalbytes, max_steps, src, dst, a, b, &steps ) )
			{
				visible = 1;
				break;
			}
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
	}
}
