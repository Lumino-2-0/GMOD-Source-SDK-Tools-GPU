// vrad_gpu_kernel.cu
//
// Noyau CUDA de visibilite point-a-point pour VRAD (phase des transferts,
// CTransferMaker / BuildVisLeafs).
//
// Reproduit, pour UN rayon, RayTracingEnvironment::Trace4Rays (raytrace.cpp) :
// meme k-d tree, memes tests d'intersection, memes epsilons (1e-10).
// Resultat = 1 si le rayon est LIBRE (transfert cree), 0 sinon, soit le test de
// CTransferMaker::Finish : HitID == -1 || HitDistance >= ray_length.
//
// Difference connue : le CPU calcule 1/x avec rcpps + Newton ; ici division IEEE.
//
// Compile en PTX par vrad_build_ptx.ps1 (vrad_gpu_kernel_ptx.h).

struct GpuTri
{
	float nx, ny, nz, d;
	float e[6];
	int cs0, cs1;
	int flags;
	int id;
};

#define KD_LEAF 3
#define STACK_SIZE 48
#define EPS 1.0e-10f

__device__ __forceinline__ float RecipSat( float x )
{
	if ( x == 0.0f )
		x = EPS;
	return 1.0f / x;
}

extern "C" __global__ void TraceVisKernel(
	const int *nodeChildren, const float *nodeSplit,
	const int *triIndexList, const GpuTri *tris,
	const float *bounds,
	const float *rays, unsigned char *out, int numRays, int mode, int skipId )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= numRays )
		return;

	float o[3], d[3];
	o[0] = rays[i * 6 + 0]; o[1] = rays[i * 6 + 1]; o[2] = rays[i * 6 + 2];
	d[0] = rays[i * 6 + 3] - o[0];
	d[1] = rays[i * 6 + 4] - o[1];
	d[2] = rays[i * 6 + 5] - o[2];

	float len = sqrtf( d[0] * d[0] + d[1] * d[1] + d[2] * d[2] );
	float inv = RecipSat( len );
	d[0] *= inv; d[1] *= inv; d[2] *= inv;

	float invd[3];
	int frontIdx[3], backIdx[3];
	for ( int c = 0; c < 3; ++c )
	{
		invd[c] = RecipSat( d[c] );
		if ( d[c] < 0.0f ) { backIdx[c] = 0; frontIdx[c] = 1; }
		else               { backIdx[c] = 1; frontIdx[c] = 0; }
	}

	float tmin = 0.0f, tmax = len;
	for ( int c = 0; c < 3; ++c )
	{
		float t0 = ( bounds[c] - o[c] ) * invd[c];
		float t1 = ( bounds[3 + c] - o[c] ) * invd[c];
		tmin = fmaxf( tmin, fminf( t0, t1 ) );
		tmax = fminf( tmax, fmaxf( t0, t1 ) );
	}
	if ( !( tmin <= tmax ) )
	{
		out[i] = 1;
		return;
	}

	float bestT = 1.0e23f;
	int bestId = 0;
	int transparentSeen = 0;
	int   stackNode[STACK_SIZE];
	float stackMin[STACK_SIZE];
	float stackMax[STACK_SIZE];
	int sp = STACK_SIZE;

	int cur = 0;
	while ( true )
	{
		int ch;
		while ( ( ( ch = nodeChildren[cur] ) & 3 ) != KD_LEAF )
		{
			int axis = ch & 3;
			int front = ch >> 2;
			float dist = ( nodeSplit[cur] - o[axis] ) * invd[axis];
			bool active = tmin <= tmax;
			if ( !( active && dist >= tmin ) )
			{
				cur = front + backIdx[axis];
				tmin = fmaxf( tmin, dist );
			}
			else if ( !( active && dist <= tmax ) )
			{
				cur = front + frontIdx[axis];
				tmax = fminf( tmax, dist );
			}
			else
			{
				--sp;
				stackNode[sp] = front + backIdx[axis];
				stackMin[sp] = fmaxf( tmin, dist );
				stackMax[sp] = tmax;
				cur = front + frontIdx[axis];
				tmax = fminf( tmax, dist );
			}
		}

		int ntris = __float_as_int( nodeSplit[cur] );
		int start = ch >> 2;
		for ( int k = 0; k < ntris; ++k )
		{
			const GpuTri &t = tris[triIndexList[start + k]];
			if ( t.id == skipId )
				continue;
			float ddotn = d[0] * t.nx + d[1] * t.ny + d[2] * t.nz;
			if ( !( ddotn > EPS || ddotn < -EPS ) )
				continue;
			float num = t.d - ( o[0] * t.nx + o[1] * t.ny + o[2] * t.nz );
			float isect = num / ddotn;
			if ( !( isect > EPS ) || !( isect < len ) || !( isect < bestT ) )
				continue;

			float c1 = o[t.cs0] + isect * d[t.cs0];
			float c2 = o[t.cs1] + isect * d[t.cs1];
			float b0 = t.e[0] * c1;
			b0 = b0 + t.e[1] * c2;
			b0 = b0 + t.e[2];
			if ( !( b0 >= EPS ) )
				continue;
			float b1 = t.e[3] * c1;
			b1 = b1 + t.e[4] * c2;
			b1 = b1 + t.e[5];
			if ( !( b1 >= EPS ) )
				continue;
			if ( !( ( b1 + b0 ) <= 1.0f ) )
				continue;

			if ( mode == 2 && ( t.flags & 1 ) )
			{
				transparentSeen = 1;
				continue;
			}
			if ( mode != 1 )
			{
				out[i] = 0;
				return;
			}
			bestT = isect;
			bestId = t.id;
		}
		if ( mode == 1 && bestT < tmax )
			break;

		if ( sp == STACK_SIZE )
			break;
		cur = stackNode[sp];
		tmin = stackMin[sp];
		tmax = stackMax[sp];
		++sp;
	}
	// mode 1 : libre si aucun impact, ou si l'impact le plus proche est du ciel (TRACE_ID_SKY)
	if ( mode == 1 && bestT < 1.0e22f && !( bestId & 0x01000000 ) )
	{
		out[i] = 0;
		return;
	}
	out[i] = ( mode == 2 && transparentSeen ) ? 2 : 1;
}

// ============================================================================
// Eclairage indirect des props statiques (ComputeIndirectLightingAtPoint).
//
// Reproduit CLightSurface::FindIntersection (vraddetailprops.cpp) :
// EnumerateNodesAlongRay_R (bsplib.cpp) en version ITERATIVE (pile explicite),
// EnumerateNode / EnumerateLeaf, TestPointAgainstSurface, ciel sur noeud,
// puis echantillonnage du lightmap, reflectivite et attenuation 1/(1+d^2).
//
// Difference volontaire : le CLightSurface CPU conserve m_HitFrac d'un rayon au
// suivant (bogue) ; ici chaque rayon part de 1.0.
// Les rayons qui traversent une feuille contenant un displacement sont rendus
// au CPU (w = 2).
// ============================================================================

struct GpuBspNode { float nx, ny, nz, dist; int type; int ch0, ch1; int firstface, numfaces; };
struct GpuBspLeaf { int firstleafface, numleaffaces, hasDisp, pad; };
struct GpuBspFace
{
	float pn[4];
	float lv[8];
	int mins[2];
	int size[2];
	int flags;
	int skip;
	int onNode;
	int lightofs;
	int style0;
	float refl[3];
	int skyStart;
	int skyCount;
};
struct IndirectData
{
	const GpuBspNode *nodes;
	const GpuBspLeaf *leaves;
	const int *leafFaces;
	const GpuBspFace *faces;
	const float *skyPts;
	const unsigned char *light;
	const float *pow2;
	int lightBytes;
	int pad;
};

#define IND_STACK 192
#define BSP_EPS 0.03125f
#define SURF_SKY_FLAG 0x0004
#define SURF_NOLIGHT_FLAG 0x0400

__device__ __forceinline__ bool IndTestPoint( const GpuBspFace &f, float x, float y, float z, float *ds, float *dt )
{
	if ( f.flags & SURF_NOLIGHT_FLAG )
		return false;
	float s = x * f.lv[0] + y * f.lv[1] + z * f.lv[2] + f.lv[3];
	float t = x * f.lv[4] + y * f.lv[5] + z * f.lv[6] + f.lv[7];
	if ( s < (float)f.mins[0] || t < (float)f.mins[1] )
		return false;
	float a = s - (float)f.mins[0];
	float b = t - (float)f.mins[1];
	if ( a > (float)f.size[0] || b > (float)f.size[1] )
		return false;
	*ds = a;
	*dt = b;
	return true;
}

__device__ __forceinline__ void IndNormalize( float *v )
{
	float r = sqrtf( v[0] * v[0] + v[1] * v[1] + v[2] * v[2] );
	if ( r > 0.0f )
	{
		float inv = 1.0f / r;
		v[0] *= inv; v[1] *= inv; v[2] *= inv;
	}
}

// PointInWinding (polylib.cpp)
__device__ bool IndPointInSky( const IndirectData &D, const GpuBspFace &f, float px, float py, float pz )
{
	int n = f.skyCount;
	if ( n < 3 )
		return false;
	const float *p = D.skyPts + f.skyStart * 3;
	float t[3] = { px - p[0], py - p[1], pz - p[2] };
	float e[3] = { p[3] - p[0], p[4] - p[1], p[5] - p[2] };
	float tc[3] = { e[1] * t[2] - e[2] * t[1], e[2] * t[0] - e[0] * t[2], e[0] * t[1] - e[1] * t[0] };
	IndNormalize( tc );
	for ( int k = 1; k < n; ++k )
	{
		const float *a = p + k * 3;
		const float *b = p + ( ( k + 1 ) % n ) * 3;
		float tt[3] = { px - a[0], py - a[1], pz - a[2] };
		float ee[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
		float c[3] = { ee[1] * tt[2] - ee[2] * tt[1], ee[2] * tt[0] - ee[0] * tt[2], ee[0] * tt[1] - ee[1] * tt[0] };
		IndNormalize( c );
		if ( c[0] * tc[0] + c[1] * tc[1] + c[2] * tc[2] < 0.0f )
			return false;
	}
	return true;
}

// rays : 6 floats (origine, delta). out : 4 floats (r,g,b, w) ; w = 0 rien, 1 contribution, 2 repli CPU.
extern "C" __global__ void IndirectKernel( IndirectData D, const float *rays, float *out, int numRays )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= numRays )
		return;

	float o[3], d[3];
	o[0] = rays[i * 6 + 0]; o[1] = rays[i * 6 + 1]; o[2] = rays[i * 6 + 2];
	d[0] = rays[i * 6 + 3]; d[1] = rays[i * 6 + 4]; d[2] = rays[i * 6 + 5];

	int hitFace = -1;
	float hitFrac = 1.0f, hds = 0.0f, hdt = 0.0f;
	int fallback = 0;
	bool done = false;

	int stN[IND_STACK];
	float stA[IND_STACK], stB[IND_STACK];
	unsigned char stV[IND_STACK];
	int sp = 0;
	stN[0] = 0; stA[0] = 0.0f; stB[0] = 1.0f; stV[0] = 0; sp = 1;

	while ( sp > 0 && !done && !fallback )
	{
		--sp;
		int n = stN[sp];
		float start = stA[sp], end = stB[sp];

		if ( stV[sp] )
		{
			// EnumerateNode( n, f = start )
			float f = start;
			float px = o[0] + f * d[0], py = o[1] + f * d[1], pz = o[2] + f * d[2];
			const GpuBspNode &nd = D.nodes[n];
			int skySurf = -1;
			for ( int k = 0; k < nd.numfaces; ++k )
			{
				int fi = nd.firstface + k;
				const GpuBspFace &fc = D.faces[fi];
				if ( !fc.onNode || fc.skip )
					continue;
				if ( fc.flags & SURF_SKY_FLAG )
				{
					if ( IndPointInSky( D, fc, px, py, pz ) )
						skySurf = fi;
					continue;
				}
				float a, b;
				if ( IndTestPoint( fc, px, py, pz, &a, &b ) )
				{
					hitFace = fi; hitFrac = f; hds = a; hdt = b;
					done = true;
					break;
				}
			}
			if ( !done && skySurf >= 0 )
			{
				hitFace = skySurf;
				done = true;
			}
			continue;
		}

		while ( n >= 0 )
		{
			const GpuBspNode &nd = D.nodes[n];
			float sdn, ddn;
			if ( nd.type <= 2 )
			{
				sdn = o[nd.type];
				ddn = d[nd.type];
			}
			else
			{
				sdn = o[0] * nd.nx + o[1] * nd.ny + o[2] * nd.nz;
				ddn = d[0] * nd.nx + d[1] * nd.ny + d[2] * nd.nz;
			}
			float front = sdn + start * ddn - nd.dist;
			float back = sdn + end * ddn - nd.dist;
			if ( front <= -BSP_EPS && back <= -BSP_EPS )
				n = nd.ch1;
			else if ( front >= BSP_EPS && back >= BSP_EPS )
				n = nd.ch0;
			else
			{
				int side = front < 0.0f;
				float sf;
				if ( ddn == 0.0f )
					sf = 1.0f;
				else
				{
					sf = ( nd.dist - sdn ) / ddn;
					if ( sf < 0.0f ) sf = 0.0f;
					else if ( sf > 1.0f ) sf = 1.0f;
				}
				if ( sp + 2 > IND_STACK )
				{
					fallback = 1;
					break;
				}
				stN[sp] = side ? nd.ch0 : nd.ch1; stA[sp] = sf; stB[sp] = end; stV[sp] = 0; ++sp;
				stN[sp] = n; stA[sp] = sf; stB[sp] = 0.0f; stV[sp] = 1; ++sp;
				n = side ? nd.ch1 : nd.ch0;
				end = sf;
			}
		}
		if ( fallback )
			break;

		// EnumerateLeaf
		const GpuBspLeaf &lf = D.leaves[-n - 1];
		if ( lf.hasDisp )
		{
			fallback = 1;
			break;
		}
		bool hit = false;
		for ( int k = 0; k < lf.numleaffaces; ++k )
		{
			int fi = D.leafFaces[lf.firstleafface + k];
			const GpuBspFace &fc = D.faces[fi];
			if ( fc.skip || fc.onNode )
				continue;
			if ( fc.pn[0] * d[0] + fc.pn[1] * d[1] + fc.pn[2] * d[2] > 0.0f )
				continue;
			float sdn = o[0] * fc.pn[0] + o[1] * fc.pn[1] + o[2] * fc.pn[2];
			float ddn = d[0] * fc.pn[0] + d[1] * fc.pn[1] + d[2] * fc.pn[2];
			float front = sdn + start * ddn - fc.pn[3];
			float back = sdn + end * ddn - fc.pn[3];
			bool side = front < 0.0f;
			if ( ( back < 0.0f ) == side )
				continue;
			float f = front / ( front - back );
			float mid = start * ( 1.0f - f ) + end * f;
			if ( mid >= hitFrac )
				continue;
			float px = o[0] + mid * d[0], py = o[1] + mid * d[1], pz = o[2] + mid * d[2];
			float a, b;
			if ( IndTestPoint( fc, px, py, pz, &a, &b ) )
			{
				hitFrac = mid; hitFace = fi; hds = a; hdt = b;
				hit = true;
			}
		}
		if ( hit )
			done = true;
	}

	float *res = out + i * 4;
	res[0] = res[1] = res[2] = 0.0f;
	res[3] = 0.0f;
	if ( fallback )
	{
		res[3] = 2.0f;
		return;
	}
	if ( hitFace < 0 )
		return;

	const GpuBspFace &fc = D.faces[hitFace];
	if ( fc.flags & SURF_SKY_FLAG )
		return;
	if ( fc.style0 == 255 || fc.lightofs < 0 )
		return;

	int smax = fc.size[0] + 1;
	int tmax = fc.size[1] + 1;
	int ds = (int)hds, dt = (int)hdt;
	ds = ds < 0 ? 0 : ( ds > smax - 1 ? smax - 1 : ds );
	dt = dt < 0 ? 0 : ( dt > tmax - 1 ? tmax - 1 : dt );
	long long off = (long long)fc.lightofs + 4LL * ( (long long)dt * smax + ds );
	if ( off + 4 > (long long)D.lightBytes )
	{
		res[3] = 2.0f;
		return;
	}
	const unsigned char *px = D.light + off;
	float scale = D.pow2[(int)(signed char)px[3] + 128];
	float cr = 255.0f * ( (float)px[0] * scale );
	float cg = 255.0f * ( (float)px[1] * scale );
	float cb = 255.0f * ( (float)px[2] * scale );

	float vx = d[0] * hitFrac * 0.0078125f;
	float vy = d[1] * hitFrac * 0.0078125f;
	float vz = d[2] * hitFrac * 0.0078125f;
	float lsq = vx * vx + vy * vy + vz * vz;
	float inv = 1.0f / ( 1.0f + lsq );
	res[0] = cr * ( inv * fc.refl[0] );
	res[1] = cg * ( inv * fc.refl[1] );
	res[2] = cb * ( inv * fc.refl[2] );
	res[3] = 1.0f;
}

