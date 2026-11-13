//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose: 
//
// $NoKeywords: $
//
//=============================================================================//
// vis.c

#include <windows.h>
#include "vis.h"
#include "threads.h"
#include "stdlib.h"
#include "flow_gpu.h"
#include "pacifier.h"
#include "tier1/strtools.h"
#include "collisionutils.h"
#include "tier0/icommandline.h"
#include "ilaunchabledll.h"
#include "tools_minidump.h"
#include "loadcmdline.h"
#include "byteswap.h"


int			g_numportals;
int			portalclusters;

char		inbase[32];

portal_t	*portals;
leaf_t		*leafs;

int			c_portaltest, c_portalpass, c_portalcheck;

byte		*uncompressedvis;

byte		*vismap, *vismap_p, *vismap_end;	// past visfile
int			originalvismapsize;

int			leafbytes;				// (portalclusters+63)>>3
int			leaflongs;

int			portalbytes, portallongs;

bool		fastvis;
bool		nosort;

int			totalvis;

portal_t	*sorted_portals[MAX_MAP_PORTALS*2];

bool		g_bUseRadius = false;
double		g_VisRadius = 4096.0f * 4096.0f;

bool		g_bLowPriority = false;
// Precision du raycast GPU : g_PassSamples[k] = nombre max de sommets de winding
// utilises par anneau a la passe k. Points par portail = 1 + 2 * min(n, sommets).
//   -RaycastType 0 : n = 4 (defaut, 9 points, jusqu'a 81 rayons par paire)
//   -RaycastType 1 : n = 5 (11 points, environ 25 % d'echantillons en plus)
//   -Ray <n>       : n choisi librement
//   -Pass / -final : plusieurs passes, les suivantes ne reprennent que les paires
//                    candidates pas encore prouvees visibles (voir flow_gpu_host.cpp)
static const int g_RaycastTypeSamples[2] = { 4, 5 };
static int		g_PassSamples[16] = { 4 };
static int		g_NumPasses = 1;

bool		g_bForceCpuFlow = false;

//=============================================================================

void PlaneFromWinding (winding_t *w, plane_t *plane)
{
	Vector		v1, v2;

// calc plane
	VectorSubtract (w->points[2], w->points[1], v1);
	VectorSubtract (w->points[0], w->points[1], v2);
	CrossProduct (v2, v1, plane->normal);
	VectorNormalize (plane->normal);
	plane->dist = DotProduct (w->points[0], plane->normal);
}


/*
==================
NewWinding
==================
*/
winding_t *NewWinding (int points)
{
	winding_t	*w;
	int			size;
	
	if (points > MAX_POINTS_ON_WINDING)
		Error ("NewWinding: %i points, max %d", points, MAX_POINTS_ON_WINDING);
	
	size = (int)(&((winding_t *)0)->points[points]);
	w = (winding_t*)malloc (size);
	memset (w, 0, size);
	
	return w;
}

void pw(winding_t *w)
{
	int		i;
	for (i=0 ; i<w->numpoints ; i++)
		Msg ("(%5.1f, %5.1f, %5.1f)\n",w->points[i][0], w->points[i][1],w->points[i][2]);
}

void prl(leaf_t *l)
{
	int			i;
	portal_t	*p;
	plane_t		pl;
	
	int count = l->portals.Count();
	for (i=0 ; i<count ; i++)
	{
		p = l->portals[i];
		pl = p->plane;
		Msg ("portal %4i to leaf %4i : %7.1f : (%4.1f, %4.1f, %4.1f)\n",(int)(p-portals),p->leaf,pl.dist, pl.normal[0], pl.normal[1], pl.normal[2]);
	}
}


//=============================================================================

/*
=============
SortPortals

Sorts the portals from the least complex, so the later ones can reuse
the earlier information.
=============
*/
int PComp (const void *a, const void *b)
{
	if ( (*(portal_t **)a)->nummightsee == (*(portal_t **)b)->nummightsee)
		return 0;
	if ( (*(portal_t **)a)->nummightsee < (*(portal_t **)b)->nummightsee)
		return -1;

	return 1;
}

void BuildTracePortals( int clusterStart )
{
	leaf_t *leaf = &leafs[g_TraceClusterStart];
	g_numportals = leaf->portals.Count();
	for ( int i = 0; i < g_numportals; i++ )
	{
		sorted_portals[i] = leaf->portals[i];
	}
}

void SortPortals (void)
{
	int		i;
	
	for (i=0 ; i<g_numportals*2 ; i++)
		sorted_portals[i] = &portals[i];

	if (nosort)
		return;
	qsort (sorted_portals, g_numportals*2, sizeof(sorted_portals[0]), PComp);
}


/*
==============
LeafVectorFromPortalVector
==============
*/
int LeafVectorFromPortalVector (byte *portalbits, byte *leafbits)
{
	int			i;
	portal_t	*p;
	int			c_leafs;


	memset (leafbits, 0, leafbytes);

	for (i=0 ; i<g_numportals*2 ; i++)
	{
		if ( CheckBit( portalbits, i ) )
		{
			p = portals+i;
			SetBit( leafbits, p->leaf );
		}
	}

	c_leafs = CountBits (leafbits, portalclusters);

	return c_leafs;
}


/*
===============
ClusterMerge

Merges the portal visibility for a leaf
===============
*/
void ClusterMerge (int clusternum)
{
	leaf_t		*leaf;
//	byte		portalvector[MAX_PORTALS/8];
	byte		portalvector[MAX_PORTALS/4];      // 4 because portal bytes is * 2
	byte		uncompressed[MAX_MAP_LEAFS/8];
	int			i, j;
	int			numvis;
	portal_t	*p;
	int			pnum;

	// OR together all the portalvis bits

	memset (portalvector, 0, portalbytes);
	leaf = &leafs[clusternum];
	for (i=0 ; i < leaf->portals.Count(); i++)
	{
		p = leaf->portals[i];
		if (p->status != stat_done)
			Error ("portal not done %d %p %p\n", i, p, portals);
		for (j=0 ; j<portallongs ; j++)
			((long *)portalvector)[j] |= ((long *)p->portalvis)[j];
		pnum = p - portals;
		SetBit( portalvector, pnum );
	}

	// convert portal bits to leaf bits
	numvis = LeafVectorFromPortalVector (portalvector, uncompressed);

#if 0
	// func_viscluster makes this happen all the time because it allows a non-convex set of portals
	// My analysis says this is ok, but it does make this check for errors in vis kind of useless
	if ( CheckBit( uncompressed, clusternum ) )
		Warning("WARNING: Cluster portals saw into cluster\n");
#endif
		
	SetBit( uncompressed, clusternum );
	numvis++;		// count the leaf itself

	// save uncompressed for PHS calculation
	memcpy (uncompressedvis + clusternum*leafbytes, uncompressed, leafbytes);

	qprintf ("cluster %4i : %4i visible\n", clusternum, numvis);
	totalvis += numvis;
}

static int CompressAndCrosscheckClusterVis( int clusternum )
{
	int		optimized = 0;
	byte	compressed[MAX_MAP_LEAFS/8];
//
// compress the bit string
//
	byte *uncompressed = uncompressedvis + clusternum*leafbytes;
	for ( int i = 0; i < portalclusters; i++ )
	{
		if ( i == clusternum )
			continue;

		if ( CheckBit( uncompressed, i ) )
		{
			byte *other = uncompressedvis + i*leafbytes;
			if ( !CheckBit( other, clusternum ) )
			{
				ClearBit( uncompressed, i );
				optimized++;
			}
		}
	}
	int numbytes = CompressVis( uncompressed, compressed );

	byte *dest = vismap_p;
	vismap_p += numbytes;
	
	if (vismap_p > vismap_end)
		Error ("Vismap expansion overflow");

	dvis->bitofs[clusternum][DVIS_PVS] = dest-vismap;

	memcpy( dest, compressed, numbytes );

	// check vis data
	DecompressVis( vismap + dvis->bitofs[clusternum][DVIS_PVS], compressed );

	return optimized;
}


/*
==================
CalcPortalVis
==================
*/
void CalcPortalVis (void)
{
	int		i;

	// fastvis just uses mightsee for a very loose bound
	if( fastvis )
	{
		for (i=0 ; i<g_numportals*2 ; i++)
		{
			portals[i].portalvis = portals[i].portalflood;
			portals[i].status = stat_done;
		}
		return;
	}

// Chemin par defaut : raycast sur GPU (CUDA sur NVIDIA, sinon OpenCL).
// On retombe sur le flow CPU recursif d'origine avec -cpu, s'il n'y a pas de
// GPU utilisable, ou si le GPU renvoie une erreur.
if ( !g_bForceCpuFlow )
{
bool bDone = false;
if ( GPU_Init() )
bDone = GPU_RunPortalFlowRaycast( g_numportals * 2, g_PassSamples, g_NumPasses );
GPU_Shutdown();
if ( bDone )
return;
Msg( "[VVIS GPU] GPU indisponible - flow CPU recursif d'origine.\n" );
}

	RunThreadsOnIndividual (g_numportals*2, true, PortalFlow);

}


void CalcVisTrace (void)
{
    RunThreadsOnIndividual (g_numportals*2, true, BasePortalVis);
	BuildTracePortals( g_TraceClusterStart );
	// NOTE: We only schedule the one-way portals out of the start cluster here
	// so don't run g_numportals*2 in this case


	RunThreadsOnIndividual (g_numportals, true, PortalFlow);
}

/*
==================
CalcVis
==================
*/
void CalcVis (void)
{
	int		i;
	RunThreadsOnIndividual (g_numportals*2, true, BasePortalVis);
	

	SortPortals ();

	CalcPortalVis ();

	//
	// assemble the leaf vis lists by oring the portal lists
	//
	for ( i = 0; i < portalclusters; i++ )
	{
		ClusterMerge( i );
	}


	int count = 0;
	// Now crosscheck each leaf's vis and compress
	for ( i = 0; i < portalclusters; i++ )
	{
		count += CompressAndCrosscheckClusterVis( i );
	}

		
	Msg ("Optimized: %d visible clusters (%.2f%%)\n", count, count*100.0/totalvis);
	Msg ("Total clusters visible: %i\n", totalvis);
	Msg ("Average clusters visible: %i\n", totalvis / portalclusters);
}


void SetPortalSphere (portal_t *p)
{
	int		i;
	Vector	total, dist;
	winding_t	*w;
	float	r, bestr;

	w = p->winding;
	VectorCopy (vec3_origin, total);
	for (i=0 ; i<w->numpoints ; i++)
	{
		VectorAdd (total, w->points[i], total);
	}
	
	for (i=0 ; i<3 ; i++)
		total[i] /= w->numpoints;

	bestr = 0;		
	for (i=0 ; i<w->numpoints ; i++)
	{
		VectorSubtract (w->points[i], total, dist);
		r = VectorLength (dist);
		if (r > bestr)
			bestr = r;
	}
	VectorCopy (total, p->origin);
	p->radius = bestr;
}

/*
============
LoadPortals
============
*/
void LoadPortals (char *name)
{
	int			i, j;
	portal_t	*p;
	leaf_t		*l;
	char		magic[80];
	int			numpoints;
	winding_t	*w;
	int			leafnums[2];
	plane_t		plane;

	FILE *f;

	// Open the portal file.
	f = fopen( name, "r" );
	

	if ( !f )
		Error ("LoadPortals: couldn't read %s\n",name);

	if (fscanf (f,"%79s\n%i\n%i\n",magic, &portalclusters, &g_numportals) != 3)
		Error ("LoadPortals %s: failed to read header", name);
	if (stricmp(magic,PORTALFILE))
		Error ("LoadPortals %s: not a portal file", name);

	Msg ("%4i portalclusters\n", portalclusters);
	Msg ("%4i numportals\n", g_numportals);

	if (g_numportals * 2 >= MAX_PORTALS)
	{
		Error("The map overflows the max portal count (%d of max %d)!\n", g_numportals, MAX_PORTALS / 2 );
	}

	// these counts should take advantage of 64 bit systems automatically
	leafbytes = ((portalclusters+63)&~63)>>3;
	leaflongs = leafbytes/sizeof(long);
	
	portalbytes = ((g_numportals*2+63)&~63)>>3;
	portallongs = portalbytes/sizeof(long);

// each file portal is split into two memory portals
	portals = (portal_t*)malloc(2*g_numportals*sizeof(portal_t));
	memset (portals, 0, 2*g_numportals*sizeof(portal_t));
	
	leafs = (leaf_t*)malloc(portalclusters*sizeof(leaf_t));
	memset (leafs, 0, portalclusters*sizeof(leaf_t));

	originalvismapsize = portalclusters*leafbytes;
	uncompressedvis = (byte*)malloc(originalvismapsize);

	vismap = vismap_p = dvisdata;
	dvis->numclusters = portalclusters;
	vismap_p = (byte *)&dvis->bitofs[portalclusters];

	vismap_end = vismap + MAX_MAP_VISIBILITY;
		
	for (i=0, p=portals ; i<g_numportals ; i++)
	{
		if (fscanf (f, "%i %i %i ", &numpoints, &leafnums[0], &leafnums[1])
			!= 3)
			Error ("LoadPortals: reading portal %i", i);
		if (numpoints > MAX_POINTS_ON_WINDING)
			Error ("LoadPortals: portal %i has too many points", i);
		if ( (unsigned)leafnums[0] > portalclusters
		|| (unsigned)leafnums[1] > portalclusters)
			Error ("LoadPortals: reading portal %i", i);
		
		w = p->winding = NewWinding (numpoints);
		w->original = true;
		w->numpoints = numpoints;
		
		for (j=0 ; j<numpoints ; j++)
		{
			double	v[3];
			int		k;

			// scanf into double, then assign to vec_t
			// so we don't care what size vec_t is
			if (fscanf (f, "(%lf %lf %lf ) "
			, &v[0], &v[1], &v[2]) != 3)
				Error ("LoadPortals: reading portal %i", i);
			for (k=0 ; k<3 ; k++)
				w->points[j][k] = v[k];
		}
		fscanf (f, "\n");
		
	// calc plane
		PlaneFromWinding (w, &plane);

	// create forward portal
		l = &leafs[leafnums[0]];
		l->portals.AddToTail(p);
		
		p->winding = w;
		VectorSubtract (vec3_origin, plane.normal, p->plane.normal);
		p->plane.dist = -plane.dist;
		p->leaf = leafnums[1];
		SetPortalSphere (p);
		p++;
		
	// create backwards portal
		l = &leafs[leafnums[1]];
		l->portals.AddToTail(p);
		
		p->winding = NewWinding(w->numpoints);
		p->winding->numpoints = w->numpoints;
		for (j=0 ; j<w->numpoints ; j++)
		{
			VectorCopy (w->points[w->numpoints-1-j], p->winding->points[j]);
		}

		p->plane = plane;
		p->leaf = leafnums[0];
		SetPortalSphere (p);
		p++;

	}
	
	fclose (f);
}


/*
================
CalcPAS

Calculate the PAS (Potentially Audible Set)
by ORing together all the PVS visible from a leaf
================
*/
void CalcPAS (void)
{
	int		i, j, k, l, index;
	int		bitbyte;
	long	*dest, *src;
	byte	*scan;
	int		count;
	byte	uncompressed[MAX_MAP_LEAFS/8];
	byte	compressed[MAX_MAP_LEAFS/8];

	Msg ("Building PAS...\n");

	count = 0;
	for (i=0 ; i<portalclusters ; i++)
	{
		scan = uncompressedvis + i*leafbytes;
		memcpy (uncompressed, scan, leafbytes);
		for (j=0 ; j<leafbytes ; j++)
		{
			bitbyte = scan[j];
			if (!bitbyte)
				continue;
			for (k=0 ; k<8 ; k++)
			{
				if (! (bitbyte & (1<<k)) )
					continue;
				// OR this pvs row into the phs
				index = ((j<<3)+k);
				if (index >= portalclusters)
					Error ("Bad bit in PVS");	// pad bits should be 0
				src = (long *)(uncompressedvis + index*leafbytes);
				dest = (long *)uncompressed;
				for (l=0 ; l<leaflongs ; l++)
					((long *)uncompressed)[l] |= src[l];
			}
		}
		for (j=0 ; j<portalclusters ; j++)
		{
			if ( CheckBit( uncompressed, j ) )
			{
				count++;
			}
		}

	//
	// compress the bit string
	//
		j = CompressVis (uncompressed, compressed);

		dest = (long *)vismap_p;
		vismap_p += j;
		
		if (vismap_p > vismap_end)
			Error ("Vismap expansion overflow");

		dvis->bitofs[i][DVIS_PAS] = (byte *)dest-vismap;

		memcpy (dest, compressed, j);	
	}

	Msg ("Average clusters audible: %i\n", count/portalclusters);
}



static void GetBoundsForFace( int faceID, Vector &faceMin, Vector &faceMax )
{
	ClearBounds( faceMin, faceMax );
	dface_t *pFace = &dfaces[faceID];
	int i;
	for( i = pFace->firstedge; i < pFace->firstedge + pFace->numedges; i++ )
	{
		int edgeID = dsurfedges[i];
		if( edgeID < 0 )
		{
			edgeID = -edgeID;
		}
		dedge_t *pEdge = &dedges[edgeID];
		dvertex_t *pVert0 = &dvertexes[pEdge->v[0]];
		dvertex_t *pVert1 = &dvertexes[pEdge->v[1]];
		AddPointToBounds( pVert0->point, faceMin, faceMax );	
		AddPointToBounds( pVert1->point, faceMin, faceMax );	
	}
}

// FIXME: should stick this in mathlib
static float GetMinDistanceBetweenBoundingBoxes( const Vector &min1, const Vector &max1, 
												 const Vector &min2, const Vector &max2 )
{
	if( IsBoxIntersectingBox( min1, max1, min2, max2 ) )
	{
		return 0.0f;
	}

	Vector axisDist;
	int i;
	for( i = 0; i < 3; i++ )
	{
		if( min1[i] <= max2[i] && max1[i] >= min2[i] )
		{
			// the intersection in this dimension.
			axisDist[i] = 0.0f;
		}
		else
		{
			float dist1, dist2;
			dist1 = min1[i] - max2[i];
			dist2 = min2[i] - max1[i];
			axisDist[i] = dist1 > dist2 ? dist1 : dist2;
			Assert( axisDist[i] > 0.0f );
		}
	}

	float mag = axisDist.Length();
	Assert( mag > 0.0f );
	return mag;
}

static float CalcDistanceFromLeafToWater( int leafNum )
{
	byte	uncompressed[MAX_MAP_LEAFS/8];

	int j, k;

	// If we know that this one doesn't see a water surface then don't bother doing anything.
	if( ((dleafs[leafNum].contents & CONTENTS_TESTFOGVOLUME) == 0) && ( dleafs[leafNum].leafWaterDataID == -1 ) )
		return 65535; // FIXME: make a define for this.
	
	// First get the vis data..
	int cluster = dleafs[leafNum].cluster;
	if (cluster < 0)
		return 65535; // FIXME: make a define for this.
	
	DecompressVis( &dvisdata[dvis->bitofs[cluster][DVIS_PVS]], uncompressed );
	
	float minDist = 65535.0f; // FIXME: make a define for this.
	
	Vector leafMin, leafMax;
	
	leafMin[0] = ( float )dleafs[leafNum].mins[0];
	leafMin[1] = ( float )dleafs[leafNum].mins[1];
	leafMin[2] = ( float )dleafs[leafNum].mins[2];
	leafMax[0] = ( float )dleafs[leafNum].maxs[0];
	leafMax[1] = ( float )dleafs[leafNum].maxs[1];
	leafMax[2] = ( float )dleafs[leafNum].maxs[2];

/*
	CUtlVector<listplane_t> temp;
	
	// build a convex solid out of the planes so that we can get at the triangles.
	for( j = dleafs[i].firstleafbrush; j < dleafs[i].firstleafbrush + dleafs[i].numleafbrushes; j++ )
	{
		dbrush_t *pBrush = &dbrushes[j];
		for( k = pBrush->firstside; k < pBrush->firstside + pBrush->numsides; k++ )
		{
			dbrushside_t *pside = dbrushsides + k;
			dplane_t *pplane = dplanes + pside->planenum;
			AddListPlane( &temp, pplane->normal[0], pplane->normal[1], pplane->normal[2], pplane->dist );
		}
		CPhysConvex *pConvex = physcollision->ConvexFromPlanes( (float *)temp.Base(), temp.Count(), VPHYSICS_MERGE );
		ConvertConvexToCollide(  &pConvex, 
			temp.RemoveAll();
	}
*/

	// Iterate over all potentially visible clusters from this leaf
	for (j = 0; j < dvis->numclusters; ++j)
	{
		// Don't need to bother if this is the same as the current cluster
		if (j == cluster)
			continue;
		
		// If the cluster isn't in our current pvs, then get out of here.
		if ( !CheckBit( uncompressed, j ) )
			continue;
		
		// Found a visible cluster, now iterate over all leaves
		// inside that cluster
		for (k = 0; k < g_ClusterLeaves[j].leafCount; ++k)
		{
			int nClusterLeaf = g_ClusterLeaves[j].leafs[k];
			
			// Don't bother testing the ones that don't see a water boundary.
			if( ((dleafs[nClusterLeaf].contents & CONTENTS_TESTFOGVOLUME) == 0) && ( dleafs[nClusterLeaf].leafWaterDataID == -1 ) )
				continue;	

			// Find the minimum distance between each surface on the boundary of the leaf 
			// that we have the pvs for and each water surface in the leaf that we are testing.
			int nFirstFaceID = dleafs[nClusterLeaf].firstleafface;
			for( int leafFaceID = 0; leafFaceID < dleafs[nClusterLeaf].numleaffaces; ++leafFaceID )
			{
				int faceID = dleaffaces[nFirstFaceID + leafFaceID];
				dface_t *pFace = &dfaces[faceID];
				if( pFace->texinfo == -1 )
					continue;

				texinfo_t *pTexInfo = &texinfo[pFace->texinfo];
				if( pTexInfo->flags & SURF_WARP )
				{
					// Woo hoo!!!  We found a water face.
					// compare the bounding box of the face with the bounding
					// box of the leaf that we are looking from and see
					// what the closest distance is.
					// FIXME: this could be a face/face distance between the water
					// face and the bounding volume of the leaf.
					
					// Get the bounding box of the face
					Vector faceMin, faceMax;
					GetBoundsForFace( faceID, faceMin, faceMax );
					float dist = GetMinDistanceBetweenBoundingBoxes( leafMin, leafMax, faceMin, faceMax );
					if( dist < minDist )
					{
						minDist = dist;
					}
				}
			}
		}
	}
	return minDist;
}

static void CalcDistanceFromLeavesToWater( void )
{
	int i;
	for( i = 0; i < numleafs; i++ )
	{
		g_LeafMinDistToWater[i] = ( unsigned short )CalcDistanceFromLeafToWater( i );
	}
}

//-----------------------------------------------------------------------------
// Using the PVS, compute the visible fog volumes from each leaf
//-----------------------------------------------------------------------------
static void CalcVisibleFogVolumes()
{
	byte	uncompressed[MAX_MAP_LEAFS/8];

	int i, j, k;

	// Clear the contents flags for water testing
	for (i = 0; i < numleafs; ++i)
	{
		dleafs[i].contents &= ~CONTENTS_TESTFOGVOLUME;
		g_LeafMinDistToWater[i] = 65535;
	}

	for (i = 0; i < numleafs; ++i)
	{
		// If we've already discovered that this leaf needs testing,
		// no need to go through the work again...
		if (dleafs[i].contents & CONTENTS_TESTFOGVOLUME)
		{
			Assert((dleafs[i].contents & (CONTENTS_SLIME | CONTENTS_WATER)) == 0);
			continue;
		}

		// Don't bother checking fog volumes from solid leaves
		if (dleafs[i].contents & CONTENTS_SOLID)
			continue;

		// Look only for leaves which are visible from leaves that have fluid in them.
		if ( dleafs[i].leafWaterDataID == -1 )
			continue;

		// Don't bother about looking from CONTENTS_SLIME; we're not going to treat that as interesting.
		// because slime is opaque
		if ( dleafs[i].contents & CONTENTS_SLIME )
			continue;

		// First get the vis data..
		int cluster = dleafs[i].cluster;
		if (cluster < 0)
			continue;

		DecompressVis( &dvisdata[dvis->bitofs[cluster][DVIS_PVS]], uncompressed );

		// Iterate over all potentially visible clusters from this leaf
		for (j = 0; j < dvis->numclusters; ++j)
		{
			// Don't need to bother if this is the same as the current cluster
			if (j == cluster)
				continue;

			if ( !CheckBit( uncompressed, j ) )
				continue;

			// Found a visible cluster, now iterate over all leaves
			// inside that cluster
			for (k = 0; k < g_ClusterLeaves[j].leafCount; ++k)
			{
				int nClusterLeaf = g_ClusterLeaves[j].leafs[k];

				// Don't bother checking fog volumes from solid leaves
				if ( dleafs[nClusterLeaf].contents & CONTENTS_SOLID )
					continue;

				// Don't bother checking from any leaf that's got fluid in it
				if ( dleafs[nClusterLeaf].leafWaterDataID != -1 )
					continue;

				// Here, we've found a case where a non-liquid leaf is visible from a liquid leaf
				// So, in this case, we have to do the expensive test during rendering.
				dleafs[nClusterLeaf].contents |= CONTENTS_TESTFOGVOLUME;
			}
		}
	}
}


//-----------------------------------------------------------------------------
// Compute the bounding box, excluding 3D skybox + skybox, add it to keyvalues
//-----------------------------------------------------------------------------
float DetermineVisRadius( )
{
	float flRadius = -1;

	// Check the max vis range to determine the vis radius
	for (int i = 0; i < num_entities; ++i)
	{
		char* pEntity = ValueForKey(&entities[i], "classname");
		if (!stricmp(pEntity, "env_fog_controller"))
		{
			flRadius = FloatForKey (&entities[i], "farz");
			if (flRadius == 0.0f)
				flRadius = -1.0f;
			break;
		}
	}

	return flRadius;
}

void MarkLeavesAsRadial()
{
	for ( int i = 0; i < numleafs; i++ )
	{
		dleafs[i].flags |= LEAF_FLAGS_RADIAL;
	}
}


int ParseCommandLine( int argc, char **argv )
{
	int i;
	for (i=1 ; i<argc ; i++)
	{
		if (!Q_stricmp(argv[i],"-threads"))
		{
			numthreads = atoi (argv[i+1]);
			i++;
		}
		else if (!Q_stricmp(argv[i], "-fast"))
		{
			Msg ("fastvis = true\n");
			fastvis = true;
		}
		else if (!Q_stricmp(argv[i], "-cpu"))
		{
			Msg ("CPU recursive flow = true (GPU raycast disabled)\n");
			g_bForceCpuFlow = true;
		}
		else if (!Q_stricmp(argv[i], "-opencl"))
		{
			Msg ("GPU backend forced: OpenCL\n");
			_putenv_s( "VVIS_GPU_BACKEND", "opencl" );
		}
		else if (!Q_stricmp(argv[i], "-cuda"))
		{
			Msg ("GPU backend forced: CUDA (OpenCL fallback if unavailable)\n");
			_putenv_s( "VVIS_GPU_BACKEND", "cuda" );
		}
		else if (!Q_stricmp(argv[i], "-final"))
		{
			g_PassSamples[0] = 4;
			g_PassSamples[1] = 8;
			g_NumPasses = 2;
			Msg( "final mode: pass 1 = default (4), pass 2 = refinement (8) on unproven pairs only\n" );
		}
		else if (!Q_stricmp(argv[i], "-RaycastType") && i + 1 < argc)
		{
			int t = atoi( argv[++i] );
			if ( t < 0 || t > 1 )
			{
				Warning( "-RaycastType must be 0 or 1\n" );
				return 100000;
			}
			g_PassSamples[0] = g_RaycastTypeSamples[t];
			g_NumPasses = 1;
			Msg( "RaycastType %d (%d samples cap)\n", t, g_PassSamples[0] );
		}
		else if (!Q_stricmp(argv[i], "-Ray") && i + 1 < argc)
		{
			int n = atoi( argv[++i] );
			if ( n < 1 )
			{
				Warning( "-Ray must be >= 1\n" );
				return 100000;
			}
			g_PassSamples[0] = n;
			g_NumPasses = 1;
			Msg( "Custom raycast samples cap = %d\n", n );
		}
		else if (!Q_stricmp(argv[i], "-Pass") && i + 1 < argc)
		{
			// Accepte "0:RaycastType=1,1:RaycastType=3" (virgules/espaces, un ou plusieurs arguments)
			// ou "0:Ray=6". Les arguments suivants contenant ':' sont aussi consommes.
			g_NumPasses = 0;
			while ( i + 1 < argc && argv[i+1][0] != '-' && strchr( argv[i+1], ':' ) )
			{
				char buf[256];
				V_strncpy( buf, argv[++i], sizeof( buf ) );
				for ( char *tok = strtok( buf, ", " ); tok; tok = strtok( NULL, ", " ) )
				{
					char *colon = strchr( tok, ':' );
					char *eq = colon ? strchr( colon, '=' ) : NULL;
					if ( !colon || !eq || g_NumPasses >= 16 )
					{
						Warning( "Invalid -Pass entry \"%s\"\n", tok );
						return 100000;
					}
					*colon = 0; *eq = 0;
					const char *key = colon + 1;
					int val = atoi( eq + 1 );
					int s;
					if ( !Q_stricmp( key, "RaycastType" ) && val >= 0 && val <= 1 )
						s = g_RaycastTypeSamples[val];
					else if ( !Q_stricmp( key, "Ray" ) && val >= 1 )
						s = val;
					else
					{
						Warning( "Invalid -Pass value \"%s=%s\"\n", key, eq + 1 );
						return 100000;
					}
					g_PassSamples[g_NumPasses++] = s;
				}
			}
			if ( g_NumPasses == 0 )
			{
				Warning( "-Pass: no valid pass given\n" );
				return 100000;
			}
			Msg( "%d raycast pass(es)\n", g_NumPasses );
		}
		else if (!Q_stricmp(argv[i], "-raycast"))
		{
			Msg ("%s ignored: GPU raycast is the default flow\n", argv[i]);
		}
		else if (!Q_stricmp(argv[i], "-v") || !Q_stricmp(argv[i], "-verbose"))
		{
			Msg ("verbose = true\n");
			verbose = true;
		}
		else if( !Q_stricmp( argv[i], "-radius_override" ) )
		{
			g_bUseRadius = true;
			g_VisRadius = atof( argv[i+1] );
			i++;
			Msg( "Vis Radius = %4.2f\n", g_VisRadius );
			g_VisRadius = g_VisRadius * g_VisRadius;   // so distance check can be squared
		}
		else if( !Q_stricmp( argv[i], "-trace" ) )
		{
			g_TraceClusterStart = atoi( argv[i+1] );
			i++;
			g_TraceClusterStop = atoi( argv[i+1] );
			i++;
			Msg( "Tracing vis from cluster %d to %d\n", g_TraceClusterStart, g_TraceClusterStop );
		}
		else if (!Q_stricmp (argv[i],"-nosort"))
		{
			Msg ("nosort = true\n");
			nosort = true;
		}
		else if (!Q_stricmp (argv[i],"-tmpin"))
			strcpy (inbase, "/tmp");
		else if( !Q_stricmp( argv[i], "-low" ) )
		{
			g_bLowPriority = true;
		}
		else if ( !Q_stricmp( argv[i], "-FullMinidumps" ) )
		{
			EnableFullMinidumps( true );
		}
		else if ( !Q_stricmp( argv[i], CMDLINEOPTION_NOVCONFIG ) )
		{
		}
		else if ( !Q_stricmp( argv[i], "-vproject" ) || !Q_stricmp( argv[i], "-game" ) )
		{
			++i;
		}
		else if ( !Q_stricmp( argv[i], "-allowdebug" ) || !Q_stricmp( argv[i], "-steam" ) )
		{
			// nothing to do here, but don't bail on this option
		}
		else if (argv[i][0] == '-')
		{
			Warning("VBSP: Unknown option \"%s\"\n\n", argv[i]);
			i = 100000;	// force it to print the usage
			break;
		}
		else
			break;
	}
	return i;
}


void PrintCommandLine( int argc, char **argv )
{
	Warning( "Command line: " );
	for ( int z=0; z < argc; z++ )
	{
		Warning( "\"%s\" ", argv[z] );
	}
	Warning( "\n\n" );
}


void PrintUsage( int argc, char **argv )
{
	PrintCommandLine( argc, argv );

	Warning(	
		"usage  : vvis [options...] bspfile\n"
		"example: vvis -fast c:\\hl2\\hl2\\maps\\test\n"
		"\n"
		"GPU options:\n"
		"\n"
		"  (default)       : GPU raycast (CUDA on NVIDIA, else OpenCL), CPU if no GPU\n"
		"  VVIS_GPU_DEVICE=<n> (env) : use the n-th GPU\n"
		"  -opencl         : Force OpenCL (NVIDIA, AMD, Intel)\n"
		"  -cuda           : Force CUDA (NVIDIA)\n"
		"  -cpu            : Original recursive CPU flow (no GPU)\n"
		"\n"
		"Raycast precision (GPU):\n"
		"\n"
		"  -final             : Smart multi-pass (like vrad -final): default pass, then a\n"
		"                       refinement pass (cap 8) re-testing only unproven pairs.\n"
		"  -RaycastType <0-1> : 0 = default (cap 4, 9 points/portal)\n"
		"                       1 = more precise (cap 5, 11 points/portal, ~+25%% samples)\n"
		"  -Ray <n>           : Custom sample cap per ring (1 + 2*min(n, vertices)\n"
		"                       points per portal). Default: 4 (= type 0, 9 points,\n"
		"                       up to 81 rays per portal pair). Type 1 = 5.\n"
		"                       Values above ~8 (max portal vertices) have no effect.\n"
		"  -Pass <list>       : Several passes, e.g. -Pass 0:Ray=4,1:RaycastType=1\n"
		"                       (or 0:Ray=2,1:Ray=8). Pass N+1 only re-tests the\n"
		"                       candidate pairs not yet proven visible by earlier\n"
		"                       passes; results are merged (never removes visibility).\n"
		"                       Each pass prints a progress bar, % and time.\n"
		"\n"
		"Common options:\n"
		"\n"
		"  -game <directory>     : Game directory (auto-detected for Garry's Mod)\n"
		"  -vproject <directory> : Same as -game.\n"
		"  -threads <n>    : Number of CPU threads (defaults to the processor count).\n"
		"  -fast           : Only do the first quick pass.\n"
		"  -low            : Run as an idle-priority process.\n"
		"  -v, -verbose    : Verbose output.\n"
		"  -radius_override <r> : Force a vis radius (default: env_fog_controller farz).\n"
		"  -nosort         : Don't sort portals.\n"
		"  -trace <start cluster> <end cluster> : Write a linefile tracing the vis between two clusters.\n"
		"\n"
		);

}


int RunVVis( int argc, char **argv )
{
	char	portalfile[1024];
	char		source[1024];
	char		mapFile[1024];
	double		start, end;


	Msg( "Valve + Lumastor-IronKnight Software - vvis_gpu.exe (%s)\n", __DATE__ );

	verbose = false;

	LoadCmdLineFromFile( argc, argv, source, "vvis" );
	int i = ParseCommandLine( argc, argv );

	CmdLib_InitFileSystem( argv[ argc - 1 ] );

	V_FileBase( argv[ argc - 1 ], mapFile, sizeof( mapFile ) );
	V_strncpy( mapFile, ExpandPath( mapFile ), sizeof( mapFile ) );
	V_strncat( mapFile, ".bsp", sizeof( mapFile ) );

	// Source is just the mapfile without an extension at this point...
	V_strncpy( source, mapFile, sizeof( mapFile ) );
	V_StripExtension( source, source, sizeof( source ) );

	if (i != argc - 1)
	{
		PrintUsage( argc, argv );
		DeleteCmdLine( argc, argv );
		CmdLib_Exit( 1 );
	}

	start = Plat_FloatTime();

	// Setup the logfile.
	char logFile[512];
	_snprintf( logFile, sizeof(logFile), "%s.log", source );
	SetSpewFunctionLogFile( logFile );
	

	// Run in the background?
	if( g_bLowPriority )
	{
		SetLowPriority();
	}

	ThreadSetDefault ();

	Msg ("reading %s\n", mapFile);
	LoadBSPFile (mapFile);
	if (numnodes == 0 || numfaces == 0)
		Error ("Empty map");
	ParseEntities ();

	// Check the VMF for a vis radius
	if (!g_bUseRadius)
	{
		float flRadius = DetermineVisRadius( );
		if (flRadius > 0.0f)
		{
			g_bUseRadius = true;
			g_VisRadius = flRadius * flRadius;
		}
	}

	if ( g_bUseRadius )
	{
		MarkLeavesAsRadial();
	}

	if ( inbase[0] == 0 )
	{
		strcpy( portalfile, source );
	}
	else
	{
		sprintf ( portalfile, "%s%s", inbase, argv[i] );
		Q_StripExtension( portalfile, portalfile, sizeof( portalfile ) );
	}
	strcat (portalfile, ".prt");

	Msg ("reading %s\n", portalfile);
	LoadPortals (portalfile);

	// don't write out results when simply doing a trace
	if ( g_TraceClusterStart < 0 )
	{
		CalcVis ();
		CalcPAS ();

		// We need a mapping from cluster to leaves, since the PVS
		// deals with clusters for both CalcVisibleFogVolumes and
		BuildClusterTable();

		CalcVisibleFogVolumes();
		CalcDistanceFromLeavesToWater();

		visdatasize = vismap_p - dvisdata;
		Msg ("visdatasize:%i  compressed from %i\n", visdatasize, originalvismapsize*2);

		Msg ("writing %s\n", mapFile);
		WriteBSPFile (mapFile);
	}
	else
	{
		if ( g_TraceClusterStart < 0 || g_TraceClusterStart >= portalclusters || g_TraceClusterStop < 0 || g_TraceClusterStop >= portalclusters )
		{
			Error("Invalid cluster trace: %d to %d, valid range is 0 to %d\n", g_TraceClusterStart, g_TraceClusterStop, portalclusters-1 );
		}
		CalcVisTrace ();
		WritePortalTrace(source);
	}

	end = Plat_FloatTime();

	char str[512];
	GetHourMinuteSecondsString( (int)( end - start ), str, sizeof( str ) );
	Msg( "%s elapsed\n", str );

	ReleasePakFileLumps();
	DeleteCmdLine( argc, argv );
	CmdLib_Cleanup();
	return 0;
}


/*
===========
main
===========
*/
int main (int argc, char **argv)
{
	CommandLine()->CreateCmdLine( argc, argv );

	MathLib_Init( 2.2f, 2.2f, 0.0f, 1.0f, false, false, false, false );
	InstallAllocationFunctions();
	InstallSpewFunction();

	SetupDefaultToolsMinidumpHandler();

	return RunVVis( argc, argv );
}

#ifdef VVIS_BUILD_AS_DLL
class CVVisDLL : public ILaunchableDLL
{
public:
	virtual int main( int argc, char **argv )
	{
		return ::main( argc, argv );
	}
};

EXPOSE_SINGLE_INTERFACE( CVVisDLL, ILaunchableDLL, LAUNCHABLE_DLL_INTERFACE_VERSION );
#endif
