//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose: 
//
// $NoKeywords: $
//
//=============================================================================//

#include "vrad.h"
#include "vrad_gpu.h"
#include "vrad_gpu.cpp"	// compile avec vismat.cpp (evite de modifier le vcxproj)
#include "vmpi.h"
#ifdef MPI
#include "messbuf.h"
static MessageBuffer mb;
#endif

#define	HALFBIT

extern char		source[MAX_PATH];
extern char		vismatfile[_MAX_PATH];
extern char		incrementfile[_MAX_PATH];
extern qboolean	incremental;

/*
===================================================================

VISIBILITY MATRIX

Determine which patches can see each other
Use the PVS to accelerate if available
===================================================================
*/

#define TEST_EPSILON	0.1
#define PLANE_TEST_EPSILON  0.01 // patch must be this much in front of the plane to be considered "in front"
#define PATCH_FACE_OFFSET  0.1 // push patch origins off from the face by this amount to avoid self collisions

#define STREAM_SIZE 512

static volatile __int64 s_xGpu = 0, s_xApply = 0, s_xBatches = 0, s_xRows = 0;
static __int64 XNow() { LARGE_INTEGER t; QueryPerformanceCounter( &t ); return t.QuadPart; }
void VradProf_PrintTransfer()
{
	LARGE_INTEGER f; QueryPerformanceFrequency( &f );
	Msg( "[PROF] transferts : GPU %.2f s, application %.2f s, generation rayons %.2f s (cumule threads), %d lots\n",
		(double)s_xGpu / f.QuadPart, (double)s_xApply / f.QuadPart, (double)s_xRows / f.QuadPart, (int)s_xBatches );
}

class CTransferMaker
{
public:

	CTransferMaker( transfer_t *all_transfers );
	~CTransferMaker();

	FORCEINLINE void TestMakeTransfer( Vector start, Vector stop, int ndxShooter, int ndxReciever )
	{
		if ( g_bUseCUDA )
		{
			float *r = &m_pGpuRays[m_nTests * 6];
			r[0] = start.x; r[1] = start.y; r[2] = start.z;
			r[3] = stop.x; r[4] = stop.y; r[5] = stop.z;
		}
		if ( !g_bUseCUDA || g_bCUDAVerify )
			g_RtEnv.AddToRayStream( m_RayStream, start, stop, &m_pResults[m_nTests] );
		m_pShooterPatches[m_nTests] = ndxShooter;
		m_pRecieverPatches[m_nTests] = ndxReciever;
		++m_nTests;
	}

	void Finish();

	// Mode CUDA : les rayons de plusieurs patchs sont accumules puis traces en un seul lot.
	void EndPatch( int patchnum, CPatch *patch );
	bool BatchFull() const { return m_nTests + MAX_PATCHES > m_nCapacity || m_Pending.Count() >= 2048; }
	void FlushBatch( void (*PatchCB)(int iThread, int patchnum, CPatch *patch), int iThread );

private:

	struct PendingPatch_t { int patchnum; int endTest; CPatch *patch; };
	void Resolve();
	int m_nCapacity;
	CUtlVector<PendingPatch_t> m_Pending;
	int m_nTests;
	RayTracingSingleResult *m_pResults;
	int *m_pShooterPatches;
	int *m_pRecieverPatches;
	float *m_pGpuRays;
	unsigned char *m_pGpuOut;
	RayStream m_RayStream;
	transfer_t *m_AllTransfers;
};

CTransferMaker::CTransferMaker( transfer_t *all_transfers ) :
	m_AllTransfers( all_transfers ), m_nTests( 0 )
{
	m_nCapacity = g_bUseCUDA ? MAX_PATCHES * 4 : MAX_PATCHES;
	m_pResults = (RayTracingSingleResult *)calloc( 1, m_nCapacity * sizeof ( RayTracingSingleResult ) );
	m_pShooterPatches = (int *)calloc( 1, m_nCapacity * sizeof( int ) );
	m_pRecieverPatches = (int *)calloc( 1, m_nCapacity * sizeof( int ) );
	m_pGpuRays = g_bUseCUDA ? (float *)calloc( 1, (size_t)m_nCapacity * 6 * sizeof( float ) ) : NULL;
	m_pGpuOut = g_bUseCUDA ? (unsigned char *)calloc( 1, m_nCapacity ) : NULL;
}

CTransferMaker::~CTransferMaker()
{
	free ( m_pResults );
	free ( m_pShooterPatches );
	free (m_pRecieverPatches );
	free( m_pGpuRays );
	free( m_pGpuOut );
}

// Calcule la visibilite finale de chaque test. Resultat dans m_pGpuOut (1 = visible) en mode CUDA.
void CTransferMaker::Resolve()
{
	if ( !g_bUseCUDA || g_bCUDAVerify )
		g_RtEnv.FinishRayStream( m_RayStream );

	if ( !g_bUseCUDA )
		return;

	bool gpuOk = VradGPU_TraceVisibility( m_pGpuRays, m_nTests, m_pGpuOut );
	if ( !gpuOk )
	{
		if ( !g_bCUDAVerify )
			Error( "[VRAD GPU] echec du lancement CUDA pendant les transferts\n" );
		for ( int i = 0; i < m_nTests; ++i )
			m_pGpuOut[i] = ( m_pResults[i].HitID == -1 || m_pResults[i].HitDistance >= m_pResults[i].ray_length );
		return;
	}
	if ( g_bCUDAVerify )
	{
		int mismatches = 0;
		for ( int i = 0; i < m_nTests; ++i )
		{
			bool cpuFree = ( m_pResults[i].HitID == -1 || m_pResults[i].HitDistance >= m_pResults[i].ray_length );
			mismatches += ( ( m_pGpuOut[i] != 0 ) != cpuFree );
			m_pGpuOut[i] = cpuFree;	// la verification n'altere pas le resultat
		}
		VradGPU_AddVerifyStats( m_nTests, mismatches );
	}
}

void CTransferMaker::Finish()
{
	Resolve();
	for ( int i = 0; i < m_nTests; ++i )
	{
		bool visible = g_bUseCUDA ? ( m_pGpuOut[i] != 0 )
			: ( m_pResults[i].HitID == -1 || m_pResults[i].HitDistance >= m_pResults[i].ray_length );
		if ( visible )
			MakeTransfer( m_pShooterPatches[i], m_pRecieverPatches[i], m_AllTransfers );
	}
	m_nTests = 0;
}

void CTransferMaker::EndPatch( int patchnum, CPatch *patch )
{
	PendingPatch_t p = { patchnum, m_nTests, patch };
	m_Pending.AddToTail( p );
}

void CTransferMaker::FlushBatch( void (*PatchCB)(int iThread, int patchnum, CPatch *patch), int iThread )
{
	if ( m_Pending.Count() == 0 )
		return;
	__int64 x0 = XNow();
	Resolve();
	__int64 x1 = XNow();

	int begin = 0;
	for ( int p = 0; p < m_Pending.Count(); ++p )
	{
		for ( int i = begin; i < m_Pending[p].endTest; ++i )
		{
			if ( m_pGpuOut[i] )
				MakeTransfer( m_pShooterPatches[i], m_pRecieverPatches[i], m_AllTransfers );
		}
		begin = m_Pending[p].endTest;

		MakeScales( m_Pending[p].patchnum, m_AllTransfers );
		if ( PatchCB )
			PatchCB( iThread, m_Pending[p].patchnum, m_Pending[p].patch );
	}
	m_Pending.RemoveAll();
	m_nTests = 0;
	__int64 x2 = XNow();
	InterlockedExchangeAdd64( &s_xGpu, x1 - x0 );
	InterlockedExchangeAdd64( &s_xApply, x2 - x1 );
	InterlockedIncrement64( &s_xBatches );
}


dleaf_t* PointInLeaf (int iNode, Vector const& point)
{
	if ( iNode < 0 )
		return &dleafs[ (-1-iNode) ];

	dnode_t *node = &dnodes[iNode];
	dplane_t *plane = &dplanes[ node->planenum ];

	float dist = DotProduct (point, plane->normal) - plane->dist;
	if ( dist > TEST_EPSILON )
	{
		return PointInLeaf( node->children[0], point );
	}
	else if ( dist < -TEST_EPSILON )
	{
		return PointInLeaf( node->children[1], point );
	}
	else
	{
		dleaf_t *pTest = PointInLeaf( node->children[0], point );
		if ( pTest->cluster != -1 )
			return pTest;

		return PointInLeaf( node->children[1], point );
	}
}


int ClusterFromPoint( Vector const& point )
{
	dleaf_t *leaf = PointInLeaf( 0, point );
	
	return leaf->cluster;
}

void PvsForOrigin (Vector& org, byte *pvs)
{
	int visofs;
	int cluster;

	if (!visdatasize)
	{
		memset (pvs, 255, (dvis->numclusters+7)/8 );
		return;
	}

	cluster = ClusterFromPoint( org );
	if ( cluster < 0 )
	{
		visofs = -1;
	}
	else
	{
		visofs = dvis->bitofs[ cluster ][DVIS_PVS];
	}

	if (visofs == -1)
		Error ("visofs == -1");

	DecompressVis (&dvisdata[visofs], pvs);
}


void TestPatchToPatch( int ndxPatch1, int ndxPatch2, int head, transfer_t *transfers, CTransferMaker &transferMaker, int iThread )
{
	Vector tmp;

	//
	// get patches
	//
	if( ndxPatch1 == g_Patches.InvalidIndex() || ndxPatch2 == g_Patches.InvalidIndex() )
		return;

	CPatch *patch = &g_Patches.Element( ndxPatch1 );
	CPatch *patch2 = &g_Patches.Element( ndxPatch2 );

	if (patch2->child1 != g_Patches.InvalidIndex() )
	{
		// check to see if we should use a child node instead

		VectorSubtract( patch->origin, patch2->origin, tmp );
		// SQRT( 1/4 )
		// FIXME: should be based on form-factor (ie. include visible angle, etc)
		if ( DotProduct(tmp, tmp) * 0.0625 < patch2->area )
		{
			TestPatchToPatch( ndxPatch1, patch2->child1, head, transfers, transferMaker, iThread );
			TestPatchToPatch( ndxPatch1, patch2->child2, head, transfers, transferMaker, iThread );
			return;
		}
	}

	// check vis between patch and patch2
	// if bit has not already been set
	//  && v2 is not behind light plane
	//  && v2 is visible from v1
	if ( DotProduct( patch2->origin, patch->normal ) > patch->planeDist + PLANE_TEST_EPSILON )
	{
		// push out origins from face so that don't intersect their owners
		Vector p1, p2;
		VectorAdd( patch->origin, patch->normal, p1 );
		VectorAdd( patch2->origin, patch2->normal, p2 );
		transferMaker.TestMakeTransfer( p1, p2, ndxPatch1, ndxPatch2 );
	}
}


/*
==============
TestPatchToFace

Sets vis bits for all patches in the face
==============
*/
void TestPatchToFace (unsigned patchnum, int facenum, int head, transfer_t *transfers, CTransferMaker &transferMaker, int iThread )
{
	if( faceParents.Element( facenum ) == g_Patches.InvalidIndex() || patchnum == g_Patches.InvalidIndex() )
		return;

	CPatch	*patch = &g_Patches.Element( patchnum );
	CPatch *patch2 = &g_Patches.Element( faceParents.Element( facenum ) );

	// if emitter is behind that face plane, skip all patches

	CPatch *pNextPatch;

	if ( patch2 && DotProduct(patch->origin, patch2->normal) > patch2->planeDist + PLANE_TEST_EPSILON )
	{
		// we need to do a real test
		for( ; patch2; patch2 = pNextPatch )
		{
			// next patch
			pNextPatch = NULL;
			if( patch2->ndxNextParent != g_Patches.InvalidIndex() )
			{
				pNextPatch = &g_Patches.Element( patch2->ndxNextParent );
			}

			/*
			// skip patches too far away
			VectorSubtract( patch->origin, patch2->origin, tmp );
			if (DotProduct( tmp, tmp ) > 512 * 512)
				continue;
			*/

			int ndxPatch2 = patch2 - g_Patches.Base();
			TestPatchToPatch( patchnum, ndxPatch2, head, transfers, transferMaker, iThread );
		}
	}
}


struct ClusterDispList_t
{
	CUtlVector<int>	dispFaces;
};

static CUtlVector<ClusterDispList_t> g_ClusterDispFaces;

//-----------------------------------------------------------------------------
// Helps us find all displacements associated with a particular cluster
//-----------------------------------------------------------------------------
void AddDispsToClusterTable( void )
{
	g_ClusterDispFaces.SetCount( g_ClusterLeaves.Count() );

	//
	// add displacement faces to the cluster table
	//
	for( int ndxFace = 0; ndxFace < numfaces; ndxFace++ )
	{
		// search for displacement faces
		if( g_pFaces[ndxFace].dispinfo == -1 )
			continue;

		//
		// get the clusters associated with the face
		//
		if( g_FacePatches.Element( ndxFace ) != g_FacePatches.InvalidIndex() )
		{
			CPatch *pNextPatch = NULL;
			for( CPatch *pPatch = &g_Patches.Element( g_FacePatches.Element( ndxFace ) ); pPatch; pPatch = pNextPatch )
			{
				// next patch
				pNextPatch = NULL;
				if( pPatch->ndxNext != g_Patches.InvalidIndex() )
				{
					pNextPatch = &g_Patches.Element( pPatch->ndxNext );
				}

				if( pPatch->clusterNumber != g_Patches.InvalidIndex() )
				{
					int ndxDisp = g_ClusterDispFaces[pPatch->clusterNumber].dispFaces.Find( ndxFace );
					if( ndxDisp == -1 )
					{
						ndxDisp = g_ClusterDispFaces[pPatch->clusterNumber].dispFaces.AddToTail();
						g_ClusterDispFaces[pPatch->clusterNumber].dispFaces[ndxDisp] = ndxFace;
					}
				}
			}
		}
	}
}


/*
==============
BuildVisRow

Calc vis bits from a single patch
==============
*/
void BuildVisRow (int patchnum, byte *pvs, int head, transfer_t *transfers, CTransferMaker &transferMaker, int iThread )
{
	int		j, k, l, leafIndex;
	CPatch	*patch;
	dleaf_t	*leaf;
	byte	face_tested[MAX_MAP_FACES];
	byte	disp_tested[MAX_MAP_FACES];

	patch = &g_Patches.Element( patchnum );

	memset( face_tested, 0, numfaces ) ;
	memset( disp_tested, 0, numfaces );

	for (j=0; j<dvis->numclusters; j++)
	{
		if ( ! ( pvs[(j)>>3] & (1<<((j)&7)) ) )
		{
			continue;		// not in pvs
		}

		for ( leafIndex = 0; leafIndex < g_ClusterLeaves[j].leafCount; leafIndex++ )
		{
			leaf = dleafs + g_ClusterLeaves[j].leafs[leafIndex];

			for (k=0 ; k<leaf->numleaffaces; k++)
			{
				l = dleaffaces[leaf->firstleafface + k];
				// faces can be marksurfed by multiple leaves, but
				// don't bother testing again
				if (face_tested[l])
				{
					continue;
				}
				face_tested[l] = 1;

				// don't check patches on the same face
				if (patch->faceNumber == l)
					continue;
				TestPatchToFace (patchnum, l, head, transfers, transferMaker, iThread );
			}
		}

		int dispCount = g_ClusterDispFaces[j].dispFaces.Size();
		for( int ndxDisp = 0; ndxDisp < dispCount; ndxDisp++ )
		{
			int ndxFace = g_ClusterDispFaces[j].dispFaces[ndxDisp];
			if( disp_tested[ndxFace] )
				continue;

			disp_tested[ndxFace] = 1;

			// don't check patches on the same face
			if( patch->faceNumber == ndxFace )
				continue;

			TestPatchToFace( patchnum, ndxFace, head, transfers, transferMaker, iThread );
		}
	}


	// Msg("%d) Transfers: %5d\n", patchnum, patch->numtransfers);
}



/*
===========
BuildVisLeafs

  This is run by multiple threads
===========
*/

transfer_t* BuildVisLeafs_Start()
{
	return (transfer_t *)calloc( 1,  MAX_PATCHES * sizeof( transfer_t ) );
}


// If PatchCB is non-null, it is called after each row is generated (used by MPI).
void BuildVisLeafs_Cluster( 
	int threadnum,
	transfer_t *transfers, 
	int iCluster, 
	void (*PatchCB)(int iThread, int patchnum, CPatch *patch)
	)
{
	byte	pvs[(MAX_MAP_CLUSTERS+7)/8];
	CPatch	*patch;
	int		head;
	unsigned	patchnum;
	
	DecompressVis( &dvisdata[ dvis->bitofs[ iCluster ][DVIS_PVS] ], pvs);
	head = 0;

	CTransferMaker transferMaker( transfers );

	// light every patch in the cluster
	if( clusterChildren.Element( iCluster ) != clusterChildren.InvalidIndex() )
	{
		CPatch *pNextPatch;
		for( patch = &g_Patches.Element( clusterChildren.Element( iCluster ) ); patch; patch = pNextPatch )
		{
			//
			// next patch
			//
			pNextPatch = NULL;
			if( patch->ndxNextClusterChild != g_Patches.InvalidIndex() )
			{
				pNextPatch = &g_Patches.Element( patch->ndxNextClusterChild );
			}
			
			patchnum = patch - g_Patches.Base();

			// build to all other world clusters
			__int64 xr0 = XNow();
			BuildVisRow (patchnum, pvs, head, transfers, transferMaker, threadnum );
			InterlockedExchangeAdd64( &s_xRows, XNow() - xr0 );

			if ( g_bUseCUDA )
			{
				// les transferts de ce patch seront crees au prochain FlushBatch
				transferMaker.EndPatch( patchnum, patch );
				if ( transferMaker.BatchFull() )
					transferMaker.FlushBatch( PatchCB, threadnum );
				continue;
			}

			transferMaker.Finish();

			// do the transfers
			MakeScales( patchnum, transfers );

			// Let MPI aggregate the data if it's being used.
			if ( PatchCB )
				PatchCB( threadnum, patchnum, patch );
		}
	}

	if ( g_bUseCUDA )
		transferMaker.FlushBatch( PatchCB, threadnum );
}


void BuildVisLeafs_End( transfer_t *transfers )
{
	free( transfers );
}


void BuildVisLeafs( int threadnum, void *pUserData )
{
	transfer_t *transfers = BuildVisLeafs_Start();
	
	while ( 1 )
	{
		//
		// build a minimal BSP tree that only
		// covers areas relevent to the PVS
		//
		// JAY: Now this returns a cluster index
		int iCluster = GetThreadWork();
		if ( iCluster == -1 )
			break;

		BuildVisLeafs_Cluster( threadnum, transfers, iCluster, NULL );
	}
	
	BuildVisLeafs_End( transfers );
}


/*
==============
BuildVisMatrix
==============
*/
void BuildVisMatrix (void)
{
#ifdef MPI
	if ( g_bUseMPI )
	{
		RunMPIBuildVisLeafs();
	}
	else 
#endif
	{
		RunThreadsOn (dvis->numclusters, true, BuildVisLeafs);
	}
}

void FreeVisMatrix (void)
{

}
