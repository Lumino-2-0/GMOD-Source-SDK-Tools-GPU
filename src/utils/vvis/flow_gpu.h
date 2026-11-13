// flow_gpu.h
//
// Interface entre VVIS (CPU) et le calcul de visibilite sur GPU.
//
// Deux backends, charges dynamiquement (aucun lien statique) :
//   - CUDA   : nvcuda.dll, noyau PTX embarque (flow_gpu_kernel.cu)
//   - OpenCL : OpenCL.dll, noyau embarque en texte (flow_gpu_kernel.h)
// Les deux noyaux font exactement le meme calcul. Si aucun GPU n'est
// utilisable, GPU_Init() renvoie false et VVIS reprend le flow CPU d'origine.

#pragma once

// --- GPUPortal ----------------------------------------------------------------
// Copie "a plat" d'un portal_t : plan, leaf de destination, sphere englobante
// et position des sommets du winding dans le tableau winding_pts.
// Uniquement des champs de 4 octets : la structure a la meme disposition
// cote CPU, cote CUDA et cote OpenCL C. Ne pas changer l'ordre des champs.

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

// --- GPULeaf --------------------------------------------------------------------
// Un leaf connait ses portails sortants : ils sont stockes dans le tableau
// leaf_portals, entre les indices portals_start et portals_start + portals_count.

typedef struct
{
	int portals_start;
	int portals_count;
} GPULeaf;

// --- Interface hote ------------------------------------------------------------

bool GPU_Init( void );
void GPU_Shutdown( void );
// passSamples[k] = plafond de sommets par anneau de la passe k (voir vvis.cpp).
// Les passes > 0 ne retraitent que les paires candidates non encore prouvees
// visibles ; les resultats sont fusionnes par OR.
bool GPU_RunPortalFlowRaycast( int sorted_portal_count, const int *passSamples, int numPasses );
