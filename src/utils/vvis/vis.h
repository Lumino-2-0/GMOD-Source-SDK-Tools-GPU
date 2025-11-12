#include "cmdlib.h"
#include "mathlib/mathlib.h"
#include "bsplib.h"

#define MAX_PORTALS 65536
#define PORTALFILE "PRT1"

extern bool g_bUseRadius;           // "radius vis" TF2
extern double g_VisRadius;

extern bool g_bTryGPU;    // active l'échantillonnage CPU↔GPU (-TryGPU)
extern bool g_bDebugMode; // active le tracing/logging (-debug)

struct plane_t {
    Vector normal;
    float dist;
};

#define MAX_POINTS_ON_WINDING   64
#define MAX_POINTS_ON_FIXED_WINDING     16  // A ajuster pour correspondre a cl_winding_t

struct winding_t {
    qboolean original;
    int numpoints;
    Vector points[MAX_POINTS_ON_FIXED_WINDING];
};

winding_t* NewWinding(int points);
void FreeWinding(winding_t* w);
winding_t* CopyWinding(winding_t* w);

typedef enum { stat_none, stat_working, stat_done } vstatus_t;
struct portal_t {
    plane_t plane;      // normal pointant vers le voisin
    int leaf;           // cluster voisin
    Vector origin;      // centre pour test rapide
    float radius;
    winding_t* winding;
    vstatus_t status;
    byte* portalfront;  // [portals], preliminaire
    byte* portalflood;  // [portals], intermediaire
    byte* portalvis;    // [portals], final
    int nummightsee;    // bit-count de portalflood pour tri
};

struct leaf_t {
    CUtlVector<portal_t*> portals;
};

struct pstack_t {
    byte mightsee[MAX_PORTALS / 8];
    pstack_t* next;
    leaf_t* leaf;
    portal_t* portal;
    winding_t* source;
    winding_t* pass;
    winding_t windings[3];
    int freewindings[3];
    plane_t portalplane;
};

struct threaddata_t {
    portal_t* base;
    int c_chains;
    pstack_t pstack_head;
};

extern int g_numportals;
extern int portalclusters;

extern portal_t* portals;
extern leaf_t* leafs;

extern int c_portaltest, c_portalpass, c_portalcheck;
extern int c_portalskip, c_leafskip;
extern int c_vistest, c_mighttest;
extern int c_chains;

extern byte* vismap, * vismap_p, * vismap_end;
extern int testlevel;
extern byte* uncompressed;

extern int leafbytes, leaflongs;
extern int portalbytes, portallongs;

void LeafFlow(int leafnum);

void BasePortalVis(int iThread, int portalnum);
void BetterPortalVis(int portalnum);
void PortalFlow(int iThread, int portalnum);
void WritePortalTrace(const char* source);

extern portal_t* sorted_portals[MAX_MAP_PORTALS * 2];
extern int g_TraceClusterStart, g_TraceClusterStop;

int CountBits(byte* bits, int numbits);

#define CheckBit(bitstring, bitNumber) ((bitstring)[(bitNumber)>>3] & (1 << ((bitNumber)&7)))
#define SetBit(bitstring, bitNumber)   ((bitstring)[(bitNumber)>>3] |= (1 << ((bitNumber)&7)))
#define ClearBit(bitstring, bitNumber) ((bitstring)[(bitNumber)>>3] &= ~(1 << ((bitNumber)&7)))